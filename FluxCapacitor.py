import os
import sys
import contextlib
import random
import shutil
from datetime import datetime

import torch
from PIL import Image
import torchvision.transforms as T

from diffusers import DiffusionPipeline
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer

# Optional 4-bit quantization if available (NVIDIA GPUs)
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

# Optional GUI selection; we add a safe fallback for headless envs
try:
    import tkinter as tk
    from tkinter import filedialog
    _HAS_TK = True
except Exception:
    _HAS_TK = False

import colorama
from colorama import Fore, init, Style
init(autoreset=True)

# ----------------------------
# Configuration
# ----------------------------
RESOLUTION = (1024, 1024)
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 3.5
SHOW_IMAGE = False

# Local LLM for prompt generation (Granite 3B instruct)
PROMPT_MODEL_ID = "ibm-granite/granite-4.0-h-micro"  # switch to "ibm-granite/granite-4.0-micro" if needed
PROMPT_QUANTIZE_4BIT = True  # set False if you prefer full precision

# ----------------------------
# Utilities
# ----------------------------

def _pick_device_and_ctx():
    """Return (device, dtype, autocast_ctx_fn) suitable across CUDA/MPS/CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # prefer bf16 if supported; else fp16
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        def _ctx():
            return torch.autocast(device_type="cuda", dtype=dtype)
        return device, dtype, _ctx
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32  # keep weights fp32 on MPS; autocast to fp16 for compute
        def _ctx():
            return torch.autocast(device_type="mps", dtype=torch.float16)
        return device, dtype, _ctx
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        def _ctx():
            return contextlib.nullcontext()
        return device, dtype, _ctx


def _safe_input_path(prompt_text):
    """Console fallback when Tk is unavailable or fails."""
    p = input(f"{Fore.YELLOW}{prompt_text} (enter a valid path): ").strip()
    return p if p else None


def select_folder(title="Select Folder"):
    """Try Tk file dialog; fall back to console path input in headless envs."""
    if _HAS_TK:
        try:
            root = tk.Tk()
            root.withdraw()
            selected_dir = filedialog.askdirectory(title=title)
            return selected_dir or None
        except Exception:
            pass
    # Fallback
    return _safe_input_path(title)


def set_generation_theme():
    theme = input(f"{Fore.YELLOW}Enter the theme for image generation: ").strip()
    return theme


def set_detection_classes():
    classes_input = input(f"{Fore.YELLOW}Enter detection classes (comma-separated): ").strip()
    classes = [cls.strip() for cls in classes_input.split(',') if cls.strip()]
    return classes


def set_numeric_input(prompt_msg):
    while True:
        try:
            value = int(input(prompt_msg).strip())
            if value > 0:
                return value
            else:
                print(f"{Fore.RED}Value must be greater than zero.")
        except ValueError:
            print(f"{Fore.RED}Invalid input. Please enter a positive number.")


def set_prepends():
    prepends_input = input(
        f"{Fore.YELLOW}Enter prepend modifiers (comma-separated, e.g., 'Distant faraway shot, Close-up, Side view'): "
    ).strip()
    prepends = [p.strip() for p in prepends_input.split(',') if p.strip()]
    return prepends


# ----------------------------
# Main class
# ----------------------------

class FluxCapacitor:
    """
    End-to-end: prompt -> diffuse image -> OD -> YOLO labels -> split -> YAML.
    """

    def __init__(self, num_augmentations=5):
        self.num_augmentations = num_augmentations

        self.device, self.dtype, self.autocast_ctx = _pick_device_and_ctx()
        print(f"{Fore.CYAN}Using device: {self.device}  (dtype policy: {self.dtype})")

        # Florence (object detection) components
        self.processor = None
        self.model = None

        # Diffusion pipeline
        self.pipeline = None

        # Local LLM for prompt generation
        self.prompt_tokenizer = None
        self.prompt_model = None

        # Dataset / detection
        self.dataset_dir = None
        self.target_class = None
        self.detection_class = None
        self._class_to_id = {}  # lowercased mapping

        # Options
        self.enable_background_generation = True
        self.prepends = []

        # Collection to defer splitting until the very end
        self._accepted_image_paths = []

    # ------------- Loading models -------------

    def load_pipeline(self, model_dir):
        """
        Load diffusion pipeline from a local model directory.
        Only checks for model_index.json; avoids brittle shard enumerations.
        """
        try:
            model_index_file = os.path.join(model_dir, "model_index.json")
            if not os.path.exists(model_index_file):
                print(f"{Fore.RED}File not found: {model_index_file}")
                return None

            pipe = DiffusionPipeline.from_pretrained(
                model_dir,
                torch_dtype=(self.dtype if self.dtype != torch.float32 else None),
                local_files_only=True,
                cache_dir=model_dir
            ).to(self.device)

            # Memory helpers; safe no-ops if unsupported
            with contextlib.suppress(Exception):
                pipe.enable_attention_slicing()
            with contextlib.suppress(Exception):
                pipe.enable_vae_slicing()
            with contextlib.suppress(Exception):
                pipe.enable_sequential_cpu_offload()  # if accelerate is present

            print(f"{Fore.GREEN}Diffusion model loaded successfully!")
            return pipe
        except Exception as e:
            print(f"{Fore.RED}Failed to initialize the diffusion model.")
            print(f"{Fore.RED}Error: {e}")
            return None

    def init_florence_model(self, florence_model_dir):
        """
        Initialize Florence-2 (or compatible) vision-language model for <OD>.
        """
        print(f"{Fore.CYAN}Loading Florence-2 model...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                florence_model_dir, trust_remote_code=True
            ).eval().to(self.device)

            # Only cast on CUDA to preferred dtype
            if self.device.type == "cuda":
                self.model = self.model.to(self.dtype)

            self.processor = AutoProcessor.from_pretrained(
                florence_model_dir, trust_remote_code=True
            )
            print(f"{Fore.GREEN}Florence-2 model loaded successfully!")
        except Exception as e:
            print(f"{Fore.RED}Error loading Florence-2 model: {e}")

    def init_local_prompt_model(self,
                                model_id: str = PROMPT_MODEL_ID,
                                quantize_4bit: bool = PROMPT_QUANTIZE_4BIT):
        """
        Initialize a local Granite 3B instruct model for prompt generation.
        """
        print(f"{Fore.CYAN}Loading local prompt model: {model_id}")
        try:
            trust_rc = True  # Granite uses custom components
            self.prompt_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_rc)

            model_kwargs = {"trust_remote_code": trust_rc, "device_map": "auto"}
            if self.device.type == "cuda":
                model_kwargs["torch_dtype"] = self.dtype

            if quantize_4bit and _HAS_BNB:
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=(torch.float16 if self.device.type == "cuda" else torch.float32),
                    bnb_4bit_use_double_quant=True
                )
                model_kwargs["quantization_config"] = bnb

            self.prompt_model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs).eval()
            print(f"{Fore.GREEN}Local prompt model ready.")
        except Exception as e:
            print(f"{Fore.RED}Error loading local prompt model: {e}")

    # ------------- Prompt generation via local LLM -------------

    def generate_prompt(self, theme, prompt_history, exclude_objects=False, max_retries=2):
        """
        Local Granite-backed prompt generator (replaces OpenAI).
        Returns (base_prompt, modified_prompts).
        """
        if self.prompt_model is None or self.prompt_tokenizer is None:
            print(f"{Fore.RED}Prompt model not initialized. Please load the local prompt model first.")
            return None, []

        # Build instruction
        if exclude_objects:
            user_text = (
                "Generate ONE short, unique description (<= 20 words) for a hyperrealistic, photorealistic EARTH environment "
                "with NO animals of any kind. Output only the description."
            )
        else:
            user_text = (
                f"Generate ONE short, unique description (<= 20 words) for a photorealistic {theme} in a random EARTH environment. "
                "Output only the description."
            )

        chat = [
            {"role": "system", "content": "You are a creative assistant generating diffusion image prompts. Be specific, safe, and concise."},
            {"role": "user", "content": user_text},
        ]

        # Try bounded number of retries to avoid recursion
        for attempt in range(max_retries + 1):
            try:
                tok = self.prompt_tokenizer
                mdl = self.prompt_model
                device = mdl.device

                chat_str = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                inputs = tok(chat_str, return_tensors="pt").to(device)

                with torch.no_grad():
                    gen_ids = mdl.generate(
                        **inputs,
                        do_sample=True,
                        temperature=0.9 if attempt == 0 else 1.0,
                        top_p=0.95 if attempt == 0 else 0.9,
                        max_new_tokens=64,
                        repetition_penalty=1.05 if attempt == 0 else 1.01,
                        eos_token_id=tok.eos_token_id,
                        pad_token_id=tok.eos_token_id,
                    )

                new_tokens = gen_ids[:, inputs["input_ids"].shape[1]:]
                text = tok.decode(new_tokens[0], skip_special_tokens=True).strip()
                base_prompt = text.splitlines()[0].strip()

                if base_prompt and base_prompt not in prompt_history and len(base_prompt) >= 5:
                    prompt_history.add(base_prompt)
                    modified_prompts = [f"{prepend} {base_prompt}".strip() for prepend in self.prepends]
                    prompt_history.update(modified_prompts)
                    print(f"{Fore.GREEN}Generated base prompt: {base_prompt}")
                    return base_prompt, modified_prompts

                print(f"{Fore.YELLOW}Duplicate/short prompt on attempt {attempt+1}; retrying...")
            except Exception as e:
                print(f"{Fore.RED}Local LLM prompt generation failed on attempt {attempt+1}: {e}")

        print(f"{Fore.RED}Failed to produce a novel prompt.")
        return None, []

    # ------------- Image generation / OD / Annotations -------------

    def generate_image(self, prompt, output_dir, seed):
        """
        Generate an image with the diffusion model.
        """
        print(f"{Fore.CYAN}Generating image with prompt: {prompt}, seed: {seed}")
        try:
            # Generators are safest on CPU to be device-agnostic
            g = torch.Generator(device="cpu").manual_seed(seed)

            image = self.pipeline(
                prompt,
                height=RESOLUTION[0],
                width=RESOLUTION[1],
                guidance_scale=GUIDANCE_SCALE,
                output_type="pil",
                num_inference_steps=NUM_INFERENCE_STEPS,
                generator=g
            ).images[0]

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # include microseconds to avoid collisions
            image_filename = f"flux_output_{timestamp}_seed{seed}.png"
            image_path = os.path.join(output_dir, image_filename)
            image.save(image_path)
            print(f"{Fore.GREEN}Image saved at {image_path}")

            if SHOW_IMAGE:
                with contextlib.suppress(Exception):
                    image.show()

            return image_path

        except Exception as e:
            print(f"{Fore.RED}Failed to generate the image.")
            print(f"{Fore.RED}Error: {e}")
            return None

    def run_object_detection(self, image):
        """
        Run Florence-2 object detection (<OD>) on a PIL image.
        """
        print(f"{Fore.CYAN}Running object detection...")
        try:
            task_prompt = '<OD>'
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device)
            original_size = image.size

            # Cast float tensors to proper dtype
            for k, v in inputs.items():
                if torch.is_floating_point(v):
                    inputs[k] = v.to(dtype=(self.dtype if self.device.type == "cuda" else torch.float32))

            with self.autocast_ctx():
                generated_ids = self.model.generate(
                    input_ids=inputs.get("input_ids"),
                    pixel_values=inputs.get("pixel_values"),
                    max_new_tokens=1024,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=1,
                )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            print(f"{Fore.GREEN}Generated text from object detection: {generated_text[:120]}...")
            parsed = self.processor.post_process_generation(generated_text, task=task_prompt, image_size=original_size)

            if image.size != original_size:
                raise ValueError(f"Image size changed during processing. Original: {original_size}, New: {image.size}")

            return parsed
        except Exception as e:
            print(f"{Fore.RED}Error during object detection: {e}")
            return None

    def save_yolo_annotations(self, image_path, results, image_size):
        """
        Save detection results in YOLO format.
        """
        print(f"{Fore.CYAN}Saving YOLO annotations for: {image_path}")
        try:
            txt_path = image_path.replace('.png', '.txt')

            od = results.get('<OD>') if isinstance(results, dict) else None
            if not od or 'bboxes' not in od or 'labels' not in od:
                print(f"{Fore.YELLOW}No OD schema in results; skipping annotations for {image_path}")
                return

            bboxes = od['bboxes']
            labels = od['labels']
            img_w, img_h = image_size

            with open(txt_path, 'w') as f:
                for bbox, label in zip(bboxes, labels):
                    lid = self._class_to_id.get(str(label).lower())
                    if lid is None:
                        continue
                    x1, y1, x2, y2 = bbox
                    if x2 <= x1 or y2 <= y1:
                        continue
                    x_center = max(0.0, min(1.0, ((x1 + x2) / 2.0) / img_w))
                    y_center = max(0.0, min(1.0, ((y1 + y2) / 2.0) / img_h))
                    w = max(0.0, min(1.0, (x2 - x1) / img_w))
                    h = max(0.0, min(1.0, (y2 - y1) / img_h))
                    f.write(f"{lid} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

            print(f"{Fore.GREEN}Annotations saved at: {txt_path}")
        except Exception as e:
            print(f"{Fore.RED}Error saving YOLO annotations: {e}")

    def remove_invalid_image(self, image_path):
        """Remove an image file that does not meet detection criteria."""
        try:
            os.remove(image_path)
            lbl = image_path.replace('.png', '.txt')
            with contextlib.suppress(Exception):
                if os.path.exists(lbl):
                    os.remove(lbl)
            print(f"{Fore.YELLOW}Removed image without target classes: {image_path}")
        except OSError as e:
            print(f"{Fore.RED}Error removing invalid image {image_path}: {e}")

    def apply_augmentations(self, image):
        """
        Apply a suite of augmentations and return images.
        """
        print(f"{Fore.CYAN}Applying augmentations...")
        try:
            transforms = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)),
                T.Resize(RESOLUTION),
            ])

            augmented_images = []
            for i in range(self.num_augmentations):
                augmented_images.append(transforms(image))
                print(f"{Fore.GREEN}Generated augmented image {i + 1}/{self.num_augmentations}")

            # Grayscale
            grayscale_image = T.Grayscale(num_output_channels=3)(image)
            augmented_images.append(grayscale_image)
            print(f"{Fore.GREEN}Generated grayscale image")

            # Black/white thresholded
            bw_image = image.convert("L").point(lambda x: 0 if x < 128 else 255, '1').convert("RGB")
            augmented_images.append(bw_image)
            print(f"{Fore.GREEN}Generated black-and-white image")

            return augmented_images
        except Exception as e:
            print(f"{Fore.RED}Error during augmentation: {e}")
            return []

    def process_image(self, image_path, output_dir):
        """
        Run OD/augment/annotate pipeline for a single base image.
        Returns list of accepted image paths (original + augmentations) or None.
        """
        print(f"{Fore.CYAN}Processing image: {image_path}")
        try:
            img = Image.open(image_path).convert("RGB")
            results = self.run_object_detection(img)

            if results is None:
                print(f"{Fore.YELLOW}No detections found for {image_path}")
                self.remove_invalid_image(image_path)
                return None

            detected_labels = results.get('<OD>', {}).get('labels', [])
            if not any(str(label).lower() in self._class_to_id for label in detected_labels):
                print(f"{Fore.YELLOW}Removing image with no target classes: {image_path}")
                self.remove_invalid_image(image_path)
                return None

            # Save original into final dataset root; annotations saved alongside
            original_image_path = os.path.join(output_dir, os.path.basename(image_path))
            img.save(original_image_path)
            self.save_yolo_annotations(original_image_path, results, img.size)

            accepted_paths = [original_image_path]

            augmented_images = self.apply_augmentations(img)
            for idx, augmented_img in enumerate(augmented_images):
                aug_name = f"aug_{idx}_{os.path.basename(image_path)}"
                augmented_img_path = os.path.join(output_dir, aug_name)
                augmented_img.save(augmented_img_path)

                augmented_results = self.run_object_detection(augmented_img)
                if augmented_results is not None:
                    self.save_yolo_annotations(augmented_img_path, augmented_results, augmented_img.size)
                    accepted_paths.append(augmented_img_path)
                else:
                    print(f"{Fore.YELLOW}No detections found for augmented image {augmented_img_path}")
                    self.remove_invalid_image(augmented_img_path)

            print(f"{Fore.GREEN}Processing complete for: {image_path}")
            return accepted_paths

        except Exception as e:
            print(f"{Fore.RED}Error processing image {image_path}: {e}")
            return None

    # ------------- Dataset organization -------------

    def move_files(self, paths, image_dir, label_dir):
        """
        Move image and annotation files to their split directories.
        Uses 'move' to avoid duplicated copies.
        """
        for image_path in paths:
            image_filename = os.path.basename(image_path)
            label_path = image_path.replace('.png', '.txt')

            if os.path.exists(image_path) and os.path.exists(label_path):
                try:
                    shutil.move(image_path, os.path.join(image_dir, image_filename))
                    shutil.move(label_path, os.path.join(label_dir, os.path.basename(label_path)))
                    print(f"{Fore.GREEN}Moved image and label: {image_filename}")
                except Exception as e:
                    print(f"{Fore.RED}Error moving image or label for {image_filename}: {e}")
            else:
                print(f"{Fore.YELLOW}Skipping {image_filename} due to missing or empty label.")

    def create_dataset_directories(self):
        """
        Create dataset directories (train, val, eval) under dataset_dir/images and dataset_dir/labels.
        """
        try:
            train_images_dir = os.path.join(self.dataset_dir, 'images', 'train')
            val_images_dir = os.path.join(self.dataset_dir, 'images', 'val')
            eval_images_dir = os.path.join(self.dataset_dir, 'images', 'eval')

            train_labels_dir = os.path.join(self.dataset_dir, 'labels', 'train')
            val_labels_dir = os.path.join(self.dataset_dir, 'labels', 'val')
            eval_labels_dir = os.path.join(self.dataset_dir, 'labels', 'eval')

            os.makedirs(train_images_dir, exist_ok=True)
            os.makedirs(val_images_dir, exist_ok=True)
            os.makedirs(eval_images_dir, exist_ok=True)
            os.makedirs(train_labels_dir, exist_ok=True)
            os.makedirs(val_labels_dir, exist_ok=True)
            os.makedirs(eval_labels_dir, exist_ok=True)

            print(f"{Fore.GREEN}Dataset directories created/verified.")
            return train_images_dir, val_images_dir, eval_images_dir, train_labels_dir, val_labels_dir, eval_labels_dir
        except Exception as e:
            print(f"{Fore.RED}Error creating dataset directories: {e}")
            return None, None, None, None, None, None

    def split_dataset(self, image_paths):
        """
        Split all accepted images into train/val/eval once (no repeated I/O).
        """
        try:
            print(f"{Fore.CYAN}Splitting dataset into train, validation, and eval sets...")

            dirs = self.create_dataset_directories()
            if any(d is None for d in dirs):
                print(f"{Fore.RED}Dataset directories unavailable; aborting split.")
                return

            train_images_dir, val_images_dir, eval_images_dir, train_labels_dir, val_labels_dir, eval_labels_dir = dirs

            images = list(set(image_paths))  # de-dup
            total_images = len(images)

            if total_images == 0:
                print(f"{Fore.YELLOW}No images to split.")
                return

            if total_images < 8:
                train_size = max(1, total_images - 2)
                val_size = max(1, (total_images - train_size) // 2)
                eval_size = total_images - train_size - val_size
            else:
                train_size = total_images // 2
                val_size = total_images // 4
                eval_size = total_images - train_size - val_size

            selected_train_paths = random.sample(images, train_size)
            remaining = [p for p in images if p not in selected_train_paths]
            val_paths = random.sample(remaining, val_size)
            eval_paths = [p for p in remaining if p not in val_paths]

            print(f"{Fore.GREEN}Total: {total_images}. Train: {len(selected_train_paths)}, Val: {len(val_paths)}, Eval: {len(eval_paths)}")

            self.move_files(selected_train_paths, train_images_dir, train_labels_dir)
            self.move_files(val_paths, val_images_dir, val_labels_dir)
            self.move_files(eval_paths, eval_images_dir, eval_labels_dir)

        except Exception as e:
            print(f"{Fore.RED}Error during dataset splitting: {e}")

    def generate_yaml(self):
        """
        Generate a YAML file for the dataset. Include 'test' mapped to eval.
        """
        try:
            if not os.path.exists(self.dataset_dir):
                os.makedirs(self.dataset_dir)

            yaml_content = (
                "path: " + self.dataset_dir.replace("\\", "/") + "\n"
                "train: images/train\n"
                "val: images/val\n"
                "test: images/eval\n"
                "nc: " + str(len(self.detection_class)) + "\n"
                "names: " + str(self.detection_class) + "\n"
            )

            yaml_path = os.path.join(self.dataset_dir, 'data.yaml')
            with open(yaml_path, 'w') as yaml_file:
                yaml_file.write(yaml_content)
            print(f"{Fore.GREEN}YAML file generated at: {yaml_path}")
        except Exception as e:
            print(f"{Fore.RED}Error generating YAML file: {e}")

    # ------------- Backgrounds (no objects) -------------

    def create_empty_annotation(self, annotation_path):
        try:
            with open(annotation_path, 'w'):
                pass
            print(f"{Fore.GREEN}Empty annotation file created: {annotation_path}")
        except Exception as e:
            print(f"{Fore.RED}Error creating empty annotation file for {annotation_path}: {e}")

    def generate_background_images(self, prompt_history, num_batches, num_seeds):
        """
        Generate background images with no objects; write directly to split dirs with empty labels.
        """
        if not self.enable_background_generation:
            return

        dataset_splits = ['train', 'val', 'eval']
        for split in dataset_splits:
            split_dir = os.path.join(self.dataset_dir, 'images', split)
            label_dir = os.path.join(self.dataset_dir, 'labels', split)
            os.makedirs(split_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)

            for batch_index in range(num_batches):
                for seed_index in range(num_seeds):
                    base_prompt, _ = self.generate_prompt('background', prompt_history, exclude_objects=True)
                    if base_prompt:
                        seed = random.randint(0, 10000)
                        g = torch.Generator(device="cpu").manual_seed(seed)
                        background_image = self.pipeline(
                            base_prompt,
                            height=RESOLUTION[0],
                            width=RESOLUTION[1],
                            guidance_scale=GUIDANCE_SCALE,
                            num_inference_steps=NUM_INFERENCE_STEPS,
                            generator=g
                        ).images[0]

                        img_filename = f"{split}_bg_{batch_index+1}_{seed_index+1}.png"
                        img_path = os.path.join(split_dir, img_filename)
                        background_image.save(img_path)

                        annotation_filename = img_filename.replace('.png', '.txt')
                        annotation_path = os.path.join(label_dir, annotation_filename)
                        self.create_empty_annotation(annotation_path)

                        print(f"{Fore.GREEN}Background image saved: {img_path}")
                    else:
                        print(f"{Fore.RED}Failed to generate a valid background prompt.")

# ----------------------------
# Interactive menu / main
# ----------------------------

def main():
    print(f"\n{Fore.GREEN}{Style.BRIGHT}Welcome to the FLUX Capacitor!{Style.RESET_ALL}")
    print(f"{Fore.GREEN}A seamless image diffusion to computer vision dataset pipeline!{Style.RESET_ALL}")

    flux = FluxCapacitor()

    flux_model_dir = None
    florence_model_dir = None
    output_dir = None
    theme = None
    detection_classes = None
    num_batches = 1
    num_seeds = 1

    while True:
        flux_status = f"{Fore.GREEN}Set" if flux_model_dir else f"{Fore.RED}Not Set"
        florence_status = f"{Fore.GREEN}Set" if florence_model_dir else f"{Fore.RED}Not Set"
        output_status = f"{Fore.GREEN}Set" if output_dir else f"{Fore.RED}Not Set"
        theme_status = f"{Fore.GREEN}Set" if theme else f"{Fore.RED}Not Set"
        classes_status = f"{Fore.GREEN}Set" if detection_classes else f"{Fore.RED}Not Set"
        prepends_status = f"{Fore.GREEN}Set" if flux.prepends else f"{Fore.RED}Not Set"
        prompt_status = f"{Fore.GREEN}Loaded" if flux.prompt_model else f"{Fore.RED}Not Loaded"

        print(f"\n{Fore.BLUE}{Style.BRIGHT}{'='*40}")
        print(f"{Fore.CYAN}{Style.BRIGHT}--- FLUX Capacitor Menu ---{Style.RESET_ALL}")
        print(f"{Fore.BLUE}{'='*40}{Style.RESET_ALL}")

        print(f"{Fore.YELLOW}{Style.BRIGHT}\n{'-'*10} Image Generation {'-'*10}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}1. Generate and annotate images")

        print(f"{Fore.YELLOW}{Style.BRIGHT}\n{'-'*10} Directory Setup {'-'*10}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}2. Set output directory [{output_status}]")
        print(f"{Fore.CYAN}3. Set FLUX model directory [{flux_status}]")
        print(f"{Fore.CYAN}4. Set Florence model directory [{florence_status}]")

        print(f"{Fore.YELLOW}{Style.BRIGHT}\n{'-'*10} Prompt Model & Options {'-'*10}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}5. Load local Granite prompt model ({PROMPT_MODEL_ID}, 4bit={'ON' if PROMPT_QUANTIZE_4BIT and _HAS_BNB else 'OFF'}) [{prompt_status}]")
        print(f"{Fore.CYAN}6. Set generation theme [{theme_status}]")
        print(f"{Fore.CYAN}7. Set detection classes [{classes_status}]")
        print(f"{Fore.CYAN}8. Set number of image prompts (current: {num_batches})")
        print(f"{Fore.CYAN}9. Set number of seeds per prompt (current: {num_seeds})")
        print(f"{Fore.CYAN}10. Toggle background image generation (current: {'ON' if flux.enable_background_generation else 'OFF'})")
        print(f"{Fore.CYAN}12. Set prompt prepends (current: {prepends_status})")

        print(f"{Fore.YELLOW}{Style.BRIGHT}\n{'-'*10} Exit {'-'*10}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}11. Exit")

        choice = input(f"{Fore.YELLOW}{Style.BRIGHT}Please enter your choice (1-12): {Style.RESET_ALL}").strip()

        if choice == "1":
            if not (output_dir and flux_model_dir and florence_model_dir and theme and detection_classes and flux.prompt_model):
                print(f"{Fore.RED}Error: Please ensure all settings are configured correctly (including loading the local prompt model).")
                continue

            # Set detection classes and build class-id map
            flux.detection_class = detection_classes
            flux._class_to_id = {c.lower(): i for i, c in enumerate(flux.detection_class)}
            flux.target_class = detection_classes[0]

            print(f"{Fore.GREEN}Starting image generation process...")
            prompt_history = set()

            all_paths = []  # accumulate for a single split at the end

            for batch in range(num_batches):
                base_prompt, modified_prompts = flux.generate_prompt(theme, prompt_history)

                if base_prompt:
                    for _ in range(num_seeds):
                        seed = random.randint(0, 10_000)
                        image_path = flux.generate_image(base_prompt, output_dir, seed)
                        if image_path:
                            processed = flux.process_image(image_path, output_dir)
                            if processed:
                                all_paths.extend(processed)
                else:
                    print(f"{Fore.RED}Failed to generate base prompt for batch {batch+1}.")

                for mod in modified_prompts:
                    for _ in range(num_seeds):
                        seed = random.randint(0, 10_000)
                        mod_path = flux.generate_image(mod, output_dir, seed)
                        if mod_path:
                            processed = flux.process_image(mod_path, output_dir)
                            if processed:
                                all_paths.extend(processed)

            if flux.enable_background_generation:
                flux.generate_background_images(prompt_history, num_batches, num_seeds)

            # Single split at the end
            if all_paths:
                flux.split_dataset(all_paths)

            flux.generate_yaml()
            print(f"{Fore.GREEN}Successfully generated and annotated images.\n")

        elif choice == "2":
            selected_dir = select_folder("Select Output Directory")
            if selected_dir:
                output_dir = selected_dir
                flux.dataset_dir = output_dir
                print(f"{Fore.GREEN}Output directory set to: {output_dir}")

        elif choice == "3":
            selected_dir = select_folder("Select FLUX Diffusion Model Directory")
            if selected_dir:
                flux_model_dir = selected_dir
                flux.pipeline = flux.load_pipeline(flux_model_dir)
                print(f"{Fore.GREEN}FLUX diffusion model folder set to: {flux_model_dir}")

        elif choice == "4":
            selected_dir = select_folder("Select Florence Model Directory")
            if selected_dir:
                florence_model_dir = selected_dir
                flux.init_florence_model(florence_model_dir)
                print(f"{Fore.GREEN}Florence model folder set to: {florence_model_dir}")

        elif choice == "5":
            flux.init_local_prompt_model(PROMPT_MODEL_ID, PROMPT_QUANTIZE_4BIT)
            # No status update needed; shown at top next loop

        elif choice == "6":
            theme = set_generation_theme()
            print(f"{Fore.GREEN}Generation theme set to: {theme}")

        elif choice == "7":
            detection_classes = set_detection_classes()
            print(f"{Fore.GREEN}Detection classes set to: {', '.join(detection_classes)}")

        elif choice == "8":
            num_batches = set_numeric_input(f"{Fore.YELLOW}Enter the number of image prompts: ")

        elif choice == "9":
            num_seeds = set_numeric_input(f"{Fore.YELLOW}Enter the number of seeds per prompt: ")

        elif choice == "10":
            flux.enable_background_generation = not flux.enable_background_generation
            print(f"{Fore.GREEN}Background image generation is now {'ON' if flux.enable_background_generation else 'OFF'}")

        elif choice == "12":
            flux.prepends = set_prepends()
            print(f"{Fore.GREEN}Prepend modifiers set to: {', '.join(flux.prepends)}")

        elif choice == "11":
            print(f"{Fore.YELLOW}Exiting FLUX Capacitor. Goodbye!")
            break

        else:
            print(f"{Fore.RED}Invalid choice. Please enter a valid option.")


if __name__ == "__main__":
    # A touch of determinism for augment RNGs & seeds
    random.seed(42)
    try:
        torch.manual_seed(42)
    except Exception:
        pass
    main()
