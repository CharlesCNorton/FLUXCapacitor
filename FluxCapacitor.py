import os
import torch
from diffusers import DiffusionPipeline
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from datetime import datetime
import openai
import random
import tkinter as tk
from tkinter import filedialog
import torchvision.transforms as T
import shutil
from colorama import Fore, init

init(autoreset=True)

RESOLUTION = (1024, 1024)
NUM_INFERENCE_STEPS = 50
SHOW_IMAGE = False

class FluxCapacitor:

    def __init__(self, num_augmentations=5):
        self.num_augmentations = num_augmentations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        self.target_class = None
        self.detection_class = None
        self.pipeline = None
        self.dataset_dir = None
        self.generate_background_images = True
        print(f"{Fore.CYAN}Using device: {self.device}")

    def load_pipeline(self, model_dir):
        """Load the FLUX diffusion model pipeline from the selected folder."""
        try:
            transformer_dir = os.path.join(model_dir, "transformer")
            model_files = [
                os.path.join(transformer_dir, "diffusion_pytorch_model-00001-of-00003.safetensors"),
                os.path.join(transformer_dir, "diffusion_pytorch_model-00002-of-00003.safetensors"),
                os.path.join(transformer_dir, "diffusion_pytorch_model-00003-of-00003.safetensors"),
            ]
            config_file = os.path.join(transformer_dir, "transformer_config.json")
            model_index_file = os.path.join(model_dir, "model_index.json")

            for file in model_files + [config_file, model_index_file]:
                if not os.path.exists(file):
                    print(f"{Fore.RED}File not found: {file}")
                    return None

            pipe = DiffusionPipeline.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                cache_dir=model_dir
            ).to(self.device)
            print(f"{Fore.GREEN}Diffusion model loaded successfully!")
            return pipe
        except Exception as e:
            print(f"{Fore.RED}Failed to initialize the diffusion model.")
            print(f"{Fore.RED}Error: {e}")
            return None

    def init_florence_model(self, florence_model_dir):
        """Initialize the Florence-2 model using the selected model directory."""
        print(f"{Fore.CYAN}Loading Florence-2 model...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                florence_model_dir, trust_remote_code=True
            ).eval().to(self.device).half()
            self.processor = AutoProcessor.from_pretrained(florence_model_dir, trust_remote_code=True)
            print(f"{Fore.GREEN}Florence-2 model loaded successfully!")
        except Exception as e:
            print(f"{Fore.RED}Error loading Florence-2 model: {e}")

    def generate_prompt(self, theme, prompt_history, exclude_objects=False):
        """Generate a prompt using OpenAI GPT-4 with exclusions or themes."""
        try:
            if exclude_objects:
                prompt_content = "Generate a short unique description for a hyperrealistic and photorealistic environment on Earth that contains no animals of any kind at all. Use for image generation in a diffusion model."
            else:
                prompt_content = f"Generate a single short unique description for a photorealistic {theme} in a random Earth environment."

            print(f"{Fore.CYAN}Generating prompt for theme: {theme}, exclude_objects: {exclude_objects}")
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a creative assistant generating image prompts."},
                    {"role": "user", "content": prompt_content}
                ],
                max_tokens=70,
                temperature=0.9,
                n=1,
            )
            prompt = response['choices'][0]['message']['content'].strip()
            print(f"{Fore.GREEN}Generated prompt: {prompt}")
            if prompt in prompt_history or len(prompt) < 10:
                print(f"{Fore.YELLOW}Prompt is either a duplicate or invalid, generating again...")
                return self.generate_prompt(theme, prompt_history, exclude_objects)
            prompt_history.add(prompt)
            return prompt
        except Exception as e:
            print(f"{Fore.RED}Failed to generate prompt using GPT-4 mini.")
            print(f"{Fore.RED}Error: {e}")
            return None

    def generate_image(self, prompt, output_dir, seed):
        """Generate an image from the diffusion model."""
        print(f"{Fore.CYAN}Generating image with prompt: {prompt}, seed: {seed}")
        try:
            image = self.pipeline(
                prompt,
                height=RESOLUTION[0],
                width=RESOLUTION[1],
                guidance_scale=3.5,
                output_type="pil",
                num_inference_steps=NUM_INFERENCE_STEPS,
                generator=torch.Generator(self.device).manual_seed(seed)
            ).images[0]

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"flux_output_{timestamp}_seed{seed}.png"
            image_path = os.path.join(output_dir, image_filename)
            image.save(image_path)
            print(f"{Fore.GREEN}Image saved at {image_path}")

            if SHOW_IMAGE:
                image.show()

            return image_path

        except Exception as e:
            print(f"{Fore.RED}Failed to generate the image.")
            print(f"{Fore.RED}Error: {e}")
            return None

    def run_object_detection(self, image):
        """Run object detection using Florence-2 model."""
        print(f"{Fore.CYAN}Running object detection...")
        try:
            task_prompt = '<OD>'
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device)
            original_size = image.size

            for k, v in inputs.items():
                if torch.is_floating_point(v):
                    inputs[k] = v.half()

            with torch.amp.autocast("cuda"):
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs.get("pixel_values"),
                    max_new_tokens=1024,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=1,
                )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            print(f"{Fore.GREEN}Generated text from object detection:", generated_text)
            parsed_answer = self.processor.post_process_generation(generated_text, task=task_prompt, image_size=original_size)

            if image.size != original_size:
                raise ValueError(f"Image size changed during processing. Original: {original_size}, New: {image.size}")

            return parsed_answer
        except Exception as e:
            print(f"{Fore.RED}Error during object detection: {e}")
            return None

    def save_yolo_annotations(self, image_path, results, image_size):
        """Save YOLO-style annotations based on object detection results."""
        print(f"{Fore.CYAN}Saving YOLO annotations for: {image_path}")
        try:
            txt_path = image_path.replace('.png', '.txt')

            with open(txt_path, 'w') as f:
                for bbox, label in zip(results['<OD>']['bboxes'], results['<OD>']['labels']):
                    if label.lower() in [cls.lower() for cls in self.detection_class]:
                        class_id = 0
                        x1, y1, x2, y2 = bbox
                        img_width, img_height = image_size
                        x_center = (x1 + x2) / 2 / img_width
                        y_center = (y1 + y2) / 2 / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

            print(f"{Fore.GREEN}Annotations saved for detection class '{self.detection_class}' at: {txt_path}")
        except Exception as e:
            print(f"{Fore.RED}Error saving YOLO annotations: {e}")

    def draw_and_save_bbox(self, image_path, results):
        """Display bounding boxes on the image (optional)."""
        print(f"{Fore.CYAN}Processing bounding boxes for: {image_path} (not drawing on the image)")
        try:
            img = Image.open(image_path)

            for bbox, label in zip(results['<OD>']['bboxes'], results['<OD>']['labels']):
                if label.lower() in [cls.lower() for cls in self.detection_class]:
                    x1, y1, x2, y2 = bbox
                    print(f"{Fore.YELLOW}Detected bounding box: {x1}, {y1}, {x2}, {y2}, label: {label}")

        except Exception as e:
            print(f"{Fore.RED}Error processing bounding boxes: {e}")

    def apply_augmentations(self, image):
        """Apply augmentations to an image."""
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
                augmented_image = transforms(image)
                augmented_images.append(augmented_image)
                print(f"{Fore.GREEN}Generated augmented image {i + 1}/{self.num_augmentations}")

            grayscale_image = T.Grayscale(num_output_channels=3)(image)
            augmented_images.append(grayscale_image)
            print(f"{Fore.GREEN}Generated grayscale image")

            bw_image = image.convert("L").point(lambda x: 0 if x < 128 else 255, '1').convert("RGB")
            augmented_images.append(bw_image)
            print(f"{Fore.GREEN}Generated black-and-white image")

            return augmented_images
        except Exception as e:
            print(f"{Fore.RED}Error during augmentation: {e}")
            return []

    def process_image(self, image_path, output_dir):
        """Process the image, detect objects, and save annotations."""
        print(f"{Fore.CYAN}Processing image: {image_path}")
        try:
            img = Image.open(image_path).convert("RGB")
            results = self.run_object_detection(img)

            if results is None:
                print(f"{Fore.YELLOW}No detections found for {image_path}")
                return None

            original_image_path = os.path.join(output_dir, os.path.basename(image_path))
            img.save(original_image_path)
            self.save_yolo_annotations(original_image_path, results, img.size)
            self.draw_and_save_bbox(original_image_path, results)

            image_paths = [original_image_path]

            augmented_images = self.apply_augmentations(img)
            for idx, augmented_img in enumerate(augmented_images):
                augmented_img_path = os.path.join(output_dir, f"aug_{idx}_{os.path.basename(image_path)}")
                augmented_img.save(augmented_img_path)

                augmented_results = self.run_object_detection(augmented_img)
                if augmented_results is not None:
                    self.save_yolo_annotations(augmented_img_path, augmented_results, augmented_img.size)
                    self.draw_and_save_bbox(augmented_img_path, augmented_results)
                else:
                    print(f"{Fore.YELLOW}No detections found for augmented image {augmented_img_path}")

                image_paths.append(augmented_img_path)

            print(f"{Fore.GREEN}Processing complete for: {image_path}")
            return image_paths

        except Exception as e:
            print(f"{Fore.RED}Error processing image {image_path}: {e}")
            return None

    def generate_background_images(self, prompt_history, num_retry=5, num_backgrounds=3):
        """Generate background images and save annotations."""
        theme = "background"
        generated_images = []
        prompts = []

        dataset_splits = ['train', 'val', 'eval']
        for i, split in enumerate(dataset_splits):
            for attempt in range(num_retry):
                print(f"{Fore.CYAN}Attempt {attempt + 1} for generating background image for {split}")
                prompt = self.generate_prompt(theme, prompt_history, exclude_objects=True)
                if prompt:
                    seed = random.randint(0, 10000)

                    split_dir = os.path.join(self.dataset_dir, 'images', split)
                    label_dir = os.path.join(self.dataset_dir, 'labels', split)
                    os.makedirs(split_dir, exist_ok=True)
                    os.makedirs(label_dir, exist_ok=True)

                    image_filename = f"{split}_bg_{i+1}_seed{seed}.png"
                    image_path = os.path.join(split_dir, image_filename)

                    image = self.pipeline(
                        prompt,
                        height=RESOLUTION[0],
                        width=RESOLUTION[1],
                        guidance_scale=3.5,
                        output_type="pil",
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        generator=torch.Generator(self.device).manual_seed(seed)
                    ).images[0]

                    image.save(image_path)

                    if image_path:
                        image = Image.open(image_path).convert("RGB")
                        results = self.run_object_detection(image)
                        detected_labels = results.get('<OD>', {}).get('labels', []) if results else []

                        if not any(label.lower() == self.target_class.lower() for label in detected_labels):
                            print(f"{Fore.GREEN}Non-target objects detected but background is acceptable for {split}: {detected_labels}")
                            prompts.append(prompt)

                            annotation_filename = image_filename.replace('.png', '.txt')
                            annotation_path = os.path.join(label_dir, annotation_filename)
                            self.create_empty_annotation(annotation_path)

                            generated_images.append(image_path)
                            break
                    else:
                        print(f"{Fore.RED}Failed to generate image from prompt.")
                else:
                    print(f"{Fore.RED}Failed to generate a valid prompt for background.")

        if len(generated_images) == 0:
            print(f"{Fore.YELLOW}Warning: Could not generate any valid background images after several attempts.")
        else:
            print(f"{Fore.GREEN}Generated {len(generated_images)} unique background images for each dataset split.")

        return generated_images, prompts

    def create_empty_annotation(self, image_path):
        """Create an empty annotation file."""
        annotation_path = image_path.replace('.png', '.txt')
        try:
            with open(annotation_path, 'w') as f:
                pass
            print(f"{Fore.GREEN}Empty annotation file created: {annotation_path}")
        except Exception as e:
            print(f"{Fore.RED}Error creating empty annotation file for {image_path}: {e}")

    def move_files(self, paths, image_dir, label_dir):
        """Move image and annotation files to appropriate directories."""
        for image_path in paths:
            image_filename = os.path.basename(image_path)
            label_path = image_path.replace('.png', '.txt')

            if os.path.exists(image_path) and os.path.exists(label_path):
                try:
                    shutil.copy(image_path, os.path.join(image_dir, image_filename))
                    shutil.copy(label_path, os.path.join(label_dir, os.path.basename(label_path)))
                    print(f"{Fore.GREEN}Moved image and label: {image_filename}")
                except Exception as e:
                    print(f"{Fore.RED}Error copying image or label for {image_filename}: {e}")
            else:
                print(f"{Fore.YELLOW}Skipping image {image_filename} due to missing or empty label.")

    def split_dataset(self, image_paths):
        """Split the dataset into train, validation, and evaluation sets."""
        try:
            print(f"{Fore.CYAN}Splitting dataset into train, validation, and evaluation sets...")

            train_images_dir, val_images_dir, eval_images_dir, train_labels_dir, val_labels_dir, eval_labels_dir = self.create_dataset_directories()

            def split_class(images):
                total_images = len(images)

                if total_images < 8:
                    print(f"{Fore.YELLOW}Warning: Not enough images to split evenly. Only {total_images} images available.")
                    train_size = max(1, total_images - 2)
                    val_size = max(1, (total_images - train_size) // 2)
                    eval_size = total_images - train_size - val_size
                else:
                    train_size = 4
                    val_size = 2
                    eval_size = total_images - train_size - val_size

                train_paths = random.sample(images, train_size)
                remaining = [path for path in images if path not in train_paths]

                val_paths = random.sample(remaining, val_size)
                eval_paths = [path for path in remaining if path not in val_paths]

                print(f"{Fore.GREEN}Total images: {total_images}. Train: {len(train_paths)}, Val: {len(val_paths)}, Eval: {len(eval_paths)}")
                return train_paths, val_paths, eval_paths

            train_paths, val_paths, eval_paths = split_class(image_paths)

            self.move_files(train_paths, train_images_dir, train_labels_dir)
            self.move_files(val_paths, val_images_dir, val_labels_dir)
            self.move_files(eval_paths, eval_images_dir, eval_labels_dir)

        except Exception as e:
            print(f"{Fore.RED}Error during dataset splitting: {e}")

    def create_dataset_directories(self):
        """Create dataset directories for train, validation, and evaluation splits."""
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

            print(f"{Fore.GREEN}Dataset directories created successfully.")
            return train_images_dir, val_images_dir, eval_images_dir, train_labels_dir, val_labels_dir, eval_labels_dir
        except Exception as e:
            print(f"{Fore.RED}Error creating dataset directories: {e}")
            return None, None, None, None, None, None

    def generate_yaml(self):
        """Generate YAML configuration for the dataset."""
        try:
            if not os.path.exists(self.dataset_dir):
                os.makedirs(self.dataset_dir)

            yaml_content = (
                "path: " + self.dataset_dir.replace("\\", "/") + "  # Base dataset directory\n"
                "train: images/train  # Train images directory\n"
                "val: images/val  # Validation images directory\n"
                "nc: " + str(len(self.detection_class)) + "  # Number of classes\n"
                "names: " + str(self.detection_class) + "  # Class names\n"
            )

            yaml_path = os.path.join(self.dataset_dir, 'data.yaml')

            with open(yaml_path, 'w') as yaml_file:
                yaml_file.write(yaml_content)
            print(f"{Fore.GREEN}YAML file generated at: {yaml_path}")
        except Exception as e:
            print(f"{Fore.RED}Error generating YAML file: {e}")

def select_output_directory():
    """Open a file dialog to select the output directory."""
    root = tk.Tk()
    root.withdraw()
    output_dir = filedialog.askdirectory(title="Select Output Directory")
    return output_dir

def select_folder(title="Select Folder"):
    """Open a file dialog to select a folder."""
    root = tk.Tk()
    root.withdraw()
    selected_dir = filedialog.askdirectory(title=title)
    return selected_dir

def set_api_key():
    """Set the OpenAI API key from user input."""
    api_key = input(f"{Fore.YELLOW}Enter your OpenAI API key: ").strip()
    return api_key

def main():
    """Main function to run the interactive terminal menu."""
    print(f"\n{Fore.GREEN}Welcome to the FLUX Capacitor: A seamless image diffusion to computer vision dataset pipeline!{Fore.RESET}")

    flux_capacitor = FluxCapacitor()

    flux_model_dir = None
    florence_model_dir = None
    openai_api_key = openai.api_key
    output_dir = None

    while True:
        print(f"\n{Fore.BLUE}--- FLUX Capacitor Menu ---")
        print(f"{Fore.CYAN}1. Generate images, annotate, and organize dataset")
        print(f"{Fore.CYAN}2. Set output directory")
        print(f"{Fore.CYAN}3. Set FLUX diffusion model folder")
        print(f"{Fore.CYAN}4. Set Florence model folder")
        print(f"{Fore.CYAN}5. Set OpenAI API key")
        print(f"{Fore.CYAN}6. Toggle background image generation (current: {'ON' if flux_capacitor.generate_background_images else 'OFF'})")
        print(f"{Fore.CYAN}7. Exit")
        choice = input(f"{Fore.YELLOW}Please enter your choice (1/2/3/4/5/6/7): ").strip()

        if choice == "1":
            if output_dir is None:
                print(f"{Fore.RED}Error: Output directory not set. Please select option 2.")
                continue

            if flux_model_dir is None:
                print(f"{Fore.RED}Error: FLUX model folder not set. Please select option 3.")
                continue

            if florence_model_dir is None:
                print(f"{Fore.RED}Error: Florence model folder not set. Please select option 4.")
                continue

        elif choice == "2":
            selected_dir = select_folder("Select Output Directory")
            if selected_dir:
                output_dir = selected_dir
                print(f"{Fore.GREEN}Output directory set to: {output_dir}")

        elif choice == "3":
            selected_dir = select_folder("Select FLUX Diffusion Model Directory")
            if selected_dir:
                flux_model_dir = selected_dir
                flux_capacitor.pipeline = flux_capacitor.load_pipeline(flux_model_dir)
                print(f"{Fore.GREEN}FLUX diffusion model folder set to: {flux_model_dir}")

        elif choice == "4":
            selected_dir = select_folder("Select Florence Model Directory")
            if selected_dir:
                florence_model_dir = selected_dir
                flux_capacitor.init_florence_model(florence_model_dir)
                print(f"{Fore.GREEN}Florence model folder set to: {florence_model_dir}")

        elif choice == "5":
            openai_api_key = set_api_key()
            openai.api_key = openai_api_key
            print(f"{Fore.GREEN}OpenAI API key set.")

        elif choice == "6":
            flux_capacitor.generate_background_images = not flux_capacitor.generate_background_images
            print(f"{Fore.GREEN}Background image generation is now {'ON' if flux_capacitor.generate_background_images else 'OFF'}")

        elif choice == "7":
            print(f"{Fore.YELLOW}Exiting FLUX Capacitor. Goodbye!")
            break

        else:
            print(f"{Fore.RED}Invalid choice. Please enter a valid option.")


if __name__ == "__main__":
    main()
