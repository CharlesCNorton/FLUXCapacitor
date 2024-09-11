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
import colorama
from colorama import Fore, init, Style

init(autoreset=True)

RESOLUTION = (1024, 1024)
NUM_INFERENCE_STEPS = 50
SHOW_IMAGE = False

class FluxCapacitor:
    """
    FluxCapacitor is a class that handles generating prompts, images, and processing datasets
    for fine-tuning image models using diffusion pipelines. It supports augmentation, object
    detection, and custom image annotations.
    """

    def __init__(self, num_augmentations=5):
        """
        Initializes the FluxCapacitor object.

        Args:
            num_augmentations (int): Number of augmentations to apply to generated images.
        """
        self.num_augmentations = num_augmentations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        self.target_class = None
        self.detection_class = None
        self.pipeline = None
        self.dataset_dir = None
        self.enable_background_generation = True
        self.prepends = []
        print(f"{Fore.CYAN}Using device: {self.device}")

    def load_pipeline(self, model_dir):
        """
        Loads the diffusion pipeline model from a specified directory.

        Args:
            model_dir (str): Directory containing the model files.

        Returns:
            DiffusionPipeline: The loaded diffusion pipeline object.
        """
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
        """
        Initializes the Florence-2 model from a specified directory.

        Args:
            florence_model_dir (str): Directory containing the Florence-2 model files.
        """
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
        """
        Generates a prompt for image generation and applies user-defined prepends.

        Args:
            theme (str): The theme for prompt generation.
            prompt_history (set): A set to keep track of previously generated prompts.
            exclude_objects (bool): If True, generate a prompt without objects.

        Returns:
            tuple: A tuple containing the base prompt and a list of modified prompts.
        """
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
                max_tokens=73,
                temperature=0.9,
                n=1,
            )
            base_prompt = response['choices'][0]['message']['content'].strip()
            print(f"{Fore.GREEN}Generated base prompt: {base_prompt}")

            if base_prompt in prompt_history or len(base_prompt) < 10:
                print(f"{Fore.YELLOW}Prompt is either a duplicate or invalid, generating again...")
                return self.generate_prompt(theme, prompt_history, exclude_objects)

            prompt_history.add(base_prompt)

            modified_prompts = [f"{prepend} {base_prompt}" for prepend in self.prepends]
            prompt_history.update(modified_prompts)

            return base_prompt, modified_prompts
        except Exception as e:
            print(f"{Fore.RED}Failed to generate prompt using GPT-4 mini.")
            print(f"{Fore.RED}Error: {e}")
            return None, []

    def generate_image(self, prompt, output_dir, seed):
        """
        Generates an image using the diffusion model.

        Args:
            prompt (str): The prompt used for image generation.
            output_dir (str): The directory to save the generated image.
            seed (int): The random seed for generation.

        Returns:
            str: The file path of the generated image.
        """
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
        """
        Runs object detection on an image and returns detected objects.

        Args:
            image (PIL.Image): The image to perform object detection on.

        Returns:
            dict: A dictionary containing object detection results (labels, bboxes, etc.).
        """
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
        """
        Saves the object detection results in YOLO format.

        Args:
            image_path (str): The file path of the image being annotated.
            results (dict): The object detection results (labels, bboxes, etc.).
            image_size (tuple): The size of the image (width, height).
        """
        print(f"{Fore.CYAN}Saving YOLO annotations for: {image_path}")
        try:
            txt_path = image_path.replace('.png', '.txt')

            with open(txt_path, 'w') as f:
                for bbox, label in zip(results['<OD>']['bboxes'], results['<OD>']['labels']):
                    if label.lower() in [cls.lower() for cls in self.detection_class]:
                        class_id = self.detection_class.index(label.lower())
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

    def remove_invalid_image(self, image_path):
        """
        Removes an image file that does not meet the detection criteria.

        Args:
            image_path (str): The file path of the invalid image.
        """
        try:
            os.remove(image_path)
            print(f"{Fore.YELLOW}Removed image without target classes: {image_path}")
        except OSError as e:
            print(f"{Fore.RED}Error removing invalid image {image_path}: {e}")

    def apply_augmentations(self, image):
        """
        Applies augmentations to the generated image.

        Args:
            image (PIL.Image): The image to apply augmentations to.

        Returns:
            list: A list of augmented images.
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
        """
        Processes an image by running object detection and applying augmentations.

        Args:
            image_path (str): The file path of the image to process.
            output_dir (str): The directory to save processed images and annotations.

        Returns:
            list: A list of processed image paths.
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
            if not any(label.lower() in [cls.lower() for cls in self.detection_class] for label in detected_labels):
                print(f"{Fore.YELLOW}Removing image with no target class '{self.target_class}': {image_path}")
                self.remove_invalid_image(image_path)
                return None

            original_image_path = os.path.join(output_dir, os.path.basename(image_path))
            img.save(original_image_path)
            self.save_yolo_annotations(original_image_path, results, img.size)

            image_paths = [original_image_path]

            augmented_images = self.apply_augmentations(img)
            for idx, augmented_img in enumerate(augmented_images):
                augmented_img_path = os.path.join(output_dir, f"aug_{idx}_{os.path.basename(image_path)}")
                augmented_img.save(augmented_img_path)

                augmented_results = self.run_object_detection(augmented_img)
                if augmented_results is not None:
                    self.save_yolo_annotations(augmented_img_path, augmented_results, augmented_img.size)
                else:
                    print(f"{Fore.YELLOW}No detections found for augmented image {augmented_img_path}")
                    self.remove_invalid_image(augmented_img_path)
                    continue

                image_paths.append(augmented_img_path)

            print(f"{Fore.GREEN}Processing complete for: {image_path}")
            return image_paths

        except Exception as e:
            print(f"{Fore.RED}Error processing image {image_path}: {e}")
            return None

    def generate_background_images(self, prompt_history, num_batches, num_seeds):
        """
        Generates background images for the dataset, with no objects.

        Args:
            prompt_history (set): A set to track previously generated prompts.
            num_batches (int): Number of batches of images to generate.
            num_seeds (int): Number of seeds per batch.
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
                    prompt = self.generate_prompt('background', prompt_history, exclude_objects=True)[0]
                    if prompt:
                        seed = random.randint(0, 10000)
                        background_image = self.pipeline(
                            prompt,
                            height=RESOLUTION[0],
                            width=RESOLUTION[1],
                            guidance_scale=3.5,
                            num_inference_steps=NUM_INFERENCE_STEPS,
                            generator=torch.Generator(self.device).manual_seed(seed)
                        ).images[0]

                        img_filename = f"{split}_bg_{batch_index+1}_{seed_index+1}.png"
                        img_path = os.path.join(split_dir, img_filename)
                        background_image.save(img_path)

                        annotation_filename = img_filename.replace('.png', '.txt')
                        annotation_path = os.path.join(label_dir, annotation_filename)
                        self.create_empty_annotation(annotation_path)

                        print(f"{Fore.GREEN}Background image saved: {img_path}")
                    else:
                        print(f"{Fore.RED}Failed to generate a valid prompt for background.")

    def create_empty_annotation(self, annotation_path):
        """
        Creates an empty annotation file.

        Args:
            annotation_path (str): The file path of the annotation to create.
        """
        try:
            with open(annotation_path, 'w') as f:
                pass
            print(f"{Fore.GREEN}Empty annotation file created: {annotation_path}")
        except Exception as e:
            print(f"{Fore.RED}Error creating empty annotation file for {annotation_path}: {e}")

    def move_files(self, paths, image_dir, label_dir):
        """
        Moves image and annotation files to their respective directories.

        Args:
            paths (list): List of file paths to move.
            image_dir (str): Directory to move image files to.
            label_dir (str): Directory to move annotation files to.
        """
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
        """
        Splits the dataset into train, validation, and eval sets.

        Args:
            image_paths (list): List of image paths to split.
        """
        try:
            print(f"{Fore.CYAN}Splitting dataset into train, validation, and eval sets...")

            train_images_dir, val_images_dir, eval_images_dir, train_labels_dir, val_labels_dir, eval_labels_dir = self.create_dataset_directories()

            def split_class(images):
                total_images = len(images)

                if total_images < 8:
                    train_size = max(1, total_images - 2)
                    val_size = max(1, (total_images - train_size) // 2)
                    eval_size = total_images - train_size - val_size
                else:
                    train_size = total_images // 2
                    val_size = total_images // 4
                    eval_size = total_images - train_size - val_size

                selected_train_paths = random.sample(images, train_size)
                remaining = [path for path in images if path not in selected_train_paths]

                val_paths = random.sample(remaining, val_size)
                eval_paths = [path for path in remaining if path not in val_paths]

                print(f"{Fore.GREEN}Total images: {total_images}. Train: {len(selected_train_paths)}, Val: {len(val_paths)}, Eval: {len(eval_paths)}")
                return selected_train_paths, val_paths, eval_paths

            train_paths, val_paths, eval_paths = split_class(image_paths)

            self.move_files(train_paths, train_images_dir, train_labels_dir)
            self.move_files(val_paths, val_images_dir, val_labels_dir)
            self.move_files(eval_paths, eval_images_dir, eval_labels_dir)

        except Exception as e:
            print(f"{Fore.RED}Error during dataset splitting: {e}")

    def create_dataset_directories(self):
        """
        Creates the necessary directories for the dataset (train, val, eval).

        Returns:
            tuple: Paths to the directories for images and labels for each split (train, val, eval).
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

            print(f"{Fore.GREEN}Dataset directories created successfully.")
            return train_images_dir, val_images_dir, eval_images_dir, train_labels_dir, val_labels_dir, eval_labels_dir
        except Exception as e:
            print(f"{Fore.RED}Error creating dataset directories: {e}")
            return None, None, None, None, None, None

    def generate_yaml(self):
        """
        Generates a YAML configuration file for the dataset.
        """
        try:
            if not os.path.exists(self.dataset_dir):
                os.makedirs(self.dataset_dir)

            yaml_content = (
                "path: " + self.dataset_dir.replace("\\", "/") + "  # Base dataset directory\n"
                "train: images/train  # Train images directory\n"
                "val: images/val  # Validation images directory\n"
                "eval: images/eval  # Eval images directory\n"
                "nc: " + str(len(self.detection_class)) + "  # Number of classes\n"
                "names: " + str(self.detection_class) + "  # Class names\n"
            )

            yaml_path = os.path.join(self.dataset_dir, 'data.yaml')

            with open(yaml_path, 'w') as yaml_file:
                yaml_file.write(yaml_content)
            print(f"{Fore.GREEN}YAML file generated at: {yaml_path}")
        except Exception as e:
            print(f"{Fore.RED}Error generating YAML file: {e}")


def select_folder(title="Select Folder"):
    """
    Opens a file dialog to allow the user to select a folder.

    Args:
        title (str): Title of the dialog.

    Returns:
        str: The selected directory path.
    """
    root = tk.Tk()
    root.withdraw()
    selected_dir = filedialog.askdirectory(title=title)
    return selected_dir


def set_api_key():
    """
    Prompts the user to input the OpenAI API key.

    Returns:
        str: The entered OpenAI API key.
    """
    api_key = input(f"{Fore.YELLOW}Enter your OpenAI API key: ").strip()
    return api_key


def set_generation_theme():
    """
    Prompts the user to input the theme for image generation.

    Returns:
        str: The entered theme.
    """
    theme = input(f"{Fore.YELLOW}Enter the theme for image generation: ").strip()
    return theme


def set_detection_classes():
    """
    Prompts the user to input the detection classes to be used.

    Returns:
        list: List of detection class names.
    """
    classes_input = input(f"{Fore.YELLOW}Enter detection classes (comma-separated): ").strip()
    classes = [cls.strip() for cls in classes_input.split(',')]
    return classes


def set_numeric_input(prompt):
    """
    Prompts the user for a numeric input and validates it.

    Args:
        prompt (str): The prompt message for user input.

    Returns:
        int: A valid positive integer entered by the user.
    """
    while True:
        try:
            value = int(input(prompt).strip())
            if value > 0:
                return value
            else:
                print(f"{Fore.RED}Value must be greater than zero.")
        except ValueError:
            print(f"{Fore.RED}Invalid input. Please enter a positive number.")


def set_prepends():
    """
    Prompts the user to input a list of prepend modifiers for prompt generation.

    Returns:
        list: A list of prepend strings entered by the user.
    """
    prepends_input = input(f"{Fore.YELLOW}Enter prepend modifiers (comma-separated, e.g., 'Distant faraway shot, Close-up, Side view'): ").strip()
    prepends = [p.strip() for p in prepends_input.split(',')]
    return prepends


def main():
    """
    The main function that runs the interactive terminal menu for configuring and running the image generation pipeline.
    """
    print(f"\n{Fore.GREEN}{Style.BRIGHT}Welcome to the FLUX Capacitor!{Style.RESET_ALL}")
    print(f"{Fore.GREEN}A seamless image diffusion to computer vision dataset pipeline!{Style.RESET_ALL}")

    flux_capacitor = FluxCapacitor()

    flux_model_dir = None
    florence_model_dir = None
    openai_api_key = openai.api_key
    output_dir = None
    theme = None
    detection_classes = None
    num_batches = 1
    num_seeds = 1

    while True:
        flux_status = f"{Fore.GREEN}Set" if flux_model_dir else f"{Fore.RED}Not Set"
        florence_status = f"{Fore.GREEN}Set" if florence_model_dir else f"{Fore.RED}Not Set"
        api_status = f"{Fore.GREEN}Set" if openai_api_key else f"{Fore.RED}Not Set"
        output_status = f"{Fore.GREEN}Set" if output_dir else f"{Fore.RED}Not Set"
        theme_status = f"{Fore.GREEN}Set" if theme else f"{Fore.RED}Not Set"
        classes_status = f"{Fore.GREEN}Set" if detection_classes else f"{Fore.RED}Not Set"
        prepends_status = f"{Fore.GREEN}Set" if flux_capacitor.prepends else f"{Fore.RED}Not Set"

        print(f"\n{Fore.BLUE}{Style.BRIGHT}{'='*40}")
        print(f"{Fore.CYAN}{Style.BRIGHT}--- FLUX Capacitor Menu ---{Style.RESET_ALL}")
        print(f"{Fore.BLUE}{'='*40}{Style.RESET_ALL}")

        print(f"{Fore.YELLOW}{Style.BRIGHT}\n{'-'*10} Image Generation {'-'*10}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}1. Generate and annotate images")

        print(f"{Fore.YELLOW}{Style.BRIGHT}\n{'-'*10} Directory Setup {'-'*10}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}2. Set output directory [{output_status}]")
        print(f"{Fore.CYAN}3. Set FLUX model directory [{flux_status}]")
        print(f"{Fore.CYAN}4. Set Florence model directory [{florence_status}]")

        print(f"{Fore.YELLOW}{Style.BRIGHT}\n{'-'*10} API and Options {'-'*10}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}5. Set OpenAI API key [{api_status}]")
        print(f"{Fore.CYAN}6. Set generation theme [{theme_status}]")
        print(f"{Fore.CYAN}7. Set detection classes [{classes_status}]")
        print(f"{Fore.CYAN}8. Set number of image prompts (current: {num_batches})")
        print(f"{Fore.CYAN}9. Set number of seeds per prompt (current: {num_seeds})")
        print(f"{Fore.CYAN}10. Toggle background image generation (current: {'ON' if flux_capacitor.enable_background_generation else 'OFF'})")
        print(f"{Fore.CYAN}12. Set prompt prepends (current: {prepends_status})")

        print(f"{Fore.YELLOW}{Style.BRIGHT}\n{'-'*10} Exit {'-'*10}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}11. Exit")

        choice = input(f"{Fore.YELLOW}{Style.BRIGHT}Please enter your choice (1-12): {Style.RESET_ALL}").strip()

        if choice == "1":
            if not (output_dir and flux_model_dir and florence_model_dir and theme and detection_classes):
                print(f"{Fore.RED}Error: Please ensure all settings are configured correctly.")
                continue

            flux_capacitor.detection_class = detection_classes
            flux_capacitor.target_class = detection_classes[0]

            print(f"{Fore.GREEN}Starting image generation process...")
            prompt_history = set()

            for batch in range(num_batches):
                base_prompt, modified_prompts = flux_capacitor.generate_prompt(theme, prompt_history)
                if base_prompt:
                    for seed in range(num_seeds):
                        random_seed = random.randint(0, 10000)

                        image_path = flux_capacitor.generate_image(base_prompt, output_dir, random_seed)
                        if image_path:
                            image_paths = flux_capacitor.process_image(image_path, output_dir)
                            if image_paths:
                                flux_capacitor.split_dataset(image_paths)

                        for modified_prompt in modified_prompts:
                            modified_seed = random.randint(0, 10000)
                            modified_image_path = flux_capacitor.generate_image(modified_prompt, output_dir, modified_seed)
                            if modified_image_path:
                                modified_image_paths = flux_capacitor.process_image(modified_image_path, output_dir)
                                if modified_image_paths:
                                    flux_capacitor.split_dataset(modified_image_paths)
                else:
                    print(f"{Fore.RED}Failed to generate prompt.")

            if flux_capacitor.enable_background_generation:
                flux_capacitor.generate_background_images(prompt_history, num_batches, num_seeds)

            flux_capacitor.generate_yaml()
            print(f"{Fore.GREEN}Successfully generated and annotated images.\n")

        elif choice == "2":
            selected_dir = select_folder("Select Output Directory")
            if selected_dir:
                output_dir = selected_dir
                flux_capacitor.dataset_dir = output_dir
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
            flux_capacitor.enable_background_generation = not flux_capacitor.enable_background_generation
            print(f"{Fore.GREEN}Background image generation is now {'ON' if flux_capacitor.enable_background_generation else 'OFF'}")

        elif choice == "12":
            flux_capacitor.prepends = set_prepends()
            print(f"{Fore.GREEN}Prepend modifiers set to: {', '.join(flux_capacitor.prepends)}")

        elif choice == "11":
            print(f"{Fore.YELLOW}Exiting FLUX Capacitor. Goodbye!")
            break

        else:
            print(f"{Fore.RED}Invalid choice. Please enter a valid option.")


if __name__ == "__main__":
    main()
