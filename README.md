
# FLUX Capacitor: Pioneering Automated Synthetic Dataset Generation for Vision Model Fine-Tuning

## Overview

**FLUX Capacitor** is an innovative pipeline that allows users to automatically generate custom datasets for vision model fine-tuning, with a focus on object detection and classification tasks. Leveraging the power of the **FLUX diffusion model** for synthetic image generation and the unparalleled object detection capabilities of **Florence-2**, this tool represents a major leap forward in how datasets for machine learning are created. The generated datasets are fully formatted for YOLO training, complete with the necessary annotations, directory structure, and `yaml` configuration file.

This pipeline is unprecedented in its ability to seamlessly combine synthetic image generation and large-scale object detection, allowing users to create datasets for any object or category. This approach eliminates the need for manual data collection and annotation, a time-consuming and costly process, particularly for niche or rare objects.

## Key Features

- **Synthetic Dataset Creation**: FLUX generates photorealistic images based on user-defined prompts. Florence-2 then detects and validates the objects within these images, creating high-quality datasets ready for model fine-tuning.
- **YOLO-Ready**: Automatically formats generated datasets into the structure required for YOLO training, including images, annotations, and `yaml` configuration files.
- **Large-Scale Object Detection**: Florence-2 supports over 8,000 object classes, far surpassing traditional models like YOLO, enabling robust validation and detection across a broader range of objects.
- **Image Augmentation**: Apply various augmentations (flipping, rotation, color adjustments) to further enrich the dataset and improve model generalization.
- **Dataset Splitting**: Automatically splits generated images into training, validation, and test sets, ensuring your dataset is ready for model training immediately.
- **Dynamic Prompt Generation**: Optional integration with GPT-4 to generate diverse, context-driven prompts for image generation.

## Why This Pipeline Is Unprecedented

FLUX Capacitor represents the first true integration of a **diffusion-based image generator** with a **large-scale vision-language model** like Florence-2, designed specifically for generating custom datasets. Unlike traditional methods, which rely on manual image collection and annotation, FLUX Capacitor automates the entire process:

- **Synthetic Image Generation**: The FLUX diffusion model is one of the most advanced text-to-image models available today. It can generate diverse, high-quality images for any object or scene.
- **Comprehensive Object Detection**: Florence-2, with its 8,000+ object classes, provides unprecedented validation and precision, ensuring that generated images contain the exact objects of interest.
- **YOLO Training Format**: The pipeline outputs datasets pre-formatted for YOLO, including YOLO-style annotations and a `yaml` file, enabling immediate fine-tuning of your object detection models.

To our knowledge, no other tool combines these capabilities, making FLUX Capacitor the first of its kind to automate this level of synthetic dataset creation.

## Installation

### Prerequisites

- Python 3.8 or higher
- A CUDA-capable GPU (recommended for fast processing)
- OpenAI API key (optional, for GPT-4 prompt generation)

### Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/CharlesCNorton/FLUXCapacitor.git
   cd FLUXCapacitor
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Models**:
   - Download and configure the **FLUX diffusion model** and **Florence-2**.
   - Follow instructions to place the model files in the appropriate directories.

4. (Optional) Set up the OpenAI API for prompt generation:
   ```bash
   export OPENAI_API_KEY='your_openai_api_key'
   ```

## Current Capabilities

At present, **FLUX Capacitor** automates the creation of synthetic datasets ready for training object detection models like YOLO. Key functions include:

- **Image Generation**: The FLUX model generates diverse, high-quality images based on prompts related to the objects of interest.
- **Object Detection & Annotation**: Florence-2 performs object detection on these images, providing YOLO-style bounding box annotations.
- **Dataset Splitting**: The images and annotations are automatically split into training, validation, and evaluation sets, prepared for model training.
- **Augmentation**: Optional augmentations can be applied to enhance the dataset's diversity.

The output dataset includes:

```
/dataset
  /images
    /train
    /val
    /test
  /labels
    /train
    /val
    /test
  data.yaml
```

The `data.yaml` file includes all the necessary configurations for training a YOLO model with the generated dataset.

## Future Development: Auto-Training Pipeline

The current version of FLUX Capacitor focuses on dataset generation, but future development will aim to create a fully automated training pipeline for models like YOLO, EfficientDet, or RetinaNet. Key upcoming features include:

- **Automatic Model Training**: After generating the dataset, the pipeline will automatically initiate the training of the specified object detection model.
- **Hyperparameter Tuning**: Integration of hyperparameter tuning to optimize the performance of the models trained on custom datasets.
- **Evaluation and Validation**: The pipeline will include automated evaluation of the trained models on validation sets and comparison with baseline performance metrics.

With these improvements, **FLUX Capacitor** will not only generate custom datasets but also fully automate the process of training and validating object detection models, making it an all-in-one tool for vision model fine-tuning.

## Contribution

We welcome contributions from the community! If you'd like to contribute, please fork the repository, make your changes, and submit a pull request. Let's work together to expand this unprecedented pipeline!

## License

This project is licensed under the MIT License. See the LICENSE file for details.
