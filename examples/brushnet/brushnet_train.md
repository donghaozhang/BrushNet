# BrushNet Training

This repository contains code for training BrushNet, a specialized model for image editing with brush strokes.

## Overview

BrushNet is a model that allows for controlled image generation based on brush strokes. It works by conditioning a diffusion model on both text prompts and brush stroke inputs, enabling precise control over the generated image content.

## Requirements

- Python 3.8+
- PyTorch
- Diffusers
- Transformers
- Accelerate
- Datasets
- Huggingface Hub

## Installation

```bash
pip install -r requirements.txt
```

## Usage

To train BrushNet, use the `train_brushnet.py` script:

```bash
python train_brushnet.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="path/to/your/data" \
  --output_dir="brushnet-model" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --learning_rate=5e-6 \
  --max_train_steps=15000 \
  --validation_prompt="A cake on the table." \
  --validation_image="examples/brushnet/src/test_image.jpg" \
  --validation_mask="examples/brushnet/src/test_mask.jpg" \
  --random_mask
```

## Data Format

The training script expects data in WebDataset format (`.tar` files). Each sample should contain:
- `image`: The target image
- `caption`: Text description of the image
- `height` and `width`: Image dimensions
- `segmentation`: JSON containing mask information

## Key Parameters

- `--pretrained_model_name_or_path`: Base diffusion model to use
- `--brushnet_model_name_or_path`: Optional path to pretrained BrushNet weights
- `--train_data_dir`: Directory containing training data
- `--output_dir`: Where to save the trained model
- `--resolution`: Image resolution for training
- `--random_mask`: Use randomly generated masks during training
- `--validation_prompt`, `--validation_image`, `--validation_mask`: Used for validation during training
- `--learning_rate`: Learning rate for training
- `--max_train_steps`: Maximum number of training steps
- `--push_to_hub`: Whether to push the model to Hugging Face Hub

## Validation

The script performs validation at regular intervals, generating images based on the provided validation prompts and images. These are logged to TensorBoard or Weights & Biases if enabled.

## Checkpointing

The model is saved at regular intervals specified by `--checkpointing_steps`. You can resume training from a checkpoint using `--resume_from_checkpoint`.

## Advanced Features

- Mixed precision training with `--mixed_precision`
- xFormers memory-efficient attention with `--enable_xformers_memory_efficient_attention`
- Gradient checkpointing with `--gradient_checkpointing`
- 8-bit Adam optimizer with `--use_8bit_adam`

## License

This project is licensed under the MIT License - see the LICENSE file for details. 