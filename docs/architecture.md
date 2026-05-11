# Textual Inversion Architecture

## Overview

Textual Inversion is a technique for fine-tuning Stable Diffusion models to learn new concepts from just a few images (typically 3-5). It works by learning a new embedding in the text encoder's vocabulary while keeping the rest of the model frozen.

## Core Components

The architecture consists of several key modules:

1. **Text Encoder (Frozen)**: Translates text prompts into embeddings.
2. **Diffusion Model (Frozen)**: Predicts noise based on latents and text embeddings.
3. **New Embedding (Trainable)**: A new token (e.g., `espositope`) is added to the vocabulary. Its embedding is the ONLY part of the model that is trained.

### Training Flow

1. **Image Preprocessing**: Training images are loaded, resized, and augmented (cropped, flipped).
2. **Latent Encoding**: Images are passed through the VAE encoder to get latents, which are scaled.
3. **Noise Addition**: Random noise is added to the latents based on a random timestep.
4. **Text Encoding**: The prompt containing the new token is encoded into a hidden state.
5. **Noise Prediction**: The diffusion model predicts the added noise using the noisy latents and the text hidden state.
6. **Loss Calculation**: The Mean Squared Error (MSE) between the predicted noise and actual noise is calculated.
7. **Gradient Update**: Gradients are calculated, but **only the weights corresponding to the new token's embedding are updated**. All other gradients are zeroed out.

## Refactored Implementation

The codebase has been refactored to follow SOLID principles:

- **`config.py`**: Centralizes all hyperparameters and constants.
- **`dataset.py`**: Handles image loading, augmentation, and prompt tokenization.
- **`trainer.py`**: Contains the `TextualInversionTrainer` class, handling the custom training loop and gradient manipulation.
- **`image_generator.py`**: A Keras Callback that generates and saves sample images during training.
