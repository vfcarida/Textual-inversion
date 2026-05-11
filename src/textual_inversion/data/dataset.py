"""
Dataset creation functions for Textual Inversion.
"""
import logging
from typing import List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_cv import layers as cv_layers

from textual_inversion.config import TrainingConfig
from textual_inversion.utils.logger import get_logger

logger = get_logger(__name__)


def pad_embedding(embedding: List[int], end_of_text_token: int, max_length: int) -> List[int]:
    """
    Pads the tokenized embedding to the maximum prompt length.

    Args:
        embedding (List[int]): The tokenized prompt.
        end_of_text_token (int): Token ID for end of text.
        max_length (int): Maximum length to pad to.

    Returns:
        List[int]: Padded embedding.
    """
    return embedding + [end_of_text_token] * (max_length - len(embedding))


def assemble_image_dataset(file_paths: List[str], size: int) -> tf.data.Dataset:
    """
    Creates an augmented image dataset from file paths.
    
    Args:
        file_paths (List[str]): List of absolute paths to images.
        size (int): Image size to resize to (assumes square images).
        
    Returns:
        tf.data.Dataset: Augmented image dataset.
    """
    try:
        resize = keras.layers.Resizing(height=size, width=size, crop_to_aspect_ratio=True)
        images = [keras.utils.load_img(img) for img in file_paths]
        images = [keras.utils.img_to_array(img) for img in images]
        images_array = np.array([resize(img) for img in images])

        # Normalize to [-1, 1] range for Stable Diffusion
        images_array = images_array / 127.5 - 1

        image_dataset = tf.data.Dataset.from_tensor_slices(images_array)

        # Shuffle and introduce random noise
        image_dataset = image_dataset.shuffle(50, reshuffle_each_iteration=True)
        image_dataset = image_dataset.map(
            cv_layers.RandomCropAndResize(
                target_size=(size, size),
                crop_area_factor=(0.8, 1.0),
                aspect_ratio_factor=(1.0, 1.0),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        image_dataset = image_dataset.map(
            cv_layers.RandomFlip(mode="horizontal"),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        
        logger.info(f"Successfully assembled image dataset with {len(file_paths)} images.")
        return image_dataset

    except Exception as e:
        logger.error(f"Error assembling image dataset: {e}")
        raise


def assemble_text_dataset(
    prompts: List[str], 
    placeholder_token: str, 
    tokenizer: keras.layers.Layer, 
    max_prompt_length: int
) -> tf.data.Dataset:
    """
    Tokenizes and pads prompts, returning a text dataset.

    Args:
        prompts (List[str]): List of prompt templates.
        placeholder_token (str): The new token to learn.
        tokenizer: The tokenizer from the stable diffusion model.
        max_prompt_length (int): Max prompt context length.

    Returns:
        tf.data.Dataset: Tokenized and padded text dataset.
    """
    formatted_prompts = [prompt.format(placeholder_token) for prompt in prompts]
    embeddings = [tokenizer.encode(prompt) for prompt in formatted_prompts]
    
    end_of_text = tokenizer.end_of_text
    padded_embeddings = [np.array(pad_embedding(emb, end_of_text, max_prompt_length)) for emb in embeddings]
    
    text_dataset = tf.data.Dataset.from_tensor_slices(padded_embeddings)
    return text_dataset


def assemble_dataset(
    file_paths: List[str], 
    config: TrainingConfig, 
    tokenizer: keras.layers.Layer
) -> tf.data.Dataset:
    """
    Zips image and text datasets for training.

    Args:
        file_paths (List[str]): List of absolute paths to images.
        config (TrainingConfig): Training configuration.
        tokenizer: Stable diffusion tokenizer.

    Returns:
        tf.data.Dataset: Zipped dataset of (image, text) pairs.
    """
    image_dataset = assemble_image_dataset(file_paths, config.image_size)
    text_dataset = assemble_text_dataset(
        config.prompts, 
        config.placeholder_token, 
        tokenizer, 
        config.max_prompt_length
    )

    # Repeat datasets to match batches
    image_dataset = image_dataset.repeat()
    text_dataset = text_dataset.repeat(5)
    
    zipped_dataset = tf.data.Dataset.zip((image_dataset, text_dataset))
    return zipped_dataset
