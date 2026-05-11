"""
Configuration module for Textual Inversion.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    """
    Configuration dataclass for the Textual Inversion training process.
    """
    placeholder_token: str = "espositope"
    subject: str = "cat"
    image_size: int = 512
    epochs: int = 50
    train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    initial_learning_rate: float = 1e-4
    weight_decay: float = 0.004
    epsilon: float = 1e-8
    global_clipnorm: float = 10.0
    batch_size: int = 1
    seed: int = 1337
    
    # KerasCV Stable Diffusion constants
    max_prompt_length: int = 77
    latent_scale_factor: float = 0.18215
    embedding_layer_index: int = 2
    placeholder_token_id: int = 49408
    
    prompts: List[str] = field(default_factory=lambda: [
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        "a photo of my {}",
        "a photo of the cool {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a good photo of the {}",
        "a photo of one {}",
        "a close-up photo of the {}",
        "a rendition of the {}",
        "a photo of the clean {}",
        "a rendition of a {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of the small {}",
        "a photo of the weird {}",
        "a photo of the large {}",
        "a photo of a cool {}",
        "a photo of a small {}",
    ])
