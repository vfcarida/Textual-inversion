"""
Callbacks for image generation during training.
"""
import os
import tensorflow as tf
from tensorflow import keras
import keras_cv
from datetime import datetime

from textual_inversion.utils.visualization import plot_images
from textual_inversion.utils.logger import get_logger

logger = get_logger(__name__)


class ImageGenerationCallback(keras.callbacks.Callback):
    """
    Callback that generates images at specified epoch intervals to monitor training progress.
    """

    def __init__(
        self, 
        stable_diffusion: keras_cv.models.StableDiffusion, 
        prompt: str, 
        steps: int = 25, 
        frequency: int = 20, 
        seed: int = None,
        save_dir: str = "outputs",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.stable_diffusion = stable_diffusion
        self.prompt = prompt
        self.steps = steps
        self.frequency = frequency
        self.seed = seed
        self.save_dir = save_dir
        
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        """
        Generates and saves image at epoch end based on frequency.
        """
        if (epoch + 1) % self.frequency == 0 or epoch == 0:
            logger.info(f"Generating image for epoch {epoch + 1} with prompt: '{self.prompt}'")
            try:
                images = self.stable_diffusion.text_to_image(
                    self.prompt, 
                    batch_size=1, 
                    num_steps=self.steps, 
                    seed=self.seed
                )
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.save_dir, f"epoch_{epoch + 1}_{timestamp}.png")
                
                plot_images(images, save_path=save_path)
                logger.info(f"Saved generated image to {save_path}")
                
            except Exception as e:
                logger.error(f"Failed to generate image during callback: {e}")
