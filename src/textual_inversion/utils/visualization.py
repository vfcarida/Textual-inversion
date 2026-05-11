"""
Visualization utilities for images.
"""
import os
import numpy as np
from typing import List

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def plot_images(images: List[np.ndarray], save_path: str = None) -> None:
    """
    Plots a list of images in a horizontal line.
    
    Args:
        images (List[np.ndarray]): List of image arrays to plot.
        save_path (str, optional): If provided, saves the figure to this path.
    """
    if plt is None:
        print("matplotlib is not installed. Skipping plot.")
        return

    n_images = len(images)
    plt.figure(figsize=(20, 20))
    for i in range(n_images):
        ax = plt.subplot(1, n_images, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
        
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
