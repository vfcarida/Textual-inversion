# Textual Inversion

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![KerasCV](https://img.shields.io/badge/KerasCV-0.4.0-red)

Professional implementation of Textual Inversion for Stable Diffusion using KerasCV. This repository was originally presented at Google Brasil and has been completely refactored to follow production-ready software engineering standards.

## 📖 Overview

Textual Inversion allows you to teach a pre-trained Stable Diffusion model new concepts (objects or styles) using just 3-5 images. It does this by learning a new embedding in the text encoder's vocabulary, while keeping the rest of the massive model frozen.

This repository provides a modular, typed, and tested implementation of the technique.

## 🏛️ Architecture

For a detailed explanation of how Textual Inversion works and how the codebase is structured, please see the [Architecture Documentation](docs/architecture.md).

The original presentation slides are available in `docs/Textual inversion.pdf`.

## ⚙️ Prerequisites

- Python 3.10+
- A GPU with at least 16GB VRAM is highly recommended for training.

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Textual-inversion.git
   cd Textual-inversion
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This project relies on `keras_cv==0.4.0` for compatibility with the original presentation.*

## 💡 Quick Start

The easiest way to get started is to use the provided Jupyter notebook, which walks you through the entire process.

1. Place your training images (e.g., 3-5 photos of a specific cat) in the `data/` directory or update the paths in the notebook.
2. Run the demo notebook:
   ```bash
   jupyter notebook notebooks/textual_inversion_demo.ipynb
   ```

### Programmatic Usage

You can also use the modules directly in your Python code:

```python
import keras_cv
from textual_inversion.config import TrainingConfig
from textual_inversion.data.dataset import assemble_dataset
from textual_inversion.models.trainer import TextualInversionTrainer

# 1. Configure
config = TrainingConfig(placeholder_token="espositope", subject="cat", epochs=50)

# 2. Load Base Model
stable_diffusion = keras_cv.models.StableDiffusion()
noise_scheduler = keras_cv.models.stable_diffusion.NoiseScheduler()

# 3. Create Dataset
train_ds = assemble_dataset(
    file_paths=["data/cat1.jpg", "data/cat2.jpg", "data/cat3.jpg"],
    config=config,
    tokenizer=stable_diffusion.tokenizer
)
train_ds = train_ds.batch(config.batch_size)

# 4. Initialize Trainer and Compile
trainer = TextualInversionTrainer(stable_diffusion, noise_scheduler, config)
# ... compile with optimizer ...

# 5. Train
trainer.fit(train_ds, epochs=config.epochs)
```

## 📁 Project Structure

```text
Textual-inversion/
├── docs/                      # Documentation and presentation slides
├── notebooks/                 # Jupyter notebooks for demonstration
├── src/
│   └── textual_inversion/     # Main package
│       ├── callbacks/         # Keras callbacks (image generation)
│       ├── data/              # Dataset loading and preprocessing
│       ├── models/            # Custom training loop logic
│       ├── utils/             # Visualization and logging
│       └── config.py          # Centralized hyperparameter configuration
├── tests/                     # Unit tests
├── pyproject.toml             # Project configuration
└── requirements.txt           # Dependency lockfile
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on the [Keras Tutorial: Fine-tuning via Textual Inversion](https://keras.io/examples/generative/fine_tune_via_textual_inversion/)
- Original code presented at Google Brasil.
