# Contributing to Textual-inversion

First off, thank you for considering contributing to this project!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Textual-inversion.git
   cd Textual-inversion
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Code Standards

We use `ruff` for linting and formatting, and `mypy` for static type checking.

Before submitting a pull request, please ensure your code passes all checks:

```bash
# Run tests
pytest tests/

# Run type checking
mypy src/

# Run linting
ruff check src/
```

## Pull Request Process

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
