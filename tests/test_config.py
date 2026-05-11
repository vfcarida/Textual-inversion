import pytest
from textual_inversion.config import TrainingConfig

def test_training_config_defaults():
    config = TrainingConfig()
    assert config.placeholder_token == "espositope"
    assert config.subject == "cat"
    assert config.image_size == 512
    assert config.epochs == 50
    assert len(config.prompts) > 0
    assert config.max_prompt_length == 77
