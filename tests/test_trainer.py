import pytest

def test_trainer_import():
    try:
        from textual_inversion.models.trainer import TextualInversionTrainer
        assert True
    except ImportError:
        pytest.fail("Failed to import TextualInversionTrainer")
