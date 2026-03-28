import test_setup

from src.models.surrogate_loader import load_surrogate_models
from src.config.load_config import load_config

def test_load_surrogate_models():
    """
    Only tests if the code runs without errors, no edge case testing.
    """
    print("Load config")
    config = load_config('config/example.yml')
    print("Load surrogate models")
    surrogate_models = load_surrogate_models(**config)
    print("Done")
    assert surrogate_models is not None

if __name__ == "__main__":
    test_load_surrogate_models()