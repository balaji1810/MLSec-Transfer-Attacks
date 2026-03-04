import test_setup

from src.models.target_loader import load_all_targets
from src.config.load_config import load_config

def test_load_target_models():
    """
    Only tests if the code runs without errors, no edge case testing.
    """
    print("Load config")
    config = load_config('config/example.yml')
    print("Load target models")
    target_models = load_all_targets(**config)
    print("Done")
    assert target_models is not None

if __name__ == "__main__":
    test_load_target_models()
