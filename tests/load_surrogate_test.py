import sys
from pathlib import Path

# ensure project root is on sys.path so "import src..." works when running tests
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.surrogate_loader import load_surrogate_models
from src.config.load_config import load_config

def test_load_surrogate_models():
    """
    Only tests if the code runs without errors, no edge case testing.
    """
    print("Load config")
    config = load_config('config/example.yml')
    print(config)
    print("Load surrogate models")
    surrogate_models = load_surrogate_models(**config)
    print("Done")
    assert surrogate_models is not None

if __name__ == "__main__":
    test_load_surrogate_models()