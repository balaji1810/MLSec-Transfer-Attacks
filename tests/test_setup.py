import sys
from pathlib import Path

# ensure project root is on sys.path so "import src..." works when running tests
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))