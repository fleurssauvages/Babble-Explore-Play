"""6 cubes: build a tower in any order with the primitives discovered on two cubes.

    python scripts/cubes6.py [--relearn]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bep.app import main  # noqa: E402

if __name__ == "__main__":
    main("cubes6")
