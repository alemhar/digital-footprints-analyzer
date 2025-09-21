import os
import sys

# Ensure the src/ directory is on sys.path so the package `dfa` can be imported
ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dfa.ui_app import main  # noqa: E402

if __name__ == "__main__":
    main()
