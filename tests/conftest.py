"""Make the flat module layout importable from tests.

``data/`` and ``strategies/`` hold plain modules rather than packages, so that
scripts in them can be run directly (``python fetch.py``). Tests import them by
bare name, which needs those directories on the path.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

for folder in ("data", "strategies", "backtests", "journal"):
    path = str(PROJECT_ROOT / folder)
    if path not in sys.path:
        sys.path.insert(0, path)
