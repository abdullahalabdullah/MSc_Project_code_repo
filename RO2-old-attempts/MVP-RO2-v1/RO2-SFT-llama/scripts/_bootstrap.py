"""Put the repo root on sys.path so `import sft...` works from scripts/.

Each entry script does `import _bootstrap  # noqa` as its first project import.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
