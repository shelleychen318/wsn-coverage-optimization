"""Ensure the project root is on sys.path so `import wsn` works under pytest
regardless of editable-install state."""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
