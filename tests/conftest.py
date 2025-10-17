"""Test configuration ensuring project modules are importable when running individual files."""

from __future__ import annotations

import sys
from pathlib import Path

# Pytest can execute a single test module directly (e.g. ``pytest tests/test_file.py``),
# in which case the process working directory becomes ``tests/``. Ensure the repository
# root is at the front of ``sys.path`` so package imports like ``components.*`` resolve
# without manual path tweaks inside each test module.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
