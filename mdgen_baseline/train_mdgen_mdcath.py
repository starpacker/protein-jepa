"""Launcher: run MDGen's own train.py with our compat shims pre-imported.

NEW file; MDGen source untouched.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MDGEN_SRC = Path("/liuyutian/mdgen_src/mdgen-master")
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_MDGEN_SRC))

import mdgen_baseline.mdgen_compat  # noqa: F401

os.chdir(_MDGEN_SRC)
runpy.run_path(str(_MDGEN_SRC / "train.py"), run_name="__main__")
