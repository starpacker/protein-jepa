"""Vendored official third-party packages used by protein_jepa_frame.

Currently contains a vendored copy of the official OpenFold
(aqlaboratory/openfold, Apache-2.0) pure-Python package.  The vendored tree
keeps OpenFold's own absolute ``openfold.*`` import style intact, so this
module puts the vendored directory on ``sys.path`` (registering it as the
top-level ``openfold`` package) **only if** a real ``openfold`` install is
not already importable.

This lets Stage C reuse official OpenFold geometry/atom37 routines verbatim
(e.g. ``openfold.utils.feats.torsion_angles_to_frames`` and
``frames_and_literature_positions_to_atom14_pos``) instead of reimplementing
frame -> atom37 reconstruction by hand.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# Only register the vendored copy when no real OpenFold is installed.
if importlib.util.find_spec("openfold") is None:
    if _HERE not in sys.path:
        # Append (not insert) so any real install would still win.
        sys.path.append(_HERE)

__all__ = []
