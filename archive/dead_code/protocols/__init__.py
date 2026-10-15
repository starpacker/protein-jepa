"""Physical time protocols and horizon selection for training/evaluation.

Gate 3: Physical time protocol
- Horizon candidates from real time_ps, not hard-coded frame counts
- short/medium/long selection via ACF/ITS on train+val
- Locked in protocol/horizons.yaml before test evaluation
"""

from protein_jepa_frame.protocols.horizons import (
    HorizonProtocol,
    compute_acf,
    select_horizons,
)

__all__ = [
    "HorizonProtocol",
    "compute_acf",
    "select_horizons",
]
