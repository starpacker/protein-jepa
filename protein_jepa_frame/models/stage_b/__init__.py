"""Stage B: Frozen Stage A + LaSP world model training.

Key components (Gate 2 canonical LaSP):
- projection: H_t -> Q_t, G_t = P(Q_t)
- energy: single scalar energy A(Q, c, T)
- diffusion: D_i(q_i, G, c, T) with Ito divergence
- bath: per-residue S_i:[L,16] with exact-OU update
- integrator: resolved substeps with recorded delta_t_ps
- objectives: transition NLL, FDT diagnostics, VAMP-2
"""

from protein_jepa_frame.models.stage_b.model import StageBLASP
from protein_jepa_frame.models.stage_b.objectives import stage_b_total_loss

__all__ = [
    "StageBLASP",
    "stage_b_total_loss",
]
