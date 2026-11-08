"""Stage B (v4): physically-consistent GLE over H_dyn, conditioned on H_static.

Wires the proven ``StageB1b`` GLE (energy ``U(H;T,c)``, dissipative ``velocity``
bath coupling, exact/amortised S_0 posterior, collective mobility) onto the v4
static/dynamic representation:

  H_static = E_s(c, X_0)          -- fixed static context, conditions the bath
  H_dyn,t  = E_d(X_t | H_static)  -- magnitude-carrying conformational state
  (H_dyn,t, S_t) -> (H_dyn,t+1, S_t+1)   via StageB1b
  X_hat,t  = D(H_static, H_dyn,t)

Only H_dyn and S are evolved; H_static is held fixed and passed as the ``seq_feature``
static conditioning of ``StaticBathParams`` (D, C, Gamma).  This is exactly the spec
§3 contract: H_static "may condition drift, coupling, noise, potential or metric".

The module does NOT own the encoder/decoder (they are the frozen Stage A bundle);
it owns only the GLE.  The encode/decode plumbing for the frozen bundle lives in
``training/train_stage_b_v4.py`` (``encode_static`` / ``encode_dyn`` /
``decode_paths_v4``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig


class StageBV4(nn.Module):
    """GLE over H_dyn with H_static as the static bath conditioning."""

    def __init__(self, gle_cfg: StageB1bConfig):
        super().__init__()
        self.gle = StageB1b(gle_cfg)
        self.cfg = gle_cfg

    def rollout(self, H_dyn0, H_static, residue_mask, temperature_k, n_steps,
                H_hist=None, dt_hist=None, S0=None, generator=None,
                return_bath=False, collective_mobility=True, s0_mode="exact",
                detach=True, grad_steps=None):
        """Autonomous (H_dyn, S) rollout with H_static as the bath conditioning.

        Returns stacked H_dyn trajectory [B, n_steps+1, L, dH] (and S if requested).
        """
        self.gle.s0_mode = s0_mode
        return self.gle.rollout(
            H_dyn0, residue_mask, temperature_k, n_steps,
            H_hist=H_hist, dt_hist=dt_hist, seq_feature=H_static, S0=S0,
            generator=generator, return_bath=return_bath,
            collective_mobility=collective_mobility,
            detach=detach, grad_steps=grad_steps,
        )

    def compute_forces(self, H_dyn, residue_mask, temperature_k=None):
        """Force f = -grad U over H_dyn (with static conditioning folded into bath)."""
        return self.gle.compute_forces(H_dyn, residue_mask, temperature_k)


def build_stage_b_v4(latent_dim: int = 128, bath_dim: int = 8,
                     step_ps: float = 1000.0) -> StageBV4:
    """Construct a StageBV4 with a stable, dissipative default config.

    ``bath_coupling='velocity'`` (FDT-correct, damping rather than amplifying) and
    ``collective_mobility=True`` are required for the exact S_0 Kalman posterior.
    """
    cfg = StageB1bConfig(
        latent_dim=latent_dim, bath_dim=bath_dim, step_ps=step_ps,
        bath_coupling="velocity",
    )
    return StageBV4(cfg)
