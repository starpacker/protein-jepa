"""iter_008: kinematic feature routing into Stage-B Q.

Evidence (artifacts/iter_007_phase0_audit/analysis/AB1, AB2):
  - Frozen H carries NO per-residue kinematic signal (val R^2 < 0 for
    displacement magnitude; direction ~0).
  - Q = projection(H) is therefore empty at the per-residue level
    (mag R^2 ~0.002, direction ~ -0.001).
  - The only demonstrable per-residue signal lives in RAW context kinematics:
      BA  = basin-anchor displacement  (context mean CA - anchor CA)
      VEL = context velocity           (frame-to-frame CA displacement)
    BA+VEL reaches mag R^2 = 0.267, dir R^2 = 0.151 on protein-disjoint val.

Conclusion: the signal must be ROUTED INTO Q, not regressed out of it.
This module adds an explicit, SE(3)-invariant, context-only kinematic
channel to the Q projection while leaving the canonical energy/force/
semigroup/diffusion (Gate-D) path untouched.

Design:
  Q = proj_base(h_last)  +  proj_kin([h_last, kin_feat])
The kinematic branch is zero-initialised so a freshly loaded v3_final
checkpoint reproduces the canonical Q exactly, then training grows the
kinematic contribution.  Every feature uses ONLY context frames, so the
result is legal for direct and autonomous evaluation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from protein_jepa_frame.config import ModelConfig
from protein_jepa_frame.models.stage_b.model import StageBLASP
from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame


class StageBLASP_Kinematic(StageBLASP):
    """Stage B with an SE(3)-invariant kinematic routing channel into Q."""

    # BA(3) + |BA|(1) + VEL(3) + |VEL|(1) + span(3) + |span|(1) = 12
    KIN_DIM = 12

    def __init__(self, config: ModelConfig, *, kin_hidden: int = 64, **kwargs):
        super().__init__(config, **kwargs)

        self.kin_hidden = int(kin_hidden)
        D = self.state_dim
        d_h = config.latent_dim

        # Per-residue kinematic feature encoder (context-only input).
        self.kin_encoder = nn.Sequential(
            nn.Linear(self.KIN_DIM, self.kin_hidden), nn.SiLU(),
            nn.Linear(self.kin_hidden, self.kin_hidden), nn.SiLU(),
        )

        # Kinematic branch of the Q projection: [h_last, kin_feat] -> Q.
        self.projection_kin = nn.Sequential(
            nn.Linear(d_h + self.kin_hidden, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, D),
        )
        # Zero-init the final layer: at load time Q equals the canonical
        # projection(h_last) exactly, so the v3_final representation is
        # preserved until the kinematic gradient reshapes it.
        nn.init.zeros_(self.projection_kin[-1].weight)
        nn.init.zeros_(self.projection_kin[-1].bias)

    # -- Kinematic feature extraction (context-only, SE(3)-invariant) --

    @staticmethod
    def compute_kin_features(
        coords: torch.Tensor,          # [B, T, L, 4, 3]
        residue_mask: torch.Tensor,    # [B, L]
    ) -> torch.Tensor:
        """Per-residue local-frame kinematic features from context frames.

        Features (all rotated into the anchor residue frame, so SE(3)-invariant):
          BA      = R_a^T (mean_context_CA - anchor_CA)         [3]
          |BA|    = norm of BA                                   [1]
          VEL     = R_a^T (CA_{T-1} - CA_{T-2})   (last velocity) [3]
          |VEL|   = norm of VEL                                   [1]
          SPAN    = R_a^T (CA_last - CA_first)    (total drift)   [3]
          |SPAN|  = norm of SPAN                                  [1]

        Args:
            coords:       context backbone [B, T, L, 4, 3] (N, CA, C, O)
            residue_mask: [B, L]

        Returns:
            [B, L, KIN_DIM] features.
        """
        B, T, L = coords.shape[:3]
        device, dtype = coords.device, coords.dtype
        ca = coords[..., 1, :]                 # [B, T, L, 3]

        # Anchor residue frames from the last context frame (N, CA, C)
        anchor = coords[:, -1]                 # [B, L, 4, 3]
        R_a, _ = RigidBodyFrame.from_backbone(
            anchor[..., 0, :], anchor[..., 1, :], anchor[..., 2, :]
        )                                      # [B, L, 3, 3]
        Rt = R_a.transpose(-2, -1)             # [B, L, 3, 3]

        anchor_ca = ca[:, -1]                  # [B, L, 3]
        mean_ca = ca.mean(dim=1)               # [B, L, 3]
        ba_g = mean_ca - anchor_ca             # [B, L, 3]
        ba = (Rt @ ba_g.unsqueeze(-1)).squeeze(-1)

        if T >= 2:
            vel_g = ca[:, -1] - ca[:, -2]
        else:
            vel_g = torch.zeros_like(ba_g)
        vel = (Rt @ vel_g.unsqueeze(-1)).squeeze(-1)

        span_g = ca[:, -1] - ca[:, 0]
        span = (Rt @ span_g.unsqueeze(-1)).squeeze(-1)

        feats = torch.cat(
            [ba, ba.norm(dim=-1, keepdim=True),
             vel, vel.norm(dim=-1, keepdim=True),
             span, span.norm(dim=-1, keepdim=True)],
            dim=-1,
        )  # [B, L, 12]
        if residue_mask is not None:
            feats = feats * residue_mask.to(dtype).unsqueeze(-1)
        return feats

    # -- Encode state with kinematic routing --

    def encode_state(
        self,
        coords, residue_mask, aatype, residue_index, chain_id,
        time_ps=None, temperature_k=None,
        *, use_temporal=True, return_bath=True,
        return_ca_distances=True, return_forces=False,
    ) -> dict:
        """Canonical encode_state + kinematic routing into Q.

        Identical to the parent except Q gains a kinematic branch:
            Q = proj_base(h_last) + proj_kin([h_last, kin_feat])
        """
        B, T, L = coords.shape[:3]
        device = coords.device

        if residue_mask.dim() == 3:
            residue_mask_2d = residue_mask[:, -1]
        else:
            residue_mask_2d = residue_mask

        if aatype.dim() == 1:
            aatype = aatype.unsqueeze(0).expand(B, -1)
        if residue_index is None:
            residue_index = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        elif residue_index.dim() == 1:
            residue_index = residue_index.unsqueeze(0).expand(B, -1)
        if chain_id is None:
            chain_id = torch.zeros(B, L, dtype=torch.long, device=device)
        elif chain_id.dim() == 1:
            chain_id = chain_id.unsqueeze(0).expand(B, -1)

        # -- Spatial encoding (frozen) --
        context_latents = self.forward_spatial(
            coords, residue_mask, aatype, residue_index, chain_id,
        )

        # -- Temporal history -> h_last --
        if use_temporal and self.history_aware and self.temporal_encoder is not None and T > 1:
            with torch.set_grad_enabled(not self._freeze_flag_temporal):
                temporal_out = self.temporal_encoder(
                    context_latents, residue_mask,
                    time_ps=time_ps, temperature_k=temperature_k,
                )
                frame_state = temporal_out["frame_states"][:, -1]
                h_last = context_latents[:, -1] + frame_state.unsqueeze(1)
        else:
            h_last = context_latents[:, -1]

        # -- Q = base projection + kinematic branch --
        q_base = self.project_to_q(h_last)
        kin_feat = self.compute_kin_features(coords, residue_mask_2d)
        kin_emb = self.kin_encoder(kin_feat)                       # [B,L,kin_hidden]
        q_kin = self.projection_kin(torch.cat([h_last, kin_emb], dim=-1))
        q = q_base + q_kin

        result = {"q": q, "h_last": h_last, "residue_mask": residue_mask_2d,
                  "q_base": q_base, "q_kin": q_kin, "kin_feat": kin_feat}

        if return_ca_distances:
            anchor_backbone = coords[:, -1]
            ca_positions = anchor_backbone[..., 1, :]
            result["ca_distances"] = torch.cdist(ca_positions, ca_positions)

        if return_bath and self.use_bath:
            result["bath_s0"] = self.initialize_bath(h_last, residue_mask_2d)

        if return_forces and return_ca_distances:
            force_out = self.compute_energy_and_forces(
                q, result["ca_distances"], residue_mask_2d, temperature_k,
            )
            result["forces"] = force_out["forces"]
            result["energy_total"] = force_out.get("energy_total")

        return result


__all__ = ["StageBLASP_Kinematic"]
