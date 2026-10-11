"""Latent projection and role readout for LaSP.

Contains:
  - UnifiedLatentRoleReadout: read multiple scientific views from shared state
  - msm_consistency_loss: self-supervised soft-MSM objective
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_jepa_frame.models.lasp.objectives import vamp2_loss


class UnifiedLatentRoleReadout(nn.Module):
    """Read several scientific views from one shared coarse state.

    The role dimensions describe probe capacity, not fixed coordinates in ``z``.
    Every probe sees the complete state, so the JEPA representation remains a
    single 256-dimensional object and can reuse information across objectives.
    """

    def __init__(
        self,
        state_dim: int,
        fixed_cv_dim: int,
        condition_dim: int,
        *,
        koopman_dim: int = 64,
        msm_dim: int = 32,
        geometric_dim: int = 96,
        auxiliary_dim: int = 64,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if any(v < 1 for v in (state_dim, fixed_cv_dim, condition_dim,
                                koopman_dim, msm_dim, geometric_dim, auxiliary_dim)):
            raise ValueError("unified latent role dimensions must be positive")
        if hidden_dim < 4:
            raise ValueError("hidden_dim must be at least 4")
        self.state_dim = int(state_dim)
        self.fixed_cv_dim = int(fixed_cv_dim)
        self.condition_dim = int(condition_dim)
        self.koopman_dim = int(koopman_dim)
        self.msm_dim = int(msm_dim)
        self.geometric_dim = int(geometric_dim)
        self.auxiliary_dim = int(auxiliary_dim)

        def readout(output_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.LayerNorm(state_dim),
                nn.Linear(state_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        self.koopman = readout(self.koopman_dim)
        self.msm = readout(self.msm_dim)
        self.geometric = readout(self.geometric_dim)
        self.auxiliary = readout(self.auxiliary_dim)
        self.geometric_decoder = nn.Sequential(
            nn.LayerNorm(self.geometric_dim),
            nn.Linear(self.geometric_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.fixed_cv_dim),
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        if z.ndim != 2 or z.shape[-1] != self.state_dim:
            raise ValueError(f"z must have shape [batch, {self.state_dim}]")
        return {
            "koopman": self.koopman(z),
            "msm_logits": self.msm(z),
            "geometric": self.geometric(z),
            "auxiliary": self.auxiliary(z),
        }

    def losses(
        self,
        current: torch.Tensor,
        target: torch.Tensor,
        fixed_cv: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Return role-guidance losses without partitioning the latent vector."""
        current_roles = self(current)
        with torch.no_grad():
            target_roles = self(target.detach())
        target_auxiliary = condition.detach()
        if target_auxiliary.shape[-1] < self.auxiliary_dim:
            target_auxiliary = F.pad(
                target_auxiliary, (0, self.auxiliary_dim - target_auxiliary.shape[-1])
            )
        elif target_auxiliary.shape[-1] > self.auxiliary_dim:
            target_auxiliary = target_auxiliary[..., :self.auxiliary_dim]
        geometric_prediction = self.geometric_decoder(current_roles["geometric"])
        return {
            "lspwm_koopman_loss": vamp2_loss(
                current_roles["koopman"], target_roles["koopman"], ridge=1.0e-3
            ),
            "lspwm_msm_loss": msm_consistency_loss(
                current_roles["msm_logits"], target_roles["msm_logits"]
            ),
            "lspwm_geometric_loss": F.smooth_l1_loss(
                geometric_prediction, fixed_cv.detach()
            ),
            "lspwm_auxiliary_loss": F.smooth_l1_loss(
                current_roles["auxiliary"], target_auxiliary
            ),
        }, current_roles


def msm_consistency_loss(
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    *,
    balance_weight: float = 0.1,
    confidence_weight: float = 0.02,
) -> torch.Tensor:
    """Self-supervised soft-MSM objective for paired coarse states."""
    if source_logits.shape != target_logits.shape or source_logits.ndim != 2:
        raise ValueError("MSM logits must have the same shape [batch, num_states]")
    source = source_logits.softmax(dim=-1)
    target = target_logits.softmax(dim=-1).detach()
    overlap = (source * target).sum(dim=-1).clamp(min=1.0e-8)
    marginal = source.mean(dim=0)
    num_states = source.shape[-1]
    balance = num_states * (marginal - 1.0 / num_states).square().sum()
    entropy = -(source * source.clamp(min=1.0e-8).log()).sum(dim=-1).mean()
    return -overlap.log().mean() + balance_weight * balance + confidence_weight * entropy
