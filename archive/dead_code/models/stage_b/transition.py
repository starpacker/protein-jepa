"""P1.2: Closed-loop transition module for Stage B (physical-only).

z_{t+1} = physical_step(z_t, dt), noise handled by integrator's stochastic flag.

The learned residual/noise heads (ResidualTransitionNet, NoiseScaleNet) are
REMOVED.  They were never trained (stochastic=False in rollout_mse_loss) but
were used at eval (stochastic=True), creating a train/eval mismatch.  The
physical integrator (stage_b_lasp.step) already has its own FDT-consistent
noise via ``stochastic=True``, which is both trained and evaluated.

Gate 2.6 re-entry point: when the physical-only baseline passes all gates,
re-introduce a learned correction with explicit NLL on the full closed-loop
distribution, not as a separate untrained noise injection.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dataclasses import dataclass
from protein_jepa_frame.models.stage_b.state import LatentState


@dataclass
class StepResult:
    """Result of a single transition step."""
    next_state: LatentState
    forces: torch.Tensor | None = None
    diffusion: torch.Tensor | None = None
    energy_total: torch.Tensor | None = None
    diagnostics: dict | None = None


class ClosedLoopTransition(nn.Module):
    """Physical-only closed-loop transition (no learned residual/noise).

    Delegates to stage_b_lasp.step() which handles:
      - energy-based forces
      - FDT-consistent diffusion noise (when stochastic=True)
      - bath coupling
      - sub-stepping for stability

    This is a thin wrapper that provides the LatentState interface without
    any untrained correction terms.
    """

    def __init__(
        self,
        state_dim: int = 8,
        env_dim: int = 8,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.env_dim = env_dim

    def forward(
        self,
        z_t: LatentState,
        dt_ps: float,
        stage_b_lasp,          # StageBLASP instance (for physical step)
        ca_distances: torch.Tensor,
        residue_mask: torch.Tensor,
        temperature_k: torch.Tensor | None = None,
        stochastic: bool = True,
        max_substep_dt_ps: float = 20.0,
        noise_scale: float = 1.0,
    ) -> StepResult:
        """Execute one physical transition step (force + FDT noise + bath).

        Args:
            z_t: current latent state
            dt_ps: time step in ps
            stage_b_lasp: the Stage B model for physical integration
            ca_distances: frozen CA distance graph [B, L, L]
            residue_mask: [B, L] valid residues
            temperature_k: [B] temperature
            stochastic: whether to add FDT noise
            noise_scale: linear multiplier on noise (0=deterministic, 1=full FDT)

        Returns:
            StepResult with next_state and diagnostics
        """
        q_next, m_next, diagnostics = stage_b_lasp.step(
            z_t.q, z_t.m,
            ca_distances=ca_distances,
            residue_mask=residue_mask,
            temperature_k=temperature_k,
            dt_ps=dt_ps,
            stochastic=stochastic,  # consistent train/eval
            max_substep_dt_ps=max_substep_dt_ps,
            return_diagnostics=True,
            noise_scale=noise_scale,
        )

        z_next = LatentState(q=q_next, m=m_next, e=z_t.e)

        return StepResult(
            next_state=z_next,
            forces=diagnostics.get("forces") if diagnostics else None,
            diffusion=diagnostics.get("diffusion") if diagnostics else None,
            energy_total=diagnostics.get("energy_total") if diagnostics else None,
            diagnostics=diagnostics,
        )

    def rollout(
        self,
        z_0: LatentState,
        n_steps: int,
        dt_ps: float,
        stage_b_lasp,
        ca_distances: torch.Tensor,
        residue_mask: torch.Tensor,
        temperature_k: torch.Tensor | None = None,
        stochastic: bool = True,
        record_every: int = 1,
    ) -> list[StepResult]:
        """Autonomous rollout from z_0 for n_steps.

        Returns trajectory of StepResults.
        """
        trajectory: list[StepResult] = []
        z_t = z_0.clone()

        # Record initial state
        trajectory.append(StepResult(next_state=z_t.clone()))

        for step in range(n_steps):
            result = self.forward(
                z_t, dt_ps, stage_b_lasp, ca_distances,
                residue_mask, temperature_k, stochastic,
            )
            z_t = result.next_state

            if (step + 1) % record_every == 0:
                trajectory.append(StepResult(next_state=z_t.clone()))

        return trajectory


# Keep for backward compatibility / future re-introduction.
class ResidualTransitionNet(nn.Module):
    """PLACEHOLDER: Learned residual correction (disabled pending Gate 2.6)."""
    def __init__(self, state_dim: int = 8, env_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + env_dim + 1, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, state_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, q, e, dt_ps):
        return torch.zeros_like(q)


class NoiseScaleNet(nn.Module):
    """PLACEHOLDER: Learned noise scale (disabled pending Gate 2.6)."""
    def __init__(self, state_dim: int = 8, env_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + env_dim + 1, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, q, e, dt_ps):
        return torch.zeros_like(q)


__all__ = [
    "StepResult",
    "ClosedLoopTransition",
    "ResidualTransitionNet",
    "NoiseScaleNet",
]
