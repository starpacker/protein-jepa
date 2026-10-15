"""Autonomous recursive rollout for Stage C.

Gate 6 - autonomous_recursive mode:
- Bath initialized once from history (not refreshed)
- Fixed delta_t_ps substep integration
- Each step uses only (Q, S), static conditions, frozen graph
- No sliding-window teacher refresh
- Training: truncated BPTT with step curriculum
- Validation/testing: always complete free rollout

Architecture invariant (from audit plan Gate 6):
- History encoder runs only once per residue
- No future H or coordinates as input during rollout
- Final step uses recorded remainder step if horizon not divisible by dt
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RolloutConfig:
    """Configuration for autonomous recursive rollout."""
    delta_t_ps: float = 100.0       # integration substep time in ps
    max_rollout_steps: int = 1000   # maximum steps in full rollout
    bath_dim: int = 16               # per-residue bath dimension S
    dynamics_dim: int = 8            # per-residue dynamics dimension Q
    context_dim: int = 256            # H_t0 context dimension

    # Curriculum training
    curriculum_start_steps: int = 1
    curriculum_end_steps: int = 100
    curriculum_ramp_epochs: int = 100

    # Truncated BPTT
    bptt_chunk_size: int = 10        # number of steps per BPTT chunk

    freeze_integrator: bool = True   # freeze LaSP integrator params during rollout training


def _compute_remainder_step(horizon_ps: float, delta_t_ps: float) -> tuple[int, float]:
    """Compute number of full steps and remainder for given horizon.

    Args:
        horizon_ps: total horizon time in ps
        delta_t_ps: substep time in ps

    Returns:
        num_full_steps: integer number of complete dt steps
        remainder_ps: remainder time for last step (0 if exact multiple)
    """
    num_full = int(horizon_ps / delta_t_ps)
    remainder = horizon_ps - num_full * delta_t_ps
    if remainder < 1e-3 * delta_t_ps:
        remainder = 0.0
    if remainder < 0:
        remainder = 0.0
    return num_full, remainder


def rollout_schedule(
    epoch: int,
    config: RolloutConfig,
) -> int:
    """Compute rollout steps for curriculum training.

    Linearly ramps from curriculum_start_steps to curriculum_end_steps.

    Args:
        epoch: current training epoch
        config: rollout configuration

    Returns:
        number of rollout steps for this epoch
    """
    if config.curriculum_ramp_epochs <= 0:
        return config.curriculum_end_steps

    start = config.curriculum_start_steps
    end = config.curriculum_end_steps
    ramp = config.curriculum_ramp_epochs

    fraction = min(1.0, epoch / max(1, ramp))
    steps = int(start + (end - start) * fraction)
    return max(1, steps)


class AutonomousRollout(nn.Module):
    """Autonomous recursive rollout using frozen LaSP integrator.

    Assembles the full pipeline:
    1. History encoder (Stage A frozen) → H_t0
    2. Projection H_t0 → Q_t0 (Stage B learned)
    3. Bath initialized from history once → S_0
    4. For each step: LaSP integrator advances (Q_t, S_t) → (Q_{t+dt}, S_{t+dt})
    5. At each target time: Q_t fed to StructureDecoder → backbone

    No future information access during rollout.
    """

    def __init__(
        self,
        stage_b_lasp: nn.Module | None = None,
        config: RolloutConfig | None = None,
    ):
        super().__init__()
        self.config = config or RolloutConfig()
        self.lasp = stage_b_lasp  # must be frozen for the rollout

    def initialize(
        self,
        h_history: torch.Tensor,
        q_initial: torch.Tensor,
        bath_initial: torch.Tensor | None = None,
        residue_mask: torch.Tensor | None = None,
        ca_distances: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Initialize rollout state from history.

        Args:
            h_history: [B, L, 256] frozen H from context encoder
            q_initial: [B, L, 8] initial Q at t0
            bath_initial: [B, L, 16] initial bath state (optional)

        Returns:
            state dict with Q, S, step counter
        """
        B, L = q_initial.shape[:2]
        device = q_initial.device

        if bath_initial is None:
            bath_initial = torch.zeros(B, L, self.config.bath_dim, device=device)
        if residue_mask is None:
            residue_mask = torch.ones(B, L, dtype=torch.bool, device=device)
        if ca_distances is None:
            raise ValueError(
                "autonomous rollout requires the frozen anchor CA-distance graph"
            )
        if ca_distances.shape != (B, L, L):
            raise ValueError(
                f"ca_distances shape {tuple(ca_distances.shape)} != {(B, L, L)}"
            )

        return {
            "Q": q_initial.clone(),
            "S": bath_initial.clone(),
            "step": 0,
            "h_frozen": h_history.clone(),
            "residue_mask": residue_mask.clone(),
            "ca_distances": ca_distances.clone(),
        }

    def step(
        self,
        state: dict[str, torch.Tensor],
        static_condition: torch.Tensor | None = None,
        temperature_k: torch.Tensor | None = None,
        dt_ps: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Advance state by one integration step.

        Args:
            state: current state dict
            static_condition: [B, L, D] optional
            temperature_k: [B] or scalar
            dt_ps: override delta_t_ps

        Returns:
            new_state dict
        """
        dt = dt_ps if dt_ps is not None else self.config.delta_t_ps
        Q = state["Q"]
        S = state["S"]

        if self.lasp is not None:
            # Use the frozen LaSP step
            if hasattr(self.lasp, "transition"):
                new_Q, new_S, _ = self.lasp.transition(
                    Q, S,
                    ca_distances=state["ca_distances"],
                    residue_mask=state["residue_mask"],
                    temperature_k=temperature_k,
                    dt_ps=dt,
                    stochastic=False,
                )
            else:
                new_Q, new_S = self.lasp.step(
                    Q, S,
                    ca_distances=state["ca_distances"],
                    residue_mask=state["residue_mask"],
                    temperature_k=temperature_k,
                    dt_ps=dt,
                )
        else:
            # Passthrough (no dynamics)
            new_Q = Q
            new_S = S

        return {
            "Q": new_Q,
            "S": new_S,
            "step": state["step"] + 1,
            "h_frozen": state["h_frozen"],
            "residue_mask": state["residue_mask"],
            "ca_distances": state["ca_distances"],
        }

    def rollout(
        self,
        state: dict[str, torch.Tensor],
        n_steps: int,
        static_condition: torch.Tensor | None = None,
        temperature_k: torch.Tensor | None = None,
        dt_ps: float | None = None,
        record_every: int = 1,
        remainder_ps: float = 0.0,
    ) -> list[dict[str, torch.Tensor]]:
        """Roll out dynamics for n_steps.

        Args:
            state: initial state dict from initialize()
            n_steps: number of full steps
            static_condition: [B, L, D]
            temperature_k: [B]
            dt_ps: override dt
            record_every: record state every N steps
            remainder_ps: remainder time for final partial step

        Returns:
            list of state dicts at recorded steps
        """
        dt = dt_ps if dt_ps is not None else self.config.delta_t_ps
        records = [state]

        for step_idx in range(n_steps):
            state = self.step(state, static_condition, temperature_k, dt)

            if (step_idx + 1) % record_every == 0:
                records.append(state)

        # A horizon shorter than one full step still has to advance once.
        # Keeping this outside the loop also makes the elapsed time exactly
        # n_steps * dt + remainder rather than coupling it to record_every.
        if remainder_ps > 0:
            state = self.step(state, static_condition, temperature_k, remainder_ps)
            records.append(state)
        elif records[-1] is not state:
            records.append(state)

        return records

    def rollout_to_horizon(
        self,
        h_history: torch.Tensor,
        q_initial: torch.Tensor,
        horizon_ps: float,
        static_condition: torch.Tensor | None = None,
        temperature_k: torch.Tensor | None = None,
        bath_initial: torch.Tensor | None = None,
        residue_mask: torch.Tensor | None = None,
        ca_distances: torch.Tensor | None = None,
        delta_t_ps: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Roll out to specified horizon time.

        Args:
            h_history: [B, L, 256]
            q_initial: [B, L, 8]
            horizon_ps: target horizon in ps
            static_condition: [B, L, D]
            temperature_k: [B]
            bath_initial: [B, L, 16]

        Returns:
            final state at horizon (Q, S, h_frozen, step)
        """
        dt = float(delta_t_ps if delta_t_ps is not None else self.config.delta_t_ps)
        n_steps, remainder = _compute_remainder_step(horizon_ps, dt)

        state = self.initialize(
            h_history,
            q_initial,
            bath_initial,
            residue_mask=residue_mask,
            ca_distances=ca_distances,
        )
        records = self.rollout(
            state, n_steps, static_condition, temperature_k, dt,
            remainder_ps=remainder,
        )
        return records[-1]


__all__ = [
    "RolloutConfig",
    "AutonomousRollout",
    "rollout_schedule",
    "_compute_remainder_step",
]
