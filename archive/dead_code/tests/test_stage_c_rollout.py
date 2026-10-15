from __future__ import annotations

import torch
import torch.nn as nn

from protein_jepa_frame.models.stage_c.rollout import AutonomousRollout, RolloutConfig


class RecordingLaSP(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[dict] = []

    def step(
        self,
        q,
        s,
        *,
        ca_distances,
        residue_mask,
        temperature_k,
        dt_ps,
        static_condition=None,
    ):
        self.calls.append(
            {
                "ca_distances": ca_distances.clone(),
                "residue_mask": residue_mask.clone(),
                "dt_ps": float(dt_ps),
            }
        )
        mask = residue_mask.unsqueeze(-1).to(q.dtype)
        return q + float(dt_ps) * mask, s + mask


def _inputs():
    h = torch.randn(2, 5, 16)
    q = torch.zeros(2, 5, 3)
    s = torch.zeros(2, 5, 4)
    mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )
    ca = torch.randn(2, 5, 3)
    distances = torch.cdist(ca, ca)
    return h, q, s, mask, distances


def test_rollout_passes_frozen_graph_and_mask_to_every_step():
    lasp = RecordingLaSP()
    rollout = AutonomousRollout(
        lasp,
        RolloutConfig(
            delta_t_ps=10.0,
            bath_dim=4,
            dynamics_dim=3,
            context_dim=16,
        ),
    )
    h, q, s, mask, distances = _inputs()

    final = rollout.rollout_to_horizon(
        h,
        q,
        horizon_ps=25.0,
        bath_initial=s,
        residue_mask=mask,
        ca_distances=distances,
        temperature_k=torch.tensor([300.0, 310.0]),
    )

    assert [call["dt_ps"] for call in lasp.calls] == [10.0, 10.0, 5.0]
    assert all(torch.equal(call["residue_mask"], mask) for call in lasp.calls)
    assert all(torch.equal(call["ca_distances"], distances) for call in lasp.calls)
    assert final["step"] == 3
    assert torch.equal(final["residue_mask"], mask)
    assert torch.equal(final["ca_distances"], distances)
    assert torch.all(final["Q"][~mask] == 0)


def test_rollout_executes_remainder_when_horizon_is_shorter_than_dt():
    lasp = RecordingLaSP()
    rollout = AutonomousRollout(
        lasp,
        RolloutConfig(
            delta_t_ps=10.0,
            bath_dim=4,
            dynamics_dim=3,
            context_dim=16,
        ),
    )
    h, q, s, mask, distances = _inputs()

    final = rollout.rollout_to_horizon(
        h,
        q,
        horizon_ps=4.0,
        bath_initial=s,
        residue_mask=mask,
        ca_distances=distances,
    )

    assert [call["dt_ps"] for call in lasp.calls] == [4.0]
    assert final["step"] == 1
    assert torch.any(final["Q"][mask] != 0)


def test_rollout_rejects_missing_anchor_graph():
    rollout = AutonomousRollout(
        RecordingLaSP(),
        RolloutConfig(bath_dim=4, dynamics_dim=3, context_dim=16),
    )
    h, q, s, mask, _ = _inputs()

    try:
        rollout.initialize(h, q, s, residue_mask=mask, ca_distances=None)
    except ValueError as exc:
        assert "anchor CA-distance graph" in str(exc)
    else:
        raise AssertionError("missing anchor graph must be rejected")

