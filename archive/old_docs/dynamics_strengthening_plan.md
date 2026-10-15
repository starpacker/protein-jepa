# Dynamics Strengthening Plan

## Goal

Make the contextual JEPA state representation stronger for protein dynamics prediction, with evidence from strict VAMP-CV, trajectory-heldout MSM probes, linear dynamics, and FastFolding downstream probes.

## Step 1: Prevent Evaluation Regressions

- Use multi-frame clips for contextual models: `sequence_frames > 1`.
- Use `state` or `clip_state` readout for contextual claims.
- Use `--require-contextual` in the paper pipeline for formal runs.
- Keep MSM probe, VAMP-CV, and linear dynamics on the same clip protocol.

Formal quick evaluation template:

```bash
PYTHONPATH=/liuyutian/protein_jepa_frame \
/liuyutian/protein_jepa/.venv/bin/python -m protein_jepa_frame.eval.paper_eval pipeline \
  --checkpoint model=artifacts/mdcath_state_jepa_longhorizon_msm/checkpoint_step20000.pt \
  --output-dir artifacts/paper_eval/pipeline_quick_longhorizon_msm_step20000 \
  --preset quick \
  --device auto \
  --batch-size 4 \
  --sequence-frames 16 \
  --window-stride 5 \
  --linear-window-stride 8 \
  --readout state \
  --lambda-selection train_holdout \
  --require-contextual
```

## Step 2: Improve Long-Horizon Dynamics

The current multi-lag JEPA improves lag-1 linear dynamics, but lag-5 and lag-20 linear R2 remain weak. The next training config adds a linear multi-lag transition objective:

- `linear_dynamics_lags: [1, 2, 4, 8, 12, 16]`
- `linear_dynamics_weight: 0.35`

This explicitly encourages contextual frame states to support linear lagged prediction, which is aligned with the linear dynamics probe.

## Step 3: Improve MSM State Information

The next config also enables a small MSM auxiliary loss:

- `msm_state_weight: 0.02`
- labels are read from the existing deeptime-derived training labels

This should be treated as a separate semi-supervised/auxiliary variant, not as a pure self-supervised JEPA baseline.

## Step 4: Acceptance Criteria

The strengthened model is worth promoting only if it improves under the same strict protocol:

- VAMP-CV normalized score above single-lag and multi-lag baselines.
- MSM trajectory-heldout accuracy improves, with more proteins above baseline.
- Linear dynamics improves at lag 1 and does not collapse at lag 5/20.
- FastFolding smoke/full downstream probes improve without writing to the source dataset.
