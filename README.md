# Protein JEPA Frame

A compact research framework for protein-dynamics representation learning and
latent-state prediction.

The canonical evaluation contract is:

```text
X_{1:K} [B,K,L,4,3]
  -> frame-wise encoder
H_context [B,K,L,dH]
  -> predictor(H_context, tau, temperature)
H_future [B,L,dH]
  -> shared structure decoder
X_hat_future [B,L,3]
```

For the current Stage-A checkpoints, the predictor emits OPF factor-coordinate
deltas `[B,L,J,r]`. Evaluation reconstructs the absolute future latent as:

```text
H_future = OPF.compose(OPF.decompose(H_anchor) + pred_factors)
```

The autonomous H+S path instead initializes a bath state from the full history
and rolls out the learned GLE transition from `H_anchor`.

## Repository layout

- `protein_jepa_frame/` — model, data, training, and vendored geometry code.
- `evaluation/` — leakage-audited CA-RMSD evaluation and checkpoint adapters.
- `tests/` — unit and protocol tests.
- `scripts/` and `run_*.sh` — dataset preparation and training entry points.
- `EVALUATION_PLAN.md` — the authoritative evaluation protocol and acceptance criteria.
- `STAGE_AB_REDESIGN_PLAN.md` — model/training design specification.

Checkpoints, trajectories, generated reports, caches, and scratch experiments
are intentionally not part of this repository. Their canonical paths and
hashes are recorded in `evaluation/MODELS.md` when an evaluation run uses them.

## Install and test

```bash
pip install -e ".[dev]"
python -m pytest -q
```

The evaluation package has an additional smoke path:

```bash
python -m pytest evaluation/tests -q
python -m evaluation.run_eval --config evaluation/configs/smoke.yaml \
  --max-proteins 2 --horizons 1 4
```

See `evaluation/README.md` and `evaluation/RUNBOOK.md` for the full held-out
protein and held-out-temperature protocols.
