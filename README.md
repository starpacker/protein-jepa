# Protein JEPA Frame

A compact research framework for protein-dynamics representation learning and
latent-state prediction on **mdCATH** (5 temperatures: 320 / 348 / 379 / 413 / 450 K).

---

## 📌 Start here — the two canonical documents

| # | Document | What it is |
|---|---|---|
| 1 | **[`EVALUATION_3TASKS_RESULTS_20261021.md`](EVALUATION_3TASKS_RESULTS_20261021.md)** | **The authoritative evaluation plan + full results reference.** Defines *every* metric (CA-RMSD / Kabsch, RMSF & RMSF-r, Root-Mean / Variance / MD-PCA W2, contact Jaccard, energy scores, CA geometry validity, backbone N/CA/C/O stereochemistry, BioEmu-style static + MSM dynamics, order-aware trajectory metrics, temperature generalization) with units/direction/code, plus all result tables — the 3 newest tasks **and** every earlier evaluation. Every number carries its `n` + sampling budget; static vs. dynamics metrics are separated. |
| 2 | **[`PROJECT_REPORT_20261021.md`](PROJECT_REPORT_20261021.md)** | **Full self-contained project report** — models, training recipes, results vs. persistence / MDGen / bioemu2, and limitations. |

> Its plan / companion protocol is [`NEXT_STEP_EVALUATION_3TASKS_20261021.md`](NEXT_STEP_EVALUATION_3TASKS_20261021.md).

Everything else in the repo is either **code** (below) or **archived history** (`archive/`, see [Repository layout](#repository-layout)).

---

## Secondary documents (read in this order)

- [`STAGE_A_AND_GLE.md`](STAGE_A_AND_GLE.md) — map of the **current-best code** for the two live model families (**Stage A** JEPA encoder + **GLE / from-none** latent Langevin dynamics) and the code that joins them.
- [`STAGE_B_HS_RESULTS_20261021.md`](STAGE_B_HS_RESULTS_20261021.md) — the Stage-A + GLE join ("Stage B H+S") result.
- [`EVALUATION_PLAN.md`](EVALUATION_PLAN.md) — the authoritative evaluation protocol and acceptance criteria.
- [`REPAIR_PROGRESS.md`](REPAIR_PROGRESS.md) — 2026-10-17+ repair status and pilot evidence.

Historical evaluation reports do **not** establish forecasting skill. New B1b training
uses history-only autonomous H+S predictions; future H is a detached latent target.
Its physical loss compares decoded predictions with real future coordinates.
Formal B1b runs require a passed, matching [H readout gate](evaluation/run_h_gate.py).

## The canonical evaluation contract

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

| Path | Contents |
|---|---|
| `protein_jepa_frame/` | Model, data, training, and vendored OpenFold geometry code. |
| `evaluation/` | Leakage-audited CA/backbone-RMSD evaluation, MSM & BioEmu metrics, checkpoint adapters. |
| `tests/` | Unit and protocol tests (`python -m pytest -q`). |
| `scripts/` | Dataset preparation, training and evaluation entry points (Python). |
| `run_scripts/` | The **current** shell recipes (`run_*.sh`) referenced by the canonical docs. |
| `results/` · `artifacts/` | Evaluation JSON results and derived products (large; git-ignored). |
| `archive/` | Frozen historical reports, scripts and scratch (`old_docs_20261021/`, `old_scripts_20261021/`, …). *Not part of the live code.* |

Checkpoints, trajectories, generated reports, caches, and scratch experiments
are intentionally not part of this repository. Their canonical paths and hashes
are recorded in `evaluation/MODELS.md` when an evaluation run uses them.

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

---

## Reproducing the 3-task evaluation

Each task's exact command is in the "Repro" section of
[`EVALUATION_3TASKS_RESULTS_20261021.md`](EVALUATION_3TASKS_RESULTS_20261021.md).
The corresponding shell recipes live in [`run_scripts/`](run_scripts/):

| Task | Recipe |
|---|---|
| T1 backbone suite | `scripts/eval_backbone_suite.py` |
| T2 static + MSM | `scripts/eval_states_and_msm.py` |
| T3 temperature (Model 1) | `run_scripts/run_ensemble_temp_1000dom.sh` |
| T3 temperature (Model 2) | `scripts/eval_temperature_generalization_model2.py` |
