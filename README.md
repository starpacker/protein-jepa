# Protein JEPA Frame

A compact research framework for protein-dynamics representation learning and
latent-state prediction on **mdCATH** (5 temperatures: 320 / 348 / 379 / 413 / 450 K).

---

## 📌 Start here — the two canonical documents (evaluation reference)

| # | Document | What it is |
|---|---|---|
| 1 | **[`EVALUATION_METHODS_AND_COMMANDS_20261021.md`](EVALUATION_METHODS_AND_COMMANDS_20261021.md)** | **The authoritative evaluation methods + reproduction commands + full results reference.** §1.2 lists **every evaluated method** (Model 1 / Model 2 / Lasp-final / H+S / MDGen / bioemu2 + persistence & oracle), §2 defines *every* metric (CA-RMSD / Kabsch, RMSF & RMSF-r, Root-Mean / Variance / MD-PCA W2, contact Jaccard, energy scores, CA geometry validity, backbone N/CA/C/O stereochemistry, BioEmu-style static + MSM dynamics, order-aware trajectory metrics, temperature generalization) with units/direction/code, §9 gives **15 verified runnable commands** (one per result table), and §3–§7 carry all result tables. Every number carries its `n` + sampling budget; static vs. dynamics metrics are separated. |
| 2 | **[`PROJECT_REPORT_20261021.md`](PROJECT_REPORT_20261021.md)** | **Full self-contained project report** — models, training recipes, results vs. persistence / MDGen / bioemu2, and limitations. |

> Its plan / companion protocol lives in `archive/docs_20261031/`.

Everything else in the repo is either **code** (below) or **archived history** (`archive/`, see [Repository layout](#repository-layout)).

---

## ⭐ Current framework — LASP correct-framework rebuild (2026-10-31)

This is the **baseline for further work**, replacing the frozen SOTA as the
reference implementation. It keeps the SOTA dynamics and fixes the bath so the
memory variable `S` actually enters the trajectory.

| Document | What it is |
|---|---|
| [`LASP_CORRECT_FRAMEWORK_PLAN_20261031.md`](LASP_CORRECT_FRAMEWORK_PLAN_20261031.md) | Design, pre-registered criteria, tooling. |
| [`RESULTS_LASP_CORRECT_FRAMEWORK_20261031.md`](RESULTS_LASP_CORRECT_FRAMEWORK_20261031.md) | Kalman sign fix, exact `S₀` posterior, memory bound, 9-arm ablation, verdict. |

State of the art it must beat: the frozen SOTA ensemble in
`results/sota_frozen_20261024_backup/` (never re-evaluated; kept as the yardstick).

What is fixed relative to the old SOTA:

- the Kalman gain sign (`K = −P·B_hᵀ·V⁻¹`; the covariance was fine, the mean was not);
- `S₀` is a full Kalman posterior over the history (`s0_mode=exact`), not an
  untrained amortised GRU, and not a "drive-only" recursion that is 76–95% off;
- the bath is trained at a scale where `S` moves the trajectory
  (`--coupling-scale 0.1 --gamma-scale 2e-5 --gamma-floor 2e-6`).

Known limitation, measured: history carries only **2–4%** linearly-predictable
memory beyond `H₀`, so `exact` vs `prior` does not separate on the primary
metrics, and the retrained arms do **not** beat the frozen SOTA on
`rmsf_corr` / `rmsf_mae` / `root_mean_w2`.  They do win on `md_pca_w2`.
See the results document for the full verdict and the next-step options.

Reproduce:

```bash
export PYTHONPATH=/tmp/lasp_stability_deps:.
bash run_scripts/drive_lasp_s0_arms.sh        # 9-arm training grid
bash run_scripts/run_lasp_s0_arms_eval.sh     # standard-200 ensemble metrics
bash run_scripts/analyze_lasp_s0_arms.sh      # paired + 3-seed verdicts
```

## Secondary documents

The 2026-10 Stage-A / GLE / Stage-B reports referenced by earlier revisions of
this file (`STAGE_A_AND_GLE.md`, `STAGE_B_HS_RESULTS_20261021.md`,
`EVALUATION_PLAN.md`, `REPAIR_PROGRESS.md`,
`NEXT_STEP_EVALUATION_3TASKS_20261021.md`) are kept in
`archive/docs_20261031/` — they document the superseded pipeline and are not
part of the live code.

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
[`EVALUATION_METHODS_AND_COMMANDS_20261021.md`](EVALUATION_METHODS_AND_COMMANDS_20261021.md).
The corresponding shell recipes live in [`run_scripts/`](run_scripts/):

| Task | Recipe |
|---|---|
| T1 backbone suite | `scripts/eval_backbone_suite.py` |
| T2 static + MSM | `scripts/eval_states_and_msm.py` |
| T3 temperature (Model 1) | `run_scripts/run_ensemble_temp_1000dom.sh` |
| T3 temperature (Model 2) | `scripts/eval_temperature_generalization_model2.py` |
