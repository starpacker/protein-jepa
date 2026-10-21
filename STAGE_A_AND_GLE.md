# Stage A + GLE — current-best code map

> **Purpose of this file.** A single, self-contained map of the **current best
> code** for the two live model families and the code that already joins them:
>
> 1. **Stage A** — JEPA-style *pretrained* protein-dynamics encoder
>    (`StageAV2`), trained by `train_stage_a_v2.py`.
> 2. **GLE (Stage B / "from-none")** — *from-scratch* encoder + **Generalized
>    Langevin (GLE)** latent dynamics kernel (`StageB1b`) + CA decoder, trained
>    end-to-end by `train_stage_cb1b.py`.
> 3. **A+GLE join** — freeze Stage A's encoder, replace only the static
>    Gaussian ensemble head with the trainable **GLE kernel**, trained by
>    `train_stage_b_hs.py` (the `HSEnsembleProducer`).
>
> Full numbers live in `PROJECT_REPORT_20261021.md`. This file only says
> **which file is which**.

---

## 0. TL;DR — the three entry points

| Family | Entry point | Model package | Best checkpoint (default) |
|---|---|---|---|
| **Stage A** (encoder pretrain) | `python -m protein_jepa_frame.training.train_stage_a_v2` | `protein_jepa_frame/models/stage_a_v2/` | `artifacts/stage_a_disp_relpos/checkpoint_latest.pt` |
| **GLE / from-none** (end-to-end) | `python -m protein_jepa_frame.training.train_stage_cb1b` | `protein_jepa_frame/models/stage_b1b_fromnone/` + `models/stage_b1b/` (kernel) | `artifacts/fromnone/stage_cb1b_v4e/checkpoint_latest.pt` |
| **A + GLE join** (Stage B "H+S") | `python -m protein_jepa_frame.training.train_stage_b_hs` | reuses **frozen** Stage A encoder + B1a decoder + trainable `StageB1b` kernel | `artifacts/stage_hs_1000dom/checkpoint_best.pt` |

All three default to the **`mdcath_manifest_1000dom`** dataset
(`train.jsonl` / `val.jsonl`).

---

## 1. Stage A — JEPA encoder pretraining

**What it is.** A frame-wise `ProteinGeometryEncoder` (online + EMA teacher) with
OPF factor projectors and a causal Transformer predictor. Trained with the
JEPA-Anything objective family plus a displacement/relative-position temporal
objective and a structure-reconstruction auxiliary head. Produces the latent
`H` used downstream.

**Train it**

```bash
bash run_stage_a_unified.sh          # 200dom (default in the script)
# or point it at 1000dom:
TRAIN_MANIFEST=artifacts/mdcath_manifest_1000dom/train.jsonl \
VAL_MANIFEST=artifacts/mdcath_manifest_1000dom/val.jsonl \
OUTPUT=artifacts/stage_a_disp_relpos bash run_stage_a_unified.sh
```

**Code**

| File | Role |
|---|---|
| `protein_jepa_frame/training/train_stage_a_v2.py` | training driver |
| `protein_jepa_frame/models/stage_a_v2/model.py` | `StageAV2`, `StageAV2Config` (encoder + predictor + OPF) |
| `protein_jepa_frame/models/stage_a_v2/objectives.py` | `compute_stage_a_v2_loss`, `StageAV2LossWeights` |
| `protein_jepa_frame/models/encoder.py` | shared `ProteinGeometryEncoder` (online/EMA, orientation & relative-position features) |
| `run_stage_a_unified.sh` | launch script |

**Key config ideas** (`StageAV2Config`): `latent_dim=128`, `num_factors=4` (OPF),
`orientation_features=True`, `relative_position_features=True`,
`deterministic_temporal=True`, EMA teacher (`ema_start=0.998` → `ema_end=0.9999`).

---

## 2. GLE (Stage B / "from-none") — end-to-end encoder + Langevin + decoder

**What it is.** A **randomly-initialised** `ProteinGeometryEncoder` co-trained
end-to-end with:

* `StageB1b` — the **GLE kernel**: an H-space energy `U(H)` + a per-residue bath
  `S` (Langevin with `bath_dim=8`), with a `rollout()` that samples stochastic
  H-trajectories from `H_anchor`;
* a **CA decoder** (`SimpleCADecoder` / `MessagePassingCADecoder` / `CRIDecoder`)
  mapping `H` → CA displacement (`ca = X_ctx + tanh-clipped Δx`).

Losses: point-prediction CA RMSD + latent energy score + ensemble distribution
terms (RMSF profile L1, RMSF-shape Pearson `1-r`, variance, spread floor —
added incrementally through the `v4*` generations).

**Train it**

```bash
bash run_fromnone_v4e.sh     # best checkpoint (v4e) — MLP decoder + RMSF-r loss
bash run_fromnone_v4f.sh     # from-scratch + CRIDecoder (in progress lineage)
bash run_fromnone_v4g.sh     # CRIDecoder resumed from anchor ckpt
```

**Code**

| File | Role |
|---|---|
| `protein_jepa_frame/training/train_stage_cb1b.py` | end-to-end training driver (the "cb1b" = from-none Stage B) |
| `protein_jepa_frame/models/stage_b1b/model.py` | **GLE kernel**: `StageB1b`, `StageB1bConfig`, `HSpaceEnergy`, `StaticBathParams`, `HistoryPosterior`, `rollout`, `sample_step` |
| `protein_jepa_frame/models/stage_b1b/objectives.py` | kernel-level objectives (drift/diffusion/Kalman NLL helpers) |
| `protein_jepa_frame/models/stage_b1b/residual_decoder.py` | residual CA decoder used by the kernel path |
| `protein_jepa_frame/models/stage_b1b_fromnone/decoder.py` | `FromNoneConfig`, `SimpleCADecoder` (per-residue MLP Δx head) |
| `protein_jepa_frame/models/stage_b1b_fromnone/mp_decoder.py` | `MessagePassingCADecoder` + `CRIDecoder` (residue-graph self-attention + H↔S cross-attention) |
| `protein_jepa_frame/models/stage_b1b_fromnone/objectives.py` | from-none losses: `aligned_ca_rmsd`, `energy_score`, `ensemble_rmsf_loss`, `ensemble_rmsf_corr_loss`, `ensemble_variance_loss`, `ensemble_spread_floor`, `FromNoneWeights` |
| `protein_jepa_frame/models/stage_c/geometry.py` | `RigidBodyFrame` (shared rigid-geometry primitives) |
| `run_fromnone_v4*.sh` | launch scripts (v4 / v4d / v4e best / v4f / v4g) |

**The `v4*` lineage** (see report §1.3, §5.2): `anchor → v4 → v4c/v4d → **v4e** (best)
→ v4f (CRIDecoder from scratch) → v4g (CRIDecoder resumed)`.

---

## 3. A + GLE join — "Stage B H+S on Model 1"

**What it is.** The experiment that **connects Stage A and the GLE kernel**:

* freezes the **Stage A encoder** and the **B1a anchored decoder**
  (the two pieces that give the SOTA `RMSF-r = 0.673` ensemble);
* throws away the static `GaussianHHead`;
* keeps **only the `StageB1b` GLE kernel trainable** (H-space energy `U(H)` +
  per-residue bath `S`);
* rolls out `N` GLE trajectories from the frozen `H_anchor`, decodes each through
  the **same frozen decoder**, and scores them with the **same** ensemble loss
  (`compute_ensemble_loss`: RMSF-L1 + contact-L1 + internal-distance soft-EMD).

**Result so far (2026-10-21, paired test, n=200): negative** — the static
Gaussian head remains the better inductive bias at 1 ns/frame; H+S only wins on
transient-contact Jaccard. See `STAGE_B_HS_RESULTS_20261021.md`.
**This is the join point to push on next.**

**Train / evaluate it**

```bash
bash run_stage_hs_1000dom.sh                 # 6000 steps on 1000dom
python -m scripts.paired_hs_vs_model1 --help # the paired H+S vs Model-1 test
```

**Code**

| File | Role |
|---|---|
| `protein_jepa_frame/training/train_stage_b_hs.py` | **the join**: `HSEnsembleProducer` (frozen encoder+decoder, trainable GLE kernel) |
| `protein_jepa_frame/training/train_stage_ensemble.py` | Model-1 reference driver + `load_frozen_stack` etc. reused by the join |
| `protein_jepa_frame/training/ensemble_losses.py` | shared `compute_ensemble_loss` (RMSF/contact/dist) — identical for Model 1 and the join |
| `protein_jepa_frame/training/ensemble_targets.py` | ensemble targets (`build_ensemble_targets`, `persistence_ensemble`, `oracle_ensemble`) |
| `protein_jepa_frame/training/train_stage_b1a.py` | B1a anchored-decoder training (`_extract_backbone_frames` etc. reused) |
| `protein_jepa_frame/models/stage_b1a/model.py` | `StageBOracleDecoder` — the anchored decoder (frozen in the join) |
| `protein_jepa_frame/models/ensemble_producer.py` | `GaussianHHead` + `EnsembleProducer` (Model-1 static head, for comparison) |
| `run_stage_hs_1000dom.sh` | launch script |

---

## 4. Shared infrastructure (both families depend on it)

| File | Role |
|---|---|
| `protein_jepa_frame/config.py` | `DataConfig` and shared config dataclasses |
| `protein_jepa_frame/data/dataset.py` | `build_dataset`, `TrajectoryClipDataset` |
| `protein_jepa_frame/data/collate.py` | `collate_paired_windows`, `collate_clips` |
| `protein_jepa_frame/data/protocol.py` | window/physical-time protocol, `apply_protocol`, split validation |
| `protein_jepa_frame/data/manifest.py` | manifest loading |
| `protein_jepa_frame/data/schema.py` | atom37 / residue naming |
| `protein_jepa_frame/training/train_utils.py` | `_device`, `_lr_scale`, `_move_batch`, `_set_reproducibility` |
| `protein_jepa_frame/utils/contracts.py` | prediction-contract records |
| `protein_jepa_frame/utils/backbone_geometry.py` | backbone frame helpers |
| `protein_jepa_frame/models/stage_c/structure_decoder.py` | `StructureDecoder` (frozen decode path used by Model 1 / the join) |
| `protein_jepa_frame/models/stage_c/objectives.py` | FAPE / torsion / distance losses for the decoder |
| `protein_jepa_frame/models/stage_c/openfold_atom37.py` | OpenFold-backed frame→atom37 (atom placement) |
| `protein_jepa_frame/vendored/openfold/` | vendored official OpenFold (Apache-2.0) |

---

## 5. Evaluation for these two families

| File | Role |
|---|---|
| `evaluation/ensemble_metrics.py` | RMSF / RMSF-r / root-mean-W2 / variance-W2 / MD-PCA-W2 / weak- & transient-contact Jaccard |
| `evaluation/ensemble_metrics_ext.py` | extension metrics (per-window, per-target, `*_full` variants with `n_true/n_pred`) |
| `scripts/eval_ensemble_metrics_suite.py` | the **suite** driver (model / persistence / oracle in one table) |
| `scripts/eval_ours_trajectory.py` | rollout (Kabsch) evaluation |
| `scripts/eval_b1a_manifest.py` | B1a anchored-decoder eval |
| `scripts/paired_hs_vs_model1.py` | **paired** H+S vs Model 1 test (same windows) |
| `scripts/export_eval_trajectories_pdb.py` | CA-trajectory PDB export (visual check) |
| `evaluation/baselines/` | bioemu2 zero-shot baseline runner + rep summariser |
| `tests/test_ensemble_metrics.py`, `tests/test_ensemble_losses.py`, `tests/test_ensemble_producer.py`, `tests/test_ensemble_targets.py`, `tests/test_stage_a_v2_temporal.py`, `tests/test_stage_b1b_autonomous.py`, `tests/test_stage_a_b1a_repairs.py` | unit / protocol tests for these paths |

---

## 6. What is **not** part of this map

Kept in the tree but **not** on the Stage A / GLE critical path (historical or
adjacent work): `models/stage_b2/`, `models/lspwm.py.*`, `baselines/`,
`mdgen_baseline/`, `archive/`, and the many dated `*_2026*.md` reports other than
the ones named above. When changing anything, ignore them.
