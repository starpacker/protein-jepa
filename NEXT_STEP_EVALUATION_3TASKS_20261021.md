# Next-Step Plan — Three Evaluation Tasks
> **Date:** 2026-10-21  ·  **Base report:** `PROJECT_REPORT_20261021.md`
> **Scope:** evaluation only (no retraining of the frozen encoders/decoders except where a task explicitly requires a new head or a temp-holdout run).
> **Deliverable of this doc:** a concrete, task-by-task plan grounded in the *current* code + artifacts, with exact file touch-points, commands, and acceptance criteria.

---

## 0. Investigation summary (what already exists)

### 0.1 Task 1 — OpenFold backbone reconstruction
| Fact | Evidence |
|---|---|
| A **vendored official OpenFold** already lives in-tree | `protein_jepa_frame/vendored/openfold/` (aqlaboratory/openfold, Apache-2.0) |
| An adapter **already delegates** atom placement to OpenFold | `protein_jepa_frame/models/stage_c/openfold_atom37.py` (`frames_to_atom37`, `decoder_frames_to_atom37`, `backbone_frame_from_coords = Rigid.make_transform_from_reference`) |
| The StageC decoder **already returns full backbone + atom37** | `structure_decoder.py::forward` → returns `backbone_direct` `[B,L,4,3]` (N,CA,C,O), `atom37_coords` `[B,L,37,3]`, `torsions` `[B,L,7]`. Reconstruction path: `reconstruct_atom37_from_frames(R, t, aatype, torsions)` → OpenFold `torsion_angles_to_frames` + `frames_and_literature_positions_to_atom14_pos` + `atom14_to_atom37`. |
| **BUT**: every eval / export path **throws away everything but Cα** | `ensemble_producer.py:151` and `train_stage_b_hs.py:204` both do `out["backbone_coords"][:, :, 1, :]` → CA only. `eval_ensemble_metrics_suite.py` scores `out["ca_samples"]`. `export_eval_trajectories_pdb.py` writes **CA-only** PDBs. |
| The **from-none (Model 2)** decoder is **Cα-only by construction** | `stage_b1b_fromnone/decoder.py::SimpleCADecoder` predicts only a Cα displacement `ca_pred [B,L,3]`; no frames/torsions → **no OpenFold input** today. |

**Task-1 conclusion:** the OpenFold machinery is real and correct, but it is **only wired to the SOTA StageC decoder (Model 1)** and the eval layer never consumes it. The core work is (a) expose the OpenFold backbone **through the ensemble producers and into evaluation/export**, and (b) quantify backbone quality with stereochemistry metrics. Model 2 needs a *path to a backbone* (its own frame/torsion head, or an OpenFold-based "backbone lifter" from Cα) — decided explicitly below.

### 0.2 Task 2 — BioEmu-style metrics + MSM state prediction
| Fact | Evidence |
|---|---|
| Existing 7 ensemble metrics | `evaluation/ensemble_metrics.py` + `ensemble_metrics_ext.py` (RMSF, RMSF-r, root/variance/pca W2, weak/transient contact J) |
| Geometry diagnostics (Cα-only) | `evaluation/geometry.py::ca_geometry_summary`, `ca_observable_errors` (bond range, clash, Rg, pseudo-torsion) |
| BioEmu's **actual** paper metrics are *coverage* + *metastable free-energy MAE* (NOT MSM next-state accuracy) | `BioEmu_指标梳理_蛋白质动力学基准.md` §1–2 |
| **`mdtraj` is installed**; **`deeptime` / `pyemma` are NOT** | verified at runtime |
| No TICA/HMM/MSM code exists in the tree | grep finds nothing |

**Task-2 conclusion:** implement (i) **reference-state coverage**, (ii) **metastable-state free-energy MAE**, (iii) population JSD/TV, and (iv) the **MSM state-prediction** family (lagged accuracy/F1, transition-matrix error, implied timescales, MFPT) that the source doc explicitly says BioEmu does *not* do but that we *do* need. Requires an MSM/TICA stack — `deeptime` is the clean choice (pip-installable) or a self-contained TICA+HMM.

### 0.3 Task 3 — Temperature generalization
| Fact | Evidence |
|---|---|
| 5 temperatures in mdCATH: **320, 348, 379, 413, 450 K** | verified from `artifacts/mdcath_manifest_1000dom/val.jsonl` (500 each) |
| Temperature **is** a first-class input to both model families | Model 1 path: StageA-v2 `time_embed(lag_ps, temperature_k)` (`stage_a_v2/model.py`), StageB1b GLE `temperature_proj` + `_temp_feature` (`stage_b1b/model.py`), `rollout(..., temperature_k, ...)`. Model 2: `autonomous_rollout` passes `batch["temperature_k"]`. |
| A held-out-T=450 manifest **already exists** | `artifacts/mdcath_manifest_1000dom_temp/{train_not450,val_not450,val_t450}.jsonl` |
| A temp-holdout training script + a partial run exist | `run_fromnone_temp.sh`, `artifacts/fromnone/stage_cb1b_temp/` (log shows a **non-finite-grad crash**) |
| A temp-holdout eval path exists | `evaluation/run_eval_fromnone.py --split temp --heldout-temperature`, `evaluation/build_temp_holdout.py`, `evaluation/configs/mdcath_holdout_temp.yaml` |

**Task-3 conclusion:** the *plumbing* exists but is (a) partial/crashed, (b) only for Model 2, (c) not a "leave-one-T-out + seen-protein unseen-T" matrix, and (d) never surfaced as a generalization metric. The real work is a **temperature-generalization protocol** (define the split scheme, make BOTH models temperature-aware at eval, run the matrix, and report skill-vs-ΔT), plus fixing the from-none temp run.

---

## 1. Shared conventions (apply to all three tasks)

- **Data / gauge:** all metrics computed in the same anchors and Kabsch gauge already used by `evaluation/ensemble_metrics*.py`. Never introduce a new alignment.
- **Do not regress the frozen path:** Model 1's encoder/decoder stay frozen; Task 1 only *surfaces* its existing atom37 output. Task 2/3 add eval code + (for MSM) a feature extractor.
- **New code lands as new modules**, mirroring the existing `*_ext.py` pattern; touch the originals as little as possible.
- **Every new metric gets a unit test** in `tests/` with an oracle sanity check (identity → perfect score), matching `tests/test_ensemble_metrics*.py`.
- **Report discipline (from the source doc):** keep **static/ensemble metrics** and **time-dynamics metrics** in **two separate tables**, always attach `n_true/n_pred` to contact/Jaccard-style numbers, and always state the sampling budget next to coverage.

---

## 2. TASK 1 — OpenFold-based **mainchain (backbone) atom** reconstruction

### 2.1 Goal
Produce and **evaluate** a real **N, CA, C, O (backbone)** reconstruction — not just Cα — for our model ensembles, where the atom placement is done by **OpenFold** (no hand-written ideal tables). Then add backbone-level quality metrics (stereochemistry + backbone RMSD) on top of the existing Cα metrics.

### 2.2 What is already correct (reuse, don't rewrite)
- `stage_c/openfold_atom37.py` is the single source of truth for frame→atom37. **Model 1 already gets a full backbone for free** inside `StructureDecoder.forward` (keys `backbone_direct`, `atom37_coords`, `N/CA/C/O`).
- `reconstruct_atom37(N, CA, C, O, aatype, torsions)` uses `Rigid.make_transform_from_reference` (OpenFold Algorithm 21) — this is the entry point if we ever start from backbone coords.

### 2.3 Work items

**T1.1 — Surface the OpenFold backbone through the producers (Model 1).**
- Files: `protein_jepa_frame/models/ensemble_producer.py`, `protein_jepa_frame/training/train_stage_b_hs.py`.
- In `EnsembleProducer.sample_future_ca`: after `out = self.decoder.decode_structure(flat_H, ...)`, also collect `out["backbone_direct"]` → reshape to `[B, n_samples, L, 4, 3]` and `out["atom37_coords"]` → `[B, n_samples, L, 37, 3]`, plus `out["torsions"]`. Return new keys `backbone_samples`, `atom37_samples`, `torsion_samples` **alongside** the existing `ca_samples` (do not change existing keys → no downstream breakage).
- Same change in `HSEnsembleProducer._rollout_decode` (its decoder is the same `decode_structure`), and expose `temperature_k` already passed in.
- **Check** the atom ordering: `atom37_coords` is in **OpenFold ordering**; use `openfold_atom37.to_project_ordering(...)` if any consumer assumes `data/schema.ATOM37_NAMES`.

**T1.2 — Give Model 2 (from-none) a backbone.** Two options; recommend **B**:
- **A (cheap, honest):** an OpenFold **"Cα lifter"** — from a predicted Cα trace, reconstruct N, CA, C, O by (i) building a per-residue frame from Cα + a smooth Cα-pseudo-bond geometry, or better (ii) a small **backbone-torsion head** trained to predict φ/ψ/ω, then feed `reconstruct_atom37_from_frames` (OpenFold) exactly like Model 1. This *keeps OpenFold as the atom placer*.
- **B (cleanest, matches Model 1):** add a lightweight **frame + torsion head** to `SimpleCADecoder` (quaternion frame update + 3-torsion (φ,ψ,ω) head, à la `stage_c/structure_decoder.py::FrameUpdate`+`TorsionHead`), trained with the existing torsion/FAPE-style losses, and route through `decoder_frames_to_atom37`. This gives a true N/CA/C/O + sidechain-ready structure.
- **Recommendation:** implement **A first** (a `scripts/lift_ca_to_backbone_openfold.py` that takes any `[.., L, 3]` Cα ensemble → `[.., L, 4, 3]` backbone via OpenFold), so **both models** get a backbone immediately; then optionally train B for accuracy. A is model-agnostic and needs no retraining, so it unblocks Task-1 metrics for Model 2 today.

**T1.3 — Backbone-level metrics** (new `evaluation/backbone_metrics.py`).
Implement, all mask/chain/residue-index aware, reusing `geometry.py` helpers:
- **Backbone RMSD:** Kabsch **N/CA/C/O RMSD** (and per-atom-type N/CA/C/O RMSD), via a `common_anchor_rmsd`-style routine generalized to `[.., L, 4, 3]`.
- **Stereochemistry validity** (`mdtraj`-based, authoritative):
  - bond-length distributions: N–CA (~1.46 Å), CA–C (~1.53 Å), C–N (~1.33 Å), C=O (~1.23 Å);
  - bond-angle distributions (N–CA–C, CA–C–N, C–N–CA);
  - Ramachandran φ/ψ extraction and **fraction in allowed/outlier regions** (compare against true-frame distribution, not a hard cut only);
  - **ω (peptide-bond) trans fraction** (|ω| near 180°);
  - **Cα pseudo-torsion** already exists in `ca_observable_errors` — reuse.
- **Peptide-bond / clash**: C(i)–N(i+1) distance validity, nonlocal heavy-atom clash fraction (needs sidechain-optional; do backbone-only for fairness).
- Return a dict compatible with the suite's reporting style, with `n_valid/n_total`.
- **Fairness:** compute the **same** metrics on the **true future frames** (GT backbone from the h5 / manifest) and on **persistence**, so the table is `model / persistence / oracle / ground-truth` like the existing suite.

**T1.4 — PDB export of the reconstructed backbone.**
- Extend `scripts/export_eval_trajectories_pdb.py` with a `--atoms backbone|ca` flag (default keep CA for backward-compat) that writes **N/CA/C/O** ATOM records from the new `backbone_samples`, using residue/chain ids exactly as today.
- Add a **sanity check**: for Model 1, assert `|backbone[CA] − ca_samples| < 1e-4 Å` (they must be the same atoms) so the new path is provably a superset.

**T1.5 — Tests.**
- `tests/test_backbone_metrics.py`: (i) oracle identity → RMSD 0, all bonds/angles within tolerance; (ii) a real crystal backbone → φ/ψ/ω distributions sane; (iii) UNK-residue handling still yields a valid backbone (guards the documented UNK→GLY trick).
- `tests/test_openfold_backbone_path.py`: `sample_future_ca` returns `backbone_samples` whose CA column equals `ca_samples`.

### 2.4 Commands
```bash
export PYTHONPATH=/liuyutian/protein_jepa_frame
# Model 1: backbone ensemble metrics on 1000dom val
python -m scripts.eval_backbone_suite \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --ensemble-ckpt artifacts/stage_ensemble_1000dom/checkpoint_best.pt \
  --manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
  --n-samples 8 --out results/backbone_suite_model1.json
# Model 2: Cα ensemble -> OpenFold backbone lifter -> metrics
python -m scripts.lift_ca_to_backbone_openfold --help   # new
# PDB export with backbone atoms
python scripts/export_eval_trajectories_pdb.py --atoms backbone \
  --outdir artifacts/pdbs_1000dom_backbone_20261021
```

### 2.5 Acceptance criteria
- Model 1: `backbone_samples[:, :, :, 1] == ca_samples` (bit-exact/lossless). ✓
- Backbone suite emits a table with **model / persistence / oracle**, Cα-RMSD **and** N/CA/C/O-RMSD, plus stereochemistry validity counts.
- Model 2: a documented, OpenFold-based backbone exists and its Cα matches its own `ca_pred`.
- PDBs open in PyMOL with **real backbone geometry** (no NaNs, no origin-collapsed residues).

---

## 3. TASK 2 — BioEmu-style metrics **+** MSM state-prediction accuracy

> Two families, reported in **separate tables** (source doc §5): (A) static/ensemble, (B) time-dynamics.

### 3.1 Family A — BioEmu static/ensemble metrics (`evaluation/states_metrics.py`, new)

**A1. Reference-state coverage** (source doc §2.1).
- Define reference states **per protein** by clustering the **true MD** ensemble (KMeans over (a) Cα-RMSD-to-mean, (b) internal-distance features, or optionally TICA space — make it a flag).
- Coverage: for each reference state, `1[ (fraction of generated samples with d(x, r_s) < ε) ≥ 0.001 ]`, macro-averaged over states then proteins. Report the **threshold-ε curve** and **hits/sample-budget** (source doc explicitly warns a single ε over-credits over-sampling).
- Distance `d` = Cα-RMSD after Kabsch (globular) — for domains, consider per-domain RMSD.

**A2. Metastable free-energy MAE** (source doc §2.2).
- Build a shared low-dim projection: **TICA** on the **true** trajectories (lag τ, `deeptime`), then fit an **HMM** with `K` states (sweep K).
- Project **model samples** into the **same** TICA/HMM, assign states, compute `F_k = -k_B T ln p_k` per state for model vs reference → `MAE_F`.
- Also report **population MAE / JSD / TV** and a **state-wise probability scatter** (source doc fairness list).
- Freeze & record: features, TICA training set, dim, lag, K, smoothing, T.

**A3. (Optional) stability/thermodynamics** — only if we output ΔΔG; otherwise out of scope for this model (source doc §4).

### 3.2 Family B — MSM **state-prediction** metrics (the part BioEmu does *not* do)

**B0. Label trajectories with an MSM.** Fit TICA+HMM on the **true** MD per protein (or a shared model across proteins if comparable). This yields a discrete state label `s(t)` for every true frame. Then:

**B1. Lagged-state prediction accuracy** (source doc §2.3 bullet 1).
- From the **true** state at time `t` (i.e. given real history, our rollout predicts `t+τ`), predict `ŝ(t+τ)` and compare to the true `s(t+τ)`.
- Report at **multiple τ** (e.g. 1/4/16/64 ns): **accuracy**, **macro-F1**, **balanced accuracy** (cap the majority-state inflation — the source doc's explicit warning). Add **log-loss / Brier / calibration** if we emit a distribution over next states.
- Baseline: **persistence in state space** (`ŝ(t+τ) = s(t)`) — the honest opponent, exactly like the existing CA suite.

**B2. Transition-matrix error** (bullet 2).
- Estimate `T(τ)_true` from true trajectories and `T(τ)_pred` from model rollouts. Report **per-row JSD / TV**, **Frobenius norm**, and **rare-transition rows separately**. Use **Chapman–Kolmogorov** consistency as a self-check.

**B3. Dwell-time / MFPT / implied-timescale errors** (bullet 4).
- Implied timescales from eigen-decomposition of `T`; MFPT between chosen state pairs; relative error vs reference.

**B4. State-population MAE/JSD/TV** — overlaps A2; report once, here as a dynamics-agnostic check.

**B5. Path/flux** (bullet 5) — optional, only if we claim pathway prediction (likely out of scope; list as stretch).

### 3.3 Implementation notes
- **Dependency:** install `deeptime` (`pip install deeptime`) OR implement a minimal TICA+HMM in-tree (eigen of time-lagged covariance + a simple `KMeans`/`hmmlearn`). **Recommend `deeptime`** for correctness + CK/MFPT utilities; pin the version in the report. (Fallback: `hmmlearn`.)
- **Feature definition** must be identical for true and predicted (TICA trained on truth, projected on both). Use internal Cα distances (invariant to rotation) — consistent with the existing contact/distance family and avoids Kabsch ambiguity.
- **Grouping for CIs**: reuse the per-protein resampling in `evaluation/metrics.py` (`paired_bootstrap_ci`) so numbers carry the same uncertainty treatment as the rest of the report.
- **Sampling-budget fairness:** give model and baselines the same number of (sample, τ) draws.

### 3.4 Files
- `evaluation/states_metrics.py` — coverage + free-energy + population.
- `evaluation/msm_metrics.py` — TICA/HMM fit + lagged accuracy + T-matrix + timescales + MFPT + CK.
- `scripts/eval_states_and_msm.py` — driver over a manifest (model / persistence / oracle / (bioemu2 if samples cached)).
- `tests/test_states_metrics.py`, `tests/test_msm_metrics.py` — oracle = perfect; synthetic 2-state MSM recovers known T, timescales, MFPT.

### 3.5 Commands
```bash
pip install deeptime           # record version in report
python -m scripts.eval_states_and_msm \
  --manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
  --model1-ensemble-ckpt artifacts/stage_ensemble_1000dom/checkpoint_best.pt \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --tica-lag 1 --hmm-states 5 --taus 4 16 64 \
  --out results/states_msm_model1_1000dom.json
# baseline(s) to compare against (persistence + cached bioemu2 samples)
python -m evaluation.baselines.summarize_reps --help  # reuse cached bioemu2 samples if present
```

### 3.6 Acceptance criteria
- Oracle (labels = truth) → coverage = 1.0, `MAE_F` = 0, lagged accuracy = 1.0, `‖T_pred − T_true‖_F` = 0.
- Persistence in state space gives a **finite, non-trivial** baseline (it should score ~majority-class).
- A synthetic 2-state MSM with known rates recovers the analytic T(τ), implied timescale, and MFPT within tolerance.
- Final report has **two separate tables** (static vs dynamics) with `n` and sampling budget attached.

---

## 4. TASK 3 — Temperature generalization

### 4.1 Ground truth of the setup
- **5 temperatures:** `{320, 348, 379, 413, 450} K`. All present in both train (4000 each) and val (500 each) of `mdcath_manifest_1000dom`.
- Temperature **already reaches the dynamics** in both models (GLE `temperature_proj`/`_temp_feature`, StageA `time_embed`), and the from-none rollout passes `temperature_k`. So the *capability to condition on T exists*; the task is to **define and run the generalization experiment**.

### 4.2 The generalization protocol (proposed, task-by-task)

Define **two orthogonal hold-outs**, mirroring the user's request ("train diverse on protein/time/temperature, test on unseen protein AND unseen temperature"):

**Split family 1 — Unseen temperature, seen-protein-class (`LOTO`).** For each held-out `T* ∈ {320,348,379,413,450}`:
- Train on **all 5 minus T*** (proteins 800 train domains, T ∈ 4 values).
- Evaluate on **val proteins at T*** (unseen T, unseen proteins).
- This is the true "does the model generalize to an unseen temperature" test. (Manifests already exist for `T*=450`: `mdcath_manifest_1000dom_temp/*`; generate the other 4 with `evaluation/build_temp_holdout.py --temperature T*`.)

**Split family 2 — Unseen protein × unseen temperature (double hold-out).**
- Train on `(seen proteins, seen T's)`, test on `(held-out proteins, held-out T*)`. Isolates the interaction term.

**Split family 3 — Interpolation vs extrapolation.**
- Hold out an **interior** T (e.g. 379) vs an **edge** T (320 or 450). Cold/hot-edge extrapolation is the hard case; report separately (the source doc's distance-threshold-curve philosophy → here it's a T-curve).

**Split family 4 — Temperature-conditioning ablation (crucial control).**
- Evaluate the *same* checkpoint on T* with (i) the **correct T** as input, (ii) a **wrong/fixed T** (e.g. 320), (iii) **no T**. If skill doesn't drop when T is wrong, the model isn't actually using temperature → the "generalization" is illusory. This is the single most informative control.

### 4.3 Metrics (temperature-specific)
Report, per T* and per ΔT = |T* − nearest train T|:
- All existing ensemble metrics (RMSF-r, RMSF, W2 family, contact J) **stratified by temperature**.
- **Skill vs persistence** (`1 − metric/max(metric_persist, eps)`), as in `run_eval_fromnone.py`.
- A **temperature-response curve**: e.g. predicted RMSF / variance vs T must rise with T (physics): report the **slope/correlation of predicted-eff vs T** and compare to the true MD slope. This catches "temperature-unaware" models that produce the same flexibility at all T.
- **Bootstrap CIs** per protein (reuse `evaluation/metrics.py`).

### 4.4 What must change in code
- **T3.1 — Generate the remaining LOTO manifests:** run `evaluation/build_temp_holdout.py` for T ∈ {320,348,379,413} on the 1000dom manifests (450 already exists).
- **T3.2 — Make Model 1 temperature-aware at eval.** Today `eval_ensemble_metrics_suite.py` builds `cond = [t_norm, T_norm]` **only if the ensemble head had `cond_dim>0`**; the current `stage_ensemble_1000dom` head may be `cond_dim=0`. Verify the ckpt's `cond_dims`; if 0, the *T-conditioning ablation* (§4.2 family 4) is trivially "no T" and we must **train a `cond_dim≥2` ensemble head** (cheap: head only, frozen stack — `run_ensemble_1000dom.sh` already supports cond). Record which variant is used.
- **T3.3 — from-none temp path:** the existing `stage_cb1b_temp` run **crashed on non-finite grads** (`logs/fromnone_temp.nohup`). Before any temp-eval, fix training stability (lower lr / stricter grad-clip / `error_if_nonfinite=False` guard as a diagnostic, or the `--force-clip`/`tail-init` knobs already in the config) and **re-run `run_fromnone_temp.sh`** to completion, or deliberately scope Task 3 to Model 1 only and say so.
- **T3.4 — Eval driver:** add a `--temperature` / `--stratify-by-temperature` mode to the suite (or a thin wrapper `scripts/eval_temperature_generalization.py`) that iterates the T-grid × splits and emits a **matrix** `{T*: {seen/unseen protein: metrics}}` plus the **T-conditioning ablation** columns.
- **T3.5 — Tests:** `tests/test_temperature_protocol.py` — assert the held-out T is absent from train manifests (`protocol.validate_manifest_split(..., heldout_temperature=...)` already enforces this), assert no protein overlap, assert the wrong-T control produces a different prediction than the correct-T one when `cond_dim>0`.

### 4.5 Commands
```bash
# generate the other LOTO manifests (T=450 exists)
for T in 320 348 379 413; do
  python -m evaluation.build_temp_holdout \
    --manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
    --train-manifest artifacts/mdcath_manifest_1000dom/train.jsonl \
    --temperature $T
done
# (re)train the temp-holdout from-none run after fixing stability
bash run_fromnone_temp.sh
# Model 1 temperature-aware ensemble head (only if ckpt cond_dim==0)
bash run_ensemble_1000dom.sh   # ensure --cond-dims 2 --manifest .../train_not450.jsonl
# generalization matrix + T-conditioning ablation
python -m scripts.eval_temperature_generalization \
  --splits 320 348 379 413 450 --ablation correct wrong frozen \
  --out results/temperature_generalization_1000dom.json
```

### 4.6 Acceptance criteria
- A `5 (T*) × {seen, unseen protein} × {correct T, wrong T, no T}` result matrix exists.
- At least one checkpoint is **actually T-conditioned** (wrong-T control measurably changes output).
- Clear statement of interpolation vs extrapolation behavior, with stratification and CIs.
- Every held-out T is provably absent from its train manifest (unit test + protocol audit).

---

## 5. Suggested execution order & dependencies

```
Phase 0 (both):  Task 1 T1.1/T1.2 (surface OpenFold backbone, Cα-lifter)
                 → unblocks backbone metrics for BOTH models and helps Task 2 (MSM on backbone features is a stretch, keep Cα for now)

Phase 1:         Task 2 A1–A3 (coverage + free-energy: static table)
                 Task 2 B1–B4 (MSM state-prediction: dynamics table)   [needs deeptime]

Phase 2:         Task 3 T3.1 manifests → T3.2/T3.3 model prep → T3.4 matrix → T3.5 tests

Phase 3:         Re-run unified reporting; merge into inbox report as
                 "§Static ensemble (BioEmu-style)" + "§Dynamics (MSM/lagged)" + "§Backbone" + "§Temperature generalization"
                 (four tables, all with n + sampling budget).
```

**Key risks / decisions to confirm with the user:**
1. **Model 2 backbone:** accept the OpenFold Cα-lifter (no retraining) or invest in a trained frame/torsion head? → *Recommend lifter first.*
2. **deeptime vs in-tree MSM:** install `deeptime` (fast, correct) vs implement minimal TICA/HMM (no new dep). → *Recommend deeptime.*
3. **Model 1 T-conditioning:** if the current ensemble head is `cond_dim=0`, are we allowed to train a new `cond_dim≥2` head (head-only, frozen stack)? → *Recommend yes; it's the only way Task 3 is meaningful for Model 1.*
4. **Task 3 scope:** fix & finish the from-none temp run, or scope to Model 1? → *Recommend fix it; the crash is a stability knob, not a design problem.*

---

## 6. File/asset map (quick reference)

| Purpose | Path |
|---|---|
| OpenFold adapter (authoritative) | `protein_jepa_frame/models/stage_c/openfold_atom37.py` |
| Decoder exposing backbone/atom37 | `protein_jepa_frame/models/stage_c/structure_decoder.py` |
| Model-1 producer (CA-only today) | `protein_jepa_frame/models/ensemble_producer.py`, `protein_jepa_frame/training/train_stage_b_hs.py` |
| Model-2 CA-only decoder | `protein_jepa_frame/models/stage_b1b_fromnone/decoder.py` |
| Existing metrics | `evaluation/ensemble_metrics.py`, `evaluation/ensemble_metrics_ext.py`, `evaluation/geometry.py` |
| Ensemble suite (CA-only) | `scripts/eval_ensemble_metrics_suite.py` |
| PDB export (CA-only) | `scripts/export_eval_trajectories_pdb.py` |
| Temp-holdout builder | `evaluation/build_temp_holdout.py` |
| Temp-holdout eval | `evaluation/run_eval_fromnone.py` (`--split temp`) |
| Temp manifests | `artifacts/mdcath_manifest_1000dom_temp/*` |
| Temp-trained from-none (crashed) | `artifacts/fromnone/stage_cb1b_temp/`, `logs/fromnone_temp.nohup`, `run_fromnone_temp.sh` |
| 5 temperatures | 320 / 348 / 379 / 413 / 450 K (`mdcath_manifest_1000dom`) |
