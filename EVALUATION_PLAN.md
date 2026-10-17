# Evaluation Framework — Plan & Spec

**Date:** 2026-10-17
**Scope:** A single, clean `evaluation/` package that computes the *one* number we care about
(anchor-aligned CA RMSD = internal relative displacement) for every model variant, on a
fixed protocol, with multi-horizon and held-out-protein / held-out-temperature splits on mdCATH.

---

## 1. The one question we are answering

Given a fixed-length context of a protein trajectory, we ask:

> Does the model predict the **internal conformational displacement** of the protein
> `τ` nanoseconds into the future, better than assuming nothing moves?

Everything is measured as **CA RMSD after anchor alignment** (§2.5), compared against the
**persistence baseline** (§3.4). A model is useful only if it beats persistence.

### 1.1 The canonical pipeline (identical for every variant)

```
   context frames  X_{1..K}            (K = 8, uniform stride)
          │
          ▼
   ┌─────────────┐
   │  Encoder f  │   Stage A V2 geometry encoder (SE(3)-invariant), frozen after B1a
   └─────────────┘
          │
          ▼
    H_context = H_{1:K} ∈ R^{B×K×L×128}
    H_anchor  = H_K ∈ R^{B×L×128}
          │
          ▼
   ┌─────────────────────────────────────────────┐
   │  Future-state producer  (this is the ONLY    │
   │  thing that differs between variants)        │
   └─────────────────────────────────────────────┘
          │
          ▼
    H_future ∈ R^{B×L×128}   (one H per target residue set)
          │
          ▼
   ┌─────────────┐
   │  Decoder D  │   maps H_future → CA coordinates in the anchor frame
   └─────────────┘
          │
          ▼
    X̂_CA ∈ R^{L×3}  (anchor-local)
          │
          ▼
   metric: anchor-aligned CA RMSD vs  X_true_CA   (§2.5)
```

**Key contracts (must hold for every variant, enforced in code):**
- The encoder is applied *identically* (same checkpoint, same normalization, same call),
  frame by frame, and the complete `[B,K,L,128]` history is retained.
- The same decoder interface `D(H_future, H_anchor)` is used. `h_only` and
  `h_ft` share the B1a oracle decoder; `h_plus_s` uses the B1b residual decoder
  required by its checkpoint lineage. This lineage difference is recorded as a
  comparison caveat, not hidden as if all decoder weights were identical.
- Direct prediction consumes the complete `H_context`, not only `H_anchor`.
- The anchor frame is derived from the **last context frame**, never from the target
  (no leakage). See §2.4.
- The horizon `τ` enters the **future-state producer**, not the decoder.

### 1.2 Variants table (what changes → what to expect)

| Variant | name in code | H_future comes from | Uses dynamics eq? | Uses B1b bath/S? | Purpose |
|---|---|---|---|---|---|
| **H-direct**      | `h_only`     | `predictor(H_context, τ, T)` → `H_future` | no  | no  | Stage-A encoder + temporal predictor |
| **H-finetuned-direct** | `h_ft`  | same predictor path with fine-tuned encoder | no | no | effect of B1a encoder calibration in the full prediction path |
| **H+S**           | `h_plus_s`   | **GLE iterative rollout** `rollout(H_anchor, S₀, τ)` | **yes** (Euler–Maruyama, `sample_step`) | yes | full Stage-B1b co-evolution: (H,S) → (H,S) |
| **Persistence**   | `persist`    | identity `X̂ = X_anchor`                 | no  | no  | baseline: assume the structure is unchanged |

> **Direct prediction vs persistence.** `h_only` and `h_ft` retain the complete
> `H_context = [B,K,L,dH]` and call the temporal predictor with the requested `τ`.
> `persist` is a separate no-change baseline.  An anchor-only learned decode is
> not a predictive variant; if needed, report it separately as `anchor_decode`.

For the current Stage-A checkpoints the predictor emits a factor-coordinate delta,
not an absolute latent:

```text
pred_factors = predictor(H_context, τ, T)       [B,L,J,r]
H_future = OPF.compose(OPF.decompose(H_anchor) + pred_factors)  [B,L,dH]
```

The shared decoder consumes this resulting `H_future`.  `sequence` is not an
additional predictor input in this implementation; amino-acid identity and residue
metadata enter the encoder.

> **"H+S uses the iterative equation, not a predictor."** The user's requirement: for the
> H+S variant there is no separate direct predictor `g(H_ctx,τ)`. Instead we integrate the
> learned GLE kernel step-by-step (`StageB1b.rollout` → `sample_step`), from the anchor H and
> a bath state `S₀` sampled from the history posterior. This is the **autonomous rollout** mode.

---

## 2. Metric definition (exact)

### 2.1 CA extraction
Work on the 4-atom backbone tensor `coords [T,L,4,3]` (N, CA, C, O). CA = index **1**.
Residue validity = `residue_mask` (all backbone atoms present).

### 2.2 Ground-truth internal displacement
For a pair (context window, target frame at horizon τ):
- `X_anchor = CA(context_last_frame)` → `[L,3]`
- `X_true   = CA(target_frame)`        → `[L,3]`

### 2.3 Anchor alignment (the "对齐基准")
We align **both** `X_anchor` and the prediction to `X_true` using one common Kabsch rotation
computed from `X_anchor → X_true` over valid residues. Concretely:

1. Compute `R*, t*` = Kabsch(`X_anchor`, `X_true`) — the optimal rigid-body superposition of
   the anchor onto the target (**computed once, from real coordinates only**).
2. Apply `(R*, t*)` to every prediction: `X̂_aligned = X̂ · R* + t*`.
3. RMSD = `sqrt(mean_i || X̂_aligned,i − X_true_centered,i ||²)`.

This removes the global rigid-body motion (which is *not* what H is responsible for) and
measures only the internal conformational error. **Persistence under this metric = the
RMSD of the (aligned) anchor itself against the target** — a pure "structure didn't move"
prediction.

> The single shared Kabsch implementation lives in `evaluation/geometry.py`
> (`kabsch_align`, `aligned_rmsd`). Every variant calls it. **No script may re-implement Kabsch.**
> Convention (locked, matches the corrected B1b pipeline): `R = U·diag(1,1,d)·Vᵀ`, apply `Pc @ R`.

### 2.4 No-leakage rules (audited in §5)
- Anchor frame / anchor H come from the **last context frame only**.
- Context and target windows are **disjoint** in raw frames (`paired_nonoverlap_only=True`).
- Kabsch `R*,t*` uses **only real** coordinates (both anchor and target are real; no model output).
- The decoder's "anchor" input is the real anchor CA, never the predicted one.
- Held-out proteins appear in **no** training manifest for the variant being scored.

### 2.5 Reported quantities
For each (variant, split, horizon):
- **CA RMSD** (primary): median over proteins (robust to outliers) **and** mean±SEM.
- **ΔCA RMSD vs persistence** (per-protein paired): negative = better than persistence.
- **Skill score**: `skill = (persist − model) / (persist − oracle)`,
  where `oracle` = decode the **true future** H (for h_only/h_ft) or the **true future**
  structure ceiling. Skill=1 → matches oracle; skill=0 → equals persistence; skill<0 → worse.
- **MSD(lag)**: mean squared CA displacement of the model's prediction from the anchor,
  to prove the model is actually *moving* (rollout collapse detector). MSD≈0 ⇒ pure persistence.
- Optional: **contact-map Jaccard** and **per-residue RMSF correlation** as secondary
  sanity signals (report only, not gating).

---

## 3. Evaluation protocol

### 3.1 Horizons
mdCATH frame interval = **1000 ps ⇒ 1 frame = 1 ns**.
Multi-horizon grid (ns = frames): **`[1, 2, 4, 8, 16, 24, 32, 48, 64, 96]`**.
- Requires ≥ 96 + K context frames of trajectory; mdCATH has 440 frames ⇒ fine.
- Rollout cost scales with max horizon (96 Euler steps); run short horizons on all proteins,
  long horizons on a capped subset if needed (recorded in the result header).
- `frame_stride=1` for horizons ≤ 32; allow `frame_stride>1` option but default 1 (documented).

### 3.2 Context
Fixed `K = 8` context frames, uniform stride ending at the anchor frame (`paired_history_mode="uniform"`).

### 3.3 Splits — two axes, both required
On mdCATH we evaluate **both**:
1. **Held-out protein**: 20 val proteins, disjoint from all train proteins (verified: 0 overlap).
2. **Held-out temperature**: mdCATH val spans T ∈ {320, 348, 379, 413, 450} K. We report
   per-temperature (a "train at 320 → test at 450" question is handled by the manifest used,
   not by changing the script). The script takes `--temperatures` to subset, and always
   prints the per-temperature breakdown so a leave-one-temperature-out story can be read off.

Splits are realized by pointing `--manifest` at the desired jsonl
(`.../val.jsonl` for protein-holdout; a filtered jsonl for temperature-holdout, §5 task T2).

### 3.4 Persistence baseline
`persist_rmsd(anchor, target)` via the same §2.3 alignment. Computed from **real** coordinates
only; needs no model. This is the denominator of every skill score.

### 3.5 Oracle / ceiling
- For `h_only`/`h_ft`: `oracle` = decode the **true** future H (`H_seq[τ]` from encoding the
  real target frame) with the same decoder. This is the decoder ceiling for that representation.
- For `h_plus_s`: oracle is the same representation ceiling (the representation is frozen; the
  rollout only gets to choose H_future). Reporting the same oracle makes the skill scores comparable.
- Also report the **structure ceiling** ("what if we fed the true target structure"): trivially 0 Å;
  included only as the skill denominator anchor.

### 3.6 Sampling / reproducibility
- Fixed seed for rollout sampling (`--seed 0`), `--n-rollout-reps 4` independent rollouts per
  window; report both mean-over-reps and a per-rep spread.
- Deterministic dataset iteration order (no shuffling).
- Every run writes `meta.json`: git SHA, checkpoint SHA256, manifest path + SHA, seed, horizons,
  split, device, timestamp.

---

## 4. Deliverables — `evaluation/` package

```
evaluation/
  README.md               # what each script is, how to run, example commands (the "logic is clear" doc)
  geometry.py             # THE single Kabsch/alignment + CA extract + mask utils. No duplicates anywhere.
  data.py                 # WINDOW PROTOCOL: build context/target windows for all horizons from a manifest;
                          #   exposes an iterable of {protein, T, anchor_frame_idx, horizon, X_anchor, X_true, mask}
  loaders.py              # checkpoint loaders: encoder (stage A), B1a decoder, B1b model + residual decoder;
                          #   returns a uniform `ModelBundle` with .encode() and
                          #   .predict_future(H_context,τ,T) methods
  variants.py             # the 4 future-state producers: h_only, h_ft, h_plus_s, persist.
                          #   single source of truth for "what differs between variants"
  metrics.py              # rmsd/skill/msd/contact-jaccard aggregation, bootstrap CI, per-protein table
  report.py               # writes results.json + results.md (tables) from a MetricsResult
  run_eval.py             # THE entrypoint.  `python -m evaluation.run_eval --config evaluation/configs/<x>.yaml`
                          #   or CLI flags. Loops variants × splits × horizons, writes reports.
  configs/
    mdcath_holdout_protein.yaml     # 20 val proteins, all 5 temps, horizons 1..96
    mdcath_holdout_temp.yaml        # train-temp subset vs. one held-out temperature
    smoke.yaml                      # 2 proteins, horizons [1,4], for CI
  legacy/                 # (optional) symlink/notes to the old tmp_claude/eval_* for provenance
```

### 4.1 Responsibility of each script (the "写清楚干什么")
- **geometry.py** — pure functions, unit-tested. Kabsch, centered RMSD, CA indexing,
  anchor-frame construction, contact maps. *Nothing else imports scipy Kabsch.*
- **data.py** — turns a manifest into evaluation windows. Owns the *disjointness* guarantee and
  the horizon→frame-index mapping. No model knowledge.
- **loaders.py** — knows checkpoint file formats (stage A `{'model', 'stage_a_v2_config'}`,
  B1a `{'oracle','encoder',...}`, B1b `{'model','decoder','stage_b1b_config',...}`).
  Loads the Stage-A predictor and OPF in addition to the encoder. Returns frozen
  modules. No metric knowledge.
- **variants.py** — implements exactly the 4 rows of §1.2. Direct variants retain
  `H_context [B,K,L,dH]`, call the predictor, construct `H_future [B,L,dH]`, and
  pass it to the shared decoder.
- **metrics.py** — given arrays of (X̂, X_true, X_anchor, mask) → per-protein table + aggregate.
- **run_eval.py** — orchestration only; prints a per-horizon table and writes artifacts.

### 4.2 Output layout
```
artifacts/eval/<run_name>/
  meta.json
  results.json              # nested: split → temperature → horizon → variant → {rmsd_median, rmsd_mean, sem, delta, skill, msd, n}
  results.md                # human table, one block per (split, temperature)
  per_protein.jsonl         # one row per (protein, T, horizon, variant)
```

---

## 5. Task list (step-by-step execution)

### Phase 0 — groundwork
- **T0.1** `mkdir evaluation/`, write `EVALUATION_PLAN.md` (this file). ✅
- **T0.2** Confirm the exact checkpoint set for each variant (see §6). Verify each loads and
  the expected sub-keys are present. Write `evaluation/MODELS.md` recording the canonical paths
  + SHA256 + what each represents.

### Phase 1 — the shared library
- **T1.1** `geometry.py`: `kabsch_align`, `aligned_rmsd`, `center`, `ca_from_backbone`,
  `contact_map`, `anchor_frame_from_last`. Unit tests in `evaluation/tests/test_geometry.py`
  (compare against `scipy.spatial.transform.Rotation.align_vectors` on random clouds; target <1e-5 Å).
- **T1.2** `data.py`: `EvalWindow` dataclass + `iter_windows(manifest, horizons, K, context_stride,
  max_proteins, temperatures, temperature_holdout, seed)`. Reuse `protein_jepa_frame.data.dataset`
  / `collate` so the protocol matches training exactly (no re-implemented windowing).
- **T1.3** `loaders.py`: `load_bundle(stage_a_ckpt, b1a_ckpt=None, b1b_ckpt=None,
  stage_a_predictor_ckpt=None, device=...) -> ModelBundle`. The bundle exposes
  `encode(coords, mask, aat, ridx, cid) -> H_context` and
  `predict_future(H_context, tau, temperature, valid_mask) -> H_future`.
- **T1.4** `metrics.py` + `report.py`: aggregation, bootstrap CI (paired vs persistence), MD/JSON writers.

### Phase 2 — variants & runner
- **T2.1** `variants.py`: 4 producers. For `h_only`/`h_ft`, call the Stage-A
  predictor on the full `H_context`; for `h_plus_s`, wrap `StageB1b.rollout`
  with a fixed `n_steps=τ`, seed, `n_reps`; `persist` is the identity baseline.
- **T2.2** `run_eval.py` + `configs/*.yaml`. Cast the variants always to the **same** decoder
  (§6): default = B1a oracle decoder (if the variant's encoder is the B1a one) and the
  ResidualCADecoder for B1b-based variants — documented and switchable via `--decoder`.

### Phase 3 — build the held-out-temperature manifest
- **T3.1** `evaluation/build_temp_holdout.py` (or reuse `scripts/merge_manifests.py`):
  produce `artifacts/mdcath_manifest_200dom/val_temp<T>.jsonl` = all val proteins at a single T,
  plus a matching `train_notemp<T>.jsonl` if a train-without-that-T run is desired.
- **T3.2** Document in `evaluation/README.md` how the two held-out axes map to manifests.

### Phase 4 — run, verify, report
- **T4.1** Smoke: `python -m evaluation.run_eval --config evaluation/configs/smoke.yaml`
  → expect `h_plus_s == persist` currently (per `B1B_B2_FULL_PIPELINE_RESULTS.md`), a *known*
  reproducible starting point, proving the harness is correct end-to-end.
- **T4.2** Full protein-holdout run, horizons 1..96, all 5 temps, all 4 variants.
- **T4.3** Temperature-holdout run (per T).
- **T4.4** Write `EVALUATION_RESULTS.md` (top-level) summarizing the four variants × horizons,
  with the honest verdict (currently: representation ceiling issue, ΔH not decodable →
  variants ≈ persistence; the *point* of this harness is to make that claim rigorous and
  re-runnable after the next H retrain).

### Phase 5 — cleanup
- **T5.1** Move the surviving logic from `tmp_claude/eval_b1b_pipeline.py`, `eval_b1b_gate.py`
  into the new package (the gate/NLL eval becomes `evaluation/run_gate.py`, reusing `geometry.py`).
- **T5.2** Delete/mark-deprecated the ad-hoc duplicates; keep a short `evaluation/legacy/NOTES.md`
  mapping old script → new module so nothing is silently lost.
- **T5.3** Remove root-level scratch scripts superseded by the package; update `.gitignore` for
  the new `artifacts/eval/` outputs.

---

## 6. Model / checkpoint mapping (canonical, to be SHA-locked in `evaluation/MODELS.md`)

| Variant | encoder | B1a decoder/H-norm | B1b (H+S) | notes |
|---|---|---|---|---|
| `h_only`   | `artifacts/abc_v3/stage_a_fix4/checkpoint_latest.pt` (step 14000) | Stage-A predictor + OPF from `stage_a_predictor_ckpt`; B1a oracle decoder | none | direct predictor on full `H_context` |
| `h_ft`     | `artifacts/abc_v3/stage_b1a_fix4_unfrozen/checkpoint_latest.pt` key `encoder` | Stage-A predictor + OPF from explicit `stage_a_predictor_ckpt`; B1a oracle decoder | none | direct predictor on full `H_context` with fine-tuned encoder |
| `h_plus_s` | same encoder as `h_ft` (frozen after B1a) | history initializes `S₀`; transition starts at `H_anchor` | `artifacts/abc_v3/stage_b1b_mdcath/checkpoint_latest.pt` (or `_mixed`) | GLE rollout; decoder = ResidualCADecoder from B1b ckpt |
| `persist`  | none | none | none | identity |

> **Open item T0.2:** decide whether the headline `h_plus_s` uses the mdCATH-only B1b
> (`stage_b1b_mdcath`) or the mixed one; and whether `h_only`/`h_ft` share one decoder
> checkpoint or each has its own. This is recorded as an explicit config field, not a hard-code.

**Decoder consistency rule:** `h_only` and `h_ft` must use the same B1a decoder
weight-set. The current `h_plus_s` checkpoint supplies a different residual
decoder and is therefore not a clean decoder-controlled comparison; the runner
logs this explicitly in `meta.json`.
The runner asserts this and logs it in `meta.json`.

---

## 7. Acceptance criteria for the harness itself (not the science)
1. `smoke.yaml` runs end-to-end in < 5 min on 1 GPU and produces `results.md` + `results.json`.
2. `geometry.py` unit tests pass (Kabsch vs scipy < 1e-5 Å).
3. No-leakage audit (§2.4) is enforced by assertions in `data.py`; a deliberately-leaky test fails.
4. Re-running the same config twice gives byte-identical `per_protein.jsonl` (determinism).
5. `h_only` and `h_ft` must show in the run metadata that they loaded a Stage-A
   predictor and OPF, and their producer must receive `[B,K,L,dH]`, not only
   `[B,L,dH]`.  A shape-contract test must fail if the history is dropped.

---

## 8. What this plan is NOT doing
- It is **not** retraining H (that is the separate "路 A" in `B1B_DIAGNOSIS_AND_PLAN.md`).
- It does not retrain model weights, but it does contain checkpoint-compatible
  evaluation adapters for the Stage-A predictor and OPF.
- It is **not** keeping the old scripts as-is — they get folded into `evaluation/` and deprecated.

---

## 9. Immediate next action
Start at **Phase 0 → T0.2** (lock checkpoints into `evaluation/MODELS.md`), then **T1.1**
(`geometry.py`, the single source of truth for Kabsch). Everything downstream depends on those two.

---

## 10. Execution status (2026-10-17)

| Task | Status | Notes |
|---|---|---|
| T0.1 plan + `evaluation/` | ✅ | this file; `evaluation/` created |
| T0.2 lock checkpoints → `MODELS.md` | ✅ | SHA256 recorded; **found a lineage inconsistency** (B1a descends from `stage_a_fix4_frozen`, B1b from `stage_a_mixed`) |
| T1.1 `geometry.py` + tests | ✅ | 7/7 tests pass (ground-truth, optimality, mask, CA, contacts) |
| T1.2 `data.py` | ✅ | reuses training readers; disjointness asserted; verified on real manifest |
| T1.3 `loaders.py` | ✅ | version-aware decoder plus Stage-A predictor/OPF loading via `compat/`; auto-infers encoder flags |
| T1.4 `metrics.py` + report | ✅ | median/mean/SEM, paired bootstrap CI, skill, MSD |
| T2.1 `variants.py` | ✅ | direct h_only/h_ft use full history predictor; H+S = GLE rollout |
| T2.2 `run_eval.py` + configs | ✅ | anchor-major loop; direct predictor is called once per horizon |
| T3.1 temp-holdout manifest | ✅ | `build_temp_holdout.py`; wrote `val_temp450.jsonl` / `train_notemp450.jsonl` |
| T3.2 README | ✅ | `evaluation/README.md` |
| T4.1 smoke | ✅ | validates direct `H_context → predictor → H_future → decoder`, H+S rollout, persistence, and output reports; numerical results are not reused from the old anchor-only harness |
| T4.2 full protein-holdout run | 🔄 | launched 2026-10-17 (`artifacts/eval/mdcath_holdout_protein`) |
| T4.3 temperature-holdout run | ⏳ | manifests ready; run after T4.2 |
| T4.4 top-level results doc | ✅ | `EVALUATION_RESULTS.md` |
| T5.1 fold gate eval | ✅ | `run_gate.py` (functional; NLL ablations + CI) |
| T5.2 legacy notes | ✅ | `evaluation/legacy/NOTES.md` |
| T5.3 cleanup | ✅ | plan/README written; `artifacts/eval/` gitignored |

### Key discoveries during execution
1. **The working tree rewrote `StructureDecoder`**, making every B1a oracle checkpoint
   unloadable with current code. Fixed with `evaluation/compat/` (vendored v1), selected
   automatically by probing the state dict.
2. **Stage-A checkpoint configs lie** (`stage_a_fix4` stores `struct_recon_dim=384` but its
   weights are 256-wide) and the encoder feature width changed across the lineage. Loader now
   infers encoder flags (`orientation_features`) from weight shapes.
3. **The B1a oracle decoder is a weak structure decoder** (~5–6 Å), not the 0.6 Å of the MLP
   probe. So `oracle` is a *loose* ceiling; this is documented, not hidden.
4. **The harness reproduces the prior ad-hoc scripts' numbers** (cross-check §7.5 passes),
   confirming the clean re-implementation is faithful.
