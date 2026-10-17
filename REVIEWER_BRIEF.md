# Reviewer Brief — methods & results under strict evaluation

**Scope of this request.** We are asking a reviewer to *strictly* assess the
**methods** and **results** of this framework, not to rubber-stamp them. Below is
(a) what the code/files are, (b) exactly what to attack, (c) where the evidence
lives, and (d) what a defensible verdict looks like. Do not trust any claim that
is not backed by a re-runnable command in this repo.

---

## 1. What this repository is

A protein-dynamics representation-learning framework: a per-frame SE(3)-invariant
encoder `E` produces a latent `H`; a **future-state producer** turns a context of
`K` frames into a predicted future latent `H_future`; a **decoder** `D` maps
`H_future` back to CA coordinates. The claim under test is narrow and falsifiable:

> Given a fixed-length context of a trajectory, does the model predict the
> **internal conformational displacement** of the protein `τ` ns ahead better
> than assuming nothing moves (persistence)?

The metric is **CA RMSD after anchor alignment** (internal relative displacement),
which removes global rigid-body motion. See `EVALUATION_PLAN.md` §1–§2.

## 2. File map (methods)

| area | path | what it is |
|---|---|---|
| encoder / stages | `protein_jepa_frame/models/stage_a_v2/`, `stage_b1a/`, `stage_b1b/` | Stage-A geometry encoder; B1a supervised structure alignment; B1b GLE (H,S) dynamics |
| data protocol | `protein_jepa_frame/data/` | manifest, windows, collate, masks (shared with training) |
| training entrypoints | `protein_jepa_frame/training/`, `run_*.sh` | how each checkpoint was produced |
| vendored geometry | `protein_jepa_frame/vendored/openfold/` | upstream OpenFold (not ours) |
| **evaluation package** | `evaluation/` | the measuring instrument (see below) |
| protocol / spec | `EVALUATION_PLAN.md` | authoritative evaluation contract + acceptance criteria |
| model/training design | `STAGE_AB_REDESIGN_PLAN.md` | the model spec |
| checkpoints + hashes | `evaluation/MODELS.md` | SHA-locked checkpoint mapping (checkpoints are **not** in-repo) |

### The `evaluation/` package (one number, one implementation)
- `geometry.py` — **THE** single Kabsch / aligned-RMSD / CA extraction. No other
  file may re-implement Kabsch. Convention is locked (see module docstring).
- `data.py` — manifest → `{context, target}` windows per horizon; guarantees
  context/target **disjointness** and anchor = last context frame (no leakage).
- `loaders.py` — checkpoints → frozen `ModelBundle`; version-aware decoder choice.
- `variants.py` — the **only** place variants differ (the 4 producers).
- `metrics.py` — per-protein table → median/mean±SEM, paired bootstrap CI, skill, MSD.
- `run_eval.py` — entrypoint (orchestration only); writes `results.{json,md}`,
  `per_protein.jsonl`, `meta.json` (git sha + checkpoint sha256 + manifest sha).
- `run_gate.py` — likelihood-side NLL ablations (companion, not the RMSD metric).
- `configs/` — smoke, held-out-protein, held-out-temperature.
- `compat/` — vendored legacy decoder so old checkpoints still load (see caveats).

### The four variants (`EVALUATION_PLAN.md` §1.2)
| variant | `H_future` from | dynamics eq? |
|---|---|---|
| `h_only` | `predictor(H_context, τ, T)` (Stage-A encoder) | no |
| `h_ft` | same predictor path, B1a fine-tuned encoder | no |
| `h_plus_s` | GLE iterative rollout `rollout(H_anchor, S₀, τ)` | **yes** (Euler–Maruyama) |
| `persist` | identity `X̂ = X_anchor` | no |

## 3. Results as currently stated (`EVALUATION_RESULTS.md`)
1. `h_plus_s ≈ persist` (Δ≈0, MSD≈0.001 Å²) — the rollout H barely moves.
2. `h_only` / `h_ft` ≈ 5–8 Å — **worse** than persistence.
3. `oracle` (decode the TRUE future H) ≈ 6 Å ≈ decoding the anchor H ⇒ the
   inter-frame change in H is **not geometrically decodable**. This is the crux
   claim: no current variant achieves `skill > 0`.

## 4. What to attack (strict-evaluation checklist)

**Methods**
- M1. Is the anchor-aligned CA RMSD the *right* metric, and is it computed
  correctly? Verify `geometry.py::aligned_rmsd` / `kabsch_align` against
  `scipy.spatial.transform.Rotation.align_vectors` independently.
- M2. **No-leakage** (§2.4): anchor from the last context frame; context/target
  disjoint; Kabsch uses real coords only; decoder "anchor" input is real.
  Do the assertions in `data.py` + `tests/test_data_no_leakage.py` actually cover this?
- M3. The oracle is a **representation ceiling**, but the B1a oracle decoder is
  weak (~5–6 Å). Is the ceiling *too loose* for the skill score to be meaningful?
- M4. **Lineage inconsistency** (`MODELS.md`): `h_only` (stage_a_fix4),
  `h_ft` (fix4_frozen→unfrozen), `h_plus_s` (stage_a_mixed) sit on **three**
  different Stage-A bases. Is the 4-way comparison confounded? Should the
  headline claim be blocked until a shared-base retrain (CONFIG-B)?
- M5. Is the persistence baseline fair (same windows, same alignment, real coords)?
- M6. Sampling/reproducibility: seed, `n_rollout_reps`, deterministic iteration,
  byte-identical `per_protein.jsonl` on re-run — is determinism actually enforced?

**Results**
- R1. Do the numbers reproduce? Run the smoke and full configs yourself; compare
  to `EVALUATION_RESULTS.md`. Is the legacy cross-check (§7.5) convincing?
- R2. Are medians/means/SEM and paired CIs computed on the right unit
  (per-protein, paired vs persistence)? Is "sig?" correctly defined?
- R3. Are the held-out **protein** and **temperature** axes both truly held out
  (0 train overlap)? Is per-temperature breakdown reported?
- R4. Is the conclusion overstated? State exactly which claims the evidence
  supports vs. does not.

## 5. How to reproduce (evidence)
```bash
# geometry correctness (Kabsch vs independent ground truth)
python -m pytest evaluation/tests/ -q                      # 9 tests

# fast end-to-end harness check (~smoke, 2 proteins, 2 horizons)
python -m evaluation.run_eval --config evaluation/configs/smoke.yaml --max-proteins 2

# full held-out-protein axis (needs checkpoints + mdCATH manifest, 1 GPU)
python -m evaluation.run_eval --config evaluation/configs/mdcath_holdout_protein.yaml

# held-out-temperature axis (450 K)
python -m evaluation.build_temp_holdout --manifest artifacts/mdcath_manifest_200dom/val.jsonl \
    --train-manifest artifacts/mdcath_manifest_200dom/train.jsonl --temperature 450
python -m evaluation.run_eval --config evaluation/configs/mdcath_holdout_temp.yaml \
    --manifest-override artifacts/mdcath_manifest_200dom/val_temp450.jsonl
```
Artifacts: `artifacts/eval/<run>/{results.json, results.md, per_protein.jsonl, meta.json}`.

**Reproducibility caveat the reviewer must respect:** checkpoints (`.pt`), mdCATH
trajectories (`.h5`), and `artifacts/` are **gitignored / not in this repo**. A
reviewer without them can validate the *method* (geometry, leakage, aggregation,
metric definitions, determinism) and re-run the smoke path on a synthetic/mock
encoder, but cannot re-derive the exact Å numbers. Canonical paths + SHA256 are
in `evaluation/MODELS.md`.

## 6. Acceptance criteria for the harness itself (`EVALUATION_PLAN.md` §7)
1. `smoke.yaml` runs end-to-end < 5 min on 1 GPU, produces `results.{md,json}`.
2. `geometry.py` tests pass (Kabsch vs independent ground truth < 1e-5 Å).
3. No-leakage audit enforced by `data.py` assertions; a deliberately-leaky test fails.
4. Re-running the same config gives byte-identical `per_protein.jsonl`.
5. `h_only`/`h_ft` must load a Stage-A predictor + OPF and their producer must
   receive `[B,K,L,dH]` (not only `[B,L,dH]`); a shape-contract test fails if the
   history is dropped.

## 7. Known limitations we are NOT hiding
- Representation ceiling issue: ΔH is not decodable to Δstructure ⇒ every variant
  ≈ persistence or worse with the current checkpoints.
- Lineage inconsistency (three Stage-A bases) — see `MODELS.md` §"Lineage inconsistency".
- The oracle decoder is weak (~5–6 Å), so `oracle` is a loose ceiling.
- Held-out-temperature run (T4.3) and the full protein-holdout run (T4.2) are
  launched but their artifacts are not in-repo.
- `compat/` exists because the working tree rewrote `StructureDecoder`, making
  legacy checkpoints otherwise unloadable.

## 8. What a defensible reviewer verdict looks like
- **Methods:** state whether the metric, leakage controls, alignment convention,
  and aggregation are correct and sufficient; flag any confound (esp. M4) that
  invalidates cross-variant claims.
- **Results:** state whether the conclusion ("no variant beats persistence; ΔH not
  decodable") is *supported*, *overstated*, or *understated*, and which numbers
  are reproducible without the private artifacts.
- **Verdict:** accept / revise / reject the **methodological claim**, with the
  single most important blocking issue named explicitly.
