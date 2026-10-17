# `evaluation/` — what each script does and how to run it

This package computes **one number** for every model variant: the **anchor-aligned
CA RMSD** (internal relative displacement) at horizons 1–96 ns, on mdCATH held-out
protein and held-out temperature splits, always compared against a **persistence**
baseline.

Full spec + rationale: `../EVALUATION_PLAN.md`. Checkpoint mapping: `MODELS.md`.

---

## The pipeline (identical for every variant)

```
context X_{1..K}  ->  Encoder E  ->  H_context [B,K,L,dH]
                                      │
                                      ├─ H_anchor = H_context[:,-1]
                                      │
                                      └─> {future-state producer} -> H_future [B,L,dH]
                                                                           │
                                                        shared Decoder D ───┘
                                                                           │
                                                                     CA_pred
                                                                                      │
                                       anchor-aligned CA RMSD  <── X_true  <──────────┘
```

Only the **future-state producer** differs between variants:

| variant | producer | dynamics |
|---|---|---|
| `persist` | X_anchor (identity) | none |
| `h_only` | `predictor(H_context, τ, T)` from Stage-A encoder, then decode | Stage-A predictor + OPF |
| `h_ft` | same predictor path with B1a fine-tuned encoder | Stage-A predictor + OPF |
| `h_plus_s` | decode(GLE rollout of (H,S)) | **iterative equation** (Euler–Maruyama), no predictor |

The direct Stage-A path is not an anchor-only decode:

```text
H_context = E(X_{1:K})                         [B,K,L,dH]
pred_factors = predictor(H_context, tau, T)     [B,L,J,r]
H_future = OPF.compose(OPF.decompose(H_anchor) + pred_factors)
X_hat = D(H_future, H_anchor)
```

`H_future` is therefore the complete `[B,L,dH]` latent consumed by the same
decoder `D`.  `pred_factors` is a delta in OPF factor coordinates, not the
absolute future latent.  The `sequence`/`aatype` information is already used by
the encoder; it is not an additional predictor input in the current checkpoint
contract.  `anchor_decode` (decoding `H_anchor` directly) is an optional
representation diagnostic, not one of the main predictive variants.

---

## Modules

| file | responsibility | depends on |
|---|---|---|
| `geometry.py` | **THE** Kabsch / aligned-RMSD / CA extraction. Unit-tested. | torch |
| `data.py` | manifest → evaluation windows (context + target per horizon); guarantees context/target disjointness. | data.dataset readers |
| `loaders.py` | checkpoint → frozen `ModelBundle`; version-aware decoder choice. | model packages |
| `variants.py` | the 4 future-state producers (the only place variants differ). | geometry, models |
| `metrics.py` | per-protein table → aggregate + paired bootstrap CI + skill. | numpy |
| `report.py` (in `run_eval.py`) | writes `results.json` / `results.md` / `per_protein.jsonl` / `meta.json`. | — |
| `run_eval.py` | **entrypoint**; orchestration only. | all of the above |
| `build_temp_holdout.py` | build held-out-temperature manifests. | — |
| `compat/` | vendored legacy model code so old checkpoints still load. | — |
| `tests/test_geometry.py` | Kabsch correctness (ground-truth + optimality + mask). | pytest |

**Rule:** nothing re-implements Kabsch or CA extraction outside `geometry.py`.

---

## Running

```bash
# fast harness check (2 proteins, 2 horizons)
python -m evaluation.run_eval --config evaluation/configs/smoke.yaml --max-proteins 2

# held-out-protein axis, horizons 1..96 ns, all 5 temperatures
python -m evaluation.run_eval --config evaluation/configs/mdcath_holdout_protein.yaml

# held-out-temperature axis (450 K)
python -m evaluation.build_temp_holdout --manifest artifacts/mdcath_manifest_200dom/val.jsonl \
    --train-manifest artifacts/mdcath_manifest_200dom/train_manifest_200dom/train.jsonl --temperature 450
python -m evaluation.run_eval --config evaluation/configs/mdcath_holdout_temp.yaml \
    --manifest-override artifacts/mdcath_manifest_200dom/val_temp450.jsonl   # (see run_eval --help)

# geometry tests
python -m pytest evaluation/tests/test_geometry.py -q
```

Useful overrides: `--max-proteins`, `--horizons 1 2 4`, `--variants h_plus_s persist`,
`--anchor-stride N` (sub-sample anchors), `--name`, `--outdir`.

Outputs land in `artifacts/eval/<name>/`:
`results.json` (machine), `results.md` (tables), `per_protein.jsonl` (raw rows), `meta.json` (git + checkpoint hashes).

---

## Reading the tables

- **RMSD median / mean±SEM** — the metric (Å, lower better).
- **persist** — the baseline at the same (split, horizon).
- **oracle** — decode the TRUE future H with the same decoder (representation ceiling).
- **Δ vs persist** — paired per-protein difference; **sig?** = 95% bootstrap CI excludes 0.
- **skill** — `(persist - model)/(persist - oracle)`; 1 = oracle, 0 = persist, <0 = worse.
  Reported only when the oracle beats persistence (`persist - oracle > 0`).
- **MSD** — mean squared CA displacement of the prediction from the anchor;
  ≈0 ⇒ the model is not moving (rollout collapse / pure persistence).

---

## Known checkpoint caveats (READ `MODELS.md`)

1. The B1a oracle decoder (v1) reconstructs a frame at **~5–6 Å**, not the 0.6 Å
   of the MLP probe (`H_DECODABILITY_REPORT.md` §9). So `oracle` here is a *weak*
   ceiling; `h_only`/`h_ft` inherit that weakness.
2. The H lineage is **not one chain**: `h_ft`/`h_plus_s` descend from
   `stage_a_fix4_frozen`, while the current B1b weights were trained on
   `stage_a_mixed`. See `MODELS.md` §"Lineage inconsistency". Cross-variant
   claims must respect this.
3. Legacy checkpoints need `compat/` (the working tree rewrote `StructureDecoder`).
