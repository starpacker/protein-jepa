# RUNBOOK — running the evaluation end to end

## 0. Prerequisites
- Data: `artifacts/mdcath_manifest_200dom/{train,val}.jsonl` (+ underlying `.h5`).
- Checkpoints: see `MODELS.md`.
- One GPU (the encoder/decoders are run in `no_grad`).

## 1. Sanity (minutes)
```bash
python -m pytest evaluation/tests/ -q                       # 9 tests
python -m evaluation.run_eval --config evaluation/configs/smoke.yaml \
    --max-proteins 2 --horizons 1 4 --anchor-stride 30
```
The smoke run must report that `h_only`/`h_ft` loaded a Stage-A predictor and
that their producer receives `[B,K,L,dH]`.  `persist` is the no-change baseline;
the direct variants are horizon-conditioned predictions, while `h_plus_s` is
the autonomous GLE rollout.

## 2. Full held-out-protein run (hours)
```bash
python -m evaluation.run_eval --config evaluation/configs/mdcath_holdout_protein.yaml
# -> artifacts/eval/mdcath_holdout_protein/{results.md,results.json,per_protein.jsonl,meta.json}
```
Cost control (config knobs, all CLI-overridable):
- `anchor_stride` (default 30 in the full configs) — anchors sampled per trajectory.
- `oracle_max_horizon` — the oracle decode is the expensive part; only computed ≤ this.
- `--max-proteins`, `--horizons`, `--variants` for partial runs.

## 3. Held-out-temperature run
```bash
python -m evaluation.build_temp_holdout \
    --manifest artifacts/mdcath_manifest_200dom/val.jsonl \
    --train-manifest artifacts/mdcath_manifest_200dom/train.jsonl --temperature 450
python -m evaluation.run_eval --config evaluation/configs/mdcath_holdout_temp.yaml \
    --manifest-override artifacts/mdcath_manifest_200dom/val_temp450.jsonl
```

## 4. Likelihood gate
```bash
python -m evaluation.run_gate \
    --ckpt artifacts/abc_v3/stage_b1b_mdcath/checkpoint_latest.pt \
    --b1a-ckpt artifacts/abc_v3/stage_b1a_fix4_unfrozen/checkpoint_latest.pt \
    --manifest artifacts/mdcath_manifest_200dom/val.jsonl --n-prot 60
```

## 5. Interpreting results.md
- `Δ vs persist` < 0 **and** `sig? = YES` ⇒ the variant beats persistence.
- `skill > 0` ⇒ beats persistence relative to the oracle ceiling (only shown when oracle < persist).
- `MSD ≈ 0` ⇒ the rollout is not moving (collapse); `h_plus_s` "matches" persistence only because it does nothing.

## 6. Adding a new variant
1. Add a producer in `variants.py` (one branch; reuse `encode_clip`/`decode`).
2. Add it to `variants:` in the config **only if** a `--`-style encoder/decoder source exists in `loaders.build_bundle`.
3. No other file changes. Metrics/report are variant-agnostic.
