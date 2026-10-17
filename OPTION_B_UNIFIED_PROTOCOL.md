# Option B — Unified train/eval window protocol

**Date:** 2026-10-17 · **Status:** Stage A training launched.

## Why (the bug this fixes)

The old checkpoints were trained and evaluated under **two different window
protocols**, which mechanically produced "all variants ≈ persistence":

| | training (old: fix4 / mixed) | evaluation (`data.py`) |
|---|---|---|
| context | multiscale, offsets `[0,32,64,96,97,98,99,100]`, span 100 ns | uniform K=8, offsets `[-7..0]`, span 7 ns |
| horizons | 101 / 150 / 200 ns | 1 … 96 ns |
| τ seen by predictor | ~101–200 ns | 1–96 ns (out of distribution) |

Two independent mismatches → the predictor was fed context/lag geometry it had
never seen, so `H_future ≈ H_anchor` and every variant collapsed to the anchor
(MSD ≈ 0, `h_only`/`h_ft` ≈ 5–8 Å because a *wrong* τ decode is worse than not
moving). The decoder was never the problem.

## The fix (one protocol, both sides, multi-lag)

Uniform context **K=8, stride 1** (plan §3.2), and **one shared (horizon, lag)
grid**:

- base horizons `H ∈ {8,16,24,32,48,64,96}` ns
- lags `l ∈ {0,4,6,7}` (counted back from the future-clip END, as in training;
  `l=0` = full horizon, larger `l` = closer to the anchor)
- physical gap (frames after anchor) = `H − l`
- ⟹ supervised gaps =
  `{1,2,4,8,9,10,12,16,17,18,20,24,25,26,28,32,41,42,44,48,57,58,60,64,89,90,92,96}` ns

This **superset covers the plan grid `[1,2,4,8,16,24,32,48,64,96]`**, and both
`train_stage_a_v2` and `evaluation.run_eval` use the *same* grid — so no horizon
seen at eval is unseen in training, and the predictor is exercised on multiple
lags (the user's requirement: "both sides should see multi-lag").

## What changed in code

- `evaluation/data.py`: `EvalProtocol` gained `temporal_lags` + `emit_multi_lag`;
  windows now emit one target per `(horizon, lag)` with the same
  backward-counted lag convention as training. `EvalWindow.lag` records it.
- `evaluation/run_eval.py` / `evaluation/metrics.py`: `lag` threaded through to
  `per_protein.jsonl`; `temporal_lags` read from the config.
- `evaluation/configs/*.yaml`: `temporal_lags: [0,4,6,7]`, `emit_multi_lag: true`,
  checkpoint paths repointed to `*_unified`.
- `run_stage_a_unified.sh`: uniform K=8, horizons `{8..96}`, lags `{0,4,6,7}`,
  same architecture/losses as `stage_a_mixed`, from scratch.

## Run order (Option B)

1. **Stage A unified** — `bash run_stage_a_unified.sh` (launched 2026-10-17,
   GPU 2, 15k steps). Output `artifacts/abc_v3/stage_a_unified/`.
2. **B1a unified** — supervised structure alignment on the unified encoder.
3. **B1b unified** — GLE dynamics on the unified H + fixed bath (see
   `B1B_DIAGNOSIS_AND_PLAN.md` 目标 2: `coupling_scale` 1e-3→0.1,
   `diffusion_scale` 1e-4→1e-2, normal `c_net` init).
4. **Eval** — `python -m evaluation.run_eval --config evaluation/configs/smoke.yaml`
   first, then the two holdout configs.

Success criterion (unchanged, `EVALUATION_RESULTS.md` §4): `h_only`/`h_ft`
beating persistence (`skill > 0`, Δ vs persist < 0, sig=YES) at short–mid
horizons with **MSD growing with τ**.
