# Plan A — Stage-A encoder + GLE, jointly fine-tuned (frozen B1a decoder)

Status: infrastructure implemented, first full run launched (`run_plan_a_hs_1000dom.sh`).

## 1. What we are doing (and why)

Keep Stage A's encoder, because its H already scores well on the ensemble metrics
(Model 1: RMSF-r 0.673, RMSF 0.837, pca_W2 0.067) and is linearly decodable to
CA displacement.  On top of it, train the GLE kernel to generate the ensemble.
Weights: Stage-A encoder (fine-tuned at low LR) + GLE kernel; decoder frozen.

## 2. Corrected diagnosis of the old `stage_hs` failure

Measured (see `.tmpwork/planA/DIAGNOSIS_EVIDENCE.md`):

| claim | verdict |
|---|---|
| "Stage-A H is not decodable" (伤1) | **WRONG** for the current ckpt: ridge R² **+0.29**, decode ~0.4 Å, cos(H_t,H_0) ~0.49. (Old result was a different checkpoint.) |
| "the bath is numerically dead" | **mis-framed**: coupling_scale/diffusion_scale are INIT scales; H does move. |
| the real bug | the kernel is **unstable at inference**: ||H_rollout−H_0|| = 33 (1 step) → 9220 (4 steps), decoding to CA **worse than the anchor** (3.17 vs 0.96 Å). Ablation: setting `coupling_scale=0` makes the SAME trained weights stable ⇒ the `h·C@S` bath feedback diverges. |

**Root cause of the divergence**: training optimised a **1-step** rollout against a
**whole-window** ensemble target ⇒ the multi-step `C@S` feedback was never in the
loss, so training drove `C` into an unstable regime that only appears at rollout
time.  A *fresh* kernel is stable at the same config; only the *trained* one blows
up.

## 3. Plan-A fixes

1. `--train-encoder --encoder-lr-scale 0.1` — unfreeze Stage A, fine-tune jointly.
2. `--rollout-steps H == --paired-horizons-frames H` — the terminal GLE step lands
   on the scored window; the multi-step bath feedback is now inside the loss.
3. `--rollout-grad-steps H` — truncated BPTT to bound memory over the H-step rollout.
4. Keep Model 1's **frozen B1a decoder** (it is the decoder that is compatible with
   Stage-A H; v4e's MLP decoder is bound to the v4e H dialect — measured: it decodes
   v4e H to 0.12 Å but Stage-A H to 9.27 Å).

## 4. Evidence that Plan A fixes the divergence

Same conditions that made the old kernel explode:

| kernel | n=1 | n=4 | n=8 |
|---|---|---|---|
| OLD `stage_hs` ||H_t−H0|| | 19.1 | 2758 | 6862 |
| PLAN-A (200 steps) | **5.5** | **11.5** | **15.5** |

Validation at step 100/200: `val/ens/total` 1.96/1.95 vs persistence 2.42 (a ~19%
improvement on the ensemble loss), NaN-free.

## 5. How to run

```bash
GPU=4 H=8 OUT=artifacts/plan_a_hs_1000dom bash run_plan_a_hs_1000dom.sh
```

Then evaluate (with the fine-tuned encoder loaded from the checkpoint):

```bash
python -m scripts.paired_hs_vs_model1 \
  --ensemble-ckpt artifacts/stage_ensemble_1000dom/checkpoint_best.pt \
  --kernel-ckpt artifacts/plan_a_hs_1000dom/checkpoint_best.pt \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
  --horizons 1 2 4 8 --n-samples 32 --batches 50 --device cuda \
  --out results/plan_a_vs_model1.json
```

## 6. Open levers (if Plan A still falls short)

- Long-horizon horizons (16/32/64 ns) need the rollout to match, at higher memory
  cost; consider `embed-fixture` chunking (see `--encode-frame-chunk` in cb1b).
- If the encoder fine-tune drifts, try `--encoder-lr-scale 0.03`, or the
  `--freeze-encoder` ablation for a pure-kernel baseline.
- Kabsch absolute-position accuracy remains the shared weakness (§3.6); the GLE now
  at least does not hurt it (rollout decodes near the anchor rather than diverging).
