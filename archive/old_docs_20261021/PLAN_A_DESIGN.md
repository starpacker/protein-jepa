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

---

## 7. Dense-trajectory supervision (2026-10-21, follow-up)

**Question:** the rollout produces the whole path `[anchor, H_1, ..., H_T]` (T+1
frames); are we using all T+1 of them, or only the last?

**Answer (measured):** only the last.  `_rollout_decode` keeps `traj[:, -1]` and
the loss decodes that single terminal H against a whole-window summary target.
At H=8 that is **1 of 9 produced frames supervised**; at H=16, **1 of 17**.

### What we now do (`--dense-trajectory`)

Score **every** rollout frame against a *contiguous* future window
`[anchor, t1, ..., tH]` (so rollout step i ↔ truth frame i), reusing the proven
from-none loss family so behaviour is identical to the recipe that already
produced v4e:

| term | weight flag | role |
|---|---|---|
| per-step Kabsch CA point loss | `--dense-ca-weight` | beats persistence; clean per-frame encoder gradient |
| energy score on the CA path | `--dense-es-weight` | distributional / spread (proper scoring rule) |
| ensemble RMSF | `--dense-rmsf-weight` | match flexibility **amplitude** along the path |
| ensemble RMSF-r | `--dense-rmsf-corr-weight` | match RMSF **profile shape** (the eval metric) |
| ensemble variance | `--dense-var-weight` | log-domain variance match |

Why it should help: RMSF / RMSF-r / W2 are *ensemble* metrics — they need the
cloud of rollouts to spread like the true walk, not just land near one point.
Dense supervision gives that signal at every step instead of one endpoint.

### Data alignment (required, and the reason for the contiguous loader)

The paired loader returns a FLAT future "window" (H frames at stride H), so step i
has *no* matching true frame.  Dense mode therefore reuses the from-none
contiguous protocol (`collate_clips` + `split_contiguous_window`, K+rollout_steps
frames) so the correspondence is exact.  Trajectories are 40–450 frames, so H up
to 64 is fully available (`--rollout-steps 64`).

### Train-time safety

- **Terminal-only** run: `run_plan_a_hs_1000dom.sh` (GPU 4) — Plan A, single terminal step.
- **Dense** run: `run_plan_a_dense_1000dom.sh` (GPU 2) — all steps.
Both share the same Stage-A init / frozen B1a decoder / GLE kernel; the only
difference is how much of the rollout path the loss sees.

### Rollback

Everything is additive and flag-gated.  To revert to the previous behaviour:
drop `--dense-trajectory` (terminal-only Plan A) or drop `--train-encoder`
(frozen-encoder join).  No existing code path changes when the flags are absent;
the `H_traj` key added to the producer's return dict is ignored by the old path.

## 8. Corrected paired-eval result (supersedes §7 numbers)

Two bugs in the first paired-eval pass made the reported numbers invalid:

1. **Global RNG was never seeded.** `GaussianHHead.sample` and the GLE kernel
   draw noise from the *global* torch generator, but the script only seeded a
   local dataset-subset Generator. Hence the frozen Model 1 produced a different
   ensemble in every process. Fixed: `torch.manual_seed` / `np.random.seed` added.
2. **Model 1 shared HS's encoder object.** `build()` built `m1` and `hs` on the
   *same* encoder instance; HS then reloaded the *fine-tuned* encoder over it,
   silently swapping Model 1's encoder too. So each run's "Model 1" was actually
   that run's fine-tuned encoder, which is what moved the baseline (1.0376 vs
   1.4248). Fixed: `m1` gets its own `deepcopy(encoder)`.

After both fixes the Model1 baseline is **identical in both runs** (rmsf_mae
0.8699), i.e. a genuine apples-to-apples paired test. Corrected conclusion:

- **Terminal-only Plan A** is essentially *on par* with Model 1: significant only
  on `rmsf_corr` (+0.0316 [+0.0020,+0.0629]) and `transient_contacts_j`
  (+0.0087 [+0.0017,+0.0157]); `rmsf_mae`, `root_mean_w2`, `variance_w2`,
  `md_pca_w2` are statistically indistinguishable.
- **Dense Plan A is WORSE than Model 1**: `rmsf_mae` +0.1230 [+0.0391,+0.2125],
  `variance_w2` +1.0356 [+0.6925,+1.4398], `md_pca_w2` +0.0043 [+0.0012,+0.0072],
  `transient_contacts_j` -0.0129 [-0.0213,-0.0049]; `rmsf_corr` flat.

So the earlier "beats Model 1 on all 7" was an artefact of the encoder aliasing.
The dense-supervision idea does **not** win on the ensemble metrics; terminal-only
is the safer variant, and neither yet clears Model 1 outright.

### Protocol caveat (must be fixed before any "final" claim)

Training used `step_ps=1000` while the MD data is 1 frame / 1000 ps, so the dense
correspondence "rollout step i <-> truth frame i" is temporally exact *only* for
the training/contiguous loader. The eval uses the **paired** loader with
`--horizons 8` (a scalar `num_frames=4` context + a stride-8 future window), which
is a *different* protocol: the future window is not the same contiguous window the
dense run trained on. Before declaring a winner, re-run both train and eval on one
shared contiguous protocol (`split_contiguous_window`, `step_ps` matched) so the
paired comparison is exact.
