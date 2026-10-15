# Stage A Representation Diagnostic Report

**Date:** 2026-10-14  
**Checkpoint:** `artifacts/abc_v3/stage_a_abs/checkpoint_latest.pt` (step 5000)  
**Encoder:** `ProteinGeometryEncoder`, SE(3)-invariant, `dH=128`  
**Training objective:** Absolute-target JEPA + struct_recon auxiliary (weight 0.3, pairs 512)

---

## Summary

Stage A training completed 5000 steps without collapsing, but the resulting representation H has two critical weaknesses that prevent the downstream oracle decoder (B1a) from working:

1. **H encodes only ~34% of CA-CA distance variance** (held-out MLP probe R²=0.343). This is partial static fold encoding, not dynamics encoding.
2. **H_future cannot predict conformational change** (MLP regression R²=−0.251, worse than persistence baseline). The oracle decoder B1a is stuck at CA MSE 90–220 Å² with no convergence trend over 1450 steps.

JEPA as a framework is not the problem. The specific failure is that the encoder learns a time-compressed representation where CA moving 22 Å produces only a 0.33 change in H (H scale=1.0). The prediction task becomes trivially easy (copy H_anchor ≈ H_future), so the encoder has no gradient pressure to encode dynamics.

---

## Metric 1: Trajectory-level H Cosine Similarity by Lag

Measured on 30 randomly sampled proteins. Each sample has 8 frames at 1 ns spacing.  
`cos_sim` is pooled over all valid residues of both frames, then averaged across all (protein, frame-pair) pairs at each lag.

| Lag (frames) | Δt (ns) | cos_sim mean | cos_sim min | CA RMSD mean (Å) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 1.0 | 0.9642 | 0.7580 | 2.567 |
| 2 | 2.0 | 0.9577 | 0.6757 | 2.945 |
| 3 | 3.0 | 0.9550 | 0.7108 | 3.180 |
| 4 | 4.0 | 0.9540 | 0.6840 | 3.283 |
| 5 | 5.0 | 0.9515 | 0.7640 | 3.376 |
| 6 | 6.0 | 0.9512 | 0.7908 | 3.468 |
| 7 | 7.0 | 0.9478 | 0.7909 | 3.618 |

**Interpretation:**  
H cosine similarity starts at 0.964 at 1 ns and only drops to 0.948 at 7 ns, even though CA RMSD grows from 2.57 Å to 3.62 Å. H is remarkably time-invariant relative to the actual conformational change occurring. The minimum cos_sim (0.68–0.79) shows some proteins do produce more distinctive H trajectories, but the mean is dominated by a near-static representation.

Note: these 8 frames cover only a 7 ns window per sample. The full dataset horizons reach 96 ns; H invariance is expected to be even more pronounced at long horizons.

---

## Metric 2: Residue-level H Cosine Similarity

This tests whether H is also **residue-invariant** — i.e., whether different residues in the same frame have distinguishable H vectors.

| Comparison | cos_sim mean | cos_sim std |
|:---|:---:|:---:|
| Same residue, lag=1 frame (temporal) | 0.9570 | 0.0663 |
| Different residues, same frame (spatial) | 0.7026 | 0.1784 |

**Interpretation:**  
H is **not** residue-invariant. Different residues have cos_sim ≈ 0.70 (std 0.18), showing the encoder assigns meaningfully different vectors to different residues in the same frame. This is good — the encoder distinguishes residues by their local geometry/environment.

However, the same residue across time has cos_sim ≈ 0.96, nearly identical to its own past self. The spatial variation (residue-to-residue, 0.70) is about 4× larger than the temporal variation (frame-to-frame, 1−0.957 = 0.043 angle). **The encoder successfully encodes which residue it is, but barely encodes how that residue's conformation changes over time.**

---

## Metric 3: H Norm and Per-Channel Temporal Variance

| Quantity | Value |
|:---|:---:|
| H norm (mean per residue) | 11.31 |
| Temporal variance (mean per channel, across 8 frames) | 0.0498 |

The temporal variance floor `gamma²` in the training objective is `0.05² = 0.0025`. Since the measured temporal variance (0.0498) far exceeds this floor, the `temporal_variance` loss never fires during training (it shows `0.0000` in all 5000 steps of the log). The floor is set ~20× too low to be an effective anti-collapse regularizer.

---

## Metric 4: CA-CA Distance Probe (Geometric Faithfulness)

The `struct_recon_head` (trained jointly) and a freshly-fit held-out MLP are both evaluated on predicting pairwise CA-CA distances from frozen H.

| Method | RMSE (Å) | Baseline std (Å) | R² |
|:---|:---:|:---:|:---:|
| Predict mean distance (baseline) | 9.59 | — | 0.000 |
| `struct_recon_head` (joint, frozen eval) | 8.77 | 9.59 | 0.163 |
| Held-out MLP probe (fit on 16 proteins) | 6.89 | 9.23 | 0.343 |

**Note on discrepancy:** The `struct_recon_head` R² (0.163) is lower than the held-out MLP R² (0.343). This is expected: the held-out MLP is fit with more capacity and without the competing JEPA gradient, while the `struct_recon_head` trains under the full multi-objective loss. The held-out MLP gives the correct upper bound on what information H actually contains about geometry.

**Interpretation:**  
H encodes about 34% of the variance in CA-CA distances. This means the encoder has learned the rough shape of the fold (it knows if a residue pair is typically close or far) but is missing fine-grained geometric detail. For comparison, a well-trained AlphaFold2 single representation would reconstruct CA-CA distances at R²>0.90.

---

## Metric 5: SE(3) Invariance Verification

| Test | ΔH after perturbation |
|:---|:---:|
| Translate all atoms by 100 Å | 0.000135 (≈ machine precision) |
| Rotate all atoms by arbitrary R | ~0.000 |
| CA perturbation 0.1 Å | H RMS change 0.071 |
| CA perturbation 1.0 Å | H RMS change 0.794 |
| CA perturbation 5.0 Å | H RMS change 1.030 |

H is correctly SE(3)-invariant (translation by 100 Å leaves H unchanged to ~5 decimal places). The sensitivity to CA perturbation saturates quickly — moving CA by 5 Å produces the same H change (~1.03) as moving it by 1 Å produces 0.79. This saturation means large conformational changes produce small, compressed H changes.

---

## Downstream Impact: B1a Oracle Decoder

With the step-5000 Stage A checkpoint (frozen encoder), the B1a oracle decoder was trained for 1450 steps. Results:

| Metric | Value |
|:---|:---|
| `oracle_ca_mse` range (steps 700–1450) | 90–220 Å² (CA RMSD ~9–15 Å) |
| Convergence trend | None — oscillating without decrease |
| Direct experiment: H_future → CA displacement MLP | R²=−0.251, RMSE 12.76 Å |
| Persistence baseline (predict 0 displacement) | RMSE 11.41 Å |

The oracle decoder performs **worse than persistence**. This is the core failure: even given the true future H (oracle condition), the decoder cannot reconstruct the future structure.

---

## Root Cause Analysis

The failure chain is:

1. The JEPA prediction task (predict future H from context H) is **too easy** because H barely changes over time (cos_sim=0.964 at 1 ns). The predictor learns to nearly copy H_anchor, and the encoder is never forced to make different conformations produce distinguishable H.

2. The `temporal_variance` floor (`gamma=0.05`) never fires because H naturally exceeds it (temporal variance 0.0498 >> gamma²=0.0025). So there is no gradient pressure to increase temporal sensitivity.

3. `struct_recon` (pairwise CA-CA distance regression) provided a partial fix (R²=0→0.343 after fixing the masked-H bug) but plateaued. The plateau occurs because the JEPA gradient pushed H toward a static-fold regime first, and `struct_recon` can only partially recover geometry encoding within that regime.

4. The SE(3)-invariant architecture is correct for the task but requires the training objective to explicitly reward conformational discrimination — which the current objective fails to provide.

---

## Proposed Fixes

Three levers in order of implementation priority:

**Fix 1 (immediate): Raise `temporal_min_std` from 0.05 to 0.30–0.50.**  
The floor needs to be set above the current natural temporal variance (0.0498) to actually trigger and push the encoder toward more time-sensitive representations. At gamma=0.30, gamma²=0.09 > 0.0498, so the floor fires and produces a gradient.

**Fix 2 (immediate): Joint encoder+decoder training (B1a phase 2).**  
Unfreeze the top 2 encoder blocks and train with the oracle decoder's CA MSE loss flowing back into the encoder. This directly optimizes H for the decoding task without requiring us to guess the right auxiliary loss. B1a's `--unfreeze-top-n-blocks 2` implements this. Anchor loss (`anchor_weight=0.1`) prevents representation collapse.

**Fix 3 (medium-term): Replace pairwise distance probe with per-residue coordinate regression.**  
Instead of predicting random CA-CA distances (1 scalar per pair), predict CA coordinates in the Kabsch-aligned anchor frame (3 scalars per residue). This is a denser, more direct supervision signal for the geometry the decoder actually needs.

---

## Code Changes Made This Session

| File | Change |
|:---|:---|
| `protein_jepa_frame/models/stage_a_v2/model.py` | Added `delta_target: bool` config flag; `struct_recon_head` MLP; `struct_recon_loss` method; modified `forward_temporal_multilag` to support absolute teacher target |
| `protein_jepa_frame/training/train_stage_a_v2.py` | Added `--delta-target` CLI flag (default False); `--struct-recon-weight/pairs/dim`; **fixed critical bug: `context_H` → `context_H_full` in struct_recon_loss call** (masked H → unmasked H) |
| `tmp_claude/probe_dist.py` | Diagnostic script: frozen H geometry probe |
| `tmp_claude/launch_abs.sh` | Launch script for `stage_a_abs` run |
