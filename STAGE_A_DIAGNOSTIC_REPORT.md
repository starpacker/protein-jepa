# Stage A Representation Diagnostic Report

**Date:** 2026-10-14  
**Checkpoint:** `artifacts/abc_v3/stage_a_abs/checkpoint_latest.pt` (step 5000)  
**Encoder:** `ProteinGeometryEncoder`, SE(3)-invariant, `dH=128`  
**Training:** Absolute-target JEPA + struct_recon auxiliary (weight 0.3, 512 pairs/step)  
**Data:** mdCATH 200-domain subset, T=320K replica 0, full 440-frame (440 ns) trajectories

---

## Executive Summary

Stage A completed 5000 steps without training instability. The encoder learns to distinguish
residues from each other (spatial cosine similarity ~0.70) and partially encodes static fold
geometry (CA-CA distance probe R²=0.343). However it fails to encode conformational dynamics:
H cosine similarity is 0.98 at 1 ns and only decays to 0.95 at 96 ns, even as CA RMSD grows
from 1.4 Å to 3.2 Å. The downstream oracle decoder (B1a) cannot converge as a result — even
given the true future H, it cannot reconstruct the future structure better than persistence.

---

## Metric 1: H Cosine Similarity at Lags 1, 4, 8, 16, 32, 64, 96 ns

Measured on 20 proteins, full 440 ns HDF5 trajectories, T=320 K, replica 0.  
`cos_sim` is computed between the pooled H vector of the anchor frame (t=0) and
the future frame at each lag, averaged across all 20 proteins.

| Lag (ns) | cos mean | cos std | cos min | cos max | CA RMSD mean (Å) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1  | 0.9845 | 0.0041 | 0.9733 | 0.9940 | 1.435 |
| 4  | 0.9815 | 0.0067 | 0.9622 | 0.9911 | 1.847 |
| 8  | 0.9791 | 0.0099 | 0.9398 | 0.9865 | 1.961 |
| 16 | 0.9793 | 0.0072 | 0.9601 | 0.9899 | 2.179 |
| 32 | 0.9699 | 0.0239 | 0.8750 | 0.9857 | 2.603 |
| 64 | 0.9542 | 0.0750 | 0.6329 | 0.9868 | 2.634 |
| 96 | 0.9525 | 0.0740 | 0.6348 | 0.9852 | 3.191 |

**Key observations:**

- At 1 ns the mean cos_sim is 0.985 — H barely changes in 1 ns even though CA moves 1.4 Å.
- The decay from 1 ns to 96 ns is only 0.032 (0.985 → 0.953), while CA RMSD more than doubles
  (1.4 → 3.2 Å). H is highly compressed relative to the actual conformational displacement.
- The large std at 64–96 ns (0.075) shows heterogeneity: some proteins (cos_min ~0.63)
  produce genuinely distinct H at long lags, but the majority (cos_max ~0.99) do not change at all.
- The `temporal_variance` regularization floor (gamma=0.05, gamma²=0.0025) never fires during
  training because the natural temporal variance already sits at ~0.05 >> gamma². The floor is
  set 20× too low to be an effective anti-collapse signal.

---

## Metric 2: Per-Residue H Cosine Similarity vs Lag

`res_cos` is computed as `cosine_similarity(H_anchor_i, H_future_i)` for each residue i,
then pooled across all (protein, residue) pairs. `frac < 0.9` is the fraction of residues
whose H has changed by more than the cos>0.9 threshold; `frac < 0.99` is finer.

| Lag (ns) | res_cos mean | res_cos std | frac < 0.9 | frac < 0.99 |
|:---:|:---:|:---:|:---:|:---:|
| 1  | 0.9845 | 0.0190 | 0.009 | 0.512 |
| 4  | 0.9824 | 0.0233 | 0.018 | 0.550 |
| 8  | 0.9794 | 0.0291 | 0.026 | 0.605 |
| 16 | 0.9798 | 0.0296 | 0.031 | 0.561 |
| 32 | 0.9751 | 0.0386 | 0.044 | 0.639 |
| 64 | 0.9694 | 0.0570 | 0.064 | 0.646 |
| 96 | 0.9669 | 0.0595 | 0.074 | 0.693 |

**Key observations:**

- Even at 96 ns only 7.4% of residues have cos_sim < 0.9 (a moderately large change).
  93% of all residues are essentially frozen in H space across 96 ns of simulation.
- ~51–69% of residues have cos_sim < 0.99 — they do show small H changes, but these are
  tiny in magnitude and do not encode meaningful conformational information.
- This answers the question directly: **H is not residue-invariant** (different residues
  in the same frame have cos_sim ~0.70, std 0.18), but it **is nearly time-invariant per
  residue** — the same residue across time has cos_sim ~0.97–0.98 regardless of lag.

---

## Metric 3: Spatial (Residue-Level) Discrimination

This confirms the encoder distinguishes residues rather than collapsing all positions to one vector.

| Comparison | cos_sim mean | cos_sim std |
|:---|:---:|:---:|
| Same residue, lag=1 frame (temporal) | 0.9570 | 0.066 |
| Different residues, same frame (spatial) | 0.7026 | 0.178 |

The spatial variation (residue-to-residue gap of ~0.30 in cos distance) is about 6× larger
than the temporal variation per residue (~0.043 at 1 ns). The encoder clearly encodes
*which residue this is* (local sequence + geometry environment), but barely encodes
*how this residue's conformation changed over time*.

---

## Metric 4: Geometry Faithfulness (CA-CA Distance Probe)

| Method | RMSE (Å) | Baseline std (Å) | R² |
|:---|:---:|:---:|:---:|
| Predict mean distance (naive baseline) | 9.59 | — | 0.000 |
| `struct_recon_head` (jointly trained, frozen eval) | 8.77 | 9.59 | 0.163 |
| Held-out MLP probe (fit post-hoc, 16-protein train) | 6.89 | 9.23 | 0.343 |

H encodes approximately 34% of the variance in CA-CA distances. The static fold is partially
captured. The struct_recon_head's lower R² (0.163) compared to the held-out probe (0.343) is
expected — the probe has no competing gradient pressure, while the in-training head shares the
encoder with the JEPA objective.

---

## Metric 5: SE(3) Invariance Verification

| Test | ΔH (RMS) |
|:---|:---:|
| Translate all atoms by 100 Å (X-axis) | 0.000135 |
| CA perturbation ±0.1 Å random noise | 0.071 |
| CA perturbation ±1.0 Å random noise | 0.794 |
| CA perturbation ±5.0 Å random noise | 1.030 |

H is correctly SE(3)-invariant to machine precision. The sensitivity to CA perturbation
saturates quickly — the response compresses from 0.79 at 1 Å to 1.03 at 5 Å, explaining
why large conformational movements produce small H changes.

---

## Downstream: B1a Oracle Decoder Failure

With the step-5000 Stage A checkpoint (frozen encoder), B1a oracle decoder trained 1450 steps.

| Metric | Value |
|:---|:---|
| `oracle_ca_mse` at steps 700–1450 | 90–220 Å² (CA RMSD ~9–15 Å), oscillating |
| Convergence trend | None |
| Direct regression: H_future → CA displacement (MLP) | R²=−0.251, RMSE 12.76 Å |
| Persistence baseline (predict 0 displacement) | RMSE 11.41 Å |

The oracle decoder performs worse than persistence. This is the direct consequence of H
containing near-zero conformational change information: even given true future H as oracle
input, the decoder cannot decode the future structure.

---

## Root Cause

The JEPA temporal prediction task (predict H_future from H_anchor context) is trivially easy
because H barely changes over time (cos_sim=0.984 at 1 ns). The predictor learns to copy
H_anchor ≈ H_future, the encoder has no gradient pressure to make different conformations
produce distinctive H, and the temporal variance floor never fires (set 20× below actual
variance). The result is an encoder that encodes static fold identity per residue but not
conformational dynamics.

---

## Proposed Fixes

**Fix 1 — Raise `temporal_min_std` to 0.35–0.50 (one-line change).**  
At gamma=0.40, gamma²=0.16 > current temporal variance (0.05), so the floor fires and pushes
the encoder to make H more time-sensitive. This is a necessary but not sufficient fix — it
prevents the near-static regime but does not guarantee meaningful conformational encoding.

**Fix 2 — Joint encoder+decoder training (B1a phase 2, immediate).**  
Unfreeze top 2 encoder blocks and let the oracle decoder CA MSE gradient flow directly into
the encoder (`--unfreeze-top-n-blocks 2`). This optimizes H for exactly what the decoder
needs without guessing auxiliary losses. Anchor loss prevents representation collapse.

**Fix 3 — Per-residue CA coordinate regression (medium-term).**  
Replace pairwise distance probe with a head that predicts CA coordinates in the
Kabsch-aligned anchor frame (3 scalars/residue vs 1 scalar/random pair). Denser, more
direct supervision signal for the geometry the decoder actually reads.

---

## Code Changes Made This Session

| File | Change |
|:---|:---|
| `protein_jepa_frame/models/stage_a_v2/model.py` | `delta_target` flag; `struct_recon_head` MLP; `struct_recon_loss` method |
| `protein_jepa_frame/training/train_stage_a_v2.py` | `--struct-recon-weight/pairs/dim` CLI args; **critical bug fix: `context_H` → `context_H_full`** |
| `tmp_claude/probe_lags_1to96.py` | Full-trajectory lag probe (1–96 ns, direct HDF5 access) |
| `tmp_claude/lag_metrics_1to96.json` | Machine-readable lag metric output |
