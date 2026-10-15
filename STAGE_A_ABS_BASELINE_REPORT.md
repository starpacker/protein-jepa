# Stage A Baseline Report — `stage_a_abs`

**Date:** 2026-10-14  
**Checkpoint:** `artifacts/abc_v3/stage_a_abs/checkpoint_latest.pt`  
**Training steps:** 5000  
**Encoder:** `ProteinGeometryEncoder`, SE(3)-invariant, `dH=128`  
**Objective:** Absolute-target JEPA + struct_recon auxiliary (weight 0.3, 512 pairs/step)  
**Data:** mdCATH 200-domain subset, T=320 K, replica 0, 440-frame (440 ns) trajectories

---

## Training Loss Curve (final steps)

| Step | total | future | masked | orth | struct_recon |
|---:|---:|---:|---:|---:|---:|
| 4850 | 0.0287 | 0.0160 | 0.0254 | 0.0012 | 84.50 |
| 4900 | 0.0289 | 0.0162 | 0.0255 | 0.0012 | 84.45 |
| 4950 | 0.0290 | 0.0162 | 0.0255 | 0.0012 | 87.88 |
| 5000 | 0.0285 | 0.0159 | 0.0252 | 0.0012 | 84.58 |

The JEPA losses (`future`, `masked`) converged cleanly. `struct_recon` (CA-CA distance head) plateaued at ~84 Å² with no further reduction, indicating the head saturated.  `temporal_variance` was 0.000 throughout — the anti-collapse floor never fired.

---

## Gate 1 — H Cosine Similarity at Lags 1–96 ns

Measured on **20 proteins**, full 440-frame HDF5 trajectories, T=320 K, replica 0.  
`cos_sim` is the trajectory-level cosine between the pooled H of the anchor frame (t=0) and the future frame, averaged over proteins.  CA RMSD is Kabsch-aligned.

| Lag (ns) | cos mean | cos std | cos min | cos max | CA RMSD (Å) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1  | 0.9845 | 0.0041 | 0.9733 | 0.9940 | 1.435 |
| 4  | 0.9815 | 0.0067 | 0.9622 | 0.9911 | 1.847 |
| 8  | 0.9791 | 0.0099 | 0.9398 | 0.9865 | 1.961 |
| 16 | 0.9793 | 0.0072 | 0.9601 | 0.9899 | 2.179 |
| 32 | 0.9699 | 0.0239 | 0.8750 | 0.9857 | 2.603 |
| 64 | 0.9542 | 0.0750 | 0.6329 | 0.9868 | 2.634 |
| 96 | 0.9525 | 0.0740 | 0.6348 | 0.9852 | 3.191 |

**Gate threshold (design doc):** cos_sim @ 96 ns < 0.95 to indicate meaningful temporal diversity.  
**Result: FAIL** — mean cos_sim @ 96 ns = 0.953, just above threshold. The decay from 1 ns to 96 ns is only 0.032 even as CA RMSD more than doubles (1.4 → 3.2 Å).

The large std at 64–96 ns (0.075) reflects heterogeneity: a minority of proteins (cos_min ~0.63) show genuinely distinct H at long lags, but the majority do not change at all (cos_max ~0.99).

---

## Gate 2 — Per-Residue H Cosine Similarity vs Lag

`res_cos` is `cosine_similarity(H_anchor_i, H_future_i)` per residue, pooled over all (protein, residue) pairs.

| Lag (ns) | res_cos mean | res_cos std | frac < 0.9 | frac < 0.99 |
|:---:|:---:|:---:|:---:|:---:|
| 1  | 0.9845 | 0.0190 | 0.009 | 0.512 |
| 4  | 0.9824 | 0.0233 | 0.018 | 0.550 |
| 8  | 0.9794 | 0.0291 | 0.026 | 0.605 |
| 16 | 0.9798 | 0.0296 | 0.031 | 0.561 |
| 32 | 0.9751 | 0.0386 | 0.044 | 0.639 |
| 64 | 0.9694 | 0.0570 | 0.064 | 0.646 |
| 96 | 0.9669 | 0.0595 | 0.074 | 0.693 |

At 96 ns, only **7.4%** of residues have cos_sim < 0.9. The other 92.6% are effectively frozen in H-space despite up to 3+ Å of spatial motion.

**Is H residue-invariant?** No. Different residues in the same frame have cos_sim ~0.70 (std 0.18) — the encoder clearly encodes local sequence and geometry environment per residue. But the same residue across frames has cos_sim ~0.97–0.98 regardless of lag. H encodes *identity* faithfully but *dynamics* almost not at all.

---

## Gate 3 — Geometry Faithfulness (CA-CA Distance Probe)

A held-out MLP probe is fit on 70% of proteins to predict pairwise CA-CA distances from pairs of H vectors, then evaluated on the remaining 30%.

| Method | RMSE (Å) | R² |
|:---|:---:|:---:|
| Naive baseline (predict mean distance) | 9.59 | 0.000 |
| `struct_recon_head` (in-training head, frozen eval) | 8.77 | 0.163 |
| Held-out MLP probe (fit post-hoc) | 6.89 | 0.343 |

H encodes ~34% of the variance in CA-CA distances. The static fold is partially captured.

**Gate threshold:** R² > 0.30 for the held-out probe.  
**Result: PASS** — R² = 0.343.

---

## Gate 4 — ΔH Decodability (Does ΔH Carry Conformational-Change Information?)

Ridge regression of `ΔH = H_future − H_anchor` onto the per-residue CA displacement in the Kabsch-aligned anchor frame. Trained on 70% of 40 proteins, evaluated on 30%. This is the definitive test of whether ΔH encodes dynamics.

| Lag (ns) | N residues | Ridge R² | Basin R² | Displacement RMS (Å) |
|:---:|:---:|:---:|:---:|:---:|
| 1  | 6 722 | −0.031 | −0.000 | 1.558 |
| 4  | 6 722 | −0.025 | −0.002 | 2.016 |
| 16 | 6 722 | −0.010 | −0.003 | 2.487 |
| 64 | 6 722 | −0.034 | −0.004 | 3.089 |
| 96 | 6 722 | −0.036 | −0.000 | 3.392 |

**Gate threshold:** R² > 0.0 (ΔH must beat predicting the mean displacement).  
**Result: FAIL** — R² is negative at every lag, worse than predicting zero displacement. ΔH carries no recoverable information about the direction of conformational change.

*Note: these results were measured on the `stage_a_fix2` checkpoint, which uses the same encoder architecture and JEPA objective as `stage_a_abs`. The fix2 run showed the same pattern.*

---

## Gate 5 — SE(3) Invariance

| Test | ΔH (RMS) |
|:---|:---:|
| Translate all atoms 100 Å along X | 0.000135 |
| CA perturbation ±0.1 Å random noise | 0.071 |
| CA perturbation ±1.0 Å random noise | 0.794 |
| CA perturbation ±5.0 Å random noise | 1.030 |

**Result: PASS** — H is SE(3)-invariant to machine precision.  

Note: the saturation of H response (0.79 at 1 Å noise vs 1.03 at 5 Å noise) is consistent with the time-invariance problem: the encoder's sensitivity to spatial change compresses quickly, so multi-Å conformational movements produce small ΔH.

---

## Gate 6 — Internal Geometry Signal Availability

A critical sanity check: does the SE(3)-invariant input actually carry information about conformational change? If the distance matrix barely changes, no encoder can recover dynamics.

| Lag (ns) | CA RMSD (Å) | Dist-matrix RMSD (Å) | Internal/Spatial ratio |
|:---:|:---:|:---:|:---:|
| 1  | 1.546 | 1.093 | 0.707 |
| 4  | 1.970 | 1.379 | 0.700 |
| 16 | 2.392 | 1.654 | 0.692 |
| 64 | 2.865 | 2.109 | 0.736 |
| 96 | 3.176 | 2.438 | 0.767 |

The pairwise CA-CA distance matrix changes by ~70% of the CA RMSD. **The conformational-change signal is present in the encoder's input.** The problem is not architectural blindness — the encoder discards this information because the training objective never rewards keeping it in ΔH.

---

## Gate 7 — Downstream Oracle Decoder (B1a)

With the step-5000 Stage A checkpoint as a frozen encoder, B1a oracle decoder was trained for 1450 steps.

| Metric | Value |
|:---|:---|
| `oracle_ca_mse` range (steps 700–1450) | 90–220 Å² (CA RMSD ~9–15 Å), oscillating |
| Convergence trend | None |
| Direct regression H_future → CA displacement | R² = −0.251, RMSE 12.76 Å |
| Persistence baseline (predict zero displacement) | RMSE 11.41 Å |

**Result: FAIL** — Even given the true future H as oracle input, the decoder cannot reconstruct the future structure. The oracle performs worse than persistence, which is the direct consequence of Gate 4 failing.

---

## Summary: Gate Scorecard

| Gate | Description | Threshold | Result | Pass? |
|:---|:---|:---|:---|:---:|
| G1 | cos_sim @ 96 ns | < 0.95 | 0.953 | ❌ |
| G2 | Residues with cos_sim < 0.9 @ 96 ns | > 10% | 7.4% | ❌ |
| G3 | CA-CA distance probe R² | > 0.30 | 0.343 | ✅ |
| G4 | ΔH → CA displacement R² | > 0.0 | −0.031 to −0.036 | ❌ |
| G5 | SE(3) invariance | < 0.001 ΔRMS on translation | 0.000135 | ✅ |
| G6 | Internal geometry signal present | dist-mat RMSD / CA RMSD > 0.5 | 0.70–0.77 | ✅ |
| G7 | Oracle decoder CA MSE | < persistence RMSE | Worse than persistence | ❌ |

**Overall: 3 / 7 gates pass. Stage A does not meet the acceptance criteria to proceed to B1a.**

---

## Root Cause

The JEPA temporal prediction task is trivially satisfied without encoding dynamics. Because H barely changes over time (cos_sim = 0.984 at 1 ns), the predictor learns to copy `H_anchor ≈ H_future` and gets near-zero `L_future`. The encoder has no gradient pressure to make different conformations produce distinctive ΔH. The anti-collapse variance floor (`temporal_min_std=0.05`) never fires because the natural temporal variance (~0.05) already exceeds `gamma²=0.0025` — the floor is set 20× too low.

The result: H encodes *which residue this is* (local sequence and geometry environment) but not *how this residue's conformation changed*. ΔH is noise, not signal.

---

## What Was Tried and Why It Didn't Work

| Attempt | What changed | Outcome |
|:---|:---|:---|
| Raise `temporal_min_std` 0.05→0.40 (fix1) | Variance floor now fires (L_temporal=0.25) | cos_sim @ 96 ns *increased* to 0.960 — encoder satisfied floor with noise in orthogonal directions |
| Supervised temporal contrastive loss (fix2) | Regressed `‖ΔH‖` toward `tanh(CA RMSD / 2Å)` | cos_sim @ 96 ns *increased* to 0.984 — encoder learned correct magnitude but noise direction |
| B1a oracle with unfrozen top-2 encoder blocks | CA MSE gradient flows to encoder | oracle_ca_mse still oscillating 108–200 Å², no downward trend (encoder had no structural change info to fine-tune) |

All three interventions made cos_sim worse, not better. The root problem is that cos_sim and the loss targets are not in conflict: an encoder can satisfy magnitude-based constraints while keeping ΔH direction uncorrelated with structural change.

---

## Proposed Next Steps

The decisive fix is **differential structure reconstruction**: force the encoder to predict the *change* in pairwise CA-CA distances, `ΔD_ij = D_ij(t+Δ) − D_ij(t)`, from `[ΔH_i, ΔH_j]`. This directly supervises the direction of ΔH with respect to the distance-matrix change that is demonstrably present in the input (Gate 6).

Secondary option: lower EMA momentum from 0.998 to 0.99 during early training, so the teacher target itself varies more, forcing the predictor to actually extract changing information from ΔH rather than copying a near-constant target.
