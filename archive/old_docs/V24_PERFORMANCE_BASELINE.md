# V24 Performance Baseline

**Date:** 2026-09-23
**Config:** `configs/lspwm_v44_stage_b_v24_mori_stable.yaml`
**Best Checkpoint:** `checkpoint_step5400.pt` (before rollout explosion at step 5500)
**Training:** 6000 steps, 4-phase curriculum (representation → thermo_id → dynamics → rollout)

---

## Training Summary

| Phase | Steps | Key Observations |
|-------|-------|------------------|
| representation | 0-1000 | paired_skill converged to ~0.83, force_score_Q ~0.04 |
| thermodynamic_identification | 1000-2500 | force_score_G dropped from 212 → 0.22, mori_NLL stuck at 500 (clamped) |
| dynamics | 2500-5000 | loss ~51.0 stable, bath_drift norm ~1.5, dissipative ~5.8e-5 |
| rollout | 5000-6000 | explosion at step 5500 (rollout NLL → 2.9×10¹²). Best ckpt: 5400 |

---

## M1: Thermodynamic Closure — 2×2 {force}×{memory} Ablation

| Mode | R² (Skill) | ± Std | MSE | n |
|:-----|:----------:|:-----:|:---:|:---:|
| **M_Full** (force + memory) | **0.1869** | 0.1528 | 7.40e-6 | 1800 |
| M_Force (conservative only) | 0.0574 | 0.0755 | 8.36e-6 | 1800 |
| M_Memory (bath only) | 0.0959 | 0.1169 | 8.14e-6 | 1800 |
| M_None (persistence) | 0.0000 | — | 8.76e-6 | 1800 |

### Pairwise Ablation Deltas (bootstrap 95% CI):

| Comparison | Δ (R²) | 95% CI | Significant? | Interpretation |
|:-----------|---:|:------:|:---:|:---|
| M_Full − M_None | **+0.1869** | [+0.1800, +0.1939] | ✅ | Full model > persistence |
| M_Full − M_Memory | +0.0910 | [+0.0864, +0.0957] | ✅ | Adding force helps |
| M_Full − M_Force | +0.1295 | [+0.1235, +0.1357] | ✅ | Adding memory helps |
| M_Force − M_None | +0.0574 | [+0.0541, +0.0610] | ✅ | Force alone has signal |
| M_Memory − M_None | +0.0959 | [+0.0908, +0.1012] | ✅ | Memory alone has signal |

**Key finding:** M_Full = 0.1869 > M_Force (0.0574) + M_Memory (0.0959) = 0.1533, showing force+memory provide complementary information in joint 2D projection (+0.0334 synergy).

---

## K1: Folding Kinetics — Per-Horizon Prediction Skill

Metric: `skill = 1 - MSE(paired_delta_pred, delta_target) / MSE(0, delta_target)`

**Mean Skill Across Horizons: 0.7838**

| Horizon (frames) | Skill | ± Std | Δ MSE | n |
|---:|---:|---:|---:|---:|
| 24 | 0.7889 | 0.1634 | 0.0213 | 600 |
| 48 | 0.7841 | 0.1622 | 0.0229 | 600 |
| 96 | 0.7785 | 0.1695 | 0.0242 | 600 |

---

## R1: Representation Closure

- Gate pass: **0/0** (representation is Markov-sufficient — closed ✅)
- Ridge probe test at all horizons and history lengths shows no benefit from history

---

## Key Model Diagnostics (checkpoint_step5400)

| Diagnostic | Value | Interpretation |
|:---|---:|:---|
| `cos(force, bath)` | 0.8574 | 🔴 Bath drift ≈ linear transform of force (degenerate ablation) |
| `cos(force, dG)` | 0.2053 | Force direction modestly aligned with ΔG |
| `cos(bath, dG)` | 0.2868 | Bath direction modestly aligned with ΔG |
| `|force_G|` | 1.411 | Conservative force magnitude |
| `|bath_drift|` | 1.498 | Bath drift magnitude (→ ~200× observed dG) |
| `|dissipative_drift|` | 0.000058 | Dissipative drift (→ ~0.008× observed dG) |
| `|dG|` | 0.00715 | Typical state change between context and future |
| `force_score_G_loss` | 0.362 | KDE gradient loss (high, noisy) |
| `force_score_Q_loss` | 0.040 | Sliced score loss (stable) |
| `self_consistency_loss` | 2.255 | Energy decomposition not accurate |
| `residue_force_cosine` | 0.040 | Per-residue force near-orthogonal to Δq |
| `mori_nll` | 500.0 (clamped) | FDT transition NLL at ceiling |
| `cv_vamp2_loss` | -2.98 | VAMP-2 score (good signal) |

---

## Identified Issues → V25 Fixes

| # | Issue | Root Cause | V25 Fix |
|---|-------|-----------|---------|
| 1 | cos(force,bath)=0.857 | aux_ss = -C·force/γ makes bath ∝ force | History-predicted aux (`history_to_bath` MLP) |
| 2 | dissipative ~5e-5 vs dG ~0.007 | diffusion_scale=5e-5 too small | Learnable per-dim diffusion correction |
| 3 | force_score_G=0.362 (noisy) | bandwidth=1.0 too broad | Reduce to 0.5 |
| 4 | self_consistency=2.255 | No residual corrector | Increase weight 0.05→0.10 |
| 5 | mori_nll=500 | Caused by #1+#2 scale mismatch | Re-enable at weight=0.001 |
| 6 | residue_force_cosine=0.04 | Direction loss too weak | Increase weight 0.3→1.0 |
