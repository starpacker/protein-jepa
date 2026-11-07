# ROOT CAUSE — v3 CA-trajectory local-geometry collapse

**Date:** 2026-11-07 · **Model:** v3 SOTA (`artifacts/lasp_v3_velocity`, `HSEnsembleProducer`)
**Input PDB:** `results/todo_1107/e31_pdb_v3_regenerated/1ka1A01_1ka1A01_T320_R1_h64_s0.pdb`

---

## 1. Reproduced defect

Adjacent Cα–Cα bond distances across the 65-frame trajectory (96 residues):

| frame | bond mean | `<3.0 Å` | in [3.6,4.0] | min |
|---|---:|---:|---:|---:|
| t=0 (anchor) | 3.694 | 2/95 | 81/95 | 1.306 |
| t=1 | 2.371 | **74/95** | 8/95 | 0.472 |
| t=32 | 2.481 | 71/95 | 8/95 | 0.343 |
| t=64 | 2.655 | 70/95 | 8/95 | 0.739 |

Same as the audit document: RMSD/Rg stay sane, local geometry collapses at the **first** step.

---

## 2. Decisive isolation test (P0-A / GT-1): encode→decode with **zero GLE**

Measured on 3 val proteins (1nhwC00, 1ka1A01, 1kafD00), `decode(encode(anchor))` with **no dynamics**:

| stage | bond mean | `<3.0Å` | in [3.6,4.0] |
|---|---:|---:|---:|
| **true anchor** | 3.835 | 0.0% | 99.3% |
| **encode→decode (NO GLE)** | 3.609 | 4.7% | 66.9% |
| GLE step 1 | 2.647 | **67.6%** | 7.2% |
| GLE step 64 | 2.907 | 59.4% | 8.2% |

**The encoder/decoder round-trip is healthy.** Pure reconstruction preserves bonds
(mean 3.61 Å, only 4.7% short). The collapse is **introduced by a single GLE step**.

(The anchor's resid 2–3 = 1.306 Å is a pre-existing data artifact specific to 1ka1A01;
it is the exception, not the mechanism. 1nhwC00/1kafD00 anchors are 100% valid.)

---

## 3. Why one GLE step destroys geometry — quantified

Per-step H update (velocity coupling, `sample_step`):

```
dH = h·D·p,   p = f - C·S  (v3: S inert, coupling 1e-3, so p ≈ f)
H_{t+1} = H + dH + sqrt(2·h·D)·ε
```

Measured magnitudes (1ka1A01 @320 K, deployed `h=1000 ps`):

| quantity | value |
|---|---:|
| ‖H₀‖ (latent norm) | **11.31** |
| ‖f‖ = ‖−∇U(H)‖ mean | **272.6** (unclipped: `force_clip=None`) |
| D (mobility) mean | 4.98e-4 |
| drift ‖h·D·f‖ | 17.0 (**clamped** at `max_step_norm=20`) |
| noise ‖sqrt(2hD)·ε‖ | 11.3 |
| **total H displacement / ‖H‖** | **2.5×** |

The drift `h·D·f` is intrinsically ~135 (272×5e-4×1000) and is **hard-clamped at 20**,
which is already ~1.8× the latent norm; adding noise pushes the single-step
displacement to **2.5× the entire H vector**. The decoder was trained to decode
the encoder's output manifold, not arbitrary H displacements of this magnitude,
so it emits a collapsed structure.

**Root cause (single sentence):** the learned H-space force/potential `U(H)` is
unconstrained in magnitude, so one Euler step of the GLE moves H far outside the
manifold the frozen decoder can invert — the geometry collapse is a
**dynamics ↔ decoder mismatch**, not an encoder/decoder information loss.

---

## 4. dt-scan confirmation (E4)

Same anchor, same total physical time (1 ns), varying `step_ps`:

| step_ps | step-1 `<3Å` | step-1 mean |
|---|---:|---:|
| 1000 | 0.768 | 2.43 |
| 200 | 0.442 | 3.00 |
| 50 | 0.095 | 3.51 |
| 10 | **0.032** | 3.65 |

Small dt ⇒ the *first* step is geometrically valid; the collapse scales with
per-step displacement, confirming the drift magnitude is the driver. (Even at
small dt, 1 ns of accumulated drift still degrades to ~45% short bonds, because
the drift `f` is a learned function with no geometry-respecting restoring force.)

---

## 5. Root-cause chain (code paths)

1. `protein_jepa_frame/models/stage_b1b/model.py:449` `compute_forces` — `f=-∇U(H)`,
   **no bound** because v3 config has `force_clip=None` (line 463 `if self.cfg.force_clip is not None`).
2. `model.py:711-720` `sample_step` velocity branch — `dH = h·D·p`, clamp only at
   `max_step_norm=20` (a magnitude cap on the *step*, not the *force*).
3. `model.py:1168` `rollout` — iterates `sample_step` `n_steps` times.
4. `protein_jepa_frame/models/stage_c/structure_decoder.py:727` anchor-free decode —
   `x_int = coil_prior + coord_scale·tanh(head(s))`; the decoder is a bounded
   correction on a coil prior, so it cannot absorb a 2.5×-norm latent displacement.
5. `scripts/export_v3_autoregressive_pdb.py:66` — single H-space rollout, decode per frame.

**Confidence: high** — every link verified by direct tensor measurement, and the
dt-scan is a clean causal confirmation (smaller displacement ⇒ valid geometry).

---

## 6. Fix plan (minimal, in v3 code, per spec §7.5)

The failure is the dynamics pushing H off the decoder manifold. Two complementary,
verifiable fixes — NOT a PDB post-processor:

1. **Bound the force at the source** (`force_clip`): set a physically-scaled
   `force_clip` so `‖f‖` cannot be 24× the latent norm. This alone stops the
   2.5×-norm single-step displacement. Verified hypothesis: `force_clip≈10–20`
   keeps per-step displacement ≲ O(1)×‖H‖.

2. **Geometry-aware multi-step training objective** (spec §7.5.3): add a
   differentiable bond/angle/clash penalty on the *decoded* rollout CA
   (`L_geom(decoder.decode_structure(H_t))` for t = 1,2,4,8,…), with confirmed
   gradient flow back to the kernel (energy/bath parameters). This teaches the
   dynamics to stay inside the decodable manifold rather than merely capping it.

The encoder/decoder are **not** the problem and should not be re-designed (GT-1
passes). The fix belongs in the GLE integrator/objective.

---

## 7. Reproduction commands

```bash
export PYTHONPATH=/tmp/lasp_stability_deps:/liuyutian/protein_jepa_frame
# GT-decode isolation + bond stats
python /tmp/diag_gt_decode.py --n-windows 3
# step-magnitude attribution
python /tmp/diag_step_magnitude.py
# dt-scan (E4 convergence)
python /tmp/diag_dt_scan.py
```

---

## 8. ADDITIONAL FINDING (2026-11-07, GT-1 variance) — decoder over-averages conformational change

Re-examining `decode(encode(GT))` across MULTIPLE true frames (not just the anchor)
reveals a second, distinct defect: the decoder collapses conformational VARIANCE.

Per-residue CA variance over the true future window vs the encode→decode window
(variance = Σᵢ tr Cov_t(rᵢ(t))):

| protein | var(true) | var(encdec) | ratio |
|---|---:|---:|---:|
| 1nhwC00 | 429.2 | 3.3 | 0.01 |
| 1ka1A01 | 41.9 | 0.7 | 0.02 |
| 1kafD00 | 48.0 | 0.4 | 0.01 |

The decoder reconstructs the anchor (frame-0 RMSD 0.2–0.6 Å) but maps DIFFERENT
true conformations to nearly the same output (1–2% of true variance). This is
the spec §A3's "decoder / posterior-mean collapse": `D(E(X_GT[t]))` is over-averaged.

So the v3 system has TWO separable defects:
1. **Decoder variance collapse** — `D(E(X_GT))` flattens motion (this section).
2. **GLE one-step geometry collapse** — unbounded force moves H off-manifold (§3).

Both must be fixed. Fix (1) is a decoder/representation training issue (the
encoder's H may not retain enough per-frame displacement signal, or the decoder's
bounded coil-prior correction saturates). Fix (2) is the force/step-magnitude issue.

Artifacts: results/todo_1107/gt_decode_pdb/{1nhwC00,1ka1A01,1kafD00}_{true,encdec,gle}.pdb
(true frames Kabsch-aligned to anchor, so motion is centered for inspection).
