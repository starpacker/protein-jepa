# Stage A/B Investigation Plan v2

Date: 2026-10-12
Based on: Phase 1 probes (iter_007), corrected Phase-2 gate audit

---

## 1. What the Phase-1/2 audit established

### 1.1 Confirmed findings (protein-disjoint, canonical metrics)

| finding | evidence |
|---|---|
| Frozen H (SE(3)-invariant, distances only) cannot represent frame-dependent vectors | probe_controls_v3, probe_orient_diagnosis; vector valR²=−0.08 |
| H CAN represent frame-invariant scalars (displacement magnitude R²=+0.49) | probe_controls_v3 C1 |
| Context velocity predicts future local displacement: cos=+0.46, R²=+0.23 out-of-sample | probe_decisive_vector (leak-free, 2000/300 train/val split) |
| Q is globally collapsed: corr(Rg, Q_mean)=0.797, per-residue corr=0.024 | iter_006 diagnosis |
| Contact-J gate is biphasic: intermediate accuracy HURTS contacts | probe_gate_threshold_v2 (canonical Kabsch, residue_index, chain_id, protein-clustered bootstrap) |
| Basin is current ceiling: skCJ=+0.056 [+0.043,+0.070] 95% CI | probe_gate_threshold_v2 |
| BA+VEL destroys contacts (skCJ=−1.17) despite R²=+0.23 on displacement | probe_gate_threshold_v2 |
| Hard contact-J gate is correctly implemented, NOT a metric bug | corrected probe matches prior qualitative finding |

### 1.2 Key revised understanding

The contact gate is NOT unreachable because "no signal exists". The directional
signal IS present in raw kinematics (cos=0.46 out-of-sample). It is unreachable
because:
(a) the current encoder cannot carry the signal into Q (invariant architecture),
(b) the contact metric requires near-oracle per-residue accuracy, and
(c) the velocity signal, while real, lives on high-amplitude residues whose
    directional error destroys the binary contact map.

This means: a better Q can legitimately improve many metrics EXCEPT the hard
contact-J gate, which requires near-oracle directional accuracy that 28ns context
physically cannot supply.

---

## 2. What a multi-component dynamical Q should carry

Q must NOT be reduced to any single component. The four components below are
independently predictable and independently testable:

### Component 1 — Thermodynamic/kinetic scalars (T-invariant, global)
- Basin centre position (already partially in Q, weakly — Rg correlation 0.79)
- Radius of gyration change (ΔRg, invariant scalar)
- Per-residue RMSF (time-averaged flexibility, frame-invariant)
- Global Rg, end-to-end distance, asphericity
- Basin occupancy / dwell time
Target: Gate C (collective observables) — R² on these scalars from Q should
significantly exceed shuffled-Q on held-out val proteins.

### Component 2 — Collective modes (frame-invariant projections)
- Projection onto top-3 PCA eigenvectors of the context displacement field
  (effective rank ≈3 confirmed by probe_max_legal)
- Inter-domain distance changes (if multi-domain)
- Global hinge/shear mode amplitude
These are predictable (they are invariant amplitudes), trainable from context,
and constitute what "global conformation" means beyond Rg.
Target: Gate C, Gate E (rollout predictability of mode amplitudes).

### Component 3 — Equivariant kinematic information (per-residue, frame-dependent)
- Context velocity (confirmed: cos=0.46 to future direction, R²=0.23)
- Second-order kinematic signal (acceleration)
- Per-residue local frame change between context frames
These REQUIRE an equivariant encoder (orientation_features=True or a full
equivariant backbone) to be accessible from H. Currently blocked by invariant
architecture.
Target: Gate B (future-specific per-residue information) — after equivariant
encoder is added, Q_context must outperform Q_shuffled on per-residue
displacement magnitude and direction (not just Jaccard).

### Component 4 — Uncertainty / distributional state
- Variance of context trajectory in Q-space (aleatoric uncertainty)
- Bath coupling strength (temperature-dependent fluctuation scale)
- Multi-modality indicator: does the context span multiple basins?
These feed the diffusion net and bath dynamics; they determine whether Q
predicts a narrow basin or a broad ensemble.
Target: Gate D (thermodynamic integrity), Gate A (no collapse).

---

## 3. What NOT to do (explicit prohibitions from the instruction set)

- Do NOT turn Q into a displacement-only vector (destroys thermodynamic/kinetic info)
- Do NOT train a future-Q autoencoder as the sole objective (oracle Q improvement
  without improving direct/autonomous Q is not progress)
- Do NOT redefine or remove the canonical Contact-J gate until the corrected audit
  shows whether the problem is implementation or objective-level mismatch (now
  confirmed: objective-level — gate stays UNCHANGED)
- Do NOT add displacement loss as Q's only objective while dropping
  energy/semigroup/temperature targets
- Do NOT use per-residue MLP probe R² as the only oracle upper bound

---

## 4. Supplementary metrics added (canonical gate UNCHANGED)

Module: `protein_jepa_frame/eval/contact_supplementary.py`

| metric | definition | direction |
|---|---|---|
| soft_contact_error | MAE of sigmoid soft-contact occupancies | lower better |
| contact_brier | Brier score: MSE(predicted prob, binary target) | lower better |
| contact_rank_corr | Spearman rank-corr of predicted vs target closeness | higher better |

All three use the same nonlocal pair mask (min_seq_sep=3, cross-chain eligible).
Verified monotone on oracle sweep (test_supplementary_contact.py).
These are reported ALONGSIDE the hard gate, never as replacements.

---

## 5. Stage-A/B information probe plan (Phase 2, before any training)

### Probe AB-1: Multi-component Q information content (current Q)

File: `tmp_claude/probe_ab1_q_components.py` (to be written)
For each of the four components above, measure:
- How much current Q_context carries (vs shuffled/zero)
- How much delta-Q (Q_future - Q_context) adds
- Whether the signal is per-residue or only global
Protocol: train probe MLP on TRAIN split, evaluate on VAL split (protein-disjoint).
Report: protein-clustered bootstrap 95% CI on each target.

### Probe AB-2: Equivariant encoder information content

File: `tmp_claude/probe_ab2_equivariant.py` (to be written)
Build a FRESH encoder with orientation_features=True (same architecture, random init).
Measure whether raw torsion/orientation features (without pretraining) carry
per-residue kinematic signal that the invariant encoder cannot.
Purpose: establish that an equivariant retrain is NECESSARY (not just sufficient)
for Component 3, and estimate the ceiling after fixing the architecture.
This is a low-cost preflight that does NOT require a full Stage-A retrain.

### Probe AB-3: Semigroup and temperature probe (current Q)

File: `tmp_claude/probe_ab3_semigroup.py` (to be written)
Re-evaluate Gate D invariants on val split:
- Q rollout divergence (does Q stay finite?)
- Semigroup composition consistency (short × short vs long)
- Temperature response (does Q change with temperature in a physically consistent
  direction?)
- No NaN/Inf, no artificial clipping hiding instability
Purpose: before any Stage-B modification, confirm whether Gates D/E already fail.
If they do, Stage-B changes must simultaneously fix those; if they pass, fixing
Component 3 (equivariant encoder) is the only necessary change.

---

## 6. Decision tree after probes complete

```
AB-1: does current Q carry Component 1 (scalars)?
  No  → Stage B: add scalar (Rg, RMSF) auxiliary targets, keep frozen Stage A
  Yes → proceed

AB-1: does current Q carry Component 2 (collective modes)?
  No  → Stage B: add PCA-mode projection target, keep frozen Stage A
  Yes → proceed

AB-2: does orientation_features=True raw encoder outperform invariant on kinematics?
  Yes → Stage A retrain needed for Component 3
        (minimal change: set orientation_features=True, retrain with same config)
  No  → Component 3 ceiling is data-limited even with equivariant encoder;
        do NOT retrain Stage A

AB-3: Gate D/E failures?
  Yes → Stage B training must include energy/semigroup/temperature terms;
        cannot drop them for displacement targets
  No  → Stage B training priorities are purely Components 1-3

Only after probes AB-1/2/3 complete: write a one-paragraph controlled training
proposal, with exact hypothesis, config delta, expected metric change, and
go/no-go criterion before touching any checkpoint.
```

---

## 7. Canonical gate pass criteria (UNCHANGED)

These are carried forward verbatim from the original gate spec:

- Contact-J gate: skCJ ≥ 0.10 with protein-clustered bootstrap 95% CI lower bound ≥ 0.10
- RMSD skill: skR > 0 with 95% CI lower bound > 0
- Q-dependence: oracle Q significantly better than shuffled/zero, with
  direct/autonomous Q significantly better than shuffled/zero separately
  (oracle improvement alone is NOT sufficient)
- Supplementary metrics reported alongside but never substituted for the canonical gate

---

## 8. What has been done

- [x] Phase 0 audit: `artifacts/iter_007_phase0_audit/analysis/PHASE0_AUDIT.md`
- [x] Phase 1 probes (10 probes): all in `tmp_claude/probe_*.py` with results recorded
- [x] Phase 2 corrected gate probe: `tmp_claude/probe_gate_threshold_v2.py`
- [x] Gate audit recorded: `artifacts/iter_007_phase0_audit/analysis/PHASE2_CANONICAL_GATE.md`
- [x] Supplementary contact metrics module: `protein_jepa_frame/eval/contact_supplementary.py`

## 9. What remains (no training yet)

- [x] Probe AB-1: multi-component Q information content (`AB1_Q_COMPONENTS.md`)
- [x] Probe AB-2: equivariant / raw-feature information ceiling (`AB2_EQUIVARIANT.md`)
- [x] Probe AB-3: semigroup/temperature Gate D/E check (`AB3_SEMIGROUP.md`)
- [x] Decision tree evaluation → controlled training proposal (§10, §11 below)

---

## 10. Decision tree evaluation (all probes complete)

```
AB-1: does current Q carry Component 1 (scalars)?
  → PARTIAL. Q carries global scalars (Rg corr 0.79) but per-residue structure is
    collapsed (per-residue corr 0.024). Component 1 needs explicit per-residue
    scalar targets (RMSF, local ΔRg) — Stage B auxiliary target, frozen Stage A OK.

AB-1: does current Q carry Component 2 (collective modes)?
  → NO at per-residue resolution. Oracle Q_future no better than Q_context on
    per-residue displacement. Collective-mode amplitudes are invariant and
    trainable from context → Stage B auxiliary PCA-mode target, frozen Stage A OK.

AB-2: does orientation_features=True raw encoder outperform invariant on kinematics?
  → NO. Raw torsion/orientation input features add ~0 directional signal
    (dir R²: A=−0.001, B=−0.001; mag R²: A=0.004, B=0.024). The directional signal
    lives in TEMPORAL velocity (BA+VEL dir R²=0.151), not static geometry.
  → REVISED CONCLUSION: do NOT retrain Stage A for orientation_features.
    Instead route the temporal velocity channel into Q (Stage B / temporal path).

AB-3: Gate D/E failures?
  → Gate D (energy/semigroup/forces) HEALTHY — must be KEPT, cannot be dropped.
    Gate E (rollout/temperature) WEAK, as a downstream symptom of empty Q, not an
    independent failure. Fixing Q routing should lift Gate E without separate work.

Contact-J gate (canonical, UNCHANGED): objective-level mismatch, not a bug.
  Basin ceiling skCJ=+0.056 [+0.043,+0.070] < 0.10. Intermediate accuracy HURTS
  (biphasic). Gate stays as-is; supplementary soft/Brier/rank metrics reported
  alongside.
```

### Net architectural verdict

The bottleneck is NOT the SE(3)-invariant encoder's lack of orientation features.
It is that the **temporal velocity signal — which demonstrably carries the only
usable per-residue directional information (dir R²=0.15, mag R²=0.23) — is averaged
out before it reaches Q.** The fix is a Stage-B / temporal-routing change, with a
frozen (or lightly fine-tuned) invariant Stage A, NOT an orientation-feature
Stage-A retrain.

---

## 11. Controlled training proposal (requires explicit user authorization)

**Status: NOT STARTED. Training remains unauthorized pending user confirmation.**

**Hypothesis.** The usable per-residue kinematic signal (dir R²≈0.15, mag R²≈0.23)
exists in context velocity (BA+VEL) but is not routed into Q. Adding explicit
multi-component auxiliary targets to Stage B — (1) per-residue scalars (RMSF,
local ΔRg), (2) top-3 collective PCA-mode amplitudes, (3) per-residue context
velocity / future local displacement with an equivariant (frame-relative) head —
while KEEPING all existing Gate-D objectives (energy, semigroup, forces,
temperature) and the canonical Contact-J gate, will make Q carry held-out
per-residue dynamical information without collapsing.

**Config delta (minimal, frozen Stage A):**
- Stage A: FROZEN (orientation_features stays False — AB-2 shows it adds nothing).
- Stage B: add three auxiliary heads + loss terms (scalars, modes, equivariant
  velocity), each weighted so existing energy/semigroup/force losses are not
  reduced below their current contribution. Keep temperature/bath terms.
- Route the temporal velocity channel (frame-to-frame local displacement) into the
  Q update path rather than collapsing it.

**Expected metric change (go/no-go, held-out val, protein-clustered bootstrap CI):**
- GO if direct/autonomous (not just oracle) Q beats shuffled-Q on per-residue
  displacement magnitude (target mag R² > 0.10, 95% CI lower bound > 0) AND
  direction (cos > 0.10, 95% CI lower bound > 0).
- GO if Gate D invariants stay healthy (no regression vs current).
- GO if supplementary contact rank-corr improves with 95% CI lower bound > 0.
- NO-GO / STOP if direct Q improvement is absent while only oracle Q improves
  (the forbidden future-Q-autoencoder failure mode).
- The canonical Contact-J gate (skCJ ≥ 0.10) is NOT expected to pass (objective-level
  mismatch) and is NOT a go/no-go criterion for this experiment; it is reported
  unchanged for the record.

**Before launching:** present this proposal to the user and obtain explicit
authorization. Do not touch any checkpoint until then.
