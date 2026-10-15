# Reviewer Response Plan: Protein Latent Dynamics → World Model

> **Current state**: history-conditioned latent dynamics with endpoint structural decoding.
> **Target**: learned-state → transition → free-rollout → observation → intervention closed loop.

This plan addresses all 9 reviewer concerns, organized P0→P4 by priority.
Each step is independently executable, verifiable, and produces a concrete artifact.

---

## P0: Canonical World-Model Evaluation Framework

*The single most important fix: establish an uncheatable eval entry point.*

### P0.1 — Fix `max_frames` config-hardcode mismatch

**File**: `scripts/verify_m02_rollout.py:82-85`
**Problem**: `max_frames=64` hardcoded, but canonical checkpoint `mdcath_stage_b_nonoverlap` uses `max_frames=128`.
Loading produces state-dict shape mismatch → verification silently fails or is invalid.

**Action**:
- Read `max_frames` from the checkpoint's saved config, not hardcode.
- If checkpoint lacks config, scan state-dict for position encoding size and infer.

```python
# In verify_m02_rollout.py, replace:
cfg = ModelConfig(latent_dim=128, encoder_depth=2, temporal_depth=1,
    num_heads=2, num_rbf=8, max_frames=64, max_residues=128, ...)

# With:
max_frames = _infer_max_frames_from_checkpoint(ckpt_state)  # scan temporal_encoder.*position* shapes
cfg = ModelConfig(..., max_frames=max_frames, ...)
```

**Validation**: `python scripts/verify_m02_rollout.py artifacts/mdcath_stage_b_nonoverlap/checkpoint_latest.pt` must load without shape errors.

---

### P0.2 — Create unified `scripts/eval_world_model.py`

**Design**: A single eval script with 5 disjoint columns. No cherry-picking allowed.

| Mode | Q Source | Decoder input | Use case |
|------|----------|---------------|----------|
| `oracle_endpoint` | future coords → Stage B encode → `q_oracle` | decoder(q_oracle, H, anchor) | Upper bound |
| `direct_endpoint` | DirectHead(H, tau, T) → `q_direct` | decoder(q_direct, H, anchor) | Learned endpoint |
| `autonomous_recursive` | H → q0, roll forward n steps → `q_rec` | decoder(q_rec, H, anchor) | **World model** (main result) |
| `persistence` | N/A | anchor coords (no decoder) | Naive baseline |
| `q_shuffle` | shuffled oracle Q across batch | decoder(q_shuffled, H, anchor) | Does Q matter? |

**Per-horizon breakdown** (1, 5, 20, 100 step free rollout):
```
horizon_ps  | oracle  | direct  | autonomous | persist | Q-shuffle
100ps       | ...     | ...     | ...        | ...     | ...
500ps       | ...     | ...     | ...        | ...     | ...
2000ps      | ...     | ...     | ...        | ...     | ...
10000ps     | ...     | ...     | ...        | ...     | ...
```

**Metrics per cell**:
- Kabsch RMSD_CA
- Kabsch-aligned CA point error (rename from "FAPE")
- DistMAE_CA (pairwise)
- Contact Jaccard (0-8Å)
- Weak Contact J (8-15Å)
- Transient Contact J (15-25Å)
- RMWD

**Split guarantees**:
- protein-disjoint: train proteins ≠ val proteins
- trajectory-disjoint: different trajectory windows for same protein not in both splits
- Seeds: 3 random seeds per sample, report mean ± std

**Output artifacts**:
- `artifacts/world_model_eval/per_protein_results.jsonl` — one line per protein × mode × horizon
- `artifacts/world_model_eval/summary_table.md` — main result table
- `artifacts/world_model_eval/full_report.json` — machine-readable aggregate

---

### P0.3 — Implement ablation decoder variants

**New file**: `protein_jepa_frame/models/stage_c/ablations.py`

Three critical decoder variants to isolate Q's contribution:

```python
class QZeroDecoder(nn.Module):
    """Decoder with Q replaced by zeros — tests if H alone drives structure."""
    def forward(self, aatype, h_frozen, q_original, R_anchor, t_anchor, mask):
        q_zero = torch.zeros_like(q_original)
        return shared_decoder(aatype, h_frozen, q_zero, R_anchor, t_anchor, mask)

class QShuffleDecoder(nn.Module):
    """Decoder with Q shuffled across batch — tests if Q semantics matter."""
    def forward(self, aatype, h_frozen, q_original, R_anchor, t_anchor, mask):
        idx = torch.randperm(q_original.shape[0])
        q_shuffled = q_original[idx]
        return shared_decoder(aatype, h_frozen, q_shuffled, R_anchor, t_anchor, mask)

class HZeroDecoder(nn.Module):
    """Decoder with H replaced by zeros — tests if H alone drives structure."""
    def forward(self, aatype, h_frozen, q_original, R_anchor, t_anchor, mask):
        h_zero = torch.zeros_like(h_frozen)
        return shared_decoder(aatype, h_zero, q_original, R_anchor, t_anchor, mask)
```

**Validation decision table**:
| Ablation | If RMSD ≈ oracle | Implication |
|----------|-------------------|-------------|
| Q=0 | Yes | Q unused — decoder learned from H alone |
| Q=0 | No (worse) | Q genuinely contributes |
| Q-shuffle | Yes ≈ oracle | Q carries no residue-specific information |
| H=0 | Yes | H unused — decoder learned from Q alone |
| H=0 | No (catastrophic) | H critical — geometry comes from frozen context |

Run on 200 val samples after P0.2 is complete.

---

### P0.4 — Rename "FAPE" to "Kabsch-aligned CA point error"

**Files to modify**:
- `scripts/eval_stage_c_full_metrics.py` (all print/compute references to fape_ca_kabsch)
- `scripts/eval_stage_c_heldout.py`
- `scripts/eval_stage_c_endtoend.py`
- `artifacts/mdcath_stage_c_compact/full_metrics.json` → rename key `fape_ca_kabsch` → `kabsch_ca_point_error`

**Rationale**: True FAPE uses residue-local frame alignment. Our metric uses global Kabsch rotation on CA atoms. This is a valid metric but must be accurately named.

---

### P0.5 — Add persistence baseline

**Implementation** in `scripts/eval_world_model.py`:

```python
def persistence_metrics(anchor_coords, future_coords, residue_mask):
    """Anchor structure as prediction — how much does the protein move?"""
    anchor_ca = anchor_coords[:, 1, :]  # [L, 3]
    future_ca = future_coords[:, 1, :]
    rmsd = kabsch_rmsd(anchor_ca, future_ca)
    # ... compute all 7 metrics treating anchor as prediction
    return metrics_dict
```

This establishes the "do nothing" floor. Autonomous rollout must beat persistence at all horizons to be useful.

---

## P1: Latent State Architecture Redesign

### P1.1 — Define canonical state `z_t = (q_t, m_t, e_t)`

**New file**: `protein_jepa_frame/models/stage_b/state.py`

```python
@dataclass
class LatentState:
    """Canonical latent state for protein dynamics world model.
    
    z_t = (q_t, m_t, e_t) where:
      q_t: [B, L, 8] — internal conformational state (SE(3)-invariant)
      m_t: [B, L, 16] — Mori/finite-memory bath state
      e_t: [B, 8]     — environment state (temperature-dependent equilibrium info)
    """
    q: torch.Tensor   # [B, L, 8]
    m: torch.Tensor   # [B, L, 16]
    e: torch.Tensor   # [B, 8]

    @property
    def batch_size(self): return self.q.shape[0]
    
    @property
    def n_residues(self): return self.q.shape[1]
    
    def clone(self): return LatentState(
        self.q.clone(), self.m.clone(), self.e.clone()
    )
    
    def detach(self): return LatentState(
        self.q.detach(), self.m.detach(), self.e.detach()
    )
```

### P1.2 — Implement closed-loop transition

**Modify**: `protein_jepa_frame/models/stage_b/model.py`

Replace the current "encode future → q_next" teacher-forcing path with a true closed-loop transition:

```python
def transition(
    self, 
    z_t: LatentState, 
    dt_ps: float, 
    temperature_k: torch.Tensor,
    stochastic: bool = True,
    ca_distances: torch.Tensor | None = None,
) -> LatentState:
    """Closed-loop transition: p(z_{t+1} | z_t, e, T, dt).
    
    Provides: z_{t+1} = integrator(z_t, dt) + learned_residual(z_t, dt) + noise
    
    No access to future coordinates or future Q.
    """
    # 1. Physical integrator (Euler-Maruyama with energy graph)
    q_phys, m_phys = self.step(
        z_t.q, z_t.m,
        ca_distances=ca_distances,
        residue_mask=residue_mask,
        temperature_k=temperature_k,
        dt_ps=dt_ps,
        stochastic=stochastic,
    )
    
    # 2. Learned residual transition (optional, initially zero-initialized)
    q_residual = self._residual_transition(z_t.q, z_t.e, temperature_k, dt_ps)
    
    # 3. Calibrated noise
    noise_scale = self._noise_scale(z_t.q, z_t.e, temperature_k)
    
    q_next = q_phys + q_residual
    if stochastic:
        q_next = q_next + noise_scale * torch.randn_like(q_next)
    
    return LatentState(q=q_next, m=m_phys, e=z_t.e)
```

### P1.3 — Train with closed-loop supervision

**Modify**: `protein_jepa_frame/training/stage_b.py`

Training must simultaneously optimize:

| Loss | Weight | Description |
|------|--------|-------------|
| one-step NLL | 1.0 | `-log p(z_{t+1} | z_t)` — matches current transition_nll |
| multi-step teacher forcing | 0.5 | NLL at horizons 2, 4, 8 with intermediate oracle Q |
| free-rollout endpoint | 0.3 | NLL at horizon T from autonomous rollout |
| semigroup consistency | 0.1 | `z_{t→t+2} ≈ transition(transition(z_t, 1), 1)` |
| uncertainty calibration | 0.05 | empirical coverage vs predicted variance |

---

### P1.4 — Fix Stage B training → rollout state inconsistency

**Problem** (reviewer point #7): Stage B currently trains:
```
q_t (history-aware) → future-frame spatial q_next (teacher-forced)
```
But rollout uses:
```
q_t → transition(q_t, dt) → q_{t+1} (no future access)
```
The distribution shift between training states and rollout states causes semantic drift.

**Fix**: Add rollout-NLL loss (P1.3's free-rollout endpoint) and ensure rollout trajectory visits the same latent regions as teacher-forced training by interleaving rollout and teacher-forcing steps.

**Implementation** in Stage B training loop:
```python
# Interleaved training: 50% teacher-forcing, 50% autonomous rollout
if torch.rand(1) < 0.5:
    # Teacher-forced one-step NLL
    loss = stage_b_module(batch)["transition_nll"]
else:
    # Autonomous rollout NLL at random horizon 1..max_rollout
    q_traj, _ = model.rollout(q0, s0, n_steps=horizon_steps, ...)
    loss = nll(q_traj[-1], q_target)
```

---

## P2: Passive World Model (before interventions)

### P2.1 — Train autonomous rollout end-to-end

**Config**: `configs/mdcath_stage_b_rollout.yaml`
```yaml
output_dir: artifacts/mdcath_stage_b_rollout
init_checkpoint: artifacts/mdcath_stage_b_nonoverlap/checkpoint_latest.pt
model:
  lspwm_per_residue_enabled: true
  lspwm_use_bath: true
  lspwm_history_aware: true
  lspwm_rollout_enabled: true
  lspwm_rollout_steps: 16
  lspwm_rollout_warmup_steps: 500
  lspwm_semigroup_consistency_weight: 0.1
optimization:
  lr: 3e-5
  epochs: 30
  gradient_accumulation_steps: 4
  steps_per_epoch: 300
  lspwm_rollout_trajectory_nll_weight: 1.0
```

**Training**: `python scripts/run_training.py configs/mdcath_stage_b_rollout.yaml`

---

### P2.2 — Comprehensive rollout quality evaluation

**New script**: `scripts/eval_rollout_quality.py`

| Metric | Description |
|--------|-------------|
| RMSD vs MD at horizons 1/5/20/100 | Calculated per-protein, per-seed |
| Ensemble coverage | % of MD structures within rollout ensemble envelope |
| ACF comparison | Autocorrelation of Q over trajectory vs MD |
| VAMP-2 score | Kinetic model quality of latent trajectory |
| Transition probability | Clustering-based state transition comparison |
| Equilibrium distribution | MMD between rollout and MD Q distributions |

**Per-protein reporting** — raw .jsonl with one line per (protein, horizon, seed):
```json
{"protein_id": "1ABC", "horizon_ps": 100.0, "seed": 42, "rmsd_kabsch": 2.34, "contact_j": 0.71, ...}
```

---

## P3: Observation Layer Redesign

### P3.1 — Replace "FAPE" → canonical observation metrics

**Modify**: `protein_jepa_frame/models/stage_c/objectives.py`

True FAPE (residue-local frame-aligned) for absolute-position training with batch_size ≥ 4:
```python
def true_fape_loss(pred_coords, target_coords, target_R, target_t, mask=None, clamp=10.0):
    """Correct FAPE: for each residue i, measure error in frame i's local coordinates."""
    # Implementation using target_R, target_t, pred_coords, target_coords
    # [B, L, L, 4, 3] pairwise frame-aligned computation
    ...
```

Primary observation metrics (replacing Kabsch-only):
```python
def structure_metrics(pred_bb, target_bb, mask):
    return {
        "rmsd_kabsch": kabsch_rmsd(pred_ca, target_ca),
        "kabsch_point_error": kabsch_aligned_ca_error(pred_bb, target_bb),
        "dist_mae": pairwise_distance_mae(pred_ca, target_ca),
        "contact_jaccard": contact_jaccard(pred_ca, target_ca, 0, 8),
        "rg_error": abs(radius_of_gyration(pred_ca) - radius_of_gyration(target_ca)),
        "rmsf_pearson": rmsf_correlation(pred_ensemble, target_ensemble),  # needs ≥2 samples
    }
```

### P3.2 — Absolute COM as auxiliary variable only

**Modify**: `protein_jepa_frame/models/stage_c/structure_decoder.py`

The `q_position_head` (currently zero-initialized) should be:
1. **Never forced** to predict global COM from Q alone (identifiability problem)
2. **Only trained** when `anchor_observation` is available as input (e.g., H already carries position info)
3. **Documented** as an auxiliary head that bridges SE(3) gauge to lab frame, not as part of the core latent dynamics

```python
# In structure_decoder.py forward():
# q_position_head receives H + Q, not Q alone.
# H carries anchor position information through spatial encoding.
# The head is trained with a separate auxiliary loss, not the main FAPE loss.
delta_com = self.q_position_head(
    torch.cat([q_h_pooled, anchor_com.unsqueeze(0)], dim=-1)
)
```

---

## P4: Intervention Protocol

### P4.1 — Temperature jump experiment design

**New config**: `configs/mdcath_intervention_temp_jump.yaml`

**Protocol**:
1. Train on 4 fixed temperatures: 320K, 348K, 379K, 413K
2. Intervention test:
   ```
   348K equilibrium (1000ps)
   → t0: jump to 413K
   → record autonomous trajectory at 413K for 5000ps
   ```
3. Compare model response vs real MD at 413K:
   - Does unfolding rate increase?
   - Does RMSF increase at expected residues?
   - Does Rg trajectory match MD statistics?

**Requires**: Temperature-dependent energy graph (`ResidueMessagePassingPotential` already has `reference_temperature_k`; need to verify it's actually wired through `compute_energy_and_forces` — currently `temperature_k` is passed but not used inside the GNN).

### P4.2 — Wire temperature through energy graph

**Modify**: `protein_jepa_frame/models/lasp/energy.py:ResidueMessagePassingPotential`

```python
def forward(self, q, ca_distances, residue_mask, condition=None, temperature_k=None):
    # Add temperature conditioning to energy computation
    if temperature_k is not None:
        temp_feat = torch.log(temperature_k / self.reference_temperature_k)
        # Broadcast to per-residue
        ...
    # Existing GNN forward, enhanced with temperature features
```

### P4.3 — Fix `compute_energy_and_forces` to pass temperature

**Modify**: `protein_jepa_frame/models/stage_b/model.py:175-193`

Current signature receives `temperature_k` but doesn't pass it to `self.energy_graph(...)`. Must add:
```python
def compute_energy_and_forces(self, q, ca_distances=None, residue_mask=None, temperature_k=None):
    energy = self.energy_graph(
        q_detached, ca_distances, residue_mask,
        condition=None,
        temperature_k=temperature_k,  # NEW: actually forward temperature
    )
```

---

## P5: Side-chain and All-atom Validation

### P5.1 — Side-chain torsion accuracy

**New metrics** in validation scripts:
- **χ1 accuracy**: % of residues with χ1 within ±30° of MD
- **χ1-4 accuracy**: all rotatable torsions
- **Side-chain RMSD**: after Kabsch backbone alignment, measure side-chain heavy atom RMSD
- **Clash rate**: % of atom pairs < 2.0Å in predicted structure

### P5.2 — Backbone geometry checks

- **Peptide bond planarity**: ω angle deviation from 180° (or 0° for cis)
- **Ramachandran outliers**: % of φ/ψ in disallowed regions
- **Chirality violations**: Cα chirality check

---

## P6: Baseline Comparisons

### P6.1 — Implement baselines

| Baseline | Description | Implementation |
|----------|-------------|----------------|
| **Linear interpolation** | `coords_t0 + (coords_future - coords_t0) * t/tau` | Pure interpolation in coordinate space |
| **Transformer predictor** | 4-layer transformer: H_t0 + tau + T → backbone coords | New model module |
| **RSSM (RSSM-like)** | Deterministic path + stochastic path, 1-step NLL | Adapt from tmjepa codebase |

### P6.2 — Baseline comparison table

Use P0.2 eval framework, add baselines as additional columns.

---

## P7: Paper Positioning

Update all documentation, abstract text, and model names from "protein dynamics world model" to:

> **"History-conditioned thermodynamic latent dynamics with endpoint structural decoding"**

Only upgrade to "world model" after P1-P3 are complete AND autonomous rollout beats persistence + linear + Transformer baselines on held-out proteins.

---

## Implementation Order

```
Week 1-2:  P0.1, P0.2, P0.3, P0.4, P0.5  (Eval framework)
Week 3-4:  P1.1, P1.2, P1.3, P1.4          (Latent state redesign)
Week 5:    P2.1, P2.2                        (Rollout training + eval)
Week 6:    P3.1, P3.2                        (Observation layer)
Week 7-8:  P4.1, P4.2, P4.3                  (Temperature intervention)
Week 9:    P5.1, P5.2                        (Side-chain validation)
Week 10:   P6.1, P6.2                        (Baselines)
Week 11:   P7                                (Paper positioning)
```

### ### P0 RESULTS (COMPLETED)

#### P0.1 — Fix `max_frames` config-hardcode mismatch ✅
- `verify_m02_rollout.py` now auto-infers config from checkpoint
- All 5 M02 tests pass: step-size sensitivity, sub-stepping consistency, noise scale, long-horizon, future leakage
- Fixed OOM from `create_graph=True` accumulation by adding `.detach()` after each step
- Known issue: |Q| grows monotonically to ~18641 over 1000 steps (documented, not suppressed)

#### P0.2 — Unified `scripts/eval_world_model.py` ✅
- 5 columns (oracle/direct/autonomous/persistence/q_shuffle) × 3 horizons (100/500/2000ps) × 2 seeds
- 600 records across 100 val samples
- **Key finding**: oracle ≈ direct ≈ autonomous ≈ q_shuffle within 0.003Å RMSD — Q contributes nothing

#### P0.3 — Ablation decoder variants ✅
- 6 variants tested: oracle, direct, q_zero, h_zero, q_noise, q_random
- **Q=0 ≈ oracle** (diff < 0.0001Å): Q is unused
- **H=0 ≈ oracle** (diff < 0.02Å): H contributes < 1% of structural information
- **Q_random ≈ oracle**: Q content completely ignored
- **Conclusion**: The decoder is an anchor refinement autoencoder — anchor + aatype + IPA blocks drive all 2.19Å RMSD

#### P0.4 — Rename "FAPE" → "Kabsch-aligned CA point error" ✅
- All 3 eval scripts + full_metrics.json updated
- True FAPE uses residue-local frames; our metric uses global Kabsch

#### P0.5 — Persistence baseline ✅
- Integrated into `eval_world_model.py`
- Persistence RMSD = 2.18Å (no decoding — just Kabsch-align anchor to target)
- Persistence is LOWER on Kabsch point error (1.59 vs 1.92Å decoder) because decoder introduces noise

---

Dependency graph
```
P0 (eval)  ──┐
             ├──→ P1 (latent state) ──→ P2 (rollout train) ──→ P4 (intervention)
             │                                                  │
             ├──→ P3 (observation) ─────────────────────────────┤
             │                                                  │
             └──→ P5 (sidechain) ──→ P6 (baselines) ──→ P7 (paper)
```

---

## Success Criteria (Nature-grade)

Each item is a gate: next step only after current gate passes.

| Gate | Criterion | Measurement |
|------|-----------|-------------|
| G1 | Autonomous rollout RMSD < persistence RMSD on ≥80% held-out proteins | P0.2 eval framework |
| G2 | Error grows monotonically with horizon (not saturating or chaotic) | P0.2 per-horizon breakdown |
| G3 | Ensemble coverage ≥ 80% (MD structures within rollout 2σ) | P2.2 coverage metric |
| G4 | ACF of latent Q matches MD ACF (correlation > 0.5 at lag 10) | P2.2 ACF comparison |
| G5 | VAMP-2 score > 0 → non-trivial kinetics | P2.2 VAMP score |
| G6 | Temperature jump produces >2× faster unfolding at 413K vs 348K | P4.1 intervention test |
| G7 | Ablations confirm Q≠0, H≠0 both degrade performance | P0.3 ablation table |
| G8 | 3-seed mean ± std reported for all metrics | P0.2 seed protocol |
| G9 | No oracle/future data in any autonomous metric | Code audit (already partially done) |

### P1 RESULTS (COMPLETED)

#### P1.1 — Canonical state z_t = (q_t, m_t, e_t) ✅
- `protein_jepa_frame/models/stage_b/state.py`: `LatentState` dataclass with q, m, e tensors
- `TransitionResult` dataclass for step results with diagnostics
- Properties: batch_size, n_residues, device; methods: clone/detach/to/apply_mask/zeros

#### P1.2 — Closed-loop transition ✅
- `protein_jepa_frame/models/stage_b/transition.py`: `ClosedLoopTransition`
  - `ResidualTransitionNet`: zero-initialized (residual = 0 at start)
  - `NoiseScaleNet`: matches physical diffusion initially
  - `ClosedLoopTransition.forward()`: integrator + learned_residual + calibrated_noise
  - `ClosedLoopTransition.rollout()`: n-step autonomous trajectory
- All params zero-init: initial behavior identically physical integrator

#### P1.3 — Multi-objective training losses ✅
- `protein_jepa_frame/models/stage_b/objectives.py` extended with:
  - `rollout_nll_loss()`: autonomous N-step endpoint NLL vs target Q
  - `semigroup_consistency_loss()`: z_{t+2} ≈ 2×1-step vs 1×2-step
- Training loop updated with config-controlled weights
- Config: `configs/mdcath_stage_b_rollout.yaml` (lr=3e-5, rollout_w=0.3, semigroup_w=0.1)

#### P1.4 — Stage B training → rollout state inconsistency fix ✅
- Training loop now interleaves teacher-forced NLL + autonomous rollout NLL + semigroup
- Configurable via `lspwm_rollout_trajectory_nll_weight` and `lspwm_semigroup_consistency_weight`
- Random horizon (1..8 steps) per batch for rollout — visits same latent regions as teacher-forced

---

### CRITICAL FINDING FROM P0 (DECODER AUDIT)

**The decoder does not use Q or H for structure generation.** This was confirmed by:

| Ablation | RMSD (Å) | Δ from Oracle |
|----------|----------|---------------|
| Oracle | 2.1881 | — |
| Q=0 | 2.1882 | +0.0001 |
| H=0 | 2.2099 | +0.0218 |
| Q_random | 2.1960 | +0.0079 |
| Persistence | 2.1794 | -0.0087 (better!) |

The decoder is effectively an **anchor refinement autoencoder**: aatype + anchor frames + IPA blocks produce a 2.19Å RMSd structure via a near-identity mapping. Q and H contribute < 1% of the information.

**Implication for P3+**: The Stage C decoder must be retrained with a fundamentally different objective that forces Q-dependence. Options:
1. Train decoder from scratch with Q as the ONLY input (no H, no aatype)
2. Train decoder to predict Δcoords (anchor→target displacement) rather than absolute coords
3. Add adversarial Q-discriminator to force Q utilization
4. Use Q-conditioned IPA instead of anchor-conditioned IPA

This will be addressed in later phases as the decoder redesign.

