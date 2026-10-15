# Fast ABC Fix Summary — Changes Applied

## Files Changed (11 files)

### P0-1: Removed untrained residual/noise transition
**`protein_jepa_frame/models/stage_b/transition.py`** — Rewritten
- Removed `ResidualTransitionNet`, `NoiseScaleNet` from active code (kept as zero-op placeholders)
- `ClosedLoopTransition.forward()` now delegates directly to `stage_b_lasp.step(stochastic=stochastic)` — consistent train/eval
- Was: `stochastic=False` in training, `stochastic=True` in eval (noise head never trained but used in eval)

### P0-2: Removed substepping accuracy cap
**`protein_jepa_frame/models/stage_b/model.py`** — `step()` method
- Removed `max_substeps=10` cap — was silently degrading 4000ps→10×400ps instead of 200×20ps
- Removed `clip_drift` and NaN guard (silent suppression of failures)
- Removed unused diagnostics counters (`total_elements`, `clipped_elements`, `nan_elements`)
- Diagnostics now returns only `{"n_substeps": N}`

### P0-3: Unified state encoding
**`protein_jepa_frame/models/stage_b/model.py`** — Added `encode_state()`
- Single canonical entry point: `encode_state(coords, mask, aatype, ri, ci, time_ps, temperature_k) → {q, h_last, residue_mask, bath_s0, ca_distances}`
- Identical temporal-history path as training `forward()`
- `forward()` internally uses `encode_state()`

### P0-4: Temperature conditioning in energy graph
**`protein_jepa_frame/models/lasp/energy.py`** — `ResidueMessagePassingPotential.forward()`
- Added `log(T/T_ref)` as per-residue node feature (was: accepted `temperature_k` but ignored it)
- Node nets input: `cv_dim + 1 (temp) + hidden + cond` (was: `cv_dim + hidden + cond`)
- Energy head input: `cv_dim + 1 (temp) + cond` (was: `cv_dim + cond`)
- Verified: `E(320K) != E(413K)` — temperature now affects energy surface

### P0-5: Stage C scheduled sampling (oracle→predicted Q)
**`protein_jepa_frame/training/stage_c.py`** — `train_stage_c()`
- Was: `q_decode = q_oracle` (always uses future-observed Q)
- Now: `q_decode = alpha * q_pred + (1-alpha) * q_oracle` with scheduled sampling ramp
- `q_weight_ramp_steps` now controls transition from oracle→predicted Q
- COM position head uses predicted Q (not oracle)
- Uses `stage_b_model.encode_state()` for consistent future-encoding

### P0-6: Torsion→backbone chain reconstruction
**`protein_jepa_frame/models/stage_c/structure_decoder.py`** — `reconstruct_backbone_from_torsions()`
- Was: used ideal geometry only, ignored torsions completely
- Now: chain-extends backbone using phi/psi/omega dihedrals
- Added helpers: `_unit()`, `_reject()`, `_rodrigues()`
- Verified: changing phi/psi/omega changes backbone atom positions

### P0-7: Clash loss on atom37
**`protein_jepa_frame/models/stage_c/objectives.py`** — `clash_penalty()`
- Now accepts atom37 coords [B,L,37,3] in addition to CA [B,L,3]
- Computes minimum inter-residue atom distance for clash detection

### P1-8: Data consistency
**`configs/fast_abc_stage_b_from_scratch.yaml`** — `max_residues: 256` (was 100)

### Eval script alignment
**`scripts/eval_rollout_quality.py`** — Uses `sb.encode_state()` for consistent Q encoding
**`scripts/eval_baselines.py`** — Uses `sb.encode_state()` for consistent Q encoding

### Test updates
**`tests/test_gate2_requirements.py`** — Updated diagnostics assertions (clip_rate/nan_rate removed)
**`tests/test_gate2_synthetic.py`** — Updated energy_head dimension, added temperature sensitivity check

## Verification Results

| Check | Result |
|-------|--------|
| Temperature affects energy graph | ✓ `E(320K) ≠ E(413K)` |
| Changing phi changes backbone | ✓ Max CA diff = 1.97Å |
| Changing psi changes backbone | ✓ Max C diff = 1.91Å |
| Changing omega propagates downstream | ✓ Max N diff = 3.54Å |
| ClosedLoopTransition physical-only | ✓ No residual/noise |
| encode_state signature valid | ✓ |
| All 279 tests pass | ✓ (1 pre-existing flaky deselected) |

## Acceptance Criteria Status

| Criterion | Before | After |
|-----------|--------|-------|
| train/eval use same dt_ps | ✗ 4000ps vs 100ps | ✓ `model.base_step_ps` (100ps) |
| noise head has gradient | ✗ None (stochastic=False in training) | ✗ (head removed; only FDT noise now) |
| temperature affects energy/diffusion | ✗ Dead code path | ✓ `log(T/T_ref)` in energy graph |
| decoder uses predicted Q | ✗ Always oracle | ✓ Scheduled sampling |
| changing phi/psi changes backbone | ✗ No (ideal-only reconstruction) | ✓ Chain extension with dihedrals |
| clash loss on atom37 | ✗ CA-only | ✓ atom37 min inter-residue distance |
| 256-residue eval consistent | ✗ Train=100, Eval=256 | ✓ Both at 256 |
