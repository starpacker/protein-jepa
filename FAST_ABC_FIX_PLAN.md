# Fast ABC Fix Plan — Unified State, Time, Loss, and Evaluation Contracts

## Fix Priority Order

| # | Priority | Problem | Root Cause | Fix |
|---|----------|---------|------------|-----|
| 1 | P0 | Train/eval use different dt_ps | eval_rollout_quality.py infers 4000ps from data; eval_baselines.py hardcodes 100ps; model tolerance mismatch | Unify to `model.base_step_ps` (100ps), all eval scripts use same dt |
| 2 | P0 | substepping silently degrades accuracy | max_substeps=10 for 4000ps → 400ps actual substep; 20× error amplification | Remove substepping cap; use exact substeps; add CANONICAL_DT_PS config |
| 3 | P0 | State encoding: train uses temporal history, eval uses spatial-only | Training: h_last + frame_state → Q; Eval: spatial only | Add `encode_state()` unified interface; all callers use it |
| 4 | P0 | Learned residual/noise untrained but used in eval | rollout_mse_loss uses stochastic=False, eval uses stochastic=True | Remove residual/noise from closed-loop; keep physical-only transition |
| 5 | P0 | Temperature dead in energy graph | ResidueMessagePassingPotential accepts temperature_k but ignores it | Add log(T/T_ref) to node input; write unit test |
| 6 | P0 | Stage C decoder trained on oracle Q | q_decode = q_oracle (future-observed), never uses predicted Q | Scheduled sampling: q_decode = blend(q_pred, q_oracle, ramp) |
| 7 | P0 | reconstruct_backbone_from_torsions ignores torsions | Function receives torsions but computes ideal geometry only | Rewrite to use torsions for backbone dihedrals |
| 8 | P1 | clash loss uses CA-only, eval uses atom37 | objectives.py clash_penalty on CA coords | Expand clash to atom37 with atom mask |
| 9 | P1 | Train/eval max_residues mismatch | Stage B config: 100, eval: 256 | Unify; add per-length bucket support |
| 10 | P1 | Validation samples from single protein | Sequential manifest iteration hits 1b4uB00 repeatedly | Stratified sampling by protein_id |

## Implementation Phases

### Phase 1: Stage B — Unified time & state contract
Files: config.py, stage_b/model.py, stage_b/objectives.py, stage_b/transition.py,
       training/stage_b.py, eval_rollout_quality.py, eval_baselines.py

### Phase 2: Stage B — Temperature pathway
Files: energy.py, stage_b/model.py

### Phase 3: Stage C — Oracle→predicted Q + geometry fix  
Files: training/stage_c.py, structure_decoder.py, objectives.py

### Phase 4: Data consistency
Files: configs/*.yaml, dataset.py

## Verification Gates After Each Phase
- Phase 1: Run rollout at 100ps, norm should match baseline
- Phase 2: Run at 320K vs 413K, q_diff should differ
- Phase 3: q_zero/q_shuffle should change decoder output; changing phi/psi changes backbone
- Phase 4: Validation covers ≥5 distinct proteins
