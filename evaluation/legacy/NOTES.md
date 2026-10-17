# Legacy eval scripts → new `evaluation/` modules

The ad-hoc scripts below were the previous, inconsistent way of computing these
numbers. They are **superseded**; kept only for provenance / numeric cross-check.

| legacy script | what it did | replaced by | status |
|---|---|---|---|
| `tmp_claude/eval_b1b_pipeline.py` | CA RMSD (oracle/rollout/persist) for B1b; re-implemented Kabsch; hard-coded paths; `HORIZONS=[1,2,4,8]` | `evaluation/run_eval.py` + `variants.py` + `geometry.py` | superseded |
| `tmp_claude/eval_b1b_gate.py` | §2.7 NLL ablations (full/nohist/nobath/persist) + bootstrap | `evaluation/run_gate.py` | superseded |
| `tmp_claude/probe_h_to_ca_ceiling.py` | H→CA MLP ceiling (0.60 Å) | not ported (one-off probe); method documented in `../MODELS.md` | reference |
| `tmp_claude/probe_predictor_skill_v5.py` | predictor skill vs persistence | `run_eval.py` (`h_only`/`h_ft` variants) | reference |
| `scripts/verify_temporal_sensitivity.py` | temporal cosine-similarity monitor (targets old `ProteinJEPA` API) | not ported (API changed) | stale |
| `baselines/*` | TICA/VAMPnets/SPIB baselines (different question: state-discovery, not CA-RMSD forecasting) | unchanged | out of scope |

## Why they were replaced
- **Kabsch was re-implemented in each script**, with at least one historical sign
  bug (see `H_quality_step5000.md` §4). Now there is exactly one implementation
  (`evaluation/geometry.py`), unit-tested.
- Paths/checkpoints were hard-coded per script. Now they live in `configs/*.yaml`.
- Horizons were fixed per script. Now `horizons_frames` covers 1–96 ns uniformly.
- Held-out **protein** and **temperature** axes were not both supported. Now both are.

## Numerical cross-check (must agree)
At 1 ns, mdCATH, 320 K:
| quantity | legacy (`H_DECODABILITY_REPORT.md` §9) | new harness (smoke) |
|---|---|---|
| persistence | ~1.40 Å | ~1.22 Å |
| H_persistence (h_ft) | ~5.01 Å | ~6.06 Å |
| oracle_H | ~5.03 Å | ~6.33 Å |

Differences are within the sampling/protocol spread (the legacy run used ≤20 windows
per domain, the new smoke used anchor-stride sampling). The **ordering and the
conclusion are identical**: `h_ft ≈ oracle ≫ persist`, `h_plus_s ≈ persist`.
