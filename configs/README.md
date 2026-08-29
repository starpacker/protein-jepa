# Experiment configurations

## Current framework

The canonical V4-D workflow uses:

- `phase4_v4_d1_scratch_dynamic_1k.yaml`: learn the dynamic-only six-state
  representation from scratch.
- `phase4_v4_d2_reversible_kinetics_1k.yaml`: load D1 step 750, freeze the
  representation, and fit the reversible continuous-time generator.

These two configs use repository-relative paths and expect
`manifests/mdcath.train.jsonl`.

## Research history

The remaining YAML files are exact experiment records for Phase 1 through V4-C
and failed or diagnostic variants. Many preserve the absolute paths used on the
original training machine. Update `output_dir`, `manifest_path`, and checkpoint
paths before rerunning them elsewhere.

Configurations are not equivalent evaluation protocols. Use the result and
design documents under `docs/` when comparing checkpoints.
