# Model and checkpoint contracts

New evaluation uses content-verified **schema-v2** bundles; a path or tensor shape
match is insufficient. Existing legacy numbers are diagnostic artifacts and cannot
establish H+S, JEPA or temperature-generalization gains.

A B1a bundle contains `encoder`, `stage_a_v2_config`, `oracle`,
`stage_b1a_config`, `representation_contract`, and saved training provenance. Its
normalizer must be fitted on training H; the decoder input is raw H. A B1b bundle
copies the complete frozen representation and adds `model`, `stage_b1b_config`,
its autonomous training contract, run scope and oracle-gate report. Its dynamics
input is normalized H, with inverse normalization before the **same** oracle.

Direct prediction requires a Stage-A `prediction_contract` declaring
`raw_H_history → delta_factors`. Actual predictor encoder state/config must match
the calibrated representation. Fine-tuning the encoder therefore requires
adapting/retraining the predictor; matching `latent_dim=128` is insufficient.

Formal autonomous results require `run_scope=oracle_gate_passed`, a schema-v2
passed held-out-protein oracle gate for that exact representation, and hash-locked
real-data lineage. Contract-smoke checkpoints always produce diagnostics.

The loader checks actual encoder/normalizer/decoder/config hashes and upstream
checkpoint/manifest snapshots. The runner records effective config/source/manifest
hashes and requires every actual training manifest to be declared for holdout
verification. Do not overwrite a source `checkpoint_latest.pt` referenced by a
calibrated bundle; archive the locked snapshot first.

The repaired pilot example paths in `configs/` are under
`artifacts/repaired_v2/pilot_20261017/`. They must be created by the repaired
training sequence; they are not claims of completed experiments. CLI checkpoint
overrides allow explicit immutable snapshots.

Legacy files are rejected by default. Explicit `--allow-legacy` enables diagnostic
adapters only. A legacy direct model additionally needs an explicit
`absolute_factors|delta_factors` output declaration; a legacy H+S model needs its
explicit training encoder source. Architecture compatibility does not certify
old target semantics, normalizer lineage or scientific comparability.

The earlier SHA table below identifies archived files only; these are **not**
headline checkpoint mappings for the repaired evaluator:

| archived checkpoint | SHA256 |
|---|---|
| `artifacts/abc_v3/stage_a_fix4/checkpoint_latest.pt` | `144fe92a1656a97b73186dfc284f4d764d982d9b026784d28856561f9bddeb6b` |
| `artifacts/abc_v3/stage_a_mixed/checkpoint_latest.pt` | `fa78d675a8d94da6aeace1e48cd25d9e838f36579de1cee9704fc64530fa1a56` |
| `artifacts/abc_v3/stage_a_fix4_frozen/checkpoint_step6450.pt` | `3db3ed9cca4554ab224a0c94a5a48bce052d8e9fbbe8513b31cc1c3320192e04` |
| `artifacts/abc_v3/stage_b1a_fix4_unfrozen/checkpoint_latest.pt` | `0b163398c4ef5e3f8b1d046da5ad515d6032b3b90bd00712268dcae99b5ba568` |
| `artifacts/abc_v3/stage_b1a_frozen/checkpoint_latest.pt` | `08e78ff7401b186b629961b3ccb66e1ffe24197c65cf924399ce2cf45fd37e08` |
| `artifacts/abc_v3/stage_b1b_mdcath/checkpoint_latest.pt` | `2651ad98c3c8acefd55bc7b417c89d23d05b41076898728149d48f0faa63019c` |
| `artifacts/abc_v3/stage_b1b_mixed/checkpoint_latest.pt` | `e7662bd8f40cb9d1d302f1ec5aa384efff317c6e51ab271de56bff618c843851` |
