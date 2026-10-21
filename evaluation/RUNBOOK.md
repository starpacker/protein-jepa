# Repaired evaluation runbook

Read [README.md](README.md) for schema, metrics and diagnostic/headline separation.
Historical outputs under `artifacts/eval/smoke` and the old result documents were
produced with invalid pairing/lineage contracts and are not scientific benchmarks.
Do not overwrite them to suggest continuity with repaired results.

1. Train Stage A with the current delta schema and pass representation calibration.
2. Use a schema-v2 B1a oracle bundle. A fine-tuned encoder requires an adapted or
   retrained direct predictor; the loader rejects an unchanged incompatible head.
3. Pass the held-out oracle gate before formal B1b training; preserve its report.
4. Run `python -m evaluation.run_eval --help`, then explicitly supply checkpoint,
   training and validation manifest paths. Configs contain repaired pilot example
   locations, not promises that completed files already exist.
5. Inspect `headline_eligible`, representation IDs and the split audit before
   interpreting the equal-protein table. Contract smoke, mock and legacy output
   are diagnostics only.
6. Use a new output directory for every locked experiment. Retain full window,
   sample and protein tables, effective config, source/checkpoint hashes and seed.
7. Once the pilot passes, freeze acceptance thresholds and evaluate untouched test
   proteins with multiple independent training seeds and physical trajectory scores.

A filtered 450 K validation set is a temperature subset. A temperature-holdout run
requires all train-time modules to exclude that temperature, and the matching
training manifests must be passed to the runner for audit.
