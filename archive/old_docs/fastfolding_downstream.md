# FastFolding Downstream Benchmark

FastFolding is treated as a read-only benchmark. Do not write manifests, labels,
features, checkpoints, or temporary files under the source data directory.

Observed local source path:

```text
/liuyutian/folddata/FastFolding
```

The user-provided `/share/liuyutian/folddata/FastFolding` path was not mounted
in this runtime. If it is available in another environment, pass it with
`--root`.

## Build a Read-Only Manifest

```bash
cd /liuyutian/protein_jepa_frame
PYTHONPATH=/liuyutian/protein_jepa_frame \
/liuyutian/protein_jepa/.venv/bin/python -m protein_jepa_frame.cli fastfolding-manifest \
  --root /liuyutian/folddata/FastFolding \
  --output artifacts/fastfolding/manifest.jsonl
```

The manifest records protein directories, reference PDBs, and frame count
metadata. It does not modify the original data.

For shared filesystem smoke tests, scan a bounded subset first:

```bash
PYTHONPATH=/liuyutian/protein_jepa_frame \
/liuyutian/protein_jepa/.venv/bin/python -m protein_jepa_frame.cli fastfolding-manifest \
  --root /liuyutian/folddata/FastFolding \
  --protein CLN025-0-protein \
  --max-scan-per-protein 50 \
  --output artifacts/fastfolding/manifest.cln025.jsonl
```

## Derive Frame-Level Labels

```bash
PYTHONPATH=/liuyutian/protein_jepa_frame \
/liuyutian/protein_jepa/.venv/bin/python -m protein_jepa_frame.cli fastfolding-labels \
  --manifest artifacts/fastfolding/manifest.jsonl \
  --output artifacts/fastfolding/frame_labels.jsonl \
  --stride 100 \
  --max-frames-per-protein 2000 \
  --max-scan-entries-per-protein 200000
```

Derived labels include:

- `rmsd_to_ref`: Kabsch-aligned CA RMSD to the reference PDB.
- `q_native`: fraction of native CA contacts present in the frame.
- `radius_of_gyration`: CA radius of gyration.
- `state`: folded / transition / unfolded based on Q thresholds.
- `direction`: folding / stationary / unfolding based on local `dQ/dt`.
- `committor_proxy`: whether the future window reaches folded before unfolded.

For quick smoke:

```bash
PYTHONPATH=/liuyutian/protein_jepa_frame \
/liuyutian/protein_jepa/.venv/bin/python -m protein_jepa_frame.cli fastfolding-labels \
  --manifest artifacts/fastfolding/manifest.cln025.jsonl \
  --output artifacts/fastfolding/labels.cln025.smoke.jsonl \
  --stride 10 \
  --max-frames-per-protein 20 \
  --max-scan-entries-per-protein 200
```

## Run Contextual Latent Probes

```bash
PYTHONPATH=/liuyutian/protein_jepa_frame \
/liuyutian/protein_jepa/.venv/bin/python -m protein_jepa_frame.cli fastfolding-probe \
  --checkpoint state_jepa_20k=artifacts/mdcath_state_jepa_future/checkpoint_step20000.pt \
  --labels artifacts/fastfolding/labels.cln025.smoke.jsonl \
  --output artifacts/fastfolding/probe.cln025.smoke.json \
  --num-frames 8 \
  --readout state \
  --max-samples 200
```

The probe builds temporal PDB windows ending at labeled frames, extracts model
latents from the chosen readout, and evaluates:

- progress regression for `q_native`, `rmsd_to_ref`, and `radius_of_gyration`.
- direction classification for folding vs unfolding frames.
- committor proxy classification when future basin labels are available.

These labels are intended for downstream probes:

- folding progress regression: RMSD, Q, and Rg.
- direction-aware classification: folding vs unfolding near similar structures.
- committor prediction: future basin reachability.
- transition-path retrieval: nearest neighbors should share dynamics context,
  not only geometry.

## Modeling Implication

FastFolding should primarily evaluate contextual state latents. A single-frame
geometry latent cannot distinguish folding and unfolding paths that pass through
similar conformations. Downstream probes should therefore prefer state readouts
computed from temporal windows.

## Dynamics Benchmark Runner

The paper-oriented event benchmark is implemented in
`protein_jepa_frame.fastfolding.benchmark` and
`scripts/evaluate_dynamics_benchmarks.py`. It assigns numbered replicas to a
shared scaffold group, keeps each group in exactly one split, and evaluates
state forecasts and transition operators separately at every horizon.

Run the persistence baseline first:

```bash
PYTHONPATH=/liuyutian/protein_jepa_frame \
python scripts/evaluate_dynamics_benchmarks.py fastfolding \
  --labels artifacts/fastfolding/frame_labels.jsonl \
  --output artifacts/fastfolding/persistence_benchmark.json \
  --context-frames 8 \
  --horizons 1 4 8 16
```

To score a model adapter, provide a JSONL prediction file with one row per
sample id:

```json
{"sample_id":"CLN025-0-protein|CLN025-0-protein|800|8",
 "state_probs":[0.15,0.25,0.60],
 "committor_probability":0.73}
```

The state order is `unfolded transition folded` by default. The prediction file
must contain model outputs only; future labels are joined internally from the
held-out label file. The output reports per-horizon NLL, Brier score, macro-F1,
calibration error, committor-proxy metrics, and transition-operator metrics
including stationary distance, implied timescales, and MFPT error.

This runner evaluates event and operator predictions. It does not treat a
linear probe over frozen latents as a generated path. A future model adapter
should therefore emit state probabilities from the actual transition model or
from an explicitly declared sampled rollout, and should add path samples for
the transition-path ensemble benchmark.

For checkpoints with `paired_forecast=True`, the repository includes a causal
transition-head adapter:

```bash
PYTHONPATH=/liuyutian/protein_jepa_frame \
python scripts/generate_fastfolding_predictions.py \
  --checkpoint model=/path/to/checkpoint.pt \
  --labels artifacts/fastfolding/frame_labels.jsonl \
  --output artifacts/fastfolding/model_predictions.jsonl \
  --context-frames 8 \
  --horizons 1 4 8 16

PYTHONPATH=/liuyutian/protein_jepa_frame \
python scripts/evaluate_dynamics_benchmarks.py fastfolding \
  --labels artifacts/fastfolding/frame_labels.jsonl \
  --predictions artifacts/fastfolding/model_predictions.jsonl \
  --output artifacts/fastfolding/model_benchmark.json
```

The adapter fits only a train-scaffold readout from the predicted future
latent state to the three FastFolding states. It is therefore a transition-head
forecast evaluation, not a claim that the latent itself has a unique physical
state interpretation.
