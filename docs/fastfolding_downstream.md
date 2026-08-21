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
