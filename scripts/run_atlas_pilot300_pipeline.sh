#!/usr/bin/env bash
set -euo pipefail

FRAME_REPO=/liuyutian/protein_jepa_frame
OLD_REPO=/liuyutian/protein_jepa
PY=/liuyutian/protein_jepa/.venv/bin/python

RAW_MANIFEST="$FRAME_REPO/artifacts/manifests/atlas.raw.pilot300.jsonl"
CACHED_DIR="$FRAME_REPO/artifacts/cache/atlas_h5_pilot300"
CACHED_MANIFEST="$FRAME_REPO/artifacts/manifests/atlas.cached.pilot300.jsonl"
MDCATH_MANIFEST=/liuyutian/protein_jepa/artifacts/manifests/mdcath.train.jsonl
MIXED_MANIFEST="$FRAME_REPO/artifacts/manifests/mdcath_atlas_pilot300.train.jsonl"
CONFIG="$FRAME_REPO/configs/mdcath_atlas_pilot300_state_jepa_multilag_scratch20k.yaml"
RUN_DIR="$FRAME_REPO/artifacts/mdcath_atlas_pilot300_state_jepa_multilag_scratch20k"
EVAL_DIR="$FRAME_REPO/artifacts/paper_eval/pipeline_aligned24_mdcath_atlas_pilot300_contextual"

mkdir -p "$FRAME_REPO/artifacts/manifests" "$CACHED_DIR" "$RUN_DIR"

echo "[$(date -Is)] materialize ATLAS pilot300"
cd "$OLD_REPO"
"$PY" -m protein_jepa.cli materialize-cache \
  "$RAW_MANIFEST" \
  "$CACHED_DIR" \
  "$CACHED_MANIFEST" \
  --config "$OLD_REPO/configs/phase1.yaml" \
  --chunk-frames 128

echo "[$(date -Is)] build mixed mdCATH+ATLAS manifest"
cd "$FRAME_REPO"
cat "$MDCATH_MANIFEST" "$CACHED_MANIFEST" > "$MIXED_MANIFEST"
wc -l "$MDCATH_MANIFEST" "$CACHED_MANIFEST" "$MIXED_MANIFEST"

echo "[$(date -Is)] smoke-load mixed manifest"
PYTHONPATH="$FRAME_REPO" "$PY" - <<'PY'
from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.dataset import TrajectoryClipDataset
from protein_jepa_frame.data.manifest import load_manifest

records = load_manifest("artifacts/manifests/mdcath_atlas_pilot300.train.jsonl")
dataset = TrajectoryClipDataset(
    records,
    DataConfig(num_frames=24, frame_stride=2, window_stride=8, max_residues=384),
)
sample = dataset[len(dataset) - 1]
print(
    {
        "records": len(records),
        "clips": len(dataset),
        "last_sample_shape": tuple(sample["coords"].shape),
        "last_sample_dataset_traj": sample["trajectory_id"],
    },
    flush=True,
)
PY

echo "[$(date -Is)] train mixed scratch 20k"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$FRAME_REPO" "$PY" -m protein_jepa_frame.cli train --config "$CONFIG"

echo "[$(date -Is)] evaluate mixed scratch 20k"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$FRAME_REPO" "$PY" -m protein_jepa_frame.eval.paper_eval pipeline \
  --checkpoint baseline20=artifacts/mdcath_state_jepa_multilag/checkpoint_step20000.pt \
  --checkpoint atlas_pilot300=artifacts/mdcath_atlas_pilot300_state_jepa_multilag_scratch20k/checkpoint_latest.pt \
  --output-dir "$EVAL_DIR" \
  --preset quick \
  --readout state \
  --sequence-frames 24 \
  --window-stride 8 \
  --linear-window-stride 8 \
  --require-contextual

echo "[$(date -Is)] done"
