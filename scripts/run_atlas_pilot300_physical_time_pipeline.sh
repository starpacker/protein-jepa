#!/usr/bin/env bash
set -euo pipefail

FRAME_REPO=/liuyutian/protein_jepa_frame
PY=/liuyutian/protein_jepa/.venv/bin/python

CONFIG="$FRAME_REPO/configs/mdcath_atlas_pilot300_state_jepa_multilag_physical_time_scratch20k.yaml"
MIXED_MANIFEST="$FRAME_REPO/artifacts/manifests/mdcath_atlas_pilot300.train.jsonl"
RUN_DIR="$FRAME_REPO/artifacts/mdcath_atlas_pilot300_state_jepa_multilag_physical_time_scratch20k"
EVAL_DIR="$FRAME_REPO/artifacts/paper_eval/pipeline_aligned24_mdcath_atlas_pilot300_physical_time_contextual"

if [[ ! -s "$MIXED_MANIFEST" ]]; then
  echo "missing mixed manifest: $MIXED_MANIFEST" >&2
  echo "run scripts/run_atlas_pilot300_pipeline.sh first to materialize ATLAS pilot300" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

cd "$FRAME_REPO"

echo "[$(date -Is)] smoke-load mixed mdCATH+ATLAS manifest"
PYTHONPATH="$FRAME_REPO" "$PY" - <<'PY'
from collections import Counter

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.dataset import TrajectoryClipDataset
from protein_jepa_frame.data.manifest import load_manifest

records = load_manifest("artifacts/manifests/mdcath_atlas_pilot300.train.jsonl")
dataset = TrajectoryClipDataset(
    records,
    DataConfig(num_frames=24, frame_stride=2, window_stride=8, max_residues=384),
)
counts = Counter(record.dataset for record in records)
print({"records": len(records), "records_by_dataset": dict(counts), "clips": len(dataset)}, flush=True)
PY

echo "[$(date -Is)] train physical-time mixed scratch 20k"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$FRAME_REPO" "$PY" -m protein_jepa_frame.cli train --config "$CONFIG"

echo "[$(date -Is)] evaluate physical-time mixed scratch 20k"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$FRAME_REPO" "$PY" -m protein_jepa_frame.eval.paper_eval pipeline \
  --checkpoint baseline20=artifacts/mdcath_state_jepa_multilag/checkpoint_step20000.pt \
  --checkpoint atlas_pilot300=artifacts/mdcath_atlas_pilot300_state_jepa_multilag_scratch20k/checkpoint_latest.pt \
  --checkpoint atlas_pilot300_physical_time=artifacts/mdcath_atlas_pilot300_state_jepa_multilag_physical_time_scratch20k/checkpoint_latest.pt \
  --output-dir "$EVAL_DIR" \
  --preset quick \
  --readout state \
  --sequence-frames 24 \
  --window-stride 8 \
  --linear-window-stride 8 \
  --require-contextual

echo "[$(date -Is)] done"
