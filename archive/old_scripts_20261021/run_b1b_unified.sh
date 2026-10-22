#!/usr/bin/env bash
# History-only autonomous H+S. A failed/mismatched oracle gate stops training.
set -euo pipefail
cd /liuyutian/protein_jepa_frame
export TMPDIR=/tmp
OUTPUT=${OUTPUT:-artifacts/repaired_v2/full/stage_b1b}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-artifacts/mdcath_manifest_200dom/train.jsonl}
VAL_MANIFEST=${VAL_MANIFEST:-artifacts/mdcath_manifest_200dom/val.jsonl}
B1A_CKPT=${B1A_CKPT:-artifacts/repaired_v2/full/stage_b1a/checkpoint_best.pt}
ORACLE_GATE_REPORT=${ORACLE_GATE_REPORT:-artifacts/repaired_v2/full/h_gate/report.json}
mkdir -p "$OUTPUT"
python -u -m protein_jepa_frame.training.train_stage_b1b \
  --manifest "$TRAIN_MANIFEST" --val-manifest "$VAL_MANIFEST" \
  --b1a-ckpt "$B1A_CKPT" --oracle-gate-report "$ORACLE_GATE_REPORT" --output "$OUTPUT" \
  --steps 20000 --warmup-steps 500 --batch-size 8 --num-workers 0 \
  --history-frames 8 --horizon-steps 8 --frame-stride 1 --step-ps 1000 \
  --bath-dim 8 --energy-depth 3 --nll-weight 0 --es-weight 1 \
  --physical-es-weight 0.1 --es-num-samples 4 --rmsd-weight 0 \
  --lr 0.0001 --weight-decay 0.01 --grad-clip 1 \
  --validation-every 200 --validation-batches 16 --validation-horizon-steps 32 \
  --log-every 20 --save-every 1000 "$@" 2>&1 | tee "$OUTPUT/train.log"
