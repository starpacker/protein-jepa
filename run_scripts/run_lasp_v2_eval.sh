#!/bin/bash
# Evaluate a Lasp checkpoint on the MULTI-PROTEIN protocol and compare against
# bioemu2's own multi-protein summary.
#
# Round-robin coverage is the default now, so a 200-window budget spans 50
# proteins x 5 temperatures x 4 horizons instead of ONE 60-residue domain.
# Confidence intervals are protein-clustered bootstraps.
set -euo pipefail
cd /liuyutian/protein_jepa_frame
CKPT=${CKPT:?set CKPT to a Lasp kernel checkpoint}
OUT=${OUT:?set OUT to a results json path}
GPU=${GPU:-3}
MAXW=${MAXW:-200}
CUDA_VISIBLE_DEVICES=$GPU python -u scripts/eval_lasp_final.py \
  --kernel-ckpt "$CKPT" \
  --ensemble-ckpt artifacts/stage_ensemble_1000dom/checkpoint_best.pt \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
  --device cuda --horizons-ns 1 4 16 64 --n-samples 12 \
  --max-windows "$MAXW" --max-windows-per-protein 8 \
  --out "$OUT" 2>&1 | grep -v "UserWarning\|_transformer\|return data" | tail -70
