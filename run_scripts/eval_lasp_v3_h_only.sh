#!/usr/bin/env bash
# Evaluate the Correct H-only control (v3 recipe, bath_dim=0) and compare vs v3 SOTA.
#
# v3 SOTA is the H+S arm (artifacts/lasp_v3_velocity/checkpoint_best.pt, seed 42).
# H-only is the SAME recipe with --bath-dim 0 (artifacts/lasp_v3_h_only_20261107).
#
# Both use the v3 eval entries (v3 checkpoints have checkpoint_kind=None, no
# component_checkpoint, so the exact-arm noisemix0 entries DO NOT apply):
#   1. ensemble   -> scripts/eval_lasp_final.py   (extended_ensemble_summary + Model1 + persist)
#   2. trajectory -> scripts/eval_ca_ordered_paths_v3.py (ACF Error / Path Error / Latent Stable)
#
# Usage: OUTROOT=results/v3_h_only_20261107 bash run_scripts/eval_lasp_v3_h_only.sh
set -euo pipefail
cd /liuyutian/protein_jepa_frame

OUTROOT=${OUTROOT:-results/v3_h_only_20261107}
MANIFEST=${MANIFEST:-artifacts/mdcath_manifest_full/val.jsonl}
SEEDS=${SEEDS:-42 43 44}
GPU=${GPU:-0}
N_SAMPLES=${N_SAMPLES:-12}
STAGE_A=artifacts/stage_a_disp_relpos/checkpoint_latest.pt
B1A=artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt
ENSEMBLE_CKPT=artifacts/stage_ensemble_1000dom/checkpoint_best.pt
mkdir -p "$OUTROOT"

export CUDA_VISIBLE_DEVICES="$GPU"

for seed in $SEEDS; do
  ckpt="artifacts/lasp_v3_h_only_20261107/v3honly_seed${seed}/checkpoint_best.pt"
  [ -e "$ckpt" ] || { echo "MISSING checkpoint $ckpt (training not done?)"; continue; }

  echo "=== v3 H-only seed${seed} ensemble eval (eval_lasp_final) ==="
  python -u -m scripts.eval_lasp_final \
    --kernel-ckpt "$ckpt" \
    --stage-a-ckpt "$STAGE_A" --b1a-ckpt "$B1A" \
    --ensemble-ckpt "$ENSEMBLE_CKPT" \
    --manifest "$MANIFEST" \
    --horizons-ns 1 4 16 64 \
    --max-windows 200 --n-samples "$N_SAMPLES" --seed 0 --max-residues 96 \
    --out "$OUTROOT/v3honly_seed${seed}.ensemble.json" 2>&1 | tail -3

  echo "=== v3 H-only seed${seed} trajectory eval (eval_ca_ordered_paths_v3) ==="
  python -u scripts/eval_ca_ordered_paths_v3.py \
    --kernel-ckpt "$ckpt" \
    --stage-a-ckpt "$STAGE_A" --b1a-ckpt "$B1A" \
    --manifest "$MANIFEST" \
    --out "$OUTROOT/v3honly_seed${seed}.ordered.json" \
    --horizons 1 4 16 64 2>&1 | tail -3
done

echo "=== v3 H-only eval complete -> $OUTROOT ==="
