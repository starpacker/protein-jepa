#!/usr/bin/env bash
# bioemu2 ZERO-SHOT baseline, 3 independent sampling replicates (seed 0,1,2).
# Deterministic: PYTHONHASHSEED=0 + cuDNN deterministic + fp32 (32-true).
# Each replicate is a full 200-window sweep; results are averaged as mean +/- std.
set -euo pipefail
cd /liuyutian/protein_jepa_frame
export TMPDIR=/tmp
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/octopus-runtime/cache/huggingface/hub}
export BIOEMU2_MSA_CACHE=${BIOEMU2_MSA_CACHE:-$PWD/baselines/bioemu2/msa_cache}
export PYTHONHASHSEED=0
CKPT=$PWD/baselines/bioemu2/checkpoints/bioemu2-preview/checkpoint.ckpt
mkdir -p "$BIOEMU2_MSA_CACHE" results/bioemu2_reps

for S in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$S python -u -m evaluation.baselines.run_bioemu2_zeroshot \
    --manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
    --horizons-ns 1 4 16 64 --n-samples 12 --max-windows 200 \
    --max-anchors-per-trajectory 4 --max-residues 256 \
    --num-steps 128 --device cuda --seed $S \
    --checkpoint-path "$CKPT" \
    --out results/bioemu2_reps/seed$S.json \
    > results/bioemu2_reps/seed$S.log 2>&1 &
done
wait
echo "all replicates done"
