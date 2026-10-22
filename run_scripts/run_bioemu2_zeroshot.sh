#!/usr/bin/env bash
# bioemu2 (microsoft/bioemu2, bioemu2-preview) ZERO-SHOT dynamics baseline.
# Published weights used as-is: no fine-tuning, no mdCATH training data.
# Scored with this repo's ensemble protocol on the same 200 windows / horizons.
# See BIOEMU2_ZEROSHOT_BASELINE_20261020.md for the fairness caveats.
set -euo pipefail
cd /liuyutian/protein_jepa_frame
export TMPDIR=/tmp
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/octopus-runtime/cache/huggingface/hub}
export BIOEMU2_MSA_CACHE=${BIOEMU2_MSA_CACHE:-$PWD/baselines/bioemu2/msa_cache}

OUT=${OUT:-results/bioemu2_zeroshot.json}
CKPT=${CKPT:-$PWD/baselines/bioemu2/checkpoints/bioemu2-preview/checkpoint.ckpt}
mkdir -p "$(dirname "$OUT")" "$BIOEMU2_MSA_CACHE"

python -u -m evaluation.baselines.run_bioemu2_zeroshot \
  --manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
  --horizons-ns 1 4 16 64 \
  --n-samples 12 \
  --max-windows 200 \
  --max-anchors-per-trajectory 4 \
  --max-residues 256 \
  --num-steps 128 \
  --device cuda \
  --sampler edm \
  --checkpoint-path "$CKPT" \
  --out "$OUT" \
  "$@" 2>&1 | tee "${OUT%.json}.log"
