#!/usr/bin/env bash
# v4e: v4d (MLP decoder, warm start from anchor, ensemble losses) + RMSF-r (shape) loss.
# Motivation: §7 eval showed the amplitude-only ens_rmsf improved rmsf_mae but
# HURT rmsf_corr (RMSF profile shape).  This run adds --ens-rmsf-corr-weight, a
# differentiable 1 - Pearson(model_rmsf, true_rmsf), to recover profile shape
# while keeping v4d's gains on the amplitude/contact metrics.
set -euo pipefail
cd /liuyutian/protein_jepa_frame
export TMPDIR=/tmp
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

OUTPUT=${OUTPUT:-artifacts/fromnone/stage_cb1b_v4e}
RESUME=${RESUME:-artifacts/fromnone/stage_cb1b_anchor/checkpoint_latest.pt}
TRAIN=artifacts/mdcath_manifest_1000dom/train.jsonl
VAL=artifacts/mdcath_manifest_1000dom/val.jsonl
mkdir -p "$OUTPUT"

python -u -m protein_jepa_frame.training.train_stage_cb1b \
  --manifest "$TRAIN" --val-manifest "$VAL" \
  --output "$OUTPUT" \
  --resume "$RESUME" \
  --steps 70000 --warmup-steps 2000 --lr 2e-4 --encoder-lr-scale 2.0 \
  --weight-decay 0.01 --grad-clip 1.0 \
  --batch-size 4 --num-workers 6 --max-residues 192 --device cuda \
  --history-frames 8 --multi-lag --frame-stride 1 --step-ps 1000 \
  --latent-dim 128 --encoder-depth 4 --encoder-heads 4 --dropout 0.1 \
  --orientation-features --relative-position-features --anchor-conditioned \
  --bath-dim 8 --energy-depth 3 --energy-heads 4 --energy-hidden 128 \
  --history-hidden 128 --coupling-scale 1e-3 --diffusion-scale 1e-4 \
  --c-init-std 0.1 --gamma-scale 2e-4 --gamma-floor 2e-5 --max-gamma-step 1.9 \
  --force-clip 1.0 --tail-init 1.0 \
  --decoder-type mlp --decoder-hidden 256 --decoder-depth 3 --max-update 12.0 \
  --mp-depth 2 --mp-heads 4 \
  --es-weight 0.1 \
  --ca-weight 1.0 \
  --tf-ca-weight 0.5 \
  --phys-weight 0.05 \
  --ens-rmsf-weight 0.3 \
  --ens-rmsf-corr-weight 0.5 \
  --ens-var-weight 0.3 \
  --ens-spread-weight 0.1 \
  --es-num-samples 4 \
  --ca-decode-steps 8 \
  --validation-every 500 --validation-batches 24 --validation-horizon-steps 16 \
  --save-every 2000 --log-every 50 --seed 42 \
  "$@" 2>&1 | tee "$OUTPUT/train.log"
