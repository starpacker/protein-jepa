#!/usr/bin/env bash
# v4g recipe applied to the HELD-OUT-TEMPERATURE split (T=450 unseen).
#
# The first temperature run (run_fromnone_temp.sh) used the OLD cold-start recipe
# (LR 3e-4, no resume, no tf-ca term) and exploding gradients killed it at step
# ~4500. This script reuses the mature v4g recipe (resume from the anchor
# checkpoint, LR 1e-4, --tf-ca-weight, ensemble-spread losses) but trains on the
# temperature split: T in {320,348,379,413}, T=450 held out entirely.
#
# Train/val proteins are DISJOINT (val_not450 has 100 held-out proteins), so this
# measures generalization to BOTH unseen proteins and an unseen temperature.
set -euo pipefail
cd /liuyutian/protein_jepa_frame
export TMPDIR=/tmp
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

OUTPUT=${OUTPUT:-artifacts/fromnone/stage_cb1b_temp_v2}
TRAIN=${TRAIN:-artifacts/mdcath_manifest_1000dom_temp/train_not450.jsonl}
VAL=${VAL:-artifacts/mdcath_manifest_1000dom_temp/val_not450.jsonl}
ANCHOR=${ANCHOR:-artifacts/fromnone/stage_cb1b_anchor/checkpoint_latest.pt}
mkdir -p "$OUTPUT"

/opt/conda/envs/b2t/bin/python -u -m protein_jepa_frame.training.train_stage_cb1b \
  --manifest "$TRAIN" --val-manifest "$VAL" \
  --output "$OUTPUT" \
  --resume "$ANCHOR" \
  --steps 70000 --warmup-steps 500 --lr 1e-4 --encoder-lr-scale 2.0 \
  --weight-decay 0.01 --grad-clip 1.0 \
  --batch-size 4 --num-workers 3 --max-residues 192 --device cuda \
  --history-frames 8 --multi-lag --frame-stride 1 --step-ps 1000 \
  --latent-dim 128 --encoder-depth 4 --encoder-heads 4 --dropout 0.1 \
  --orientation-features --relative-position-features --anchor-conditioned \
  --bath-dim 8 --energy-depth 3 --energy-heads 4 --energy-hidden 128 \
  --history-hidden 128 --coupling-scale 1e-3 --diffusion-scale 1e-4 \
  --c-init-std 0.1 --gamma-scale 2e-4 --gamma-floor 2e-5 --max-gamma-step 1.9 \
  --force-clip 1.0 --tail-init 1.0 \
  --decoder-hidden 256 --decoder-depth 3 --max-update 12.0 \
  --es-weight 0.1 \
  --ca-weight 1.0 \
  --tf-ca-weight 0.5 \
  --phys-weight 0.05 \
  --ens-rmsf-weight 0.2 \
  --ens-rmsf-corr-weight 0.8 \
  --ens-var-weight 0.3 \
  --ens-spread-weight 0.1 \
  --es-num-samples 4 \
  --ca-decode-steps 16 \
  --validation-every 500 --validation-batches 24 --validation-horizon-steps 16 \
  --save-every 2000 --log-every 50 --seed 43 \
  "$@" 2>&1 | tee "$OUTPUT/train.log"
