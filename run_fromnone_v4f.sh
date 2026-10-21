#!/usr/bin/env bash
# v4f: from-scratch training with two targeted fixes + CRIDecoder
#
# Trained from scratch (no resume) so the encoder, kernel, and CRIDecoder
# all learn together under the corrected losses from step 0.
#
# Fix A (修复A) — MULTILAG_CYCLE_NS_1NS:
#   At step_ps=1000 the old cycle had 62.9% of batches as 1-frame rollouts
#   (0.1/0.2/0.5 ns entries all collapse to 1 frame), disabling all ensemble
#   losses via the T>=2 guard.  The new cycle keeps the same timescale coverage
#   but removes sub-1ns dead weight; T=1 fraction drops to 27.3% so ensemble
#   supervision fires on 72.7% of batches (vs 37.1% before).
#
# Fix B (修复B) — ensemble_rmsf_corr_loss rewrite:
#   Old: used _ensemble_stats cumulative variance (M-sample spatial spread at
#        each step t separately) — structurally different from the evaluator.
#   New: flatten M*T decoded frames per protein → Kabsch-align all onto
#        frame 0 → rmsf_profile, exactly matching align_ensemble+rmsf_profile
#        in run_eval_fromnone_ensemble.py.  Gradient now points in the direction
#        that actually improves the scored metric.
#
# Architecture: CRIDecoder (replaces MLP decoder of v4e)
#   Per-residue MLP cannot learn collective motion; CRIDecoder adds:
#     * residue self-attention with geometric edge bias (CA-distance RBF +
#       sequence separation RBF + cos(dH_i,dH_j))
#     * H<->S cross-attention at every layer (bath-H coupling)
#   These let the decoder see that residue i's displacement depends on all
#   other residues j — essential for capturing domain-level flexibility patterns
#   in the RMSF profile shape (rmsf_corr).
#
# NOTE: uses /opt/conda/envs/b2t python (torch 2.4.1+cu121) because the base
#       env torch was upgraded to 2.14.1+cu130 which is incompatible with the
#       570 driver (max CUDA 12.9).  Prepend b2t/bin to PATH before running.
set -euo pipefail
cd /liuyutian/protein_jepa_frame
export TMPDIR=/tmp
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

OUTPUT=${OUTPUT:-artifacts/fromnone/stage_cb1b_v4f}
TRAIN=artifacts/mdcath_manifest_1000dom/train.jsonl
VAL=artifacts/mdcath_manifest_1000dom/val.jsonl
mkdir -p "$OUTPUT"

python -u -m protein_jepa_frame.training.train_stage_cb1b \
  --manifest "$TRAIN" --val-manifest "$VAL" \
  --output "$OUTPUT" \
  --steps 70000 --warmup-steps 1000 --lr 2e-4 --encoder-lr-scale 2.0 \
  --weight-decay 0.01 --grad-clip 1.0 \
  --batch-size 4 --num-workers 3 --max-residues 192 --device cuda \
  --history-frames 8 --multi-lag --frame-stride 1 --step-ps 1000 \
  --latent-dim 128 --encoder-depth 4 --encoder-heads 4 --dropout 0.1 \
  --orientation-features --relative-position-features --anchor-conditioned \
  --bath-dim 8 --energy-depth 3 --energy-heads 4 --energy-hidden 128 \
  --history-hidden 128 --coupling-scale 1e-3 --diffusion-scale 1e-4 \
  --c-init-std 0.1 --gamma-scale 2e-4 --gamma-floor 2e-5 --max-gamma-step 1.9 \
  --force-clip 1.0 --tail-init 1.0 \
  --decoder-type cri --decoder-hidden 256 --decoder-depth 3 --max-update 12.0 \
  --mp-depth 2 --mp-heads 4 --hs-heads 4 \
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
  --save-every 2000 --log-every 50 --seed 42 \
  "$@" 2>&1 | tee "$OUTPUT/train.log"
