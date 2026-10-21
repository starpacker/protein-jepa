#!/bin/bash
# Direction 3: train B1a oracle decoder on the BEST Stage-A encoder
# (stage_a_disp_relpos: struct_recon~0.02, position R2~0.13).
# Metric to watch: b1a/diag_ca_rmsd_real vs _zeroH/_persist. Goal: <1.2 A.
#
# Overfit probe (.tmpwork/b1a_overfit_probe.log) proved the decoder DOES
# converge on real mdCATH once past its slow start (ca_mse 139->0.47 in 300
# steps); earlier "flat ~200" observations were simply too early. Use a short
# warmup so the breakthrough is reached quickly, then ride it down.
set -e
cd /liuyutian/protein_jepa_frame
export TMPDIR=/liuyutian/protein_jepa_frame/.tmpwork
OUT=artifacts/stage_b1a_relpos
mkdir -p "$OUT" .tmpwork/logs
CUDA_VISIBLE_DEVICES=${GPU:-5} python -u -m protein_jepa_frame.training.train_stage_b1a \
  --manifest artifacts/mdcath_manifest_200dom/train.jsonl \
  --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --output "$OUT" --device cuda --torch-threads 4 \
  --steps 10000 --warmup-steps 150 --lr 0.0005 --weight-decay 0.01 --grad-clip 1.0 \
  --batch-size 4 --num-workers 6 --history-frames 4 --frame-stride 2 \
  --max-residues 256 --paired-horizons-frames 4 8 16 24 32 48 \
  --unfreeze-top-n-blocks 0 --encoder-lr-scale 0.1 \
  --decoder-hidden 128 --decoder-blocks 4 --decoder-heads 4 \
  --anchor-free --coord-scale 40.0 \
  --fape-weight 1.0 --torsion-weight 0.5 --distance-weight 0.2 \
  --ca-mse-weight 1.0 --phys-future-weight 1.0 --anchor-weight 0.1 \
  --diag-zero-h --val-every 500 --val-batches 8 --fit-normalizer-batches 20 \
  --log-every 10 --save-every 250 --seed 42 \
  2>&1 | tee .tmpwork/logs/stage_b1a_relpos.log
