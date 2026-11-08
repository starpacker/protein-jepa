#!/usr/bin/env bash
# Correct H-only control for v3 SOTA: v3 recipe with --bath-dim 0.
#
# This is run_lasp_v3_velocity.sh with exactly ONE change:
#   * --bath-dim 0  (disables the bath S; StageB1b use_bath=False -> S=None)
#
# Everything else is byte-identical to the accepted v3 SOTA
# (artifacts/lasp_v3_velocity/checkpoint_best.pt), so the ONLY axis of
# variation is "does the bath S add anything over H alone":
#   * --train-encoder --train-decoder (in-place, encoder/decoder lr-scale 0.1)
#   * --bath-coupling velocity (FDT-correct), coupling_scale default 1e-3
#   * --lr 5e-4, --steps 20000, --max-residues 192, --history-frames 4
#   * --multi-horizon --paired-horizons-frames 1 4 16 64 --multi-horizon-max-grad 16
#   * --rollout-steps 8, --n-samples 6, --per-sample-temperature, --roundtrip 0.5
#   * --seed 42 (primary, matches v3 SOTA) + 43/44 for seed variance
#
# NOTE (important, honest): v3's S is INERT at the default coupling_scale=1e-3
# (bath_var/R ~ 1e-8), so v3-SOTA vs v3-bath0 is expected to be nearly identical.
# This is the *as-shipped* ablation: it answers "does v3's configured S add
# anything", NOT "can a correctly-scaled S add anything".
#
# Usage: SEEDS="42 43 44" OUTROOT=artifacts/lasp_v3_h_only_20261107 \
#          bash run_scripts/run_lasp_v3_h_only.sh
set -euo pipefail
cd /liuyutian/protein_jepa_frame

OUTROOT=${OUTROOT:-artifacts/lasp_v3_h_only_20261107}
MANIFEST=${MANIFEST:-artifacts/mdcath_manifest_full/train.jsonl}
VAL=${VAL:-artifacts/mdcath_manifest_full/val.jsonl}
SEEDS=${SEEDS:-"42 43 44"}
GPUS=${GPUS:-"0 1"}
STEPS=${STEPS:-20000}
export TMPDIR=/tmp/lasp_review_runtime
export PYTHONPATH=/tmp/lasp_stability_deps:.
mkdir -p "$OUTROOT" "$TMPDIR"

read -r -a gpu_arr <<< "$GPUS"
read -r -a seed_arr <<< "$SEEDS"

run_job() {
  local seed=$1 gpu=$2 out=$3
  echo "=== v3 H-only seed $seed on GPU $gpu -> $out ==="
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m protein_jepa_frame.training.train_stage_b_hs \
    --manifest "$MANIFEST" --val-manifest "$VAL" \
    --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
    --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
    --output "$out" --device cuda --torch-threads 4 \
    --max-train-samples 200000 --max-residues 192 --frame-stride 1 \
    --history-frames 4 --n-samples 6 \
    --multi-horizon --paired-horizons-frames 1 4 16 64 --multi-horizon-max-grad 16 \
    --ens-rmsf-corr-weight 0.5 \
    --per-sample-temperature \
    --train-encoder --encoder-lr-scale 0.1 \
    --train-decoder --decoder-lr-scale 0.1 \
    --roundtrip-weight 0.5 --roundtrip-frames 2 \
    --rmsf-weight 1.0 --contact-weight 10.0 --dist-weight 0.3 --pos-weight 0.0 \
    --rollout-steps 8 \
    --bath-coupling velocity --max-step-norm 20 --bath-dim 0 \
    --energy-depth 3 --energy-heads 4 --energy-hidden 128 \
    --steps "$STEPS" --warmup-steps 200 --lr 5e-4 --weight-decay 0.01 --grad-clip 1.0 \
    --batch-size 4 --num-workers 4 --val-every 250 --val-batches 8 \
    --log-every 10 --save-every 500 --seed "$seed" \
    > "$out.log" 2>&1
}

i=0
while [ $i -lt ${#seed_arr[@]} ]; do
  pids=()
  for ((slot=0; slot<${#gpu_arr[@]} && i<${#seed_arr[@]}; slot++)); do
    seed=${seed_arr[$i]}
    gpu=${gpu_arr[$slot]}
    out="$OUTROOT/v3honly_seed${seed}"
    i=$((i+1))
    if [ -e "$out/checkpoint_best.pt" ]; then
      echo "skip $out (exists)"
      continue
    fi
    run_job "$seed" "$gpu" "$out" &
    pids+=($!)
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  [ "$status" -eq 0 ] || echo "WARNING: a job in this wave failed" >&2
done

echo "=== all v3 H-only arms finished ==="
