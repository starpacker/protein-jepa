#!/usr/bin/env bash
# Train the three S_0 arms (exact / prior / gru) at the ACTIVE bath scale.
#
# Identical to the lasp_full_v1 SOTA recipe except:
#   * --history-conditioning is ON for every arm (SOTA had it off, so its GRU
#     stayed at its zero init and never influenced anything),
#   * the bath sits where S is authoritative (x100 coupling, Gamma x0.1);
#     at the historical scale the bath contributes ~1e-8 of R and S_0 cannot
#     move a forecast no matter how it is initialised,
#   * --s0-mode selects the arm.
#
# Jobs run in WAVES of JOBS_PER_GPU per GPU.  The 2-GPU box is CPU-bound, not
# GPU-bound: one training process uses ~86% of a core and leaves the A800 at
# 10-25% utilization, so the default 2 per GPU nearly halves wall-clock without
# slowing any individual run meaningfully.  Set JOBS_PER_GPU=1 to restore the
# strictly one-per-GPU schedule.
#
# Usage: STEPS=7000 SEEDS="42 43 44" ARMS="exact prior gru" JOBS_PER_GPU=2 \
#          bash run_scripts/run_lasp_s0_arms.sh
set -euo pipefail
cd /liuyutian/protein_jepa_frame

STEPS=${STEPS:-7000}
SEEDS=${SEEDS:-"42 43 44"}
ARMS=${ARMS:-"exact prior gru"}
OUTROOT=${OUTROOT:-artifacts/lasp_s0_arms_20261031}
GPUS=${GPUS:-"0 1"}
JOBS_PER_GPU=${JOBS_PER_GPU:-2}
export TMPDIR=/tmp/lasp_review_runtime
export PYTHONPATH=/tmp/lasp_stability_deps:.
mkdir -p "$TMPDIR" "$OUTROOT"

read -r -a gpu_arr <<< "$GPUS"
read -r -a seed_arr <<< "$SEEDS"
read -r -a arm_arr <<< "$ARMS"
nper=$(( ${#gpu_arr[@]} * JOBS_PER_GPU ))
# The slot -> GPU map repeats each GPU JOBS_PER_GPU times.
gpu_slots=()
for g in "${gpu_arr[@]}"; do
  for ((k=0; k<JOBS_PER_GPU; k++)); do gpu_slots+=("$g"); done
done

jobs=()
for arm in "${arm_arr[@]}"; do
  for seed in "${seed_arr[@]}"; do
    jobs+=("$arm:$seed")
  done
done

run_job() {
  local arm=$1 seed=$2 gpu=$3 out=$4
  echo "=== $arm seed $seed on GPU $gpu -> $out ==="
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m protein_jepa_frame.training.train_lasp_ca_dynamics \
    --component artifacts/lasp_full_v1/representation/checkpoint_best.pt \
    --output "$out" \
    --collective-mobility --noise-mix 0 --noise-length 8 \
    --history-conditioning --s0-mode "$arm" \
    --coupling-scale 0.1 --gamma-scale 2e-5 --gamma-floor 2e-6 \
    --steps "$STEPS" --train-windows 0 --stateless-data --cosine --warmup 1500 \
    --val-windows 64 --val-every 500 \
    --n-samples 12 --grad-steps 64 --target path_mixture --max-residues 96 \
    --lr .0001 --latent-weight .1 --contact-objective brier \
    --contact-weight 20 --rmsf-weight 1 --clash-weight 40 --mode-weight 2 \
    --seed "$seed" > "$out.log" 2>&1
}

i=0
while [ $i -lt ${#jobs[@]} ]; do
  pids=()
  for ((slot=0; slot<nper && i<${#jobs[@]}; slot++)); do
    job=${jobs[$i]}
    arm=${job%%:*}; seed=${job##*:}
    gpu=${gpu_slots[$slot]}
    out="$OUTROOT/${arm}_seed${seed}"
    i=$((i+1))
    if [ -e "$out/metrics.jsonl" ]; then
      echo "skip $out (exists)"
      continue
    fi
    run_job "$arm" "$seed" "$gpu" "$out" &
    pids+=($!)
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  [ "$status" -eq 0 ] || echo "WARNING: a job in this wave failed" >&2
done

echo "=== all S_0 arms finished ==="
