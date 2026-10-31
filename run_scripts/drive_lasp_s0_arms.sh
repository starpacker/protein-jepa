#!/usr/bin/env bash
# Keep up to MAX_JOBS of the S_0 arm grid running until every arm is done.
#
# Unlike run_lasp_s0_arms.sh (fixed waves of JOBS_PER_GPU per GPU), this driver
# is slot-based: whenever a slot frees it starts the next missing arm, so a
# failed or already-finished arm never leaves a GPU idle.  It counts trainers
# that are ALREADY running, so it can be started mid-grid without double
# launching anything.
#
# Idempotent: an arm with checkpoint_best.pt is done; an arm with metrics.jsonl
# but no checkpoint is still running (or crashed) and is left to the slot scan.
#
# Usage: nohup setsid bash run_scripts/drive_lasp_s0_arms.sh > out.log 2>&1 &
set -uo pipefail
cd /liuyutian/protein_jepa_frame

STEPS=${STEPS:-7000}
OUTROOT=${OUTROOT:-artifacts/lasp_s0_arms_20261031}
GPUS=${GPUS:-"0 1"}
MAX_JOBS=${MAX_JOBS:-6}
export TMPDIR=/tmp/lasp_review_runtime
export PYTHONPATH=/tmp/lasp_stability_deps:.
mkdir -p "$TMPDIR" "$OUTROOT"
read -r -a gpu_arr <<< "$GPUS"

arms=(exact prior gru)
seeds=(42 43 44)

# An arm is DONE only when its latest checkpoint has reached the target step.
# checkpoint_best.pt is written at step 0 as well (val-selected), so its mere
# existence proves nothing and would hide a crashed run.
_ckpt_step() {
  [ -e "$1" ] || { echo -1; return; }
  python - "$1" <<'PY' 2>/dev/null || echo -1
import sys, torch
print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["step"]))
PY
}

all_done() {
  for arm in "${arms[@]}"; do for seed in "${seeds[@]}"; do
    local s
    s=$(_ckpt_step "$OUTROOT/${arm}_seed${seed}/checkpoint_latest.pt")
    [ "$s" -ge "$STEPS" ] || return 1
  done; done
  return 0
}

# PIDs of OUR trainers, anchored to the python command line so a shell command
# of mine that merely contains the pattern is never counted or matched.
_trainer_pids() {
  pgrep -f "^python -u -m protein_jepa_frame.training.train_lasp_ca_dynamics.*lasp_s0_arms" 2>/dev/null
}

# PID whose --output is exactly $1, or empty.
_pid_for_output() {
  local want=$1 pid out
  for pid in $(_trainer_pids); do
    out=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -o -- "--output [^ ]*" | cut -d' ' -f2)
    [ "$out" = "$want" ] && { echo "$pid"; return; }
  done
}

# Pick the GPU with the fewest of our trainers, read from each PID's own
# environment (nvidia-smi cannot tell which script a compute process belongs to).
pick_gpu() {
  local min=999 target=${gpu_arr[0]} g pid dev c
  for g in "${gpu_arr[@]}"; do
    c=0
    for pid in $(_trainer_pids); do
      dev=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')
      [ "$dev" = "$g" ] && c=$((c+1))
    done
    if [ "$c" -lt "$min" ]; then min=$c; target=$g; fi
  done
  echo "$target"
}

echo "=== driver start $(date -Is) MAX_JOBS=$MAX_JOBS ==="
while true; do
  if all_done; then echo "=== ALL ARMS DONE $(date -Is) ==="; break; fi

  running=$(echo $(_trainer_pids) | wc -w)
  if [ "$running" -lt "$MAX_JOBS" ]; then
    for arm in "${arms[@]}"; do
      for seed in "${seeds[@]}"; do
        [ "$running" -ge "$MAX_JOBS" ] && break 2
        out="$OUTROOT/${arm}_seed${seed}"
        [ "$(_ckpt_step "$out/checkpoint_latest.pt")" -ge "$STEPS" ] && continue
        # Already running?  A live trainer with this output dir occupies a slot.
        if [ -n "$(_pid_for_output "$out")" ]; then
          continue
        fi
        gpu=$(pick_gpu)
        echo "--- $(date -Is) start $arm seed $seed on GPU $gpu (running=$running)"
        CUDA_VISIBLE_DEVICES="$gpu" nohup python -u -m protein_jepa_frame.training.train_lasp_ca_dynamics \
          --component artifacts/lasp_full_v1/representation/checkpoint_best.pt --output "$out" \
          --collective-mobility --noise-mix .8 --noise-length 8 \
          --history-conditioning --s0-mode "$arm" \
          --coupling-scale 0.1 --gamma-scale 2e-5 --gamma-floor 2e-6 \
          --steps "$STEPS" --train-windows 0 --stateless-data --cosine --warmup 1500 \
          --val-windows 64 --val-every 500 --n-samples 12 --grad-steps 64 \
          --target path_mixture --max-residues 96 --lr .0001 --latent-weight .1 \
          --contact-objective brier --contact-weight 20 --rmsf-weight 1 \
          --clash-weight 40 --mode-weight 2 --seed "$seed" \
          >> "$out.log" 2>&1 &
        running=$((running+1))
      done
    done
  fi
  sleep 120
done
