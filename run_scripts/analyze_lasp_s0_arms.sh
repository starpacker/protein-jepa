#!/usr/bin/env bash
# Step 6 of LASP_CORRECT_FRAMEWORK_PLAN_20261031.md: the pre-registered paired
# analysis of the S_0 arms.
#
# Per seed pair we compare, at exact window-key pairing:
#   exact vs prior   -- the primary question: does a history-conditioned S_0
#                       (full Kalman posterior) beat an uninformed S_0, with
#                       every other part of the recipe held fixed?
#   exact vs sota    -- does it beat the frozen baseline?
#   prior vs sota    -- is the active-bath retraining itself the whole story?
#   gru   vs prior   -- does the exact posterior beat the amortised history net?
#
# Then the 3-seed aggregation applies the decision rule fixed before training
# (WIN = 3-seed mean improvement > 0 AND >= 2/3 seeds one-sided lower bound > 0).
#
# SOTA and BioEmu2 are never re-evaluated: the frozen files are read as-is.
set -euo pipefail
cd /liuyutian/protein_jepa_frame

export TMPDIR=/tmp/lasp_review_runtime
export PYTHONPATH=/tmp/lasp_stability_deps:.
ARM_RES=${ARM_RES:-results/lasp_s0_arms_20261031}
SOTA_RES=${SOTA_RES:-results/sota_frozen_20261024_backup}
OUT=${OUT:-results/lasp_s0_arms_20261031}
mkdir -p "$OUT"

# seed -> the frozen SOTA seed it is paired against
sota_file() {
  case "$1" in
    42) echo "$SOTA_RES/dynamics_seed42_0.ensemble.json" ;;
    43) echo "$SOTA_RES/dynamics_seed43_1.ensemble.json" ;;
    44) echo "$SOTA_RES/dynamics_seed44_2.ensemble.json" ;;
    *)  echo "unknown seed $1" >&2; return 1 ;;
  esac
}

for seed in 42 43 44; do
  runs=()
  for arm in exact prior gru; do
    f="$ARM_RES/${arm}_seed${seed}.ensemble.json"
    [ -e "$f" ] || { echo "MISSING $f"; exit 3; }
    runs+=(--run "${arm}_seed${seed}=$f")
  done
  runs+=(--run "sota_seed${seed}=$(sota_file "$seed")")
  python -m scripts.compare_s0_arms "${runs[@]}" \
    --compare "exact_seed${seed}:prior_seed${seed}" \
    --compare "exact_seed${seed}:sota_seed${seed}" \
    --compare "prior_seed${seed}:sota_seed${seed}" \
    --compare "gru_seed${seed}:prior_seed${seed}" \
    --out "$OUT/paired_seed${seed}.json"
done

# 3-seed aggregation, one report per comparison
pairs=()
for comp in exact_vs_prior exact_vs_sota prior_vs_sota gru_vs_prior; do
  for seed in 42 43 44; do
    pairs+=(--pair "${comp}=s${seed}=$OUT/paired_seed${seed}.json")
  done
done
PYTHONPATH=/tmp/lasp_stability_deps:. /opt/conda/bin/python -m scripts.report_s0_arms "${pairs[@]}" --out "$OUT/decision_3seed.json"
