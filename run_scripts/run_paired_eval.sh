#!/bin/bash
# Apples-to-apples paired eval: Plan A (terminal-only) vs dense Plan A, both
# against the SAME frozen Model 1. Global RNG is seeded inside the script so the
# Model1 baseline is identical across the two invocations.
set -e
cd /liuyutian/protein_jepa_frame
export TMPDIR=/liuyutian/protein_jepa_frame/.tmpwork
GPU=${GPU:-5}
MAN=artifacts/mdcath_manifest_1000dom/val.jsonl
COMMON=(--manifest "$MAN" --device cuda \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --ensemble-ckpt artifacts/stage_ensemble_1000dom/checkpoint_best.pt \
  --history-frames 4 --frame-stride 1 --horizons 8 \
  --max-residues 192 --n-samples 8 --batches 40 --batch-size 1 \
  --num-workers 2 --seed 0)

for RUN in plan_a_hs plan_a_dense; do
  if [ "$RUN" = plan_a_hs ]; then CK=artifacts/plan_a_hs_1000dom/checkpoint_best.pt; else CK=artifacts/plan_a_dense_1000dom/checkpoint_best.pt; fi
  echo "===== $RUN ====="
  CUDA_VISIBLE_DEVICES=$GPU python -u -m scripts.paired_hs_vs_model1 \
    --kernel-ckpt "$CK" --out "results/${RUN}_vs_model1.json" \
    "${COMMON[@]}" 2>&1 | tee ".tmpwork/logs/eval_${RUN}.log"
done
