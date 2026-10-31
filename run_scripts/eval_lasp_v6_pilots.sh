#!/usr/bin/env bash
set -euo pipefail
cd /liuyutian/protein_jepa_frame
mkdir -p results/lasp_v6
export CUDA_VISIBLE_DEVICES=${GPU:-2}
for model in v3 path history; do
  case "$model" in
    v3) ck=artifacts/lasp_v6_pilot_input/v3_best.pt ;;
    path) ck=artifacts/lasp_v6_path_pilot/checkpoint_best.pt ;;
    history) ck=artifacts/lasp_v6_history_pilot/checkpoint_best.pt ;;
  esac
  for mode in endpoint path; do
    extra=()
    if [ "$mode" = path ]; then extra=(--path-ensemble); fi
    python -u scripts/eval_lasp_final.py \
      --kernel-ckpt "$ck" --ensemble-ckpt artifacts/stage_ensemble_1000dom/checkpoint_best.pt \
      --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
      --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
      --manifest artifacts/mdcath_manifest_full/val.jsonl --device cuda \
      --history-frames 4 --max-residues 96 --n-samples 6 --max-windows 40 \
      --max-windows-per-protein 4 --seed 0 "${extra[@]}" \
      --out "results/lasp_v6/${model}_${mode}.json"
  done
done
for model in path history; do
  for mode in endpoint path; do
    python -m scripts.compare_paired_ensembles \
      "results/lasp_v6/${model}_${mode}.per_window.json" \
      "results/lasp_v6/v3_${mode}.per_window.json" --reference-field lasp \
      --out "results/lasp_v6/${model}_${mode}_vs_v3.json"
  done
done
