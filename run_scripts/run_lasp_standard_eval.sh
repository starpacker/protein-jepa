#!/usr/bin/env bash
# STANDARD evaluation for an autonomous CA (LASP) checkpoint.
#
# From now on every evaluation of a dynamics checkpoint goes through this script,
# so the following are always reported together instead of opportunistically:
#
#   1. the harmonized BioEmu2-protocol ensemble metrics on 200 windows
#      (rmsf_corr / rmsf_mae / root_mean_w2 / variance_w2 / md_pca_w2 /
#       weak_contact_j / transient_contacts_j, plus geometry and position),
#   2. the per-temperature sweep and the temperature-conditioning matrix
#      correct / wrong / none -- the mechanism test for "is temperature used?",
#   3. the BioEmu-style static-ensemble metrics (T2A): reference-state coverage,
#      free-energy MAE (kcal/mol), population MAE / TV / JSD.
#
# Definitions: EVALUATION_METHODS_AND_COMMANDS_20261021.md 2.6 (T2A) and 2.9 (T3).
# All of it is a development diagnostic on the val split; no test claim.
#
# Usage:
#   bash run_scripts/run_lasp_standard_eval.sh <tag> <ckpt> [<ckpt> ...]
#
# e.g.
#   bash run_scripts/run_lasp_standard_eval.sh round2 \
#       artifacts/lasp_full_v1/dynamics_seed42/checkpoint_best.pt ...
set -euo pipefail
cd /liuyutian/protein_jepa_frame

TAG=${1:?usage: run_lasp_standard_eval.sh <tag> <ckpt> [<ckpt> ...]}
shift
[ "$#" -ge 1 ] || { echo "need at least one checkpoint" >&2; exit 2; }

export TMPDIR=/tmp
OUTDIR=${OUTDIR:-results/standard_eval/$TAG}
GPUS=${GPUS:-"3 4 6"}
PROTEINS_PER_TEMP=${PROTEINS_PER_TEMP:-12}
ANCHORS=${ANCHORS:-2}
N_SAMPLES=${N_SAMPLES:-12}
WINDOWS=${WINDOWS:-200}
mkdir -p "$OUTDIR"

read -r -a gpu_arr <<< "$GPUS"
ckpts=("$@")
pids=()
for i in "${!ckpts[@]}"; do
  ck="${ckpts[$i]}"
  gpu=${gpu_arr[$((i % ${#gpu_arr[@]}))]}
  name=$(basename "$(dirname "$ck")")_$i
  echo "=== $name: harmonized 200-window ensemble metrics (GPU $gpu) ==="
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m scripts.eval_lasp_ca_dynamics \
    --checkpoint "$ck" --out "$OUTDIR/$name.ensemble.json" \
    --manifest artifacts/mdcath_manifest_full/val.jsonl \
    --max-windows "$WINDOWS" --n-samples "$N_SAMPLES" \
    > "$OUTDIR/$name.ensemble.log" 2>&1 &
  pids+=($!)

  echo "=== $name: per-temperature + T3 matrix + T2A (GPU $gpu) ==="
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m scripts.eval_lasp_ca_suite \
    --checkpoint "$ck" --out "$OUTDIR/$name.suite.json" --t2a \
    --modes correct wrong none --wrong-temp 320 \
    --proteins-per-temp "$PROTEINS_PER_TEMP" \
    --anchors-per-trajectory "$ANCHORS" --n-samples "$N_SAMPLES" \
    --horizons 1 4 16 64 --max-residues 96 \
    > "$OUTDIR/$name.suite.log" 2>&1 &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
echo "=== evaluation jobs finished (status=$status) ==="
if [ "$status" -ne 0 ]; then
  echo "ERROR: evaluation failed; refusing to publish reports or completion manifest" >&2
  exit "$status"
fi

# Exact BioEmu2 comparison protocol. Run one checkpoint at a time only after
# every full-suite child has exited, so this adds no concurrent GPU load.
for i in "${!ckpts[@]}"; do
  ck="${ckpts[$i]}"
  gpu=${gpu_arr[$((i % ${#gpu_arr[@]}))]}
  name=$(basename "$(dirname "$ck")")_$i
  echo "=== $name: matched8 correct-only T2A (GPU $gpu) ==="
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m scripts.eval_lasp_ca_suite \
    --checkpoint "$ck" --out "$OUTDIR/$name.matched8.json" --t2a \
    --manifest artifacts/mdcath_manifest_full/val.jsonl \
    --modes correct --proteins-per-temp 8 --anchors-per-trajectory 1 \
    --n-samples 12 --context-frames 8 --horizons 1 4 16 64 --max-residues 96 \
    > "$OUTDIR/$name.matched8.log" 2>&1
done

python -u -m scripts.report_lasp_ca_suite \
  "$OUTDIR"/*.suite.json --out "$OUTDIR/suite_report.json" > "$OUTDIR/report.txt"

# T2A reference anchors (oracle / persistence). Model-free: reads true MD frames
# only, so it can run after the samplers have released the GPUs. The anchor
# protocol must match the suite's, otherwise the numbers are not comparable.
if [ "${SKIP_ANCHORS:-0}" != "1" ]; then
  echo "=== T2A anchors (oracle / persistence) ===" | tee -a "$OUTDIR/report.txt"
  python -u -m scripts.t2a_baselines \
    --out "$OUTDIR/t2a_anchors.json" \
    --proteins-per-temp "$PROTEINS_PER_TEMP" \
    --anchors-per-trajectory "$ANCHORS" --max-residues 96 \
    2>&1 | tee -a "$OUTDIR/report.txt"
fi

cat "$OUTDIR/report.txt"

TAG="$TAG" STATUS="$status" WINDOWS="$WINDOWS" N_SAMPLES="$N_SAMPLES" \
PROTEINS_PER_TEMP="$PROTEINS_PER_TEMP" ANCHORS="$ANCHORS" OUTDIR="$OUTDIR" \
python - "${ckpts[@]}" <<'PY'
import json, os, sys
from pathlib import Path
out = Path(os.environ['OUTDIR'])
out.joinpath('manifest.json').write_text(json.dumps({
    'tag': os.environ['TAG'],
    'checkpoints': sys.argv[1:],
    'windows': int(os.environ['WINDOWS']),
    'n_samples': int(os.environ['N_SAMPLES']),
    'proteins_per_temp': int(os.environ['PROTEINS_PER_TEMP']),
    'anchors_per_trajectory': int(os.environ['ANCHORS']),
    'noise_protocol_version': 'shared-window-noise-v1',
    'matched8': {
        'proteins_per_temp': 8, 'anchors_per_trajectory': 1, 'n_samples': 12,
        'context_frames': 8, 'horizons': [1, 4, 16, 64], 'modes': ['correct'],
        'outputs': [str(out / f'{Path(ck).parent.name}_{i}.matched8.json')
                    for i, ck in enumerate(sys.argv[1:])],
    },
    'definition_sections': ['2.6 (T2A)', '2.9 (T3)'],
    'status': int(os.environ['STATUS']),
}, indent=2))
PY
echo "=== standard evaluation '$TAG' complete -> $OUTDIR ==="