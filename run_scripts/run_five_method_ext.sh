#!/usr/bin/env bash
# Run ONE method over a list of cells on ONE GPU, in ONE process so the model
# and the per-sequence bioemu2 MSA/embedding caches are reused across cells.
#
#   bash run_scripts/run_five_method_ext.sh <gpu> <method> <suite...>
#
# Env: NS (samples/window), PROT (proteins/cell), THREADS (CPU threads),
#      EXTRA (extra CLI flags, e.g. "--temp-split" or "--bioemu2-walk"),
#      OUTDIR, TAG
set -uo pipefail
cd /liuyutian/protein_jepa_frame
export TMPDIR=/tmp
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/octopus-runtime/cache/huggingface/hub}
export BIOEMU2_MSA_CACHE=${BIOEMU2_MSA_CACHE:-$PWD/baselines/bioemu2/msa_cache}
export PYTHONPATH=$PWD
GPU=${1:?gpu}; M=${2:?method}; shift 2; SUITES=${*:?suites}
NS=${NS:-8}; PROT=${PROT:-20}; OUTDIR=${OUTDIR:-results/five_method}
TAG=${TAG:-}; EXTRA=${EXTRA:-}
mkdir -p "$OUTDIR" logs/five_method
SUFFIX=${TAG:+__$TAG}
OUT=${OUTDIR}/${M}${SUFFIX}.jsonl
rm -f "$OUT"
echo "[launch] gpu=$GPU method=$M cells=$SUITES -> $OUT" > "logs/five_method/${M}${SUFFIX}.log"
CUDA_VISIBLE_DEVICES=$GPU python -u -m scripts.five_method.run_five_method \
  --torch-threads "${THREADS:-4}" \
  --suite $SUITES --methods "$M" --device cuda --n-samples "$NS" \
  --max-proteins "$PROT" --max-anchors-per-trajectory 1 \
  --max-trajectories-per-protein 1 --out "$OUT" $EXTRA \
  >> "logs/five_method/${M}${SUFFIX}.log" 2>&1
