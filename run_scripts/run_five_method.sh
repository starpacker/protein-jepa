#!/usr/bin/env bash
# Five-method comparison sweep (Model1 / Model2 / Lasp / MDGen / bioemu2).
#
# One cell = (protein split x temperature).  Each process scores ONE method on
# ONE cell and writes its own JSONL, so the five methods can run in parallel on
# separate GPUs without interleaving writes; scripts/five_method/make_tables.py
# merges them afterwards.
#
#   usage: bash run_scripts/run_five_method.sh <gpu> <suite...>
set -uo pipefail
cd /liuyutian/protein_jepa_frame
export TMPDIR=/tmp
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/octopus-runtime/cache/huggingface/hub}
export BIOEMU2_MSA_CACHE=${BIOEMU2_MSA_CACHE:-$PWD/baselines/bioemu2/msa_cache}
export PYTHONPATH=$PWD

GPU=${1:?gpu}; shift
SUITES=${*:?suites}
NS=${NS:-8}
PROT=${PROT:-20}
OUTDIR=${OUTDIR:-results/five_method}
mkdir -p "$OUTDIR" logs/five_method

run() {  # run <method> <suite> <extra...>
  local m=$1 s=$2; shift 2
  CUDA_VISIBLE_DEVICES=$GPU python -u -m scripts.five_method.run_five_method \
    --suite "$s" --methods "$m" --device cuda \
    --n-samples "$NS" --max-proteins "$PROT" \
    --max-anchors-per-trajectory 1 --max-trajectories-per-protein 1 \
    --out "$OUTDIR/${s}__${m}.jsonl" "$@" \
    >> "logs/five_method/${s}__${m}.log" 2>&1
}

for s in $SUITES; do
  case " $* " in
    *" --temp-split "*) TS=(--temp-split);;
    *) TS=();;
  esac
  run model1  "$s" "${TS[@]}"
  run model2  "$s" "${TS[@]}"
  run lasp    "$s" "${TS[@]}"
done
