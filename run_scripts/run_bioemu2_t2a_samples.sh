#!/usr/bin/env bash
# bioemu2 zero-shot samples for the BioEmu-style static-ensemble metrics (T2A).
#
# Purpose: give bioemu2 the SAME T2A treatment as our own model, i.e. score it
# with evaluation/t2a_static.py against the SAME per-trajectory reference
# ensembles built by scripts/eval_lasp_ca_suite.py::reference_ensemble.
#
# Scope choice (documented, not hidden): our suite scores 12 proteins per
# temperature x 5 replicas x 2 anchors x 4 horizons = 480 windows per
# temperature.  bioemu2 at ~20 s/window would need ~13 h for that.  We instead
# take a strict PREFIX of exactly those windows -- the first 8 proteins per
# temperature (the 8-protein set is a subset of the 12-protein set, verified),
# the first anchor of each trajectory and all four horizons
# (8 x 5 x 1 x 4 = 160 windows per temperature, 800 windows total) -- so
# bioemu2's T2A covers exactly the same trajectories, residue masks and
# reference ensembles as our model restricted to the same windows.
#
# The matching protocol for our own model is
#
#   python -m scripts.eval_lasp_ca_suite --checkpoint <ckpt> --out <out> --t2a \
#       --modes correct --proteins-per-temp 8 --anchors-per-trajectory 1 \
#       --n-samples 12 --horizons 1 4 16 64 --max-residues 96
#
# i.e. identical window set, identical reference ensembles, identical pooling
# (every window of a trajectory pooled, per (protein, trajectory, temperature)).
# scripts/t2a_from_samples.py reads the per-window .npz dumps.
#
# Sampling is anchor-conditioned and never sees the future; the true MD frames
# written alongside are reference data for the metric only.
set -euo pipefail
cd /liuyutian/protein_jepa_frame

export TMPDIR=/tmp
export HF_HUB_CACHE=${HF_HUB_CACHE:-/octopus-runtime/cache/huggingface/hub}
export BIOEMU2_MSA_CACHE=${BIOEMU2_MSA_CACHE:-$PWD/baselines/bioemu2/msa_cache}

CKPT=${CKPT:-$PWD/baselines/bioemu2/checkpoints/bioemu2-preview/checkpoint.ckpt}
OUTDIR=${OUTDIR:-results/lasp_full_v1/t2a_bioemu2}
SAMPLES=${SAMPLES:-$OUTDIR/samples}
TEMPS=${TEMPS:-"320 348 379 413 450"}
GPUS=${GPUS:-"2 5 7 0 1"}
PROTEINS=${PROTEINS:-8}
mkdir -p "$OUTDIR" "$SAMPLES" "$BIOEMU2_MSA_CACHE"

# One temperature per GPU, concurrently: the MSA embedding dominates the cost and
# is per (protein, temperature), so the five splits share no work anyway.
read -r -a gpu_arr <<< "$GPUS"
i=0
pids=()
for T in $TEMPS; do
  gpu=${gpu_arr[$((i % ${#gpu_arr[@]}))]}
  i=$((i + 1))
  echo "=== bioemu2 T2A sampling T=$T on GPU $gpu ==="
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m evaluation.baselines.run_bioemu2_zeroshot \
    --manifest artifacts/mdcath_manifest_full/val.jsonl \
    --temperatures "$T" \
    --horizons-ns 1 4 16 64 \
    --n-samples 12 \
    --max-proteins "$PROTEINS" \
    --max-anchors-per-trajectory 1 \
    --max-residues 96 \
    --history-frames 8 \
    --num-steps 128 \
    --device cuda \
    --sampler edm \
    --checkpoint-path "$CKPT" \
    --save-samples "$SAMPLES" \
    --out "$OUTDIR/bioemu2_T${T}.json" \
    > "$OUTDIR/bioemu2_T${T}.log" 2>&1 &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
echo "=== all temperatures finished (status=$status) ==="
if [ "$status" -ne 0 ]; then
  echo "ERROR: sampling failed; refusing to publish a T2A report" >&2
  exit "$status"
fi

python -u -m scripts.t2a_from_samples \
  --samples "$SAMPLES" \
  --manifest artifacts/mdcath_manifest_full/val.jsonl \
  --max-residues 96 \
  --source bioemu2 \
  --temps $TEMPS --proteins-per-temp "$PROTEINS" \
  --anchors-per-trajectory 1 --context-frames 8 --horizons 1 4 16 64 \
  --out "$OUTDIR/bioemu2_t2a.json" | tee "$OUTDIR/t2a_summary.log"
echo "=== bioemu2 T2A complete ==="