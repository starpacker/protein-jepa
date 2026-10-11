#!/bin/bash
# Launch full Stage A training on mdCATH
# Expects: artifacts/mdcath_manifest/{train,val,test}.jsonl already built
# Usage: bash scripts/launch_mdcath_stage_a.sh

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.

echo "=== mdCATH Stage A Training ==="
echo "Manifest: $(wc -l < artifacts/mdcath_manifest/train.jsonl) train trajectories"
echo "Config: configs/mdcath_stage_a.yaml"
echo "Output: artifacts/mdcath_stage_a"
echo ""

python3 -u -c "
from protein_jepa_frame.training.train import train
from protein_jepa_frame.config import load_config

cfg = load_config('configs/mdcath_stage_a.yaml')
print(f'Data: {cfg.data.num_frames} frames × stride {cfg.data.frame_stride}')
print(f'Horizons: {cfg.data.paired_horizons_frames}')
print(f'Model: dim={cfg.model.latent_dim}, enc_depth={cfg.model.encoder_depth}, temporal_depth={cfg.model.temporal_depth}')
print(f'Time cond: {cfg.model.time_conditioning}')
print(f'Batch size: {cfg.data.batch_size}')
print(f'Epochs: {cfg.optimization.epochs}')
print(f'Starting training...')
print()

output = train(cfg)
print(f'\nTraining complete: {output}')
"
