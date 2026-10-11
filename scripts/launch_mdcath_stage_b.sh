#!/bin/bash
# Launch Stage B LSPWM training (requires Stage A checkpoint)
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.

# Wait for Stage A checkpoint
A_CKPT="artifacts/mdcath_stage_a_200dom/checkpoint_latest.pt"
echo "=== mdCATH Stage B Training ==="
echo "Stage A checkpoint: $A_CKPT"
if [ ! -f "$A_CKPT" ]; then
    echo "ERROR: Stage A checkpoint not found!"
    echo "Run Stage A training first: python3 -c \"from protein_jepa_frame.training.train import train; from protein_jepa_frame.config import load_config; train(load_config('configs/mdcath_stage_a_200dom.yaml'))\""
    exit 1
fi

# Verify temporal sensitivity
echo ""
echo "Verifying temporal sensitivity..."
python3 scripts/verify_temporal_sensitivity.py "$A_CKPT"
echo ""

python3 -u -c "
from protein_jepa_frame.training.train import train
from protein_jepa_frame.config import load_config

cfg = load_config('configs/mdcath_stage_b.yaml')
cfg.init_checkpoint = 'artifacts/mdcath_stage_a_200dom/checkpoint_latest.pt'
cfg.model.freeze_context_encoder = True
cfg.model.freeze_temporal_encoder = True
cfg.model.freeze_target_encoders = True
cfg.output_dir = 'artifacts/mdcath_stage_b'
cfg.data.batch_size = 16
cfg.data.num_workers = 4

print(f'Device: {cfg.device}')
print(f'Hidden dim: {cfg.model.lspwm_learned_cv_dim}')
print(f'Rollout steps: {cfg.model.lspwm_rollout_steps}')
print(f'Starting Stage B training...')
print()

output = train(cfg)
print(f'\nTraining complete: {output}')
"
