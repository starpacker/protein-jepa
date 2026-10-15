#!/bin/bash
set -euo pipefail

ROOT=/liuyutian/protein_jepa_frame
CONFIG=$ROOT/configs/lspwm_v44_stage_b_v24_mori_stable.yaml
ARTIFACT_DIR=$ROOT/artifacts/lspwm_v44_stage_b_v24_mori_stable
LOG=$ARTIFACT_DIR/train_and_eval.log

exec >> $LOG 2>&1

echo "=========================================="
echo "V24 Train + M1/K1/R1 Eval Pipeline"
echo "Started: $(date)"
echo "=========================================="

# ─── Phase 0: Clean Environment ───────────────────────────────────────────
echo "[0/4] Cleaning environment..."
cd $ROOT

# Kill any zombie training processes
pkill -f "cli.py train" 2>/dev/null || true
pkill -f "protein_jepa_frame.*train" 2>/dev/null || true
sleep 2

# Clear all Python caches
find $ROOT/protein_jepa_frame -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
find $ROOT/protein_jepa_frame -name '*.pyc' -delete 2>/dev/null || true

# Clear any stale installed package
rm -rf /opt/conda/lib/python3.13/site-packages/protein_jepa_frame* 2>/dev/null || true
rm -rf /opt/conda/lib/python3.13/site-packages/protein_jepa_frame-*.egg-link 2>/dev/null || true

# Reinstall fresh from source
echo "[0/4] Reinstalling package..."
PYTHONPATH="" pip install -e $ROOT --no-build-isolation --no-deps 2>&1 | tail -3

# Quick import check
echo "[0/4] Validating imports..."
python3 -c "
from protein_jepa_frame.models.lspwm import (
    UnifiedEnergyGraph, force_score_G_loss, force_score_Q_loss,
    self_consistency_loss, stationary_distribution_loss,
    mori_transition_nll,
)
from protein_jepa_frame.eval.m1k1r1_benchmark import (
    run_m1k1r1_benchmarks, print_summary_table,
)
print('All imports OK (V24 fixes + M1/K1/R1 eval)')
"

# Verify the force_G_init fix is in the source
grep -q "torch.enable_grad():" $ROOT/protein_jepa_frame/models/lspwm.py && echo "FIX: torch.enable_grad in force_G_init path" || echo "WARNING: fix missing!"
grep -q "sigma2 = noise_std \*\* 2" $ROOT/protein_jepa_frame/models/lspwm.py && echo "FIX: sigma^2-scaled DSM" || echo "WARNING: fix missing!"
grep -q "nll = nll.clamp" $ROOT/protein_jepa_frame/models/lspwm.py && echo "FIX: NLL clamping" || echo "WARNING: fix missing!"

# Clean up old stdout.log to get fresh logs
rm -f $ARTIFACT_DIR/stdout.log

# ─── Phase 1: Training ────────────────────────────────────────────────────
echo ""
echo "[1/4] Starting V24 training..."
echo "Config: $CONFIG"
echo "Artifacts: $ARTIFACT_DIR"

mkdir -p $ARTIFACT_DIR
mkdir -p $ROOT/claude_tmp
mkdir -p $ROOT/tmp
export CLAUDE_CODE_TMPDIR=$ROOT/claude_tmp
export TMPDIR=$ROOT/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $ROOT
TRAIN_LOG=$ARTIFACT_DIR/stdout.log

nohup python3 -u protein_jepa_frame/cli.py train \
  --config $CONFIG \
  >> $TRAIN_LOG 2>&1 &

TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"

# Wait for training to actually start
sleep 10
if ! ps -p $TRAIN_PID > /dev/null 2>&1; then
    echo "FATAL: Training failed to start!"
    tail -30 $TRAIN_LOG
    exit 1
fi
echo "Training running (PID=$TRAIN_PID)"

# Monitor: poll train.jsonl every 2 min, check for phase transitions and explosions
echo "[1/4] Monitoring training progress..."
LAST_STEP=-1
while ps -p $TRAIN_PID > /dev/null 2>&1; do
    sleep 120  # Check every 2 minutes
    
    # Get latest metrics
    if [ -f $ARTIFACT_DIR/train.jsonl ]; then
        METRICS=$(python3 -c "
import json
with open('$ARTIFACT_DIR/train.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
if not lines: exit()
# Find latest run (last reset at step 0)
reset_idx = max(i for i, l in enumerate(lines) if l['step'] == 0)
run = lines[reset_idx:]
l = run[-1]
s = l['step']
p = l.get('lspwm_active_phase','?')
loss = l['loss']
mori = l.get('lspwm_mori_nll', 0)
fsG = l.get('lspwm_force_score_G_loss', 0)
fsQ = l.get('lspwm_force_score_Q_loss', 0)
bath = l.get('lspwm_bath_drift_norm', 0)
roll = l.get('lspwm_rollout_trajectory_nll', 0)
skill = l.get('paired_skill', 0)
print(f'{s}|{p}|{loss:.2f}|{mori:.2f}|{fsG:.4f}|{fsQ:.4f}|{bath:.4f}|{roll:.2f}|{skill:.4f}')
")
        if [ -n "$METRICS" ]; then
            IFS='|' read -r STEP PHASE LOSS MORI FSG FSQ BATH ROLL SKILL <<< "$METRICS"
            if [ "$STEP" != "$LAST_STEP" ]; then
                echo "[$(date +%H:%M:%S)] step=$STEP phase=$PHASE loss=$LOSS mori=$MORI G=$FSG Q=$FSQ bath=$BATH roll=$ROLL skill=$SKILL"
                LAST_STEP=$STEP
            fi
            
            # Explosion detection
            if python3 -c "exit(0 if $MORI < 1000 else 1)" 2>/dev/null; then
                : # OK
            else
                echo "CRITICAL: mori_nll=$MORI — LOSS EXPLOSION at step $STEP!"
                kill $TRAIN_PID 2>/dev/null || true
                echo "Training killed due to explosion."
                echo "Check $ARTIFACT_DIR/stdout.log for details."
                exit 1
            fi
            
            if python3 -c "exit(0 if $ROLL < 500 else 1)" 2>/dev/null; then
                : # OK
            else
                echo "CRITICAL: rollout_nll=$ROLL — ROLLOUT EXPLOSION at step $STEP!"
                kill $TRAIN_PID 2>/dev/null || true
                echo "Training killed due to rollout explosion."
                exit 1
            fi
        fi
    fi
    
    # Check if training completed (checkpoint_step6000 exists)
    if [ -f $ARTIFACT_DIR/checkpoint_step6000.pt ]; then
        echo "[1/4] Training completed (checkpoint_step6000.pt found)"
        # Wait for process to finish naturally
        wait $TRAIN_PID 2>/dev/null || true
        break
    fi
done

echo "[1/4] Training phase done."
echo ""

# ─── Phase 2: Find Best Checkpoint ────────────────────────────────────────
echo "[2/4] Finding best checkpoint..."

BEST_CKPT=$(python3 -c "
import json
with open('$ARTIFACT_DIR/train.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
# Latest run
reset_idx = max(i for i, l in enumerate(lines) if l['step'] == 0)
run = lines[reset_idx:]

# Find step with lowest loss in each phase that's been stable
# Use last few hundred steps
phases_seen = set()
best_step = 0
best_loss = float('inf')
for l in run:
    if l['step'] > 0:
        total = l['loss']
        if total < best_loss:
            best_loss = total
            best_step = l['step']
print(best_step)
")

BEST_CHECKPOINT=$ARTIFACT_DIR/checkpoint_step${BEST_CKPT}.pt
if [ ! -f "$BEST_CHECKPOINT" ]; then
    # Fall back to latest
    BEST_CHECKPOINT=$ARTIFACT_DIR/checkpoint_latest.pt
fi
echo "[2/4] Best checkpoint: $BEST_CHECKPOINT"
echo ""

# ─── Phase 3: V23-Style Final Evaluation ──────────────────────────────────
echo "[3/4] Running comprehensive eval on best checkpoint..."
EVAL_DIR=$ARTIFACT_DIR/eval
mkdir -p $EVAL_DIR

python3 -c "
import json, torch, numpy as np, math
from pathlib import Path
from torch.utils.data import DataLoader
from protein_jepa_frame.config import load_config
from protein_jepa_frame.models.jepa import ProteinJEPA, upgrade_lspwm_state_dict
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.collate import collate_paired_windows

CKPT = '$BEST_CHECKPOINT'
CFG = '$CONFIG'
N_BATCHES = 100
device = 'cuda'

config = load_config(CFG)
model = ProteinJEPA(config.model).to(device).eval()
ckpt = torch.load(CKPT, map_location=device)
state = upgrade_lspwm_state_dict(model, ckpt.get('model', ckpt))
model.load_state_dict(state, strict=False)

ds = build_dataset(config.data)
loader = DataLoader(ds, batch_size=config.data.batch_size, shuffle=False,
                    num_workers=0, collate_fn=collate_paired_windows, drop_last=False)

all_results = {
    'paired_skills': [], 'per_horizon_skills': {'24':[],'48':[],'96':[]},
    'force_G_losses': [], 'force_Q_losses': [], 'mori_nlls': [],
    'residue_direction': [], 'residue_amplitude': [], 'residue_conditional': [],
    'residue_force_cosines': [], 'endpoint_skills': [],
    'energy_globals': [], 'energy_coupling_vars': [],
    'global_force_norms': [], 'gauge_norms': [],
    'stationary_losses': [], 'self_consistency': [],
    'mori_force_G_norms': [], 'mori_bath_drift_norms': [],
    'mori_dissipative_norms': [], 'mori_total_drift_norms': [],
    'rollout_nlls': [], 'paired_endpoint_variances': [],
}

n = 0
for batch in loader:
    if n >= N_BATCHES: break
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
    with torch.no_grad():
        with torch.enable_grad():
            out = model.forward_paired(batch)
    for key, arr in [
        ('paired_skill', 'paired_skills'),
        ('lspwm_force_score_G_loss', 'force_G_losses'),
        ('lspwm_force_score_Q_loss', 'force_Q_losses'),
        ('lspwm_mori_nll', 'mori_nlls'),
        ('lspwm_residue_direction_loss', 'residue_direction'),
        ('lspwm_residue_amplitude_loss', 'residue_amplitude'),
        ('lspwm_residue_conditional_loss', 'residue_conditional'),
        ('lspwm_residue_force_cosine', 'residue_force_cosines'),
        ('lspwm_endpoint_energy_skill', 'endpoint_skills'),
        ('lspwm_energy_global', 'energy_globals'),
        ('lspwm_energy_coupling_var', 'energy_coupling_vars'),
        ('lspwm_global_force_norm', 'global_force_norms'),
        ('lspwm_gauge_norm', 'gauge_norms'),
        ('lspwm_stationary_loss', 'stationary_losses'),
        ('lspwm_self_consistency_loss', 'self_consistency'),
        ('lspwm_mori_force_G_norm', 'mori_force_G_norms'),
        ('lspwm_mori_bath_drift_norm', 'mori_bath_drift_norms'),
        ('lspwm_mori_dissipative_norm', 'mori_dissipative_norms'),
        ('lspwm_mori_total_drift_norm', 'mori_total_drift_norms'),
        ('lspwm_rollout_trajectory_nll', 'rollout_nlls'),
        ('paired_endpoint_variance', 'paired_endpoint_variances'),
    ]:
        v = out.get(key)
        if v is not None:
            if hasattr(v, 'numel') and v.numel() > 1:
                all_results[arr].append(float(v.mean().item()))
            elif hasattr(v, 'item'):
                all_results[arr].append(v.item())
            elif isinstance(v, (int, float)):
                all_results[arr].append(v)
    ph = out.get('paired_by_horizon', {})
    for h in ['24','48','96']:
        if h in ph:
            all_results['per_horizon_skills'][h].append(ph[h]['skill'])
    n += 1

print(f'Collected {n} batches')
print('='*70)
print('V24 FINAL EVALUATION RESULTS')
print('='*70)

def summarize(arr, label, fmt='.4f'):
    if not arr: return {'mean':0,'std':0,'n':0}
    a = np.array(arr); a = a[np.isfinite(a)]
    mean, std, nn = float(a.mean()), float(a.std()), len(a)
    print(f'  {label:30s} = {mean:{fmt}} ± {std:{fmt}}  (n={nn})')
    return {'mean':mean,'std':std,'n':nn}

results = {}
print()
print('--- JEPA Backbone ---')
results['paired_skill'] = summarize(all_results['paired_skills'],'Paired Skill')
for h in ['24','48','96']:
    results[f'paired_skill_h{h}'] = summarize(all_results['per_horizon_skills'][h], f'  Horizon {h} frames')

print()
print('--- Force Matching ---')
results['force_score_G'] = summarize(all_results['force_G_losses'],'Force Score G (KDE MSE)')
results['force_score_Q'] = summarize(all_results['force_Q_losses'],'Force Score Q (DSM)')
results['self_consistency'] = summarize(all_results['self_consistency'],'Self-Consistency Loss')

print()
print('--- Dynamics ---')
results['mori_nll'] = summarize(all_results['mori_nlls'],'Mori NLL (FDT bath)')
results['endpoint_skill'] = summarize(all_results['endpoint_skills'],'Endpoint Energy Skill')
results['rollout_nll'] = summarize(all_results['rollout_nlls'],'Rollout NLL')

print()
print('--- Per-Residue Forces ---')
results['residue_direction'] = summarize(all_results['residue_direction'],'Residue Direction Loss')
results['residue_amplitude'] = summarize(all_results['residue_amplitude'],'Residue Amplitude Loss')
results['residue_conditional'] = summarize(all_results['residue_conditional'],'Residue Conditional NLL')
results['residue_force_cos'] = summarize(all_results['residue_force_cosines'],'Force-Displacement Cosine')

print()
print('--- Energy Graph ---')
results['energy_global'] = summarize(all_results['energy_globals'],'Global PMF Energy')
results['energy_coupling_var'] = summarize(all_results['energy_coupling_vars'],'Coupling Energy Var', '.6f')
results['global_force_norm'] = summarize(all_results['global_force_norms'],'Global Force Norm')
results['gauge_norm'] = summarize(all_results['gauge_norms'],'Gauge Norm')

print()
print('--- Mori Bath Diagnostics ---')
results['mori_force_G_norm'] = summarize(all_results['mori_force_G_norms'],'Mori Force G Norm')
results['mori_bath_drift'] = summarize(all_results['mori_bath_drift_norms'],'Mori Bath Drift Norm')
results['mori_dissipative'] = summarize(all_results['mori_dissipative_norms'],'Mori Dissipative Drift Norm')
results['mori_total_drift'] = summarize(all_results['mori_total_drift_norms'],'Mori Total Drift Norm')
results['endpoint_variance'] = summarize(all_results['paired_endpoint_variances'],'Endpoint Variance')

with open('$EVAL_DIR/v24_final_eval.json','w') as f:
    json.dump(results, f, indent=2)
print(f'Results saved to $EVAL_DIR/v24_final_eval.json')
" 2>&1 | tee -a $LOG

echo ""
echo "[3/4] Eval done."
echo ""

# ─── Phase 4: M1/K1/R1 Benchmarks ─────────────────────────────────────────
echo "[4/4] Running M1/K1/R1 benchmarks..."
echo "Checkpoint: $BEST_CHECKPOINT"

mkdir -p $EVAL_DIR

python3 $ROOT/protein_jepa_frame/eval/m1k1r1_benchmark.py \
  --checkpoint "$BEST_CHECKPOINT" \
  --config "$CONFIG" \
  --output-dir "$EVAL_DIR" \
  --m1-batches 50 \
  --k1-batches 100 \
  --r1-batches 30 \
  2>&1 | tee -a $LOG

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE: $(date)"
echo "=========================================="
echo ""
echo "Artifacts:"
echo "  Model checkpoint: $BEST_CHECKPOINT"
echo "  Detailed eval:    $EVAL_DIR/v24_final_eval.json"
echo "  M1/K1/R1 results: $EVAL_DIR/m1k1r1_benchmark.json"
echo "  Full log:         $LOG"
echo "  Training log:     $TRAIN_LOG"
