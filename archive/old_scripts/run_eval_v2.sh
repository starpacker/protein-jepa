#!/bin/bash
# Run comprehensive evaluation with step3000 checkpoint

cd /liuyutian/protein_jepa_frame

ARTIFACTS="/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v3_improved_thermodynamics"
EVAL_DIR="${ARTIFACTS}/eval_improved"

# Ensure eval config exists with batch_size=4
python3 -c "
import json, yaml
with open('${ARTIFACTS}/../configs/lspwm_v44_stage_b_v3_improved_thermodynamics.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['data']['batch_size'] = 4
cfg['model']['lspwm_potential_hidden_depth'] = 2
cfg['model']['lspwm_potential_hidden_dim'] = 128
import os; os.makedirs('${EVAL_DIR}', exist_ok=True)
with open('${EVAL_DIR}/eval_config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print('eval_config.json created')
"

echo "Starting comprehensive evaluation with step3000 checkpoint..."
echo "Log: ${EVAL_DIR}/eval_v2.log"

python3 -u -m protein_jepa_frame.eval.comprehensive_eval \
    --checkpoint "${ARTIFACTS}/checkpoint_step3000.pt" \
    --config "${EVAL_DIR}/eval_config.json" \
    --output-dir "${EVAL_DIR}" \
    --max-batches 20 \
    --bandwidth 0.1 \
    --device cuda:0 \
    --fastfolding-labels /tmp/lspwm_fastfolding_labels.jsonl \
    2>&1 | tee "${EVAL_DIR}/eval_v2.log"

EVAL_EXIT=$?
echo "Eval exit code: ${EVAL_EXIT}"
date > "${EVAL_DIR}/_eval_v2_done.txt"
echo "Done"