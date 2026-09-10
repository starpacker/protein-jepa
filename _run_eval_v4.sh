#!/bin/bash
cd /liuyutian/protein_jepa_frame
python3 -u -m protein_jepa_frame.eval.comprehensive_eval \
    --checkpoint /liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v3_improved_thermodynamics/checkpoint_step3000.pt \
    --config /liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v3_improved_thermodynamics/eval_improved/eval_config.json \
    --output-dir /liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v3_improved_thermodynamics/eval_improved \
    --max-batches 20 --bandwidth 0.1 --device cuda:0 \
    --fastfolding-labels /tmp/lspwm_fastfolding_labels.jsonl \
    2>&1 | tee /liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v3_improved_thermodynamics/eval_improved/eval_v4.log
echo "EXIT:$?" > /liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v3_improved_thermodynamics/eval_improved/_eval_v4_exit.txt
