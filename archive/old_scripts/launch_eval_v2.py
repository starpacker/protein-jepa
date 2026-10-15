#!/usr/bin/env python3
"""Spawn the comprehensive evaluation as a fully-detached subprocess and exit."""
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

CHECKPOINT = "/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v3_improved_thermodynamics/checkpoint_step3000.pt"
EVAL_DIR = Path("/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v3_improved_thermodynamics/eval_improved")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# 1. Prepare eval config
with open("/liuyutian/protein_jepa_frame/configs/lspwm_v44_stage_b_v3_improved_thermodynamics.yaml") as f:
    cfg = yaml.safe_load(f)
cfg['data']['batch_size'] = 4
cfg['model']['lspwm_potential_hidden_depth'] = 2
cfg['model']['lspwm_potential_hidden_dim'] = 128
with open(EVAL_DIR / "eval_config.json", "w") as f:
    json.dump(cfg, f, indent=2)
print("eval_config.json ready")

# 2. Fork and detach
pid = os.fork()
if pid > 0:
    print(f"Forked child PID={pid}, parent exiting.")
    sys.exit(0)

# Child: become session leader, detach from parent
os.setsid()
log_path = EVAL_DIR / "eval_v2.log"

with open(log_path, "w") as log:
    log.write("=== Comprehensive Eval v2 (step3000) ===\n")
    log.flush()
    proc = subprocess.Popen(
        [
            sys.executable, "-u", "-m", "protein_jepa_frame.eval.comprehensive_eval",
            "--checkpoint", CHECKPOINT,
            "--config", str(EVAL_DIR / "eval_config.json"),
            "--output-dir", str(EVAL_DIR),
            "--max-batches", "20",
            "--bandwidth", "0.1",
            "--device", "cuda:0",
            "--fastfolding-labels", "/tmp/lspwm_fastfolding_labels.jsonl",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        cwd="/liuyutian/protein_jepa_frame",
    )
    with open(EVAL_DIR / "_eval_v2_pid.txt", "w") as pf:
        pf.write(str(proc.pid))
    proc.wait()
    log.write(f"\n=== Eval exited with code {proc.returncode} ===\n")
    with open(EVAL_DIR / "_eval_v2_done.txt", "w") as df:
        df.write(f"exit_code={proc.returncode}\n")
