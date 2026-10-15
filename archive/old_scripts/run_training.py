#!/usr/bin/env python3
"""Launch script for training with a config file."""
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from protein_jepa_frame.config import load_config
from protein_jepa_frame.training import train

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/lspwm_v44_stage_b_detskill_fix_v2.yaml"
    cfg = load_config(config_path)
    ckpt = train(cfg)
    print(f"Training complete. Checkpoint: {ckpt}")