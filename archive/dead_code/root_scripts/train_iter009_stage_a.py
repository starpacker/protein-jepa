"""iter_009 Stage A fine-tune: add per-residue displacement prediction objective.

Fine-tunes from artifacts/full_abc_stage_a/checkpoint_latest.pt.
Adds displacement_head supervision on top of existing paired objective.
All existing JEPA objectives (mask + paired) are preserved and unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from protein_jepa_frame.config import TrainConfig, config_to_dict
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.stage_a import StageAJEPA
from protein_jepa_frame.models.stage_a.objectives import (
    stage_a_displacement_loss,
    stage_a_total_loss,
)
from protein_jepa_frame.training.train_utils import (
    _device,
    _lr_scale,
    _momentum,
    _move_batch,
    _save_checkpoint,
    _set_reproducibility,
)
from torch.utils.data import DataLoader

# ── Config ──────────────────────────────────────────────────────────────────
STAGE_A_CKPT = "artifacts/full_abc_stage_a/checkpoint_latest.pt"
OUTPUT_DIR    = "artifacts/iter_009_stage_a_disp"
MANIFEST_TRAIN = "artifacts/mdcath_manifest_200dom/train.jsonl"

TOTAL_STEPS  = 5000
WARMUP_STEPS = 200
LR           = 1e-4
ENCODER_LR_SCALE = 0.1   # freeze-like: encoder at 10% of head LR
WEIGHT_DECAY = 1e-2
GRAD_CLIP    = 1.0
EMA_START    = 0.996
EMA_END      = 0.999
BATCH_SIZE   = 8
NUM_WORKERS  = 4
LOG_EVERY    = 10
SAVE_EVERY   = 500
SEED         = 42

# Displacement loss weights
DISP_WEIGHT  = 1.0    # total weight on displacement loss
DIR_WEIGHT   = 1.0    # direction cosine sub-weight
MAG_WEIGHT   = 1.0    # magnitude smooth-L1 sub-weight
PAIRED_WEIGHT = 0.5   # keep the existing paired endpoint objective

# ── Build ─────────────────────────────────────────────────────────────────
def main():
    _set_reproducibility(SEED, False)
    device = _device("cuda" if torch.cuda.is_available() else "cpu")

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load reference config from original Stage A run ──
    ref_cfg_path = Path("artifacts/full_abc_stage_a/config.json")
    with ref_cfg_path.open() as f:
        ref_cfg_dict = json.load(f)

    # Build a minimal TrainConfig from it
    from dataclasses import replace as dc_replace
    from protein_jepa_frame.config import DataConfig, ModelConfig, OptimizationConfig, MaskConfig
    from protein_jepa_frame.data.dataset import load_manifest

    model_d  = ref_cfg_dict.get("model", {})
    data_d   = ref_cfg_dict.get("data", {})
    optim_d  = ref_cfg_dict.get("optimization", {})

    model_cfg = ModelConfig(**{k: v for k, v in model_d.items()
                                if k in ModelConfig.__dataclass_fields__})

    from dataclasses import fields as dc_fields
    data_cfg_defaults = {f.name: f.default for f in dc_fields(DataConfig)}
    data_cfg_overrides = {k: v for k, v in data_d.items()
                          if k in DataConfig.__dataclass_fields__}
    data_cfg = DataConfig(**{**data_cfg_defaults, **data_cfg_overrides,
                              "manifest_path": MANIFEST_TRAIN,
                              "batch_size": BATCH_SIZE,
                              "num_workers": NUM_WORKERS,
                              "training_mode": "paired_window",
                              "paired_horizons_frames": [4, 8, 16, 32, 64, 96],
                              "paired_all_horizons": True,
                              "paired_history_mode": "uniform"})

    # ── Dataset ──
    dataset = build_dataset(data_cfg)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        collate_fn=collate_paired_windows,
        drop_last=True,
    )

    # ── Model ──
    model = StageAJEPA(model_cfg).to(device)

    # Load checkpoint strictly for all keys that exist; displacement_head is new
    print(f"Loading Stage A checkpoint: {STAGE_A_CKPT}")
    raw = torch.load(STAGE_A_CKPT, map_location="cpu")
    state = raw.get("model") or raw.get("model_state_dict") or raw.get("state_dict") or raw
    if not isinstance(state, dict) or not any("." in k for k in state):
        raise RuntimeError(f"could not locate model state dict in checkpoint; keys={list(raw.keys())}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    disp_keys = {k for k in missing if k.startswith("displacement_head")}
    non_disp_missing = [k for k in missing if not k.startswith("displacement_head")]
    if non_disp_missing:
        raise RuntimeError(
            f"Unexpected missing keys (non-displacement): {non_disp_missing[:10]}"
        )
    if unexpected:
        print(f"  WARNING: unexpected keys in checkpoint (ignored): {unexpected[:5]}")
    print(f"  displacement_head keys initialised fresh: {sorted(disp_keys)}")
    print("  All other keys loaded from checkpoint.")

    # EMA target encoders: sync from online
    model.sync_target_encoders()

    # ── Optimiser: separate LRs ──
    # displacement_head + paired_delta_predictor: full LR
    # context_encoder: 10x lower (fine-tune gently)
    # everything else: full LR
    disp_params   = list(model.displacement_head.parameters())
    enc_params    = list(model.context_encoder.parameters())
    other_params  = [p for n, p in model.named_parameters()
                     if not n.startswith("displacement_head")
                     and not n.startswith("context_encoder")
                     and not n.startswith("target_")         # EMA, no grad
                     and p.requires_grad]

    param_groups = [
        {"params": disp_params,  "lr": LR,                    "base_lr_scale": 1.0},
        {"params": enc_params,   "lr": LR * ENCODER_LR_SCALE, "base_lr_scale": ENCODER_LR_SCALE},
        {"params": other_params, "lr": LR,                    "base_lr_scale": 1.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    log_path = out / "train.jsonl"
    (out / "stage.txt").write_text("stage_a_iter009\n")
    (out / "config.json").write_text(json.dumps({
        "stage_a_ckpt": STAGE_A_CKPT,
        "total_steps": TOTAL_STEPS,
        "lr": LR,
        "encoder_lr_scale": ENCODER_LR_SCALE,
        "disp_weight": DISP_WEIGHT,
        "paired_weight": PAIRED_WEIGHT,
        "dir_weight": DIR_WEIGHT,
        "mag_weight": MAG_WEIGHT,
        "batch_size": BATCH_SIZE,
        "paired_horizons_frames": data_cfg.paired_horizons_frames,
        "model": model_d,
    }, indent=2))

    # ── Training loop ──
    global_step = 0
    data_iter   = iter(loader)

    print(f"Starting iter_009 Stage A displacement fine-tune for {TOTAL_STEPS} steps")
    print(f"  Output: {out}")

    while global_step < TOTAL_STEPS:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = _move_batch(batch, device)

        # LR schedule
        lr_mult = _lr_scale(global_step, TOTAL_STEPS, WARMUP_STEPS)
        for group in optimizer.param_groups:
            group["lr"] = LR * float(group["base_lr_scale"]) * lr_mult

        optimizer.zero_grad(set_to_none=True)

        # ── Forward: paired endpoint prediction (existing objective) ──
        paired_out = model.forward_paired(batch)
        paired_loss, paired_log = stage_a_total_loss(
            mask_outputs=None,
            paired_outputs=paired_out,
            paired_weight=PAIRED_WEIGHT,
            mask_weight=0.0,
        )

        # ── Forward: displacement prediction (new objective) ──
        disp_out  = model.forward_displacement(batch)
        disp_loss = stage_a_displacement_loss(
            disp_out,
            dir_weight=DIR_WEIGHT,
            mag_weight=MAG_WEIGHT,
        )

        loss = paired_loss + DISP_WEIGHT * disp_loss

        loss.backward()

        if GRAD_CLIP is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        # EMA update
        momentum = _momentum(global_step, TOTAL_STEPS, EMA_START, EMA_END)
        model.update_target_encoder(momentum)

        # Logging
        if global_step % LOG_EVERY == 0:
            # Fraction of valid residues with |target| > 0.1 (displacement gate)
            n_valid = disp_out["valid_mask"].sum().item()
            n_moving = (
                disp_out["valid_mask"] &
                (disp_out["disp_target"].norm(dim=-1) > 0.1)
            ).sum().item()
            record = {
                "step": global_step,
                "lr": optimizer.param_groups[0]["lr"],
                "ema": momentum,
                "loss": float(loss.detach()),
                "paired_loss": float(paired_log.get("paired_delta_loss", 0.0)),
                "disp_loss": float(disp_loss.detach()),
                "n_valid_res": int(n_valid),
                "n_moving_res": int(n_moving),
            }
            with log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
            print(json.dumps(record), flush=True)

        if (global_step + 1) % SAVE_EVERY == 0:
            _save_checkpoint(out, model, optimizer, 0, global_step + 1, None,
                             write_step_file=True)

        global_step += 1

    _save_checkpoint(out, model, optimizer, 0, global_step, None, write_step_file=False)
    print(f"iter_009 Stage A fine-tune complete. Checkpoint: {out}/checkpoint_latest.pt")


if __name__ == "__main__":
    main()
