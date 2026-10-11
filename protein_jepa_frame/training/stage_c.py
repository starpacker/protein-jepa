"""Stage C training: decode structure from frozen Stage A/B latent states.

Gate 5/6 canonical training:
- Load frozen Stage B checkpoint (contains frozen Stage A + trainable B)
- Freeze encoders, only train decoder + direct_head
- Use real per-residue H_last and Q from frozen pipeline
- Direct+endpoint training with oracle latent reconstruction benchmark
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from protein_jepa_frame.config import TrainConfig, config_to_dict
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset, validate_real_paired_window_protocol
from protein_jepa_frame.models.stage_b.model import StageBLASP
from protein_jepa_frame.models.stage_c import (
    StructureDecoder,
    DecoderConfig,
    DirectEndpointHead,
    DirectHeadConfig,
    stage_c_total_loss,
)
from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame
from protein_jepa_frame.training.train_utils import (
    _device, _set_reproducibility, _move_batch,
    _lr_scale, _save_checkpoint, _validate_init_freeze_contract,
    _write_run_manifest,
)


def _extract_backbone_frames(backbone_coords):
    N = backbone_coords[..., 0, :]
    CA = backbone_coords[..., 1, :]
    C = backbone_coords[..., 2, :]
    return RigidBodyFrame.from_backbone(N, CA, C)


def _dihedral(p1, p2, p3, p4):
    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
    b2_n = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
    b1_p = b1 - (b1 * b2_n).sum(dim=-1, keepdim=True) * b2_n
    b3_p = b3 - (b3 * b2_n).sum(dim=-1, keepdim=True) * b2_n
    n1 = torch.cross(b2_n, b1_p / (b1_p.norm(dim=-1, keepdim=True) + 1e-8), dim=-1)
    n2 = torch.cross(b2_n, b3_p / (b3_p.norm(dim=-1, keepdim=True) + 1e-8), dim=-1)
    dot = (n1 * n2).sum(dim=-1)
    cross = (torch.cross(n1, n2, dim=-1) * b2_n).sum(dim=-1)
    return torch.atan2(cross, dot)


def _compute_target_torsions(backbone_coords):
    N = backbone_coords[..., 0, :]
    CA = backbone_coords[..., 1, :]
    C = backbone_coords[..., 2, :]
    B, L = N.shape[:2]
    torsions = torch.zeros(B, L, 7, device=N.device)
    for i in range(L):
        if i > 0:
            torsions[:, i, 0] = _dihedral(C[:, i - 1], N[:, i], CA[:, i], C[:, i])
        if i < L - 1:
            torsions[:, i, 1] = _dihedral(N[:, i], CA[:, i], C[:, i], N[:, i + 1])
        if i < L - 1:
            torsions[:, i, 2] = _dihedral(CA[:, i], C[:, i], N[:, i + 1], CA[:, i + 1])
    return torsions


def _load_stage_c_state(decoder, direct_head, state):
    """Load legacy flat or explicitly prefixed Stage-C state dictionaries."""
    decoder_keys = set(decoder.state_dict())
    direct_keys = set(direct_head.state_dict())
    if any(k.startswith(("decoder.", "direct_head.")) for k in state):
        dec_state = {
            k.removeprefix("decoder."): v for k, v in state.items()
            if k.startswith("decoder.")
        }
        dh_state = {
            k.removeprefix("direct_head."): v for k, v in state.items()
            if k.startswith("direct_head.")
        }
    else:
        # Existing checkpoints store decoder and direct-head keys in one flat
        # mapping. Exact membership makes the split unambiguous.
        dec_state = {k: v for k, v in state.items() if k in decoder_keys}
        dh_state = {k: v for k, v in state.items() if k in direct_keys}
    decoder.load_state_dict(dec_state, strict=True)
    direct_head.load_state_dict(dh_state, strict=True)


def _stage_c_checkpoint_payload(
    decoder,
    direct_head,
    optimizer,
    epoch,
    step,
    config,
    stage_b_model=None,
):
    """Build a Stage-C checkpoint, including adapted encoder weights.

    Frozen-AB checkpoints historically contained only decoder/direct-head
    parameters.  Once Stage-C fine-tunes an encoder, saving only those two
    modules would make the reported checkpoint silently revert to the old
    representation at evaluation time.  Embed the complete Stage-B state
    whenever any Stage-B parameter is trainable.
    """
    payload = {
        "model": {**decoder.state_dict(), **direct_head.state_dict()},
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": config_to_dict(config),
    }
    if stage_b_model is not None and any(
        p.requires_grad for p in stage_b_model.parameters()
    ):
        payload["stage_b_model"] = stage_b_model.state_dict()
    return payload


def _stage_c_finetune_summary(stage_b_model) -> dict:
    """Return a compact, serializable summary of Stage-B trainability."""
    if stage_b_model is None:
        return {"enabled": False, "trainable_parameters": 0, "modules": []}
    modules = []
    trainable = 0
    for name, module in (
        ("spatial_encoder", stage_b_model.spatial_encoder),
        ("temporal_encoder", stage_b_model.temporal_encoder),
    ):
        if module is None:
            continue
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if n:
            modules.append(name)
            trainable += n
    return {
        "enabled": bool(modules),
        "trainable_parameters": trainable,
        "modules": modules,
    }


def train_stage_c(config: TrainConfig) -> Path:
    _validate_init_freeze_contract(config)
    validate_real_paired_window_protocol(config.data)
    _set_reproducibility(config.seed, config.deterministic)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dict = config_to_dict(config)
    (output_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

    # Write run manifest (P0-11 provenance)
    _write_run_manifest(
        output_dir,
        config_dict=config_dict,
        init_checkpoint_path=config.init_checkpoint,
        data_paths=[str(config.data.data_dir)] if hasattr(config.data, 'data_dir') and config.data.data_dir else None,
    )

    device = _device(config.device)

    # Build data
    dataset = build_dataset(config.data, msm_labels_path=config.data.msm_labels_path)
    loader = DataLoader(
        dataset, batch_size=config.data.batch_size, shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=str(device).startswith("cuda"),
        collate_fn=collate_paired_windows, drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("dataloader is empty")

    steps_per_epoch = config.optimization.steps_per_epoch or len(loader)
    # Support explicit max_steps
    max_steps = getattr(config.optimization, 'max_steps', None)
    if max_steps is not None and max_steps > 0:
        total_steps = max_steps
        config.optimization.epochs = max(1, int(math.ceil(total_steps / steps_per_epoch)))
    else:
        total_steps = steps_per_epoch * config.optimization.epochs

    # ── Load Stage A+B ──
    stage_b_model = None
    finetune_spatial = bool(
        getattr(config.model, "stage_c_finetune_spatial_encoder", False)
    )
    finetune_temporal = bool(
        getattr(config.model, "stage_c_finetune_temporal_encoder", False)
    )
    if config.init_checkpoint:
        stage_b_model = StageBLASP(
            config.model,
            freeze_spatial_encoder=not finetune_spatial,
            freeze_temporal_encoder=not finetune_temporal,
        ).to(device)
        ckpt = torch.load(config.init_checkpoint, map_location=device)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        # Preserve a canonical Stage B checkpoint in full.  Legacy Stage A/B
        # checkpoints used context_encoder.* for the same spatial module, so
        # remap only that prefix instead of dropping all dynamics parameters.
        remapped = {}
        for k, v in state.items():
            if k.startswith("context_encoder."):
                remapped["spatial_encoder." + k.removeprefix("context_encoder.")] = v
            else:
                remapped[k] = v
        state = remapped
        incompatible = stage_b_model.load_state_dict(state, strict=False)
        # Keep the pretrained Stage-B dynamics and projection frozen.  Only
        # explicitly requested representation encoders are adapted.  eval()
        # disables dropout while retaining autograd for those encoders, which
        # is the stable fine-tuning behavior for this small-batch protocol.
        stage_b_model.eval()
        for p in stage_b_model.parameters():
            p.requires_grad = False
        if finetune_spatial:
            for p in stage_b_model.spatial_encoder.parameters():
                p.requires_grad = True
        if finetune_temporal and stage_b_model.temporal_encoder is not None:
            for p in stage_b_model.temporal_encoder.parameters():
                p.requires_grad = True
        finetune_summary = _stage_c_finetune_summary(stage_b_model)
        print(json.dumps({"event": "stage_c_loaded_ab", "checkpoint": str(config.init_checkpoint),
                          "loaded_keys": len(state),
                          "missing_keys": list(incompatible.missing_keys),
                          "unexpected_keys": list(incompatible.unexpected_keys),
                          "finetune": finetune_summary}), flush=True)

    # ── Build decoder (match encoder output dims) ──
    q_dim = int(getattr(config.model, "q_dim", 8))
    decoder_cfg = DecoderConfig(
        context_dim=config.model.latent_dim,
        q_dim=q_dim,
        c_hidden=config.model.latent_dim,
        max_translation_update=float(
            getattr(config.model, "decoder_max_translation_update", 0.5)
        ),
        max_com_update=float(
            getattr(config.model, "decoder_max_com_update", 4.0)
        ),
        max_q_translation_update=float(
            getattr(config.model, "decoder_max_q_translation_update", 2.0)
        ),
    )
    decoder = StructureDecoder(decoder_cfg).to(device)
    direct_cfg = DirectHeadConfig(
        context_dim=config.model.latent_dim,
        condition_dim=0,
        latent_dim=q_dim,
        hidden_dim=config.model.latent_dim,
    )
    direct_head = DirectEndpointHead(direct_cfg).to(device)
    decoder_params = list(decoder.parameters()) + list(direct_head.parameters())
    encoder_params = (
        [p for p in stage_b_model.parameters() if p.requires_grad]
        if stage_b_model is not None else []
    )
    all_params = decoder_params + encoder_params
    param_groups = [
        {
            "params": decoder_params,
            "lr": config.optimization.lr,
            "base_lr_scale": 1.0,
            "group_name": "stage_c_decoder",
        }
    ]
    if encoder_params:
        encoder_lr_scale = float(
            getattr(config.optimization, "encoder_lr_scale", 0.2)
        )
        if encoder_lr_scale <= 0.0:
            raise ValueError(
                "Stage-C encoder fine-tuning requires optimization.encoder_lr_scale > 0"
            )
        param_groups.append({
            "params": encoder_params,
            "lr": config.optimization.lr * encoder_lr_scale,
            "base_lr_scale": encoder_lr_scale,
            "group_name": "stage_b_representation",
        })

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.optimization.lr,
        weight_decay=config.optimization.weight_decay,
    )
    print(json.dumps({
        "event": "stage_c_optimizer_ready",
        "decoder_parameters": sum(p.numel() for p in decoder_params),
        "encoder_parameters": sum(p.numel() for p in encoder_params),
        "encoder_lr_scale": (
            float(getattr(config.optimization, "encoder_lr_scale", 0.2))
            if encoder_params else 0.0
        ),
    }), flush=True)
    global_step = 0
    log_path = output_dir / "train.jsonl"
    start_epoch = 0

    # Gradient accumulation (only needed for Q→position coupling with batch_size=1)
    grad_accum = getattr(config.optimization, 'gradient_accumulation_steps', 1)
    q_only_steps = max(
        0, int(getattr(config.model, "stage_c_q_only_steps", 0))
    )

    # ── Load pre-trained Stage C decoder weights (excluding q_position_head) ──
    # This preserves the converged internal geometry from centered-FAPE training
    # while allowing q_position_head to learn Q→position from scratch.
    stage_c_pretrained = getattr(config, "stage_c_pretrained", None)
    if stage_c_pretrained:
        pt_ckpt = torch.load(stage_c_pretrained, map_location="cpu")
        pt_state = pt_ckpt.get("model", pt_ckpt)
        # Load decoder weights.  Preserve q_position_head for fine-tune
        # continuations when requested; the legacy default remains unchanged.
        # This handles the legacy flat checkpoint layout as well.
        exclude_q_position = bool(
            getattr(config, "stage_c_pretrained_exclude_q_position", True)
        )
        dec_state = {k: v for k, v in pt_state.items()
                     if k in decoder.state_dict()
                     and not (
                         exclude_q_position and k.startswith("q_position_head.")
                     )}
        decoder.load_state_dict(dec_state, strict=False)
        dh_state = {k: v for k, v in pt_state.items() if k in direct_head.state_dict()}
        if dh_state:
            direct_head.load_state_dict(dh_state, strict=False)
        print(json.dumps({"event": "stage_c_loaded_pretrained_decoder",
                          "checkpoint": str(stage_c_pretrained),
                          "loaded_decoder_keys": len(dec_state),
                          "loaded_directhead_keys": len(dh_state),
                          "skipped_q_position": exclude_q_position}), flush=True)

    if config.resume_checkpoint:
        resume = torch.load(config.resume_checkpoint, map_location=device)
        if not isinstance(resume, dict) or "model" not in resume or "optimizer" not in resume:
            raise ValueError(
                f"Stage-C resume checkpoint {config.resume_checkpoint} must contain model and optimizer state"
            )
        _load_stage_c_state(decoder, direct_head, resume["model"])
        if stage_b_model is not None and "stage_b_model" in resume:
            stage_b_model.load_state_dict(resume["stage_b_model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        global_step = int(resume.get("step", 0))
        print(json.dumps({
            "event": "stage_c_resumed_checkpoint",
            "checkpoint": config.resume_checkpoint,
            "epoch": resume.get("epoch"),
            "step": global_step,
        }), flush=True)

    if global_step >= total_steps:
        torch.save(
            _stage_c_checkpoint_payload(
                decoder, direct_head, optimizer,
                config.optimization.epochs, global_step, config, stage_b_model,
            ),
            output_dir / "checkpoint_latest.pt",
        )
        return output_dir / "checkpoint_latest.pt"

    start_epoch = global_step // steps_per_epoch
    last_grad_norm = None
    for epoch in range(start_epoch, config.optimization.epochs):
        first_local_step = global_step % steps_per_epoch if epoch == start_epoch else 0
        iterator = iter(loader)
        for local_step in range(first_local_step, steps_per_epoch):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = _move_batch(batch, device)

            lr_mult = _lr_scale(global_step, total_steps, config.optimization.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = (
                    config.optimization.lr
                    * float(group.get("base_lr_scale", 1.0))
                    * lr_mult
                )

            # Gradient accumulation: only zero grad at start of accumulation cycle
            if local_step % grad_accum == 0:
                optimizer.zero_grad(set_to_none=True)

            # Parse batch (flat format)
            B = batch["context_coords"].shape[0]
            L = batch["context_coords"].shape[2]
            target_coords_raw = batch["future_coords"]  # [B, T_fut, L, 4, 3]
            aatype = batch.get("future_aatype", batch.get("aatype"))
            if aatype.dim() == 1:
                aatype = aatype.unsqueeze(0).expand(B, -1)

            # Horizon time & temperature — must be defined BEFORE encode_state / encode_context
            tau_ps = batch.get("horizon_time_ps", torch.tensor([1000.0], device=device))
            if tau_ps.dim() == 0:
                tau_ps = tau_ps.unsqueeze(0)
            temperature_k = batch.get("context_temperature_k", torch.ones(B, device=device) * 348.0)
            if temperature_k.dim() > 1:
                temperature_k = temperature_k[:, -1]

            # ── Get real H and Q from Stage B representation ──
            if stage_b_model is not None:
                # During the canonical run this is no_grad() and H is
                # detached.  In controlled Stage-C fine-tuning, preserve the
                # context graph so structure loss can adapt the selected
                # spatial/temporal encoder parameters.  Future Q remains a
                # detached teacher target below to avoid moving the target
                # representation and causing representational collapse.
                encoder_trainable = bool(encoder_params)
                with torch.set_grad_enabled(encoder_trainable):
                    # Encode context through frozen Stage B encoder
                    ctx_coords = batch["context_coords"]
                    ctx_res_mask = batch["context_residue_mask"]
                    # Topology
                    t_aatype, t_res_idx, t_chain = aatype, batch.get("context_residue_index", batch.get("residue_index")), batch.get("context_chain_id", batch.get("chain_id"))
                    if t_res_idx is None:
                        t_res_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
                    if t_chain is None:
                        t_chain = torch.zeros(B, L, dtype=torch.long, device=device)
                    if t_aatype.dim() == 1:
                        t_aatype = t_aatype.unsqueeze(0).expand(B, -1)
                    if t_res_idx.dim() == 1:
                        t_res_idx = t_res_idx.unsqueeze(0).expand(B, -1)
                    if t_chain.dim() == 1:
                        t_chain = t_chain.unsqueeze(0).expand(B, -1)

                    # CANONICAL: use encode_state so H includes temporal history,
                    # matching what evaluation scripts produce via the same path.
                    ctx_state = stage_b_model.encode_state(
                        ctx_coords, ctx_res_mask, t_aatype, t_res_idx, t_chain,
                        time_ps=batch.get("context_time_ps"),
                        temperature_k=temperature_k,
                        use_temporal=True,
                        return_bath=False,
                        return_ca_distances=False,
                        return_forces=False,
                    )
                    h_frozen = ctx_state["h_last"]
                    if not encoder_trainable:
                        h_frozen = h_frozen.detach()  # [B, L, D_H]
            else:
                h_frozen = torch.zeros(B, L, 256, device=device)

            # Anchor frames from last context frame
            anchor_backbone = batch["context_coords"][:, -1]  # [B, L, 4, 3]
            R_anchor, t_anchor = _extract_backbone_frames(anchor_backbone)

            # Basin-reversion prior: aligned context-mean CA displacement in the
            # anchor local frame.  Computed from the CLEAN anchor/context so the
            # geometric prior is exact; anchor noise (below) only perturbs the
            # decoder's starting translation, not this eval-legal signal.
            ctx_res_mask_last = batch["context_residue_mask"][:, -1]
            basin_dt_local = StructureDecoder.compute_basin_dt_local(
                batch["context_coords"], R_anchor, t_anchor,
                residue_mask=ctx_res_mask_last.bool(),
            )

            anchor_noise = float(getattr(config.model, "anchor_noise_level", 0.0))
            if anchor_noise > 0.0:
                # Corrupt only the decoder anchor.  H and the oracle Q still
                # come from the uncorrupted observation, making Q useful for
                # endpoint reconstruction instead of allowing a copy shortcut.
                t_anchor = t_anchor + torch.randn_like(t_anchor) * anchor_noise

            # Target structure (last future frame)
            target_backbone = target_coords_raw[:, -1]  # [B, L, 4, 3]
            R_target, t_target = _extract_backbone_frames(target_backbone)
            target_torsions = _compute_target_torsions(target_backbone)
            target_mask = batch["future_residue_mask"][:, -1]

            # Direct endpoint prediction
            q_pred_mean, _ = direct_head(h_frozen, None, tau_ps, temperature_k, target_mask)

            # ── Oracle Q from frozen Stage B encoding of future coords ──
            q_oracle = None
            if stage_b_model is not None:
                with torch.no_grad():
                    # Use encode_state for consistent temporal-history encoding
                    fut_state = stage_b_model.encode_state(
                        batch["future_coords"],
                        batch["future_residue_mask"],
                        t_aatype, t_res_idx, t_chain,
                        time_ps=batch.get("future_time_ps"),
                        temperature_k=temperature_k,
                        use_temporal=True,  # CANONICAL: future Q must use same temporal path as context Q
                        return_bath=False,
                        return_ca_distances=False,
                        return_forces=False,
                    )
                    q_oracle = fut_state["q"]  # [B, L, 8]

            # ── Scheduled sampling: blend predicted Q with oracle Q ──
            # Early training: oracle-dominant (teacher forcing to learn Q→structure)
            # Late training: predicted-dominant (autonomous decoding)
            # q_weight_ramp controls the transition.
            ramp_steps = int(getattr(config.model, "q_weight_ramp_steps", 0))
            ramp_start = int(getattr(config.model, "q_weight_ramp_start", 0))
            q_ramp = 1.0 if ramp_steps <= 0 else min(
                1.0, max(0.0, (global_step - ramp_start) / max(ramp_steps, 1))
            )
            # q_ramp = 0 → all oracle; q_ramp = 1 → all predicted
            q_only_phase = global_step < q_only_steps
            if q_only_phase and q_oracle is not None:
                # Phase 1 uses teacher-forced Q so the explicit Q→position
                # path learns a coordinate-aligned signal before autonomous
                # direct-Q errors are introduced.
                q_decode = q_oracle
            elif q_oracle is not None and q_pred_mean is not None:
                # Blend: early steps use oracle, late steps use predicted
                alpha = q_ramp  # weight on predicted Q
                q_decode = alpha * q_pred_mean + (1.0 - alpha) * q_oracle
            elif q_oracle is not None:
                q_decode = q_oracle
            else:
                q_decode = q_pred_mean

            decoder_outputs = decoder(
                aatype, h_frozen, q_decode, R_anchor, t_anchor, target_mask,
                q_only_translation=q_only_phase,
                basin_dt_local=basin_dt_local,
            )

            target = {
                "backbone_coords": target_backbone,
                "CA": target_backbone[..., 1, :],
                "R": R_target,
                "t": t_target,
                "torsion_angles": target_torsions,
            }

            # Eval-aligned loss weights.  The Kabsch/contact objectives are the
            # ones that actually map onto the reported metrics, so they carry
            # the bulk of the position signal; the legacy anchor-relative
            # terms are demoted to auxiliary regularisers.
            sc_weights = {
                "fape": float(getattr(config.optimization, "sc_fape_weight", 0.2)),
                "torsion": float(getattr(config.optimization, "sc_torsion_weight", 0.5)),
                "distance": float(getattr(config.optimization, "sc_distance_weight", 0.3)),
                "bond": float(getattr(config.optimization, "sc_bond_weight", 0.1)),
                "clash": float(getattr(config.optimization, "sc_clash_weight", 0.05)),
                "ca_position": float(getattr(config.optimization, "sc_ca_position_weight", 0.05)),
                "kabsch_ca": float(getattr(config.optimization, "sc_kabsch_weight", 1.0)),
                "contact_distance": float(getattr(config.optimization, "sc_contact_distance_weight", 1.0)),
                "contact_binary": float(getattr(config.optimization, "sc_contact_binary_weight", 1.0)),
            }
            total_loss, components = stage_c_total_loss(
                decoder_outputs, target, target_mask,
                weights=sc_weights,
                use_frame_aligned=True,
                anchor_CA=anchor_backbone[..., 1, :],
                anchor_R=R_anchor,
                residue_index=t_res_idx,
                chain_id=t_chain,
            )

            # Explicit supervision for the Q-only translation path.  The
            # target is the internal anchor-relative deformation, expressed
            # in each residue's anchor frame; global COM drift is removed.
            q_translation_loss = None
            q_update = decoder_outputs.get("q_translation_update_local")
            if q_update is not None:
                mf_q = target_mask.float().unsqueeze(-1)
                target_disp = target_backbone[..., 1, :] - anchor_backbone[..., 1, :]
                target_disp = target_disp - (
                    target_disp * mf_q
                ).sum(dim=1, keepdim=True) / mf_q.sum(dim=1, keepdim=True).clamp(min=1)
                target_disp_local = (
                    R_anchor.transpose(-2, -1)
                    @ target_disp.unsqueeze(-1)
                ).squeeze(-1)
                # Match the normalization to the motion scale.  The previous
                # fixed 4 A divisor was paired with a 2 A/component bound,
                # while the paired targets are ~12.6 A RMS in local relative
                # displacement; that combination left the Q path saturated.
                q_translation_scale = max(
                    float(getattr(config.optimization,
                                  "q_translation_target_scale", 4.0)),
                    1.0,
                )
                q_translation_loss = F.smooth_l1_loss(
                    q_update / q_translation_scale,
                    target_disp_local.detach() / q_translation_scale,
                )
                q_translation_weight = float(
                    getattr(
                        config.optimization,
                        "q_only_translation_weight" if q_only_phase else "q_translation_weight",
                        10.0 if q_only_phase else 2.0,
                    )
                )
                total_loss = total_loss + q_translation_weight * q_translation_loss

            # Semantic anti-bypass loss.  With a batch size of one, residue
            # permutation is the deterministic negative example available
            # without mixing proteins.  The correct Q must explain the target
            # local displacement better than shuffled Q by a small margin.
            q_shuffle_loss = None
            q_shuffle_weight = float(
                getattr(config.optimization, "q_shuffle_weight", 0.0)
            )
            q_shuffle_margin = float(
                getattr(config.optimization, "q_shuffle_margin", 0.0)
            )
            if (
                q_update is not None
                and q_oracle is not None
                and q_shuffle_weight > 0.0
                and target_disp_local is not None
            ):
                q_shuffle = q_decode.roll(shifts=1, dims=1)
                q_update_shuffle = decoder.q_translation_update(q_shuffle)
                scale = q_translation_scale
                correct_err = F.smooth_l1_loss(
                    q_update / scale,
                    target_disp_local.detach() / scale,
                    reduction="none",
                ).mean(dim=-1)
                shuffled_err = F.smooth_l1_loss(
                    q_update_shuffle / scale,
                    target_disp_local.detach() / scale,
                    reduction="none",
                ).mean(dim=-1)
                valid = target_mask.float()
                q_shuffle_loss = (
                    F.relu(q_shuffle_margin + correct_err - shuffled_err) * valid
                ).sum() / valid.sum().clamp(min=1.0)
                total_loss = total_loss + q_shuffle_weight * q_shuffle_loss

            # Global COM displacement is intentionally not supervised: it is
            # a gauge degree of freedom under the relative-position objective.
            q_position_loss = None

            # ── Gradient accumulation: scale loss for effective batch_size=N ──
            # Accumulate gradients over grad_accum batches before stepping.
            # Essential when batch_size=1 with 60Å anchor→target displacements.
            total_loss = total_loss / grad_accum

            # ── DirectHead MSE loss against oracle Q (direct Q-space supervision) ──
            direct_head_mse = None
            if q_oracle is not None:
                direct_head_mse = F.mse_loss(q_pred_mean, q_oracle)
                total_loss = total_loss + direct_head_mse * 0.5

            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"non-finite Stage C loss at step {global_step}")
            total_loss.backward()

            grad_norm = None
            # Step only after gradient accumulation is complete
            if (local_step + 1) % grad_accum == 0:
                if config.optimization.grad_clip is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        all_params, config.optimization.grad_clip, error_if_nonfinite=True
                    )
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, float("inf"))
                optimizer.step()
                last_grad_norm = float(grad_norm.detach().cpu())

            # Log: use the raw (unscaled) loss for monitoring

            if global_step % config.log_every == 0:
                record = {"epoch": epoch, "step": global_step, "loss": float(total_loss.detach().cpu()), "lr": optimizer.param_groups[0]["lr"]}
                for name, v in components.items():
                    record[f"loss_{name}"] = float(v.detach().cpu())
                if direct_head_mse is not None:
                    record["loss_direct_mse"] = float(direct_head_mse.detach().cpu())
                if q_position_loss is not None:
                    record["loss_q_position"] = float(q_position_loss.detach().cpu())
                if q_translation_loss is not None:
                    record["loss_q_translation"] = float(q_translation_loss.detach().cpu())
                if q_shuffle_loss is not None:
                    record["loss_q_shuffle"] = float(q_shuffle_loss.detach().cpu())
                record["q_only_phase"] = bool(q_only_phase)
                # Accumulation-only records have no current optimizer grad
                # norm; zero would be misleading in the diagnostic log.
                record["grad_norm"] = (
                    last_grad_norm if grad_norm is None
                    else float(grad_norm.detach().cpu())
                )
                record["optimizer_step"] = bool(grad_norm is not None)
                record["translation_update_norm"] = float(
                    decoder_outputs["translation_update_local"].detach().norm(dim=-1).mean().cpu()
                )
                record["q_translation_update_norm"] = float(
                    decoder_outputs["q_translation_update_local"].detach().norm(dim=-1).mean().cpu()
                )
                record["q_translation_target_norm"] = float(
                    target_disp_local.detach().norm(dim=-1).mean().cpu()
                )
                record["delta_com_norm"] = float(
                    decoder_outputs["delta_com_local"].detach().norm(dim=-1).mean().cpu()
                )
                with log_path.open("a") as f:
                    f.write(json.dumps(record) + "\n"); f.flush()
                print(json.dumps(record), flush=True)

            if config.checkpoint_every_steps > 0 and (global_step + 1) % config.checkpoint_every_steps == 0:
                ckpt_path = output_dir / f"step_{global_step+1}.pt"
                torch.save(
                    _stage_c_checkpoint_payload(
                        decoder, direct_head, optimizer,
                        epoch, global_step + 1, config, stage_b_model,
                    ),
                    ckpt_path,
                )

            global_step += 1

            # ── Explicit max_steps guard ──
            if max_steps is not None and global_step >= max_steps:
                torch.save(
                    _stage_c_checkpoint_payload(
                        decoder, direct_head, optimizer,
                        epoch + 1, global_step, config, stage_b_model,
                    ),
                    output_dir / "checkpoint_latest.pt",
                )
                (output_dir / "canonical_state_version.txt").write_text(
                    "v1\nstage_c_canonical\nall encode_state calls use_temporal=True\n"
                    + (
                        "stage_b_representation_finetuned=true\n"
                        if encoder_params else ""
                    )
                )
                return output_dir / "checkpoint_latest.pt"

        if (epoch + 1) % config.save_every == 0 or epoch + 1 == config.optimization.epochs:
            ckpt_path = output_dir / "checkpoint_latest.pt"
            torch.save(
                _stage_c_checkpoint_payload(
                    decoder, direct_head, optimizer,
                    epoch + 1, global_step, config, stage_b_model,
                ),
                ckpt_path,
            )

    return output_dir / "checkpoint_latest.pt"


__all__ = ["train_stage_c"]
