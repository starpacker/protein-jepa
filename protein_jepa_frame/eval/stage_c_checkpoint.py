"""Strict, backward-compatible Stage-C checkpoint loading for evaluation."""

from __future__ import annotations

from typing import Mapping

import torch

from protein_jepa_frame.config import ModelConfig
from protein_jepa_frame.models.stage_c.direct_head import DirectEndpointHead, DirectHeadConfig
from protein_jepa_frame.models.stage_c.structure_decoder import DecoderConfig, StructureDecoder
from protein_jepa_frame.models.stage_b.model import StageBLASP


def split_stage_c_state(decoder, direct_head, state: Mapping[str, torch.Tensor]):
    """Split legacy flat or explicit decoder/direct-head state dictionaries."""
    decoder_keys = set(decoder.state_dict())
    direct_keys = set(direct_head.state_dict())
    if any(k.startswith(("decoder.", "direct_head.")) for k in state):
        dec = {k.removeprefix("decoder."): v for k, v in state.items()
               if k.startswith("decoder.")}
        dh = {k.removeprefix("direct_head."): v for k, v in state.items()
              if k.startswith("direct_head.")}
    else:
        dec = {k: v for k, v in state.items() if k in decoder_keys}
        dh = {k: v for k, v in state.items() if k in direct_keys}
    missing_dec = decoder_keys - set(dec)
    missing_dh = direct_keys - set(dh)
    # Legacy canonical Stage-C checkpoints were saved before several decoder
    # modules were added (q_translation_head, basin_gate).  All of these use
    # zero final-layer initialisation so filling them from the freshly
    # constructed decoder exactly reproduces the legacy identity/zero path.
    _LEGACY_ZERO_PREFIXES = (
        "q_translation_head.",
        "basin_gate.",
    )
    legacy_missing = {
        key for key in missing_dec
        if any(key.startswith(p) for p in _LEGACY_ZERO_PREFIXES)
    }
    if missing_dec and missing_dec == legacy_missing:
        dec.update({key: decoder.state_dict()[key] for key in legacy_missing})
        missing_dec = set()
    if missing_dec or missing_dh:
        raise RuntimeError(
            f"incomplete Stage-C checkpoint: decoder_missing={sorted(missing_dec)}, "
            f"direct_head_missing={sorted(missing_dh)}"
        )
    return dec, dh


def stage_c_configs_from_checkpoint(checkpoint, context_dim: int = 128):
    raw = checkpoint.get("config", {}) if isinstance(checkpoint, Mapping) else {}
    model = raw.get("model", {}) if isinstance(raw, Mapping) else {}
    latent_dim = int(model.get("latent_dim", context_dim))
    q_dim = int(model.get("q_dim", 8))
    decoder_cfg = DecoderConfig(
        context_dim=latent_dim,
        q_dim=q_dim,
        c_hidden=latent_dim,
        max_translation_update=float(model.get("decoder_max_translation_update", 0.5)),
        max_com_update=float(model.get("decoder_max_com_update", 4.0)),
        max_q_translation_update=float(model.get("decoder_max_q_translation_update", 2.0)),
    )
    direct_cfg = DirectHeadConfig(
        context_dim=latent_dim, condition_dim=0,
        latent_dim=q_dim, hidden_dim=latent_dim,
    )
    return decoder_cfg, direct_cfg


def load_stage_c_modules(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    decoder_cfg, direct_cfg = stage_c_configs_from_checkpoint(checkpoint)
    decoder = StructureDecoder(decoder_cfg).to(device)
    direct_head = DirectEndpointHead(direct_cfg).to(device)
    dec_state, dh_state = split_stage_c_state(decoder, direct_head, state)
    decoder.load_state_dict(dec_state, strict=True)
    direct_head.load_state_dict(dh_state, strict=True)
    decoder.eval()
    direct_head.eval()
    return decoder, direct_head, checkpoint


def load_stage_b_for_stage_c(
    stage_c_checkpoint,
    stage_b_checkpoint_path: str | None,
    device: torch.device,
):
    """Load the Stage-B representation paired with a Stage-C checkpoint.

    Fine-tuned Stage-C checkpoints embed the adapted Stage-B state.  Prefer
    that state over the original Stage-B checkpoint; otherwise evaluation
    would silently decode with a different representation than training.
    Legacy frozen-AB checkpoints continue to use ``stage_b_checkpoint_path``.
    """
    embedded = stage_c_checkpoint.get("stage_b_model")
    if embedded is not None:
        sb_checkpoint = stage_c_checkpoint
        state = embedded
    else:
        if not stage_b_checkpoint_path:
            raise ValueError(
                "Stage-C checkpoint has no embedded Stage-B state and no "
                "stage-B checkpoint was supplied"
            )
        sb_checkpoint = torch.load(
            stage_b_checkpoint_path, map_location="cpu", weights_only=False
        )
        state = sb_checkpoint.get("model", sb_checkpoint)

    raw = sb_checkpoint.get("config", {}) if isinstance(sb_checkpoint, dict) else {}
    sb_cfg = raw.get("model", {}) if isinstance(raw, dict) else {}
    mc = ModelConfig(
        latent_dim=sb_cfg.get("latent_dim", 128),
        encoder_depth=sb_cfg.get("encoder_depth", 4),
        temporal_depth=sb_cfg.get("temporal_depth", 2),
        num_heads=sb_cfg.get("num_heads", 4),
        num_rbf=sb_cfg.get("num_rbf", 16),
        max_frames=sb_cfg.get("max_frames", 128),
        max_residues=sb_cfg.get("max_residues", 256),
        dropout=0.0,
        orientation_features=False,
        time_conditioning=sb_cfg.get("time_conditioning", "physical"),
        thermodynamic_conditioning=sb_cfg.get("thermodynamic_conditioning", True),
        lspwm_base_step_ps=sb_cfg.get("lspwm_base_step_ps", 100.0),
        lspwm_diffusion_scale=sb_cfg.get("lspwm_diffusion_scale", 1.0e-6),
        lspwm_coupling_scale=sb_cfg.get("lspwm_coupling_scale", 1.0e-3),
    )
    remapped = {}
    for key, value in state.items():
        if key.startswith("context_encoder."):
            remapped["spatial_encoder." + key.removeprefix("context_encoder.")] = value
        else:
            remapped[key] = value
    # The early canonical Stage-B checkpoints were trained with an energy
    # potential that had no condition channel (its first node layer is 137
    # wide); a later decoder experiment instantiated a one-channel placeholder
    # to avoid a zero-width tensor (138 wide).  The checkpoint, not the current
    # default constructor, is authoritative for this compatibility dimension.
    # Infer it from the saved layer and instantiate the matching module before
    # loading.  This prevents evaluation from silently using a different
    # Stage-B representation or failing before any sample is scored.
    node_weight = remapped.get("energy_graph.node_nets.0.0.weight")
    edge_weight = remapped.get("energy_graph.edge_net.0.weight")
    head_weight = remapped.get("energy_graph.energy_head.0.weight")
    if node_weight is not None and edge_weight is not None and head_weight is not None:
        node_in = int(node_weight.shape[1])
        edge_in = int(edge_weight.shape[1])
        head_in = int(head_weight.shape[1])
        # Canonical dimensions: q=8, RBF=8, message hidden=128.  Older
        # checkpoints differ only in whether the temperature feature was
        # present in node/head inputs; the edge condition remains explicit.
        condition_dim = edge_in - 2 * 8 - 8
        has_temperature = node_in == 8 + 128 + condition_dim + 1
        expected_node = 8 + 128 + condition_dim + (1 if has_temperature else 0)
        expected_head = 8 + condition_dim + (1 if has_temperature else 0)
        if condition_dim not in (0, 1) or node_in != expected_node or head_in != expected_head:
            raise RuntimeError(
                "unsupported Stage-B energy layout: "
                f"edge={edge_in}, node={node_in}, head={head_in}, "
                f"condition={condition_dim}"
            )
    else:
        condition_dim = 0
        has_temperature = True
    sb = StageBLASP(
        mc,
        condition_dim=condition_dim,
        use_bath=sb_cfg.get("lspwm_use_bath", True),
        history_aware=sb_cfg.get("lspwm_history_aware", True),
        energy_temperature_feature=has_temperature,
    ).to(device)
    sb.load_state_dict(remapped, strict=True if embedded is not None else False)
    sb.eval()
    for parameter in sb.parameters():
        parameter.requires_grad = False
    return sb, mc, ("embedded_stage_b" if embedded is not None else "external_stage_b")


__all__ = [
    "split_stage_c_state",
    "stage_c_configs_from_checkpoint",
    "load_stage_c_modules",
    "load_stage_b_for_stage_c",
]
