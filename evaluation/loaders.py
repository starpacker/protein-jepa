"""Checkpoint loading → a uniform ModelBundle.

Knows the on-disk formats of Stage A V2 / B1a / B1b checkpoints.  Knows nothing
about metrics.  Every variant is driven through the SAME encoder call so that a
comparison is apples-to-apples.

Formats (see evaluation/MODELS.md):
  Stage A V2 : {'model': <StageAV2 sd>, 'stage_a_v2_config': {...}, 'step': int}
  B1a        : {'oracle': <StageBOracleDecoder sd>, 'encoder': <enc sd>|absent,
                'stage_b1a_config': {...}, 'step': int}
  B1b        : {'model': <StageB1b sd>, 'decoder': <ResidualCADecoder sd>,
                'decoder_config': {...}, 'stage_b1b_config': {...}, 'step': int}
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from protein_jepa_frame.models.stage_a_v2 import StageAV2, StageAV2Config
from protein_jepa_frame.models.stage_a_v2.model import CausalTransformerPredictor
from protein_jepa_frame.models.stage_b1a import StageBOracleConfig, StageBOracleDecoder
from protein_jepa_frame.models.stage_b1a.model import HNormalizer
from protein_jepa_frame.models.stage_b1b import (
    ResidualCADecoder,
    ResidualDecoderConfig,
    StageB1b,
    StageB1bConfig,
)

from jepa_anything_core.opf import OrthogonalFactorProjection

from .compat.stage_a_predictor_v1 import LegacyCausalTransformerPredictor


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ModelBundle:
    """Frozen modules + a single canonical encode() call."""
    encoder: nn.Module
    h_normalizer: nn.Module
    decoder: nn.Module | None            # learned decoder (None => identity/persistence)
    decoder_kind: str                    # 'oracle' | 'residual' | 'identity'
    predictor: nn.Module | None = None   # H_context -> predicted factor delta
    opf: nn.Module | None = None         # factor delta <-> latent H coordinates
    b1b: StageB1b | None = None          # GLE dynamics for h_plus_s
    b1b_cfg: StageB1bConfig | None = None
    device: torch.device = torch.device("cpu")
    provenance: dict = field(default_factory=dict)
    latent_dim: int = 128

    @torch.no_grad()
    def predict_future(
        self,
        h_context: torch.Tensor,
        lag_ps: torch.Tensor,
        temperature_k: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one future latent frame from the complete context history.

        The Stage-A checkpoint contract is delta-factor prediction, not direct
        absolute-H prediction:

            pred_factors = predictor(H_context, tau, T)
            H_future = OPF.compose(OPF.decompose(H_anchor) + pred_factors)

        Thus the returned tensor is the actual ``H_future`` consumed by the
        shared decoder and has shape ``[B,L,dH]``.
        """
        if self.predictor is None or self.opf is None:
            raise ValueError("this bundle has no Stage-A predictor/OPF")
        if h_context.ndim != 4:
            raise ValueError(f"h_context must be [B,K,L,dH], got {tuple(h_context.shape)}")
        pred_factors = self.predictor(
            h_context, lag_ps, temperature_k, valid_mask=valid_mask, task_id=1,
        )
        h_anchor = h_context[:, -1]
        return self.opf.compose(self.opf.decompose(h_anchor) + pred_factors)

    @torch.no_grad()
    def encode(self, coords, residue_mask, aatype, residue_index, chain_id,
               normalize: bool = False) -> torch.Tensor:
        """Encode a batch of T frames per protein → H [B, T, L, dH].

        Args:
            coords: [B, T, L, 4, 3]; others: [B, L] or [B, T, L]
            normalize: apply the B1a H-normalizer.  DEFAULT False because the
                structure decoder / B1b dynamics were trained on RAW H
                (H_DECODABILITY_REPORT §9.1; train_stage_b1a `_encode_frame`
                returns raw H).  Only use True where normalization was actually
                part of the trained path.
        """
        B, T, L = coords.shape[:3]

        def _expand(idx):
            # idx: [B,L] or [B,T,L] -> [B*T, L]
            if idx.dim() == 2:
                return idx[:, None].expand(B, T, L).reshape(B * T, L)
            return idx.reshape(B * T, L)

        H = self.encoder(
            coords.reshape(B * T, 1, L, 4, 3),
            residue_mask.reshape(B * T, 1, L),
            _expand(aatype),
            _expand(residue_index),
            _expand(chain_id),
            token_mask=residue_mask.reshape(B * T, 1, L),
        )
        H = H.reshape(B, T, L, -1)
        return self.h_normalizer(H) if normalize else H


def _filter_kwargs(cls, raw: dict) -> dict:
    """Keep only keys accepted by `cls` (checkpoints may carry stale/removed keys)."""
    return {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}


def _extract_encoder_state(state: dict) -> dict:
    """Return a bare ProteinGeometryEncoder state dict.

    Accepts either a bare encoder state dict (keys like `geom_proj.weight`) or a
    full StageAV2 state dict (keys prefixed `context_encoder.`).
    """
    if any(k.startswith("context_encoder.") for k in state):
        prefix = "context_encoder."
        return {k[len(prefix):]: v for k, v in state.items()
                if k.startswith(prefix) and not k.endswith("_ema_step")}
    return dict(state)


def _infer_encoder_flags(state: dict) -> dict:
    """Infer encoder/config knobs from weight shapes.

    Handles the two lineages we have on disk, whose *saved config dicts are
    unreliable* (edited after training):
      * geom_proj.weight in-dim 12 => orientation_features=True (else 6)
      * struct_recon_head.0.weight in-dim => encoder feature width used at train
        time (only relevant for loading StageAV2 strictly; not for the encoder).
    """
    flags: dict = {}
    s = _extract_encoder_state(state)
    gp = s.get("geom_proj.weight")
    if gp is not None:
        flags["orientation_features"] = bool(gp.shape[1] == 12)
    return flags


def _repair_stage_a_config(cfg_kwargs: dict, state: dict) -> dict:
    kw = dict(cfg_kwargs)
    kw.update(_infer_encoder_flags(state))
    return kw


@torch.no_grad()
def _load_encoder_from_state(state: dict, cfg: StageAV2Config, device) -> nn.Module:
    """Build a ProteinGeometryEncoder that matches `state` and load it strictly.

    We build the encoder directly (not via StageAV2) so that a StageAV2-level
    aux head (struct_recon) shape drift does not block us -- we only ever need
    the encoder.
    """
    from protein_jepa_frame.models.encoder import ProteinGeometryEncoder
    enc = ProteinGeometryEncoder(
        latent_dim=cfg.latent_dim,
        depth=cfg.encoder_depth,
        num_heads=cfg.num_heads,
        num_rbf=cfg.num_rbf,
        dropout=cfg.dropout,
        max_residues=cfg.max_residues,
        orientation_features=cfg.orientation_features,
        identity_embed_scale=cfg.identity_embed_scale,
    ).to(device)
    enc.load_state_dict(_extract_encoder_state(state), strict=True)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def load_encoder_stage_a(ckpt_path: str, device, provenance: dict) -> tuple[nn.Module, StageAV2Config]:
    """Load only the Stage A context encoder (frozen).

    Prefers the StageAV2 full checkpoint's `model` state dict; falls back to a
    bare B1a `encoder` state dict when the caller passes one.
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ck.get("model")
    if state is None:
        state = ck["encoder"]
    cfg_raw = ck.get("stage_a_v2_config") or ck.get("stage_b1a_config", {})
    cfg = StageAV2Config(**_filter_kwargs(StageAV2Config, cfg_raw))
    flags = _infer_encoder_flags(state)
    for k, v in flags.items():
        setattr(cfg, k, v)
    if "stage_a_v2_config" in ck:
        enc = _load_encoder_from_state(ck["model"], cfg, device)
    else:
        enc = _load_encoder_from_state(ck["encoder"], cfg, device)
    provenance[f"encoder:{ckpt_path}"] = sha256_file(ckpt_path)
    provenance["encoder_step"] = ck.get("step")
    provenance["encoder_flags"] = flags
    return enc, cfg


def _load_b1a_oracle(ck, device):
    """Load a B1a oracle decoder, choosing the architecture version by probing
    the state-dict keys.

    The working tree rewrote StructureDecoder AFTER these checkpoints were made,
    so all legacy oracles need the vendored v1 code (evaluation/compat/).  We
    pick v1 if the state dict has the legacy key `decoder.h_input_norm.weight`.
    """
    sd = ck["oracle"]
    if "decoder.h_input_norm.weight" in sd:
        from evaluation.compat.stage_b1a_v1 import StageBOracleDecoder as _Oracle
        from evaluation.compat.stage_b1a_v1 import StageBOracleConfig as _Cfg
        version = "v1(legacy)"
    else:
        _Oracle = StageBOracleDecoder
        _Cfg = StageBOracleConfig
        version = "current"
    cfg = _Cfg(**_filter_kwargs(_Cfg, ck["stage_b1a_config"]))
    oracle = _Oracle(cfg).to(device)
    report = oracle.load_state_dict(sd, strict=False)
    if report.missing_keys or report.unexpected_keys:
        raise RuntimeError(
            f"B1a oracle decoder mismatch [{version}]: "
            f"missing={report.missing_keys[:4]} unexpected={report.unexpected_keys[:4]}"
        )
    oracle.eval()
    for p in oracle.parameters():
        p.requires_grad_(False)
    return oracle, version


def load_b1a(ckpt_path: str, device, provenance: dict):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    oracle, version = _load_b1a_oracle(ck, device)
    provenance[f"b1a:{ckpt_path}"] = sha256_file(ckpt_path)
    provenance["b1a_step"] = ck.get("step")
    provenance["b1a_decoder_version"] = version
    return None, oracle, oracle.h_normalizer


def load_b1b(ckpt_path: str, device, provenance: dict) -> tuple[StageB1b, ResidualCADecoder, StageB1bConfig]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    b_cfg = StageB1bConfig(**_filter_kwargs(StageB1bConfig, ck["stage_b1b_config"]))
    model = StageB1b(b_cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dec_cfg = ck.get("decoder_config")
    dec_cfg = ResidualDecoderConfig(**_filter_kwargs(ResidualDecoderConfig, dec_cfg)) if dec_cfg else ResidualDecoderConfig(latent_dim=b_cfg.latent_dim)
    dec = ResidualCADecoder(dec_cfg).to(device)
    dec.load_state_dict(ck["decoder"])
    dec.eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    provenance[f"b1b:{ckpt_path}"] = sha256_file(ckpt_path)
    provenance["b1b_step"] = ck.get("step")
    return model, dec, b_cfg


@torch.no_grad()
def load_stage_a_predictor(
    ckpt_path: str,
    device,
    provenance: dict,
) -> tuple[nn.Module, nn.Module, StageAV2Config]:
    """Load the predictor and OPF from a full Stage-A checkpoint.

    Old ``stage_a_fix4`` checkpoints use ``predictor.transformer``; newer
    checkpoints use ``predictor.temporal_transformer`` plus a spatial block.
    Both implement the same public contract and are selected from the state
    dict rather than guessed from mutable config files.
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ck.get("model")
    if state is None:
        raise ValueError(f"Stage-A predictor source has no 'model': {ckpt_path}")
    cfg = StageAV2Config(**_filter_kwargs(StageAV2Config, ck.get("stage_a_v2_config", {})))
    cfg = StageAV2Config(**_repair_stage_a_config(ck.get("stage_a_v2_config", {}), state))

    pred_state = {
        k[len("predictor."):]: v for k, v in state.items()
        if k.startswith("predictor.")
    }
    if any(k.startswith("transformer.") for k in pred_state):
        predictor = LegacyCausalTransformerPredictor(
            latent_dim=cfg.latent_dim,
            num_factors=cfg.num_factors,
            factor_dim=cfg.factor_dim,
            depth=cfg.predictor_depth,
            num_heads=cfg.predictor_heads,
            max_frames=cfg.max_frames,
            dropout=cfg.dropout,
        ).to(device)
        predictor_version = "v1-legacy"
    else:
        predictor = CausalTransformerPredictor(
            latent_dim=cfg.latent_dim,
            num_factors=cfg.num_factors,
            factor_dim=cfg.factor_dim,
            depth=cfg.predictor_depth,
            num_heads=cfg.predictor_heads,
            max_frames=cfg.max_frames,
            dropout=cfg.dropout,
            spatial_depth=cfg.predictor_spatial_depth,
        ).to(device)
        predictor_version = "v2-current"
    report = predictor.load_state_dict(pred_state, strict=True)
    if report.missing_keys or report.unexpected_keys:
        raise RuntimeError(f"Stage-A predictor mismatch: {report}")

    opf_state = {
        k[len("opf."):]: v for k, v in state.items() if k.startswith("opf.")
    }
    opf = OrthogonalFactorProjection(
        state_dim=cfg.latent_dim,
        num_factors=cfg.num_factors,
        factor_dim=cfg.factor_dim,
        learnable=cfg.opf_learnable,
        orthogonality_mode="soft_gram",
    ).to(device)
    opf.load_state_dict(opf_state, strict=True)

    predictor.eval()
    opf.eval()
    for module in (predictor, opf):
        for p in module.parameters():
            p.requires_grad_(False)
    provenance[f"predictor:{ckpt_path}"] = sha256_file(ckpt_path)
    provenance["predictor_version"] = predictor_version
    return predictor, opf, cfg


def build_bundle(
    *,
    variant: str,
    stage_a_ckpt: str | None = None,
    b1a_ckpt: str | None = None,
    b1b_ckpt: str | None = None,
    stage_a_predictor_ckpt: str | None = None,
    device: str | torch.device = "cpu",
    mock_encoder: bool = False,
) -> ModelBundle:
    """Assemble the frozen modules for one variant.

    variant:
      'h_only'   -> Stage A encoder + Stage-A predictor/OPF + B1a decoder
      'h_ft'     -> B1a fine-tuned encoder + the explicit Stage-A predictor/OPF + B1a decoder
      'h_plus_s' -> B1a fine-tuned encoder + B1a normalizer + B1b GLE + residual decoder
      'persist'  -> identity (no learned modules)

    Encoder source:
      * 'h_only'   uses stage_a_ckpt.
      * 'h_ft'/'h_plus_s' use the fine-tuned encoder stored inside b1a_ckpt.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    prov: dict = {"variant": variant, "device": str(device)}

    if variant == "persist":
        return ModelBundle(encoder=nn.Identity(), h_normalizer=nn.Identity(), decoder=None,
                           decoder_kind="identity", device=device, provenance=prov)

    if mock_encoder:
        return _mock_bundle(variant, device, prov)

    # --- load B1a (oracle decoder + normalizer) first; it also supplies the
    #     fine-tuned encoder for h_ft / h_plus_s -------------------------------
    oracle = None
    h_norm: nn.Module = nn.Identity()
    b1a_ck = None
    if b1a_ckpt is not None:
        b1a_ck = torch.load(b1a_ckpt, map_location=device, weights_only=False)
        oracle, b1a_version = _load_b1a_oracle(b1a_ck, device)
        h_norm = oracle.h_normalizer
        prov[f"b1a:{b1a_ckpt}"] = sha256_file(b1a_ckpt)
        prov["b1a_step"] = b1a_ck.get("step")
        prov["b1a_decoder_version"] = b1a_version

    # --- choose the encoder checkpoint --------------------------------------
    if variant in ("h_ft", "h_plus_s"):
        if b1a_ck is None or b1a_ck.get("encoder") is None:
            raise ValueError(f"{variant} needs b1a_ckpt containing a fine-tuned 'encoder'")
        enc_ckpt = b1a_ckpt
        prov["encoder_from_b1a"] = True
    else:  # h_only
        if stage_a_ckpt is None:
            raise ValueError("h_only needs --stage-a-ckpt")
        enc_ckpt = stage_a_ckpt

    enc, a_cfg = load_encoder_stage_a(enc_ckpt, device, prov)
    latent_dim = a_cfg.latent_dim

    predictor = opf = None
    if variant in ("h_only", "h_ft"):
        if oracle is None:
            raise ValueError(f"{variant} needs b1a_ckpt for the decoder")
        predictor_ckpt = stage_a_predictor_ckpt or stage_a_ckpt
        if predictor_ckpt is None:
            raise ValueError(f"{variant} needs stage_a_predictor_ckpt (or stage_a_ckpt)")
        predictor, opf, p_cfg = load_stage_a_predictor(
            predictor_ckpt, device, prov,
        )
        if p_cfg.latent_dim != latent_dim:
            raise ValueError(
                f"encoder/predictor latent mismatch: {latent_dim} vs {p_cfg.latent_dim}"
            )
        return ModelBundle(encoder=enc, h_normalizer=h_norm, decoder=oracle,
                           decoder_kind="oracle", predictor=predictor, opf=opf,
                           device=device, provenance=prov,
                           latent_dim=latent_dim)

    # h_plus_s
    if b1b_ckpt is None:
        raise ValueError("h_plus_s requires --b1b-ckpt")
    model, dec, b1b_cfg = load_b1b(b1b_ckpt, device, prov)
    return ModelBundle(encoder=enc, h_normalizer=h_norm, decoder=dec,
                       decoder_kind="residual", b1b=model, b1b_cfg=b1b_cfg,
                       device=device, provenance=prov, latent_dim=latent_dim)


def _mock_bundle(variant, device, prov):
    """Zero-parameter encoder for fast harness smoke tests (no checkpoint needed).

    Returns a fixed random projection so downstream modules run without weights.
    """
    dH = 128

    class _MockEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4 * 3 + 3, dH)

        def forward(self, coords, residue_mask, aatype, residue_index, chain_id, token_mask=None):
            B, T, L = coords.shape[:3]
            feat = torch.cat([coords.reshape(B * T, L, 12),
                              aatype.float().unsqueeze(-1).expand(B * T, L, 1),
                              residue_index.float().unsqueeze(-1).expand(B * T, L, 1),
                              chain_id.float().unsqueeze(-1).expand(B * T, L, 1)], dim=-1)
            return self.proj(feat).unsqueeze(1)

    prov["mock_encoder"] = True
    return ModelBundle(encoder=_MockEnc().to(device), h_normalizer=nn.Identity(),
                       decoder=None, decoder_kind="identity", device=device,
                       provenance=prov)
