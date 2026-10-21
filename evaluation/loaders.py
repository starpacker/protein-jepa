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
    representation_contract: dict | None = None
    predictor_output: str = "delta_factors"
    diagnostic_only: bool = False

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
        if self.predictor is None:
            raise ValueError("this bundle has no predictor")
        if h_context.ndim != 4:
            raise ValueError(f"h_context must be [B,K,L,dH], got {tuple(h_context.shape)}")
        if self.predictor_output in {"normalized_H_delta","normalized_H_absolute"}:
            normalized = self.normalize(h_context)
            pred = self.predictor(normalized,lag_ps,temperature_k,valid_mask=valid_mask)
            return self.inverse(normalized[:,-1]+pred if self.predictor_output == "normalized_H_delta" else pred)
        if self.opf is None:
            raise ValueError("factor prediction requires its Stage-A OPF")
        pred_factors = self.predictor(
            h_context, lag_ps, temperature_k, valid_mask=valid_mask, task_id=1,
        )
        h_anchor = h_context[:, -1]
        if self.predictor_output == "delta_factors":
            return self.opf.compose(self.opf.decompose(h_anchor) + pred_factors)
        if self.predictor_output == "absolute_factors" and self.diagnostic_only:
            return self.opf.compose(pred_factors)
        raise ValueError(f"unsupported predictor output contract: {self.predictor_output}")

    def normalize(self, h_raw):
        return self.h_normalizer(h_raw)

    def inverse(self, h_normalized):
        if isinstance(self.h_normalizer, nn.Identity):
            return h_normalized
        if hasattr(self.h_normalizer, "fitted_") and not self.h_normalizer.fitted_.item():
            return h_normalized
        return self.h_normalizer.inverse(h_normalized)

    @torch.no_grad()
    def encode(self, coords, residue_mask, aatype, residue_index, chain_id,
               normalize: bool = False) -> torch.Tensor:
        """Encode a batch of T frames per protein → H [B, T, L, dH].

        Args:
            coords: [B, T, L, 4, 3]; others: [B, L] or [B, T, L]
            normalize: optional explicit B1a standardization. The canonical
                encoder/predictor/decoder interface is raw H. Autonomous
                dynamics normalize the complete history exactly once.
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
    # B1b model is a dynamics state dict; the representation encoder is explicit.
    state = ck["encoder"] if "encoder" in ck else ck["model"]
    cfg_raw = ck.get("stage_a_v2_config") or ck.get("stage_b1a_config", {})
    # Encoder construction does not interpret predictor semantics. Legacy False
    # cannot enter the new training config; its output contract is explicit below.
    cfg = StageAV2Config(**{**_filter_kwargs(StageAV2Config, cfg_raw), "delta_target": True})
    flags = _infer_encoder_flags(state)
    for k, v in flags.items():
        setattr(cfg, k, v)
    enc = _load_encoder_from_state(state, cfg, device)
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
    cfg = StageAV2Config(**{**_repair_stage_a_config(
        _filter_kwargs(StageAV2Config, ck.get("stage_a_v2_config", {})), state), "delta_target": True})

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
        orthogonality_mode=("soft_gram" if cfg.opf_mode == "fixed_orthogonal" else cfg.opf_mode),
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


def _validate_bundle_checkpoint(ck: dict, *, allow_legacy: bool) -> bool:
    """Legacy files are opt-in diagnostics and never headline evidence."""
    if not ck.get("representation_contract"):
        if not allow_legacy:
            raise ValueError("checkpoint has no representation_contract; legacy results require explicit allow_legacy and are diagnostic only")
        return False
    required = ["encoder", "stage_a_v2_config", "oracle", "stage_b1a_config"]
    missing = [k for k in required if k not in ck]
    if missing:
        raise ValueError(f"incomplete self-contained representation bundle: {missing}")
    return True


def _training_provenance(ck, contract):
    """Audit saved lineage hashes without treating filenames as compatibility."""
    from pathlib import Path
    source = contract.get("source") or {}
    records = []
    def add(manifest, expected, stage):
        if manifest:
            actual = sha256_file(manifest)
            if expected and actual != expected:
                raise ValueError(f"{stage} training manifest changed since checkpoint calibration")
            records.append(dict(stage=stage,manifest=str(Path(manifest).resolve()),sha256=actual,
                                saved_hash_verified=bool(expected)))
    own_args = ck.get("train_args",{})
    own = own_args.get("manifest")
    split = ck.get("manifest_split") or ck.get("split_audit") or {}
    own_expected = (ck.get("train_manifest_sha256") or split.get("train_manifest_sha256")
                    or source.get("b1a_train_manifest_sha256"))
    add(own, own_expected, "checkpoint")
    stage_a_path = source.get("stage_a_checkpoint")
    stage_a_manifest = None
    upstream_synthetic = False
    if stage_a_path:
        actual = sha256_file(stage_a_path)
        expected = source.get("stage_a_checkpoint_sha256")
        if expected and actual != expected:
            raise ValueError("Stage-A source checkpoint changed since calibration; use the locked source snapshot")
        stage_a = torch.load(stage_a_path,map_location="cpu",weights_only=False)
        args = stage_a.get("train_args", stage_a.get("args",{}))
        stage_a_manifest = args.get("manifest")
        upstream_synthetic = bool(args.get("synthetic",False))
        if not stage_a_manifest and not upstream_synthetic:
            raise ValueError("Stage-A source checkpoint has no actual training manifest provenance")
        add(stage_a_manifest,source.get("stage_a_train_manifest_sha256"),"stage_a")
    complete = bool(own and own_expected and stage_a_path
                    and source.get("stage_a_checkpoint_sha256") and stage_a_manifest
                    and source.get("stage_a_train_manifest_sha256"))
    if (ck.get("training_contract") == "autonomous_history_only_v2"
            or ck.get("comparison_model") == "direct_jepa"):
        # B1b's own manifest does not establish which data fitted its frozen
        # B1a readout. Keep every learned stage in the declared split audit.
        b1a_path = own_args.get("b1a_ckpt")
        if b1a_path:
            b1a = torch.load(b1a_path,map_location="cpu",weights_only=False)
            from protein_jepa_frame.utils.contracts import assert_same_representation
            assert_same_representation(contract,b1a.get("representation_contract"))
            gate_hash = (ck.get("oracle_gate") or {}).get("checkpoint_sha256")
            if gate_hash and sha256_file(b1a_path) != gate_hash:
                raise ValueError("B1a source checkpoint changed since its oracle gate")
            b1a_prov = _training_provenance(b1a,b1a["representation_contract"])
            records.extend(b1a_prov["training_manifests"])
            complete = complete and b1a_prov["training_lineage_complete"]
            upstream_synthetic = upstream_synthetic or b1a_prov["synthetic_lineage"]
        else:
            complete = False
    return dict(training_manifests=records,upstream_source_hash_verified=bool(
        stage_a_path and source.get("stage_a_checkpoint_sha256")),
        training_lineage_complete=complete,
        synthetic_lineage=bool(source.get("synthetic",False) or own_args.get("synthetic",False)
                               or upstream_synthetic))


def build_bundle(
    *, variant: str, stage_a_ckpt: str | None = None,
    b1a_ckpt: str | None = None, b1b_ckpt: str | None = None,
    stage_a_predictor_ckpt: str | None = None, device="cpu",
    direct_ckpt: str | None = None,
    mock_encoder: bool = False, allow_legacy: bool = False,
    legacy_encoder_ckpt: str | None = None,
    legacy_predictor_output: str | None = None,
    diagnostic_comparison: bool = False,
) -> ModelBundle:
    """Load one verified representation; dynamics use normalized H, decoder raw H.

    B1b carries its frozen B1a bundle. An external B1a, when provided, must be
    identical. A direct predictor is valid only for its actual encoder weights.
    Legacy checkpoint adapters are explicitly diagnostic and cannot be ranked.
    """
    device = torch.device(device)
    prov = {"variant": variant, "device": str(device)}
    if variant == "persist":
        return ModelBundle(nn.Identity(), nn.Identity(), None, "identity", device=device, provenance=prov)
    if mock_encoder:
        return _mock_bundle(variant, device, prov)
    if variant not in {"h_only", "h_ft", "direct_jepa", "h_markov", "h_plus_s", "oracle"}:
        raise ValueError(f"unknown variant: {variant}")
    dynamics_variant = variant in {"h_markov", "h_plus_s"}
    source = b1b_ckpt if dynamics_variant else (direct_ckpt if variant=="direct_jepa" and direct_ckpt else b1a_ckpt)
    if not source:
        raise ValueError(f"{variant} requires its representation checkpoint")
    ck = torch.load(source, map_location=device, weights_only=False)
    verified = _validate_bundle_checkpoint(ck, allow_legacy=allow_legacy)
    if not verified:
        return _legacy_bundle(variant, source, stage_a_ckpt, b1a_ckpt,
                              stage_a_predictor_ckpt, legacy_encoder_ckpt,
                              legacy_predictor_output, device, prov)
    from protein_jepa_frame.utils.contracts import (
        validate_representation_contract, validate_prediction_contract,
        assert_same_representation, state_sha256,
    )
    enc, enc_cfg = load_encoder_stage_a(source, device, prov)
    oracle, version = _load_b1a_oracle(ck, device)
    if not oracle.h_normalizer.fitted_.item():
        raise ValueError("schema-v2 representation requires its fitted train-only H normalizer")
    contract = ck["representation_contract"]
    validate_representation_contract(
        contract, encoder_state=enc.state_dict(), encoder_config=ck["stage_a_v2_config"],
        normalizer_state=oracle.h_normalizer.state_dict(), decoder_state=oracle.decoder.state_dict(),
        decoder_config=ck["stage_b1a_config"],
    )
    if contract.get("calibration_mode") != "oracle":
        raise ValueError("headline evaluation requires a true-H oracle/calibration bundle")
    if dynamics_variant and b1a_ckpt:
        other = torch.load(b1a_ckpt, map_location="cpu", weights_only=False)
        _validate_bundle_checkpoint(other, allow_legacy=False)
        assert_same_representation(contract, other["representation_contract"])
    if variant == "h_only" and stage_a_ckpt:
        stage_a = torch.load(stage_a_ckpt, map_location="cpu", weights_only=False)
        if state_sha256(_extract_encoder_state(stage_a["model"])) != state_sha256(enc.state_dict()):
            raise ValueError("Stage-A encoder differs from decoder representation; train a matched oracle bundle")
    prov.update(representation_checkpoint=source, representation_sha256=sha256_file(source),
                representation_contract=contract, b1a_decoder_version=version,
                diagnostic_only=False, latent_input="raw_H", dynamics_input="normalized_H",
                training_variant=ck.get("training_variant",variant))
    prov.update(_training_provenance(ck,contract))
    if ck.get("run_scope") == "contract_smoke_only":
        prov.update(diagnostic_only=True, run_scope="contract_smoke_only")
    if ck.get("run_scope") == "diagnostic_comparison":
        if not diagnostic_comparison:
            raise ValueError("failed/unpassed oracle gate requires explicit diagnostic_comparison evaluation")
        prov.update(diagnostic_only=True, run_scope="diagnostic_comparison",
                    oracle_gate=ck.get("oracle_gate"),
                    limitation="oracle readout gate not established; comparisons are diagnostic")
    common = dict(encoder=enc, h_normalizer=oracle.h_normalizer, decoder=oracle,
                  decoder_kind="oracle", device=device, provenance=prov,
                  latent_dim=enc_cfg.latent_dim, representation_contract=contract,
                  diagnostic_only=prov["diagnostic_only"])
    if variant == "oracle":
        return ModelBundle(**common)
    if variant == "direct_jepa" and direct_ckpt:
        prediction = ck.get("prediction_contract") or {}
        if (prediction.get("schema_version") != 2
                or prediction.get("representation_id") != contract["representation_id"]
                or prediction.get("predictor_input") != "normalized_H_history"
                or prediction.get("predictor_output") not in {"normalized_H_delta","normalized_H_absolute"}):
            raise ValueError("cached direct JEPA requires explicit matched normalized-H prediction contract")
        from protein_jepa_frame.training.train_cached_comparison import DirectJEPAPredictor
        predictor = DirectJEPAPredictor(**ck["direct_predictor_config"]).to(device)
        predictor.load_state_dict(ck["direct_predictor"],strict=True)
        predictor.eval().requires_grad_(False)
        prov.update(direct_predictor_config=ck["direct_predictor_config"],
                    predictor_parameter_count=sum(p.numel() for p in predictor.parameters()),
                    predictor_output=prediction["predictor_output"])
        return ModelBundle(**common,predictor=predictor,predictor_output=prediction["predictor_output"])
    if dynamics_variant:
        if ck.get("training_contract") != "autonomous_history_only_v2":
            raise ValueError("schema-v2 autonomous checkpoint requires history-only training_contract")
        if not prov["diagnostic_only"]:
            gate = ck.get("oracle_gate") or {}
            if (ck.get("run_scope") != "oracle_gate_passed" or gate.get("passed") is not True
                    or gate.get("schema_version") != 2 or gate.get("scope") != "heldout_protein"
                    or gate.get("representation_id") != contract["representation_id"]):
                raise ValueError("headline autonomous checkpoint requires its passed held-out oracle gate; contract smoke is diagnostic only")
        cfg = StageB1bConfig(**_filter_kwargs(StageB1bConfig, ck["stage_b1b_config"]))
        if variant == "h_markov" and cfg.bath_dim != 0:
            raise ValueError("h_markov requires its independently trained bath_dim=0 checkpoint")
        if variant == "h_plus_s" and cfg.bath_dim <= 0:
            raise ValueError("h_plus_s requires bath_dim>0; use h_markov for the no-bath baseline")
        model = StageB1b(cfg).to(device)
        model.load_state_dict(ck["model"], strict=True)
        model.eval()
        model.requires_grad_(False)
        prov["b1b_step"] = ck.get("step")
        prov["dynamics_parameter_count"] = sum(p.numel() for p in model.parameters())
        prov["dynamics_config"] = ck["stage_b1b_config"]
        prov["training_variant"] = ck.get("training_variant",variant)
        return ModelBundle(**common, b1b=model, b1b_cfg=cfg)
    predictor_source = stage_a_predictor_ckpt or stage_a_ckpt
    if not predictor_source:
        raise ValueError("direct variants require an explicit predictor checkpoint")
    predictor_ck = torch.load(predictor_source, map_location="cpu", weights_only=False)
    pred_contract = predictor_ck.get("prediction_contract")
    if not pred_contract:
        raise ValueError("predictor checkpoint lacks prediction_contract; cannot use for headline evaluation")
    validate_prediction_contract(pred_contract, encoder_state=enc.state_dict(),
                                 encoder_config=ck["stage_a_v2_config"])
    # Compare actual states as well as schema: fine-tuning invalidates the old predictor.
    if state_sha256(_extract_encoder_state(predictor_ck["model"])) != state_sha256(enc.state_dict()):
        raise ValueError("predictor/encoder weights differ; adapt or retrain predictor on the selected representation")
    if pred_contract.get("predictor_output", pred_contract.get("output_contract")) != "delta_factors":
        raise ValueError("headline predictor must declare delta_factors")
    predictor, opf, p_cfg = load_stage_a_predictor(predictor_source, device, prov)
    if p_cfg.latent_dim != enc_cfg.latent_dim:
        raise ValueError("encoder/predictor dimension mismatch")
    return ModelBundle(**common, predictor=predictor, opf=opf)


def _legacy_bundle(variant, source, stage_a_ckpt, b1a_ckpt, predictor_source,
                   legacy_encoder_ckpt, legacy_predictor_output, device, prov):
    prov.update(diagnostic_only=True, legacy_reason="missing verified representation/output schema")
    if variant == "h_plus_s":
        if not legacy_encoder_ckpt:
            raise ValueError("legacy h_plus_s diagnostics require explicit legacy_encoder_ckpt matching training")
        enc, cfg = load_encoder_stage_a(legacy_encoder_ckpt, device, prov)
        model, dec, bc = load_b1b(source, device, prov)
        return ModelBundle(enc, nn.Identity(), dec, "residual", b1b=model, b1b_cfg=bc,
                           device=device, provenance=prov, latent_dim=cfg.latent_dim, diagnostic_only=True)
    if not b1a_ckpt:
        raise ValueError("legacy direct diagnostic requires b1a_ckpt")
    ck = torch.load(b1a_ckpt, map_location=device, weights_only=False)
    oracle, version = _load_b1a_oracle(ck, device)
    enc_source = stage_a_ckpt if variant == "h_only" else b1a_ckpt
    if not enc_source:
        raise ValueError("legacy direct diagnostic requires encoder checkpoint")
    enc, cfg = load_encoder_stage_a(enc_source, device, prov)
    predictor_source = predictor_source or stage_a_ckpt
    predictor, opf, pc = load_stage_a_predictor(predictor_source, device, prov)
    output = legacy_predictor_output
    if output not in {"delta_factors", "absolute_factors"}:
        raise ValueError("legacy diagnostics require explicit legacy_predictor_output; old flags do not establish semantics")
    prov.update(b1a_decoder_version=version, predictor_output=output,
                warning="legacy metadata is ambiguous; diagnostic adapter is not a scientific comparison")
    return ModelBundle(enc, oracle.h_normalizer, oracle, "oracle", predictor=predictor, opf=opf,
                       device=device, provenance=prov, latent_dim=cfg.latent_dim,
                       predictor_output=output, diagnostic_only=True)


def _mock_bundle(variant, device, prov):
    """Deterministic, explicitly diagnostic persistence fixture for harness tests."""
    dH = 128
    class MockEncoder(nn.Module):
        def forward(self, coords, residue_mask, aatype, residue_index, chain_id, token_mask=None):
            shape = (*coords.shape[:3], dH)
            return torch.zeros(shape, dtype=coords.dtype, device=coords.device)
    class MockPredictor(nn.Module):
        def forward(self, h, *args, **kwargs):
            return h.new_zeros(h.shape[0], h.shape[2], 4, 32)
    class MockOPF(nn.Module):
        def decompose(self, h): return h.reshape(*h.shape[:-1], 4, 32)
        def compose(self, h): return h.flatten(-2)
    class MockDynamics(nn.Module):
        cfg = StageB1bConfig()
        def rollout(self, h, *args, n_steps, **kwargs):
            return h[:,None].expand(-1,n_steps+1,-1,-1).clone()
    prov.update(mock_encoder=True, diagnostic_only=True)
    return ModelBundle(MockEncoder().to(device), nn.Identity(), None, "identity",
                       predictor=MockPredictor(), opf=MockOPF(),
                       b1b=MockDynamics() if variant in {"h_markov","h_plus_s"} else None,
                       b1b_cfg=StageB1bConfig() if variant in {"h_markov","h_plus_s"} else None,
                       device=device, provenance=prov, diagnostic_only=True)
