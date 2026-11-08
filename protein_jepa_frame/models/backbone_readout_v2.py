"""Independent, fixed-CA N/CA/C/O readout initialized from anchored B1a.

Inputs are raw encoder H (never HNormalizer Z), historical dataset embedding
IDs, and separately supplied OpenFold chemical IDs. IPA uses the supplied CA
cloud; orientation updates are relative to the observed decoder-native frames.
This is a new readout contract, not a reinterpretation of legacy checkpoints.
Only psi affects O in the canonical OpenFold backbone. The other six legacy
torsion outputs are therefore omitted, rather than registered as dead trainable
parameters. No position head, probe, encoder, or H normalizer is retained.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import torch
from torch import nn

from protein_jepa_frame.models.stage_c.geometry import rotation_matrix_from_quaternion
from protein_jepa_frame.models.stage_c.structure_decoder import (
    DecoderConfig,
    StructureDecoder,
    _torsion_logits_to_angles,
    reconstruct_atom37_from_frames,
)


HEAD_VERSION = "fixed_ca_backbone_readout_v2"
BACKBONE_ATOM_ORDER = ("N", "CA", "C", "O")


@dataclass(frozen=True)
class BackboneReadoutConfig:
    raw_h_dim: int = 128
    c_hidden: int = 128
    n_heads: int = 4
    n_ipa_blocks: int = 4
    torsion_bins: int = 36
    dropout: float = 0.1
    global_h_context: bool = True

    def __post_init__(self):
        if min(self.raw_h_dim, self.c_hidden, self.n_heads, self.n_ipa_blocks) < 1:
            raise ValueError("readout dimensions and block count must be positive")
        if self.c_hidden % self.n_heads or self.torsion_bins < 2:
            raise ValueError("hidden width must divide into heads; at least two torsion bins")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BackboneReadoutV2(nn.Module):
    """Predict actual backbone atoms while preserving every valid input CA.

    ``config`` + strict ``state_dict`` suffice to restore this independent head.
    ``initialization`` is provenance metadata, to be copied into its checkpoint
    by the caller. The frozen key/bias constants preserve legacy initialization:
    adding a constant to every attention logit is an unidentifiable softmax
    direction, so ``linear_k.bias`` and ``linear_b.bias`` are not trainable.
    """

    def __init__(self, config: BackboneReadoutConfig | None = None):
        super().__init__()
        self.config = config or BackboneReadoutConfig()
        c = self.config
        # Reuse the legacy module definitions and initialization. Only the named
        # orientation/state modules are registered on this head; the temporary
        # decoder's positional modules are discarded and never used in forward.
        source = StructureDecoder(DecoderConfig(
            q_dim=c.raw_h_dim, c_hidden=c.c_hidden, n_heads=c.n_heads,
            n_ipa_blocks=c.n_ipa_blocks, torsion_bins=c.torsion_bins,
            dropout=c.dropout, global_h_context=c.global_h_context,
            anchor_free=False, bath_extra_dim=0,
        ))
        for name in ("aatype_embed", "state_input_norm", "state_proj",
                     "ipa_blocks", "transitions", "layer_norms", "dropout",
                     "frame_update"):
            setattr(self, name, getattr(source, name))
        if c.global_h_context:
            self.global_h_proj = source.global_h_proj
        self.psi_head = nn.Sequential(
            source.torsion_head.net[0], nn.SiLU(), nn.Linear(c.c_hidden, c.torsion_bins)
        )
        with torch.no_grad():
            psi_slice = slice(c.torsion_bins, 2 * c.torsion_bins)
            self.psi_head[2].weight.copy_(source.torsion_head.net[2].weight[psi_slice])
            self.psi_head[2].bias.copy_(source.torsion_head.net[2].bias[psi_slice])
        for block in self.ipa_blocks:
            block.linear_k.bias.requires_grad_(False)
            block.linear_b.bias.requires_grad_(False)
        self.initialization: dict = {"kind": "random", "head_version": HEAD_VERSION}

    @classmethod
    def from_legacy_decoder(cls, decoder: StructureDecoder) -> "BackboneReadoutV2":
        """Copy an anchored raw-H decoder; never share tensors with the source.

        The legacy positional output is intentionally discarded. This control
        uses corrected chemical IDs and current CA at IPA and reconstruction;
        it is not equal to the legacy translated backbone route.
        """
        c = decoder.config
        if decoder.anchor_free or c.bath_extra_dim:
            raise ValueError("initialization requires anchored B1a raw-H decoder without bath input")
        config = BackboneReadoutConfig(
            raw_h_dim=c.q_dim, c_hidden=c.c_hidden, n_heads=c.n_heads,
            n_ipa_blocks=c.n_ipa_blocks, torsion_bins=c.torsion_bins,
            dropout=c.dropout, global_h_context=decoder.global_h_context,
        )
        # Loading an initialization must not perturb a caller's CPU RNG stream.
        with torch.random.fork_rng(devices=[]):
            result = cls(config)
        source_state = decoder.state_dict()
        target_state = {}
        for name in result.state_dict():
            source_name = name.replace("psi_head.", "torsion_head.net.")
            value = source_state[source_name].detach().clone()
            if name.startswith("psi_head.2."):
                value = value[c.torsion_bins:2 * c.torsion_bins].clone()
            if not torch.isfinite(value).all():
                raise ValueError(f"nonfinite legacy initialization tensor: {source_name}")
            target_state[name] = value
        # Preserve the source floating-point dtype/device for direct callers.
        reference = decoder.aatype_embed.weight
        result.to(device=reference.device, dtype=reference.dtype)
        result.load_state_dict(target_state, strict=True)
        result.train(decoder.training)
        result.initialization = {
            "kind": "legacy_anchored_structure_decoder", "head_version": HEAD_VERSION,
            "input_state": "raw_H_without_HNormalizer",
            "embedding_order": "ACDEFGHIKLMNPQRSTVWYX",
            "chemical_order": "ARNDCQEGHILKMFPSTWYVX",
            "unknown_chemical_template": "OpenFold adapter GLY backbone; identity remains UNK",
            "ipa_positions": "fixed_current_CA_Angstrom",
            "orientation_basis": "observed decoder-native R",
            "translation": "exact supplied CA on valid residues",
            "trainable_torsion_slots": [1],
            "unused_torsion_slots": "phi/omega/chi1-4 fixed to zero; not backbone dihedrals",
            "frozen_softmax_gauges": ["ipa_blocks.*.linear_k.bias", "ipa_blocks.*.linear_b.bias"],
            "atom_order": list(BACKBONE_ATOM_ORDER),
            "atom_producer": "official OpenFold adapter (float32 internal geometry)",
            "legacy_translation_head_retained": False,
        }
        return result

    @classmethod
    def from_legacy_checkpoint(
        cls, path: str | Path, *, expected_sha256: str | None = None,
    ) -> "BackboneReadoutV2":
        """Strictly load the deployment ``oracle`` from a trusted B1a file on CPU.

        The full oracle state is checked before selecting head tensors, including
        its probe and normalizer. Those modules are not retained by this head.
        ``training_oracle`` and encoder weights are never substituted for it.
        """
        from protein_jepa_frame.models.stage_b1a.model import StageBOracleConfig, StageBOracleDecoder

        path = Path(path).resolve()
        digest = _sha256(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("legacy checkpoint SHA256 mismatch")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if "stage_b1a_config" not in checkpoint or "oracle" not in checkpoint:
            raise ValueError("expected B1a stage_b1a_config and deployment oracle")
        with torch.random.fork_rng(devices=[]):
            source = StageBOracleDecoder(StageBOracleConfig.from_saved(checkpoint["stage_b1a_config"]))
            source.load_state_dict(checkpoint["oracle"], strict=True)
            result = cls.from_legacy_decoder(source.decoder.eval())
        if _sha256(path) != digest:
            raise RuntimeError("legacy checkpoint changed while loading")
        result.initialization.update({
            "legacy_checkpoint": str(path), "legacy_checkpoint_sha256": digest,
            "legacy_step": checkpoint.get("step"), "legacy_state_key": "oracle",
            "legacy_config": checkpoint["stage_b1a_config"],
        })
        return result

    def checkpoint_dict(self) -> dict:
        """Head-only payload; training state/step/protocol remain caller-owned.

        Save the initial payload before updates as the corrected-chemical,
        fixed-CA zero-update control. ``initialization`` records the origin,
        not a claim that a later state is still untrained.
        """
        return {"head_version": HEAD_VERSION, "head_config": asdict(self.config),
                "head": self.state_dict(), "initialization": dict(self.initialization)}

    def forward(
        self,
        raw_H: torch.Tensor,
        observed_R: torch.Tensor,
        fixed_current_CA: torch.Tensor,
        embedding_aatype: torch.Tensor,
        chemical_aatype: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return N/CA/C/O [B,L,4,3] in Angstrom and matching boolean atom mask.

        R columns use ``RigidBodyFrame.from_backbone``'s decoder convention.
        Supplied frames/CA must use the same observed-anchor gauge. There are
        no future-coordinate/mask arguments. Missing/padded residues return
        zero atoms, translation, logits and torsions, false atom masks, and
        identity R; their input NaNs/IDs are ignored. Valid CA is copied exactly
        even for float64 callers (the existing atom producer runs in float32).
        Topology is a separate loss contract, not inferred from array adjacency.
        Caller controls whether upstream input tensors are frozen/detached.
        """
        if raw_H.ndim != 3 or raw_H.shape[-1] != self.config.raw_h_dim:
            raise ValueError("raw_H must be [B,L,raw_h_dim]")
        shape = raw_H.shape[:2]
        if min(shape) < 1 or observed_R.shape != (*shape, 3, 3) or fixed_current_CA.shape != (*shape, 3):
            raise ValueError("nonempty observed_R [B,L,3,3] and fixed_current_CA [B,L,3] required")
        if residue_mask.shape != shape or residue_mask.dtype != torch.bool:
            raise ValueError("residue_mask must be bool [B,L]")
        for name, value in (("embedding_aatype", embedding_aatype), ("chemical_aatype", chemical_aatype)):
            if value.shape != shape or value.dtype != torch.long:
                raise ValueError(f"{name} must be int64 [B,L]")
        tensors = (observed_R, fixed_current_CA, embedding_aatype, chemical_aatype, residue_mask)
        if any(value.device != raw_H.device for value in tensors):
            raise ValueError("all head inputs must be on the same device")
        if not raw_H.is_floating_point() or observed_R.dtype != raw_H.dtype or fixed_current_CA.dtype != raw_H.dtype:
            raise ValueError("H, R and CA must share a floating dtype")
        for name, value in (("raw_H", raw_H), ("observed_R", observed_R), ("fixed_current_CA", fixed_current_CA)):
            if not torch.isfinite(value[residue_mask]).all():
                raise ValueError(f"nonfinite valid {name}")
        for name, value in (("embedding_aatype", embedding_aatype), ("chemical_aatype", chemical_aatype)):
            if ((value[residue_mask] < 0) | (value[residue_mask] > 20)).any():
                raise ValueError(f"valid {name} outside 0..20")

        mask = residue_mask.unsqueeze(-1)
        H = torch.where(mask, raw_H, 0.0)
        t = torch.where(mask, fixed_current_CA, 0.0)
        identity = torch.eye(3, dtype=raw_H.dtype, device=raw_H.device)
        observed_R = torch.where(mask.unsqueeze(-1), observed_R, identity)
        embedding = torch.where(residue_mask, embedding_aatype, 20)
        chemical = torch.where(residue_mask, chemical_aatype, 20)
        state = self.state_input_norm(H)
        s = self.aatype_embed(embedding) + self.state_proj(state)
        if self.config.global_h_context:
            pooled = (state * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp(min=1)
            s = s + self.global_h_proj(pooled)
        s = torch.where(mask, s, 0.0)
        for block, norm, transition in zip(self.ipa_blocks, self.layer_norms, self.transitions):
            residual, _ = block(s, observed_R, t, residue_mask)
            s = norm(s + self.dropout(residual))
            s = torch.where(mask, s + transition(s), 0.0)

        R = observed_R @ rotation_matrix_from_quaternion(self.frame_update(s))
        R = torch.where(mask.unsqueeze(-1), R, identity)
        psi_logits = torch.where(mask, self.psi_head(s), 0.0)
        psi = _torsion_logits_to_angles(psi_logits.unsqueeze(-2)).squeeze(-1)
        psi = torch.where(residue_mask, psi, 0.0)
        zero = torch.zeros_like(psi)
        torsions = torch.stack([zero, psi, zero, zero, zero, zero, zero], dim=-1)
        atom37, atom37_mask = reconstruct_atom37_from_frames(R, t, chemical, torsions)
        atom_mask = atom37_mask[..., [0, 1, 2, 4]].bool() & mask
        # Restore exact CA after the official adapter's internal float32 cast.
        backbone = torch.stack((atom37[..., 0, :], t, atom37[..., 2, :], atom37[..., 4, :]), dim=-2)
        backbone = torch.where(atom_mask.unsqueeze(-1), backbone, 0.0)
        return {"backbone": backbone, "atom_mask": atom_mask, "R": R, "t": t,
                "torsions": torsions, "psi_logits": psi_logits}
