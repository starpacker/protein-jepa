"""In-memory Stage A adaptation wrapper with an explicit relpos-mode contract.

Bare state_dict tensors contain NO feature-mode identity. Generic strict state
loading and intentional warm starts do not certify semantic compatibility. A
future checkpoint/file loader must call validate_checkpoint_metadata first.
Named buffers are explicit caller inputs for warm starts; nonpersistent buffers
may be reconstructed from pinned configuration, not claimed as checkpoint bytes.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import torch
from torch import nn

from protein_jepa_frame.models.relpos_mode_encoder_v1 import FEATURE_MODES, RelposModeProteinGeometryEncoder
from protein_jepa_frame.models.stage_a_v2 import StageAV2, StageAV2Config

SCHEMA_VERSION = 'stage_a_relpos_v1'
READOUT_ENCODERS = ('online', 'ema')


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _feature_mode(value: str) -> str:
    _require(type(value) is str and value in FEATURE_MODES, 'explicit valid feature_mode required')
    return value


def _readout(value: str) -> str:
    _require(type(value) is str and value in READOUT_ENCODERS, 'readout_encoder must explicitly be online or ema')
    return value


def _validate_tensors(expected: Mapping[str, torch.Tensor], supplied: Mapping[str, torch.Tensor], label: str) -> None:
    _require(isinstance(supplied, Mapping) and set(supplied) == set(expected), label+' keys differ')
    for key, target in expected.items():
        value = supplied[key]
        _require(isinstance(value, torch.Tensor) and value.shape == target.shape
                 and value.dtype == target.dtype and value.layout == target.layout,
                 label+' shape/dtype/layout differs: '+key)
        _require(bool(torch.isfinite(value).all()), label+' nonfinite tensor: '+key)


def _consistent_buffers(state: Mapping[str, torch.Tensor], buffers: Mapping[str, torch.Tensor], label: str) -> None:
    for key, value in buffers.items():
        if key in state:
            _require(torch.equal(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                                 state[key].detach().cpu().contiguous().reshape(-1).view(torch.uint8)),
                     label+' contradicts persistent state buffer: '+key)


@torch.no_grad()
def _copy_buffers(module: nn.Module, buffers: Mapping[str, torch.Tensor]) -> None:
    for key, target in module.named_buffers():
        target.copy_(buffers[key])


def _encoder_kwargs(config: StageAV2Config) -> dict[str, Any]:
    return dict(latent_dim=config.latent_dim, depth=config.encoder_depth,
        num_heads=config.num_heads, num_rbf=config.num_rbf, dropout=config.dropout,
        max_residues=config.max_residues, orientation_features=config.orientation_features,
        identity_embed_scale=config.identity_embed_scale,
        relative_position_features=config.relative_position_features, anchor_conditioned=False)


class StageARelposV1(StageAV2):
    """Unchanged StageAV2 computation with mode-labelled online/EMA encoders."""

    def __init__(self, config: StageAV2Config, *, feature_mode: str):
        _require(torch.get_default_device().type == 'cpu'
                 and torch.get_default_dtype() == torch.float32,
                 'construction requires default CPU device and float32 dtype; move afterward')
        selected = _feature_mode(feature_mode)
        _require(type(config) is StageAV2Config and config.relative_position_features is True,
                 'requires unchanged StageAV2Config with relpos projection enabled')
        super().__init__(config)
        self._feature_mode = selected
        replacements = []
        # The original full Stage A constructor consumes exactly its usual RNG.
        # Replacement construction consumes CPU RNG only inside this fork.
        with torch.random.fork_rng(devices=[]):
            for old in (self.context_encoder, self.teacher_encoder):
                new = RelposModeProteinGeometryEncoder(**_encoder_kwargs(config), feature_mode=selected)
                first = next(old.parameters())
                new.to(device=first.device, dtype=first.dtype)
                new.load_state_dict(old.state_dict(), strict=True)
                old_buffers = dict(old.named_buffers())
                _validate_tensors(dict(new.named_buffers()), old_buffers, 'replacement buffers')
                _copy_buffers(new, old_buffers)
                old_parameters = dict(old.named_parameters())
                for name, parameter in new.named_parameters():
                    parameter.requires_grad_(old_parameters[name].requires_grad)
                new.train(old.training)
                replacements.append(new)
        self.context_encoder, self.teacher_encoder = replacements
        self._enforce_teacher()
        self._assert_modes()

    @property
    def feature_mode(self) -> str:
        return self._feature_mode

    def _assert_modes(self) -> None:
        selected = _feature_mode(self.feature_mode)
        _require(isinstance(self.context_encoder, RelposModeProteinGeometryEncoder)
                 and isinstance(self.teacher_encoder, RelposModeProteinGeometryEncoder)
                 and self.context_encoder.feature_mode == self.teacher_encoder.feature_mode == selected,
                 'online/teacher feature modes contradict wrapper')

    def _enforce_teacher(self) -> None:
        self.teacher_encoder.eval()
        for parameter in self.teacher_encoder.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None

    def _runtime_contract(self) -> None:
        self._assert_modes()
        _require(all(not m.training for m in self.teacher_encoder.modules())
                 and all(not p.requires_grad and p.grad is None for p in self.teacher_encoder.parameters()),
                 'teacher must remain eval, frozen, and gradient-free')

    def train(self, mode: bool = True):
        self._assert_modes()
        super().train(mode)
        self._enforce_teacher()
        return self

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Strict tensor loading only; this is NOT mode-aware checkpoint loading."""
        self._assert_modes()
        _require(strict is True and assign is False, 'only strict copy-based loading is supported')
        _validate_tensors(self.state_dict(), state_dict, 'full state')
        result = super().load_state_dict(state_dict, strict=True, assign=False)
        self._enforce_teacher()
        self._assert_modes()
        return result

    @torch.no_grad()
    def sync_teacher_from_online(self) -> None:
        self._assert_modes()
        state, buffers = self.context_encoder.state_dict(), dict(self.context_encoder.named_buffers())
        _validate_tensors(self.teacher_encoder.state_dict(), state, 'teacher sync state')
        _validate_tensors(dict(self.teacher_encoder.named_buffers()), buffers, 'teacher sync buffers')
        self.teacher_encoder.load_state_dict(state, strict=True)
        _copy_buffers(self.teacher_encoder, buffers)
        self._enforce_teacher()

    @torch.no_grad()
    def warm_start_in_memory(
        self, original_full_state: Mapping[str, torch.Tensor],
        current_encoder_state: Mapping[str, torch.Tensor], *,
        original_named_buffers: Mapping[str, torch.Tensor],
        current_encoder_named_buffers: Mapping[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Fresh adaptation initialization, never optimizer/checkpoint resume.

        Validate both tensor inventories before changing any tensor. Full
        original Stage A state/buffers load first; current online encoder state
        and all its buffers load second. Teacher sync follows transplantation,
        then EMA step is reset to zero and all parameter gradients are cleared.
        """
        self._assert_modes()
        _validate_tensors(self.state_dict(), original_full_state, 'original full state')
        _validate_tensors(dict(self.named_buffers()), original_named_buffers, 'original named buffers')
        _validate_tensors(self.context_encoder.state_dict(), current_encoder_state, 'current encoder state')
        _validate_tensors(dict(self.context_encoder.named_buffers()), current_encoder_named_buffers, 'current encoder buffers')
        _consistent_buffers(original_full_state, original_named_buffers, 'original buffers')
        _consistent_buffers(current_encoder_state, current_encoder_named_buffers, 'current buffers')
        # Caller mappings can alias this model (state_dict values are views).
        # Snapshot every validated input before the first target mutation.
        original_full_state = {k: v.detach().clone() for k, v in original_full_state.items()}
        original_named_buffers = {k: v.detach().clone() for k, v in original_named_buffers.items()}
        current_encoder_state = {k: v.detach().clone() for k, v in current_encoder_state.items()}
        current_encoder_named_buffers = {k: v.detach().clone() for k, v in current_encoder_named_buffers.items()}
        original_step = int(original_full_state['_ema_step'])
        self.load_state_dict(original_full_state, strict=True)
        _copy_buffers(self, original_named_buffers)
        self.context_encoder.load_state_dict(current_encoder_state, strict=True)
        _copy_buffers(self.context_encoder, current_encoder_named_buffers)
        self.sync_teacher_from_online()
        self._ema_step.zero_()
        for parameter in self.parameters():
            parameter.grad = None
        self._runtime_contract()
        return dict(initialization='original_full_then_current_online_then_teacher_sync',
            feature_mode=self.feature_mode, original_ema_step=original_step, ema_step=0,
            teacher_synced_after_transplant=True, named_buffers_supplied_explicitly=True,
            optimizer_loaded=False)

    def select_readout_encoder(self, kind: str) -> RelposModeProteinGeometryEncoder:
        self._runtime_contract()
        return self.context_encoder if _readout(kind) == 'online' else self.teacher_encoder

    def encode_temporal(self, **kwargs):
        self._runtime_contract()
        return super().encode_temporal(**kwargs)

    def forward_spatial(self, *args, **kwargs):
        self._runtime_contract()
        return super().forward_spatial(*args, **kwargs)

    def forward_temporal_multilag(self, *args, **kwargs):
        self._runtime_contract()
        return super().forward_temporal_multilag(*args, **kwargs)

    @torch.no_grad()
    def update_ema(self, momentum: float | None = None) -> float:
        self._runtime_contract()
        return super().update_ema(momentum)


def build_checkpoint_metadata(model: StageARelposV1, *, readout_encoder: str) -> dict[str, Any]:
    """Build explicit semantic metadata; this does not save or authenticate weights."""
    _require(isinstance(model, StageARelposV1), 'versioned model required')
    model._runtime_contract()
    return dict(schema_version=SCHEMA_VERSION, feature_mode=model.feature_mode,
        online_feature_mode=model.context_encoder.feature_mode,
        teacher_feature_mode=model.teacher_encoder.feature_mode,
        stage_a_v2_config=asdict(model.config), readout_encoder=_readout(readout_encoder))


def validate_checkpoint_metadata(
    metadata: Mapping[str, Any], *, expected_feature_mode: str,
    expected_config: StageAV2Config, expected_readout_encoder: str,
) -> dict[str, Any]:
    """Mandatory future-loader gate; reject absent or contradictory mode labels."""
    selected, readout = _feature_mode(expected_feature_mode), _readout(expected_readout_encoder)
    _require(type(expected_config) is StageAV2Config and expected_config.relative_position_features is True,
             'expected versioned Stage A configuration required')
    expected = dict(schema_version=SCHEMA_VERSION, feature_mode=selected,
        online_feature_mode=selected, teacher_feature_mode=selected,
        stage_a_v2_config=asdict(expected_config), readout_encoder=readout)
    _require(isinstance(metadata, Mapping) and set(metadata) == set(expected), 'checkpoint metadata fields differ')
    _require(dict(metadata) == expected, 'checkpoint schema/mode/config/readout identity differs')
    return expected
