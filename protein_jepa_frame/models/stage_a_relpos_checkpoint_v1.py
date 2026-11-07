"""Explicit checkpoint boundary for the prospective Stage A relpos candidate.

Warm starts read exactly two hash-pinned whole checkpoint dictionaries and
construct only StageARelposV1. The old encoder loader is never used for a new
mode-labelled payload. Neither API restores an optimizer or resumes training.
Nonpersistent warm-start buffers are reconstructed from the explicit original
configuration; they are not represented as historical checkpoint bytes.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import math
from pathlib import Path
import re

import torch

from protein_jepa_frame.models.stage_a_v2 import StageAV2Config
from protein_jepa_frame.models.stage_a_relpos_v1 import (
    StageARelposV1, _consistent_buffers, _copy_buffers, _encoder_kwargs,
    _feature_mode, _readout, _require, _validate_tensors,
    build_checkpoint_metadata, validate_checkpoint_metadata,
)
from protein_jepa_frame.models.stage_c.frozen_h_distance_probe_v2 import (
    encoder_payload, sha256_file,
)
from protein_jepa_frame.utils.contracts import state_sha256

PAYLOAD_SCHEMA = 'stage_a_relpos_checkpoint_v1'
PAYLOAD_KEYS = {'schema_version', 'metadata', 'encoder_policy', 'model_state', 'named_buffers'}


def _cpu_defaults():
    _require(torch.get_default_device().type == 'cpu' and torch.get_default_dtype() == torch.float32,
             'checkpoint construction requires default CPU device and float32 dtype')


def _config(raw):
    defaults = asdict(StageAV2Config())
    _require(isinstance(raw, Mapping) and set(raw) == set(defaults),
             'complete explicit StageAV2Config fields required; no inferred defaults')
    for key, reference in defaults.items():
        value = raw[key]
        if type(reference) is float:
            _require(type(value) in (float, int) and math.isfinite(value), 'invalid numeric config: '+key)
        else:
            _require(type(value) is type(reference), 'invalid config type: '+key)
    config = StageAV2Config(**dict(raw))
    _require(config.relative_position_features is True, 'relpos projection must remain present')
    return config


def _pin(path, expected_sha256):
    _require(isinstance(expected_sha256, str) and re.fullmatch('[0-9a-f]{64}', expected_sha256) is not None,
             'explicit lowercase SHA256 required')
    path = Path(path).resolve()
    _require(path.is_file() and sha256_file(path) == expected_sha256,
             'checkpoint hash differs: '+str(path))
    return path


def _snapshot(mapping):
    return {key: value.detach().cpu().clone() for key, value in mapping.items()}


def _cpu_tensors(mapping, label):
    _require(isinstance(mapping, Mapping) and all(isinstance(k, str) for k in mapping), label+' must be a named mapping')
    _require(all(isinstance(v, torch.Tensor) and v.device.type == 'cpu' for v in mapping.values()),
             label+' must contain only CPU tensors')


def _reconstructed_buffers(module, state):
    # Persistent buffer values are checkpoint tensors. Other buffers arise from
    # the explicit configuration in this new CPU construction, before loading.
    return {name: (state[name] if name in state else value).detach().clone()
            for name, value in module.named_buffers()}


def _policy(readout):
    return dict(online='context_encoder', teacher='teacher_encoder',
                readout=_readout(readout), optimizer_resume=False)


def load_warm_start(original_path, *, original_sha256, current_path, current_sha256, feature_mode):
    """Return a fresh model and audit after original-full/current-online loading.

    Both files pass their SHA checks before either deserialization. Exactly two
    torch.load calls read CPU dictionaries. Hashes are rechecked before model
    construction and after loading; the caller CPU RNG state is preserved.
    """
    _cpu_defaults(); _feature_mode(feature_mode)
    original_path = _pin(original_path, original_sha256)
    current_path = _pin(current_path, current_sha256)
    _require(original_path != current_path, 'original and current checkpoint paths must differ')
    original = torch.load(original_path, map_location='cpu', weights_only=False)
    current = torch.load(current_path, map_location='cpu', weights_only=False)
    _pin(original_path, original_sha256); _pin(current_path, current_sha256)
    _require(isinstance(original, dict) and isinstance(current, dict), 'whole checkpoint dictionaries required')
    config = _config(original.get('stage_a_v2_config'))
    original_state = original.get('model')
    _cpu_tensors(original_state, 'original full state')
    component = current.get('component_checkpoint')
    _require(isinstance(component, dict) and component.get('checkpoint_kind') == 'ca_component_probe',
             'current component kind differs')
    calibration = component.get('calibration_args')
    _require(isinstance(calibration, Mapping), 'explicit calibration lineage required')
    declared = calibration.get('stage_a_ckpt')
    _require(isinstance(declared, str) and Path(declared).resolve() == original_path,
             'calibration configuration path differs from pinned original')
    lineage = {'calibration_args.stage_a_ckpt': declared}
    if 'stage_a_ckpt' in component:
        top = component['stage_a_ckpt']
        _require(isinstance(top, str) and top == declared,
                 'component stage_a_ckpt disagrees with calibration lineage')
        lineage['component_checkpoint.stage_a_ckpt'] = top
    current_state, kwargs, extraction = encoder_payload(current, 'current', configuration_checkpoint=original)
    _require(kwargs == _encoder_kwargs(config), 'current state-derived encoder flags/config disagree with original config')
    with torch.random.fork_rng(devices=[]):
        model = StageARelposV1(config, feature_mode=feature_mode)
        original_buffers = _reconstructed_buffers(model, original_state)
        current_buffers = _reconstructed_buffers(model.context_encoder, current_state)
        warm_audit = model.warm_start_in_memory(original_state, current_state,
            original_named_buffers=original_buffers, current_encoder_named_buffers=current_buffers)
    _pin(original_path, original_sha256); _pin(current_path, current_sha256)
    persistent = set(model.state_dict())
    nonpersistent = [name for name, _ in model.named_buffers() if name not in persistent]
    audit = dict(schema_version=PAYLOAD_SCHEMA, feature_mode=feature_mode,
        original_checkpoint_path=str(original_path), original_checkpoint_sha256=original_sha256,
        current_checkpoint_path=str(current_path), current_checkpoint_sha256=current_sha256,
        configuration_source='pinned_original.stage_a_v2_config', stage_a_v2_config=asdict(config),
        recorded_config_lineage=lineage, current_encoder_extraction=extraction,
        warm_start=warm_audit, checkpoint_load_count=2, cpu_rng_preserved=True,
        nonpersistent_buffers_source='reconstructed_from_explicit_pinned_original_config',
        reconstructed_nonpersistent_buffer_names=nonpersistent,
        persistent_buffers_source='checkpoint_model_state_then_current_encoder_state_and_teacher_sync',
        model_state_sha256=state_sha256(model), named_buffers_sha256=state_sha256(dict(model.named_buffers())),
        optimizer_loaded=False, model_constructed='StageARelposV1_only',
        online_training=model.context_encoder.training, teacher_training=model.teacher_encoder.training)
    return model, audit


def build_versioned_payload(model, *, readout_encoder):
    """Snapshot full tensors/buffers and explicit mode/readout metadata.

    All tensors are independent CPU clones, including nonpersistent buffers.
    This is a weights/readout artifact, not an optimizer or exact-resume file.
    Building it neither freezes the online encoder nor changes train/eval mode.
    """
    metadata = build_checkpoint_metadata(model, readout_encoder=readout_encoder)
    _config(metadata['stage_a_v2_config'])
    state, buffers = _snapshot(model.state_dict()), _snapshot(dict(model.named_buffers()))
    _validate_tensors(model.state_dict(), state, 'payload model state')
    _validate_tensors(dict(model.named_buffers()), buffers, 'payload named buffers')
    _consistent_buffers(state, buffers, 'payload named buffers')
    return dict(schema_version=PAYLOAD_SCHEMA, metadata=metadata,
        encoder_policy=_policy(readout_encoder), model_state=state, named_buffers=buffers)


def load_versioned_payload(path, *, expected_sha256, expected_feature_mode,
                           expected_config, expected_readout_encoder):
    """Strict new-schema file loader; never use an old loader for local weights.

    Expected mode/config/readout identity is mandatory, before constructing the
    model. Full tensor and buffer inventories are validated before load. Teacher
    state and EMA step are preserved, without a teacher resync or optimizer.
    Online trainability is unchanged; choosing a readout does not freeze it.
    """
    _cpu_defaults(); _feature_mode(expected_feature_mode); _readout(expected_readout_encoder)
    _require(type(expected_config) is StageAV2Config, 'explicit expected StageAV2Config required')
    _config(asdict(expected_config))
    path = _pin(path, expected_sha256)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    _pin(path, expected_sha256)
    _require(isinstance(payload, dict) and set(payload) == PAYLOAD_KEYS
             and payload.get('schema_version') == PAYLOAD_SCHEMA, 'unlabelled or invalid versioned payload schema')
    _require(isinstance(payload['metadata'], Mapping), 'explicit versioned metadata required')
    _config(payload['metadata'].get('stage_a_v2_config'))
    validate_checkpoint_metadata(payload['metadata'], expected_feature_mode=expected_feature_mode,
        expected_config=expected_config, expected_readout_encoder=expected_readout_encoder)
    _require(payload['encoder_policy'] == _policy(expected_readout_encoder), 'contradictory encoder/readout policy')
    state, buffers = payload['model_state'], payload['named_buffers']
    _cpu_tensors(state, 'versioned model state'); _cpu_tensors(buffers, 'versioned buffers')
    with torch.random.fork_rng(devices=[]):
        model = StageARelposV1(expected_config, feature_mode=expected_feature_mode)
        _validate_tensors(model.state_dict(), state, 'versioned model state')
        _validate_tensors(dict(model.named_buffers()), buffers, 'versioned named buffers')
        _consistent_buffers(state, buffers, 'versioned named buffers')
        model.load_state_dict(state, strict=True)
        _copy_buffers(model, buffers)
        model._runtime_contract()
    _pin(path, expected_sha256)
    audit = dict(schema_version=PAYLOAD_SCHEMA, checkpoint_path=str(path), checkpoint_sha256=expected_sha256,
        metadata=payload['metadata'], encoder_policy=payload['encoder_policy'],
        model_state_sha256=state_sha256(model), named_buffers_sha256=state_sha256(dict(model.named_buffers())),
        named_buffers_source='explicit_versioned_payload_including_nonpersistent',
        checkpoint_load_count=1, cpu_rng_preserved=True, optimizer_loaded=False,
        teacher_resynced=False, ema_step=int(model._ema_step),
        online_training=model.context_encoder.training, teacher_training=model.teacher_encoder.training)
    return model, audit
