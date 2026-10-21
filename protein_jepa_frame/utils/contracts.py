"""Versioned, content-addressed representation contracts.

File names and matching tensor shapes do not establish latent compatibility.
New training and evaluation paths exchange these contracts and reject mismatches.
Legacy checkpoints require an explicit diagnostic adapter; no semantics are guessed.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from collections.abc import Mapping

import torch
from torch import nn

SCHEMA_VERSION = 2


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"contract values must be JSON serializable, got {type(value).__name__}")


def config_sha256(config) -> str:
    raw = json.dumps(_plain(config), sort_keys=True, separators=(',', ':'),
                     ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def state_sha256(state_or_module) -> str:
    state = state_or_module.state_dict() if isinstance(state_or_module, nn.Module) else state_or_module
    if not isinstance(state, Mapping):
        raise TypeError('state must be a module or state_dict')
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        digest.update(name.encode() + b'\0')
        if torch.is_tensor(value):
            value = value.detach().cpu().contiguous()
            digest.update(str(value.dtype).encode() + b'\0')
            digest.update(json.dumps(list(value.shape)).encode() + b'\0')
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(config_sha256(value).encode())
    return digest.hexdigest()


def build_prediction_contract(encoder_state, encoder_config, *,
                              predictor_output='delta_factors', window_protocol=None) -> dict:
    if predictor_output != 'delta_factors':
        raise ValueError('new Stage-A training only supports delta_factors')
    return {
        'schema_version': SCHEMA_VERSION,
        'encoder_sha256': state_sha256(encoder_state),
        'encoder_config_sha256': config_sha256(encoder_config),
        'encoder_input': 'backbone_N_CA_C_O_angstrom',
        'encoder_output': 'raw_H',
        'predictor_input': 'raw_H_history',
        'predictor_output': predictor_output,
        'factor_composition': 'anchor_factors_plus_delta',
        'window_protocol': _plain(window_protocol),
    }


def validate_prediction_contract(contract, *, encoder_state=None, encoder_config=None):
    if not contract or contract.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('missing/legacy prediction contract; retrain or explicitly use diagnostic legacy mode')
    if contract.get('predictor_output') != 'delta_factors' or contract.get('predictor_input') != 'raw_H_history':
        raise ValueError('unsupported predictor input/output contract')
    for key, actual in (
        ('encoder_sha256', state_sha256(encoder_state) if encoder_state is not None else None),
        ('encoder_config_sha256', config_sha256(encoder_config) if encoder_config is not None else None),
    ):
        if actual is not None and contract.get(key) != actual:
            raise ValueError(f'prediction contract mismatch: {key}')


_REPRESENTATION_FIELDS = (
    'schema_version', 'encoder_sha256', 'encoder_config_sha256',
    'normalizer_sha256', 'decoder_sha256', 'decoder_config_sha256',
    'decoder_input', 'dynamics_input', 'calibration_mode',
)


def build_representation_contract(encoder_state, encoder_config, normalizer_state,
                                  decoder_state, decoder_config, *,
                                  window_protocol=None, calibration_mode='oracle', source=None) -> dict:
    if calibration_mode not in ('oracle', 'predicted_adaptation'):
        raise ValueError('unknown calibration mode')
    contract = {
        'schema_version': SCHEMA_VERSION,
        'encoder_sha256': state_sha256(encoder_state),
        'encoder_config_sha256': config_sha256(encoder_config),
        'normalizer_sha256': state_sha256(normalizer_state),
        'decoder_sha256': state_sha256(decoder_state),
        'decoder_config_sha256': config_sha256(decoder_config),
        'decoder_input': 'raw_H',
        'dynamics_input': 'normalized_H',
        'calibration_mode': calibration_mode,
        'window_protocol': _plain(window_protocol),
        'source': _plain(source),
    }
    contract['representation_id'] = config_sha256({k: contract[k] for k in _REPRESENTATION_FIELDS})
    return contract


def validate_representation_contract(contract, *, encoder_state=None, encoder_config=None,
                                     normalizer_state=None, decoder_state=None, decoder_config=None):
    if not contract or contract.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('missing/legacy representation contract; new training requires a calibrated schema-v2 bundle')
    missing = [k for k in _REPRESENTATION_FIELDS if k not in contract]
    if missing:
        raise ValueError(f'incomplete representation contract: {missing}')
    expected_id = config_sha256({k: contract[k] for k in _REPRESENTATION_FIELDS})
    if contract.get('representation_id') != expected_id:
        raise ValueError('representation contract content hash is invalid')
    if contract['decoder_input'] != 'raw_H' or contract['dynamics_input'] != 'normalized_H':
        raise ValueError('unsupported representation coordinate convention')
    checks = (
        ('encoder_sha256', encoder_state, state_sha256),
        ('encoder_config_sha256', encoder_config, config_sha256),
        ('normalizer_sha256', normalizer_state, state_sha256),
        ('decoder_sha256', decoder_state, state_sha256),
        ('decoder_config_sha256', decoder_config, config_sha256),
    )
    for key, value, hash_fn in checks:
        if value is not None and hash_fn(value) != contract[key]:
            raise ValueError(f'representation contract mismatch: {key}')


def assert_same_representation(a, b):
    validate_representation_contract(a)
    validate_representation_contract(b)
    if a['representation_id'] != b['representation_id']:
        raise ValueError('representation mismatch: encoder/normalizer/decoder/calibration must be identical')
