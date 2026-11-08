"""Synthetic temporary checkpoint files only; no real model/data artifacts."""
import copy
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

import protein_jepa_frame.models.stage_a_relpos_checkpoint_v1 as bridge
from protein_jepa_frame.models.stage_a_v2 import StageAV2, StageAV2Config
from protein_jepa_frame.models.stage_a_relpos_v1 import StageARelposV1


def config(latent_dim=128):
    return StageAV2Config(latent_dim=latent_dim, encoder_depth=1, num_heads=4,
        num_rbf=3, max_residues=32, predictor_depth=1, predictor_spatial_depth=1,
        predictor_heads=4, max_frames=8, struct_recon_dim=24, struct_recon_pairs=6)


def snapshots(mapping):
    return {k:v.detach().clone() for k,v in mapping.items()}


def same(a, b):
    assert list(a) == list(b)
    for k in a:
        assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape
        assert torch.equal(a[k].contiguous().reshape(-1).view(torch.uint8),
                           b[k].contiguous().reshape(-1).view(torch.uint8)), k


@pytest.fixture(scope='module')
def warm_payloads():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91); old = StageAV2(config())
        old._ema_step.fill_(13)
        torch.manual_seed(92); current = StageAV2(config()).context_encoder
    return dict(stage_a_v2_config=asdict(config()), model=snapshots(old.state_dict())), snapshots(current.state_dict())


def write_warm(tmp_path, warm_payloads, mutate=None):
    original = copy.deepcopy(warm_payloads[0])
    op, cp = tmp_path/'original.pt', tmp_path/'current.pt'
    component = dict(checkpoint_kind='ca_component_probe', calibration_args={'stage_a_ckpt':str(op)},
                     stage_a_ckpt=str(op), encoder=copy.deepcopy(warm_payloads[1]))
    current = {'component_checkpoint':component}
    if mutate: mutate(original, current)
    torch.save(original, op); torch.save(current, cp)
    args = dict(original_path=op, original_sha256=bridge.sha256_file(op),
                current_path=cp, current_sha256=bridge.sha256_file(cp))
    return args, original, current


@pytest.mark.parametrize('mode', ['pca', 'zero', 'local'])
def test_warm_start_two_hash_preflights_two_loads_rng_and_strict_transplant(tmp_path, warm_payloads, monkeypatch, mode):
    args, original, current = write_warm(tmp_path, warm_payloads)
    events = []
    sha, load = bridge.sha256_file, torch.load
    def hashed(path):
        events.append(('sha', Path(path).resolve())); return sha(path)
    def loaded(path, **kwargs):
        events.append(('load', Path(path).resolve()))
        assert kwargs == {'map_location':'cpu', 'weights_only':False}
        return load(path, **kwargs)
    monkeypatch.setattr(bridge, 'sha256_file', hashed)
    monkeypatch.setattr(torch, 'load', loaded)
    rng = torch.get_rng_state().clone()
    model, audit = bridge.load_warm_start(**args, feature_mode=mode)
    assert torch.equal(rng, torch.get_rng_state())
    assert events[:4] == [('sha', args['original_path']), ('sha', args['current_path']),
                         ('load', args['original_path']), ('load', args['current_path'])]
    assert sum(kind == 'load' for kind, _ in events) == 2
    for encoder in (model.context_encoder, model.teacher_encoder):
        same(encoder.state_dict(), current['component_checkpoint']['encoder'])
        assert encoder.feature_mode == mode
    for name, tensor in model.state_dict().items():
        if not name.startswith(('context_encoder.', 'teacher_encoder.')) and name != '_ema_step':
            same({name:tensor}, {name:original['model'][name]})
    assert int(model._ema_step) == 0 and audit['warm_start']['original_ema_step'] == 13
    assert audit['configuration_source'] == 'pinned_original.stage_a_v2_config'
    assert audit['checkpoint_load_count'] == 2 and not audit['optimizer_loaded']
    assert audit['reconstructed_nonpersistent_buffer_names'] == ['context_encoder.rbf_centers', 'teacher_encoder.rbf_centers']
    assert audit['nonpersistent_buffers_source'] == 'reconstructed_from_explicit_pinned_original_config'
    # Compare the constructor's actual buffer definition, not a checkpoint claim.
    with torch.random.fork_rng(devices=[]): reference = StageAV2(config())
    same(dict(model.context_encoder.named_buffers()), dict(reference.context_encoder.named_buffers()))
    assert all(p.requires_grad for p in model.context_encoder.parameters())
    assert not model.teacher_encoder.training and all(not p.requires_grad for p in model.teacher_encoder.parameters())


@pytest.mark.parametrize('which', ['original_sha256', 'current_sha256'])
def test_both_hashes_preflight_before_any_deserialization(tmp_path, warm_payloads, monkeypatch, which):
    args, _, _ = write_warm(tmp_path, warm_payloads)
    args[which] = '0'*64
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('must not load any checkpoint'))
    with pytest.raises(ValueError, match='hash differs'):
        bridge.load_warm_start(**args, feature_mode='local')


@pytest.mark.parametrize('drift', ['config_missing', 'config_extra', 'config_bad_type', 'config_flags',
    'component_kind', 'lineage_missing', 'lineage_wrong', 'duplicate_lineage', 'alternate_config',
    'current_missing', 'current_nan', 'original_missing', 'original_nan'])
def test_warm_start_strict_schema_lineage_and_inventory_rejection(tmp_path, warm_payloads, drift):
    def mutate(original, current):
        c = current['component_checkpoint']
        if drift == 'config_missing': original['stage_a_v2_config'].pop('struct_recon_dim')
        elif drift == 'config_extra': original['stage_a_v2_config']['anchor_conditioned'] = False
        elif drift == 'config_bad_type': original['stage_a_v2_config']['encoder_depth'] = True
        elif drift == 'config_flags': original['stage_a_v2_config']['orientation_features'] = False
        elif drift == 'component_kind': c['checkpoint_kind'] = 'legacy'
        elif drift == 'lineage_missing': c['calibration_args'].clear()
        elif drift == 'lineage_wrong': c['calibration_args']['stage_a_ckpt'] = str(tmp_path/'different.pt')
        elif drift == 'duplicate_lineage': c['stage_a_ckpt'] = str(tmp_path/'different.pt')
        elif drift == 'alternate_config': c['stage_a_v2_config'] = original['stage_a_v2_config']
        elif drift == 'current_missing': c['encoder'].pop('relpos_proj.weight')
        elif drift == 'current_nan': c['encoder']['relpos_proj.weight'].fill_(float('nan'))
        elif drift == 'original_missing': original['model'].pop('context_encoder.relpos_proj.weight')
        else: original['model']['context_encoder.relpos_proj.weight'].fill_(float('nan'))
    args, _, _ = write_warm(tmp_path, warm_payloads, mutate)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError): bridge.load_warm_start(**args, feature_mode='local')
    assert torch.equal(rng, torch.get_rng_state())


def test_warm_start_detects_hash_change_after_deserialization_before_construction(tmp_path, warm_payloads, monkeypatch):
    args, _, _ = write_warm(tmp_path, warm_payloads)
    load = torch.load
    def changed(path, **kwargs):
        payload = load(path, **kwargs)
        if Path(path) == args['current_path']:
            with args['original_path'].open('ab') as f: f.write(b'changed')
        return payload
    monkeypatch.setattr(torch, 'load', changed)
    monkeypatch.setattr(bridge, 'StageARelposV1', lambda *a, **kw: pytest.fail('hash change must precede construction'))
    with pytest.raises(ValueError, match='hash differs'): bridge.load_warm_start(**args, feature_mode='pca')


def versioned_model(mode='local'):
    model = StageARelposV1(config(16), feature_mode=mode)
    with torch.no_grad():
        model._ema_step.fill_(7)
        for p in model.context_encoder.parameters(): p.add_(.25)
        model.context_encoder.rbf_centers.add_(.125)
        model.teacher_encoder.rbf_centers.sub_(.25)
    return model


def save_versioned(tmp_path, model, readout='online', mutate=None):
    payload = bridge.build_versioned_payload(model, readout_encoder=readout)
    if mutate: mutate(payload)
    path = tmp_path/'versioned.pt'; torch.save(payload, path)
    args = dict(path=path, expected_sha256=bridge.sha256_file(path),
        expected_feature_mode=model.feature_mode, expected_config=model.config,
        expected_readout_encoder=readout)
    return payload, args


@pytest.mark.parametrize('mode', ['pca', 'zero', 'local'])
@pytest.mark.parametrize('readout', ['online', 'ema'])
def test_versioned_roundtrip_preserves_teacher_step_all_buffers_mode_rng_without_freezing(tmp_path, mode, readout):
    model = versioned_model(mode)
    payload, args = save_versioned(tmp_path, model, readout)
    before = torch.get_rng_state().clone()
    restored, audit = bridge.load_versioned_payload(**args)
    assert torch.equal(before, torch.get_rng_state())
    same(model.state_dict(), restored.state_dict())
    same(dict(model.named_buffers()), dict(restored.named_buffers()))
    assert restored.feature_mode == restored.context_encoder.feature_mode == restored.teacher_encoder.feature_mode == mode
    assert int(restored._ema_step) == 7 and audit['ema_step'] == 7
    assert not audit['teacher_resynced'] and not audit['optimizer_loaded']
    assert audit['encoder_policy']['readout'] == readout
    assert audit['named_buffers_source'] == 'explicit_versioned_payload_including_nonpersistent'
    assert restored.select_readout_encoder(readout) is (restored.context_encoder if readout == 'online' else restored.teacher_encoder)
    assert all(p.requires_grad for p in restored.context_encoder.parameters())
    assert all(not p.requires_grad for p in restored.teacher_encoder.parameters())
    assert restored.context_encoder.training and not restored.teacher_encoder.training
    assert any(not torch.equal(p, restored.teacher_encoder.state_dict()[name]) for name,p in restored.context_encoder.state_dict().items())


def test_payload_all_tensors_cloned_and_builder_preserves_modes_rng():
    model = versioned_model(); model.eval()
    before = torch.get_rng_state().clone()
    payload = bridge.build_versioned_payload(model, readout_encoder='ema')
    expected = copy.deepcopy(payload)
    with torch.no_grad():
        for p in model.parameters(): p.add_(1)
        for b in model.buffers(): b.add_(1)
    same(payload['model_state'], expected['model_state'])
    same(payload['named_buffers'], expected['named_buffers'])
    assert torch.equal(before, torch.get_rng_state())
    assert not model.training and not model.context_encoder.training
    assert all(p.requires_grad for p in model.context_encoder.parameters())


@pytest.mark.parametrize('drift', ['unlabeled', 'schema', 'mode', 'teacher_mode', 'config', 'config_type', 'readout', 'policy',
    'optimizer_extra', 'state_missing', 'state_shape', 'state_dtype', 'state_nan',
    'buffer_missing', 'buffer_nan', 'buffer_conflict'])
def test_versioned_rejects_semantic_or_tensor_drift_before_any_load_state_dict(tmp_path, monkeypatch, drift):
    model = versioned_model()
    def mutate(p):
        if drift == 'unlabeled': p.pop('metadata')
        elif drift == 'schema': p['schema_version'] = 'legacy'
        elif drift == 'mode': p['metadata']['feature_mode'] = 'pca'
        elif drift == 'teacher_mode': p['metadata']['teacher_feature_mode'] = 'zero'
        elif drift == 'config': p['metadata']['stage_a_v2_config']['dropout'] = .3
        elif drift == 'config_type': p['metadata']['stage_a_v2_config']['orientation_features'] = 1
        elif drift == 'readout': p['metadata']['readout_encoder'] = 'ema'
        elif drift == 'policy': p['encoder_policy']['online'] = 'teacher_encoder'
        elif drift == 'optimizer_extra': p['optimizer'] = {}
        elif drift == 'state_missing': p['model_state'].pop('context_encoder.relpos_proj.weight')
        elif drift == 'state_shape': p['model_state']['context_encoder.relpos_proj.weight'] = torch.zeros(1)
        elif drift == 'state_dtype': p['model_state']['context_encoder.relpos_proj.weight'] = p['model_state']['context_encoder.relpos_proj.weight'].double()
        elif drift == 'state_nan': p['model_state']['context_encoder.relpos_proj.weight'].fill_(float('nan'))
        elif drift == 'buffer_missing': p['named_buffers'].pop('context_encoder.rbf_centers')
        elif drift == 'buffer_nan': p['named_buffers']['context_encoder.rbf_centers'].fill_(float('nan'))
        else: p['named_buffers']['_ema_step'].add_(1)
    _, args = save_versioned(tmp_path, model, mutate=mutate)
    # Replacement encoder construction has its own strict initialization loads;
    # no full Stage A load may occur before all payload validation completes.
    monkeypatch.setattr(StageARelposV1, 'load_state_dict', lambda *a, **kw: pytest.fail('invalid payload must fail before load'))
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError): bridge.load_versioned_payload(**args)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('drift', ['mode', 'config', 'readout', 'hash'])
def test_versioned_expected_identity_is_mandatory(tmp_path, drift, monkeypatch):
    _, args = save_versioned(tmp_path, versioned_model())
    if drift == 'mode': args['expected_feature_mode'] = 'pca'
    elif drift == 'config': args['expected_config'] = replace(args['expected_config'], struct_recon_dim=32)
    elif drift == 'readout': args['expected_readout_encoder'] = 'ema'
    else:
        args['expected_sha256'] = '0'*64
        monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('hash failure must precede deserialization'))
    with pytest.raises(ValueError): bridge.load_versioned_payload(**args)


def test_versioned_hash_change_during_load_rejected(tmp_path, monkeypatch):
    _, args = save_versioned(tmp_path, versioned_model())
    load = torch.load
    def changed(path, **kwargs):
        result = load(path, **kwargs)
        with Path(path).open('ab') as f: f.write(b'drift')
        return result
    monkeypatch.setattr(torch, 'load', changed)
    with pytest.raises(ValueError, match='hash differs'): bridge.load_versioned_payload(**args)


def test_explicit_modes_and_cpu_defaults_checked_before_any_file_read(monkeypatch):
    monkeypatch.setattr(bridge, '_pin', lambda *a: pytest.fail('invalid contract must fail before file access'))
    with pytest.raises(ValueError):
        bridge.load_warm_start('unused', original_sha256='0'*64, current_path='unused2', current_sha256='0'*64, feature_mode='LOCAL')
    monkeypatch.setattr(torch, 'get_default_device', lambda: torch.device('cuda'))
    with pytest.raises(ValueError, match='default CPU device'):
        bridge.load_warm_start('unused', original_sha256='0'*64, current_path='unused2', current_sha256='0'*64, feature_mode='local')
    with pytest.raises(ValueError, match='default CPU device'):
        bridge.load_versioned_payload('unused', expected_sha256='0'*64, expected_feature_mode='local', expected_config=config(), expected_readout_encoder='online')
