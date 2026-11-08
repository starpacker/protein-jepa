"""In-memory synthetic Stage A candidates only; no checkpoint or dataset IO."""
import copy
import json
from dataclasses import replace

import pytest
import torch

from protein_jepa_frame.models.stage_a_v2 import StageAV2, StageAV2Config
from protein_jepa_frame.models.stage_a_relpos_v1 import (
    StageARelposV1, build_checkpoint_metadata, validate_checkpoint_metadata,
)


def config():
    return StageAV2Config(latent_dim=16, encoder_depth=1, num_heads=4,
        num_rbf=5, dropout=.2, max_residues=32, predictor_depth=1,
        predictor_spatial_depth=1, predictor_heads=4, max_frames=8,
        struct_recon_dim=24, struct_recon_pairs=6)


def clones(mapping):
    return {k: v.detach().clone() for k, v in mapping.items()}


def same(a, b):
    assert list(a) == list(b)
    for k in a:
        assert a[k].shape == b[k].shape and a[k].dtype == b[k].dtype, k
        assert torch.equal(a[k].contiguous().reshape(-1).view(torch.uint8),
                           b[k].contiguous().reshape(-1).view(torch.uint8)), k


def matched(mode='pca'):
    torch.manual_seed(31)
    state = torch.get_rng_state().clone()
    old = StageAV2(config())
    after = torch.get_rng_state().clone()
    torch.set_rng_state(state)
    new = StageARelposV1(config(), feature_mode=mode)
    assert torch.equal(after, torch.get_rng_state())
    same(old.state_dict(), new.state_dict())
    same(dict(old.named_buffers()), dict(new.named_buffers()))
    assert not new.teacher_encoder.training
    return old, new


def batches():
    g = torch.Generator().manual_seed(57)
    x = torch.randn(2, 4, 8, 4, 3, generator=g)
    mask = torch.ones(2, 4, 8, dtype=torch.bool); mask[:, :, -1] = False
    aa = torch.arange(8)[None].expand(2, -1)
    ri = aa.clone(); ci = torch.zeros_like(aa); ci[:, 4:] = 1
    spatial = dict(coords=x[:, :3], residue_mask=mask[:, :3], aatype=aa,
                   residue_index=ri, chain_id=ci, temperature_k=torch.tensor([300., 350.]))
    temporal = {f'context_{k}': v for k, v in spatial.items() if k != 'temperature_k'}
    temporal.update(future_coords=x[:, 3:], future_residue_mask=mask[:, 3:],
        context_time_ps=torch.tensor([[0., 1000., 2000.]]).expand(2, -1),
        future_time_ps=torch.tensor([[3000.], [6000.]]),
        temperature_k=spatial['temperature_k'])
    return spatial, temporal


@pytest.mark.parametrize('mode', ['pca', 'zero', 'local'])
def test_initial_state_all_buffers_rng_teacher_and_deepcopy(mode):
    _, model = matched(mode)
    assert 'context_encoder.rbf_centers' in dict(model.named_buffers())
    assert 'context_encoder.rbf_centers' not in model.state_dict()
    twin = copy.deepcopy(model)
    for obj in (model, twin):
        assert obj.feature_mode == obj.context_encoder.feature_mode == obj.teacher_encoder.feature_mode == mode
        assert all(not p.requires_grad for p in obj.teacher_encoder.parameters())
        assert obj.select_readout_encoder('online') is obj.context_encoder
        assert obj.select_readout_encoder('ema') is obj.teacher_encoder
        obj.train(); assert obj.context_encoder.training and not obj.teacher_encoder.training
        obj.eval(); assert not obj.context_encoder.training and not obj.teacher_encoder.training


@pytest.mark.parametrize('training', [False, True])
def test_full_pca_spatial_temporal_auxiliary_gradient_and_rng_parity(training):
    old, new = matched()
    old.train(training); new.train(training)
    spatial, temporal = batches()
    before = torch.get_rng_state().clone()
    def evaluate(model):
        s = model.forward_spatial(spatial)
        t = model.forward_temporal(temporal)
        loss = (s['pred_factors']-s['target_factors']).square().mean()
        loss = loss+(t['pred_factors']-t['target_factors']).square().mean()
        loss = loss+model.struct_recon_loss(t['context_H'], temporal['context_coords'], temporal['context_residue_mask'], n_pairs=6)
        loss = loss+model.displacement_readout_loss(t['pred_factors'], t['target_delta_local'], t['valid_mask'], t['synth_delta_H'])
        loss.backward()
        return s, t, loss
    a = evaluate(old); after = torch.get_rng_state().clone()
    torch.set_rng_state(before); b = evaluate(new)
    assert torch.equal(after, torch.get_rng_state())
    same(a[0], b[0]); same(a[1], b[1]); assert torch.equal(a[2], b[2])
    ap, bp = dict(old.named_parameters()), dict(new.named_parameters())
    assert list(ap) == list(bp)
    for key, p in ap.items():
        assert (p.grad is None) == (bp[key].grad is None), key
        if p.grad is not None:
            same({key:p.grad}, {key:bp[key].grad})
            assert torch.isfinite(p.grad).all(), key
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in new.context_encoder.parameters())
    assert all(p.grad is None for p in new.teacher_encoder.parameters())
    assert not b[0]['h_teacher_full'].requires_grad and not b[1]['h_teacher_full'].requires_grad


@pytest.mark.parametrize('momentum', [None, .7])
def test_ema_parameter_buffer_step_and_rng_parity(momentum):
    old, new = matched(); old.train(); new.train()
    with torch.no_grad():
        for model in (old, new):
            for p in model.context_encoder.parameters(): p.add_(.03)
            model.context_encoder.rbf_centers.add_(.2)
    before = torch.get_rng_state().clone()
    a = old.update_ema(momentum); b = new.update_ema(momentum)
    assert a == b and int(new._ema_step) == 1
    same(old.state_dict(), new.state_dict())
    same(dict(old.named_buffers()), dict(new.named_buffers()))
    assert torch.equal(before, torch.get_rng_state())


def warm_inputs():
    torch.manual_seed(6); original = StageAV2(config())
    original._ema_step.fill_(17)
    torch.manual_seed(8); current = StageAV2(config()).context_encoder
    with torch.no_grad(): current.rbf_centers.add_(.25)
    return dict(original_full_state=clones(original.state_dict()),
        current_encoder_state=clones(current.state_dict()),
        original_named_buffers=clones(dict(original.named_buffers())),
        current_encoder_named_buffers=clones(dict(current.named_buffers())))


@pytest.mark.parametrize('mode', ['pca', 'zero', 'local'])
def test_warm_start_current_online_then_teacher_and_fresh_ema(mode):
    model = StageARelposV1(config(), feature_mode=mode); inputs = warm_inputs()
    before = copy.deepcopy(inputs); rng = torch.get_rng_state().clone()
    for p in model.parameters(): p.grad = torch.ones_like(p)
    audit = model.warm_start_in_memory(**inputs)
    assert audit['original_ema_step'] == 17 and audit['ema_step'] == 0
    assert audit['teacher_synced_after_transplant'] and not audit['optimizer_loaded']
    assert torch.equal(rng, torch.get_rng_state())
    for name in inputs: same(inputs[name], before[name])
    for enc in (model.context_encoder, model.teacher_encoder):
        same(enc.state_dict(), inputs['current_encoder_state'])
        same(dict(enc.named_buffers()), inputs['current_encoder_named_buffers'])
        assert enc.feature_mode == mode
    for name, value in model.state_dict().items():
        if not name.startswith(('context_encoder.', 'teacher_encoder.')) and name != '_ema_step':
            same({name:value}, {name:inputs['original_full_state'][name]})
    assert any(not torch.equal(p, inputs['original_full_state']['teacher_encoder.'+name])
               for name, p in model.teacher_encoder.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    assert all(not p.requires_grad for p in model.teacher_encoder.parameters())
    assert all(p.requires_grad for p in model.context_encoder.parameters())
    assert not model.teacher_encoder.training
    assert next(model.context_encoder.parameters()).data_ptr() != next(model.teacher_encoder.parameters()).data_ptr()


@pytest.mark.parametrize('problem', ['original_missing', 'current_missing', 'original_buffers_missing',
    'current_buffers_missing', 'shape', 'dtype', 'nonfinite', 'buffer_conflict'])
def test_warm_start_rejected_before_any_tensor_mutation(problem):
    model = StageARelposV1(config(), feature_mode='local'); inputs = warm_inputs()
    state = clones(model.state_dict()); buffers = clones(dict(model.named_buffers()))
    if problem.endswith('_missing'):
        name = {'original_missing':'original_full_state', 'current_missing':'current_encoder_state',
            'original_buffers_missing':'original_named_buffers', 'current_buffers_missing':'current_encoder_named_buffers'}[problem]
        inputs[name].pop(next(iter(inputs[name])))
    elif problem == 'buffer_conflict': inputs['original_named_buffers']['_ema_step'].add_(1)
    else:
        key = next(iter(inputs['current_encoder_state']))
        value = inputs['current_encoder_state'][key]
        if problem == 'shape': value = value.reshape(-1)[:1]
        elif problem == 'dtype': value = value.double()
        else: value.fill_(float('nan'))
        inputs['current_encoder_state'][key] = value
    with pytest.raises(ValueError): model.warm_start_in_memory(**inputs)
    same(state, model.state_dict()); same(buffers, dict(model.named_buffers()))


@pytest.mark.parametrize('mode', ['pca', 'zero', 'local'])
def test_strict_load_preserves_labels_and_repairs_teacher_eval(mode):
    model = StageARelposV1(config(), feature_mode=mode)
    model.teacher_encoder.train()
    for p in model.teacher_encoder.parameters(): p.requires_grad_(True)
    model.load_state_dict(clones(model.state_dict()))
    assert model.feature_mode == model.context_encoder.feature_mode == model.teacher_encoder.feature_mode == mode
    assert not model.teacher_encoder.training
    assert all(not p.requires_grad for p in model.teacher_encoder.parameters())
    with pytest.raises(ValueError): model.load_state_dict(model.state_dict(), strict=False)
    with pytest.raises(ValueError): model.load_state_dict(model.state_dict(), assign=True)


@pytest.mark.parametrize('drift', ['teacher_mode', 'online_mode', 'teacher_training', 'teacher_grad'])
def test_runtime_rejects_teacher_or_mode_drift(drift):
    model = StageARelposV1(config(), feature_mode='local')
    if drift == 'teacher_mode': model.teacher_encoder.feature_mode = 'pca'
    elif drift == 'online_mode': model.context_encoder.feature_mode = 'zero'
    elif drift == 'teacher_training': model.teacher_encoder.train()
    else: next(model.teacher_encoder.parameters()).grad = torch.ones_like(next(model.teacher_encoder.parameters()))
    with pytest.raises(ValueError): model.select_readout_encoder('online')
    with pytest.raises(ValueError): model.forward_spatial(batches()[0])


@pytest.mark.parametrize('mode', ['pca', 'zero', 'local'])
@pytest.mark.parametrize('readout', ['online', 'ema'])
def test_metadata_explicit_mode_config_and_readout_json_roundtrip(mode, readout):
    model = StageARelposV1(config(), feature_mode=mode)
    metadata = build_checkpoint_metadata(model, readout_encoder=readout)
    assert validate_checkpoint_metadata(json.loads(json.dumps(metadata)), expected_feature_mode=mode,
        expected_config=config(), expected_readout_encoder=readout) == metadata
    assert 'feature_mode' not in model.state_dict()


@pytest.mark.parametrize('drift', ['missing', 'unlabeled', 'mode', 'teacher', 'online', 'schema', 'config', 'readout', 'extra'])
def test_metadata_rejects_semantic_drift(drift):
    model = StageARelposV1(config(), feature_mode='local')
    meta = build_checkpoint_metadata(model, readout_encoder='online')
    if drift == 'missing': meta.pop('feature_mode')
    elif drift == 'unlabeled': meta = model.state_dict()
    elif drift == 'mode': meta['feature_mode'] = 'pca'
    elif drift == 'teacher': meta['teacher_feature_mode'] = 'zero'
    elif drift == 'online': meta['online_feature_mode'] = 'pca'
    elif drift == 'schema': meta['schema_version'] = 'old'
    elif drift == 'config': meta['stage_a_v2_config']['dropout'] = .3
    elif drift == 'readout': meta['readout_encoder'] = 'ema'
    else: meta['extra'] = True
    with pytest.raises(ValueError):
        validate_checkpoint_metadata(meta, expected_feature_mode='local', expected_config=config(), expected_readout_encoder='online')


def test_modes_readout_and_configuration_require_explicit_supported_choices():
    before = torch.get_rng_state().clone()
    with pytest.raises(TypeError): StageARelposV1(config())
    for mode in ('', 'LOCAL', None):
        with pytest.raises(ValueError): StageARelposV1(config(), feature_mode=mode)
    with pytest.raises(ValueError): StageARelposV1(replace(config(), relative_position_features=False), feature_mode='pca')
    assert torch.equal(before, torch.get_rng_state())
    model = StageARelposV1(config(), feature_mode='zero')
    with pytest.raises(TypeError): model.select_readout_encoder()
    with pytest.raises(ValueError): model.select_readout_encoder('teacher')
    with pytest.raises(TypeError): build_checkpoint_metadata(model)


def test_constructor_rejects_non_cpu_default_before_rng_or_construction(monkeypatch):
    before = torch.get_rng_state().clone()
    monkeypatch.setattr(torch, 'get_default_device', lambda: torch.device('cuda'))
    def forbidden(*args, **kwargs):
        raise AssertionError('base constructor must not run')
    monkeypatch.setattr(StageAV2, '__init__', forbidden)
    with pytest.raises(ValueError, match='default CPU device and float32'):
        StageARelposV1(config(), feature_mode='pca')
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize('dtype', [torch.float64, torch.float16])
def test_constructor_rejects_non_float32_default_before_rng_or_construction(dtype, monkeypatch):
    before = torch.get_rng_state().clone()
    previous = torch.get_default_dtype()
    def forbidden(*args, **kwargs):
        raise AssertionError('base constructor must not run')
    monkeypatch.setattr(StageAV2, '__init__', forbidden)
    try:
        torch.set_default_dtype(dtype)
        with pytest.raises(ValueError, match='default CPU device and float32'):
            StageARelposV1(config(), feature_mode='local')
        assert torch.equal(before, torch.get_rng_state())
    finally:
        torch.set_default_dtype(previous)


def test_warm_start_snapshots_current_online_aliases_before_original_load():
    model = StageARelposV1(config(), feature_mode='local')
    original = warm_inputs()
    with torch.no_grad():
        for p in model.context_encoder.parameters(): p.add_(.375)
        model.context_encoder.rbf_centers.add_(.625)
    # These mappings intentionally retain views into the destination model.
    aliased_state = model.context_encoder.state_dict()
    aliased_buffers = dict(model.context_encoder.named_buffers())
    expected_state = clones(aliased_state)
    expected_buffers = clones(aliased_buffers)
    assert next(iter(aliased_state.values())).data_ptr() == next(model.context_encoder.parameters()).data_ptr()
    model.warm_start_in_memory(original['original_full_state'], aliased_state,
        original_named_buffers=original['original_named_buffers'],
        current_encoder_named_buffers=aliased_buffers)
    for encoder in (model.context_encoder, model.teacher_encoder):
        same(expected_state, encoder.state_dict())
        same(expected_buffers, dict(encoder.named_buffers()))
    assert int(model._ema_step) == 0
    assert not model.teacher_encoder.training
