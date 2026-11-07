"""Observed-only prediction from synthetic coordinates and initialized models."""
import copy

import pytest
import torch

from protein_jepa_frame.models.stage_a_v2 import StageAV2Config
from protein_jepa_frame.models.stage_a_relpos_v1 import StageARelposV1
from protein_jepa_frame.models.stage_a_relpos_prediction_v1 import predict_endpoint


def model(mode='pca', dim=16):
    cfg = StageAV2Config(latent_dim=dim, encoder_depth=1, num_heads=4,
        num_rbf=3, dropout=.2, max_residues=32, predictor_depth=1,
        predictor_spatial_depth=1, predictor_heads=4, max_frames=8,
        struct_recon_dim=24, struct_recon_pairs=6)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        result = StageARelposV1(cfg, feature_mode=mode)
    return result.eval()


def inputs(mode='pca', dtype=torch.float32):
    g = torch.Generator().manual_seed(43)
    coords = torch.randn(4, 8, 7, 4, 3, generator=g)
    mask = torch.ones(4, 8, 7, dtype=torch.bool); mask[:, :, -1] = False
    mask[1, :, -2] = False
    aa = torch.arange(7)[None].expand(4, -1).clone()
    ri = torch.tensor([-2, -1, 0, 0, 1, 2, 3])[None].expand(4, -1).clone()
    ci = torch.zeros(4, 7, dtype=torch.long); ci[:, 4:] = 1
    times = torch.arange(8, dtype=dtype)[None].expand(4, -1)*1000+137.5
    horizons = torch.tensor([1, 4, 16, 64])
    return dict(feature_mode=mode, history_coords=coords, history_mask=mask,
        aatype=aa, residue_index=ri, chain_id=ci, history_time_ps=times,
        endpoint_time_ps=times[:, -1]+horizons.to(dtype)*1000,
        horizon_frames=horizons, temperature_k=torch.tensor([300., 350., 400., 450.]))


def same(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                       b.contiguous().reshape(-1).view(torch.uint8))


def full_batch(data):
    g = torch.Generator().manual_seed(97)
    return dict(context_coords=data['history_coords'], context_residue_mask=data['history_mask'],
        context_aatype=data['aatype'], context_residue_index=data['residue_index'],
        context_chain_id=data['chain_id'], context_time_ps=data['history_time_ps'],
        future_coords=torch.randn(4, 1, 7, 4, 3, generator=g),
        future_residue_mask=data['history_mask'][:, -1:],
        future_time_ps=data['endpoint_time_ps'][:, None], temperature_k=data['temperature_k'])


@pytest.mark.parametrize('mode', ['pca', 'zero', 'local'])
@pytest.mark.parametrize('time_dtype', [torch.float32, torch.float64])
def test_all_horizons_exact_full_temporal_parity_without_teacher_or_target_execution(mode, time_dtype, monkeypatch):
    m = model(mode); data = inputs(mode, time_dtype)
    with torch.no_grad(): expected = m.forward_temporal_multilag(full_batch(data), lags=(0,), recent_frac=1.)
    def prohibited(*args, **kwargs):
        pytest.fail('observed-only prediction must not execute teacher, targets, or full temporal forward')
    monkeypatch.setattr(m.teacher_encoder, 'forward', prohibited)
    monkeypatch.setattr(m, '_teacher_encode_clip', prohibited)
    monkeypatch.setattr(m, '_anchor_local_ca_displacement', prohibited)
    monkeypatch.setattr(m, 'forward_temporal_multilag', prohibited)
    calls = []
    def capture(module, args, kwargs):
        assert not torch.is_grad_enabled() and not module.training
        calls.append((args, kwargs))
    hook = m.predictor.register_forward_pre_hook(capture, with_kwargs=True)
    try: actual = predict_endpoint(m, **data)
    finally: hook.remove()
    assert len(calls) == 1
    same(actual['context_H'], expected['context_H'])
    same(actual['H_observed'], expected['anchor_H'])
    same(actual['H_predicted'], expected['synth_H'])
    same(actual['pred_factors'], expected['pred_factors'])
    same(actual['anchor_factors'], expected['anchor_factors'])
    same(actual['lag_ps'], expected['lag_ps'])
    assert calls[0][1]['task_id'] == 1 and 'target_frame' not in calls[0][1]
    same(calls[0][1]['valid_mask'], data['history_mask'][:, 0])
    assert actual['audit']['horizon_frames'] == [1, 4, 16, 64]
    assert actual['audit']['encoder'] == 'online/context_encoder'
    assert not actual['audit']['teacher_called'] and not actual['audit']['future_coordinates_used']
    assert actual['audit']['normalization'] == 'none' and actual['audit']['rebasing'] == 'none'
    # The unchanged full predictor returns padded values; readout must use mask.
    assert torch.count_nonzero(actual['H_predicted'][~actual['valid_mask']]) > 0


@pytest.mark.parametrize('mode', ['pca', 'zero', 'local'])
def test_eval_rng_inputs_state_buffers_and_existing_gradients_unchanged(mode):
    m = model(mode); data = inputs(mode)
    data['history_coords'].requires_grad_(True)
    before_inputs = copy.deepcopy(data)
    state = {k:v.detach().clone() for k,v in m.state_dict().items()}
    buffers = {k:v.detach().clone() for k,v in m.named_buffers()}
    parameters = dict(m.named_parameters())
    for name, p in parameters.items():
        if p.requires_grad: p.grad = torch.full_like(p, .125)
    gradients = {k:None if p.grad is None else p.grad.clone() for k,p in parameters.items()}
    flags = {name:sub.training for name,sub in m.named_modules()}
    rng = torch.get_rng_state().clone()
    out = predict_endpoint(m, **data)
    assert torch.equal(rng, torch.get_rng_state())
    assert flags == {name:sub.training for name,sub in m.named_modules()}
    for k,v in data.items():
        if isinstance(v, torch.Tensor): same(v.detach(), before_inputs[k].detach())
        else: assert v == before_inputs[k]
    for k,v in m.state_dict().items(): same(v, state[k])
    for k,v in m.named_buffers(): same(v, buffers[k])
    for k,p in parameters.items():
        if gradients[k] is None: assert p.grad is None
        else: same(p.grad, gradients[k])
    assert data['history_coords'].grad is None
    assert all(not value.requires_grad for value in out.values() if isinstance(value, torch.Tensor))


def test_128D_output_is_compatible_with_current_distance_head_shape():
    m = model('local', dim=128)
    out = predict_endpoint(m, **inputs('local'))
    assert out['H_observed'].shape == out['H_predicted'].shape == (4, 7, 128)
    assert out['audit']['latent_dim'] == 128


@pytest.mark.parametrize('drift', ['seven_frames', 'coords_dtype', 'coords_nan', 'padding_nan',
    'mask_changes', 'mask_empty', 'mask_dtype', 'aa_dtype', 'aa_range', 'ri_shape', 'ci_dtype',
    'times_step', 'times_order', 'times_nan', 'times_integer', 'endpoint_dtype', 'endpoint_nan',
    'endpoint_zero', 'endpoint_off_grid', 'horizon_mismatch', 'horizon_unsupported', 'horizon_dtype',
    'temperature_zero', 'temperature_nan', 'temperature_dtype', 'mode_mismatch'])
def test_input_guards_before_online_execution(drift, monkeypatch):
    m = model(); data = inputs()
    if drift == 'seven_frames': data['history_coords'] = data['history_coords'][:, :-1]
    elif drift == 'coords_dtype': data['history_coords'] = data['history_coords'].double()
    elif drift == 'coords_nan': data['history_coords'][0, 0, 0, 0, 0] = float('nan')
    elif drift == 'padding_nan': data['history_coords'][0, 0, -1, 0, 0] = float('nan')
    elif drift == 'mask_changes': data['history_mask'][0, 1, 0] = False
    elif drift == 'mask_empty': data['history_mask'][0] = False
    elif drift == 'mask_dtype': data['history_mask'] = data['history_mask'].long()
    elif drift == 'aa_dtype': data['aatype'] = data['aatype'].float()
    elif drift == 'aa_range': data['aatype'][0, 0] = 21
    elif drift == 'ri_shape': data['residue_index'] = data['residue_index'][:, :-1]
    elif drift == 'ci_dtype': data['chain_id'] = data['chain_id'].int()
    elif drift == 'times_step': data['history_time_ps'][0, 2] += .5
    elif drift == 'times_order': data['history_time_ps'] = data['history_time_ps'].flip(1)
    elif drift == 'times_nan': data['history_time_ps'][0, 0] = float('nan')
    elif drift == 'times_integer': data['history_time_ps'] = data['history_time_ps'].long()
    elif drift == 'endpoint_dtype': data['endpoint_time_ps'] = data['endpoint_time_ps'].double()
    elif drift == 'endpoint_nan': data['endpoint_time_ps'][0] = float('nan')
    elif drift == 'endpoint_zero': data['endpoint_time_ps'][0] = data['history_time_ps'][0, -1]
    elif drift == 'endpoint_off_grid': data['endpoint_time_ps'][0] += .5
    elif drift == 'horizon_mismatch': data['horizon_frames'][0] = 4
    elif drift == 'horizon_unsupported': data['horizon_frames'][0] = 2
    elif drift == 'horizon_dtype': data['horizon_frames'] = data['horizon_frames'].float()
    elif drift == 'temperature_zero': data['temperature_k'][0] = 0
    elif drift == 'temperature_nan': data['temperature_k'][0] = float('nan')
    elif drift == 'temperature_dtype': data['temperature_k'] = data['temperature_k'].double()
    else: data['feature_mode'] = 'local'
    monkeypatch.setattr(m, 'encode_temporal', lambda **kw: pytest.fail('invalid input must fail before encoder'))
    with pytest.raises(ValueError): predict_endpoint(m, **data)


@pytest.mark.parametrize('drift', ['training', 'child_training', 'double', 'teacher_mode', 'teacher_grad'])
def test_model_contract_guards(drift):
    m = model(); data = inputs()
    if drift == 'training': m.train()
    elif drift == 'child_training': m.predictor.train()
    elif drift == 'double': m.double()
    elif drift == 'teacher_mode': m.teacher_encoder.feature_mode = 'local'
    else: next(m.teacher_encoder.parameters()).requires_grad_(True)
    with pytest.raises(ValueError): predict_endpoint(m, **data)


def test_autocast_future_arguments_and_missing_mode_are_rejected():
    m = model(); data = inputs()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        with pytest.raises(ValueError, match='autocast'): predict_endpoint(m, **data)
    with pytest.raises(TypeError): predict_endpoint(m, **data, future_coords=torch.zeros(1))
    data.pop('feature_mode')
    with pytest.raises(TypeError): predict_endpoint(m, **data)


def test_nonfinite_predictor_output_rejected(monkeypatch):
    m = model()
    monkeypatch.setattr(m.predictor, 'forward', lambda *args, **kwargs: torch.full((4,7,4,4), float('nan')))
    with pytest.raises(ValueError, match='finite'): predict_endpoint(m, **inputs())
