"""Synthetic candidate encoders only; no checkpoint/data/model-weight IO."""
import copy

import numpy as np
import pytest
import torch

from protein_jepa_frame.models.encoder import ProteinGeometryEncoder, _within_frame_canonical_positions
from protein_jepa_frame.models.relpos_mode_encoder_v1 import (
    RelposModeProteinGeometryEncoder, local_centroid_positions,
)


def config(orientation=True):
    return dict(latent_dim=16, depth=2, num_heads=4, num_rbf=7, dropout=.2,
                max_residues=64, orientation_features=orientation,
                identity_embed_scale=.1, relative_position_features=True,
                anchor_conditioned=False)


def batch(token=True):
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(2, 3, 9, 4, 3, generator=generator)
    mask = torch.ones(2, 3, 9, dtype=torch.bool)
    mask[:, :, -1] = False
    tokens = mask.clone()
    if token:
        tokens[:, :, 2:4] = False
        tokens[1, 1] = False
    aa = torch.arange(9)[None].expand(2, -1)
    ri = torch.arange(9)[None].expand(2, -1)
    ci = torch.zeros(2, 9, dtype=torch.long)
    ci[:, 5:] = 1
    return dict(coords=x, residue_mask=mask, aatype=aa, residue_index=ri,
                chain_id=ci, token_mask=tokens if token else None)


def matched(orientation=True, mode='pca'):
    torch.manual_seed(31)
    before = torch.get_rng_state().clone()
    old = ProteinGeometryEncoder(**config(orientation))
    after = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    new = RelposModeProteinGeometryEncoder(**config(orientation), feature_mode=mode)
    assert torch.equal(after, torch.get_rng_state())
    assert list(old.state_dict()) == list(new.state_dict())
    assert all(torch.equal(a, new.state_dict()[k]) for k, a in old.state_dict().items())
    assert list(dict(old.named_buffers())) == list(dict(new.named_buffers()))
    assert all(torch.equal(a, dict(new.named_buffers())[k]) for k, a in old.named_buffers())
    return old, new


@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('orientation', [False, True])
@pytest.mark.parametrize('token', [False, True])
def test_pca_exact_outputs_every_parameter_gradient_and_RNG(training, orientation, token):
    old, new = matched(orientation)
    old.train(training); new.train(training)
    inputs = batch(token)
    probe = torch.randn(2, 3, 9, 16, generator=torch.Generator().manual_seed(47))
    state = torch.get_rng_state().clone()
    old_out = old(**inputs)
    ((old_out*probe).sum()+.031*old_out.square().sum()).backward()
    old_rng = torch.get_rng_state().clone()
    torch.set_rng_state(state)
    new_out = new(**inputs)
    ((new_out*probe).sum()+.031*new_out.square().sum()).backward()
    assert torch.equal(old_out.view(torch.int32), new_out.view(torch.int32))
    assert torch.equal(old_rng, torch.get_rng_state())
    old_params, new_params = dict(old.named_parameters()), dict(new.named_parameters())
    assert list(old_params) == list(new_params)
    assert all(p.grad is not None for p in old_params.values())
    for name, parameter in old_params.items():
        assert (parameter.grad is None) == (new_params[name].grad is None), name
        if parameter.grad is not None:
            assert torch.equal(parameter.grad.view(torch.int32), new_params[name].grad.view(torch.int32)), name


@pytest.mark.parametrize('mode', ['pca', 'zero', 'local'])
def test_all_modes_same_inventory_initialization_and_literal_feature_inputs(mode):
    old, model = matched(mode=mode)
    model.eval(); inputs = batch()
    captured = []
    hook = model.relpos_proj.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    try:
        output = model(**inputs)
    finally:
        hook.remove()
    assert len(captured) == 1
    valid = inputs['residue_mask'] & inputs['token_mask']
    flatcoords = inputs['coords'].reshape(6, 9, 4, 3)
    flatvalid = valid.reshape(6, 9)
    actual = captured[0]
    if mode == 'pca':
        assert torch.equal(actual, _within_frame_canonical_positions(flatcoords, flatvalid))
    elif mode == 'zero':
        assert torch.equal(actual, torch.zeros_like(actual))
        (output*torch.randn_like(output)).sum().backward()
        assert model.relpos_proj.weight.grad is not None
        assert torch.count_nonzero(model.relpos_proj.weight.grad) == 0
    else:
        safe = np.where(flatvalid.numpy()[..., None, None], flatcoords.double().numpy(), 0.)
        n, ca, c = safe[:, :, 0], safe[:, :, 1], safe[:, :, 2]
        normalize = lambda value: value/np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-8)
        x = normalize(c-ca); z = normalize(np.cross(x, normalize(n-ca))); y = normalize(np.cross(z, x))
        frames = np.stack([x, y, z], axis=-2)
        center = np.sum(ca, axis=1, keepdims=True)/np.maximum(flatvalid.numpy().sum(1)[:, None, None], 1)
        centered = np.where(flatvalid.numpy()[..., None], ca-center, 0.)
        expected = np.einsum('blkc,blc->blk', frames, centered)
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-6, rtol=2e-6)
    assert torch.count_nonzero(actual[~flatvalid]) == 0
    assert torch.isfinite(output).all()


@pytest.mark.parametrize('mode', ['pca', 'zero', 'local'])
def test_hidden_finite_geometry_does_not_leak_visible_H(mode):
    _, model = matched(mode=mode); model.eval()
    inputs = batch()
    valid = inputs['residue_mask'] & inputs['token_mask']
    changed = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    changed['coords'][~valid] = torch.randn_like(changed['coords'][~valid])*100+500
    with torch.no_grad():
        baseline, mutated = model(**inputs), model(**changed)
    assert torch.equal(baseline[valid].view(torch.int32), mutated[valid].view(torch.int32))
    assert torch.count_nonzero(baseline[~valid]) == torch.count_nonzero(mutated[~valid]) == 0


def test_local_centroid_uses_effective_visibility_not_full_residue_mask():
    _, model = matched(mode='local')
    inputs = batch()
    valid = (inputs['residue_mask'] & inputs['token_mask']).reshape(6, 9)
    x = inputs['coords'].reshape(6, 9, 4, 3)
    effective = model.relpos_features(x, valid)
    leaked = model.relpos_features(x, inputs['residue_mask'].reshape(6, 9))
    assert not torch.allclose(effective[valid], leaked[valid])
    # The standalone feature itself safely sanitizes hidden NaNs. Full encoder
    # forward deliberately rejects them because its other frozen branches do not.
    poisoned = x.clone(); poisoned[~valid] = float('nan')
    assert torch.equal(effective, local_centroid_positions(poisoned, valid))


@pytest.mark.parametrize('mode', ['pca', 'zero', 'local'])
def test_teacher_deepcopy_preserves_explicit_mode_state_buffers_without_RNG(mode):
    _, online = matched(mode=mode)
    rng = torch.get_rng_state().clone()
    teacher = copy.deepcopy(online).eval().requires_grad_(False)
    assert teacher.feature_mode == online.feature_mode == mode
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, teacher.state_dict()[key]) for key, value in online.state_dict().items())
    assert torch.equal(online.rbf_centers, teacher.rbf_centers)
    assert online.rbf_centers.data_ptr() != teacher.rbf_centers.data_ptr()
    online.eval()
    with torch.no_grad():
        assert torch.equal(online(**batch()), teacher(**batch()))
    online.feature_mode = 'zero' if mode != 'zero' else 'local'
    assert teacher.feature_mode == mode


@pytest.mark.parametrize('kind', ['collapsed', 'collinear', 'near_collinear', 'tiny_bond'])
def test_local_finite_degeneracy_kept_and_forward_parameter_gradients_finite(kind):
    _, model = matched(mode='local'); model.eval()
    inputs = batch(token=False)
    x = inputs['coords']; x[:] = 0
    if kind != 'collapsed': x[..., 2, 0] = 1.
    if kind in ('collinear', 'near_collinear'): x[..., 0, 0] = 1.
    if kind == 'near_collinear': x[..., 0, 1] = 1e-10
    if kind == 'tiny_bond':
        x[..., 2, 0] = 1e-10; x[..., 0, 1] = 1.
    output = model(**inputs)
    assert torch.isfinite(output).all()
    probe = torch.randn(output.shape, generator=torch.Generator().manual_seed(7))
    (output*probe).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    feature = model.relpos_features(x.reshape(6, 9, 4, 3), inputs['residue_mask'].reshape(6, 9))
    assert feature.shape == (6, 9, 3) and torch.isfinite(feature).all()


def test_local_feature_keeps_raw_Angstrom_scale_and_coordinate_gradients():
    inputs = batch(token=False)
    x = inputs['coords'].reshape(6, 9, 4, 3).clone().requires_grad_(True)
    mask = inputs['residue_mask'].reshape(6, 9)
    a = local_centroid_positions(x, mask)
    b = local_centroid_positions(3*x, mask)
    torch.testing.assert_close(b, 3*a, atol=2e-6, rtol=2e-6)
    a.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[mask]) > 0
    assert torch.count_nonzero(x.grad[~mask]) == 0


@pytest.mark.parametrize('mode', [None, '', 'LOCAL', 'anchor', 0])
def test_invalid_mode_rejected_before_constructor_RNG(mode):
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='feature_mode'):
        RelposModeProteinGeometryEncoder(**config(), feature_mode=mode)
    assert torch.equal(rng, torch.get_rng_state())


def test_mode_mandatory_validated_attribute_and_projection_never_removed():
    with pytest.raises(TypeError): RelposModeProteinGeometryEncoder(**config())
    _, model = matched()
    with pytest.raises(ValueError): model.feature_mode = 'typo'
    assert model.feature_mode == 'pca'
    opts = config(); opts['relative_position_features'] = False
    with pytest.raises(ValueError, match='remain present'):
        RelposModeProteinGeometryEncoder(**opts, feature_mode='zero')
    model.relpos_proj = None
    with pytest.raises(ValueError, match='remain present'): model(**batch())


@pytest.mark.parametrize('argument', ['constructor', 'anchor_centroid', 'anchor_basis'])
def test_anchor_branch_mixing_rejected(argument):
    if argument == 'constructor':
        opts = config(); opts['anchor_conditioned'] = True
        with pytest.raises(ValueError, match='anchor'):
            RelposModeProteinGeometryEncoder(**opts, feature_mode='local')
    else:
        _, model = matched()
        inputs = batch(); inputs[argument] = torch.zeros(6, 3) if argument == 'anchor_centroid' else torch.eye(3).expand(6, 3, 3)
        with pytest.raises(ValueError, match='anchor'): model(**inputs)


@pytest.mark.parametrize('mode', ['pca', 'zero', 'local'])
@pytest.mark.parametrize('hidden', [False, True])
def test_nonfinite_coordinates_fail_closed_even_when_hidden(mode, hidden):
    _, model = matched(mode=mode)
    inputs = batch(); row = 2 if hidden else 0
    inputs['coords'][0, 0, row, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'): model(**inputs)


def test_all_feature_modes_preserve_training_dropout_RNG_draw_sequence():
    inputs = batch()
    models = [matched(mode=mode)[1].train() for mode in ('pca', 'zero', 'local')]
    rng = torch.get_rng_state().clone()
    probe = torch.randn(2, 3, 9, 16, generator=torch.Generator().manual_seed(213))
    end_states = []
    for model in models:
        torch.set_rng_state(rng)
        output = model(**inputs)
        (output*probe).sum().backward()
        end_states.append(torch.get_rng_state().clone())
        assert torch.isfinite(output).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.equal(end_states[0], state) for state in end_states[1:])
