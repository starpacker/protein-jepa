"""Synthetic hooks and numerical identities only; no real model/data loading."""
from itertools import permutations, product

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn.modules import module as module_hooks

from protein_jepa_frame.models.stage_c.relpos_counterfactual_v1 import (
    pair_counterfactual_audit, relpos_input_override,
)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.relpos_proj = nn.Linear(3, 7, bias=False)
        self.other_proj = nn.Linear(2, 7, bias=False)
        self.norm = nn.LayerNorm(7, elementwise_affine=False)
        self.calls = 1
        self.fail_after = False

    def forward(self, positions, other):
        projected = None
        for _ in range(self.calls):
            projected = self.relpos_proj(positions)
        if self.fail_after:
            raise RuntimeError('synthetic forward failure')
        return self.norm(torch.tanh(projected + self.other_proj(other)))


def fixture():
    torch.manual_seed(87)
    model = TinyEncoder().eval().requires_grad_(False)
    positions = torch.randn(1, 8, 3)
    positions[0, 0, 0] = -0.0
    other = torch.randn(1, 8, 2)
    mask = torch.ones(8, dtype=torch.bool)
    return model, positions, other, mask


def ps():
    for order in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            P = torch.zeros(3, 3, dtype=torch.float64)
            P[list(order), torch.arange(3)] = torch.tensor(signs, dtype=torch.float64)
            yield P, order, signs


def test_identity_exact_same_input_object_and_full_forward_parity():
    model, x, other, mask = fixture()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    before_x, before_other, rng = x.clone(), other.clone(), torch.get_rng_state().clone()
    with torch.no_grad():
        baseline = model(x, other)
        with relpos_input_override(model, x, torch.eye(3), mask) as w:
            sham = model(x, other)
        post = model(x, other)
    assert torch.equal(baseline.view(torch.int32), sham.view(torch.int32))
    assert torch.equal(baseline.view(torch.int32), post.view(torch.int32))
    assert w['call_count'] == 1 and w['is_identity'] and w['identity_input_object_preserved']
    assert w['hook_removed'] and not model.relpos_proj._forward_pre_hooks
    assert torch.equal(w['input'].view(torch.int32), w['output'].view(torch.int32))
    assert torch.equal(x.view(torch.int32), before_x.view(torch.int32))
    assert torch.equal(other, before_other) and torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())


def test_all48_signed_column_replacements_match_independent_numpy_exactly():
    model, x, other, mask = fixture()
    with torch.no_grad():
        for P, order, signs in ps():
            expected_np = x.numpy()[..., list(order)] * np.asarray(signs, dtype=np.float32)
            expected = torch.from_numpy(expected_np.copy())
            manual = model(expected, other)
            with relpos_input_override(model, x, P, mask) as w:
                actual = model(x, other)
            assert torch.equal(w['output'].view(torch.int32), expected.view(torch.int32))
            assert torch.equal(actual, manual)
            assert torch.equal(w['input'].view(torch.int32), x.view(torch.int32))
            assert w['hook_removed'] and not model.relpos_proj._forward_pre_hooks


@pytest.mark.parametrize('kind', ['zero_calls', 'two_calls', 'forward_error', 'wrong_input', 'signed_zero'])
def test_hook_removed_on_every_failure(kind):
    model, x, other, mask = fixture()
    if kind == 'zero_calls': model.calls = 0
    if kind == 'two_calls': model.calls = 2
    if kind == 'forward_error': model.fail_after = True
    actual = x.clone()
    if kind == 'wrong_input': actual[0, 0, 1] += .1
    if kind == 'signed_zero': actual[0, 0, 0] = 0.0
    with torch.no_grad():
        with pytest.raises((ValueError, RuntimeError)):
            with relpos_input_override(model, x, torch.eye(3), mask) as w:
                if kind != 'zero_calls':
                    model(actual, other)
        assert w['hook_removed']
        assert not model.relpos_proj._forward_pre_hooks


@pytest.mark.parametrize('kind', ['train', 'grad_parameter', 'parameter_grad', 'double', 'wrong_linear', 'nonlinear', 'canonical_double', 'nonfinite', 'padding', 'empty_mask', 'bad_P', 'fractional_P'])
def test_context_validation_fails_before_hook_registration(kind):
    model, x, other, mask = fixture()
    P = torch.eye(3)
    if kind == 'train': model.train()
    if kind == 'grad_parameter': model.relpos_proj.weight.requires_grad_(True)
    if kind == 'parameter_grad': model.relpos_proj.weight.grad = torch.ones_like(model.relpos_proj.weight)
    if kind == 'double': model.double()
    if kind == 'wrong_linear': model.relpos_proj = nn.Linear(4, 7).eval().requires_grad_(False)
    if kind == 'nonlinear': model.relpos_proj = nn.Identity().eval()
    if kind == 'canonical_double': x = x.double()
    if kind == 'nonfinite': x[0, 1, 0] = float('nan')
    if kind == 'padding': mask[-1] = False
    if kind == 'empty_mask': mask[:] = False
    if kind == 'bad_P': P[0, 1] = 1
    if kind == 'fractional_P': P *= .5
    with torch.no_grad(), pytest.raises(ValueError):
        with relpos_input_override(model, x, P, mask):
            pytest.fail('entered invalid context')
    assert not model.relpos_proj._forward_pre_hooks


def test_grad_enabled_and_cpu_autocast_rejected():
    model, x, other, mask = fixture()
    with pytest.raises(ValueError, match='no_grad'):
        with relpos_input_override(model, x, torch.eye(3), mask):
            pass
    with torch.no_grad(), torch.autocast('cpu'), pytest.raises(ValueError, match='autocast'):
        with relpos_input_override(model, x, torch.eye(3), mask):
            pass


@pytest.mark.parametrize('global_hook', [False, True])
def test_preexisting_hooks_rejected_without_removing_them(global_hook):
    model, x, other, mask = fixture()
    handle = (module_hooks.register_module_forward_pre_hook(lambda m, a: None) if global_hook
              else model.other_proj.register_forward_hook(lambda m, a, o: None))
    try:
        with torch.no_grad(), pytest.raises(ValueError, match='hooks'):
            with relpos_input_override(model, x, torch.eye(3), mask):
                pass
        container = module_hooks._global_forward_pre_hooks if global_hook else model.other_proj._forward_hooks
        assert handle.id in container
        assert not model.relpos_proj._forward_pre_hooks
    finally:
        handle.remove()


def pair_inputs():
    g = torch.Generator().manual_seed(59)
    h0 = torch.randn(1, 8, 6, generator=g)
    h1 = h0 + torch.randn(1, 8, 6, generator=g)
    y0 = h0 + torch.randn(1, 8, 6, generator=g) * .4
    y1 = h1 + torch.randn(1, 8, 6, generator=g) * .8
    return h0, h1, y1, y0, y1.clone(), torch.ones(8, dtype=torch.bool)


def test_pair_all_effects_and_tangents_match_independent_numpy():
    inputs = pair_inputs()
    out = pair_counterfactual_audit(*inputs)
    a, s = out['arrays'], out['summary']
    x0, x1, y1, y0, _, _ = inputs
    x0, x1, y1, y0 = [x[0].numpy().astype(float) for x in (x0, x1, y1, y0)]
    J = np.eye(6) - np.ones((6, 6))/6
    baseline, E0, E1 = x1-x0, y0-x0, y1-x1
    for name, start, end, effect in [('original', x0, x1, E1*0), ('next_only', x0, y1, E1), ('shared_P', y0, y1, E1-E0)]:
        delta = end-start
        for centered in (False, True):
            D, B, E = [x @ J if centered else x for x in (delta, baseline, effect)]
            prefix = 'centered' if centered else 'raw'
            np.testing.assert_allclose(a[name+'__'+prefix+'_cross'].numpy(), 2*np.sum(B*E, -1), atol=1e-13)
            np.testing.assert_allclose(a[name+'__'+prefix+'_increment_squared'].numpy(), np.sum(D*D, -1), atol=1e-13)
            assert s[name][prefix+'_decomposition_max_abs_error'] < 1e-13
        tangent = []
        for x, d in zip(start, delta):
            z = J @ x; u = z/np.linalg.norm(z); T = J-np.outer(u, u)
            tangent.append(np.sum((T@d)**2))
        np.testing.assert_allclose(a[name+'__tangent_increment_squared'].numpy(), tangent, atol=1e-13)
        assert s[name]['finite_chord_max_abs_error'] < 1e-13
    assert s['next_only']['tangent_reference'] == 'original_current_H'
    assert s['shared_P']['tangent_reference'] == 'shared_P_current_H'


def test_negative_cross_term_retained_without_explained_fraction():
    h0 = torch.zeros(1, 2, 3); h1 = torch.ones(1, 2, 3)
    y1 = h1*.5
    out = pair_counterfactual_audit(h0, h1, y1, h0, y1, torch.ones(2, dtype=torch.bool))
    assert out['summary']['next_only']['raw_cross_mean'] == -3
    assert out['summary']['next_only']['raw_increment_squared_mean'] == .75
    assert out['summary']['next_only']['raw_effect_squared_mean'] == .75
    assert all('fraction' not in k for k in out['summary']['next_only'])


def test_masked_invariance_and_mode_specific_undefined_directions():
    args = list(pair_inputs()); mask = args[-1]; mask[-1] = False
    args[0][:, 0] = 2.  # baseline/next-only direction undefined only for row0
    args[3][:, 1] = 3.  # shared direction undefined only for row1
    first = pair_counterfactual_audit(*args)
    for H in args[:-1]: H[:, ~mask] = float('nan')
    second = pair_counterfactual_audit(*args)
    assert first['summary'] == second['summary']
    a = second['arrays']
    assert not a['next_only__tangent_valid'][0] and a['shared_P__tangent_valid'][0]
    assert a['next_only__tangent_valid'][1] and not a['shared_P__tangent_valid'][1]
    assert second['summary']['residue_count'] == 7
    assert torch.isnan(a['next_only__tangent_increment_squared'][0])


def test_all_undefined_tangents_null_but_raw_centered_rows_retained():
    h0 = torch.ones(1, 4, 3)
    h1 = torch.arange(12.).reshape(1, 4, 3)
    out = pair_counterfactual_audit(h0, h1, h1, h0, h1, torch.ones(4, dtype=torch.bool))
    for mode in ('original', 'next_only', 'shared_P'):
        assert out['summary'][mode]['tangent_increment_squared_mean'] is None
        assert out['summary'][mode]['tangent_undefined_count'] == 4
        assert out['summary'][mode]['raw_increment_squared_mean'] > 0
        assert out['summary'][mode]['centered_increment_squared_mean'] > 0


@pytest.mark.parametrize('kind', ['next_mismatch', 'signed_zero_mismatch', 'nan_valid', 'double', 'shape'])
def test_pair_validation(kind):
    args = list(pair_inputs())
    if kind == 'next_mismatch': args[4][0, 0, 0] += .1
    if kind == 'signed_zero_mismatch':
        args[2][0, 0, 0] = -0.0; args[4][0, 0, 0] = 0.0
    if kind == 'nan_valid': args[0][0, 0, 0] = float('nan')
    if kind == 'double': args[0] = args[0].double()
    if kind == 'shape': args[0] = args[0][:, :, :3]
    with pytest.raises(ValueError): pair_counterfactual_audit(*args)


def test_shared_gauge_is_not_assumed_to_preserve_H_increment_norm():
    model, x, other, mask = fixture()
    x1 = x + torch.randn_like(x)*.2
    other1 = other + .1
    P = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    with torch.no_grad():
        h0, h1 = model(x, other), model(x1, other1)
        with relpos_input_override(model, x, P, mask) as w0: y0 = model(x, other)
        with relpos_input_override(model, x1, P, mask) as w1: y1 = model(x1, other1)
    # Input relabeling preserves the physical feature increment norm, but the
    # learned projection and nonlinear encoder are not permutation-equivariant.
    torch.testing.assert_close((w1['output']-w0['output']).square().sum(), (x1-x).square().sum())
    out = pair_counterfactual_audit(h0, h1, y1, y0, y1, mask)
    assert abs(out['summary']['shared_P']['raw_increment_squared_mean'] - out['summary']['original']['raw_increment_squared_mean']) > 1e-5


@pytest.mark.parametrize('mode', ['grad', 'autocast'])
def test_forward_mode_change_inside_context_rejected_and_hook_removed(mode):
    model, x, other, mask = fixture()
    with torch.no_grad(), pytest.raises(ValueError, match='forward requires'):
        with relpos_input_override(model, x, torch.eye(3), mask) as w:
            with (torch.enable_grad() if mode == 'grad' else torch.autocast('cpu')):
                model(x, other)
    assert w['hook_removed']
    assert not model.relpos_proj._forward_pre_hooks
