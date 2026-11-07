from types import SimpleNamespace

import pytest
import torch

from protein_jepa_frame.utils.contracts import state_sha256
from scripts import diagnose_ca_readout_retention as diagnostic


def quadratic_bundle():
    encoder = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        encoder.weight.copy_(torch.tensor([[1., -2.]], dtype=torch.float64))
    return SimpleNamespace(encoder=encoder)


def quadratic_evaluate(bundle, *_args, **_kwargs):
    vector = bundle.encoder.weight.flatten()
    values = {name: float((index + 1) * vector.square().sum().detach())
              for index, name in enumerate(diagnostic.OBJECTIVES)}
    return values, [], {}


def test_central_difference_matches_analytic_derivative_and_restores_exactly(monkeypatch):
    bundle = quadratic_bundle()
    before = state_sha256(bundle.encoder)
    vector = bundle.encoder.weight.detach().flatten()
    gradients = {name: 2 * (index + 1) * vector for index, name in enumerate(diagnostic.OBJECTIVES)}
    means, _, _ = quadratic_evaluate(bundle)
    monkeypatch.setattr(diagnostic, 'evaluate', quadratic_evaluate)
    results = diagnostic.finite_differences(bundle, [], (1., .1), gradients, means, (1e-5, 3e-5))
    assert len(results) == 4  # Every predeclared direction and step is retained.
    for result in results:
        for name in diagnostic.OBJECTIVES:
            assert result['central_finite_difference'][name] == pytest.approx(
                result['autodiff_directional_derivatives'][name], rel=1e-9)
        assert result['descent']['differences_from_joint']['teacher_weighted'] < 0
        assert result['opposite']['differences_from_joint']['teacher_weighted'] > 0
        assert result['descent']['actual_parameter_displacement_norm'] == pytest.approx(result['absolute_step'])
    assert state_sha256(bundle.encoder) == before


def test_failed_perturbation_restores_encoder_and_zero_gradient_is_undefined(monkeypatch):
    bundle = quadratic_bundle()
    before = state_sha256(bundle.encoder)
    gradient = torch.tensor([1., 2.], dtype=torch.float64)
    gradients = {name: gradient for name in diagnostic.OBJECTIVES}
    means, _, _ = quadratic_evaluate(bundle)
    def fail(*_args, **_kwargs):
        assert state_sha256(bundle.encoder) != before
        raise RuntimeError('injected evaluation failure')
    monkeypatch.setattr(diagnostic, 'evaluate', fail)
    with pytest.raises(RuntimeError, match='injected'):
        diagnostic.finite_differences(bundle, [], (1., .1), gradients, means, (1e-5,))
    assert state_sha256(bundle.encoder) == before
    assert diagnostic.cosine(torch.zeros_like(gradient), gradient) is None
