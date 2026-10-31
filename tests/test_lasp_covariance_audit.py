"""Pin the distinction between legacy and current collective bath noise."""
import numpy as np
import pytest
import torch

from scripts.audit_lasp_velocity_covariance import audit
from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig


def test_collective_harmonic_limit_is_target_not_twice_target():
    ratios = [audit(h, collective_mobility=True)['H_variance_ratio']
              for h in (.1, .01, .001)]
    assert 1 < ratios[-1] < ratios[1] < ratios[0]
    assert abs(ratios[-1]-1) < .001
    assert audit(.001, collective_mobility=True)['continuous_target_lyapunov_residual'] < 1e-10


def test_legacy_independent_noise_limit_is_not_collective():
    result = audit(.001)
    assert result['H_variance_ratio'] == pytest.approx(2., abs=.001)
    assert result['continuous_target_lyapunov_residual'] > 0


@pytest.mark.parametrize('factor', [.5, 1., 1.5])
def test_collective_sampler_matches_audit_noise_and_drift(factor):
    cfg = StageB1bConfig(latent_dim=2, bath_dim=1, energy_depth=1,
                        energy_heads=1, energy_hidden=8, history_hidden=8,
                        step_ps=.01, bath_coupling='velocity')
    kernel = StageB1b(cfg).double()
    kernel.__dict__['compute_forces'] = lambda H, *args, **kw: -H
    kernel.__dict__['bath'] = lambda seq, T: (torch.ones_like(seq),
        seq.new_tensor([1., 0.]).reshape(1, 1, 2, 1), seq.new_ones(1, 1, 1))
    H = torch.tensor([[[.3, 0.]]], dtype=torch.float64)
    S = torch.tensor([[[.7]]], dtype=torch.float64)
    mask = torch.ones(1, 1, dtype=torch.bool)
    F = torch.tensor([[[factor]]], dtype=torch.float64)
    seed = 17
    generator = torch.Generator().manual_seed(seed)
    hn, sn = kernel.sample_step(H, S, mask, None, generator=generator,
                                noise_factor=F, collective_mobility=True)
    generator = torch.Generator().manual_seed(seed)
    epsh = torch.randn(H.shape, generator=generator, dtype=H.dtype)[0, 0, 0].item()
    epss = torch.randn(S.shape, generator=generator, dtype=S.dtype)[0, 0, 0].item()
    h = cfg.step_ps
    m = factor**2
    A = np.array([[1-h*m, -h*m], [-h*m, 1-h-h*m]])
    B = np.sqrt(2*h)*np.array([[factor, 0.], [factor, 1.]])
    expected = A@np.array([.3, .7])+B@np.array([epsh, epss])
    np.testing.assert_allclose([hn[0, 0, 0].item(), sn[0, 0, 0].item()], expected, atol=1e-12)
    np.testing.assert_allclose(B@B.T, audit(h, collective_mobility=True,
                                           noise_factor=factor)['noise_covariance'])
