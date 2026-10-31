import torch
import pytest

from protein_jepa_frame.training.train_lasp_ca_dynamics import (
    anchor_noise_factor, elastic_network_kernel)


def test_elastic_kernel_is_psd_padded_and_finite():
    torch.manual_seed(5)
    anchor = torch.randn(2, 12, 3) * 6.0
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[:, -3:] = False
    factor = elastic_network_kernel(anchor, mask, cutoff=10., mix=.8)
    assert torch.isfinite(factor).all()
    cov = factor @ factor.transpose(-1, -2)
    # PSD: smallest eigenvalue is non-negative.
    assert float(torch.linalg.eigvalsh(cov.double()).min()) > -1e-8
    # Padding residues cannot exchange noise with valid residues.
    torch.testing.assert_close(cov[:, -3:, :-3], torch.zeros(2, 3, 9), atol=1e-6, rtol=0)


def test_elastic_kernel_is_se3_invariant():
    """Rotation + translation of the anchor must not change the kernel."""
    torch.manual_seed(11)
    anchor = torch.randn(1, 10, 3) * 5.0
    mask = torch.ones(1, 10, dtype=torch.bool)
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if float(torch.det(q)) < 0:
        q[:, 0] = -q[:, 0]
    moved = anchor @ q + 17.0
    a = elastic_network_kernel(anchor, mask, cutoff=10., mix=.8)
    b = elastic_network_kernel(moved, mask, cutoff=10., mix=.8)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_elastic_kernel_identity_mix_preserves_seeded_kernel_rollout():
    """mix=0 reproduces the isotropic control bitwise, as the Gaussian kernel does."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig
    torch.manual_seed(47)
    kernel = StageB1b(StageB1bConfig(energy_depth=1, bath_dim=0)).eval()
    h = torch.randn(1, 6, 128)
    mask = torch.ones(1, 6, dtype=torch.bool)
    anchor = torch.randn(1, 6, 3) * 4.0
    with sdpa_kernel(SDPBackend.MATH):
        base = kernel.rollout(h, mask, torch.tensor([320.]), 3,
                              generator=torch.Generator().manual_seed(9))
        identity = kernel.rollout(h, mask, torch.tensor([320.]), 3,
                                  generator=torch.Generator().manual_seed(9),
                                  noise_factor=torch.eye(6)[None])
        from_elastic = kernel.rollout(
            h, mask, torch.tensor([320.]), 3,
            generator=torch.Generator().manual_seed(9),
            noise_factor=elastic_network_kernel(anchor, mask, cutoff=10., mix=0.))
    torch.testing.assert_close(base, identity, atol=0, rtol=0)
    torch.testing.assert_close(base, from_elastic, atol=1e-6, rtol=1e-5)


def test_elastic_kernel_encodes_residue_flexibility_variation():
    """Unlike the isotropic kernel, the diagonal must actually vary across residues.

    The isotropic Gaussian kernel is normalized to unit marginal variance on every
    residue, so it cannot express which residues are soft. This is the property the
    new kernel exists to provide.
    """
    torch.manual_seed(3)
    # A dense core and a floppy tail: the tail must receive larger fluctuations.
    core = torch.randn(1, 12, 3) * 1.5
    tail = torch.randn(1, 6, 3) * 1.5 + torch.tensor([20.0, 0.0, 0.0])
    anchor = torch.cat([core, tail], dim=1)
    mask = torch.ones(1, 18, dtype=torch.bool)
    elastic = elastic_network_kernel(anchor, mask, cutoff=10., mix=.8)
    gaussian = anchor_noise_factor(anchor, mask, .8, 8.)
    e_diag = (elastic @ elastic.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1)
    g_diag = (gaussian @ gaussian.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1)
    assert float(g_diag.std()) < 1e-5, 'isotropic kernel is flat by construction'
    assert float(e_diag.std()) > 1e-3, 'elastic kernel must vary'
    assert float(e_diag[0, 12:].mean()) > float(e_diag[0, :12].mean()), \
        'the detached tail must be softer than the packed core'


def test_elastic_kernel_matches_noise_factor_contract():
    """Same [B, L, L] Cholesky contract as anchor_noise_factor, so plumbing is shared."""
    torch.manual_seed(7)
    anchor = torch.randn(3, 9, 3) * 5.0
    mask = torch.ones(3, 9, dtype=torch.bool)
    factor = elastic_network_kernel(anchor, mask, cutoff=12., mix=.6)
    assert factor.shape == (3, 9, 9)
    # Lower-triangular, as cholesky requires.
    torch.testing.assert_close(factor, factor.tril(), atol=1e-6, rtol=0)


@pytest.mark.parametrize('bad', [
    dict(mix=1.0), dict(mix=-0.1), dict(cutoff=0.0), dict(temperature=0.0)])
def test_elastic_kernel_rejects_invalid_parameters(bad):
    anchor = torch.randn(1, 6, 3) * 4.0
    mask = torch.ones(1, 6, dtype=torch.bool)
    with pytest.raises(ValueError):
        elastic_network_kernel(anchor, mask, **bad)


def test_cadynamics_anchor_kernel_dispatch_and_validation():
    """The dispatch helper selects by name and refuses unknown kernels."""
    from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics
    anchor = torch.randn(1, 8, 3) * 5.0
    mask = torch.ones(1, 8, dtype=torch.bool)
    model = CADynamics.__new__(CADynamics)
    model.noise_mix = .8
    model.noise_length = 8.
    model.noise_cutoff = 10.
    model.noise_kernel = 'gaussian'
    g = model.anchor_kernel(anchor, mask)
    model.noise_kernel = 'elastic'
    e = model.anchor_kernel(anchor, mask)
    assert g.shape == e.shape == (1, 8, 8)
    assert not torch.allclose(g, e), 'the two kernels must not be identical'
    model.noise_kernel = 'nonsense'
    with pytest.raises(ValueError):
        model.anchor_kernel(anchor, mask)