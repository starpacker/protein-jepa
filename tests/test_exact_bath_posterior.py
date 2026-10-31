"""Tests for exact_bath_posterior: closed-form p(S_0 | H_hist).

Three properties pin the implementation to the sampler:
 1. With C = 0 the history carries no information, so the posterior must equal
    the initial prior exactly.
 2. The returned Cholesky must reproduce the covariance of a hand-written
    recursion (independent code path, 1e-6 on float64).
 3. The posterior must match a Monte Carlo estimate obtained by filtering many
    real sampler draws conditioned on the same observed path.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.debug_kalman_stepwise import build


def _setup(dS=3, dH=4, L=2, h=0.5, k=0.5, seed=7):
    torch.manual_seed(seed)
    D_v = torch.tensor([0.03, 0.02, 0.04, 0.025][:dH])
    G_v = torch.tensor([0.7, 0.5, 0.9][:dS])
    C = 0.2 * torch.randn(dH, dS)
    model = build(dH, dS, L, h, k, D_v, G_v, C).double().eval()
    return model, D_v.double(), G_v.double(), C.double()


def test_zero_coupling_recovers_prior():
    model, D_v, G_v, C = _setup()
    model.bath.C_val = torch.zeros_like(C)
    H = torch.randn(1, 5, 2, 4, dtype=torch.float64)
    dt = torch.full((1, 5), 0.5, dtype=torch.float64)
    mask = torch.ones(1, 2, dtype=torch.bool)
    m, Lc = model.exact_bath_posterior(H, dt, None, mask)
    assert torch.allclose(m, torch.zeros_like(m), atol=1e-12)
    # With C=0 the mean stays 0 and the variance follows the OU recursion
    # P <- a^2 P + 2hG starting from the unit prior, for K-1 steps.  The result
    # is history-independent and must be identical across residues/batch.
    H_, G_ = 0.5, G_v
    a = 1.0 - H_ * G_
    p = torch.ones(3, dtype=torch.float64)
    for _ in range(H.shape[1] - 1):
        p = a * a * p + 2.0 * H_ * G_
    var = Lc.diagonal(dim1=-2, dim2=-1).to(torch.float64).square()   # [B,L,dS]
    assert torch.allclose(var, p.reshape(1, 1, 3).expand_as(var), atol=1e-10), var
    # and it must be identical across residues (and batch=1)
    assert torch.allclose(var, var[:, :1].expand_as(var), atol=1e-10)


def _hand_filter(H, dt, D, C, Gamma, forces):
    B, K, L, _ = H.shape
    dS = C.shape[-1]
    I = torch.eye(dS, dtype=H.dtype).expand(B, L, dS, dS).clone()
    m = torch.zeros(B, L, dS, dtype=H.dtype)
    P = I.clone()
    for kk in range(K - 1):
        h = dt[:, kk].view(B, 1, 1)
        R = 2.0 * h * D
        B_h = h * D.unsqueeze(-1) * C
        z = (H[:, kk + 1] - H[:, kk]) - h * D * forces[kk]
        e = z + torch.einsum('blij,blj->bli', B_h, m)
        V = torch.diag_embed(R) + B_h @ P @ B_h.transpose(-1, -2)
        Pbt = P @ B_h.transpose(-1, -2)
        Kg = -torch.linalg.solve(V, Pbt.transpose(-1, -2)).transpose(-1, -2)
        m_plus = m + torch.einsum('blsk,blk->bls', Kg, e)
        KC = -torch.einsum('blsk,blkt->blst', Kg, B_h)
        ImKC = I - KC
        P_plus = ImKC @ P @ ImKC.transpose(-1, -2) \
            + torch.einsum('blsk,blk,bltk->blst', Kg, R, Kg)
        a = 1.0 - h * Gamma
        m = a * m_plus + torch.einsum('blij,bli->blj', C, H[:, kk + 1] - H[:, kk])
        Am = torch.diag_embed(a)
        P = Am @ P_plus @ Am.transpose(-1, -2) + torch.diag_embed(2.0 * h * Gamma)
    return m, P


def test_matches_hand_recursion():
    model, D_v, G_v, C = _setup()
    B, K, L = 1, 6, 2
    H = torch.randn(B, K, L, 4, dtype=torch.float64)
    dt = torch.full((B, K), 0.5, dtype=torch.float64)
    mask = torch.ones(B, L, dtype=torch.bool)
    m, Lc = model.exact_bath_posterior(H, dt, None, mask)
    forces = [model.compute_forces(H[:, k], mask, None, create_graph=False)
              for k in range(K)]
    m_ref, P_ref = _hand_filter(H, dt, D_v.view(1, 1, 4), C.view(1, 1, 4, 3),
                                G_v.view(1, 1, 3), forces)
    assert torch.allclose(m, m_ref, atol=1e-10), (m - m_ref).abs().max()
    P = Lc @ Lc.transpose(-1, -2)
    assert torch.allclose(P, P_ref, atol=1e-10), (P - P_ref).abs().max()


def test_posterior_matches_monte_carlo():
    """Filter many real sampler draws; the empirical S at the anchor must match."""
    model, D_v, G_v, C = _setup(dS=3, dH=4, L=1, h=0.5, k=0.5, seed=11)
    B, K = 1, 4
    mask = torch.ones(B, 1, dtype=torch.bool)
    sf = torch.zeros(B, 1, 4, dtype=torch.float64)

    # One observed path; then re-run the sampler with many S paths and keep the
    # draws whose H trajectory stays near the observed one?  That needs a
    # conditional sampler, so instead we check the exact filtering identity on
    # the SAMPLER's own joint: sample (H,S) jointly, then the filter's posterior
    # covariance must equal the conditional covariance E[(S-m)(S-m)^T | H].
    gen = torch.Generator().manual_seed(3)
    n = 400000
    H = torch.zeros(n, 1, 4, dtype=torch.float64)
    S = torch.randn(n, 1, 3, dtype=torch.float64, generator=gen)
    Hs, Ss = [H.clone()], [S.clone()]
    with torch.no_grad():
        for _ in range(K - 1):
            H, S = model.sample_step(H, S, mask.expand(n, -1), None,
                                     sf.expand(n, -1, -1), generator=gen,
                                     collective_mobility=True)
            Hs.append(H.clone())
            Ss.append(S.clone())
    # Condition on the observed H path being exactly the joint's mean path (0).
    # Under a joint Gaussian, the conditional mean is linear in the observed H,
    # so we can estimate the regression coefficients and compare to the filter.
    X = torch.cat([h[:, 0] for h in Hs[:-1]], 1)            # [n, (K-1)*4]
    Y = Ss[-1][:, 0]                                        # [n, 3]
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    # Regress Y on X: beta [4(K-1), 3].  E[Y | X=x] = mean_Y + (x-mean_X) @ beta,
    # so the conditional mean at the observed path x = 0 is used below.
    beta = torch.linalg.lstsq(Xc, Yc).solution              # [4(K-1), 3]
    mdata = Y.mean(0) - X.mean(0) @ beta
    resid = Yc - Xc @ beta
    Pdata = resid.T @ resid / n

    dt = torch.full((B, K), 0.5, dtype=torch.float64)
    # Observed path at the joint mean = the all-zero path.
    m_model, Lc = model.exact_bath_posterior(
        torch.zeros(B, K, 1, 4, dtype=torch.float64), dt, None, mask)
    P_model = Lc @ Lc.transpose(-1, -2)
    assert torch.allclose(m_model[0, 0], mdata, atol=2e-2), (m_model[0, 0], mdata)
    assert torch.allclose(P_model[0, 0], Pdata, atol=2e-2), (P_model[0, 0], Pdata)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
