"""Verify kalman_nll's conditional predictive distribution against the sampler.

The Kalman recursion scores p(H_{n+1} | H_{0:n}) for the deployed kernel.  The
matching Monte-Carlo check conditions on the SAME observed path and the SAME
S posterior: draw many S_n ~ N(m_n, P_n) from the filter's own posterior, step
the real sampler once, and compare the empirical mean/covariance of H_{n+1}
with the recursion's mu and V.

Agreement here means the diagnostic and the sampler describe one process.
Marginal (unconditional) variance grows over time and is NOT the right object.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.debug_kalman_stepwise import build


def main():
    torch.manual_seed(42)
    dH, dS, L, h, k, N = 4, 3, 1, 0.5, 0.5, 4
    D_v = torch.tensor([0.3, 0.2, 0.4, 0.25])
    G_v = torch.tensor([0.7, 0.5, 0.9])
    C = 0.2 * torch.randn(dH, dS)
    model = build(dH, dS, L, h, k, D_v, G_v, C).eval()
    sf = torch.zeros(1, L, dH)
    mask = torch.ones(1, L, dtype=torch.bool)

    D, Cv, G = model.bath(sf, None)
    R = 2 * h * D
    B_h = h * D.unsqueeze(-1) * Cv
    A = 1.0 - h * G
    IdS = torch.eye(dS)

    # One representative observed path from the sampler (S0 ~ prior).
    gen = torch.Generator().manual_seed(99)
    H, S = torch.zeros(1, L, dH), torch.zeros(1, L, dS)
    path, spath = [H.clone()], [S.clone()]
    with torch.no_grad():
        for _ in range(N):
            H, S = model.sample_step(H, S, mask, None, sf, generator=gen,
                                     collective_mobility=True)
            path.append(H.clone())
            spath.append(S.clone())

    # Filter recursion, tracking m_n and P_n, and comparing each predictive.
    m = torch.zeros(1, L, dS)
    P = IdS.expand(1, L, dS, dS).clone()
    nMC = 200000
    gen_mc = torch.Generator().manual_seed(1234)
    print(f'{"n":>2} {"V_model":>9} {"V_mc":>9} {"rel":>8}   {"mu_model":>9} {"mu_mc":>9}')
    for n in range(N):
        Hn, Hn1 = path[n], path[n + 1]
        f = model.compute_forces(Hn, mask, None, create_graph=False)
        mu_model = Hn + h * D * f - torch.einsum('blij,blj->bli', B_h, m)
        V_model = torch.diag_embed(R) + B_h @ P @ B_h.transpose(-1, -2)

        # MC: S_n ~ N(m, P), step the real sampler once, compare H_{n+1}.
        Lc = torch.linalg.cholesky(P)
        eps = torch.randn(nMC, L, dS, generator=gen_mc)
        # Lc is [B, L, dS, dS]; take batch element 0 -> [L, dS, dS].
        # Sn becomes [nMC, L, dS], m is [1, L, dS] so it broadcasts.
        Sn = m + torch.einsum('lij,nlj->nli', Lc[0], eps)
        gen_step = torch.Generator().manual_seed(5000 + n)
        with torch.no_grad():
            Hnext, _ = model.sample_step(
                Hn.expand(nMC, -1, -1), Sn.expand(nMC, -1, -1),
                mask.expand(nMC, -1), None, sf.expand(nMC, -1, -1),
                generator=gen_step, collective_mobility=True)
        X = Hnext[:, 0].double()
        mu_mc = X.mean(0)
        V_mc = X.var(0)
        rel = (V_mc - V_model[0, 0].diagonal().double()).abs() / V_mc
        print(f'{n + 1:>2} {V_model[0,0].diagonal().mean():9.4f} {V_mc.mean():9.4f} '
              f'{float(rel.mean()):8.4f}   {float(mu_model[0,0].mean()):9.4f} {float(mu_mc.mean()):9.4f}')

        # Advance the filter with the observed transition.
        e = Hn1 - mu_model
        K = P @ B_h.transpose(-1, -2) @ torch.linalg.inv(V_model)
        m = m + torch.einsum('blij,blj->bli', K, e)
        IKB = IdS - K @ B_h
        P = IKB @ P @ IKB.transpose(-1, -2) + K @ torch.diag_embed(R) @ K.transpose(-1, -2)
        P = 0.5 * (P + P.transpose(-1, -2))
        P = A.unsqueeze(-1) * P * A.unsqueeze(-1) + torch.diag_embed(2 * h * G)
        P = 0.5 * (P + P.transpose(-1, -2))


if __name__ == '__main__':
    main()
