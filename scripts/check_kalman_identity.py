"""Compare kalman_nll against an independent recursion built from the
Monte-Carlo-validated augmented system.

If the independent recursion (which is validated against the sampler by
Monte Carlo) disagrees with kalman_nll, the diagnostic recursion has a bug.
If they agree, then the dense joint reference in the unit test is wrong.
"""
import math
import torch

from scripts.debug_kalman_stepwise import build


def independent_nll(F, Q, dH, dS, H_seq, mask, P0_bath):
    """Prediction-error decomposition for z=[H,S] with H observed exactly."""
    B, T, L, _ = H_seq.shape
    N = T - 1
    total = H_seq.new_zeros(()).double()
    for b in range(B):
        for l in range(L):
            if not mask[b, l]:
                continue
            m = torch.zeros(dH + dS, dtype=torch.float64)
            m[:dH] = H_seq[b, 0, l].double()
            P = torch.zeros(dH + dS, dH + dS, dtype=torch.float64)
            P[dH:, dH:] = P0_bath.double()
            for n in range(N):
                # predict
                m = F.double() @ m
                P = F.double() @ P @ F.double().T + Q.double()
                # update with H_{n+1}
                y = H_seq[b, n + 1, l].double()
                S = P[:dH, :dH]
                e = y - m[:dH]
                Lc = torch.linalg.cholesky(S)
                sol = torch.cholesky_solve(e.unsqueeze(-1), Lc)
                quad = float((e.unsqueeze(-1) * sol).sum())
                logdet = float(2.0 * torch.log(torch.diagonal(Lc)).sum())
                total = total + 0.5 * (quad + logdet + dH * math.log(2 * math.pi))
                K = P[:, :dH] @ torch.linalg.inv(S)
                m = m + K @ e
                P = P - K @ P[:dH, :]
                P = 0.5 * (P + P.T)
    return total


def main():
    torch.manual_seed(42)
    dH, dS, L, h, k = 4, 3, 2, 0.5, 0.5
    B, N = 2, 6
    D_v = torch.tensor([0.3, 0.2, 0.4, 0.25])
    G_v = torch.tensor([0.7, 0.5, 0.9])
    C = 0.2 * torch.randn(dH, dS)
    model = build(dH, dS, L, h, k, D_v, G_v, C).eval()

    H_seq = torch.randn(B, N + 1, L, dH)
    mask = torch.ones(B, L, dtype=torch.bool)

    class FixedPosterior(torch.nn.Module):
        def forward(self, H_hist, dt_hist, temperature_k, residue_mask):
            b2 = H_hist.shape[0]
            return (torch.zeros(b2, L, dS),
                    torch.eye(dS).reshape(1, 1, dS, dS).expand(b2, L, dS, dS).clone())

    model.history_posterior = FixedPosterior()
    H_hist = H_seq[:, :1].expand(B, 2, L, dH)
    dt = torch.ones(B, 2) * h
    nll_model = float(model.kalman_nll(H_hist, dt, H_seq, mask, None))

    Dm = torch.diag(D_v)
    CDC = C.T @ Dm @ C
    F = torch.zeros(dH + dS, dH + dS)
    F[:dH, :dH] = torch.eye(dH) - h * k * Dm
    F[:dH, dH:] = -h * Dm @ C
    F[dH:, :dH] = -h * k * C.T @ Dm
    F[dH:, dH:] = torch.eye(dS) - h * torch.diag(G_v) - h * CDC
    Q = torch.zeros(dH + dS, dH + dS)
    Q[:dH, :dH] = 2 * h * Dm
    Q[:dH, dH:] = 2 * h * Dm @ C
    Q[dH:, :dH] = 2 * h * C.T @ Dm
    Q[dH:, dH:] = 2 * h * CDC + torch.diag(2 * h * G_v)
    nll_ind = float(independent_nll(F, Q, dH, dS, H_seq, mask, torch.eye(dS)))
    print(f'kalman_nll      = {nll_model:.6f}')
    print(f'independent     = {nll_ind:.6f}')
    print(f'abs diff        = {abs(nll_model - nll_ind):.3e}')


if __name__ == '__main__':
    main()
