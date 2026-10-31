"""Locate the discrepancy between kalman_nll and the exact joint Gaussian.

Builds the linear model used by tests/test_stage_b1b_kalman.py, then compares
step by step: the analytic predictive mean/covariance at each n against the
Kalman filter's own predicted moments, and finally the total NLL against the
dense joint-Gaussian marginal.
"""
import math
import torch
import torch.nn as nn

from protein_jepa_frame.models.stage_b1b.model import StageB1b, StageB1bConfig, HSpaceEnergy


def build(dH, dS, L, h, k, D_v, G_v, C):
    cfg = StageB1bConfig(latent_dim=dH, bath_dim=dS, energy_depth=1, energy_heads=1,
                         energy_hidden=16, step_ps=h, jitter=1e-9)

    class LinearEnergy(HSpaceEnergy):
        def forward(self, H, residue_mask, temperature_k=None):
            return 0.5 * k * (H.pow(2).sum(-1) * residue_mask.float()).sum()

    class ConstBath(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = type("c", (), {"use_bath": True})()
            self.register_buffer("D_val", D_v)
            self.register_buffer("Gamma_val", G_v)
            self.register_buffer("C_val", C)

        def forward(self, seq_feature, temperature_k):
            b, l, _ = seq_feature.shape
            return (self.D_val.reshape(1, 1, dH).expand(b, l, dH),
                    self.C_val.reshape(1, 1, dH, dS).expand(b, l, dH, dS),
                    self.Gamma_val.reshape(1, 1, dS).expand(b, l, dS))

    model = StageB1b(cfg)
    model.energy = LinearEnergy(cfg)
    model.bath = ConstBath()
    return model


def main():
    torch.manual_seed(42)
    dH, dS, L, h, k, N = 4, 3, 1, 0.5, 0.5, 4
    D_v = torch.tensor([0.3, 0.2, 0.4, 0.25])
    G_v = torch.tensor([0.7, 0.5, 0.9])
    C = 0.2 * torch.randn(dH, dS)
    model = build(dH, dS, L, h, k, D_v, G_v, C).eval()

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

    # Analytic marginal of H_n (from z_0 = [H_0=0, S_0~N(0,I)]).
    P0z = torch.zeros(dH + dS, dH + dS)
    P0z[dH:, dH:] = torch.eye(dS)
    Phis = [torch.eye(dH + dS)]
    for _ in range(N):
        Phis.append(F @ Phis[-1])
    print('n  analytic_var(H)                          kalman_V(diag)')
    for n in range(1, N + 1):
        cov = Phis[n] @ P0z @ Phis[n].T
        for i in range(n):
            cov = cov + Phis[n - 1 - i] @ Q @ Phis[n - 1 - i].T
        print(f'{n}  {cov.diagonal()[:dH].tolist()}')
    print('F=\n', F)
    print('Q=\n', Q)


if __name__ == '__main__':
    main()
