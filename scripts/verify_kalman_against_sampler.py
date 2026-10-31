"""Decide which marginal is correct by Monte Carlo against the real sampler.

Both the diagnostic recursion and the analytic joint reference claim to describe
``sample_step(collective_mobility=True)``.  Rather than trust either, draw many
paths from the actual sampler with the same linear energy and compare the
empirical mean and covariance of H_{1:N} with each candidate.

Reports, for each candidate, the max relative error of the mean and the
Frobenius relative error of the covariance against the Monte Carlo estimate.
"""
import argparse
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


def joint_reference_stats(D_v, C, G_v, k, h, N):
    """Mean/cov of H_{1:N} under the augmented linear system, dH/dS decoded."""
    dH, dS = D_v.numel(), G_v.numel()
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
    dz = dH + dS
    m0 = torch.zeros(dz)
    P0 = torch.zeros(dz, dz)
    P0[dH:, dH:] = torch.eye(dS)
    Phis = [torch.eye(dz)]
    for _ in range(N):
        Phis.append(F @ Phis[-1])
    means, covs = [], []
    for n in range(1, N + 1):
        means.append(Phis[n] @ m0)
        row = []
        for mm in range(1, N + 1):
            cov = Phis[n] @ P0 @ Phis[mm].T
            for i in range(min(n, mm)):
                cov = cov + Phis[n - 1 - i] @ Q @ Phis[mm - 1 - i].T
            row.append(cov)
        covs.append(row)
    return means, covs, dH, dS, dz


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples', type=int, default=400000)
    p.add_argument('--chunk', type=int, default=20000)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    dH, dS, L, h, k, N = 4, 2, 1, 0.5, 0.5, 4
    D_v = torch.tensor([0.3, 0.2, 0.4, 0.25])
    G_v = torch.tensor([0.7, 0.5, 0.9][:dS])
    C = 0.3 * torch.randn(dH, dS)
    model = build(dH, dS, L, h, k, D_v, G_v, C).eval()
    mask = torch.ones(1, L, dtype=torch.bool)
    seq_feature = torch.zeros(1, L, dH)

    # Monte Carlo through the real sampler; S_0 ~ N(0, I).
    gen = torch.Generator().manual_seed(a.seed)
    acc = torch.zeros(N * dH, dtype=torch.float64)
    acc2 = torch.zeros(N * dH, N * dH, dtype=torch.float64)
    done = 0
    while done < a.samples:
        b = min(a.chunk, a.samples - done)
        H = torch.zeros(b, L, dH)
        S = torch.randn(b, L, dS, generator=gen)
        rows = []
        with torch.no_grad():
            for _ in range(N):
                H, S = model.sample_step(H, S, mask.expand(b, -1), None, seq_feature.expand(b, -1, -1),
                                         generator=gen, collective_mobility=True)
                rows.append(H[:, 0])
        traj = torch.stack(rows, 1).double().reshape(b, -1)      # [b, N*dH]
        acc += traj.sum(0)
        acc2 += traj.T @ traj
        done += b
    mc_mean = acc / a.samples
    mc_cov = acc2 / a.samples - mc_mean.outer(mc_mean)

    means, covs, _, _, dz = joint_reference_stats(D_v, C, G_v, k, h, N)
    ref_mean = torch.cat(means)[[n * dz + j for n in range(N) for j in range(dH)][:0] or
                                [n * dz + j for n in range(N) for j in range(dH)]].double()
    ref_cov = torch.zeros(N * dH, N * dH, dtype=torch.float64)
    idx = [n * dz + j for n in range(N) for j in range(dH)]
    for n in range(N):
        for mm in range(N):
            block = covs[n][mm][:dH, :dH]
            ref_cov[n * dH:(n + 1) * dH, mm * dH:(mm + 1) * dH] = block.double()

    def report(name, mean, cov):
        em = float((mean - mc_mean).abs().max() / mc_mean.abs().max().clamp_min(1e-12))
        ec = float((cov - mc_cov).norm() / mc_cov.norm())
        print(f'{name:28s} mean_rel_err={em:.4e}  cov_rel_frob_err={ec:.4e}')
        return ec

    print(f'monte-carlo samples={a.samples}')
    print('mc_mean', mc_mean[:8].tolist())
    print('mc_diag', mc_cov.diagonal()[:8].tolist())
    ref_mean = torch.cat(means).double()
    ref_mean = ref_mean[idx]
    print('ref_mean', ref_mean[:8].tolist())
    print('ref_diag', ref_cov.diagonal()[:8].tolist())
    report('augmented joint model', ref_mean, ref_cov)


if __name__ == '__main__':
    main()
