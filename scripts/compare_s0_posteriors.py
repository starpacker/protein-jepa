"""Compare two candidate posteriors p(S_anchor | H_history) on real anchors.

(A) "simple": S_{k+1} = (1 - h Gamma) S_k + C^T dH_k + eta, i.e. treat the
    realised displacement as a deterministic drive (what exact_bath_posterior
    currently does).  This counts the bath's own noise but IGNORES the fact that
    observing H_{k+1} constrains S_k through the H-noise likelihood.

(B) "kalman": the augmented linear-Gaussian filter over [H, S] with C_obs =
    -h D C.  This is the recursion MC-validated against sample_step in
    scripts/verify_kalman_against_sampler.py and verify_conditional_predictive.py.

They coincide only in the R -> inf limit (h D C / sqrt(2 h D) -> 0).  The gap
tells us whether the cheap recursion is defensible or must be replaced.
"""
import sys
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics
from scripts.diagnose_lasp_stability import _panel, _as_device


def simple_posterior(H, dt, D, C, Gamma, jitter=1e-9):
    B, K, L, d = H.shape
    dS = C.shape[-1]
    m = torch.zeros(B, L, dS, device=H.device, dtype=H.dtype)
    P = torch.ones(B, L, dS, device=H.device, dtype=H.dtype)
    for k in range(K - 1):
        h = dt[:, k].view(B, 1, 1)
        a = 1.0 - h * Gamma
        m = a * m + torch.einsum('blij,bli->blj', C, H[:, k + 1] - H[:, k])
        P = a * a * P + 2.0 * h.squeeze(-1) * Gamma
    return m, torch.diag_embed(P.clamp_min(jitter).sqrt())


def kalman_posterior(H, dt, D, C, Gamma, f_seq, jitter=1e-9):
    """Exact filter.  f_seq[k] is compute_forces(H_k), so the observation is
    z_k = H_{k+1}-H_k - h_k D f_k = -h_k D C S_k + noise."""
    B, K, L, d = H.shape
    dS = C.shape[-1]
    I = torch.eye(dS, device=H.device, dtype=H.dtype).reshape(1, 1, dS, dS)
    m = torch.zeros(B, L, dS, device=H.device, dtype=H.dtype)
    P = I.expand(B, L, dS, dS).clone()
    for k in range(K - 1):
        h = dt[:, k].view(B, 1, 1)
        R = 2.0 * h * D
        B_h = h * D.unsqueeze(-1) * C
        A = 1.0 - h * Gamma
        z = (H[:, k + 1] - H[:, k]) - h * D * f_seq[k]
        e = z + torch.einsum('blij,blj->bli', B_h, m)
        V = torch.diag_embed(R) + B_h @ P @ B_h.transpose(-1, -2)
        V = 0.5 * (V + V.transpose(-1, -2))
        Pbt = P @ B_h.transpose(-1, -2)
        K = -torch.linalg.solve(V, Pbt.transpose(-1, -2)).transpose(-1, -2)
        m_plus = m + torch.einsum('blsk,blk->bls', K, e)
        KC = -torch.einsum('blsk,blkt->blst', K, B_h)
        ImKC = I - KC
        P_plus = ImKC @ P @ ImKC.transpose(-1, -2) \
            + torch.einsum('blsk,blk,bltk->blst', K, R, K)
        P_plus = 0.5 * (P_plus + P_plus.transpose(-1, -2))
        m = A * m_plus + torch.einsum('blij,bli->blj', C, H[:, k + 1] - H[:, k])
        Am = torch.diag_embed(A)
        P = Am @ P_plus @ Am.transpose(-1, -2) + torch.diag_embed(2.0 * h * Gamma)
        P = 0.5 * (P + P.transpose(-1, -2))
    return m, torch.linalg.cholesky(P + jitter * I)


def main():
    torch.set_num_threads(4)
    seed = 42
    ckp = f'results/standard_eval/stopped_best/dynamics_seed{seed}/checkpoint_best.pt'
    ck = torch.load(ckp, map_location='cpu', weights_only=False)
    comp = Path('/tmp/lasp_review_runtime/cmp_s0.pt')
    comp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck['component_checkpoint'], comp)
    for tag, cm, gs, gf in [('trained', 1.0, None, None),
                            ('active_x100_g0.1', 100.0, 0.1, 1.0)]:
        model = CADynamics(str(comp), 'cuda', ckp, history_conditioning=False,
                           noise_mix=ck['args'].get('noise_mix', 0.),
                           noise_length=ck['args'].get('noise_length', 8.),
                           collective_mobility=ck['args'].get('collective_mobility', False)).eval()
        if cm != 1.0:
            model.kernel.cfg.coupling_scale *= cm
            model.kernel.bath.coupling_scale *= cm
        if gs is not None:
            model.kernel.cfg.gamma_scale = model.kernel.cfg.gamma_scale * gs
            model.kernel.bath.gamma_scale = model.kernel.bath.gamma_scale * gs
        print(f'=== {tag} (coupling x{cm}) ===')
        with sdpa_kernel(SDPBackend.MATH), torch.no_grad():
            for w in _panel('artifacts/mdcath_manifest_full/val.jsonl', [348.], (1,), 96, 1, 2):
                co = _as_device(w.context['coords'], 'cuda')[None]
                mk = _as_device(w.context['residue_mask'], 'cuda').all(0).bool()[None]
                aa, ri, ci = [_as_device(w.context[k], 'cuda').long()[None]
                              for k in ('aatype', 'residue_index', 'chain_id')]
                t = torch.tensor([w.temperature_k], device='cuda')
                # encode() takes one backbone frame [B,L,4,3]; stack per frame.
                T = co.shape[1]
                m = mk[0]
                for K in (2, 4, 8):
                    if T < K:
                        continue
                    H = torch.stack([model.bundle.encode(co[:, t], mk, aa, ri, ci)
                                     for t in range(T - K, T)], dim=1)   # [1,K,L,d]
                    dt = torch.full((1, K), float(model.kernel.cfg.step_ps), device='cuda')
                    D, C, G = model.kernel.bath(H[:, -1], t)
                    Dm = D[:, m].double().cpu()
                    Cm = C[:, m].double().cpu()
                    Gm = G[:, m].double().cpu()
                    dtb = dt.double().cpu()
                    Hd = H[:, :, m].double().cpu()
                    m1, L1 = simple_posterior(Hd, dtb, Dm, Cm, Gm)
                    # forces at each history frame, from the model's energy
                    fs = []
                    with torch.no_grad():
                        for kk in range(K):
                            Hk = H[:, kk]
                            fs.append(model.kernel.compute_forces(
                                Hk, mk, t, create_graph=False)[:, m].double().cpu())
                    m2, L2 = kalman_posterior(Hd, dtb, Dm, Cm, Gm, fs)
                    dmean = (m2 - m1).abs().mean() / m2.abs().mean().clamp_min(1e-30)
                    sd1 = L1.diagonal(dim1=-2, dim2=-1)
                    sd2 = L2.diagonal(dim1=-2, dim2=-1)
                    dv = (sd2 - sd1).abs().mean() / sd2.abs().mean().clamp_min(1e-30)
                    # consistency: model method vs this hand recursion (1e-6 target)
                    mm, Lm = model.kernel.exact_bath_posterior(
                        H, dt, t, mk, seq_feature=H[:, -1])
                    mref = mm[:, m].double().cpu()
                    mdiff = (mref - m2).abs().max()
                    Lref = Lm[:, m].double().cpu()
                    Ldiff = (Lref - L2).abs().max()
                    print(f'  K={K}: simple-vs-kalman mean_rel={float(dmean):.3e} '
                          f'sd_rel={float(dv):.3e} | hand-vs-model dm={float(mdiff):.2e} '
                          f'dL={float(Ldiff):.2e}  |m2|={float(m2.abs().mean()):.3e}')


if __name__ == '__main__':
    main()
