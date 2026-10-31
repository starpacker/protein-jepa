"""Measure the trained bath's calibration constants against the kernel's stated design rule.

The kernel comment claims C is initialised so that ``B_h P B_h^T`` is the same
order as ``R = 2 h D`` (B_h = h C).  This script evaluates that claim on real
anchors from the frozen checkpoint instead of assuming the init formula, because
``c_net`` is a nonlinear map whose output scale depends on the anchor feature.

Reported per residue:
  R        = 2 h D                      [dH]
  bath_var = diag(B_h P B_h^T),  P = I  [dH]
  ratio    = bath_var / R
and the bath's own relaxation time 1/Gamma against the observation interval h.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics
from scripts.diagnose_lasp_stability import _panel, _as_device, _jsonable_window


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--manifest', default='artifacts/mdcath_manifest_full/val.jsonl')
    p.add_argument('--max-residues', type=int, default=96)
    p.add_argument('--windows', type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(4)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    component = Path(a.out).with_suffix('.component.pt')
    component.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck['component_checkpoint'], component)
    model = CADynamics(str(component), 'cuda', a.checkpoint,
                       history_conditioning=False,
                       noise_mix=ck['args'].get('noise_mix', 0.),
                       noise_length=ck['args'].get('noise_length', 8.),
                       collective_mobility=ck['args'].get('collective_mobility', False)).eval()
    cfg = model.kernel.cfg
    h = float(cfg.step_ps)
    rows = []
    with sdpa_kernel(SDPBackend.MATH), torch.no_grad():
        for index, window in enumerate(_panel(a.manifest, [348.], (1,), a.max_residues,
                                              1, a.windows)):
            coords = _as_device(window.context['coords'], 'cuda')[None]
            mask = _as_device(window.context['residue_mask'], 'cuda').all(0).bool()[None]
            aa, ri, ci = [_as_device(window.context[k], 'cuda').long()[None]
                          for k in ('aatype', 'residue_index', 'chain_id')]
            temp = torch.tensor([window.temperature_k], device='cuda')
            anchor = model.bundle.encode(coords[:, -1], mask, aa, ri, ci)
            D, C, Gamma = model.kernel.bath(anchor, temp)               # [1,L,*]
            m = mask[0]
            D_m, C_m, G_m = D[0][m], C[0][m], Gamma[0][m]
            # B_h is the mean effect of S on H.  The sampler moves H by
            #   h*D*(f - C S), so its S -> H map is -h*D*C and the magnitude is
            #   B_h = h*D*C.  The old h*C here omitted the mobility and made
            # every ratio wrong by 1/D (~5e4).  Signs do not matter for the
            # variance B_h P B_h^T.
            B_h = h * D_m.unsqueeze(-1) * C_m                          # [L,dH,dS]
            # B_h P B_h^T with P = I (the prior the sampler actually starts from).
            bath_var = torch.einsum('lij,lij->li', B_h, B_h)           # [L,dH]
            R = 2.0 * h * D_m                                          # [L,dH]
            rows.append(dict(
                identity=_jsonable_window(window),
                n_residues=int(m.sum()),
                R_mean=float(R.mean()), R_median=float(R.median()),
                bath_var_mean=float(bath_var.mean()),
                bath_var_median=float(bath_var.median()),
                ratio_bath_over_R_mean=float((bath_var / R.clamp_min(1e-30)).mean()),
                ratio_bath_over_R_median=float((bath_var / R.clamp_min(1e-30)).median()),
                C_fro_per_residue=float(C_m.norm(dim=(-1, -2)).mean()),
                C_std=float(C_m.std()),
                D_mean=float(D_m.mean()),
                h_gamma_mean=float((h * G_m).mean()),
                gamma_mean=float(G_m.mean()),
                memory_time_ps=float((1.0 / G_m.clamp_min(1e-30)).mean()),
                memory_time_over_h=float((1.0 / (h * G_m.clamp_min(1e-30))).mean()),
            ))
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                              for k, v in rows[-1].items() if k != 'identity'}), flush=True)

    def avg(key):
        return sum(r[key] for r in rows) / len(rows)

    summary = {k: avg(k) for k in rows[0] if isinstance(rows[0][k], float)}
    out = dict(scope='no-update bath calibration audit; optimizer_steps=0',
               checkpoint=a.checkpoint, test_opened=False,
               config=dict(coupling_scale=float(cfg.coupling_scale),
                           c_init_std=float(getattr(cfg, 'c_init_std', float('nan'))),
                           diffusion_scale=float(cfg.diffusion_scale),
                           gamma_scale=float(cfg.gamma_scale),
                           gamma_floor=float(cfg.gamma_floor),
                           step_ps=h, bath_dim=int(cfg.bath_dim)),
               summary=summary, windows=rows, complete=True)
    Path(a.out).write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print('SUMMARY', json.dumps({k: round(v, 6) for k, v in summary.items()}, indent=2))


if __name__ == '__main__':
    main()
