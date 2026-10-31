"""Map the stable, physically-meaningful region of bath coupling for the LASP kernel.

The trained SOTA sits where ``C^T M C << Gamma``, i.e. S is driven by its own
thermostat noise and carries no information about H.  Making S a genuine memory
requires ``C^T M C ~ Gamma``, which is ~281x the trained coupling.  A previous
coarse sweep showed x300 diverging at h=64, so this script resolves the boundary
and reports, per (coupling multiplier, gamma multiplier):

  * finiteness and max |CA| (stability),
  * bath force relative to the energy force (authority),
  * trajectory-driven vs thermostat-driven amplitude of the S update,
  * stationary S RMS (should stay O(1) if the FDT is respected).
"""
import argparse
import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics
from scripts.diagnose_lasp_stability import _panel, _as_device, _jsonable_window


def rms(x):
    return float(x.detach().float().square().mean().sqrt())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--manifest', default='artifacts/mdcath_manifest_full/val.jsonl')
    p.add_argument('--horizons', type=int, nargs='+', default=[1, 4, 16, 64])
    p.add_argument('--multipliers', type=float, nargs='+',
                   default=[1., 10., 30., 50., 100., 150., 200., 281.])
    p.add_argument('--gamma-multipliers', type=float, nargs='+', default=[1.])
    p.add_argument('--n-samples', type=int, default=12)
    p.add_argument('--max-residues', type=int, default=96)
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
    kernel = model.kernel
    original_forward = kernel.bath.forward
    original_step = kernel.sample_step
    result = dict(scope='no-update bath rescale stability/authority map; optimizer_steps=0',
                  checkpoint=a.checkpoint, test_opened=False, complete=False, windows=[])

    def instrument(record):
        def wrapped(H, S, residue_mask, temperature_k, seq_feature=None, generator=None,
                    create_graph=False, noise_factor=None, collective_mobility=False):
            feature = H if seq_feature is None else seq_feature
            D, C, Gamma = kernel.bath(feature, temperature_k)
            out = original_step(H, S, residue_mask, temperature_k, seq_feature, generator,
                                create_graph, noise_factor, collective_mobility)
            with torch.no_grad():
                f = kernel.compute_forces(H, residue_mask, temperature_k, create_graph=False)
                bath = torch.einsum('blij,blj->bli', C, S) if S is not None else torch.zeros_like(f)
                disp = out[0] - H
                record.append(dict(
                    force=rms(f), bath_force=rms(bath),
                    s_drive=rms(torch.einsum('blij,bli->blj', C, disp)),
                    s_noise=rms(torch.sqrt(2.0 * kernel.cfg.step_ps * Gamma)),
                    s_rms=rms(S) if S is not None else 0.,
                    h_gamma=rms(kernel.cfg.step_ps * Gamma)))
            return out
        return wrapped

    with sdpa_kernel(SDPBackend.MATH), torch.no_grad():
        for index, window in enumerate(_panel(a.manifest, [348.], tuple(a.horizons),
                                              a.max_residues, 1, 1)):
            horizon = int(window.horizon)
            coords = _as_device(window.context['coords'], 'cuda')[None]
            mask = _as_device(window.context['residue_mask'], 'cuda').all(0).bool()[None]
            aa, ri, ci = [_as_device(window.context[k], 'cuda').long()[None]
                          for k in ('aatype', 'residue_index', 'chain_id')]
            temp = torch.tensor([window.temperature_k], device='cuda')
            seed = 20261030 + index
            base_coupling = float(kernel.bath.coupling_scale)
            base_gamma_scale = float(kernel.bath.gamma_scale)
            base_gamma_floor = float(kernel.bath.gamma_floor)

            def run(multiplier, gmul):
                kernel.bath.coupling_scale = multiplier * base_coupling
                kernel.bath.gamma_scale = gmul * base_gamma_scale
                kernel.bath.gamma_floor = gmul * base_gamma_floor
                ca, _ = model.sample(coords, mask, aa, ri, ci, temp, horizon, a.n_samples,
                                     torch.Generator(device='cuda').manual_seed(seed))
                return ca[0][:, mask[0]].float()

            trace = []
            kernel.sample_step = instrument(trace)
            reference = run(1., 1.)
            kernel.sample_step = original_step
            kernel.bath.coupling_scale = base_coupling
            kernel.bath.gamma_scale = base_gamma_scale
            kernel.bath.gamma_floor = base_gamma_floor

            entry = dict(identity=_jsonable_window(window), reference=dict(
                predicted_spread=float(reference.std(0).square().sum(-1).mean().sqrt())),
                arms={})
            for gmul in a.gamma_multipliers:
                for multiplier in a.multipliers:
                    if multiplier == 1. and gmul == 1.:
                        arm, tr = reference, trace
                    else:
                        tr = []
                        kernel.sample_step = instrument(tr)
                        arm = run(multiplier, gmul)
                        kernel.sample_step = original_step
                        kernel.bath.coupling_scale = base_coupling
                        kernel.bath.gamma_scale = base_gamma_scale
                        kernel.bath.gamma_floor = base_gamma_floor
                    finite = bool(torch.isfinite(arm).all())
                    rec = dict(
                        finite=finite,
                        max_abs=float(torch.nan_to_num(arm.abs(), nan=-1.).max()),
                        ca_rms_from_reference=(float((arm - reference).square().sum(-1).mean().sqrt())
                                               if finite else None),
                        predicted_spread=(float(arm.std(0).square().sum(-1).mean().sqrt())
                                          if finite else None))
                    if tr:
                        rec['force'] = sum(r['force'] for r in tr) / len(tr)
                        rec['bath_force'] = sum(r['bath_force'] for r in tr) / len(tr)
                        rec['s_drive'] = sum(r['s_drive'] for r in tr) / len(tr)
                        rec['s_noise'] = sum(r['s_noise'] for r in tr) / len(tr)
                        rec['s_rms'] = sum(r['s_rms'] for r in tr) / len(tr)
                        rec['h_gamma'] = sum(r['h_gamma'] for r in tr) / len(tr)
                        rec['bath_over_force'] = rec['bath_force'] / max(rec['force'], 1e-12)
                        rec['drive_over_noise'] = rec['s_drive'] / max(rec['s_noise'], 1e-12)
                    entry['arms'][f'c{multiplier:g}_g{gmul:g}'] = rec
            result['windows'].append(entry)
            print(f"--- h={horizon} ---", flush=True)
            for k, v in entry['arms'].items():
                if v['finite']:
                    print(f"  {k:<16} ok   dCA={v['ca_rms_from_reference']:.3f} "
                          f"spread={v['predicted_spread']:.3f} "
                          f"bath/f={v.get('bath_over_force', 0):.4f} "
                          f"drive/noise={v.get('drive_over_noise', 0):.4f} "
                          f"Srms={v.get('s_rms', 0):.2f}", flush=True)
                else:
                    print(f"  {k:<16} DIVERGED max|CA|={v['max_abs']:.3g}", flush=True)
            Path(a.out).write_text(json.dumps(result, indent=2) + '\n')
    result['complete'] = True
    Path(a.out).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
