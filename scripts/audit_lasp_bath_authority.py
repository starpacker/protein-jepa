"""Audit whether the bath state S has real authority over the prediction.

Three independent measurements on a frozen checkpoint, no parameter updates:

1. per-step displacement decomposition: energy-force drift, bath-force drift
   and Brownian noise, measured by wrapping the exact sampler primitive;
2. counterfactual rollouts: zero the coupling, or amplify it by k, and measure
   how far the CA forecast moves and whether the trajectory stays bounded;
3. training-pressure check: gradient norm per parameter group on the real
   objective, so we can say whether C could ever have grown.
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
                   default=[0., 1., 3., 10., 30., 100., 300.])
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
    cfg = kernel.cfg
    result = dict(scope='no-update bath authority audit; optimizer_steps=0',
                  checkpoint=a.checkpoint, test_opened=False, complete=False,
                  config={k: float(getattr(cfg, k)) for k in
                          ('coupling_scale', 'c_init_std', 'gamma_scale',
                           'gamma_floor', 'diffusion_scale', 'step_ps')},
                  windows=[])

    original_step = kernel.sample_step

    def instrument(record):
        def wrapped(H, S, residue_mask, temperature_k, seq_feature=None, generator=None,
                    create_graph=False, noise_factor=None, collective_mobility=False):
            feature = H if seq_feature is None else seq_feature
            D, C, Gamma = kernel.bath(feature, temperature_k)
            out = original_step(H, S, residue_mask, temperature_k, seq_feature, generator,
                                create_graph, noise_factor, collective_mobility)
            with torch.no_grad():
                f = kernel.compute_forces(H, residue_mask, temperature_k, create_graph=False)
                root_D = D.sqrt()

                def mix(x):
                    if noise_factor is None:
                        return x
                    x = torch.einsum('bij,bjd->bid', noise_factor.transpose(-1, -2), x)
                    return torch.einsum('bij,bjd->bid', noise_factor, x)

                bath = torch.einsum('blij,blj->bli', C, S) if S is not None else torch.zeros_like(f)
                # How much of S is driven by the trajectory versus its own
                # thermostat noise?  The two terms of S_{t+1} are C^T dH and
                # sqrt(2 h Gamma) eps_S.
                record.append(dict(
                    energy_drift=rms(kernel.cfg.step_ps * root_D * mix(root_D * f)),
                    bath_drift=rms(kernel.cfg.step_ps * root_D * mix(-root_D * bath)),
                    total_displacement=rms(out[0] - H),
                    force=rms(f), bath_force=rms(bath),
                    h_gamma=rms(kernel.cfg.step_ps * Gamma),
                    s_drive=rms(torch.einsum('blij,bli->blj', C, out[0] - H)),
                    s_noise=rms(torch.sqrt(2.0 * kernel.cfg.step_ps * Gamma)),
                    s_rms=rms(S) if S is not None else 0.))
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

            base_coupling = float(cfg.coupling_scale)

            def run(multiplier):
                kernel.bath.coupling_scale = float(multiplier) * base_coupling
                ca, _ = model.sample(coords, mask, aa, ri, ci, temp, horizon, a.n_samples,
                                     torch.Generator(device='cuda').manual_seed(seed))
                return ca[0][:, mask[0]].float()

            trace = []
            kernel.sample_step = instrument(trace)
            reference = run(1.)
            kernel.sample_step = original_step
            kernel.bath.coupling_scale = base_coupling

            entry = dict(identity=_jsonable_window(window),
                         decomposition={k: sum(r[k] for r in trace) / len(trace) for k in trace[0]},
                         s_rms_end=trace[-1]['s_rms'], s_rms_max=max(r['s_rms'] for r in trace),
                         predicted_spread=float(reference.std(0).square().sum(-1).mean().sqrt()),
                         arms={})
            for multiplier in a.multipliers:
                arm = run(multiplier) if multiplier != 1. else reference
                finite = bool(torch.isfinite(arm).all())
                entry['arms'][str(multiplier)] = dict(
                    ca_rms_from_reference=(float((arm - reference).square().sum(-1).mean().sqrt())
                                           if finite else None),
                    predicted_spread=(float(arm.std(0).square().sum(-1).mean().sqrt())
                                      if finite else None),
                    max_abs=float(torch.nan_to_num(arm.abs(), nan=-1.).max()),
                    finite=finite)
            # Restore the trained coupling before the gradient probe.
            kernel.bath.coupling_scale = base_coupling
            entry['decomposition']['bath_over_energy_force'] = (
                entry['decomposition']['bath_force'] /
                max(entry['decomposition']['force'], 1e-12))
            entry['decomposition']['bath_drift_over_total'] = (
                entry['decomposition']['bath_drift'] /
                max(entry['decomposition']['total_displacement'], 1e-12))
            result['windows'].append(entry)
            print(f"h={horizon:>2} energy={entry['decomposition']['energy_drift']:.3e} "
                  f"bath={entry['decomposition']['bath_drift']:.3e} "
                  f"total={entry['decomposition']['total_displacement']:.3e} "
                  f"bath/total={entry['decomposition']['bath_drift_over_total']:.4f} "
                  + ' '.join(f"x{k}->{v['ca_rms_from_reference']:.3f}" if v['finite']
                             else f"x{k}->NaN" for k, v in entry['arms'].items()), flush=True)
            Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')

    # Training pressure: which parameter groups actually receive gradient from
    # the objective that trained this checkpoint?
    from types import SimpleNamespace
    from protein_jepa_frame.training.train_lasp_ca_dynamics import make_loader
    from protein_jepa_frame.training.lasp_run_state import step_batch
    from protein_jepa_frame.training.train_utils import _move_batch
    model.train(); model.bundle.eval()
    args = SimpleNamespace(n_samples=a.n_samples, grad_steps=64, contact_objective='brier',
                           latent_weight=.1, contact_weight=20., rmsf_weight=1.,
                           distance_weight=0., clash_weight=40., mode_weight=2.,
                           mode_distribution_weight=0., target='path_mixture',
                           train_windows=0, val_windows=1, seed=42, stateless_data=True,
                           max_residues=a.max_residues)
    loader = make_loader(args, a.manifest, 64, False)
    batch = _move_batch(step_batch(loader, 1, 42), 'cuda')
    generator = torch.Generator(device='cuda').manual_seed(7)
    with sdpa_kernel(SDPBackend.MATH):
        loss = model.losses(batch, 64, args, generator)['loss']
        model.kernel.zero_grad(set_to_none=True)
        loss.backward()
    groups = ('bath.c_net', 'bath.gamma_net', 'bath.d_net', 'energy.',
              'history_posterior.')
    def group_norm(predicate):
        terms = [x.grad.detach().square().sum() for n, x in kernel.named_parameters()
                 if predicate(n) and x.grad is not None]
        return float(torch.stack(terms).sum().sqrt()) if terms else 0.

    norms = {g: group_norm(lambda n, g=g: n.startswith(g)) for g in groups}
    total = group_norm(lambda n: True)
    def finite(x):
        x = float(x)
        return x if x == x and abs(x) != float('inf') else None

    result['gradient'] = dict(loss=finite(loss.detach()), total=finite(total),
                              groups={g: finite(v) for g, v in norms.items()},
                              share={g: finite(v / max(total, 1e-30)) for g, v in norms.items()})
    print('gradient shares', {g: round(v / max(total, 1e-30), 5)
                              for g, v in norms.items()}, flush=True)
    result['complete'] = True
    Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
