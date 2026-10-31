"""Measure whether the training objective depends on the history-initialised S_0.

No updates are performed.  The probe answers one question before committing
GPU hours to a warm-start fine-tune: does the frozen SOTA loss change when the
bath initial state is chosen by the history posterior instead of the N(0, I)
prior?  If the objective is insensitive, no amount of training can use the
extra degree of freedom.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics, make_loader
from protein_jepa_frame.training.lasp_run_state import step_batch
from protein_jepa_frame.training.train_utils import _move_batch

TRAIN_ARGS = dict(n_samples=12, grad_steps=64, contact_objective='brier',
                  latent_weight=.1, contact_weight=20., rmsf_weight=1.,
                  distance_weight=0., clash_weight=40., mode_weight=2.,
                  mode_distribution_weight=0., target='path_mixture')


def _args(horizon):
    return SimpleNamespace(horizon=horizon, **TRAIN_ARGS)


class _SharedPosterior(torch.nn.Module):
    """Stand-in posterior with one learned mean shared by every sample.

    This is the ceiling of what a history-conditioned posterior can do for a
    single window: the history is identical for all samples, so a posterior can
    only move the mean and the scale, not individual draws.
    """

    def __init__(self, mean, bath_dim):
        super().__init__()
        # Register the caller's Parameter object itself; wrapping it again would
        # leave the optimiser holding a different leaf with no gradient.
        assert isinstance(mean, torch.nn.Parameter)
        self.mean = mean
        self.bath_dim = bath_dim

    def forward(self, H_hist, dt_hist, temperature_k, residue_mask):
        m0 = self.mean.expand(H_hist.shape[0], -1, -1)
        chol = torch.eye(self.bath_dim, device=H_hist.device,
                         dtype=H_hist.dtype).expand(H_hist.shape[0], H_hist.shape[2], -1, -1)
        return m0, chol


def _loss(model, batch, horizon, seed):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    return model.losses(_move_batch(batch, 'cuda'), horizon, _args(horizon), generator)['loss']


def _oracle_mean(model, loader, horizon, a, steps=20, lr=.1):
    """Upper bound: optimise a per-residue S_0 mean on one window, fixed noise.

    This reads the window's targets, so it is an oracle.  It answers only
    whether the S_0 degree of freedom has any leverage at all on the objective.
    """
    batch = _move_batch(step_batch(loader, 1, a.seed), 'cuda')
    gen_seed = a.seed * 7 + horizon
    length = batch['coords'].shape[2]
    # nn.Parameter(tensor) does not alias an existing leaf, so build the
    # parameter and hand that same object to the optimiser.
    mean = torch.nn.Parameter(torch.zeros(1, length, model.kernel.cfg.bath_dim, device='cuda'))
    original = model.kernel.history_posterior
    probe = _SharedPosterior(mean, model.kernel.cfg.bath_dim)
    model.kernel.history_posterior = probe
    model.history_conditioning = True
    model.train(); model.bundle.eval()
    with sdpa_kernel(SDPBackend.MATH):
        start = float(_loss(model, batch, horizon, gen_seed).detach())
        opt = torch.optim.Adam([mean], lr=lr)
        best = start
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            value = _loss(model, batch, horizon, gen_seed)
            value.backward()
            opt.step()
            best = min(best, float(value.detach()))
    model.kernel.history_posterior = original
    return dict(loss_with_prior=start, best_loss_with_optimised_mean=best,
                relative_change=(best - start) / start,
                mean_abs=float(mean.detach().abs().mean()), steps=steps)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--val-manifest', default='artifacts/mdcath_manifest_full/val.jsonl')
    p.add_argument('--horizons', type=int, nargs='+', default=[1, 4, 16, 64])
    p.add_argument('--windows', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-samples', type=int, default=12)
    p.add_argument('--grad-steps', type=int, default=64)
    p.add_argument('--max-residues', type=int, default=96)
    a = p.parse_args()
    torch.set_num_threads(4)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    record = ck['args']
    recipe = SimpleNamespace(train_windows=0, val_windows=a.windows, seed=a.seed,
                             stateless_data=True, max_residues=a.max_residues)
    component = Path(a.out).with_suffix('.component.pt')
    component.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck['component_checkpoint'], component)
    TRAIN_ARGS.update(n_samples=a.n_samples, grad_steps=a.grad_steps)
    model = CADynamics(str(component), 'cuda', a.checkpoint,
                       history_conditioning=True,
                       noise_mix=record.get('noise_mix', 0.),
                       noise_length=record.get('noise_length', 8.),
                       collective_mobility=record.get('collective_mobility', False)).to('cuda')
    posterior = model.kernel.history_posterior
    head = posterior.head
    out = dict(scope='no-update sensitivity of the frozen SOTA objective to the S_0 prior',
               checkpoint=a.checkpoint, checkpoint_sha256=ck.get('checkpoint_sha256'),
               optimizer_steps=0, test_opened=False, complete=False, horizons={})

    def perturb(mean_offset=0., chol_diag_offset=0.):
        with torch.no_grad():
            head.bias.zero_(); head.weight.zero_()
            head.bias[:posterior.cfg.bath_dim] = mean_offset
            n_tril = posterior.cfg.bath_dim * (posterior.cfg.bath_dim + 1) // 2
            tril = head.bias[posterior.cfg.bath_dim:]
            i, j = posterior._tril_idx
            tril[(i == j)] = chol_diag_offset

    with sdpa_kernel(SDPBackend.MATH):
        for horizon in a.horizons:
            loader = make_loader(recipe, a.val_manifest, horizon, False)
            entry = dict(conditions={}, gradient={})
            for index in range(a.windows):
                batch = step_batch(loader, index + 1, a.seed)
                model.train(); model.bundle.eval()
                seeds = (a.seed * 7 + horizon, index)
                gen_seed = seeds[0] * 1000 + seeds[1]
                model.history_conditioning = False
                base = _loss(model, batch, horizon, gen_seed)
                perturb(0., 0.)
                model.history_conditioning = True
                identity = _loss(model, batch, horizon, gen_seed)
                perturb(.5, 0.)
                mean_shift = _loss(model, batch, horizon, gen_seed)
                perturb(0., .5)
                chol_shift = _loss(model, batch, horizon, gen_seed)
                perturb(0., 0.)
                model.kernel.zero_grad(set_to_none=True)
                value = _loss(model, batch, horizon, gen_seed)
                value.backward()
                group = lambda prefix: float(torch.sqrt(sum(
                    (x.grad.detach() ** 2).sum() for n, x in model.kernel.named_parameters()
                    if n.startswith(prefix) and x.grad is not None)))
                entry['conditions'][str(index)] = dict(
                    base=float(base), history_prior_identity=float(identity),
                    mean_offset_half=float(mean_shift), chol_diag_offset_half=float(chol_shift),
                    identity_max_abs_diff=float((identity - base).abs()),
                    mean_relative_change=float((mean_shift - base) / base),
                    chol_relative_change=float((chol_shift - base) / base))
                entry['gradient'][str(index)] = {
                    'posterior': group('history_posterior.'),
                    'energy': group('energy.'), 'bath': group('bath.'),
                    'total': group('')}
            rows = entry['conditions'].values()
            entry['oracle_mean'] = _oracle_mean(model, loader, horizon, a)
            entry['summary'] = {
                'mean_identity_max_abs_diff': max(r['identity_max_abs_diff'] for r in rows),
                'mean_relative_change': sum(r['mean_relative_change'] for r in rows) / len(rows),
                'chol_relative_change': sum(r['chol_relative_change'] for r in rows) / len(rows),
                'posterior_grad_over_total': max(
                    entry['gradient'][k]['posterior'] / max(entry['gradient'][k]['total'], 1e-30)
                    for k in entry['gradient'])}
            out['horizons'][str(horizon)] = entry
            print(f'horizon={horizon} {entry["summary"]}', flush=True)
            Path(a.out).write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    out['complete'] = True
    Path(a.out).write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
