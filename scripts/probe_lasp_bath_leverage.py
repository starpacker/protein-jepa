"""How much can the bath initial state move a frozen-SOTA forecast?

Two quantities, both measured without any parameter update:

* the size of the bath force ``C S`` relative to the energy force ``f``;
* the CA-space displacement caused by shifting the bath mean by a known
  number of prior standard deviations, with identical RNG for every arm.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics
from scripts.diagnose_lasp_stability import _panel, _as_device, _jsonable_window


class ConstantPosterior(torch.nn.Module):
    """S_0 = mean + eps, eps ~ N(0, I); exposes only the bath mean to the test."""

    def __init__(self, bath_dim, mean=None, scale=0.):
        super().__init__()
        self.bath_dim = bath_dim
        self.mean = mean
        self.scale = scale

    def forward(self, H_hist, dt_hist, temperature_k, residue_mask):
        b, _, length, _ = H_hist.shape
        if self.mean is None:
            m0 = torch.zeros(b, length, self.bath_dim, device=H_hist.device,
                             dtype=H_hist.dtype) + self.scale
        else:
            m0 = self.mean.to(H_hist.dtype).expand(b, -1, -1) * self.scale
        chol = torch.eye(self.bath_dim, device=H_hist.device,
                         dtype=H_hist.dtype).expand(b, length, -1, -1)
        return m0, chol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--manifest', default='artifacts/mdcath_manifest_full/val.jsonl')
    p.add_argument('--horizons', type=int, nargs='+', default=[1, 4, 16, 64])
    p.add_argument('--scales', type=float, nargs='+', default=[.5, 1., 2., 4.])
    p.add_argument('--n-samples', type=int, default=12)
    p.add_argument('--max-residues', type=int, default=96)
    p.add_argument('--coupling-multiplier', type=float, nargs='+', default=[1.])
    a = p.parse_args()
    torch.set_num_threads(4)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    component = Path(a.out).with_suffix('.component.pt')
    component.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck['component_checkpoint'], component)
    model = CADynamics(str(component), 'cuda', a.checkpoint, history_conditioning=True,
                       noise_mix=ck['args'].get('noise_mix', 0.),
                       noise_length=ck['args'].get('noise_length', 8.),
                       collective_mobility=ck['args'].get('collective_mobility', False)).eval()
    cfg = model.kernel.cfg
    panel = _panel(a.manifest, [348.], tuple(a.horizons), a.max_residues, 1, 1)
    result = dict(scope='no-update leverage diagnostic; optimizer_steps=0',
                  checkpoint=a.checkpoint, test_opened=False, complete=False, windows=[])
    with sdpa_kernel(SDPBackend.MATH), torch.no_grad():
        for index, window in enumerate(panel):
            coords = _as_device(window.context['coords'], 'cuda')[None]
            mask = _as_device(window.context['residue_mask'], 'cuda').all(0).bool()[None]
            aa, ri, ci = [_as_device(window.context[k], 'cuda').long()[None]
                          for k in ('aatype', 'residue_index', 'chain_id')]
            temp = torch.tensor([window.temperature_k], device='cuda')
            anchor = model.bundle.encode(coords[:, -1], mask, aa, ri, ci)
            D, C, Gamma = model.kernel.bath(anchor, temp)
            force = model.kernel.compute_forces(anchor, mask, temp, create_graph=False)
            # Typical size of the bath force C S for S ~ N(0, I) (the prior).
            bath_norm = (C * torch.randn_like(C)).norm(dim=(-1, -2))
            force_norm = force.norm(dim=-1).clamp_min(1e-12)
            entry = dict(identity=_jsonable_window(window),
                         coupling_norm=float(C.norm(dim=-1).mean()),
                         force_norm=float(force_norm.mean()),
                         bath_force_over_force=float((bath_norm / force_norm).mean()),
                         bath_force_over_force_max=float((bath_norm / force_norm).max()),
                         scales={})
            horizon = int(window.horizon)
            # Worst-case bath direction: the leading right singular vector of C,
            # i.e. the unit S_0 that produces the largest bath force.
            top = torch.linalg.svd(C[0], full_matrices=False)[2][:, 0, :]   # [L, dS]
            arms = {}
            for scale in [0.] + list(a.scales):
                model.kernel.history_posterior = ConstantPosterior(cfg.bath_dim, scale=scale)
                ca, _ = model.sample(coords, mask, aa, ri, ci, temp, horizon, a.n_samples,
                                     torch.Generator(device='cuda').manual_seed(20261030 + index))
                arms[scale] = ca[0][:, mask[0]]
                model.kernel.history_posterior = ConstantPosterior(cfg.bath_dim, mean=top,
                                                                   scale=scale)
                ca, _ = model.sample(coords, mask, aa, ri, ci, temp, horizon, a.n_samples,
                                     torch.Generator(device='cuda').manual_seed(20261030 + index))
                arms[f'top{scale}'] = ca[0][:, mask[0]]
            reference = arms[0.].float()
            entry['scales'][str(horizon)] = {
                str(name): dict(
                    ca_rms_from_prior=float((arms[name].float() - reference)
                                            .square().sum(-1).mean().sqrt()),
                    predicted_rmsf=float(reference.std(0).square().sum(-1).mean().sqrt()))
                for name in arms}
            print(f'window={index+1} horizon={horizon} ' + json.dumps(
                {k: round(v['ca_rms_from_prior'], 4)
                 for k, v in entry['scales'][str(horizon)].items()}), flush=True)
            result['windows'].append(entry)
            Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    result['complete'] = True
    Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
