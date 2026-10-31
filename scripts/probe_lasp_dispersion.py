"""Predicted vs true ensemble RMSF per horizon on the frozen SOTA.

No updates.  Tells us whether the forecast ensemble is over- or under-dispersed,
which is the channel that actually sets MD-PCA Wasserstein distances.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics
from evaluation.ensemble_metrics import align_ensemble
from scripts.diagnose_lasp_stability import _panel, _as_device, _jsonable_window


def rmsf(x, mask):
    """x [n, L, 3] -> per-residue sqrt(mean_t |x - mean_t x|^2), averaged."""
    centred = x - x.mean(0, keepdim=True)
    return centred.square().sum(-1).mean(0).clamp_min(1e-8).sqrt().mean()


def _truth_only(a):
    """Future-path spread from the manifest alone; no checkpoint, no GPU."""
    panel = _panel(a.manifest, [348.], tuple(a.horizons), a.max_residues, 4, 2)
    rows = []
    for window in panel:
        mask = window.context['residue_mask'].all(0).bool()
        truth = torch.as_tensor(window.future_path['coords'])[:, :, 1][:, mask].float()
        anchor = torch.as_tensor(window.context['coords'][-1, :, 1])[mask].float()
        rows.append(dict(identity=_jsonable_window(window),
                         truth_rmsf=float(rmsf(truth, mask)),
                         persistence_rmsf=float(rmsf(anchor[None].expand(len(truth), -1, -1), mask)),
                         n_frames=int(truth.shape[0])))
    Path(a.out).write_text(json.dumps(dict(scope='truth-only dispersion', windows=rows,
                                           complete=True), indent=2, allow_nan=False) + '\n')
    for row in rows:
        print(row['identity'].get('horizon'), round(row['truth_rmsf'], 3))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint')
    p.add_argument('--out', required=True)
    p.add_argument('--manifest', default='artifacts/mdcath_manifest_full/val.jsonl')
    p.add_argument('--horizons', type=int, nargs='+', default=[1, 4, 16, 64])
    p.add_argument('--n-samples', type=int, default=12)
    p.add_argument('--max-residues', type=int, default=96)
    p.add_argument('--truth-only', action='store_true',
                   help='report the true future ensemble spread without a model')
    a = p.parse_args()
    torch.set_num_threads(4)
    if a.truth_only:
        return _truth_only(a)
    if not a.checkpoint:
        raise SystemExit('--checkpoint is required unless --truth-only')
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    component = Path(a.out).with_suffix('.component.pt')
    component.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck['component_checkpoint'], component)
    model = CADynamics(str(component), 'cuda', a.checkpoint,
                       history_conditioning=ck['args'].get('history_conditioning', False),
                       noise_mix=ck['args'].get('noise_mix', 0.),
                       noise_length=ck['args'].get('noise_length', 8.),
                       collective_mobility=ck['args'].get('collective_mobility', False)).eval()
    panel = _panel(a.manifest, [348.], tuple(a.horizons), a.max_residues, 1, 1)
    result = dict(scope='no-update dispersion diagnostic', optimizer_steps=0,
                  checkpoint=a.checkpoint, test_opened=False, complete=False, windows=[])
    with sdpa_kernel(SDPBackend.MATH), torch.no_grad():
        for index, window in enumerate(panel):
            coords = _as_device(window.context['coords'], 'cuda')[None]
            mask = _as_device(window.context['residue_mask'], 'cuda').all(0).bool()[None]
            aa, ri, ci = [_as_device(window.context[k], 'cuda').long()[None]
                          for k in ('aatype', 'residue_index', 'chain_id')]
            anchor = coords[0, -1, :, 1]
            ca, _ = model.sample(coords, mask, aa, ri, ci,
                                 torch.tensor([window.temperature_k], device='cuda'),
                                 int(window.horizon), a.n_samples,
                                 torch.Generator(device='cuda').manual_seed(20261030 + index))
            m = mask[0]
            pred = ca[0][:, m].float()
            truth_raw = _as_device(window.future_path['coords'], 'cuda')[:, :, 1][:, m].float()
            # Absolute MD frames carry global rigid motion; remove it before
            # computing spread, otherwise RMSF measures translation, not
            # conformational fluctuation.
            truth = align_ensemble(truth_raw)
            result['windows'].append(dict(
                identity=_jsonable_window(window),
                predicted_rmsf=float(rmsf(pred, m)), truth_rmsf=float(rmsf(truth, m)),
                persistence_rmsf=float(rmsf(anchor[None, m].expand(len(truth), -1, -1).float(), m)),
                ensemble_mean_rms_from_truth_mean=float(
                    (pred.mean(0) - truth.mean(0)).square().sum(-1).mean().sqrt())))
            print(f"window={index+1} h={int(window.horizon)} "
                  f"pred={result['windows'][-1]['predicted_rmsf']:.3f} "
                  f"truth={result['windows'][-1]['truth_rmsf']:.3f}", flush=True)
            Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    result['complete'] = True
    Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
