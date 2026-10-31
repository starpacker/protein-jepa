"""No-update history intervention on frozen SOTA, never a learned-history claim."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics
from scripts.diagnose_lasp_stability import _panel, _jsonable_window, _as_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--manifest', default='artifacts/mdcath_manifest_full/val.jsonl')
    p.add_argument('--device', default='cuda')
    p.add_argument('--n-samples', type=int, default=12)
    a = p.parse_args()
    torch.set_num_threads(2)
    device = torch.device(a.device)
    panel = _panel(a.manifest, [348., 450.], [1, 64], 96, 2, 2)
    output = Path(a.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(scope='no-update history enablement diagnostic; not trained history conditioning',
                  optimizer_steps=0, test_opened=False, complete=False, checkpoints=[])
    for checkpoint in a.checkpoint:
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        component = output.parent / f"history_seed{ck['args']['seed']}.component.pt"
        torch.save(ck['component_checkpoint'], component)
        model = CADynamics(str(component), device, checkpoint,
            history_conditioning=False, noise_mix=ck['args'].get('noise_mix', 0.),
            noise_length=ck['args'].get('noise_length', 8.),
            collective_mobility=ck['args'].get('collective_mobility', False)).eval()
        posterior = model.kernel.history_posterior
        entry = dict(checkpoint=checkpoint, training_seed=ck['args']['seed'],
            checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            posterior_head_max_abs=float(posterior.head.weight.abs().max()),
            posterior_head_bias_max_abs=float(posterior.head.bias.abs().max()), windows=[])
        for index, window in enumerate(panel):
            coords = _as_device(window.context['coords'], device)[None]
            mask = _as_device(window.context['residue_mask'], device).all(0).bool()[None]
            aa, ri, ci = [_as_device(window.context[k], device).long()[None]
                          for k in ('aatype', 'residue_index', 'chain_id')]
            temp = torch.tensor([window.temperature_k], device=device)
            common = (mask, aa, ri, ci, temp, window.horizon, a.n_samples)
            with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                model.history_conditioning = False
                prior, _ = model.sample(coords, *common,
                    generator=torch.Generator(device=device).manual_seed(20261024+index))
                model.history_conditioning = True
                history, _ = model.sample(coords, *common,
                    generator=torch.Generator(device=device).manual_seed(20261024+index))
                changed = coords.clone()
                # Deliberately replace ONLY earlier observed frames, retaining the anchor.
                changed[:, :-1] = coords[:, -1:]
                altered, _ = model.sample(changed, *common,
                    generator=torch.Generator(device=device).manual_seed(20261024+index))
                frames = coords[:, -4:]
                encoded = model.bundle.encode(frames.reshape(-1, *frames.shape[2:]),
                    mask.expand(4, -1), aa.expand(4, -1), ri.expand(4, -1), ci.expand(4, -1))
                encoded = encoded.reshape(1, 4, mask.shape[-1], -1)
                mean, chol = posterior(encoded, torch.full((1, 4), model.kernel.cfg.step_ps,
                                                           device=device), temp, mask)
                identity = torch.eye(model.kernel.cfg.bath_dim, device=device)
                entry['windows'].append(dict(identity=_jsonable_window(window),
                    prior_vs_enabled_CA_max_abs=float((prior-history).abs().max()),
                    original_vs_altered_history_CA_max_abs=float((history-altered).abs().max()),
                    posterior_mean_max_abs=float(mean.abs().max()),
                    posterior_chol_identity_max_abs=float((chol-identity).abs().max())))
            print(f"history seed={entry['training_seed']} window={index+1}/{len(panel)}", flush=True)
        result['checkpoints'].append(entry)
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    result['complete'] = True
    result['learned_history_established'] = False
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
