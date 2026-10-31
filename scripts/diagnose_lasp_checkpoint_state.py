"""No-update latent health audit using established stability helpers."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from scripts.diagnose_lasp_stability import (
    _panel, _jsonable_window, _as_device, _repeat, _decode, _finite_stats,
    _state_diagnostics,
)
from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics
from protein_jepa_frame.training.train_stage_b1a import _extract_backbone_frames


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', nargs='+', required=True)
    p.add_argument('--manifest', default='artifacts/mdcath_manifest_full/val.jsonl')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--temperatures', type=float, nargs='+', default=[348., 450.])
    p.add_argument('--horizons', type=int, nargs='+', default=[1, 64])
    p.add_argument('--max-proteins', type=int, default=2)
    p.add_argument('--max-residues', type=int, default=96)
    p.add_argument('--n-samples', type=int, default=12)
    p.add_argument('--seed', type=int, default=20261024)
    a = p.parse_args()
    torch.set_num_threads(2)
    device = torch.device(a.device)
    panel = _panel(a.manifest, a.temperatures, a.horizons, a.max_residues,
                   a.max_proteins, a.max_proteins)
    output = Path(a.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(scope='no-update latent health; not forecast accuracy',
                  optimizer_steps=0, test_opened=False, complete=False, args=vars(a),
                  manifest_sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
                  panel=[_jsonable_window(w) for w in panel], checkpoints=[])
    for cp in a.checkpoint:
        ck = torch.load(cp, map_location='cpu', weights_only=False)
        if ck.get('checkpoint_kind') != 'lasp_ca_dynamics':
            raise ValueError(f'unsupported checkpoint {cp}')
        component = output.parent / f"health_seed{ck['args']['seed']}.component.pt"
        torch.save(ck['component_checkpoint'], component)
        model = CADynamics(str(component), device, cp,
            history_conditioning=ck['args'].get('history_conditioning', False),
            noise_mix=ck['args'].get('noise_mix', 0.),
            noise_length=ck['args'].get('noise_length', 8.),
            collective_mobility=ck['args'].get('collective_mobility', False),
            noise_kernel=ck['args'].get('noise_kernel', 'gaussian'),
            noise_cutoff=ck['args'].get('noise_cutoff', 10.)).eval()
        if model.history_conditioning or not model.collective_mobility:
            raise ValueError('this audit is scoped to frozen history-free collective SOTA')
        entry = dict(checkpoint=cp, step=ck['step'], seed=ck['args']['seed'],
                     checkpoint_sha256=hashlib.sha256(Path(cp).read_bytes()).hexdigest(), windows=[])
        result['checkpoints'].append(entry)
        for index, window in enumerate(panel):
            coords = _as_device(window.context['coords'], device)[None]
            mask = _as_device(window.context['residue_mask'], device).all(0).bool()[None]
            aa, ri, ci = [_as_device(window.context[k], device).long()[None]
                          for k in ('aatype', 'residue_index', 'chain_id')]
            temperature = torch.tensor([window.temperature_k], device=device)
            with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                start_one = model.bundle.encode(coords[:, -1], mask, aa, ri, ci)
                start, rm = _repeat(start_one, a.n_samples), _repeat(mask, a.n_samples)
                temp = _repeat(temperature[:, None], a.n_samples).flatten()
                factor = _repeat(model.anchor_kernel(coords[:, -1, :, 1], mask), a.n_samples)
                generator = torch.Generator(device=device).manual_seed(a.seed+index*7919)
                path, bath = model.kernel.rollout(start, rm, temp, window.horizon,
                    seq_feature=start, generator=generator, detach=True,
                    return_bath=True, noise_factor=factor, collective_mobility=True)
                rotation, anchor = _extract_backbone_frames(coords[:, -1])
                raw, ca = _decode(model, path[:, -1], start_one, aa, ri, ci, rotation, anchor, mask)
                state = _state_diagnostics(model, path, bath, start, temp, rm, factor)
                D, C, gamma = model.kernel.bath(start, temp)
                frames = []
                for step in range(path.shape[1]):
                    force = model.kernel.compute_forces(path[:, step], rm, temp, create_graph=False)
                    tail_force = -torch.nn.functional.softplus(model.kernel.energy.log_tail)*path[:, step]
                    energy = model.kernel.energy(path[:, step], rm, temp)
                    record = dict(step=step, energy=_finite_stats(energy),
                        tail_force=_finite_stats(tail_force), residual_force=_finite_stats(force-tail_force))
                    if step+1 < path.shape[1]:
                        drive = D.sqrt()*(force-torch.einsum('blij,blj->bli', C, bath[:, step]))
                        drive = torch.einsum('bij,bjd->bid', factor.transpose(-1,-2), drive)
                        drive = torch.einsum('bij,bjd->bid', factor, drive)
                        drift = model.kernel.cfg.step_ps*D.sqrt()*drive
                        record['realised_H_noise'] = _finite_stats(path[:, step+1]-path[:, step]-drift)
                    frames.append(record)
                perturbation = torch.randn(start.shape, device=device, generator=generator)
                perturbation = perturbation/perturbation.square().mean().sqrt()*.001
                _, ca_perturbed = _decode(model, path[:, -1]+perturbation, start_one,
                                          aa, ri, ci, rotation, anchor, mask)
                _, ca_scaled = _decode(model, path[:, -1]*2., start_one,
                                       aa, ri, ci, rotation, anchor, mask)
                bonds = (ca[:, 1:]-ca[:, :-1]).double().norm(dim=-1)
                valid = _repeat(mask[:, 1:] & mask[:, :-1] & (ci[:, 1:]==ci[:, :-1])
                                & (ri[:, 1:]-ri[:, :-1]==1), a.n_samples)
                entry['windows'].append(dict(identity=_jsonable_window(window),
                    actual_n_samples=int(path.shape[0]), state=state, energy_and_noise=frames,
                    initial_H=_finite_stats(path[:, 0]), final_H=_finite_stats(path[:, -1]),
                    initial_S=_finite_stats(bath[:, 0]), final_S=_finite_stats(bath[:, -1]),
                    ca_raw=_finite_stats(raw), ca_projected=_finite_stats(ca),
                    decoder_perturbation_CA_rms=_finite_stats(ca_perturbed-ca)['rms'],
                    decoder_scale2_CA_rms=_finite_stats(ca_scaled-ca)['rms'],
                    force_clip_fraction=0. if ck['kernel_config']['force_clip'] is None else None,
                    force_clip_active=ck['kernel_config']['force_clip'] is not None,
                    geometry=dict(bond_min=float(bonds[valid].min()), bond_max=float(bonds[valid].max()),
                                  bond_rms_error=float((bonds[valid]-3.8).square().mean().sqrt()))))
            print(f"seed={ck['args']['seed']} step={ck['step']} window={index+1}/{len(panel)}", flush=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    result['complete'] = True
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
