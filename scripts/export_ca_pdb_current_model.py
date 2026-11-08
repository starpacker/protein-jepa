"""Export CA-only PDB trajectories for the S_0-arm current model (E3.1 manual check).

The current model is CA-only (``CADynamics`` decodes Cα, no backbone). This
exports a few forecast CA trajectories as multi-MODEL CA PDBs, reusing the
``write_ca_pdb`` helper from ``export_eval_trajectories_pdb`` so the format
matches the v3 exports exactly. Reconstruction uses ``noise_mix=0`` (the stale
``args.noise_mix=0.8`` is overridden, matching the training launch and the
``s0_mode='exact'`` identity-noise requirement).

Only the val manifest is read; test.jsonl is never opened.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from evaluation.ca_trajectory_adapter import sample_ca_trajectory
from evaluation.data import EvalProtocol, iter_windows
from protein_jepa_frame.data.dataset import AA_ALPHABET
from scripts.diag_lasp_ca_ceiling import _development_manifest, _window_seed
from scripts.export_eval_trajectories_pdb import write_ca_pdb
from scripts.eval_ca_ordered_paths_noisemix0 import load_model_noisemix0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--manifest', default='artifacts/mdcath_manifest_full/val.jsonl')
    p.add_argument('--n-windows', type=int, default=3)
    p.add_argument('--horizon', type=int, default=64)
    p.add_argument('--n-samples', type=int, default=4)
    p.add_argument('--max-residues', type=int, default=96)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(0)
    manifest = _development_manifest(a.manifest)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    component_path = Path(a.outdir) / '_tmp.component.pt'
    component_path.parent.mkdir(parents=True, exist_ok=True)
    model = load_model_noisemix0(checkpoint, a.checkpoint, a.device, component_path)

    protocol = EvalProtocol(manifest=manifest, context_frames=8,
        horizons_frames=(a.horizon,), max_residues=a.max_residues,
        max_anchors_per_trajectory=4, temperatures=(320., 348., 379., 413., 450.),
        seed=0, round_robin_proteins=True, round_robin_conditions=True)

    n_windows_done = 0
    for window in iter_windows(protocol):
        if n_windows_done >= a.n_windows:
            break
        n_windows_done += 1
        context = window.context
        observed = context['coords'].to(a.device)
        mask = context['residue_mask'].to(a.device).all(0).bool()
        aa, ri, ci = [context[key].to(a.device) for key in ('aatype', 'residue_index', 'chain_id')]
        seed = _window_seed(0, window)
        result = sample_ca_trajectory(model, observed[None], mask[None], aa[None], ri[None], ci[None],
            torch.tensor([window.temperature_k], dtype=observed.dtype, device=a.device),
            window.horizon, a.n_samples, torch.Generator(device=a.device).manual_seed(seed),
            context_dt_ps=float(model.kernel.cfg.step_ps))
        ca = result.ca[0].detach().cpu().double().numpy()  # [n_samples, T, L, 3]
        # One PDB per sample, multi-MODEL over the T frames.
        for s in range(a.n_samples):
            frames = ca[s]  # [T, L, 3]
            path = Path(a.outdir) / (
                f"{window.protein_id}_{window.trajectory_id}_h{window.horizon}_s{s}.pdb")
            write_ca_pdb(path, frames, aa.cpu(), ri.cpu(), ci.cpu(),
                         origin=f"current-model exact arm forecast seed42 noise_mix=0 "
                                f"temp={window.temperature_k}K")
            print(f"wrote {path.name}", flush=True)
    component_path.unlink(missing_ok=True)
    print(json.dumps(dict(n_windows=n_windows_done)), flush=True)


import json  # noqa: E402  (kept at bottom for clarity; json is stdlib)


if __name__ == '__main__':
    main()
