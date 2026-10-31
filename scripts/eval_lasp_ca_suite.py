"""Temperature generalization (T3) and static-ensemble (T2A) suite for LASP CA.

Adds two evaluation families to the autonomous CA development harness, using the
documented definitions (EVALUATION_METHODS_AND_COMMANDS_20261021.md 2.6 / 2.9):

  * per-temperature ensemble metrics plus a temperature-conditioning matrix
    (correct / wrong / none), so we can ask whether the temperature input really
    changes the generated ensemble, and how accuracy moves across temperatures;
  * BioEmu-style static-ensemble metrics (reference-state coverage, free-energy
    MAE, population MAE/TV/JSD) with one reference clustering per trajectory,
    taken from the true MD frames of that same trajectory.

Sampling never receives future information; future coordinates enter metrics and
the reference ensembles only. Development diagnostic, not a test claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluation.data import EvalProtocol, iter_windows
from evaluation.ensemble_metrics_ext import extended_ensemble_summary
from evaluation.geometry import aligned_rmsd, ca_from_backbone, common_anchor_rmsd
from evaluation.t2a_static import static_ensemble_terms
from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.dataset import TrajectoryClipDataset
from protein_jepa_frame.data.manifest import load_manifest
from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics
from protein_jepa_frame.training.train_lasp_calibration import calibration_terms
from scripts.eval_lasp_final import METRICS
from scripts.eval_lasp_ca_probe import bootstrap

MDCATH_TEMPS = (320.0, 348.0, 379.0, 413.0, 450.0)
POSITION_METRICS = ('kabsch_ca_rmsd', 'aligned_ca_rmsd')


def reference_ensemble(manifest, trajectory_ids, max_residues, max_frames=512):
    """True CA frames per trajectory id, used as the T2A reference ensemble."""
    records = {r.trajectory_id: r for r in load_manifest(manifest)}
    missing = sorted(set(trajectory_ids) - set(records))
    if missing:
        raise ValueError(f'trajectories absent from manifest: {missing[:3]}')
    out = {}
    for tid in sorted(set(trajectory_ids)):
        record = records[tid]
        total = int(record.num_frames)
        frames = np.arange(0, total, max(1, total // max_frames))
        ds = TrajectoryClipDataset([record], DataConfig(manifest_path=None,
            num_frames=len(frames), max_residues=max_residues))
        clip = ds._read_clip(record, frames)
        ca = ca_from_backbone(torch.as_tensor(clip['coords'])).float()
        mask = torch.as_tensor(np.asarray(clip['residue_mask'])).bool()
        mask = mask.all(0) if mask.ndim == 2 else mask
        out[tid] = (ca, mask)
    return out


def window_seed(suite_seed, protein_id, trajectory_id, anchor_frame, horizon):
    """Sampling seed for one window, deliberately INDEPENDENT of the T3 arm.

    Excluding the mode means the correct / wrong / none arms of the same window
    draw identical noise and differ only in the temperature handed to the model.
    Temperature changes both the drift and the noise scale, so sharing the noise
    removes the sampling variance that would otherwise mask a real mechanism
    effect.  Seeding from the window's identity (not its loop position) keeps a
    window's seed stable when the manifest is reordered or the sweep is split.
    """
    key = f'{protein_id}|{trajectory_id}|{anchor_frame}|{horizon}'
    return (int(suite_seed) * 1000003
            + int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)) % (2**31 - 1)


def temperature_tensor(mode, true_k, wrong_k, device):
    """T3 conditioning ablation: observed T, a fixed wrong T, or no T input.

    ``wrong_k`` must differ from the observed temperature, otherwise the "wrong"
    arm reproduces the "correct" arm exactly and the mechanism test is vacuous at
    that temperature (it did: the default wrong temperature is 320 K, which is
    also the lowest mdCATH temperature we evaluate).  We fall back to the furthest
    mdCATH temperature instead of silently scoring a degenerate cell.
    """
    if mode == 'correct':
        return torch.tensor([float(true_k)], device=device)
    if mode == 'wrong':
        chosen = float(wrong_k)
        if abs(chosen - float(true_k)) < 1e-6:
            chosen = max(MDCATH_TEMPS, key=lambda t: abs(t - float(true_k)))
        return torch.tensor([chosen], device=device)
    if mode == 'none':
        return None
    raise ValueError(f'unknown temperature mode {mode!r}')


def population_counts(rows):
    """Count distinct proteins and trajectories, never trajectory rows as proteins."""
    return dict(n_proteins=len({r['protein'] for r in rows}),
                n_trajectories=len({(r['protein'], r.get('trajectory', r.get('traj')))
                                    for r in rows}))


def summarize(selected):
    """Protein-clustered bootstrap for every metric under one cell."""
    if not selected:
        return {}
    out = dict(n_windows=len(selected), **population_counts(selected))
    for metric in METRICS:
        out[metric] = bootstrap([dict(protein=r['protein'], value=r['model'][metric]) for r in selected], 'value')
        out['persist_' + metric] = bootstrap([dict(protein=r['protein'], value=r['persist'][metric]) for r in selected], 'value')
    for metric in POSITION_METRICS:
        out[metric] = bootstrap([dict(protein=r['protein'], value=r['position'][metric]) for r in selected], 'value')
        out['persist_' + metric] = bootstrap([dict(protein=r['protein'],
            value=r['position']['persist_' + metric]) for r in selected], 'value')
    out['skill_rmsf'] = bootstrap([dict(protein=r['protein'],
        value=r['persist']['rmsf_mae'] - r['model']['rmsf_mae']) for r in selected], 'value')
    out['skill_position'] = bootstrap([dict(protein=r['protein'],
        value=r['position']['persist_kabsch_ca_rmsd'] - r['position']['kabsch_ca_rmsd']) for r in selected], 'value')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest', default='artifacts/mdcath_manifest_full/val.jsonl')
    p.add_argument('--out', required=True)
    p.add_argument('--temps', type=float, nargs='+', default=list(MDCATH_TEMPS))
    p.add_argument('--modes', nargs='+', default=['correct', 'wrong', 'none'],
                   choices=['correct', 'wrong', 'none'])
    p.add_argument('--wrong-temp', type=float, default=320.0)
    p.add_argument('--horizons', type=int, nargs='+', default=[1, 4, 16, 64])
    p.add_argument('--proteins-per-temp', type=int, default=12)
    p.add_argument('--anchors-per-trajectory', type=int, default=2)
    p.add_argument('--n-samples', type=int, default=12)
    p.add_argument('--context-frames', type=int, default=8)
    p.add_argument('--max-residues', type=int, default=96)
    p.add_argument('--k-states', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--t2a', action='store_true')
    a = p.parse_args()
    if a.t2a and 'correct' not in a.modes:
        p.error('--t2a requires the correct temperature mode')
    torch.set_num_threads(4)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if ck.get('checkpoint_kind') not in ('lasp_ca_dynamics', 'lasp_bounded_endpoint_v1'):
        raise ValueError('incorrect checkpoint kind')
    component = Path(a.out).with_suffix('.component.pt')
    component.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck['component_checkpoint'], component)
    if ck.get('checkpoint_kind') == 'lasp_bounded_endpoint_v1':
        from protein_jepa_frame.training.train_lasp_bounded_endpoint import load_bounded_checkpoint
        model, _ = load_bounded_checkpoint(a.checkpoint, component_path=component, device='cuda')
        model.eval()
    else:
        model = CADynamics(str(component), 'cuda', a.checkpoint,
            history_conditioning=ck['args'].get('history_conditioning', False),
            noise_mix=ck['args'].get('noise_mix', 0.), noise_length=ck['args'].get('noise_length', 8.),
            collective_mobility=ck['args'].get('collective_mobility', False),
            noise_kernel=ck['args'].get('noise_kernel', 'gaussian'),
            noise_cutoff=ck['args'].get('noise_cutoff', 10.)).eval()

    rows = []
    pool = {}
    for T in a.temps:
        for mode in a.modes:
            protocol = EvalProtocol(manifest=a.manifest, horizons_frames=tuple(a.horizons),
                context_frames=a.context_frames, frame_stride=1, max_residues=a.max_residues,
                max_proteins=a.proteins_per_temp, max_anchors_per_trajectory=a.anchors_per_trajectory,
                temperatures=(T,), seed=a.seed, round_robin_proteins=True, round_robin_conditions=True)
            n_before = len(rows)
            for w in iter_windows(protocol):
                mask = w.context['residue_mask'].all(0).bool().cuda()
                if int(mask.sum()) < 4:
                    continue
                seed = window_seed(a.seed, w.protein_id, w.trajectory_id,
                                   w.anchor_frame, w.horizon)
                generator = torch.Generator(device='cuda').manual_seed(seed)
                temp = temperature_tensor(mode, w.temperature_k, a.wrong_temp, 'cuda')
                used_k = None if temp is None else float(temp[0])
                with torch.no_grad():
                    ca, _ = model.sample(w.context['coords'][None].cuda(), mask[None],
                        *[w.context[k][None].cuda() for k in ('aatype', 'residue_index', 'chain_id')],
                        temp, w.horizon, a.n_samples, generator)
                    anchor = w.context['coords'][-1, :, 1].cuda()
                    geometry = calibration_terms(ca,
                        anchor[None, None].expand(1, a.n_samples, -1, -1), mask[None],
                        w.context['residue_index'][None].cuda(), w.context['chain_id'][None].cuda())
                    geometry = {k: float(geometry[k]) for k in
                                ('ca_bond', 'ca_clash_fraction', 'ca_angle_cos_mae')}
                    truth = w.future_path['coords'][:, :, 1].cuda()
                    # Pre-masked with mask=None, matching eval_lasp_ca_dynamics.
                    pred = ca[0][:, mask].float()
                    true_ca = truth[:, mask].float()
                    anchor_ca = anchor[mask]
                    metrics = extended_ensemble_summary(pred, true_ca, None, cutoff=8., pca_k=3)
                    persist = extended_ensemble_summary(anchor_ca[None].expand(len(truth), -1, -1).float(),
                                                        true_ca, None, cutoff=8., pca_k=3)
                    target_ca = truth[-1][mask]
                    unit = torch.ones(int(mask.sum()), dtype=torch.bool, device='cuda')
                    kabsch = float(np.nanmean([common_anchor_rmsd(pred[i], anchor_ca, target_ca, unit)
                                               for i in range(pred.shape[0])]))
                    aligned = float(np.nanmean([aligned_rmsd(pred[i], target_ca, unit)
                                                for i in range(pred.shape[0])]))
                    p_kabsch = common_anchor_rmsd(anchor_ca, anchor_ca, target_ca, unit)
                    p_aligned = aligned_rmsd(anchor_ca, target_ca, unit)
                rows.append(dict(protein=w.protein_id, traj=w.trajectory_id,
                    temperature_k=float(w.temperature_k), T_star=float(T), mode=mode,
                    temperature_used_k=used_k,
                    horizon=float(w.horizon), anchor_frame=int(w.anchor_frame),
                    geometry_vs_anchor=geometry,
                    model={m: float(metrics[m]) for m in METRICS},
                    persist={m: float(persist[m]) for m in METRICS},
                    position=dict(kabsch_ca_rmsd=kabsch, aligned_ca_rmsd=aligned,
                                  persist_kabsch_ca_rmsd=p_kabsch, persist_aligned_ca_rmsd=p_aligned)))
                if a.t2a and mode == 'correct':
                    entry = pool.setdefault((w.protein_id, w.trajectory_id, float(w.temperature_k)),
                                            dict(samples=[], mask=mask.detach().cpu()))
                    if not torch.equal(entry['mask'], mask.detach().cpu()):
                        raise ValueError('inconsistent residue mask within one trajectory')
                    entry['samples'].append(ca[0].detach().cpu())
            if len(rows) == n_before:
                raise ValueError(f'no eligible suite windows for T={T:g} mode={mode}')
            print(f'[suite] T={T:g} mode={mode} windows={len(rows) - n_before}', flush=True)

    result = dict(checkpoint=a.checkpoint, args=vars(a), n_windows=len(rows),
        **population_counts(rows),
        noise_protocol=dict(version='shared-window-noise-v1',
            seed_identity=['protein_id', 'trajectory_id', 'anchor_frame', 'horizon'],
            shared_across_modes=True, hash='sha256-first12',
            seed_formula='(suite_seed * 1000003 + window_hash) % (2**31 - 1)'),
        note='Development diagnostic. Future inputs are metrics/reference only.',
        by_temperature={f'{T:g}': {mode: summarize([r for r in rows if r['T_star'] == T and r['mode'] == mode])
                                   for mode in a.modes} for T in a.temps},
        by_horizon={str(h): {mode: summarize([r for r in rows if r['horizon'] == h and r['mode'] == mode])
                             for mode in a.modes} for h in a.horizons},
        aggregate={mode: summarize([r for r in rows if r['mode'] == mode]) for mode in a.modes},
        per_window=rows)
    if a.t2a:
        refs = reference_ensemble(a.manifest, [traj for _, traj, _ in pool], a.max_residues)
        per_protein = {}
        for (protein, traj, T), entry in sorted(pool.items()):
            ca_ref, ref_mask = refs[traj]
            use = entry['mask']
            if int(use.sum()) < 4:
                continue
            if not torch.equal(use, ref_mask):
                raise ValueError('protocol and reference residue masks differ for ' + traj)
            # Both populations are pre-restricted to the evaluated residues, so
            # the metric receives mask=None (a full-length mask would index the
            # compacted arrays out of range).
            pooled = torch.cat(entry['samples'], 0)[:, use]
            terms = static_ensemble_terms(pooled, ca_ref[:, use], None, T,
                                          k_states=a.k_states, seed=a.seed)
            per_protein.setdefault(f'{T:g}', []).append(dict(protein=protein, trajectory=traj, **terms))
        result['t2a'] = {T: dict(**population_counts(v), **{k: bootstrap(
            [dict(protein=r['protein'], value=r[k]) for r in v], 'value') for k in
            ('coverage', 'free_energy_mae_kcal', 'population_mae', 'population_tv', 'population_jsd')})
            for T, v in per_protein.items()}
        result['t2a_per_protein'] = per_protein
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(n_windows=len(rows), t2a=bool(a.t2a))), flush=True)


if __name__ == '__main__':
    main()