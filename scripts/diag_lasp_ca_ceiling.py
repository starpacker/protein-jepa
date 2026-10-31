"""Oracle reconstruction probe on forecast windows; never a forecast or capacity bound."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from evaluation.data import EvalProtocol, iter_windows
from evaluation.ensemble_metrics_ext import extended_ensemble_summary, _align_to
from protein_jepa_frame.models.ca_component_bundle import CAComponentBundle
from protein_jepa_frame.training.train_lasp_ca_dynamics import energy_score
from protein_jepa_frame.training.train_stage_b1a import _extract_backbone_frames
from scripts.eval_lasp_final import METRICS


def endpoint_probe(bundle, h, anchor_h, aa, ri, ci, rotation, anchor, mask,
                   target, steps=30, lr=.05):
    """Optimize only an oracle endpoint latent; retain its best attainable fit."""
    latent = torch.nn.Parameter(h.detach().clone())
    opt = torch.optim.Adam([latent], lr=lr)
    decode = lambda z: bundle.decode_ca(z, anchor_h, aa, ri, ci, rotation, anchor, mask)
    def error(pred):
        return (pred[:, mask[0]]-target[:, mask[0]]).square().sum(-1).mean()
    with torch.no_grad():
        initial = error(decode(latent)).item()
    best = initial
    best_latent = latent.detach().clone()
    for _ in range(steps):
        opt.zero_grad()
        loss = error(decode(latent))
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite oracle reconstruction loss')
        loss.backward()
        if latent.grad is None or not torch.isfinite(latent.grad).all():
            raise FloatingPointError('nonfinite oracle latent gradient')
        opt.step()
        with torch.no_grad():
            value_tensor = error(decode(latent))
            if not torch.isfinite(value_tensor):
                raise FloatingPointError('nonfinite updated oracle reconstruction loss')
            value = value_tensor.item()
            if value < best:
                best, best_latent = value, latent.detach().clone()
    return dict(encoded_endpoint_rmsd=initial**.5, optimized_endpoint_rmsd=best**.5,
                latent_change_rms=float((best_latent-h).square().mean().sqrt()))


def _development_manifest(manifest):
    """Allow only the approved development manifest, resolving aliases first."""
    approved = (Path(__file__).resolve().parents[1]
                / 'artifacts/mdcath_manifest_full/val.jsonl').resolve()
    resolved = Path(manifest).resolve()
    if resolved != approved:
        raise ValueError('diagnostics require the approved development val.jsonl manifest')
    return str(resolved)


def _same_component(left, right):
    """Compare component contents without depending on pickle serialization."""
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left, right))
    if isinstance(left, np.ndarray):
        return (isinstance(right, np.ndarray) and left.dtype == right.dtype
                and np.array_equal(left, right))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_component(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _same_component(a, b) for a, b in zip(left, right))
    return left == right


def _window_seed(seed, window):
    """Match the stateless seed used by the standard endpoint evaluator."""
    key = f'{window.protein_id}|{window.trajectory_id}|{window.anchor_frame}|{window.horizon}'
    return (int(seed) * 1000003
            + int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)) % (2**31 - 1)


def latent_transition_terms(generated_h, anchor_h, target_h, mask):
    """Compare generated and observed-to-target latent transitions.

    The target is one realized future endpoint, so these are conditional
    calibration diagnostics rather than a proper estimate of the full target
    distribution. Energy scores retain the sample ensemble and are normalized
    by the flattened latent dimension.
    """
    valid = mask[0].bool()
    if not bool(valid.any()) or generated_h.shape[1] < 2:
        raise ValueError('latent diagnostics require valid residues and at least two samples')
    pred = generated_h[0, :, valid].flatten(1)
    anchor = anchor_h[0, valid].flatten()
    target = target_h[0, valid].flatten()
    pred_delta = pred - anchor
    target_delta = target - anchor
    target_delta_batch = target_delta[None]
    target_batch = target[None]
    target_delta_norm = target_delta.norm()
    pred_delta_norm = pred_delta.norm(dim=-1)
    mean_delta = pred_delta.mean(0)
    cosine = F.cosine_similarity(
        pred_delta, target_delta_batch.expand_as(pred_delta), dim=-1, eps=1e-8)
    mean_cosine = F.cosine_similarity(mean_delta[None], target_delta_batch, dim=-1, eps=1e-8)[0]
    norm_ratio = pred_delta_norm / target_delta_norm.clamp_min(1e-8)
    mean_norm_ratio = mean_delta.norm() / target_delta_norm.clamp_min(1e-8)
    n = int(valid.sum())
    pred_collective = pred_delta.reshape(len(pred), n, -1).sum(1) / n**.5
    true_collective = target_delta.reshape(n, -1).sum(0, keepdim=True) / n**.5
    # This orthogonal constant-residue projection can cancel internal motions.
    true_energy = target_delta.double().square().sum()
    pred_energy = pred_delta.double().square().sum()
    values = dict(
        true_collective_energy_fraction=float(
            true_collective.double().square().sum() / true_energy.clamp_min(1e-16)),
        generated_collective_energy_fraction=float(
            pred_collective.double().square().sum() / pred_energy.clamp_min(1e-16)),
        collective_displacement_energy_score=float(energy_score(pred_collective, true_collective)),
        collective_persistence_energy_score=float(true_collective.norm() / true_collective.shape[-1]**.5),
        latent_persistence_energy_score=float(target_delta_norm / pred.shape[-1]**.5),
        true_displacement_zero=bool(target_delta_norm == 0),
        latent_dimension=int(pred.shape[-1]),
        generated_delta_norm_mean=float(pred_delta_norm.mean()),
        true_delta_norm=float(target_delta_norm),
        latent_displacement_cosine_mean=float(cosine.mean()),
        latent_displacement_cosine_of_mean=float(mean_cosine),
        latent_displacement_norm_ratio_mean=float(norm_ratio.mean()),
        latent_displacement_norm_ratio_of_mean=float(mean_norm_ratio),
        latent_displacement_energy_score=float(energy_score(pred_delta, target_delta_batch)),
        latent_endpoint_energy_score=float(energy_score(pred, target_batch)),
    )
    if not all(math.isfinite(value) for value in values.values()
               if isinstance(value, float)):
        raise FloatingPointError('nonfinite latent transition diagnostic')
    return values


def _latent_transition_row(model, window, n_samples, seed, device):
    context = torch.as_tensor(window.context['coords'], device=device)
    future = torch.as_tensor(window.future_path['coords'], device=device)
    mask = torch.as_tensor(window.context['residue_mask'], device=device).all(0).bool()
    future_mask = torch.as_tensor(window.future_path['residue_mask'], device=device).bool()
    if not bool(future_mask[:, mask].all()):
        raise ValueError('future reference mask differs on evaluated residues')
    observed = context[-1:]
    aa = torch.as_tensor(window.context['aatype'], device=device).long()[None]
    ri = torch.as_tensor(window.context['residue_index'], device=device).long()[None]
    ci = torch.as_tensor(window.context['chain_id'], device=device).long()[None]
    temperature = torch.tensor([float(window.temperature_k)], device=device)
    sampling_seed = _window_seed(seed, window)
    generator = torch.Generator(device=device).manual_seed(sampling_seed)
    with torch.no_grad():
        anchor_h = model.bundle.encode(observed, mask[None], aa, ri, ci)
        target_h = model.bundle.encode(future[-1:], mask[None], aa, ri, ci)
        generated_ca, generated_h = model.sample(
            context[None], mask[None], aa, ri, ci, temperature,
            int(window.horizon), n_samples, generator)
        truth_ca = future[:, :, 1]
        pred_ca = generated_ca[0, :, mask].float()
        target_ca = truth_ca[:, mask].float()
        aligned_endpoint = _align_to(truth_ca[-1:], observed[0, :, 1], mask)[:, mask].float()
        coordinate_es = float(energy_score(pred_ca.flatten(1), aligned_endpoint.flatten(1)))
        coordinate_persistence_es = float(
            (aligned_endpoint - observed[:, mask, 1]).flatten(1).norm(dim=-1).mean()
            / (3*int(mask.sum()))**.5)
        decoded = extended_ensemble_summary(
            pred_ca, target_ca, None, cutoff=8., pca_k=3)
    latent = latent_transition_terms(generated_h, anchor_h, target_h, mask[None])
    latent.update(coordinate_endpoint_energy_score=coordinate_es,
                  coordinate_persistence_energy_score=coordinate_persistence_es)
    decoded = {key: float(decoded[key]) for key in METRICS}
    return dict(
        protein=window.protein_id,
        trajectory=window.trajectory_id,
        temperature_k=float(window.temperature_k),
        anchor_frame=int(window.anchor_frame),
        horizon=int(window.horizon),
        sampling_seed=int(sampling_seed),
        latent=latent,
        decoded=decoded,
    )


def _mean_nested(rows, field):
    keys = sorted(key for key in rows[0][field] if key != 'latent_dimension')
    return {key: sum(row[field][key] for row in rows) / len(rows) for key in keys}


def run_latent_transition_probe(args):
    """Run a fixed standard200 no-update transition calibration panel."""
    args.manifest = _development_manifest(args.manifest)
    device = torch.device(args.device)
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    component = torch.load(args.component, map_location='cpu', weights_only=False)
    if not _same_component(source.get('component_checkpoint'), component):
        raise ValueError('component differs from the selected checkpoint embedded component')
    from protein_jepa_frame.training.train_lasp_bounded_endpoint import load_bounded_checkpoint
    model, source = load_bounded_checkpoint(
        args.checkpoint, component_path=args.component, device=device)
    model.eval()
    protocol = EvalProtocol(
        manifest=args.manifest, horizons_frames=(1, 4, 16, 64), context_frames=8,
        frame_stride=1, max_residues=96, max_anchors_per_trajectory=4,
        seed=args.seed, round_robin_proteins=True, round_robin_conditions=True)
    rows = []
    for window in iter_windows(protocol):
        if len(rows) >= args.max_windows:
            break
        mask = torch.as_tensor(window.context['residue_mask']).all(0).bool()
        if int(mask.sum()) < 4:
            continue
        rows.append(_latent_transition_row(
            model, window, args.n_samples, args.seed, device))
        if len(rows) % 10 == 0:
            print(f'latent transition scored {len(rows)}', flush=True)
    if len(rows) != args.max_windows:
        raise RuntimeError(f'expected {args.max_windows} standard windows, got {len(rows)}')
    by_horizon = {}
    for horizon in (1, 4, 16, 64):
        selected = [row for row in rows if row['horizon'] == horizon]
        if not selected:
            raise RuntimeError(f'missing latent transition horizon {horizon}')
        by_horizon[str(horizon)] = dict(
            n_windows=len(selected), latent=_mean_nested(selected, 'latent'),
            decoded=_mean_nested(selected, 'decoded'))
    result = dict(
        scope='development no-update latent transition calibration diagnostic; not a forecast or test result',
        checkpoint=args.checkpoint, component=args.component,
        checkpoint_kind=source['checkpoint_kind'], test_opened=False,
        optimizer_steps=0, n_windows=len(rows),
        n_proteins=len({row['protein'] for row in rows}),
        n_trajectories=len({(row['protein'], row['trajectory']) for row in rows}),
        protocol=dict(
            manifest=args.manifest, horizons=[1, 4, 16, 64], context_frames=8,
            max_residues=96, max_windows=args.max_windows, n_samples=args.n_samples,
            evaluator_seed=args.seed, shared_stateless_window_seed=True,
            seed_formula='(seed*1000003 + sha256(protein|trajectory|anchor|horizon)[:12]) % (2**31 - 1)',
            future_coordinates_used='target encoding and metrics only; never passed to sampler'),
        by_horizon=by_horizon, per_window=rows)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(by_horizon, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['oracle', 'latent_transition'], default='oracle')
    p.add_argument('--component', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--out', required=True)
    p.add_argument('--manifest', default='artifacts/mdcath_manifest_full/val.jsonl')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n-samples', type=int, default=12)
    p.add_argument('--max-windows', type=int, default=200)
    p.add_argument('--oracle-steps', type=int, default=30)
    p.add_argument('--oracle-lr', type=float, default=.05)
    a = p.parse_args()
    a.manifest = _development_manifest(a.manifest)
    torch.set_num_threads(4)
    if a.max_windows < 1 or a.oracle_steps < 0 or a.oracle_lr <= 0:
        p.error('positive window budget/lr and nonnegative oracle steps required')
    if a.mode == 'latent_transition':
        if not a.checkpoint:
            p.error('--checkpoint is required for --mode latent_transition')
        if a.n_samples < 2:
            p.error('--n-samples must be at least two')
        run_latent_transition_probe(a)
        return
    device = torch.device(a.device)
    bundle = CAComponentBundle(a.component, device).eval()
    rows = []
    proto = EvalProtocol(manifest=a.manifest, horizons_frames=(1, 4, 16, 64), context_frames=8,
        max_residues=96, max_anchors_per_trajectory=4, round_robin_proteins=True, round_robin_conditions=True)
    for w in iter_windows(proto):
        if len(rows) >= a.max_windows:
            break
        mask = w.context['residue_mask'].all(0).bool().to(device)
        coords = w.future_path['coords'].to(device)
        if not bool(w.future_path['residue_mask'][:, mask.cpu()].all()):
            raise ValueError('future reference mask differs on evaluated residues')
        observed = w.context['coords'][-1:].to(device)
        aa, ri, ci = [w.context[k][None].to(device) for k in ('aatype', 'residue_index', 'chain_id')]
        n = len(coords)
        m = mask[None].expand(n, -1)
        with torch.no_grad():
            h = bundle.encode(coords, m, aa.expand(n, -1), ri.expand(n, -1), ci.expand(n, -1))
            ha = bundle.encode(observed, mask[None], aa, ri, ci)
            R, anchor = _extract_backbone_frames(observed)
            pred = bundle.decode_ca(h, ha.expand_as(h), aa.expand(n, -1), ri.expand(n, -1),
                ci.expand(n, -1), R.expand(n, -1, -1, -1), anchor.expand(n, -1, -1), m)
            truth = coords[:, :, 1]
            metrics = extended_ensemble_summary(pred[:, mask], truth[:, mask], None)
            target = _align_to(truth[-1:], observed[0, :, 1], mask)
            persist = (observed[:, mask, 1]-target[:, mask]).square().sum(-1).mean().sqrt()
        endpoint = endpoint_probe(bundle, h[-1:], ha, aa, ri, ci, R, anchor, mask[None],
                                  target, a.oracle_steps, a.oracle_lr)
        rows.append(dict(protein=w.protein_id, traj=w.trajectory_id, temperature_k=float(w.temperature_k),
            anchor_frame=int(w.anchor_frame), horizon=float(w.horizon),
            oracle={k: float(metrics[k]) for k in METRICS},
            persist_endpoint_rmsd=float(persist), **endpoint))
        if len(rows) % 10 == 0:
            print(f'oracle scored {len(rows)}', flush=True)
    result = dict(scope='Oracle diagnostic using future encoding/latent optimization; never a forecast or formal capacity bound',
        args=vars(a), n_windows=len(rows), test_opened=False,
        by_horizon={str(h): {k: sum(r[k] for r in rows if r['horizon'] == h)/sum(r['horizon'] == h for r in rows)
                            for k in ('encoded_endpoint_rmsd', 'optimized_endpoint_rmsd', 'persist_endpoint_rmsd')}
                    for h in (1, 4, 16, 64) if any(r['horizon'] == h for r in rows)}, per_window=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result['by_horizon'], indent=2), flush=True)


if __name__ == '__main__':
    main()
