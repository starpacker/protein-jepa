"""Versioned ordered-path diagnostics; historical trajectory_metrics is unchanged.

The joint energy scores use the entire future path as ONE observation. Coordinate
scores use the fixed observed-anchor gauge, align only the real reference, and
exclude time zero. Distance scores concatenate the same physical residue-pair
features at every time. Both use the off-diagonal U-statistic (M >= 2).
All public entry points promote coordinate arithmetic to float64.

Lag diagnostics preserve each ensemble member/replica as a separate trajectory.
Their MSD is mean squared INTERNAL-DISTANCE change in Angstrom^2, not atomic
diffusion. ACF is finite-path, mean-centered pair covariance divided by that
path's variance; it is not a relaxation-time/MSM estimator. These diagnostics do
not establish T2B, stationary kinetics, or velocity/increment correlations.
"""
from __future__ import annotations

import math

import torch

from evaluation.geometry import kabsch_align


TASK_VERSION = 'ca_ordered_path_metrics_v2'
DEFAULT_LAGS = (1, 2, 4, 8, 16, 32)


def _mask_and_pairs(mask, residue_index, chain_id, length, device, max_pairs=None):
    if mask.shape != (length,) or mask.dtype != torch.bool or mask.device != device:
        raise ValueError('residue_mask must be a boolean [L] tensor on the coordinate device')
    if int(mask.sum()) < 4:
        raise ValueError('ordered-path metrics require at least four valid residues')
    for value in (residue_index, chain_id):
        if value.shape != (length,) or value.device != device or value.dtype not in (
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError('residue_index and chain_id must be integer [L] tensors')
    identities = torch.stack((chain_id[mask], residue_index[mask]), -1)
    if len(torch.unique(identities, dim=0)) != len(identities):
        raise ValueError('valid physical residue identities must be unique')
    i, j = torch.triu_indices(length, length, offset=1, device=device)
    keep = mask[i] & mask[j] & (
        (chain_id[i] != chain_id[j]) | ((residue_index[i] - residue_index[j]).abs() >= 3))
    pairs = torch.stack((i[keep], j[keep]), -1)
    if not len(pairs):
        raise ValueError('no valid physical nonlocal residue pairs')
    if max_pairs is not None:
        if isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or max_pairs < 1:
            raise ValueError('max_pairs must be a positive integer or None')
        if len(pairs) > max_pairs:
            select = torch.linspace(0, len(pairs) - 1, max_pairs, device=device).round().long()
            pairs = pairs[select]
    return pairs


def _time_vector(value, frames, device):
    times = torch.as_tensor(value, dtype=torch.float64, device=device)
    if times.shape != (frames,) or not torch.isfinite(times).all():
        raise ValueError('times_ps must contain one finite timestamp per stored frame')
    if not bool((times[1:] > times[:-1]).all()):
        raise ValueError('stored timestamps must be strictly increasing')
    return times


def _validate_paths(predicted, reference, anchor, residue_mask, *, predicted_times_ps,
                    reference_times_ps, reference_mask, residue_index, chain_id, max_pairs,
                    min_members=2):
    if predicted.ndim != 4 or predicted.shape[-1] != 3 or predicted.shape[0] < min_members:
        raise ValueError(f'predicted must be [M,T+1,L,3] with M>={min_members}')
    members, frames, length, _ = predicted.shape
    if frames < 2 or reference.shape != (frames, length, 3) or anchor.shape != (length, 3):
        raise ValueError('prediction/reference must share a nonempty future path and anchor shape')
    if any(not x.is_floating_point() or x.device != predicted.device
           for x in (predicted, reference, anchor)):
        raise ValueError('coordinates must be floating tensors on one device')
    pairs = _mask_and_pairs(residue_mask, residue_index, chain_id, length, predicted.device, max_pairs)
    if reference_mask.dtype != torch.bool or reference_mask.device != predicted.device:
        raise ValueError('reference_mask must be boolean on the coordinate device')
    if reference_mask.shape == (length,):
        reference_mask = reference_mask[None].expand(frames, -1)
    if reference_mask.shape != (frames, length) or not torch.equal(
            reference_mask, residue_mask[None].expand(frames, -1)):
        raise ValueError('reference validity must exactly match the observed mask at every time')
    pt = _time_vector(predicted_times_ps, frames, predicted.device)
    rt = _time_vector(reference_times_ps, frames, predicted.device)
    if not torch.equal(pt, rt):
        raise ValueError('prediction and reference timestamps must match exactly')
    for x in (predicted, reference, anchor):
        if not torch.isfinite(x[..., residue_mask, :]).all():
            raise ValueError('nonfinite coordinates on evaluated residues')
    if not torch.equal(predicted[:, 0, residue_mask], anchor[residue_mask].expand(members, -1, -1)):
        raise ValueError('every predicted path must start at the observed anchor')
    if not torch.equal(reference[0, residue_mask], anchor[residue_mask]):
        raise ValueError('reference path must start at the observed anchor')
    predicted, reference, anchor = predicted.double(), reference.double(), anchor.double()
    # Never align a prediction to the target. The target's rigid-body motion is
    # removed using only its own coordinates and the fixed observed anchor.
    aligned_reference = torch.stack([kabsch_align(frame, anchor, residue_mask) for frame in reference])
    aligned_reference[0] = anchor
    return predicted, reference, anchor, aligned_reference, pt, pairs


def _distance_features(path, pairs):
    return (path[..., pairs[:, 0], :] - path[..., pairs[:, 1], :]).norm(dim=-1)


def _u_energy_score(samples, observation):
    cross = (samples - observation).norm(dim=-1).mean()
    spread = torch.pdist(samples).mean()
    return float((cross - .5 * spread) / math.sqrt(samples.shape[-1]))


def _energy_scores(predicted, reference, mask, pairs, raw_reference):
    if len(predicted) < 2:
        return dict(coordinate_path_energy_score=None, physical_distance_path_energy_score=None)
    return dict(
        coordinate_path_energy_score=_u_energy_score(
            predicted[:, 1:, mask].flatten(1), reference[1:, mask].flatten()),
        physical_distance_path_energy_score=_u_energy_score(
            _distance_features(predicted[:, 1:], pairs).flatten(1),
            _distance_features(raw_reference[1:], pairs).flatten()),
    )


@torch.no_grad()
def joint_path_energy_scores(predicted, reference, anchor, residue_mask, *,
                             predicted_times_ps, reference_times_ps, reference_mask,
                             residue_index, chain_id, max_pairs=None):
    """Whole-path U-ES in Angstrom, normalized by sqrt(3*L*T) / sqrt(P*T).

    The masks must agree at every time; missing future residues are rejected,
    never silently intersected away. Both inputs include time zero, which must
    equal ``anchor`` on valid residues. Scores use only subsequent frames.
    Finite-ensemble U-scores may be negative and are not clamped.
    """
    predicted, reference, anchor, target, _, pairs = _validate_paths(predicted, reference, anchor, residue_mask,
        predicted_times_ps=predicted_times_ps, reference_times_ps=reference_times_ps,
        reference_mask=reference_mask, residue_index=residue_index, chain_id=chain_id,
        max_pairs=max_pairs)
    return _energy_scores(predicted, target, residue_mask, pairs, reference)


def _lag_statistics(paths, times, pairs, lags):
    lags = tuple(lags)
    if not lags or len(set(lags)) != len(lags) or any(
            isinstance(lag, bool) or not isinstance(lag, int) or lag < 0 for lag in lags):
        raise ValueError('lags must be distinct nonnegative integer frame offsets')
    per_path, step_ps = [], None
    for index, (path, time) in enumerate(zip(paths, times)):
        dt = time[1:] - time[:-1]
        current_step = float(dt[0])
        if not torch.allclose(dt, dt[:1].expand_as(dt), rtol=1e-9, atol=1e-9):
            raise ValueError('lag statistics require uniformly spaced timestamps')
        if step_ps is not None and not math.isclose(step_ps, current_step, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError('all trajectories must use the same physical time step')
        step_ps = current_step
        feature = _distance_features(path, pairs)
        centered = feature - feature.mean(0, keepdim=True)
        # Exact constant paths can otherwise acquire rounding residue when
        # subtracting a floating mean. Their ACF is explicitly undefined.
        constant = torch.equal(feature, feature[:1].expand_as(feature))
        variance = 0. if constant else float(centered.square().mean())
        values = []
        for lag in lags:
            count = max(len(path) - lag, 0)
            if count:
                msd = float((feature[lag:] - feature[:count]).square().mean())
                acf = float((centered[lag:] * centered[:count]).mean()) / variance if variance > 0 else None
            else:
                msd, acf = None, None
            values.append(dict(lag=lag, lag_ps=lag * step_ps, n_pairs=count, acf=acf, msd=msd))
        per_path.append(dict(path_index=index, n_frames=len(path), variance=variance, by_lag=values))
    aggregate = []
    for position, lag in enumerate(lags):
        values = [path['by_lag'][position] for path in per_path]
        count = sum(row['n_pairs'] for row in values)
        acf_count = sum(row['n_pairs'] for row in values if row['acf'] is not None)
        aggregate.append(dict(lag=lag, lag_ps=lag * step_ps, n_pairs=count, n_acf_pairs=acf_count,
            msd=sum(row['n_pairs'] * row['msd'] for row in values if row['msd'] is not None) / count if count else None,
            acf=sum(row['n_pairs'] * row['acf'] for row in values if row['acf'] is not None) / acf_count if acf_count else None))
    return dict(per_path=per_path, aggregate=aggregate)


@torch.no_grad()
def path_lag_statistics(paths, times_ps, residue_mask, *, residue_index, chain_id,
                        lags=DEFAULT_LAGS, max_pairs=None):
    """Per-path internal-distance ACF/MSD with explicit legal time-pair counts.

    ``paths`` is [M,T,L,3] or a list of [T_i,L,3] arrays; no boundary joins are
    introduced. ``times_ps`` is a common [T] tensor or one vector per path.
    An ACF uses mean lagged centered feature products / full-path variance.
    This finite-sample normalization need not stay within [-1,1]. Aggregation
    weights each defined per-path value by its actual T_i-lag pair count.
    Constant paths contribute zero MSD but no ACF or ACF pair count.
    """
    if isinstance(paths, torch.Tensor):
        if paths.ndim != 4:
            raise ValueError('paths must be [M,T,L,3] or a list of independent paths')
        paths = list(paths.unbind(0))
    else:
        paths = list(paths)
    if not paths:
        raise ValueError('at least one independent path is required')
    if not isinstance(paths[0], torch.Tensor) or paths[0].ndim != 3:
        raise ValueError('each independent path must be a [T,L,3] tensor')
    length, device = paths[0].shape[-2], paths[0].device
    pairs = _mask_and_pairs(residue_mask, residue_index, chain_id, length, device, max_pairs)
    if isinstance(times_ps, torch.Tensor) and times_ps.ndim == 1:
        time_values = [times_ps] * len(paths)
    else:
        time_values = list(times_ps)
    if len(time_values) != len(paths):
        raise ValueError('one timestamp vector is required per path')
    times = []
    for path, value in zip(paths, time_values):
        if (path.ndim != 3 or path.shape[1:] != (length, 3) or len(path) < 2
                or path.device != device or not path.is_floating_point()):
            raise ValueError('each independent path must be [T>=2,L,3] on one device')
        if not torch.isfinite(path[:, residue_mask]).all():
            raise ValueError('nonfinite coordinates on evaluated residues')
        times.append(_time_vector(value, len(path), device))
    return _lag_statistics([path.double() for path in paths], times, pairs, lags)


def _pearson(left, right):
    left, right = left - left.mean(), right - right.mean()
    denom = left.norm() * right.norm()
    return float((left * right).sum() / denom) if float(denom) > 0 else None


@torch.no_grad()
def ordered_path_summary(predicted, reference, anchor, residue_mask, *,
                         predicted_times_ps, reference_times_ps, reference_mask,
                         residue_index, chain_id, lags=DEFAULT_LAGS, max_pairs=None):
    """JSON-safe order diagnostics; one true reference path, M generated paths.

    Coordinate error keeps predictions in the fixed observed gauge. Internal
    path length/tortuosity and walk RMSF align each prediction to the anchor
    using its own coordinates only; those shape diagnostics remove rigid drift.
    RMSF uses future frames and is undefined for a one-frame reference. ACF/MSD
    include the initial anchor and always preserve individual path boundaries.
    """
    predicted, reference, anchor, target, times, pairs = _validate_paths(predicted, reference, anchor, residue_mask,
        predicted_times_ps=predicted_times_ps, reference_times_ps=reference_times_ps,
        reference_mask=reference_mask, residue_index=residue_index, chain_id=chain_id,
        max_pairs=max_pairs, min_members=1)
    scalar = _energy_scores(predicted, target, residue_mask, pairs, reference)
    error = (predicted[:, 1:, residue_mask] - target[None, 1:, residue_mask]).square().sum(-1)
    rmsd_curve = error.mean(-1).sqrt().mean(0)
    valid = predicted[:, 1:, residue_mask]
    rg = (valid - valid.mean(-2, keepdim=True)).square().sum(-1).mean(-1).sqrt()
    shape_paths = torch.stack([torch.stack([kabsch_align(frame, anchor, residue_mask)
                                          for frame in path]) for path in predicted])[:, :, residue_mask]
    lengths = (shape_paths[:, 1:] - shape_paths[:, :-1]).norm(dim=-1).mean(-1).sum(-1)
    endpoint_distances = (shape_paths[:, -1] - shape_paths[:, 0]).norm(dim=-1).mean(-1)
    per_path = [dict(path_index=i, path_length=float(lengths[i]), end_to_end=float(endpoint_distances[i]),
                     tortuosity=float(lengths[i] / endpoint_distances[i]) if float(endpoint_distances[i]) > 1e-12 else None,
                     rg_mean=float(rg[i].mean())) for i in range(len(predicted))]
    tortuosities = [row['tortuosity'] for row in per_path if row['tortuosity'] is not None]
    scalar.update(per_step_rmsd_mean=float(rmsd_curve.mean()), path_length=float(lengths.mean()),
                  end_to_end=float(endpoint_distances.mean()),
                  tortuosity=sum(tortuosities) / len(tortuosities) if tortuosities else None,
                  rg_mean=float(rg.mean()))
    if len(reference) > 2:
        pooled = shape_paths[:, 1:].flatten(0, 1)
        pred_rmsf = (pooled - pooled.mean(0)).square().sum(-1).mean(0).sqrt()
        real = target[1:, residue_mask]
        real_rmsf = (real - real.mean(0)).square().sum(-1).mean(0).sqrt()
        scalar.update(walk_rmsf_mae=float((pred_rmsf - real_rmsf).abs().mean()),
                      walk_rmsf_corr=_pearson(pred_rmsf, real_rmsf))
    else:
        scalar.update(walk_rmsf_mae=None, walk_rmsf_corr=None)
    prediction_lags = _lag_statistics(list(predicted.unbind(0)), [times] * len(predicted), pairs, lags)
    reference_lags = _lag_statistics([reference], [times], pairs, lags)
    for prefix, result in (('', prediction_lags), ('reference_', reference_lags)):
        lag_one = next((row for row in result['aggregate'] if row['lag'] == 1), None)
        scalar[prefix + 'distance_acf_lag1'] = None if lag_one is None else lag_one['acf']
        scalar[prefix + 'distance_msd_lag1'] = None if lag_one is None else lag_one['msd']
    matching_acf = [(p['lag'], p['acf'], r['acf']) for p, r in zip(
        prediction_lags['aggregate'], reference_lags['aggregate'])
        if p['acf'] is not None and r['acf'] is not None]
    scalar['distance_acf_corr_n_lags'] = len(matching_acf)
    scalar['distance_acf_corr'] = _pearson(
        predicted.new_tensor([row[1] for row in matching_acf]),
        predicted.new_tensor([row[2] for row in matching_acf])) if len(matching_acf) >= 3 else None
    return dict(task_version=TASK_VERSION, scalars=scalar, per_path=per_path,
        curves=dict(times_ps=times[1:].tolist(), per_step_rmsd=rmsd_curve.tolist(), rg_mean=rg.mean(0).tolist()),
        lag_statistics=dict(prediction=prediction_lags, reference=reference_lags),
        conventions=dict(n_samples=len(predicted), n_reference_paths=1, n_future_frames=len(reference)-1,
            arithmetic_dtype='float64',
            n_valid_residues=int(residue_mask.sum()), n_nonlocal_pairs=len(pairs), pair_indices=pairs.tolist(),
            energy_score='joint path, off-diagonal U-statistic; time zero excluded; finite scores may be negative',
            coordinate_normalization='sqrt(3 * valid_residues * future_frames)',
            distance_normalization='sqrt(fixed_physical_pairs * future_frames)',
            coordinate_gauge='prediction unchanged; real reference rigidly aligned to fixed observed anchor',
            shape_gauge='each prediction rigidly aligned to observed anchor using itself only',
            walk_rmsf='pooled M*future_time conformers; includes between-member spread; order-insensitive auxiliary metric, not single-trajectory fluctuation',
            lag_feature='fixed physical nonlocal CA distances; initial anchor included',
            lag_aggregation='per-path estimates weighted by legal time-pair count; never join path boundaries',
            requested_lags=list(lags), acf_correlation_valid_lags=[row[0] for row in matching_acf],
            acf_correlation='Pearson of prediction/reference aggregate ACF on the same requested lags, >=3 defined lags; diagnostic only, no threshold; larger ACF itself is not necessarily better',
            msd_units='Angstrom^2 in internal-distance features, not atomic diffusion',
            undefined='None; U-energy scores with M<2, zero-variance ACF and one-frame-reference RMSF are undefined',
            scope='ordered-path diagnostics only; not an MSM/T2B or stationary-kinetics validation'))
