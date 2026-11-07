"""Fixed development panels for continuous, observed-only CA trajectories.

Each window calls the existing trajectory adapter ONCE. All future states share
the actual sampled S0 and one continuous H/S rollout; predictions are never
re-encoded, re-anchored to truth, or restarted. Real MD is one observed path,
not twelve independent reference trajectories. Reverse/shuffle are diagnostic
permutations of the same forecast, with t0 fixed, never competing models.

Default: the standard200 50-protein prefix, horizons 1/4/16/64 raw frames,
12 paths, seed 0. The separately named --horizons 192 panel defaults to 50
windows. Shorter complete prefixes are smoke tests, not full-panel evidence.
Only the approved development manifest is accepted. No optimizer, model
selection, BioEmu2 rerun, held-out test access, MSM or T2B success claim.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys

import torch

from evaluation.ca_trajectory_adapter import sample_ca_trajectory
from evaluation.data import EvalProtocol, iter_windows
from evaluation.geometry import kabsch_align
from evaluation.loaders import sha256_file
from evaluation.trajectory_metrics_v2 import ordered_path_summary
from protein_jepa_frame.utils.contracts import state_sha256
from scripts.diag_lasp_ca_ceiling import _development_manifest, _window_seed
from scripts.eval_ca_endpoint_scores import SOURCES as ENDPOINT_SOURCES, load_model
from scripts.gate_ca_representation import audit_sources


TASK_VERSION = 'ca_ordered_paths_development_v1'
DEFAULT_HORIZONS = (1, 4, 16, 64)
LAGS = (1, 2, 4, 8, 16, 32)
MAX_PAIRS = 512
ROLES = {
    'forecast': 'one continuous forecast per member; observed-only initialization',
    'reversed_forecast': 'temporal-order diagnostic: reverse future positions, retain t0',
    'shuffled_forecast': 'temporal-order diagnostic: one fixed future permutation shared across members; retain t0',
    'persistence': 'deterministic forecast control: observed anchor at every future time',
    'reference_MD': 'one real MD path in observed anchor gauge; oracle reference, not a forecast',
}
SOURCES = tuple(dict.fromkeys((*ENDPOINT_SOURCES,
    'scripts/eval_ca_ordered_paths.py', 'evaluation/ca_trajectory_adapter.py',
    'evaluation/trajectory_metrics_v2.py',
    'protein_jepa_frame/training/train_stage_b1a.py',
    'protein_jepa_frame/models/stage_c/geometry.py')))


def repository_sources():
    """Snapshot loaded project dependencies as well as explicitly declared sources."""
    root = Path(__file__).resolve().parents[1]
    paths = set(SOURCES)
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if path.suffix == '.py' and relative.parts[0] in {'evaluation','protein_jepa_frame','scripts'}:
            paths.add(str(relative))
    return sorted(paths)


def panel_spec(horizons, max_windows=None):
    horizons = tuple(horizons)
    if horizons == DEFAULT_HORIZONS:
        name, complete_windows = 'standard200', 200
    elif horizons == (192,):
        name, complete_windows = 'long192_separate50', 50
    else:
        raise ValueError('fixed panels allow only horizons 1 4 16 64 or the separate horizon 192')
    count = complete_windows if max_windows is None else max_windows
    if (isinstance(count, bool) or not isinstance(count, int) or count < len(horizons)
            or count > complete_windows or count % len(horizons)):
        raise ValueError('window budget must be a complete horizon-group prefix of the fixed panel')
    return dict(name=name, horizons=horizons, max_windows=count,
                complete_panel_windows=complete_windows, full_panel=count == complete_windows,
                expected_full_panel_proteins=50)


def validate_panel(rows, panel):
    """Verify the registered prefix without filling, dropping, or replacing rows."""
    if len(rows) != panel['max_windows']:
        raise RuntimeError('incomplete ordered development panel')
    proteins = defaultdict(list)
    for row in rows:
        proteins[row['protein']].append(row)
    if len(proteins) != panel['max_windows'] // len(panel['horizons']):
        raise RuntimeError('panel must contain one anchor group per protein')
    for windows in proteins.values():
        conditions = {(w['trajectory'], w['temperature_k'], w['anchor_frame']) for w in windows}
        if len(conditions) != 1 or Counter(w['horizon'] for w in windows) != Counter(panel['horizons']):
            raise RuntimeError('protein must have one condition/anchor and exactly the registered horizons')
    if panel['full_panel']:
        expected_temperature = panel['complete_panel_windows'] // 5
        if (len(proteins) != panel['expected_full_panel_proteins']
                or Counter(w['temperature_k'] for w in rows) !=
                   Counter({t: expected_temperature for t in (320.,348.,379.,413.,450.)})):
            raise RuntimeError('full panel protein or temperature allocation differs from registration')


def _write_json_exclusive(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def _snapshot_sources(snapshot, sources):
    snapshot.mkdir(parents=True, exist_ok=False)
    for source, expected in sources.items():
        destination = snapshot/source
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Path(source).open('rb') as source_handle, destination.open('xb') as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle)
        if sha256_file(destination) != expected:
            raise RuntimeError(f'source snapshot differs from registration: {source}')


def observed_time_grid(window, model):
    """Validate only observed timestamps and requested (known) horizon metadata."""
    context = window.context
    observed = context['coords']
    frames = torch.as_tensor(window.context_frames, dtype=torch.long)
    times = torch.as_tensor(context['time_ps'], dtype=torch.float64).cpu()
    step_ps = float(model.kernel.cfg.step_ps)
    if (not math.isfinite(step_ps) or step_ps <= 0 or observed.ndim != 4
            or observed.shape[-2:] != (4, 3) or len(observed) < 2
            or times.shape != (len(observed),) or frames.shape != times.shape):
        raise ValueError('observed history requires explicit finite physical timestamps and compatible frames')
    if (not bool(torch.isfinite(times).all()) or not bool((frames[1:]-frames[:-1] == 1).all())
            or int(frames[-1]) != window.anchor_frame
            or not torch.equal(times[1:]-times[:-1], torch.full_like(times[1:], step_ps))):
        raise ValueError('observed physical time grid must be consecutive at kernel.step_ps')
    if (window.horizon < 1 or window.target_frame != window.anchor_frame + window.horizon
            or not math.isclose(window.horizon_ps, window.horizon * step_ps,
                                rel_tol=1e-9, abs_tol=1e-6)):
        raise ValueError('requested endpoint and physical horizon disagree')
    relative = torch.arange(window.horizon + 1, dtype=torch.float64) * step_ps
    return times, relative, step_ps


def temporal_controls(paths, seed):
    """Same fixed permutation for every member; no target information or RNG drift."""
    if paths.ndim != 4 or paths.shape[1] < 2:
        raise ValueError('controls require [M,T+1,L,3] including t0 and future')
    steps = paths.shape[1]-1
    reverse = [0, *range(steps, 0, -1)]
    permutation_seed = (int(seed) ^ 0x5F3759DF) % (2**31-1)
    permutation = torch.randperm(steps, generator=torch.Generator().manual_seed(permutation_seed)) + 1
    shuffled = [0, *permutation.tolist()]
    return {'reversed_forecast': paths[:, reverse], 'shuffled_forecast': paths[:, shuffled]}, {
        'reverse_time_indices': reverse, 'shuffle_time_indices': shuffled,
        'shuffle_seed': permutation_seed, 'same_permutation_across_members': True,
        'reverse_changes_order': reverse != list(range(steps+1)),
        'shuffle_changes_order': shuffled != list(range(steps+1)),
        't0_preserved': True, 'role': 'diagnostic permutations; not alternative models',
    }


def _reference(window, observed_times, relative_times, mask, anchor):
    """Read labels only after forecast construction; never shrink the observed mask."""
    future = window.future_path
    if future is None:
        raise ValueError('ordered metrics require the complete actual MD path')
    expected_shape = (window.horizon + 1, len(mask), 4, 3)
    if future['coords'].shape != expected_shape:
        raise ValueError('MD path must contain anchor and every requested future frame')
    real_mask = future['residue_mask'].detach().cpu().bool()
    if real_mask.shape != expected_shape[:2] or not bool(real_mask[:, mask].all()):
        raise ValueError('MD path is missing an observed-mask residue; no future-dependent mask selection')
    times = torch.as_tensor(future['time_ps'], dtype=torch.float64).cpu()
    if times.shape != relative_times.shape or not torch.equal(times-observed_times[-1], relative_times):
        raise ValueError('MD physical time grid differs from the continuous forecast grid')
    real = future['coords'].detach().cpu().double()[:, :, 1]
    if not bool(torch.isfinite(real[:, mask]).all()) or not torch.equal(real[0, mask], anchor[mask]):
        raise ValueError('MD path must start at the observed anchor and have finite valid CA')
    target = window.target
    if (target['coords'].shape != (1, len(mask), 4, 3)
            or not bool(target['residue_mask'].detach().cpu().bool()[:, mask].all())
            or not torch.equal(target['coords'].detach().cpu().double()[0, mask, 1], real[-1, mask])
            or not torch.equal(torch.as_tensor(target['time_ps'], dtype=torch.float64).cpu(), times[-1:])):
        raise ValueError('MD path endpoint differs from the matched target label')
    # Reference-only alignment. Forecast coordinates are never fitted here.
    aligned = torch.stack([kabsch_align(frame, anchor, mask) for frame in real])
    aligned[0] = anchor
    return real, aligned, real_mask, times


@torch.no_grad()
def score_window(model, window, *, device='cpu', n_samples=12, decode_chunk_size=64):
    if model.training or getattr(model, 'prediction_kind', None) != 'endpoint':
        raise ValueError('requires an eval-mode endpoint sampler')
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 2:
        raise ValueError('joint path U-scores require at least two independent forecast members')
    observed_times, relative_times, step_ps = observed_time_grid(window, model)
    context = window.context
    observed = context['coords'].to(device)
    mask = context['residue_mask'].to(device).all(0).bool()
    if mask.shape != (observed.shape[1],) or int(mask.sum()) < 4:
        raise ValueError('requires at least four observed-valid residues')
    if not bool(torch.isfinite(observed[:, mask]).all()):
        raise ValueError('nonfinite valid observed backbone')
    aa, ri, ci = [context[key].to(device) for key in ('aatype', 'residue_index', 'chain_id')]
    seed = _window_seed(0, window)
    # This is the only sampling call. Future coordinates/masks/timestamps/H/S
    # are neither read nor passed until its continuous paths are fully formed.
    result = sample_ca_trajectory(model, observed[None], mask[None], aa[None], ri[None], ci[None],
        torch.tensor([window.temperature_k], dtype=observed.dtype, device=device),
        window.horizon, n_samples, torch.Generator(device=device).manual_seed(seed),
        decode_chunk_size=decode_chunk_size, context_dt_ps=step_ps)
    expected = (1, n_samples, window.horizon+1, len(mask), 3)
    if (result.ca.shape != expected or result.step_ps != step_ps
            or model.last_observation_steps != [window.horizon]*n_samples):
        raise ValueError('trajectory adapter did not return the requested continuous endpoint paths')
    forecast = result.ca[0].detach().cpu().double()
    mask, ri, ci = mask.cpu(), ri.cpu(), ci.cpu()
    anchor = observed[-1, :, 1].detach().cpu().double()
    if not torch.equal(forecast[:, 0, mask], anchor[mask].expand(n_samples, -1, -1)):
        raise ValueError('forecast time zero differs from the observed anchor')
    controls, control_metadata = temporal_controls(forecast, seed)
    reference, aligned_reference, raw_reference_mask, reference_times = _reference(
        window, observed_times, relative_times, mask, anchor)
    paths = {'forecast': forecast, **controls,
        'persistence': anchor[None, None].expand(n_samples, window.horizon+1, -1, -1),
        'reference_MD': aligned_reference[None]}
    arms = {}
    for name, coordinates in paths.items():
        metrics = ordered_path_summary(coordinates, reference, anchor, mask,
            predicted_times_ps=relative_times, reference_times_ps=reference_times-observed_times[-1],
            reference_mask=mask[None].expand(window.horizon+1, -1),
            residue_index=ri, chain_id=ci, lags=LAGS, max_pairs=MAX_PAIRS)
        arms[name] = dict(role=ROLES[name], n_paths=len(coordinates), metrics=metrics)
    fingerprints = dict(
        observed=state_sha256({key: context[key] for key in (
            'coords', 'residue_mask', 'aatype', 'residue_index', 'chain_id', 'time_ps')}),
        reference=state_sha256(dict(ca=reference, residue_mask=raw_reference_mask, time_ps=reference_times)),
        forecast_CA_path=state_sha256({'ca': forecast}))
    return dict(protein=window.protein_id, trajectory=window.trajectory_id,
        temperature_k=float(window.temperature_k), anchor_frame=int(window.anchor_frame),
        target_frame=int(window.target_frame), horizon=int(window.horizon), horizon_ps=float(window.horizon_ps),
        context_frames=torch.as_tensor(window.context_frames).tolist(), valid_residues=int(mask.sum()),
        sampling_seed=seed, n_forecast_paths=n_samples, n_reference_paths=1,
        time_grid=dict(step_ps=step_ps, n_frames=window.horizon+1,
                       anchor_time_ps=float(observed_times[-1]), endpoint_time_ps=float(reference_times[-1])),
        continuous_path=dict(n_sample_calls=1, future_reencodings=0, bath_resets=0,
                             includes_sampled_S0=result.bath is not None,
                             latent_shape=list(result.latent.shape),
                             bath_shape=None if result.bath is None else list(result.bath.shape)),
        input_fingerprints=fingerprints, temporal_controls=control_metadata, arms=arms)


def _count_key(key):
    return key.startswith('n_') or key.endswith('_n_lags')


def summarize(rows):
    """Equal windows within protein then equal proteins, separately per horizon."""
    if not rows:
        raise ValueError('no ordered windows to summarize')
    groups, seen = defaultdict(list), set()
    for row in rows:
        key = tuple(row[k] for k in ('protein', 'trajectory', 'temperature_k', 'anchor_frame', 'horizon'))
        if key in seen:
            raise ValueError('duplicate ordered-path window')
        seen.add(key)
        groups[(row['horizon'], row['protein'])].append(row)
    keys = tuple(rows[0]['arms']['forecast']['metrics']['scalars'])
    counts = [key for key in keys if _count_key(key)]
    metrics = [key for key in keys if not _count_key(key)]
    for row in rows:
        for arm in ROLES:
            values = row['arms'][arm]['metrics']['scalars']
            if set(values) != set(keys) or any(v is not None and not math.isfinite(v) for v in values.values()):
                raise ValueError('nonfinite or inconsistent ordered scalar schema')
    result = {}
    for horizon in sorted({h for h, _ in groups}):
        proteins = []
        for (h, protein), windows in sorted(groups.items()):
            if h != horizon:
                continue
            arms = {}
            for arm in ROLES:
                scores = {}
                for metric in metrics:
                    vals = [w['arms'][arm]['metrics']['scalars'][metric] for w in windows]
                    vals = [v for v in vals if v is not None]
                    scores[metric] = dict(mean=sum(vals)/len(vals) if vals else None, n_windows=len(vals))
                arms[arm] = scores
            proteins.append(dict(protein=protein, n_windows=len(windows), arms=arms))
        means = {}
        for arm in ROLES:
            means[arm] = {}
            for metric in metrics:
                vals = [p['arms'][arm][metric]['mean'] for p in proteins]
                vals = [v for v in vals if v is not None]
                means[arm][metric] = dict(mean=sum(vals)/len(vals) if vals else None, n_proteins=len(vals))
        result[str(horizon)] = dict(n_proteins=len(proteins), n_windows=sum(p['n_windows'] for p in proteins),
            means=means, per_protein=proteins, count_fields_retained_per_window=counts)
    return result


def run(args):
    args.manifest = _development_manifest(args.manifest)
    panel = panel_spec(args.horizons, args.max_windows)
    if args.torch_threads < 1 or args.decode_chunk_size < 1:
        raise ValueError('thread count and decode chunk size must be positive')
    output = Path(args.out)
    registration, snapshot = output.with_suffix('.protocol.json'), output.with_suffix('.source_snapshot')
    if any(path.exists() for path in (output, registration, snapshot)):
        raise FileExistsError('refusing to overwrite an ordered-path artifact or source snapshot')
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(0)
    inputs = {args.checkpoint: sha256_file(args.checkpoint), args.manifest: sha256_file(args.manifest)}
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = load_model(args.checkpoint, checkpoint, args.device)
    if float(model.kernel.cfg.step_ps) != 1000.:
        raise ValueError('these fixed development panels require 1000 ps per stored frame')
    provenance = audit_sources(checkpoint, checkpoint['component_checkpoint'],
                              'artifacts/mdcath_manifest_full/train.jsonl', args.manifest)
    for upstream in provenance['upstream_checkpoints'].values():
        inputs[upstream['path']] = upstream['current_sha256']
    sources = {path: sha256_file(path) for path in repository_sources()}
    before_model = state_sha256(model)
    protocol = EvalProtocol(manifest=args.manifest, context_frames=8,
        horizons_frames=panel['horizons'], max_residues=96, max_anchors_per_trajectory=4,
        temperatures=(320.,348.,379.,413.,450.), seed=0,
        round_robin_proteins=True, round_robin_conditions=True)
    report = dict(task_version=TASK_VERSION, registered_utc=datetime.now(timezone.utc).isoformat(),
        dataset_role='development_validation', args=vars(args), panel=panel,
        protocol=asdict(protocol), arm_roles=ROLES, input_hashes=inputs, source_hashes=sources,
        provenance=provenance, optimizer_steps=0, model_selection=False,
        runtime=dict(torch_version=torch.__version__, metric_device='cpu', metric_dtype='float64'),
        test_opened=False, bioemu2_rerun=False, future_information_given_to_sampler=False,
        sampling=dict(seed=0, n_paths=12, model_history_frames=4 if model.history_conditioning else 1,
                      prediction_kind='endpoint', s0_mode=model.kernel.s0_mode,
                      one_continuous_rollout_per_member=True, no_future_reencoding=True, no_bath_reset=True),
        metrics=dict(arithmetic='CPU float64', registered_lags=list(LAGS), max_physical_pairs=MAX_PAIRS,
            coordinate_gauge='real path frames independently aligned to same observed CA anchor; prediction unchanged',
            shape_gauge='prediction-only alignment to observed anchor for shape diagnostics, separately labelled',
            joint_scores='whole future path U-energy scores; time zero excluded; M=1 reference has no U-score',
            aggregation='equal windows within protein then equal proteins per horizon; no CI/pass claim',
            per_path_counts='ACF/MSD computed within each path; never concatenate member boundaries',
            reference_count='one real MD path per window; no replication to inflate reference counts',
            storage='scalars, short curves, per-path diagnostics and hashes; no CA/H/S coordinate arrays'),
        limitations=['adaptive development panel, not independent confirmation',
            'single reference path supports expectation-over-windows scoring, not a per-window conditional reference ensemble',
            'short finite-path ACF/MSD are diagnostics, not stationary relaxation times or diffusion constants',
            'distance-path scores use a fixed observable map and do not identify every atomic-path distribution',
            'walk RMSF pools future frames and members; between-member spread enters and temporal order is ignored',
            'reverse/shuffle preserve the future frame bag and t0, but change time marginals and may change the endpoint; not a causal memory test',
            'manifest, consumed window tensors, checkpoints and listed source files are fingerprinted; whole MD files are not hashed',
            'no MSM/T2B, full-atom validation, BioEmu2 superiority or overall-goal completion claim'])
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_exclusive(registration, report)
    _snapshot_sources(snapshot, sources)
    rows = []
    for window in iter_windows(protocol):
        if len(rows) == panel['max_windows']:
            break
        rows.append(score_window(model, window, device=args.device, n_samples=12,
                                 decode_chunk_size=args.decode_chunk_size))
        if len(rows) % 10 == 0:
            print(f'Ordered paths {len(rows)}/{panel["max_windows"]}', flush=True)
    validate_panel(rows, panel)
    if state_sha256(model) != before_model:
        raise RuntimeError('model tensors changed during ordered evaluation')
    if set(repository_sources()) - sources.keys():
        raise RuntimeError('an unregistered project dependency was imported during evaluation')
    for path, expected in {**inputs, **sources}.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f'input/source changed during ordered evaluation: {path}')
    for split in provenance['training_splits']:
        if (sha256_file(split['manifest']) != split['train_manifest_sha256']
                or sha256_file(args.manifest) != split['val_manifest_sha256']):
            raise RuntimeError('manifest changed during ordered evaluation')
    report.update(completed_utc=datetime.now(timezone.utc).isoformat(), frozen_inputs_verified=True,
        model_tensor_sha256=before_model, n_windows=len(rows), n_proteins=len({r['protein'] for r in rows}),
        per_window=rows, by_horizon=summarize(rows))
    _write_json_exclusive(output, report)
    print(json.dumps(dict(output=str(output), panel=panel['name'], n_windows=len(rows),
                          n_proteins=report['n_proteins'])), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--manifest', default='artifacts/mdcath_manifest_full/val.jsonl')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--horizons', nargs='+', type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument('--max-windows', type=int, default=None,
                        help='Default200 for standard horizons,50 for192; smaller complete prefixes are smoke tests.')
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--decode-chunk-size', type=int, default=64)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
