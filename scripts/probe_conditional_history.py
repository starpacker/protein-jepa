"""Train-only ridge selection for incremental endpoint readout from observed H history.

Compare [residue H0, masked protein-mean H0, temperature] with those same
features plus every observed H(-j)-H0 and its protein mean. The encoder is
frozen. Each horizon has separate regressions; future coordinates supply ONLY
the endpoint label, aligned to the observed anchor and expressed in its local
N-CA-C frames. Neither future H nor a future-path mixture enters the features.

This measures generalization within the stated features and linear probe family.
It is not conditional mutual information, an information/capacity upper bound,
evidence for/against Markovianity, or a comparison with BioEmu2. Positive gains
can also reflect restrictions of the anchor-only probe; null gains do not show
that history is uninformative. Validation is development data used once after
all train-protein cross-validation choices are fixed, not a final test set.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from evaluation.data import EvalProtocol, iter_windows
from evaluation.geometry import kabsch_align
from evaluation.loaders import sha256_file
from protein_jepa_frame.models.ca_component_bundle import CAComponentBundle
from protein_jepa_frame.training.train_stage_b1a import _extract_backbone_frames
from protein_jepa_frame.utils.contracts import state_sha256
from scripts.diag_lasp_ca_ceiling import _development_manifest
from scripts.gate_ca_representation import SOURCE_PATHS as GATE_SOURCES, audit_sources


ARMS = ('anchor', 'history')
SOURCES = tuple(dict.fromkeys((*GATE_SOURCES, 'scripts/probe_conditional_history.py',
    'scripts/diag_lasp_ca_ceiling.py', 'protein_jepa_frame/training/train_stage_b1a.py',
    'protein_jepa_frame/training/train_stage_ensemble.py',
    'protein_jepa_frame/models/stage_c/geometry.py')))


def feature_arrays(h, mask, temperature_k):
    """Pure observation-only features, oldest-to-newest H in [K,L,D].

    History columns are newest-to-oldest, excluding the anchor. Protein means
    use exactly the context-valid residues, never target-derived masks.
    """
    if h.ndim != 3 or len(h) < 2 or mask.dtype != torch.bool or int(mask.sum()) < 4:
        raise ValueError('features require >=2 observed frames and >=4 valid residues')
    if not torch.isfinite(h[:, mask]).all() or not np.isfinite(temperature_k):
        raise ValueError('nonfinite observed features')
    h = h[:, mask]
    count = h.shape[1]
    means = h.mean(1, keepdim=True).expand(-1, count, -1)
    base = torch.cat((h[-1], means[-1], h.new_full((count, 1), temperature_k)), dim=-1)
    delta = torch.cat((h[:-1] - h[-1:], means[:-1] - means[-1:]), dim=-1)
    delta = delta.flip(0).permute(1, 0, 2).reshape(count, -1)
    return {'anchor': base.cpu().numpy().astype(np.float64),
            'history': torch.cat((base, delta), dim=-1).cpu().numpy().astype(np.float64)}


def endpoint_label(context, target, mask):
    """[L,3] internal endpoint displacement in each observed residue's frame."""
    rotation, anchor = _extract_backbone_frames(context[-1:])
    rotation, anchor = rotation[0], anchor[0]
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    if not torch.allclose((rotation.transpose(-1, -2) @ rotation)[mask],
                          identity.expand(int(mask.sum()), -1, -1), atol=2e-4, rtol=2e-4):
        raise ValueError('degenerate observed backbone local frame')
    aligned = kabsch_align(target[-1, :, 1], anchor, mask)
    delta = torch.einsum('lij,li->lj', rotation, aligned - anchor)
    label = delta[mask].cpu().numpy().astype(np.float64)
    if not np.isfinite(label).all():
        raise ValueError('nonfinite endpoint label')
    return label


def window_key(row):
    return (row['protein'], row['trajectory'], row['temperature_k'],
            row['anchor_frame'], row['horizon'])


@torch.no_grad()
def collect_windows(bundle, protocol, limit, device):
    rows, seen = [], set()
    cached_key, features = None, None
    for window in iter_windows(protocol):
        if len(rows) == limit:
            break
        context = window.context['coords'].to(device)
        mask = window.context['residue_mask'].to(device).all(0).bool()
        target = window.target['coords'].to(device)
        if not bool(window.target['residue_mask'].to(device)[:, mask].all()):
            raise ValueError('endpoint mask differs on context-valid residues')
        key = (window.protein_id, window.trajectory_id, float(window.temperature_k),
               tuple(window.context_frames.tolist()))
        if key != cached_key:
            count = len(context)
            aa, ri, ci = [window.context[k][None].to(device) for k in
                          ('aatype', 'residue_index', 'chain_id')]
            expand = lambda value: value.expand(count, *value.shape[1:])
            # Only observed coordinates are ever passed to the encoder.
            h = bundle.encode(context, expand(mask[None]), expand(aa), expand(ri), expand(ci))
            features = feature_arrays(h, mask, window.temperature_k)
            cached_key = key
        row = dict(protein=window.protein_id, trajectory=window.trajectory_id,
                   temperature_k=float(window.temperature_k), anchor_frame=int(window.anchor_frame),
                   target_frame=int(window.target_frame), horizon=int(window.horizon),
                   horizon_ps=float(window.horizon_ps), context_frames=window.context_frames.tolist(),
                   valid_residues=int(mask.sum()), features=features,
                   target=endpoint_label(context, target, mask))
        if window_key(row) in seen:
            raise ValueError('duplicate probe window')
        seen.add(window_key(row))
        rows.append(row)
        if len(rows) % 20 == 0:
            print(f'Encoded {len(rows)}/{limit} windows from {Path(protocol.manifest).name}', flush=True)
    if len(rows) != limit or set(r['horizon'] for r in rows) != set(protocol.horizons_frames):
        raise RuntimeError('incomplete probe panel')
    return rows


def pack_rows(rows, arm):
    """Fit weights give equal mass to proteins, their windows, then residues."""
    if not rows or arm not in ARMS:
        raise ValueError('empty/unknown probe rows')
    counts = Counter(row['protein'] for row in rows)
    seen, xs, ys, weights = set(), [], [], []
    for row in rows:
        key = window_key(row)
        if key in seen:
            raise ValueError('duplicate probe window')
        seen.add(key)
        x, y = row['features'][arm], row['target']
        if x.ndim != 2 or y.shape != (len(x), 3) or not len(x):
            raise ValueError('probe arrays must be [residue,features] and [residue,3]')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('nonfinite probe arrays')
        xs.append(x)
        ys.append(y)
        weights.append(np.full(len(x), 1 / (len(counts) * counts[row['protein']] * len(x))))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)


@dataclass
class RidgeProbe:
    mean: np.ndarray
    scale: np.ndarray
    target_mean: np.ndarray
    coefficients: np.ndarray
    regularization: float

    def predict(self, x):
        result = ((x - self.mean) / self.scale) @ self.coefficients + self.target_mean
        if not np.isfinite(result).all():
            raise FloatingPointError('nonfinite ridge prediction')
        return result


def ridge_path(x, y, weights, lambdas):
    """Weighted standardized ridge with an unpenalized intercept, fitted once.

    Objective: weighted mean squared residual per output plus lambda*||W||^2.
    All means/scales and the eigensystem use ONLY the supplied fitting rows.
    """
    x, y, weights = [np.asarray(value, dtype=np.float64) for value in (x, y, weights)]
    lambdas = sorted(set(float(value) for value in lambdas))
    if (x.ndim != 2 or y.shape != (len(x), 3) or weights.shape != (len(x),)
            or not len(x) or not lambdas or any(not np.isfinite(v) or v <= 0 for v in lambdas)
            or not all(np.isfinite(value).all() for value in (x, y, weights))
            or np.any(weights <= 0)):
        raise ValueError('invalid ridge data or regularization')
    weights = weights / weights.sum()
    mean, ym = weights @ x, weights @ y
    xc = x - mean
    scale = np.sqrt(weights @ xc ** 2)
    scale = np.where(scale > 1e-8, scale, 1.)
    xc = xc / scale
    gram = xc.T @ (xc * weights[:, None])
    eigenvalues, vectors = np.linalg.eigh((gram + gram.T) / 2)
    rhs = vectors.T @ (xc.T @ ((y - ym) * weights[:, None]))
    return {lam: RidgeProbe(mean, scale, ym,
                           vectors @ (rhs / (np.maximum(eigenvalues, 0)[:, None] + lam)), lam)
            for lam in lambdas}


def protein_errors(rows, predictor, arm):
    grouped = defaultdict(list)
    for row in rows:
        pred = np.zeros_like(row['target']) if predictor is None else predictor.predict(row['features'][arm])
        mse = float(np.mean(np.sum((pred - row['target']) ** 2, axis=-1)))
        grouped[row['protein']].append((mse, np.sqrt(mse)))
    return {protein: dict(mse=float(np.mean([e[0] for e in errors])),
                          rmsd=float(np.mean([e[1] for e in errors])), windows=len(errors))
            for protein, errors in sorted(grouped.items())}


def fit_probes(train_rows, *, lambdas, folds=5, seed=20261101):
    """Select each arm/horizon lambda using only out-of-fold TRAIN proteins."""
    proteins = sorted({row['protein'] for row in train_rows})
    if not 2 <= folds <= len(proteins):
        raise ValueError('group CV needs at least one distinct training protein per fold')
    rng = np.random.default_rng(seed)
    rng.shuffle(proteins)
    assignments = {protein: index % folds for index, protein in enumerate(proteins)}
    lambdas = sorted(set(float(value) for value in lambdas))
    fitted, selection = {}, {}
    for horizon in sorted({row['horizon'] for row in train_rows}):
        rows = [row for row in train_rows if row['horizon'] == horizon]
        if {r['protein'] for r in rows} != set(proteins):
            raise ValueError('every training protein must have every probe horizon')
        fitted[horizon], selection[str(horizon)] = {}, {}
        for arm in ARMS:
            scores = {lam: [] for lam in lambdas}
            fold_audits = []
            for fold in range(folds):
                fitting = [r for r in rows if assignments[r['protein']] != fold]
                heldout = [r for r in rows if assignments[r['protein']] == fold]
                path = ridge_path(*pack_rows(fitting, arm), lambdas)
                for lam, model in path.items():
                    scores[lam].extend(p['mse'] for p in protein_errors(heldout, model, arm).values())
                fold_audits.append(dict(fold=fold, fit_proteins=sorted({r['protein'] for r in fitting}),
                                        heldout_proteins=sorted({r['protein'] for r in heldout})))
            cv = {lam: float(np.mean(values)) for lam, values in scores.items()}
            # Prefer stronger regularization only on an exact numerical tie.
            chosen = min(lambdas, key=lambda lam: (cv[lam], -lam))
            fitted[horizon][arm] = ridge_path(*pack_rows(rows, arm), [chosen])[chosen]
            selection[str(horizon)][arm] = dict(selected_lambda=chosen,
                cv_equal_protein_mse={str(lam): value for lam, value in cv.items()},
                selected_grid_boundary=chosen in (min(lambdas), max(lambdas)),
                folds=fold_audits, feature_count=rows[0]['features'][arm].shape[1])
            print(f'Fitted horizon={horizon} arm={arm} lambda={chosen:g} train-CV MSE={cv[chosen]:.6f}', flush=True)
    return fitted, dict(protein_fold=assignments, by_horizon=selection,
                        selection_metric='out-of-fold equal-protein mean endpoint squared-vector error (A^2)')


def evaluate_probes(fitted, val_rows, *, bootstrap_reps=10000, seed=20261101):
    """One locked validation evaluation; paired bootstrap resamples proteins."""
    if bootstrap_reps < 1:
        raise ValueError('bootstrap_reps must be positive')
    if set(fitted) != {row['horizon'] for row in val_rows}:
        raise ValueError('validation and fitted horizons differ')
    report = {}
    for horizon, models in sorted(fitted.items()):
        rows = [row for row in val_rows if row['horizon'] == horizon]
        for arm in ARMS:
            pack_rows(rows, arm)  # Validate uniqueness and shapes before reporting.
        errors = {arm: protein_errors(rows, model, arm) for arm, model in models.items()}
        errors['persistence'] = protein_errors(rows, None, 'anchor')
        proteins = sorted(errors['anchor'])
        if len(proteins) < 2:
            raise ValueError('validation bootstrap needs at least two proteins')
        rng = np.random.default_rng(np.random.SeedSequence([seed, horizon]))
        index = rng.integers(0, len(proteins), size=(bootstrap_reps, len(proteins)))
        def estimate(values):
            return dict(mean=float(values.mean()), ci95=np.quantile(values[index].mean(1), [.025, .975]).tolist())
        arrays = {arm: {metric: np.array([values[p][metric] for p in proteins]) for metric in ('mse', 'rmsd')}
                  for arm, values in errors.items()}
        base, history = arrays['anchor']['mse'], arrays['history']['mse']
        base_boot = base[index].mean(1)
        relative = None
        if base.mean() > 1e-12 and np.all(base_boot > 1e-12):
            relative = dict(mean=float(1 - history.mean() / base.mean()),
                            ci95=np.quantile(1 - history[index].mean(1) / base_boot, [.025, .975]).tolist())
        report[str(horizon)] = dict(n_proteins=len(proteins), n_windows=len(rows),
            metrics={arm: {m: estimate(v) for m, v in values.items()} for arm, values in arrays.items()},
            paired_history_minus_anchor={m: estimate(arrays['history'][m] - arrays['anchor'][m])
                                         for m in ('mse', 'rmsd')},
            fractional_mse_reduction=relative, per_protein=errors)
    return report


def _metadata(rows):
    return [{key: value for key, value in row.items() if key not in ('features', 'target')} for row in rows]


def run(args):
    val_manifest = _development_manifest(args.val_manifest)
    approved_train = Path(__file__).resolve().parents[1] / 'artifacts/mdcath_manifest_full/train.jsonl'
    if Path(args.train_manifest).resolve() != approved_train.resolve():
        raise ValueError('probe requires the approved development train.jsonl manifest')
    if (len(set(args.horizons)) != len(args.horizons) or args.history_frames < 2
            or args.torch_threads < 1 or args.encoder_batch_device not in ('cpu', 'cuda:0', 'cuda:1')):
        raise ValueError('invalid probe protocol or encoder device')
    if any(n < len(args.horizons) or n % len(args.horizons) for n in (args.train_windows, args.val_windows)):
        raise ValueError('window budgets must contain complete horizon groups')
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.encoder_batch_device)
    checkpoint_hash = sha256_file(args.component)
    checkpoint = torch.load(args.component, map_location='cpu', weights_only=False)
    component = checkpoint.get('component_checkpoint', checkpoint)
    if component.get('checkpoint_kind') != 'ca_component_probe':
        raise ValueError('requires a CA component or dynamics checkpoint embedding one')
    provenance = audit_sources(checkpoint, component, args.train_manifest, val_manifest)
    source_hashes = {path: sha256_file(path) for path in SOURCES}
    def protocol(manifest):
        return EvalProtocol(manifest=str(Path(manifest).resolve()), context_frames=args.history_frames,
            horizons_frames=tuple(args.horizons), frame_stride=1, max_residues=args.max_residues,
            max_anchors_per_trajectory=4, temperatures=(320., 348., 379., 413., 450.),
            seed=args.seed, round_robin_proteins=True, round_robin_conditions=True)
    train_protocol, val_protocol = protocol(args.train_manifest), protocol(val_manifest)
    with tempfile.TemporaryDirectory(prefix='conditional_history_probe_') as temp:
        path = Path(temp) / 'component.pt'
        torch.save(component, path)
        bundle = CAComponentBundle(path, device).eval()
        encoder_hash = state_sha256(bundle.encoder)
        if encoder_hash != state_sha256(component['encoder']):
            raise ValueError('loaded encoder differs from selected component')
        train_rows = collect_windows(bundle, train_protocol, args.train_windows, device)
        fitted, selection = fit_probes(train_rows, lambdas=args.lambdas, folds=args.folds, seed=args.cv_seed)
        # No validation trajectory has been read before all choices are fixed.
        val_rows = collect_windows(bundle, val_protocol, args.val_windows, device)
        train_proteins, val_proteins = [{r['protein'] for r in rows} for rows in (train_rows, val_rows)]
        if train_proteins & val_proteins:
            raise ValueError('probe train/validation protein overlap')
        if state_sha256(bundle.encoder) != encoder_hash:
            raise RuntimeError('frozen encoder changed during probe')
    frame_intervals = {round(r['horizon_ps'] / r['horizon'], 6) for r in train_rows + val_rows}
    if len(frame_intervals) != 1:
        raise ValueError('one probe comparison requires a common physical frame interval')
    results = evaluate_probes(fitted, val_rows, bootstrap_reps=args.bootstrap_reps, seed=args.bootstrap_seed)
    if sha256_file(args.component) != checkpoint_hash or any(sha256_file(p) != h for p, h in source_hashes.items()):
        raise RuntimeError('probe checkpoint or source changed during run')
    for audit in provenance['training_splits']:
        if sha256_file(audit['manifest']) != audit['train_manifest_sha256']:
            raise RuntimeError('training manifest changed during probe')
    if sha256_file(val_manifest) != provenance['training_splits'][0]['val_manifest_sha256']:
        raise RuntimeError('validation manifest changed during probe')
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_path = output.with_suffix('.probes.npz')
    arrays = {}
    for horizon, arms in fitted.items():
        for arm, model in arms.items():
            for key, value in asdict(model).items():
                arrays[f'h{horizon}_{arm}_{key}'] = value
    np.savez_compressed(model_path, **arrays)
    report = dict(schema_version=1, purpose='incremental_endpoint_readout_within_specified_probe_family',
        interpretation='neither an information upper bound nor evidence of absence/presence of physical memory',
        dataset_role='development_validation', test_opened=False, encoder_optimizer_steps=0,
        checkpoint=args.component, checkpoint_sha256=checkpoint_hash, encoder_tensor_sha256=encoder_hash,
        provenance=provenance, source_hashes=source_hashes, args=vars(args),
        protocols={'train': asdict(train_protocol), 'validation': asdict(val_protocol)},
        n_train_proteins=len(train_proteins), n_val_proteins=len(val_proteins), protein_overlap=0,
        frame_interval_ps=next(iter(frame_intervals)), train_selection=selection,
        feature_semantics=dict(anchor='per-residue H0, context-mask protein-mean H0, temperature in kelvin',
            history='anchor columns plus residue and protein-mean H(-j)-H0, newest-to-oldest observed frames',
            target='single future CA endpoint aligned to observed anchor, then anchor N-CA-C local displacement',
            preprocessing='feature weighted mean/SD and target mean fitted separately within each training fold; final refit on train only',
            weighting='equal protein, equal windows within protein, equal valid residues within window',
            selection='separate ridge lambda per arm/horizon; grouped protein CV entirely within train',
            confidence='equal-protein paired cluster bootstrap; marginal two-sided CI95, conditional on fitted models',
            limitation='finite selected features; linear readout; validation only once per invocation; no physical-memory or BioEmu2 superiority claim'),
        train_windows=_metadata(train_rows), validation_windows=_metadata(val_rows), by_horizon=results,
        fitted_probes=dict(path=str(model_path), sha256=sha256_file(model_path)))
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(output), n_train_proteins=len(train_proteins), n_val_proteins=len(val_proteins),
                         fractional_mse_reduction={h: r['fractional_mse_reduction'] for h, r in results.items()}), indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--train-manifest', default='artifacts/mdcath_manifest_full/train.jsonl')
    parser.add_argument('--val-manifest', default='artifacts/mdcath_manifest_full/val.jsonl')
    parser.add_argument('--encoder-batch-device', default='cpu')
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--train-windows', type=int, default=200)
    parser.add_argument('--val-windows', type=int, default=200)
    parser.add_argument('--max-residues', type=int, default=96)
    parser.add_argument('--history-frames', type=int, default=8)
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 4, 16, 64])
    parser.add_argument('--lambdas', type=float, nargs='+', default=[1e-5, 1e-3, .1, 1., 10., 100., 1000., 10000.])
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cv-seed', type=int, default=20261101)
    parser.add_argument('--bootstrap-seed', type=int, default=20261101)
    parser.add_argument('--bootstrap-reps', type=int, default=10000)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
