"""Serialization-only evaluation correction; frozen training and statistics unchanged."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json

import torch

from run_scripts import run_backbone_transport_pilot as frozen
from protein_jepa_frame.utils.contracts import state_sha256

base, ev, stats = frozen.base, frozen.ev, frozen.stats
REPO = frozen.REPO
VERSION = 'backbone_local_frame_transport_evaluation_serialization_v2'
DOCUMENT = 'docs/BACKBONE_TRANSPORT_EVAL_SERIALIZATION_V2_20261101.md'
FRAME_BASIS, TRAIN, VAL = frozen.FRAME_BASIS, frozen.TRAIN, frozen.VAL
INITIAL, CONTROL, OLD_EVAL = frozen.INITIAL, frozen.CONTROL, frozen.OLD_EVAL
PINNED, CONTRASTS = frozen.PINNED, frozen.CONTRASTS
require, plain = frozen.require, frozen.plain
load_checkpoint, validate_match = frozen.load_checkpoint, frozen.validate_match
transport, transport_audit = frozen.transport, frozen.transport_audit
decode, describe = frozen.decode, frozen.describe
SOURCES = tuple(sorted(set(frozen.SOURCES + (
    DOCUMENT, 'scripts/eval_backbone_transport_pilot.py',
    'tests/test_backbone_transport_evaluation.py',
))))
DEFAULT_OUTPUT = 'results/hs_joint_20261101/backbone_transport_v1/evaluation200_v2.json'


def canonical_json(value):
    """Only JSON's tuple/list serialization normalization; rejects NaN/Inf."""
    return json.loads(json.dumps(value, allow_nan=False))


def require_canonical_json_equal(actual, expected, message):
    """Exact equality after JSON round-trip; no numerical/hash tolerance."""
    require(canonical_json(actual) == canonical_json(expected), message)


def register(args, tracked, **extras):
    sources = {p: base.sha(tracked.track(p)) for p in SOURCES}
    out = base.local(args.out)
    if args.mode == 'train':
        require(not out.exists(), 'refusing existing training directory')
        out.mkdir(parents=True)
        protocol_path, snapshot = out/'protocol.json', out/'source_snapshot'
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        protocol_path, snapshot = out.with_suffix('.protocol.json'), out.with_suffix('.source_snapshot')
        require(not any(p.exists() for p in (out, protocol_path, snapshot)), 'refusing existing output')
    snapshot.mkdir()
    for path, digest in sources.items():
        dest = snapshot/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as stream:
            stream.write((REPO/path).read_bytes())
        require(base.sha(dest) == digest, 'source snapshot differs')
    protocol = dict(schema_version=VERSION, mode=args.mode,
        registered_utc=datetime.now(timezone.utc).isoformat(), frame_basis=FRAME_BASIS,
        seed=42, optimizer='AdamW', learning_rate=1e-4, final_learning_rate=1e-5,
        warmup_steps=25, weight_decay=1e-4, gradient_clip=1., geometry_lambda=0.,
        train_cache=TRAIN, initial=INITIAL, observed_control=CONTROL,
        checkpoint_selection='fixed latest; no validation selection',
        validation_used_for_training=False, encoder_CA_dynamics_frozen=True,
        target_backbone_or_mask_given_to_transport=False,
        test_opened=False, bioemu2_rerun=False, formal=True,
        coordinate_objective='mean squared Euclidean N/C/O error; fixed CA excluded; no realignment',
        contrast_pairs=CONTRASTS, inferential_metrics=stats.METRICS,
        groups=stats.GROUPS, bootstrap=dict(n=10000, seed=20261101, unit='protein',
            interval='percentile95, marginal exploratory, no multiplicity adjustment'),
        parameter_precision='float32; no AMP; TF32 disabled', transport_precision='float64 SVD, cast seed to float32',
        runtime=dict(device=args.device, torch_version=torch.__version__, torch_threads=torch.get_num_threads()),
        **extras)
    protocol['serialization_correction'] = dict(document=DOCUMENT, historical_comparison='strict equality after json.loads(json.dumps(value, allow_nan=False))', numerical_tolerance=None, scientific_protocol_version=frozen.VERSION)
    info = dict(protocol=protocol, source_hashes=sources, input_hashes=dict(tracked.hashes))
    base.write_json(protocol_path, info)
    return out, info


def evaluate(args):
    require(args.checkpoint is not None, 'evaluation requires transport checkpoint')
    tracked = base.Inputs()
    cache = torch.load(tracked.track(VAL, PINNED[VAL]), map_location='cpu', weights_only=False)
    ev.validate_cache(cache)
    tracked.declared(cache)
    saved, heads = {}, {}
    for name, path in (('init', INITIAL), ('coordinate_observed', CONTROL), ('coordinate_transport', args.checkpoint)):
        heads[name], saved[name] = load_checkpoint(path, tracked, args.device, expected=PINNED.get(path))
        heads[name].eval().requires_grad_(False)
    validate_match(saved['init'], saved['coordinate_observed'], saved['coordinate_transport'])
    previous = json.loads(tracked.track(OLD_EVAL, PINNED[OLD_EVAL]).read_text())
    old_rows = {stats.window_key(r): r for r in previous['per_window']}
    before = {k: state_sha256(h) for k, h in heads.items()}
    out, info = register(args, tracked, dataset_role='development_validation', n_windows=200,
        n_forecast_samples=12, optimizer_steps=0, checkpoint=str(base.local(args.checkpoint).relative_to(REPO)),
        prior_evaluation=OLD_EVAL, observed_controls_must_match_prior=True)
    rows = []
    matches = 0
    for index, row in enumerate(cache['rows']):
        ev.validate_row(row)
        routes, audits = {}, {}
        for variant in ('oracle', 'forecast'):
            ca, h = row[variant+'_CA'], row[variant+'_H']
            moved = transport(row, ca)
            audits[variant] = transport_audit(moved, row['observed_mask'])
            for name, head, frames in (
                ('init_observed', heads['init'], row['observed_R']),
                ('init_transport', heads['init'], moved['frame_seed']),
                ('coordinate_observed', heads['coordinate_observed'], row['observed_R']),
                ('coordinate_transport', heads['coordinate_transport'], moved['frame_seed'])):
                bb = decode(head, row, h, ca, frames)
                routes[f'{name}/{variant}'] = ev.score_backbone(bb, row,
                    role='future-derived oracle readout' if variant == 'oracle' else 'observed-only cached forecast', fixed_ca=ca)
        routes['persistence_MD'] = ev.score_backbone(row['observed_backbone'], row, role='observed MD persistence')
        routes['target_MD'] = ev.score_backbone(row['target_backbone'], row, role='aligned target MD reference')
        old = old_rows[ev.window_key(row)]
        for name, old_name in (('init_observed', 'init'), ('coordinate_observed', 'coordinate')):
            for variant in ('oracle', 'forecast'):
                a, b = routes[f'{name}/{variant}'], old['routes'][f'{old_name}/{variant}']
                for field in ('backbone_sha256', 'error', 'legacy_geometry', 'explicit_geometry', 'fixed_ca_sha256'):
                    require_canonical_json_equal(a[field], b[field], f'observed control differs from frozen prior: {field}')
                matches += 1
        rows.append(dict(**row['metadata'], topology=ev.topology_audit(row), routes=routes, transport=audits))
        if (index+1) % 10 == 0:
            print(f'transport evaluation windows={index+1}/200', flush=True)
    require(all(state_sha256(h) == before[k] for k, h in heads.items()), 'evaluation changed head')
    require(all(p.grad is None for h in heads.values() for p in h.parameters()), 'evaluation created gradients')
    contrasts = {}
    for group in stats.GROUPS:
        selected = stats.selected(rows, group)
        contrasts[group] = {}
        for candidate, reference in CONTRASTS:
            for variant in ('oracle', 'forecast'):
                label = f'{candidate}-{reference}/{variant}'
                contrasts[group][label] = {metric: stats.paired(selected, f'{candidate}/{variant}',
                    f'{reference}/{variant}', metric) for metric in stats.METRICS}
    tracked.verify()
    result = dict(**info, completed_utc=datetime.now(timezone.utc).isoformat(),
        frozen_inputs_verified=True, n_windows=200, n_proteins=50, prior_control_matches=matches,
        per_window=rows, descriptive=describe(rows), contrasts=contrasts)
    base.write_json(out, result)
    print(json.dumps(dict(output=str(out), sha256=base.sha(out), prior_control_matches=matches)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('evaluate',), default='evaluate')
    parser.add_argument('--out', default=DEFAULT_OUTPUT)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    base.seed_all()
    evaluate(args)


if __name__ == '__main__':
    main()
