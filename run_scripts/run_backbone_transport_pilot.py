"""Registered parameter-free frame transport diagnosis, training and evaluation.

Historical runners, heads, metrics and cache contracts are reused unchanged.
Only the supplied orientation seed differs. Every output is exclusive.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import torch

from run_scripts import run_backbone_readout_pilot as base
from scripts import eval_backbone_readout_pilot as ev
from scripts import summarize_backbone_readout_pilot as stats
from protein_jepa_frame.models.backbone_frame_transport_v1 import transport_observed_frames
from protein_jepa_frame.models.backbone_readout_v2 import BackboneReadoutConfig, BackboneReadoutV2
from protein_jepa_frame.models.stage_c.structure_decoder import reconstruct_atom37_from_frames
from protein_jepa_frame.utils.contracts import state_sha256
from openfold.data.data_transforms import atom37_to_torsion_angles

REPO = base.REPO
VERSION = 'backbone_local_frame_transport_pilot_v1'
FRAME_BASIS = 'observed_CA_to_current_CA_explicit_PSF_2hop_centered_proper_Kabsch_v1'
DOCUMENT = 'docs/BACKBONE_TRANSPORT_PILOT_V1_20261101.md'
TRAIN = 'results/hs_joint_20261101/backbone_repair/readout_train500_v2.pt'
VAL = 'results/hs_joint_20261101/backbone_repair/readout_val200_v2.pt'
INITIAL = 'artifacts/hs_joint_20261101/backbone_repair_v2/initial.pt'
CONTROL = 'artifacts/hs_joint_20261101/backbone_repair_v2/coordinate/checkpoint_latest.pt'
OLD_EVAL = 'results/hs_joint_20261101/backbone_repair_v2/evaluation200.json'
PINNED = {
    TRAIN: 'd48faeb6e7385fa0b37a1daf440bf4708ed3472d83e2073b8a44d9073a216718',
    VAL: 'd1a04f5861badc004f39d9496e3f245b6c32739a8d2deed5bb74840eaf605fdc',
    INITIAL: '501e9c55fd586f07962ea793371ee0ac527bf0f606517ea92aedf64c56aecfe2',
    CONTROL: 'af611531bc6ea7c359346e92382f8215b2c1e4331512a7efb12e5719b5f55ace',
    OLD_EVAL: '1c2a2e99c54e1c16f0fc09fad34c9afe09b01e93825271223a41c930422fb762',
}
SOURCES = tuple(sorted(set(base.SOURCES + ev.SOURCES + (
    DOCUMENT, 'run_scripts/run_backbone_transport_pilot.py',
    'protein_jepa_frame/models/backbone_frame_transport_v1.py',
    'scripts/summarize_backbone_readout_pilot.py',
))))
CONTRASTS = (('coordinate_transport', 'coordinate_observed'),
             ('init_transport', 'init_observed'),
             ('coordinate_transport', 'init_transport'))
require, plain = base.require, base.plain


def transport(row, ca):
    """No future atom/label/mask dependency; oracle CA is explicitly diagnostic."""
    return transport_observed_frames(row['observed_backbone'][..., 1, :], ca,
        row['observed_R'], residue_mask=row['observed_mask'],
        peptide_edges=row['peptide_edges'], geometry_row_ids=row['geometry_row_ids'])


def transport_audit(value, mask):
    audit = value['audit']
    result = {k: plain(v) for k, v in audit.items()}
    result['valid_residues_per_member'] = int(mask.sum())
    result['frame_seed_sha256'] = state_sha256({'frame_seed': value['frame_seed']})
    return result


def transported_losses(head, row):
    moved = transport(row, row['oracle_CA'])
    modified = dict(row, observed_R=moved['frame_seed'])
    return (*base.actual_losses(head, modified), moved)


def load_checkpoint(path, tracked, device, *, expected=None):
    ck = torch.load(tracked.track(path, expected), map_location='cpu', weights_only=False)
    tracked.declared(ck)
    require(ck['frozen_inputs_verified'] and ck['head_tensor_sha256'] == state_sha256(ck['head']),
            'checkpoint freeze/hash mismatch')
    with torch.random.fork_rng(devices=[]):
        head = BackboneReadoutV2(BackboneReadoutConfig(**ck['head_config']))
    head.load_state_dict(ck['head'], strict=True)
    head.initialization = ck['initialization']
    return head.to(device), ck


def validate_match(initial, control, transport_ck=None):
    require(initial['step'] == 0 and control['step'] == 500 and control['arm'] == 'coordinate',
            'wrong initial or coordinate control')
    require(control['initial_head_tensor_sha256'] == initial['head_tensor_sha256'], 'control init mismatch')
    cp = control['protocol']
    expected = dict(seed=42, steps=500, optimizer_steps=500, optimizer='AdamW',
        learning_rate=1e-4, final_learning_rate=1e-5, warmup_steps=25,
        weight_decay=1e-4, gradient_clip=1., geometry_lambda=0.,
        train_cache=TRAIN, initial=INITIAL, validation_used_for_training=False,
        encoder_CA_dynamics_frozen=True, checkpoint_selection='fixed latest; no validation selection')
    require(all(cp.get(k) == v for k, v in expected.items()), 'observed control training protocol differs')
    if transport_ck is not None:
        p = transport_ck['protocol']
        require(transport_ck['step'] == 500 and transport_ck['arm'] == 'coordinate_transport'
                and transport_ck['frame_basis'] == FRAME_BASIS
                and p['schema_version'] == VERSION and p['mode'] == 'train'
                and p['frame_basis'] == FRAME_BASIS, 'transport checkpoint contract differs')
        require(all(p.get(k) == v for k, v in expected.items()), 'transport training differs beyond frame seed')
        require(transport_ck['initial_head_tensor_sha256'] == initial['head_tensor_sha256'], 'transport init mismatch')
        require(transport_ck['head_tensor_sha256'] != initial['head_tensor_sha256']
                and transport_ck['parameter_delta_l2'] > 0, 'transport training did not update')
        for ck in (control, transport_ck):
            require(ck['head_config'] == initial['head_config'] and ck['initialization'] == initial['initialization'],
                    'architecture/weight initialization differs')
            for path in (TRAIN, INITIAL):
                require(ck['input_hashes'].get(path) == PINNED[path], 'trained cache/initial hash mismatch')


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
    info = dict(protocol=protocol, source_hashes=sources, input_hashes=dict(tracked.hashes))
    base.write_json(protocol_path, info)
    return out, info


def observed_psi(row):
    bb = row['observed_backbone'].double()
    atoms = bb.new_zeros((*bb.shape[:-2], 37, 3))
    mask = bb.new_zeros((*bb.shape[:-2], 37))
    atoms[..., [0, 1, 2, 4], :] = bb
    mask[..., [0, 1, 2, 4]] = (row['atom_mask'] & row['observed_mask'][:, None]).double()
    parsed = atom37_to_torsion_angles()(dict(aatype=row['chemical_aatype'][None],
        all_atom_positions=atoms, all_atom_mask=mask))
    angles = parsed['torsion_angles_sin_cos'][..., 2, :]
    return torch.atan2(angles[..., 0], angles[..., 1]).float()


@torch.no_grad()
def seed_backbone(row, frames, ca):
    torsions = ca.new_zeros((*ca.shape[:-1], 7))
    torsions[..., 1] = observed_psi(row)
    atoms, _ = reconstruct_atom37_from_frames(frames, ca, row['chemical_aatype'][None], torsions)
    return torch.stack((atoms[..., 0, :], ca, atoms[..., 2, :], atoms[..., 4, :]), dim=-2)


@torch.no_grad()
def decode(head, row, h, ca, frames):
    device = next(head.parameters()).device
    n = len(h)
    output = head(h.to(device), frames.expand(n, -1, -1, -1).to(device), ca.to(device),
        row['embedding_aatype'].expand(n, -1).to(device),
        row['chemical_aatype'].expand(n, -1).to(device), row['observed_mask'].expand(n, -1).to(device))
    require(bool(output['atom_mask'].cpu()[:, row['atom_mask'] & row['observed_mask'][:, None]].all()),
            'head omits observed atoms')
    return output['backbone'].cpu()


def describe(rows):
    result = {}
    for group in stats.GROUPS:
        selected = stats.selected(rows, group)
        accumulated = defaultdict(list)
        for row in selected:
            for route, values in row['routes'].items():
                for metric, value in stats.descriptive(values).items():
                    if value is not None:
                        accumulated[route, metric, row['protein']].append(value)
        grouped = defaultdict(list)
        for (route, metric, _), values in accumulated.items():
            grouped[route, metric].append(float(np.mean(values)))
        result[group] = {}
        for (route, metric), values in grouped.items():
            result[group].setdefault(route, {})[metric] = dict(mean=float(np.mean(values)), n_proteins=len(values))
    return result


def diagnose(args):
    tracked = base.Inputs()
    tracked.track(TRAIN, PINNED[TRAIN])
    cache = base.load_training_cache(TRAIN, tracked)
    initial, init_ck = load_checkpoint(INITIAL, tracked, args.device, expected=PINNED[INITIAL])
    control, ctrl_ck = load_checkpoint(CONTROL, tracked, args.device, expected=PINNED[CONTROL])
    validate_match(init_ck, ctrl_ck)
    heads = dict(init=initial.eval().requires_grad_(False), coordinate=control.eval().requires_grad_(False))
    before = {k: state_sha256(h) for k, h in heads.items()}
    out, info = register(args, tracked, dataset_role='training_diagnostic', n_windows=32, optimizer_steps=0)
    rows = []
    for row in cache['rows'][:32]:
        ca = row['oracle_CA']
        moved = transport(row, ca)
        routes = {}
        for basis, frames in (('observed', row['observed_R']), ('transport', moved['frame_seed'])):
            for name, head in heads.items():
                bb = decode(head, row, row['oracle_H'], ca, frames)
                routes[f'{name}_{basis}/oracle'] = ev.score_backbone(bb, row, role='training oracle diagnostic', fixed_ca=ca)
            routes[f'seed_{basis}/oracle'] = ev.score_backbone(seed_backbone(row, frames, ca), row,
                role='training seed-only prior using observed O-derived psi', fixed_ca=ca)
        rows.append(dict(**row['metadata'], routes=routes,
            transport=transport_audit(moved, row['observed_mask'])))
    require(all(state_sha256(head) == before[name] for name, head in heads.items()), 'diagnostic changed head')
    tracked.verify()
    base.write_json(out, dict(**info, completed_utc=datetime.now(timezone.utc).isoformat(),
        frozen_inputs_verified=True, per_window=rows, descriptive=describe(rows)))
    print(json.dumps(dict(output=str(out), n_windows=32, sha256=base.sha(out))), flush=True)


def train(args):
    tracked = base.Inputs()
    tracked.track(TRAIN, PINNED[TRAIN])
    cache = base.load_training_cache(TRAIN, tracked)
    tracked.track(INITIAL, PINNED[INITIAL])
    head, initial = base.load_head(INITIAL, tracked)
    control = torch.load(tracked.track(CONTROL, PINNED[CONTROL]), map_location='cpu', weights_only=False)
    tracked.declared(control)
    validate_match(initial, control)
    head = head.to(args.device)
    before = state_sha256(head)
    initial_tensors = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    out, info = register(args, tracked, arm='coordinate_transport', steps=500,
        optimizer_steps=500, train_row_order='cache order, one row per update, one pass')
    params = [p for p in head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    base.seed_all()
    head.train()
    with (out/'metrics.jsonl').open('x') as stream:
        for step, row in enumerate(cache['rows'], 1):
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr'] = base.lr_at(step, 500)
            coord, geom, details, moved = transported_losses(head, row)
            objective = coord + 0. * geom
            objective.backward()
            require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params), 'missing/nonfinite gradient')
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            record = dict(step=step, metadata=row['metadata'], lr=optimizer.param_groups[0]['lr'],
                coordinate_loss=float(coord.detach()), weighted_geometry_loss=float(geom.detach()),
                geometry_losses=plain(details['losses']), geometry_counts=plain(details['counts']),
                objective=float(objective.detach()), gradient_norm_before_clip=float(norm),
                transport=transport_audit(moved, row['observed_mask']))
            stream.write(json.dumps(record, allow_nan=False)+'\n')
            stream.flush()
            if step % 25 == 0:
                print(json.dumps(dict(step=step, coordinate_loss=record['coordinate_loss'],
                    geometry_losses=record['geometry_losses'])), flush=True)
    require(all(torch.isfinite(v).all() for v in head.state_dict().values()), 'nonfinite final head')
    tracked.verify()
    final = state_sha256(head)
    delta = math.sqrt(sum(float((v.detach().cpu().double()-initial_tensors[k].double()).square().sum())
                          for k, v in head.state_dict().items()))
    require(final != before and delta > 0, 'head did not update')
    ck = dict(head.checkpoint_dict(), **info, step=500, arm='coordinate_transport', frame_basis=FRAME_BASIS,
        optimizer=optimizer.state_dict(), head_tensor_sha256=final, initial_head_tensor_sha256=before,
        parameter_delta_l2=delta, frozen_inputs_verified=True)
    validate_match(initial, control, ck)
    checkpoint = out/'checkpoint_latest.pt'
    with checkpoint.open('xb') as stream:
        torch.save(ck, stream)
    base.write_json(out/'completion.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        step=500, checkpoint_sha256=base.sha(checkpoint), head_tensor_sha256=final,
        parameter_delta_l2=delta, frozen_inputs_verified=True, **info))
    print(json.dumps(dict(output=str(checkpoint), sha256=base.sha(checkpoint), step=500)), flush=True)


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
                    require(a[field] == b[field], f'observed control differs from frozen prior: {field}')
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
    parser.add_argument('--mode', choices=('diagnose', 'train', 'evaluate'), required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    base.seed_all()
    dict(diagnose=diagnose, train=train, evaluate=evaluate)[args.mode](args)


if __name__ == '__main__':
    main()
