"""Train-only local gradient/finite-difference diagnosis of frozen CA readout drift.

No optimizer, sampling, validation/test access, or persistent checkpoint write occurs.
Observed retention sees exactly four observed frames; future H is used only for
explicitly labelled 16/64-frame oracle readout diagnostics. These derivatives
are not autonomous forecast gradients or evidence about information loss.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from evaluation.data import EvalProtocol, iter_windows
from evaluation.geometry import kabsch_align
from evaluation.loaders import sha256_file
from protein_jepa_frame.models.ca_component_bundle import CAComponentBundle
from protein_jepa_frame.training.train_lasp_calibration import teacher_terms
from protein_jepa_frame.training.train_stage_b1a import _extract_backbone_frames
from protein_jepa_frame.utils.contracts import state_sha256


OBJECTIVES = ('teacher_weighted', 'observed_readout_mse', 'oracle16_mse', 'oracle64_mse')
SOURCES = ('scripts/diagnose_ca_readout_retention.py', 'evaluation/data.py',
    'evaluation/geometry.py', 'protein_jepa_frame/data/dataset.py',
    'protein_jepa_frame/models/ca_component_bundle.py', 'protein_jepa_frame/models/encoder.py',
    'protein_jepa_frame/models/geometric_delta_decoder.py', 'protein_jepa_frame/models/ca_geometry.py',
    'protein_jepa_frame/training/train_lasp_calibration.py',
    'protein_jepa_frame/training/train_stage_ensemble.py',
    'protein_jepa_frame/training/train_stage_b1a.py', 'protein_jepa_frame/utils/contracts.py')


def encode(bundle, coordinates, group):
    n = len(coordinates)
    expand = lambda value: value.expand(n, *value.shape[1:])
    return bundle.encode(coordinates, expand(group['mask'][None]),
                         *[expand(group[k][None]) for k in ('aatype', 'residue_index', 'chain_id')])


def decode(bundle, observed_h, future_h, group):
    # Anchor latent is deliberately in the graph, as in deployed delta decoding.
    h = torch.cat((observed_h[:-1], future_h), dim=0)
    n = len(h)
    expand = lambda value: value.expand(n, *value.shape[1:])
    return bundle.decode_ca(h, expand(observed_h[-1:]),
        *[expand(group[k][None]) for k in ('aatype', 'residue_index', 'chain_id')],
        expand(group['rotation']), expand(group['anchor']), expand(group['mask'][None]))


def collect_groups(protocol, maximum):
    groups = OrderedDict()
    count = 0
    for window in iter_windows(protocol):
        if count == maximum:
            break
        key = (window.protein_id, window.trajectory_id, float(window.temperature_k), int(window.anchor_frame))
        if key not in groups:
            context = window.context
            groups[key] = dict(protein=window.protein_id, trajectory=window.trajectory_id,
                temperature_k=float(window.temperature_k), anchor_frame=int(window.anchor_frame),
                context_frames=window.context_frames.tolist(), observed=context['coords'],
                mask=context['residue_mask'].all(0).bool(),
                **{k: context[k] for k in ('aatype', 'residue_index', 'chain_id')}, targets={}, horizon_ps={})
        group = groups[key]
        if window.horizon in group['targets']:
            raise ValueError('duplicate oracle diagnostic window')
        if not bool(window.target['residue_mask'][:, group['mask']].all()):
            raise ValueError('target is missing observed-valid residues')
        group['targets'][window.horizon] = window.target['coords'][0]
        group['horizon_ps'][window.horizon] = float(window.horizon_ps)
        count += 1
    result = list(groups.values())
    if (count != maximum or any(set(g['targets']) != {16, 64} for g in result)
            or len({g['protein'] for g in result}) != len(result)):
        raise ValueError('diagnosis requires complete16/64 pairs and one anchor per protein')
    return result


@torch.no_grad()
def prepare(groups, frozen, device):
    for group in groups:
        for key in ('observed', 'mask', 'aatype', 'residue_index', 'chain_id'):
            group[key] = group[key].to(device)
        if int(group['mask'].sum()) < 4 or len(group['observed']) != 4:
            raise ValueError('need four observed frames and >=4 valid residues')
        group['future'] = torch.stack([group['targets'][h] for h in (16, 64)]).to(device)
        del group['targets']
        group['rotation'], group['anchor'] = _extract_backbone_frames(group['observed'][-1:])
        coordinates = torch.cat((group['observed'][:-1], group['future']))[:, :, 1]
        group['aligned_truth'] = torch.stack([
            kabsch_align(ca, group['anchor'][0], group['mask']) for ca in coordinates])
        group['teacher_h'] = encode(frozen, group['observed'], group).detach()
        group['teacher_future_h'] = encode(frozen, group['future'], group).detach()
        group['baseline_readout'] = decode(frozen, group['teacher_h'], group['teacher_future_h'], group).detach()


def losses(bundle, group, weights):
    observed_h = encode(bundle, group['observed'], group)
    # Future H is oracle diagnostic supervision, never retention input.
    future_h = encode(bundle, group['future'], group)
    prediction = decode(bundle, observed_h, future_h, group)
    m = group['mask']
    squared = (prediction[:, m] - group['aligned_truth'][:, m]).square().sum(-1).mean(-1)
    absolute, delta = teacher_terms(observed_h[None], group['teacher_h'][None], m[None])
    result = dict(teacher_absolute=absolute, teacher_delta=delta,
        teacher_weighted=weights[0] * absolute + weights[1] * delta,
        observed_readout_mse=squared[:3].mean(), oracle16_mse=squared[3], oracle64_mse=squared[4],
        observed_mean_frame_rmsd=squared[:3].sqrt().mean(), oracle16_rmsd=squared[3].sqrt(),
        oracle64_rmsd=squared[4].sqrt(),
        observed_readout_shift_rms=(prediction[:3, m] - group['baseline_readout'][:3, m]).square().sum(-1).mean().sqrt(),
        oracle16_readout_shift_rms=(prediction[3, m] - group['baseline_readout'][3, m]).square().sum(-1).mean().sqrt(),
        oracle64_readout_shift_rms=(prediction[4, m] - group['baseline_readout'][4, m]).square().sum(-1).mean().sqrt(),
        latent_anchor_drift_rms=(observed_h[-1, m] - group['teacher_h'][-1, m]).square().mean().sqrt(),
        latent_observed_delta_drift_rms=((observed_h[:-1, m] - observed_h[-1:, m])
            - (group['teacher_h'][:-1, m] - group['teacher_h'][-1:, m])).square().mean().sqrt())
    if not all(bool(torch.isfinite(value)) for value in result.values()):
        raise FloatingPointError('nonfinite readout diagnostic loss')
    return result


def flatten_grad(loss, parameters, *, retain_graph):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten().cpu().double()
                      for p, g in zip(parameters, gradients)])


def cosine(left, right):
    denominator = float(left.norm() * right.norm())
    return float(left @ right / denominator) if denominator > 1e-20 else None


def evaluate(bundle, groups, weights, *, gradients=False, progress=False):
    parameters = list(bundle.encoder.parameters())
    accumulated, rows = {}, []
    with torch.set_grad_enabled(gradients):
        for index, group in enumerate(groups):
            values = losses(bundle, group, weights)
            row = {key: float(value.detach()) for key, value in values.items()}
            if gradients:
                local = {name: flatten_grad(values[name], parameters, retain_graph=i < len(OBJECTIVES)-1)
                         for i, name in enumerate(OBJECTIVES)}
                for name, vector in local.items():
                    if not torch.isfinite(vector).all():
                        raise FloatingPointError('nonfinite encoder gradient')
                    accumulated[name] = accumulated.get(name, torch.zeros_like(vector)) + vector / len(groups)
                row['gradient_norms'] = {name: float(g.norm()) for name, g in local.items()}
                row['teacher_observed_gradient_cosine'] = cosine(local['teacher_weighted'], local['observed_readout_mse'])
                row['observed_oracle_gradient_cosine'] = {name: cosine(local['observed_readout_mse'], local[name])
                                                        for name in ('oracle16_mse', 'oracle64_mse')}
            rows.append(row)
            if progress and (index + 1) % 8 == 0:
                print(f'Oracle diagnostic {index+1}/{len(groups)} anchor groups; gradients={gradients}', flush=True)
    means = {key: float(np.mean([row[key] for row in rows])) for key in values}
    return means, rows, accumulated


@torch.no_grad()
def assign_parameters(parameters, vector):
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.copy_(vector[offset:offset+count].reshape_as(parameter).to(parameter))
        offset += count
    if offset != len(vector):
        raise ValueError('parameter-vector size mismatch')


def gradient_summary(gradients, drift):
    return dict(norms={name: float(g.norm()) for name, g in gradients.items()},
        cosines={name: {other: cosine(g, h) for other, h in gradients.items()} for name, g in gradients.items()},
        actual_joint_minus_frozen_drift={name: dict(cosine=cosine(g, drift), dot=float(g @ drift))
                                        for name, g in gradients.items()})


def finite_differences(bundle, groups, weights, gradients, reference_means, relative_steps):
    parameters = list(bundle.encoder.parameters())
    original = torch.cat([p.detach().flatten().cpu().double() for p in parameters])
    original_hash = state_sha256(bundle.encoder)
    reference_norm = float(original.norm())
    results = []
    try:
        for objective in ('teacher_weighted', 'observed_readout_mse'):
            gradient = gradients[objective]
            if float(gradient.norm()) <= 1e-20:
                results.append(dict(direction=objective, undefined_zero_gradient=True))
                continue
            direction = -gradient / gradient.norm()
            derivatives = {name: float(g @ direction) for name, g in gradients.items()}
            for relative in relative_steps:
                step = relative * reference_norm
                sides = {}
                for sign, label in ((1., 'descent'), (-1., 'opposite')):
                    # Every point starts from the same clone; never undo by subtraction.
                    assign_parameters(parameters, original + sign * step * direction)
                    actual = torch.cat([p.detach().flatten().cpu().double() for p in parameters]) - original
                    values, _, _ = evaluate(bundle, groups, weights)
                    sides[label] = dict(losses=values, actual_parameter_displacement_norm=float(actual.norm()),
                                        differences_from_joint={k: values[k] - reference_means[k] for k in values})
                central = {name: (sides['descent']['losses'][name] - sides['opposite']['losses'][name]) / (2 * step)
                           for name in OBJECTIVES}
                results.append(dict(direction=f'negative_{objective}_gradient', relative_parameter_norm_step=relative,
                    absolute_step=step, direction_sha256=state_sha256({'direction': direction}),
                    autodiff_directional_derivatives=derivatives, central_finite_difference=central,
                    derivative_relative_disagreement={name: abs(central[name] - derivatives[name]) /
                        max(abs(central[name]), abs(derivatives[name]), 1e-10) for name in OBJECTIVES}, **sides))
                print(f'FD direction={objective} relative_step={relative:g}: {central}', flush=True)
    finally:
        assign_parameters(parameters, original)
        if state_sha256(bundle.encoder) != original_hash:
            raise RuntimeError('finite differences failed to restore encoder exactly')
    return results


def run(args):
    root = Path(__file__).resolve().parents[1]
    train_manifest = root / 'artifacts/mdcath_manifest_full/train.jsonl'
    if args.max_windows < 2 or args.max_windows % 2:
        raise ValueError('window budget must contain complete16/64 pairs')
    output = Path(args.out)
    if output.exists():
        raise ValueError('diagnosis refuses to overwrite an existing result')
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(0)
    locked = {path: sha256_file(path) for path in (args.frozen_component, args.joint_checkpoint, str(train_manifest))}
    source_hashes = {path: sha256_file(path) for path in SOURCES}
    frozen_source = torch.load(args.frozen_component, map_location='cpu', weights_only=False)
    joint_checkpoint = torch.load(args.joint_checkpoint, map_location='cpu', weights_only=False)
    joint_source = joint_checkpoint['component_checkpoint']
    if state_sha256(joint_checkpoint['target_encoder']) != state_sha256(frozen_source['encoder']):
        raise ValueError('joint authoritative target_encoder differs from frozen component')
    if (state_sha256(joint_source['delta_ca_decoder']) != state_sha256(frozen_source['delta_ca_decoder'])
            or joint_source['calibration_args']['bond_projection_steps'] != frozen_source['calibration_args']['bond_projection_steps']):
        raise ValueError('comparison requires identical decoder and projection settings')
    weights = (float(joint_checkpoint['args']['teacher_weight']), float(joint_checkpoint['args']['teacher_delta_weight']))
    protocol = EvalProtocol(manifest=str(train_manifest), context_frames=4, horizons_frames=(16, 64),
        max_residues=96, max_anchors_per_trajectory=4, temperatures=(320., 348., 379., 413., 450.),
        seed=0, round_robin_proteins=True, round_robin_conditions=True)
    with tempfile.TemporaryDirectory(prefix='readout_retention_') as temp:
        joint_path = Path(temp) / 'joint_component.pt'
        torch.save(joint_source, joint_path)
        frozen = CAComponentBundle(args.frozen_component, args.device).eval()
        joint = CAComponentBundle(joint_path, args.device).eval()
        for bundle, source in ((frozen, frozen_source), (joint, joint_source)):
            if state_sha256(bundle.encoder) != state_sha256(source['encoder']):
                raise ValueError('loaded encoder differs from selected checkpoint')
            if state_sha256(bundle.decoder) != state_sha256(source['delta_ca_decoder']):
                raise ValueError('loaded decoder differs from selected checkpoint')
            bundle.encoder.requires_grad_(True)
            bundle.decoder.requires_grad_(False)
        if [(name, tuple(p.shape)) for name, p in frozen.encoder.named_parameters()] != [
                (name, tuple(p.shape)) for name, p in joint.encoder.named_parameters()]:
            raise ValueError('encoder parameter layouts differ')
        before = {'frozen': state_sha256(frozen), 'joint': state_sha256(joint)}
        groups = collect_groups(protocol, args.max_windows)
        prepare(groups, frozen, args.device)
        original = torch.cat([p.detach().flatten().cpu().double() for p in frozen.encoder.parameters()])
        current = torch.cat([p.detach().flatten().cpu().double() for p in joint.encoder.parameters()])
        drift = current - original
        baseline, baseline_rows, baseline_grad = evaluate(frozen, groups, weights, gradients=True, progress=True)
        current_losses, joint_rows, joint_grad = evaluate(joint, groups, weights, gradients=True, progress=True)
        finite = finite_differences(joint, groups, weights, joint_grad, current_losses, (1e-5, 3e-5))
        if before != {'frozen': state_sha256(frozen), 'joint': state_sha256(joint)}:
            raise RuntimeError('diagnostic changed a model state')
    if any(sha256_file(path) != value for path, value in {**locked, **source_hashes}.items()):
        raise RuntimeError('input or diagnostic dependency changed during run')
    report = dict(task_version='train_only_readout_retention_gradient_v1',
        dataset_role='training_diagnostic_only', validation_opened=False, test_opened=False,
        optimizer_steps=0, model_states_restored_exactly=True, persistent_checkpoint_writes=0,
        temporary_component_copy_for_loading=True,
        args=vars(args), input_hashes=locked, source_hashes=source_hashes, protocol=asdict(protocol),
        teacher_weights={'absolute': weights[0], 'delta': weights[1]}, n_windows=args.max_windows,
        cuda_peak_allocated_bytes=(torch.cuda.max_memory_allocated(args.device)
            if torch.device(args.device).type == 'cuda' else None),
        n_proteins=len(groups), n_anchor_groups=len(groups),
        parameter_drift=dict(norm=float(drift.norm()), relative_l2=float(drift.norm() / original.norm())),
        frozen_means=baseline, joint_means=current_losses,
        joint_minus_frozen={key: current_losses[key] - baseline[key] for key in baseline},
        gradients={'frozen': gradient_summary(baseline_grad, drift), 'joint': gradient_summary(joint_grad, drift)},
        finite_differences=finite, per_anchor_group=[dict(
            **{key: group[key] for key in ('protein', 'trajectory', 'temperature_k', 'anchor_frame', 'context_frames', 'horizon_ps')},
            valid_residues=int(group['mask'].sum()), frozen=base, joint=now)
            for group, base, now in zip(groups, baseline_rows, joint_rows)],
        semantics=dict(retention='original teacher_terms on each four-frame observed clip, first-frame-relative delta and per-clip normalization; actual training batch size1',
            observed='three earlier observed true-H states decoded relative to last observed anchor; true observed CA aligned to same anchor; mean squared vector error in angstrom^2',
            oracle='true future H at16/64 raw frames decoded with the same observed anchor; not a forecast; both anchor and future encoder derivatives retained',
            aggregation='one anchor per protein; residue mean, then equal observed frames/proteins; separate16/64 objectives',
            finite_difference='all preset1e-5/3e-5 relative total-parameter-norm steps retained; this norm is only a step convention, including unused embedding rows',
            interpretation='local loss geometry only; direction agreement is not a historical Adam causal explanation or H information bound; no weight chosen'))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(output), frozen=baseline, joint=current_losses,
                         joint_gradient_cosines=report['gradients']['joint']['cosines']), indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-component', default='artifacts/lasp_full_v1/representation/checkpoint_best.pt')
    parser.add_argument('--joint-checkpoint', default='artifacts/hs_joint_20261101/pilot_v1/joint/checkpoint_best.pt')
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-windows', type=int, default=64)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--torch-threads', type=int, default=2)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
