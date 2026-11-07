"""Versioned real-MD backbone diagnostics for a CA-only dynamics checkpoint.

The two forecast routes use observed context only. Oracle routes deliberately
use a true endpoint and are labelled readout diagnostics, never predictions.
An ideal CA lift is deterministic postprocessing. The legacy learned B1a head
is an alternative readout whose CA need not match the current CA head; it is
not silently substituted into the current model's prediction contract.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil

import torch

from evaluation.backbone_metrics_v2 import backbone_error_summary, backbone_geometry_summary
from evaluation.data import EvalProtocol, iter_windows
from evaluation.geometry import _kabsch_rotation
from evaluation.loaders import sha256_file
from protein_jepa_frame.data.dataset import AA_ALPHABET
from protein_jepa_frame.models.stage_c.openfold_ca_lifter import lift_ca_to_backbone
from protein_jepa_frame.training.train_stage_b1a import _extract_backbone_frames
from protein_jepa_frame.training.train_stage_ensemble import load_frozen_stack
from protein_jepa_frame.utils.contracts import state_sha256
from scripts.diag_lasp_ca_ceiling import _development_manifest, _window_seed
from scripts.eval_ca_endpoint_scores import SOURCES as ENDPOINT_SOURCES, load_model
from scripts.gate_ca_representation import audit_sources


ROLES = {
    'forecast_ca_ideal_lift': 'forecast: current CA head plus deterministic ideal lift',
    'forecast_H_legacy_b1a': 'forecast compatibility diagnostic: generated H with alternative learned B1a readout',
    'oracle_current_H_ca_ideal_lift': 'oracle: true endpoint H through current CA decoder then ideal lift',
    'oracle_MD_CA_ideal_lift': 'oracle: real endpoint CA then ideal lift; isolates lift error',
    'oracle_current_H_legacy_b1a': 'oracle compatibility diagnostic: current encoder true endpoint H through old B1a',
    'oracle_original_H_legacy_b1a': 'oracle native-basis control: original Stage A true endpoint H through old B1a',
    'persistence_MD': 'forecast control: observed real backbone',
    'target_MD': 'reference: real endpoint backbone; self-error zero',
}
SOURCES = tuple(dict.fromkeys((*ENDPOINT_SOURCES,
    'scripts/eval_ca_backbone_diagnostic.py', 'evaluation/backbone_metrics_v2.py',
    'protein_jepa_frame/models/stage_b1a/model.py',
    'protein_jepa_frame/models/stage_c/structure_decoder.py',
    'protein_jepa_frame/models/stage_c/geometry.py',
    'protein_jepa_frame/models/stage_c/openfold_ca_lifter.py',
    'protein_jepa_frame/models/stage_c/openfold_atom37.py',
    'protein_jepa_frame/training/train_stage_b1a.py',
    'protein_jepa_frame/vendored/openfold/np/residue_constants.py',
    'protein_jepa_frame/vendored/openfold/resources/stereo_chemical_props.txt',
    'protein_jepa_frame/vendored/openfold/utils/feats.py',
    'protein_jepa_frame/vendored/openfold/utils/rigid_utils.py',
    'protein_jepa_frame/vendored/openfold/utils/tensor_utils.py',
    'protein_jepa_frame/vendored/openfold/data/data_transforms.py',
    'protein_jepa_frame/vendored/PROVENANCE.json')))


def align_target_backbone(endpoint, anchor_ca, mask):
    """Fit real endpoint CA once, applying the SAME transform to every atom."""
    if endpoint.shape != (len(mask), 4, 3) or anchor_ca.shape != (len(mask), 3):
        raise ValueError('expected endpoint [L,4,3] and anchor CA [L,3]')
    if mask.dtype != torch.bool or int(mask.sum()) < 4:
        raise ValueError('at least four valid residues required')
    if not (torch.isfinite(endpoint[mask]).all() and torch.isfinite(anchor_ca[mask]).all()):
        raise ValueError('nonfinite valid endpoint/anchor')
    p, q = endpoint[mask, 1].double(), anchor_ca[mask].double()
    rotation = _kabsch_rotation(p - p.mean(0), q - q.mean(0))
    aligned = (endpoint.double() - p.mean(0)) @ rotation + q.mean(0)
    return torch.where(mask[:, None, None], aligned, 0).to(endpoint.dtype)


def continuous_segments(mask, residue_index, chain_id):
    """Do not construct a CA-triplet frame across a mask, chain or index gap."""
    segments, current = [], []
    for i in range(len(mask)):
        if not bool(mask[i]):
            if current:
                segments.append(current)
                current = []
            continue
        if current and (int(chain_id[i]) != int(chain_id[current[-1]]) or
                        int(residue_index[i]) != int(residue_index[current[-1]]) + 1):
            segments.append(current)
            current = []
        current.append(i)
    if current:
        segments.append(current)
    return segments


def chemical_aatype(dataset_aatype):
    """Convert dataset embedding IDs to OpenFold chemical IDs, preserving UNK.

    Encoder and legacy decoder embeddings retain their trained dataset IDs.
    This conversion applies ONLY to the ideal lift and evaluation chemistry.
    Unknown aliases cannot be reconstructed from an integer UNK alone.
    """
    from protein_jepa_frame.vendored.openfold.np import residue_constants as rc
    if bool(((dataset_aatype < 0) | (dataset_aatype >= len(AA_ALPHABET))).any()):
        raise ValueError('invalid dataset aatype index')
    lookup = torch.tensor([rc.restype_order.get(letter, 20) for letter in AA_ALPHABET],
                          device=dataset_aatype.device, dtype=torch.long)
    return lookup[dataset_aatype]


@torch.no_grad()
def lift_segmented(ca, aa, mask, ri, ci):
    """All samples, segmented lifting, with endpoint/degeneracy accounting."""
    if ca.ndim != 3 or ca.shape[1:] != (len(mask), 3):
        raise ValueError('CA lift expects [S,L,3]')
    if not torch.isfinite(ca[:, mask]).all():
        raise ValueError('nonfinite valid CA')
    if bool(((aa[mask] < 0) | (aa[mask] > 20)).any()):
        raise ValueError('CA lift requires OpenFold chemical types 0..20')
    segments = continuous_segments(mask, ri, ci)
    result = ca.new_zeros((len(ca), len(mask), 4, 3))
    fallback = torch.zeros((len(ca), len(mask)), device=ca.device, dtype=torch.bool)
    for indices in segments:
        segment = ca[:, indices]
        backbone, _, _ = lift_ca_to_backbone(segment, aatype=aa[indices][None].expand(len(ca), -1))
        result[:, indices] = backbone
        # Mirror the lifter's numerical frame validity criterion for reporting.
        # Endpoints extrapolate collinearly and therefore use identity fallback.
        nxt, prv = torch.empty_like(segment), torch.empty_like(segment)
        if len(indices) > 1:
            nxt[:, :-1] = segment[:, 1:]
            nxt[:, -1] = 2 * segment[:, -1] - segment[:, -2]
            prv[:, 1:] = segment[:, :-1]
            prv[:, 0] = 2 * segment[:, 0] - segment[:, 1]
        else:
            unit = segment.new_tensor([1., 0., 0.])
            nxt, prv = segment + unit, segment - unit
        unit = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        x = unit(nxt-segment)
        y = unit(torch.cross(x, unit(segment-prv), dim=-1))
        z = unit(torch.cross(x, y, dim=-1))
        y = unit(torch.cross(z, x, dim=-1))
        rotation = torch.stack((x, y, z), -1)
        fallback[:, indices] = (~torch.isfinite(rotation).all(dim=(-1, -2)) |
                               (rotation.norm(dim=-2).min(dim=-1).values < .5))
    if not torch.equal(result[:, mask, 1], ca[:, mask]):
        raise ValueError('lift changed the CA column')
    return result, dict(n_segments=len(segments), segment_lengths=[len(s) for s in segments],
        unknown_gly_backbone_template_residues=int((mask & (aa == 20)).sum()),
        terminal_residue_fraction=sum(min(2, len(s)) for s in segments)/int(mask.sum()),
        frame_identity_fallback_fraction_per_sample=fallback[:, mask].float().mean(-1).tolist(),
        interpretation='identity fallback may break rotation equivariance; no inter-residue closure guarantee')


@torch.no_grad()
def score_window(model, original_encoder, legacy_decoder, window, *, device='cpu', n_samples=12):
    if model.training or getattr(model, 'prediction_kind', None) != 'endpoint':
        raise ValueError('requires eval-mode endpoint sampling')
    if window.target_frame != window.anchor_frame + window.horizon or not math.isclose(
        window.horizon_ps, window.horizon * model.kernel.cfg.step_ps, rel_tol=1e-6, abs_tol=1e-3):
        raise ValueError('physical endpoint mismatch')
    context = window.context
    observed = context['coords'].to(device)
    mask = context['residue_mask'].to(device).all(0).bool()
    aa, ri, ci = [context[key].to(device) for key in ('aatype', 'residue_index', 'chain_id')]
    chemical_aa = chemical_aatype(aa)
    # Both existing OpenFold paths explicitly use GLY backbone placement for
    # UNK. Keep its backbone atoms in every route; only residue-specific z
    # references exclude unknown chemistry, with their own metric counts.
    scoring_mask = mask
    if int(scoring_mask.sum()) < 4:
        raise ValueError('requires >=4 valid residues for matched backbone diagnostics')
    seed = _window_seed(0, window)
    ca, generated_h = model.sample(observed[None], mask[None], aa[None], ri[None], ci[None],
        torch.tensor([window.temperature_k], device=device), window.horizon, n_samples,
        torch.Generator(device=device).manual_seed(seed))
    if ca.shape != (1, n_samples, len(mask), 3) or model.last_observation_steps != [window.horizon]*n_samples:
        raise ValueError('sampler did not return the requested endpoints')
    rotation, anchor = _extract_backbone_frames(observed[-1:])
    expand = lambda value, n: value.expand(n, *value.shape[1:])

    def old_decode(h):
        n = len(h)
        out = legacy_decoder.decode_structure(h, expand(aa[None], n), expand(rotation, n),
                                               expand(anchor, n), expand(mask[None], n))
        if not bool(out['atom37_mask'][..., [0, 1, 2, 4]][:, scoring_mask].all()):
            raise ValueError('legacy decoder lacks evaluated backbone atoms')
        return out['backbone_coords']

    # Freeze both forecast branches BEFORE accessing the endpoint label.
    predictions, lift_info = {}, {}
    predictions['forecast_ca_ideal_lift'], lift_info['forecast_ca_ideal_lift'] = lift_segmented(
        ca[0], chemical_aa, scoring_mask, ri, ci)
    predictions['forecast_H_legacy_b1a'] = old_decode(generated_h[0])
    if window.target['coords'].shape != (1, len(mask), 4, 3) or not bool(
            window.target['residue_mask'].to(device)[:, mask].all()):
        raise ValueError('real MD endpoint missing evaluated backbone')
    endpoint = window.target['coords'].to(device)
    truth = align_target_backbone(endpoint[0], anchor[0], mask)
    current_h = model.bundle.encode(endpoint, mask[None], aa[None], ri[None], ci[None])
    anchor_h = model.bundle.encode(observed[-1:], mask[None], aa[None], ri[None], ci[None])
    oracle_ca = model.bundle.decode_ca(current_h, anchor_h, aa[None], ri[None], ci[None],
                                       rotation, anchor, mask[None])
    original_h = original_encoder(endpoint[:, None], mask[None, None], aa[None], ri[None], ci[None],
                                  token_mask=mask[None, None])[:, 0]
    for name, coordinates in [('oracle_current_H_ca_ideal_lift', oracle_ca),
                              ('oracle_MD_CA_ideal_lift', truth[None, :, 1])]:
        predictions[name], lift_info[name] = lift_segmented(coordinates, chemical_aa, scoring_mask, ri, ci)
    predictions.update(oracle_current_H_legacy_b1a=old_decode(current_h),
        oracle_original_H_legacy_b1a=old_decode(original_h),
        persistence_MD=observed[-1:], target_MD=truth[None])
    # Metrics use float64 internally; transfer once per route instead of
    # synchronizing CUDA for every scalar geometric statistic.
    truth_cpu, mask_cpu, aa_cpu, ri_cpu, ci_cpu = [value.cpu() for value in
        (truth, scoring_mask, chemical_aa, ri, ci)]
    routes = {}
    for name in ROLES:
        backbone = predictions[name]
        backbone_cpu = backbone.cpu()
        routes[name] = dict(role=ROLES[name], n_samples=len(backbone),
            error=backbone_error_summary(backbone_cpu, truth_cpu, mask_cpu),
            geometry=backbone_geometry_summary(backbone_cpu, mask_cpu, aatype=aa_cpu,
                                               residue_index=ri_cpu, chain_id=ci_cpu))
        if name in lift_info:
            routes[name]['lift'] = lift_info[name]
    ca_readout_difference = (predictions['forecast_H_legacy_b1a'][:, scoring_mask, 1]-ca[0, :, scoring_mask]).square().sum(-1).mean(-1).sqrt()
    return dict(protein=window.protein_id, trajectory=window.trajectory_id,
        temperature_k=float(window.temperature_k), anchor_frame=int(window.anchor_frame),
        target_frame=int(window.target_frame), horizon=int(window.horizon), horizon_ps=float(window.horizon_ps),
        context_frames=window.context_frames.tolist(), valid_residues=int(mask.sum()),
        scored_residues=int(scoring_mask.sum()), unknown_residues_excluded=0,
        chemical_unknown_residues=int((mask & (chemical_aa == 20)).sum()),
        scored_residue_index=ri[scoring_mask].tolist(), unknown_residue_index=ri[mask & (chemical_aa == 20)].tolist(),
        scoring_residue_coverage=float(scoring_mask.sum()/mask.sum()), sampling_seed=seed,
        legacy_vs_current_forecast_CA_rmsd_per_sample=ca_readout_difference.tolist(), routes=routes)


def summarize(rows):
    """Equal samples within window, windows within protein, then proteins."""
    seen, groups = set(), defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ('protein', 'trajectory', 'temperature_k', 'anchor_frame', 'horizon'))
        if key in seen:
            raise ValueError('duplicate backbone window')
        seen.add(key)
        groups[(row['horizon'], row['protein'])].append(row)
    result = {}
    for h in sorted({key[0] for key in groups}):
        per_protein = []
        for (horizon, protein), windows in sorted(groups.items()):
            if horizon != h:
                continue
            values = {}
            for route in ROLES:
                values[route] = {}
                for category in ('error', 'geometry'):
                    metrics = {}
                    for name in windows[0]['routes'][route][category]['mean']:
                        finite = [r['routes'][route][category]['mean'][name] for r in windows
                                  if r['routes'][route][category]['mean'][name] is not None]
                        metrics[name] = sum(finite)/len(finite) if finite else None
                    values[route][category] = metrics
            per_protein.append(dict(protein=protein, n_windows=len(windows), routes=values))
        means = {route: {} for route in ROLES}
        for route in ROLES:
            for category in ('error', 'geometry'):
                means[route][category] = {}
                for name in per_protein[0]['routes'][route][category]:
                    finite = [p['routes'][route][category][name] for p in per_protein
                              if p['routes'][route][category][name] is not None]
                    means[route][category][name] = dict(mean=sum(finite)/len(finite) if finite else None,
                                                      n_proteins=len(finite))
        result[str(h)] = dict(n_proteins=len(per_protein), means=means, per_protein=per_protein)
    return result


def audit_legacy_basis(encoder, decoder, checkpoint, stage_a_sha256):
    """A matching width alone cannot establish the legacy head's native H."""
    contract = checkpoint.get('representation_contract', {})
    actual_encoder = state_sha256(encoder)
    actual_decoder = state_sha256(decoder.decoder)
    if (contract.get('decoder_input') != 'raw_H'
            or contract.get('encoder_sha256') != actual_encoder
            or state_sha256(checkpoint['encoder']) != actual_encoder
            or contract.get('decoder_sha256') != actual_decoder
            or contract.get('source', {}).get('stage_a_checkpoint_sha256') != stage_a_sha256):
        raise ValueError('legacy B1a native-basis encoder/decoder/source contract mismatch')
    return dict(encoder_sha256=actual_encoder, decoder_sha256=actual_decoder,
                stage_a_sha256=stage_a_sha256, native_basis_verified=True)


def run(args):
    args.manifest = _development_manifest(args.manifest)
    if args.max_windows < 4 or args.max_windows % 4:
        raise ValueError('requires complete standard200 four-horizon groups')
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(0)
    output = Path(args.out)
    registration = output.with_suffix('.protocol.json')
    if output.exists() or registration.exists():
        raise FileExistsError('refusing to overwrite diagnostic artifacts')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    component = checkpoint['component_checkpoint']
    upstream = component['calibration_args']
    stage_a, b1a = upstream['stage_a_ckpt'], upstream['b1a_ckpt']
    input_paths = (args.checkpoint, stage_a, b1a, args.manifest)
    inputs = {p: sha256_file(p) for p in input_paths}
    provenance = audit_sources(checkpoint, component, 'artifacts/mdcath_manifest_full/train.jsonl', args.manifest)
    sources = {p: sha256_file(p) for p in SOURCES}
    protocol = EvalProtocol(manifest=args.manifest, context_frames=8, horizons_frames=(1,4,16,64),
        max_residues=96, max_anchors_per_trajectory=4, temperatures=(320.,348.,379.,413.,450.),
        seed=0, round_robin_proteins=True, round_robin_conditions=True)
    report = dict(schema_version='ca_backbone_readout_diagnostic_v2',
        registered_utc=datetime.now(timezone.utc).isoformat(), dataset_role='development_validation',
        args=vars(args), protocol=asdict(protocol), route_roles=ROLES, input_hashes=inputs,
        metric_arithmetic='float64 on CPU; prediction and truth-fit use requested model device',
        source_hashes=sources, provenance=provenance, optimizer_steps=0, test_opened=False,
        bioemu2_rerun=False, forecast_receives_future=False,
        residue_type_conventions=dict(dataset_order=AA_ALPHABET, chemistry_order='ARNDCQEGHILKMFPSTWYVX',
            conversion='map dataset to OpenFold only for ideal lift and chemical metric reference',
            legacy_decoder='preserve historical dataset IDs for its learned embedding and its existing OpenFold placement; legacy placement has residue-type mismatch',
            scoring='all routes share all context-valid residues; do not infer UNK identity; OpenFold paths use existing GLY backbone fallback, residue-specific z excludes UNK and reports counts',
            anchor_fit='all context-valid CA residues including UNK determine the fixed target-to-observed transform'),
        limitations=['CA ideal lift is not a learned backbone and has no chain closure guarantee',
            'UNK residues retain all backbone errors/geometry but use existing GLY placement; unknown chemistry has no residue-specific z reference',
            'legacy B1a retains its historical dataset/OpenFold residue-index mismatch; current ideal-lift route corrects that boundary explicitly',
            'legacy B1a is an alternative readout; generated H compatibility is unproven and its CA can differ',
            'oracles access future structure solely to diagnose readout and never enter the forecasts',
            'box Ramachandran is not MolProbity; no full-backbone/all-atom clash metric; T1 incomplete',
            'single realized endpoint RMSD is not a calibrated conditional distribution score',
            'adaptive development panel without independent confirmation or new BioEmu2 comparison'],
        aggregation='equal samples per window, equal windows per protein, equal proteins per horizon; no pass or CI claim')
    output.parent.mkdir(parents=True, exist_ok=True)
    registration.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for source in sources:
        destination = output.with_suffix('.source_snapshot')/source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    model = load_model(args.checkpoint, checkpoint, args.device)
    original, legacy, _ = load_frozen_stack(stage_a, b1a, args.device)
    legacy_audit = audit_legacy_basis(original, legacy,
        torch.load(b1a, map_location='cpu', weights_only=False), inputs[stage_a])
    hashes = [state_sha256(x) for x in (model, original, legacy)]
    rows = []
    for window in iter_windows(protocol):
        if len(rows) == args.max_windows:
            break
        rows.append(score_window(model, original, legacy, window, device=args.device))
        if len(rows) % 10 == 0:
            print(f'Backbone diagnostic {len(rows)}/{args.max_windows}', flush=True)
    if len(rows) != args.max_windows or set(r['horizon'] for r in rows) != {1,4,16,64}:
        raise RuntimeError('incomplete backbone panel')
    if [state_sha256(x) for x in (model, original, legacy)] != hashes:
        raise RuntimeError('model tensors changed during evaluation')
    for path, expected in {**inputs, **sources}.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f'input/source changed: {path}')
    for audit in provenance['training_splits']:
        if sha256_file(audit['manifest']) != audit['train_manifest_sha256']:
            raise RuntimeError('training manifest changed')
    report.update(completed_utc=datetime.now(timezone.utc).isoformat(), frozen_inputs_verified=True,
        model_tensor_hashes=hashes, legacy_basis_audit=legacy_audit,
        n_windows=len(rows), n_proteins=len({r['protein'] for r in rows}),
        per_window=rows, by_horizon=summarize(rows))
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output), n_windows=len(rows), n_proteins=report['n_proteins'])), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--manifest', default='artifacts/mdcath_manifest_full/val.jsonl')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-windows', type=int, default=200)
    parser.add_argument('--torch-threads', type=int, default=4)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
