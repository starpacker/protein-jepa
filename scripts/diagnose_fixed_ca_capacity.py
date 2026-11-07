#!/usr/bin/env python3
"""Independent, predeclared fixed-CA peptide-bond capacity lower bound.

This consumes only the pinned validation cache, never readout evaluation results,
checkpoints, summaries or raw trajectories. No model is trained or evaluated.
It uses the official reconstruction only to extract immutable N/CA/C radii.

Triangle inequality gives CN >= max(0, D-rC-rN). Thus a sufficient obstruction
to the 12-SD interval is D-rC-rN > mu+12*sigma. The positive difference is a
lower bound on the unsmoothed flat-bottom C-N penalty in Angstrom. It is also a
lower bound on the v3 smoothed penalty on this too-long side. This ignores
angles, other bonds, clashes and the short-distance obstruction. Zero is not
proof of feasibility, and this is not evidence that complete T1 passes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from protein_jepa_frame.models.stage_c.structure_decoder import reconstruct_atom37_from_frames
from protein_jepa_frame.vendored.openfold.np import residue_constants as rc


REPO = Path(__file__).resolve().parents[1]
VERSION = 'fixed_ca_capacity_v1'
CACHE = 'results/hs_joint_20261101/backbone_repair/readout_val200_v2.pt'
CACHE_SHA256 = 'd1a04f5861badc004f39d9496e3f245b6c32739a8d2deed5bb74840eaf605fdc'
OUTPUT = 'results/hs_joint_20261101/backbone_repair_v2/fixed_ca_capacity.json'
HORIZONS = (1, 4, 16, 64)
ROUTES = ('oracle_CA', 'forecast_CA', 'observed_MD_CA', 'target_MD_CA')
TOLERANCE_SD = 12.0
ABSOLUTE_MARGIN_ANGSTROM = 1e-4
ROUNDING_MULTIPLIER = 64.0
FLOAT32_EPS = float(torch.finfo(torch.float32).eps)
METRICS = ('impossible_edge_fraction', 'mean_lower_bound_angstrom',
           'mean_conservative_lower_bound_angstrom')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json_exclusive(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def numerical_margin(ca_i, ca_j, r_c, r_n, upper):
    """Fixed conservative engineering tolerance, not interval certification."""
    return (ABSOLUTE_MARGIN_ANGSTROM + ROUNDING_MULTIPLIER * FLOAT32_EPS *
            (torch.linalg.vector_norm(ca_i, dim=-1) + torch.linalg.vector_norm(ca_j, dim=-1)
             + r_c + r_n + upper))


def peptide_lower_bound(ca_i, ca_j, r_c, r_n, upper):
    """All arithmetic is float64; raw mathematical and margin-adjusted bounds."""
    ca_i, ca_j = torch.as_tensor(ca_i).double(), torch.as_tensor(ca_j).double()
    r_c, r_n, upper = [torch.as_tensor(x, dtype=torch.float64) for x in (r_c, r_n, upper)]
    distance = torch.linalg.vector_norm(ca_i-ca_j, dim=-1)
    raw = torch.relu(distance-r_c-r_n-upper)
    margin = numerical_margin(ca_i, ca_j, r_c, r_n, upper)
    conservative = torch.relu(raw-margin)
    return dict(distance=distance, raw=raw, margin=margin, conservative=conservative,
                impossible=raw > margin)


def official_template_radii():
    """Identity native decoder R, zero CA and torsions; UNK is official GLY fallback."""
    aa = torch.arange(21, dtype=torch.long)[None]
    rotation = torch.eye(3, dtype=torch.float64).expand(1, 21, 3, 3).clone()
    ca = torch.zeros(1, 21, 3, dtype=torch.float64)
    torsions = torch.zeros(1, 21, 7, dtype=torch.float64)
    atoms, mask = reconstruct_atom37_from_frames(rotation, ca, aa, torsions)
    require(torch.equal(atoms[..., 1, :], ca), 'official identity template changed CA')
    require(bool(mask[..., [0, 1, 2]].all()), 'official template missing N/CA/C')
    r_c = torch.linalg.vector_norm(atoms[0, :, 2]-atoms[0, :, 1], dim=-1)
    r_n = torch.linalg.vector_norm(atoms[0, :, 0]-atoms[0, :, 1], dim=-1)
    require(bool((r_c > 0).all() and (r_n > 0).all()), 'nonpositive official template radius')
    return r_c, r_n


def verify_official_reconstruction(r_c, r_n):
    """Predetermined rotations/translations/torsions; no validation outcomes used."""
    rotations = [torch.eye(3, dtype=torch.float64)]
    for angle in (.37, -1.2, 2.1):
        axis = torch.tensor([1., 2., 3.], dtype=torch.float64)
        axis = axis / axis.norm()
        x, y, z = axis
        skew = torch.tensor([[0., -z, y], [z, 0., -x], [-y, x, 0.]], dtype=torch.float64)
        a = torch.tensor(angle, dtype=torch.float64)
        rotations.append(torch.eye(3, dtype=torch.float64) + a.sin()*skew + (1-a.cos())*(skew@skew))
    rotation = torch.stack(rotations)[:, None].expand(4, 21, 3, 3).clone()
    # Exactly representable float32 translations: official adapter casts to float32.
    ca = torch.tensor([[0., 0., 0.], [16., -8., 4.], [-32., 64., 16.], [256., -128., 32.]],
                      dtype=torch.float64)[:, None].expand(4, 21, 3).clone()
    aa = torch.arange(21, dtype=torch.long)[None].expand(4, 21)
    torsions = torch.arange(4*21*7, dtype=torch.float64).reshape(4, 21, 7) / 113.
    atoms, mask = reconstruct_atom37_from_frames(rotation, ca, aa, torsions)
    require(torch.equal(atoms[..., 1, :], ca), 'official rotated reconstruction changed CA')
    require(bool(mask[..., [0, 1, 2]].all()), 'official rotated reconstruction missing atoms')
    actual_c = torch.linalg.vector_norm(atoms[..., 2, :]-ca, dim=-1)
    actual_n = torch.linalg.vector_norm(atoms[..., 0, :]-ca, dim=-1)
    error = torch.maximum((actual_c-r_c).abs(), (actual_n-r_n).abs())
    tolerance = numerical_margin(ca, ca, r_c, r_n, torch.tensor(1.533, dtype=torch.float64))
    require(bool((error <= tolerance).all()), 'official rotated radius exceeds fixed numerical margin')
    return dict(n_residue_types=21, n_rotations=4, ca_bitwise_preserved=True,
                radius_max_absolute_error_angstrom=float(error.max()),
                verification_margin_min_angstrom=float(tolerance.min()),
                verification_margin_max_angstrom=float(tolerance.max()),
                internal_reconstruction_dtype='float32; radii/distances assessed in float64',
                probe='fixed rotations, exactly representable translations, nonzero torsions; no data-dependent tuning')


def integer_vector(value, length, name):
    x = torch.as_tensor(value)
    require(x.shape == (length,) and x.dtype != torch.bool and not x.is_floating_point()
            and not x.is_complex(), f'{name} must be integer [{length}]')
    return x.long()


def prepare_edges(row):
    mask = torch.as_tensor(row['observed_mask'])
    require(mask.ndim == 1 and mask.dtype == torch.bool, 'observed mask must be bool [L]')
    length = len(mask)
    require(0 < length <= 96, 'invalid residue budget')
    aa = integer_vector(row['chemical_aatype'], length, 'chemical_aatype')
    require(bool(((aa >= 0) & (aa <= 20)).all()), 'chemical IDs outside OpenFold 0..20')
    ids = integer_vector(row['geometry_row_ids'], length, 'geometry_row_ids')
    require(len(ids.unique()) == length, 'duplicate geometry row identities')
    edges = torch.as_tensor(row['peptide_edges'])
    require(edges.ndim == 2 and edges.shape[1] == 2 and edges.dtype != torch.bool
            and not edges.is_floating_point() and not edges.is_complex(), 'invalid explicit peptide edge tensor')
    edges = edges.long()
    require(bool(((edges >= 0) & (edges < length)).all()), 'peptide index out of bounds')
    require(bool((edges[:, 0] != edges[:, 1]).all()), 'self peptide edge')
    require(len(torch.unique(edges, dim=0)) == len(edges), 'duplicate peptide edges')
    atoms, target_atoms = torch.as_tensor(row['atom_mask']), torch.as_tensor(row['target_atom_mask'])
    require(atoms.shape == target_atoms.shape == (length, 4) and atoms.dtype == target_atoms.dtype == torch.bool,
            'atom masks must be bool [L,4]')
    require(bool(target_atoms[mask].all()), 'target incomplete on observed-valid residue')
    i, j = edges.T
    available = (mask[i] & mask[j] & atoms[i, 1] & atoms[j, 1] & atoms[i, 2] & atoms[j, 0])
    unknown = available & (aa[j] == 20)
    eligible = available & ~unknown
    selected = edges[eligible]
    counts = dict(n_topology_edges=len(edges), n_masked_or_missing_edges=int((~available).sum()),
                  n_atom_candidate_edges=int(available.sum()), n_unknown_next_edges_excluded=int(unknown.sum()),
                  n_eligible_edges=int(eligible.sum()),
                  n_eligible_previous_unknown_edges=int((eligible & (aa[i] == 20)).sum()),
                  n_eligible_next_proline_edges=int((eligible & (aa[j] == rc.restype_order['P'])).sum()))
    return aa, selected, counts


def verify_cached_reconstruction(row, ca, r_c, r_n):
    """Verify the official family at actual stored CA/observed frame inputs."""
    valid = row['observed_mask']
    aa = row['chemical_aatype'][valid][None]
    ca = ca[:, valid].double()
    rotation = row['observed_R'][:, valid].double().expand(ca.shape[0], -1, -1, -1)
    require(bool(torch.isfinite(ca).all() and torch.isfinite(rotation).all()), 'nonfinite valid reconstruction input')
    atoms, atom_mask = reconstruct_atom37_from_frames(rotation, ca, aa.expand(ca.shape[0], -1),
                                                     torch.zeros(*ca.shape[:-1], 7, dtype=torch.float64))
    require(torch.equal(atoms[..., 1, :], ca), 'official cached-coordinate reconstruction changed CA')
    require(bool(atom_mask[..., [0, 1, 2]].all()), 'official cached-coordinate reconstruction missing atoms')
    errors = []
    for atom, radii in ((2, r_c), (0, r_n)):
        radius = radii[aa]
        error = (torch.linalg.vector_norm(atoms[..., atom, :]-ca, dim=-1)-radius).abs()
        # Two atom allowances sum to less than the main edge allowance.
        allowance = (ABSOLUTE_MARGIN_ANGSTROM/2 + ROUNDING_MULTIPLIER/2 * FLOAT32_EPS *
                     (torch.linalg.vector_norm(ca, dim=-1)+radius))
        require(bool((error <= allowance).all()), 'cached official radius exceeds registered numerical allowance')
        errors.append(error)
    return dict(ca_bitwise_preserved=True, n_residue_members=ca.shape[0]*ca.shape[1],
                max_radius_absolute_error_angstrom=float(torch.stack(errors).max()),
                all_radius_errors_within_registered_allowance=True)


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def from_totals(counts, raw_sum, conservative_sum, positive):
    n = counts['n_eligible_edge_members']
    return dict(counts=counts, n_impossible_edge_members=int(positive),
                sum_lower_bound_angstrom=float(raw_sum), sum_conservative_lower_bound_angstrom=float(conservative_sum),
                impossible_edge_fraction=ratio(positive, n),
                mean_lower_bound_angstrom=ratio(raw_sum, n),
                mean_conservative_lower_bound_angstrom=ratio(conservative_sum, n))


def score_ca(ca, aa, edges, topology_counts, r_c, r_n):
    ca = torch.as_tensor(ca)
    require(ca.ndim == 3 and ca.shape[0] > 0 and ca.shape[1:] == (len(aa), 3) and ca.is_floating_point(),
            'CA must be floating [M,L,3]')
    ca = ca.double()
    i, j = edges.T
    require(bool(torch.isfinite(ca[:, torch.unique(edges)]).all()), 'nonfinite eligible CA')
    proline = (aa[j] == rc.restype_order['P']).long()
    means = torch.tensor(rc.between_res_bond_length_c_n, dtype=torch.float64)[proline]
    stds = torch.tensor(rc.between_res_bond_length_stddev_c_n, dtype=torch.float64)[proline]
    result = peptide_lower_bound(ca[:, i], ca[:, j], r_c[aa[i]], r_n[aa[j]], means+TOLERANCE_SD*stds)
    members, n_edges = ca.shape[0], len(edges)
    per_member = []
    for k in range(members):
        counts = {key.replace('_edges', '_edge_members'): int(value) for key, value in topology_counts.items()}
        counts['n_eligible_edge_members'] = n_edges
        per_member.append(from_totals(counts, float(result['raw'][k].sum()),
                                     float(result['conservative'][k].sum()), int(result['impossible'][k].sum())))
    counts = {key: sum(member['counts'][key] for member in per_member) for key in per_member[0]['counts']}
    aggregate = from_totals(counts, float(result['raw'].sum()), float(result['conservative'].sum()),
                            int(result['impossible'].sum()))
    aggregate.update(n_members=members, per_member=per_member,
                     numerical_margin_min_angstrom=float(result['margin'].min()) if n_edges else None,
                     numerical_margin_max_angstrom=float(result['margin'].max()) if n_edges else None)
    return aggregate


def score_row(row, r_c, r_n):
    aa, edges, counts = prepare_edges(row)
    ca_routes = dict(oracle_CA=row['oracle_CA'], forecast_CA=row['forecast_CA'],
                     observed_MD_CA=row['observed_backbone'][..., 1, :], target_MD_CA=row['target_backbone'][..., 1, :])
    for name, value in ca_routes.items():
        require(tuple(value.shape) == ((12 if name == 'forecast_CA' else 1), len(aa), 3), f'{name} shape differs')
    return dict(metadata=dict(row['metadata']), topology=dict(sha256=row['topology']['sha256'], **counts),
                actual_official_reconstruction={name: verify_cached_reconstruction(row, ca, r_c, r_n)
                                                for name, ca in ca_routes.items()},
                routes={name: score_ca(ca, aa, edges, counts, r_c, r_n) for name, ca in ca_routes.items()})


def aggregate_rows(rows):
    """Pooled member-edge, equal-window, then equal-protein window means."""
    output = {}
    for route in ROUTES:
        records = [row['routes'][route] for row in rows]
        counts = {key: sum(record['counts'][key] for record in records) for key in records[0]['counts']}
        pooled = from_totals(counts, sum(r['sum_lower_bound_angstrom'] for r in records),
                             sum(r['sum_conservative_lower_bound_angstrom'] for r in records),
                             sum(r['n_impossible_edge_members'] for r in records))
        pooled.update(n_windows=len(rows), n_proteins=len({r['metadata']['protein'] for r in rows}),
                      n_members=sum(r['n_members'] for r in records), weighting='each eligible member-edge equally')
        valid = [r for r in records if r['impossible_edge_fraction'] is not None]
        equal_window = {metric: float(np.mean([r[metric] for r in valid])) if valid else None for metric in METRICS}
        equal_window.update(n_valid_windows=len(valid), n_windows=len(rows), weighting='member-edge mean within window; equal nonempty windows')
        proteins = defaultdict(list)
        for row in rows:
            if row['routes'][route]['impossible_edge_fraction'] is not None:
                proteins[row['metadata']['protein']].append(row['routes'][route])
        protein_means = {protein: {metric: float(np.mean([r[metric] for r in values])) for metric in METRICS}
                         for protein, values in sorted(proteins.items())}
        equal_protein = {metric: float(np.mean([r[metric] for r in protein_means.values()])) if protein_means else None
                         for metric in METRICS}
        equal_protein.update(n_valid_proteins=len(protein_means), n_valid_windows=len(valid),
                             weighting='member-edge mean within window, window mean within protein, equal proteins')
        output[route] = dict(all_members=pooled, equal_window=equal_window, equal_protein=equal_protein,
                             per_protein=protein_means)
    return output


def validate_cache(cache):
    require(cache['schema_version'] == 'backbone_readout_cache_v1', 'wrong cache schema')
    p = cache['protocol']
    expected = dict(mode='val', dataset_role='development_validation', seed=0, n_windows=200,
                    full_panel=True, complete_panel_windows=200, horizons=list(HORIZONS), max_residues=96,
                    context_frames=8, n_forecast_samples=12, forecast_observed_only=True,
                    before_target_read=True, optimizer_steps=0, test_opened=False, bioemu2_rerun=False,
                    frame_interval_ps=1000.)
    require(all(p.get(key) == value for key, value in expected.items()), 'cache protocol mismatch')
    done = cache['completion']
    require(done.get('frozen_inputs_verified') is True and bool(done.get('completed_utc'))
            and done['model_sha256_before'] == done['model_sha256_after'] == p['model_tensor_sha256'], 'incomplete frozen cache')
    rows = cache['rows']
    require(len(rows) == 200, 'must consume exactly 200 windows')
    grouped, keys = defaultdict(list), set()
    for row in rows:
        m = row['metadata']
        key = tuple(m[k] for k in ('protein', 'trajectory', 'anchor_frame', 'horizon'))
        require(key not in keys, 'duplicate window')
        keys.add(key)
        require(m['anchor_frame'] >= 7 and m['target_frame'] == m['anchor_frame']+m['horizon'], 'invalid temporal metadata')
        require(row['boundary']['forecast_before_target_read'] is True
                and row['boundary']['future_reencoding_for_forecast'] == 0, 'forecast boundary missing')
        grouped[m['protein']].append(m)
    require(len(grouped) == 50, 'must contain 50 proteins')
    for entries in grouped.values():
        require(sorted(m['horizon'] for m in entries) == list(HORIZONS), 'missing protein horizons')
        require(len({(m['trajectory'], m['anchor_frame'], m['temperature_k']) for m in entries}) == 1,
                'protein must have shared anchor and condition across horizons')
    counts = Counter(row['metadata']['temperature_k'] for row in rows)
    require(len(counts) == 5 and set(counts.values()) == {40}, 'temperature panel imbalance')
    return rows


def source_hashes():
    files = {Path(__file__).resolve(), REPO/'tests/test_fixed_ca_capacity.py',
             REPO/'protein_jepa_frame/vendored/PROVENANCE.json'}
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path:
            path = Path(path).resolve()
            if path.is_relative_to(REPO) and path.suffix == '.py' and path.is_file():
                files.add(path)
    return {str(path.relative_to(REPO)): sha256_file(path) for path in sorted(files)}


def protocol(cache_path, hashes, verification):
    return dict(schema_version=VERSION, registered_utc=datetime.now(timezone.utc).isoformat(),
                input_cache=str(cache_path.relative_to(REPO)), input_sha256=CACHE_SHA256,
                source_hashes=hashes, n_windows=200, n_proteins=50, horizons=list(HORIZONS),
                routes=list(ROUTES), forecast_members=12, other_route_members=1, compute_device='cpu',
                metric_dtype='float64 after official float32 template reconstruction',
                registration='written exclusively before loading cache; no evaluation or summary inputs',
                formula=dict(cn_ge='max(0, D-rC_i-rN_j)',
                             lower_bound='relu(D-rC_i-rN_j-(mu_next+12*sigma_next))',
                             conservative_lower_bound='relu(lower_bound-numerical_margin)',
                             impossible='lower_bound > numerical_margin', units='Angstrom', tolerance_sd=TOLERANCE_SD,
                             cn_mean_nonproline_proline=list(rc.between_res_bond_length_c_n),
                             cn_std_nonproline_proline=list(rc.between_res_bond_length_stddev_c_n)),
                numerical_margin=dict(formula='1e-4 Angstrom + 64*eps_float32*(norm(CA_i)+norm(CA_j)+rC_i+rN_j+mu_next+12*sigma_next)',
                                      absolute_angstrom=ABSOLUTE_MARGIN_ANGSTROM, multiplier=ROUNDING_MULTIPLIER,
                                      eps_float32=FLOAT32_EPS, selected_before_cache_load=True,
                                      scope='conservative engineering roundoff allowance; not formal interval-arithmetic certification'),
                masks='same observed-valid source C/CA and target N/CA cohort for all routes; target MD must be complete there',
                topology='explicit directed PSF edges only; no legacy renumbering or inferred adjacency',
                unknown_policy='next UNK excluded from typed CN prior, counted; previous UNK uses official GLY fallback radius',
                radius_verification=verification,
                actual_radius_verification='each stored CA route with observed_R and zero torsions; CA bitwise unchanged; each radius error <= 5e-5 Angstrom + 32*eps_float32*(norm(CA)+radius), so two allowances sum below the edge allowance',
                aggregation='pooled member-edge; equal nonempty window; equal protein of window means, by horizon and overall',
                primary_display='equal_protein; mean lower bound averages all eligible edges including zero bounds',
                inference='descriptive only; no confidence interval or threshold selection from outcomes',
                limitations=['Sufficient long-distance impossibility condition only; zero does not establish feasibility.',
                             'Ignores angular, short-distance, other bond and steric constraints; does not certify full T1.',
                             'Only fixed-CA official-template readout family; not a bound on an architecture allowed to move CA.',
                             'Oracle CA is future-derived diagnostic, forecast CA is context-only; MD CA routes are template-capacity references, not actual MD C-N loss.',
                             'Observed MD CA repeats across horizons and forecast members are dependent; no independent sample inference.'])


def run(cache_path, output):
    require(Path.cwd().resolve() == REPO, 'run from repository root')
    cache_path, output = Path(cache_path).resolve(), Path(output).resolve()
    require(cache_path == REPO/CACHE and output == REPO/OUTPUT, 'only registered frozen cache/output allowed')
    require(sha256_file(cache_path) == CACHE_SHA256, 'frozen cache SHA mismatch')
    protocol_path = output.with_suffix('.protocol.json')
    snapshot = output.with_suffix('.source_snapshot')
    require(not any(path.exists() for path in (output, protocol_path, snapshot)), 'refusing to overwrite capacity evidence')
    output.parent.mkdir(parents=True, exist_ok=True)
    r_c, r_n = official_template_radii()
    verification = verify_official_reconstruction(r_c, r_n)
    hashes = source_hashes()
    prereg = protocol(cache_path, hashes, verification)
    # Exclusive registration precedes deserialization / any data-dependent calculation.
    write_json_exclusive(protocol_path, prereg)
    snapshot.mkdir(exist_ok=False)
    for name, expected in hashes.items():
        destination = snapshot/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = (REPO/name).read_bytes()
        require(hashlib.sha256(data).hexdigest() == expected, f'source changed before snapshot: {name}')
        with destination.open('xb') as handle:
            handle.write(data)
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    rows = validate_cache(cache)
    scored = [score_row(row, r_c, r_n) for row in rows]
    result = dict(schema_version=VERSION, protocol=prereg, protocol_sha256=sha256_file(protocol_path),
                  template_radii=[dict(chemical_aatype=k, residue='UNK(GLY fallback)' if k == 20 else rc.restypes[k],
                                       ca_c_angstrom=float(r_c[k]), n_ca_angstrom=float(r_n[k])) for k in range(21)],
                  input_cache_protocol=cache['protocol'], input_cache_completion=cache['completion'],
                  per_window=scored, overall=aggregate_rows(scored),
                  by_horizon={str(h): aggregate_rows([r for r in scored if r['metadata']['horizon'] == h]) for h in HORIZONS})
    require(sha256_file(cache_path) == CACHE_SHA256, 'frozen cache changed during diagnostic')
    for name, expected in hashes.items():
        require(sha256_file(REPO/name) == expected and sha256_file(snapshot/name) == expected,
                f'source/snapshot changed during diagnostic: {name}')
    result['completion'] = dict(completed_utc=datetime.now(timezone.utc).isoformat(),
                                input_and_source_hashes_verified=True, no_evaluation_or_summary_read=True,
                                no_model_training_or_checkpoint_load=True, n_windows=len(scored), n_proteins=50)
    write_json_exclusive(output, result)
    print(json.dumps(dict(output=str(output.relative_to(REPO)), sha256=sha256_file(output),
                          overall_equal_protein={route: result['overall'][route]['equal_protein'] for route in ROUTES}), indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default=CACHE)
    parser.add_argument('--output', default=OUTPUT)
    args = parser.parse_args()
    run(args.cache, args.output)


if __name__ == '__main__':
    main()
