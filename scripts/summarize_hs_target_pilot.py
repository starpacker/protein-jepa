"""Preregister and audit the fixed-step CA training-target comparison.

CPU-only statistics from completed, immutable artifacts. No model sampling,
training, test access, BioEmu rerun, checkpoint selection, or overwrite.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from scripts import summarize_hs_readout_pilot as shared


REPO = Path(__file__).resolve().parents[1]
BASE = Path('results/hs_joint_20261101/target_v1')
ARTIFACTS = Path('artifacts/hs_joint_20261101/target_v1')
INITIAL = Path('artifacts/hs_joint_20261101/spatial_probe_v1/spatial_exact.pt')
BIOEMU = Path('results/lasp_v19/bioemu2_complete.json')
BASE_GATE = Path('results/hs_joint_20261101/retention_v1/frozen.h_gate.json')
SCRIPT = Path('scripts/summarize_hs_target_pilot.py')
SHARED = Path('scripts/summarize_hs_readout_pilot.py')
VERSION = 'hs_matched_target_paired_summary_v1'
ARMS = ('endpoint', 'path_mixture')
HORIZONS = (1, 4, 16, 64)
TEMPERATURES = (320., 348., 379., 413., 450.)
GROUPS = ('overall', 'h1', 'h4', 'h16', 'h64', 'T320', 'T348', 'T379', 'T413', 'T450')
METRICS, SCORES, GEOMETRY = shared.METRICS, shared.SCORES, shared.GEOMETRY
T2A_METRICS = ('coverage', 'free_energy_mae_kcal', 'population_mae', 'population_tv', 'population_jsd')
BOOTSTRAPS, SEED = 10000, 20261101
JOBS = ('train', 'ensemble', 'endpoint', 'suite', 'matched8')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local(value):
    file = Path(value)
    file = (REPO / file).resolve() if not file.is_absolute() else file.resolve()
    require(file.is_relative_to(REPO.resolve()), 'path outside repository')
    require(file.name != 'test.jsonl', 'test manifest is sealed')
    return file


def label(value):
    return str(local(value).relative_to(REPO.resolve()))


def sha(value):
    digest = hashlib.sha256()
    with local(value).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def number(value):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), 'nonfinite/non-numeric statistic')
    return float(value)


class Inputs:
    def __init__(self):
        self.hashes = {}

    def track(self, value, expected=None):
        file = local(value)
        actual = sha(file)
        require(expected is None or actual == expected, f'SHA256 mismatch: {file}')
        require(file not in self.hashes or self.hashes[file] == actual, f'input changed: {file}')
        self.hashes[file] = actual
        return file

    def read(self, value):
        return json.loads(self.track(value).read_text())

    def verify(self):
        for file, digest in self.hashes.items():
            require(sha(file) == digest, f'input changed during summary: {file}')


def statistical_contract():
    directions = {m: 'higher' if m in shared.HIGHER_BETTER else 'lower' for m in METRICS}
    return dict(schema_version=VERSION, role='single_seed_adaptive_development_pilot',
        arms=list(ARMS), contrast='endpoint minus path_mixture',
        primary=dict(metric='coordinate_energy_score', horizons=[16, 64],
            definition='Within each of the same 50 proteins, average its 16 and 64 ns window scores; '
                       'subtract arms within protein, then average 50 paired protein differences.',
            n_windows=100, n_proteins=50, better='lower', unit='angstrom'),
        groups=list(GROUPS), secondary_endpoint_scores=list(SCORES),
        historical_seven_metrics=list(METRICS), historical_metric_directions=directions,
        historical_reference='endpoint draws versus future-path mixture including observed anchor; unchanged',
        bioemu2_comparisons='each arm minus saved BioEmu2, historical seven metrics only, same 200 keys',
        canonical_ca_geometry=list(GEOMETRY),
        geometry_directions=dict(ca_bond_invalid_fraction='lower', ca_nonlocal_clash_fraction='lower',
            ca_bond_mean_angstrom=None, ca_radius_gyration_angstrom=None),
        endpoint_units=dict(coordinate_energy_score='angstrom', physical_distance_energy_score='dimensionless distance/1 angstrom',
            contact_brier='dimensionless'),
        endpoint_estimator='Coordinate off-diagonal U; distance diagonal-inclusive V; empirical contact Brier. '
            'M=12 fixed. Single endpoint proper-score interpretation is expectation across windows.',
        endpoint_reference='only future endpoint fitted to observation anchor; future is scoring-only',
        bootstrap=dict(reps=BOOTSTRAPS, seed=SEED, unit='protein', interval='percentile_95',
            rng='numpy.default_rng; sorted protein IDs; same draws restarted for each comparison',
            averaging='equal windows within protein, then equal proteins; no intersection fallback',
            inference='marginal exploratory; no multiplicity, training-seed or adaptive-selection adjustment'),
        suite=dict(full=dict(n_windows=7200, proteins_per_temperature=12, replicas=5, anchors=2,
                    horizons=list(HORIZONS), modes=['correct', 'wrong', 'none'], n_samples=12),
            matched8=dict(n_windows=800, proteins_per_temperature=8, replicas=5, anchors=1,
                    horizons=list(HORIZONS), modes=['correct'], n_samples=12),
            report='Recompute equal-protein descriptive means/counts from all raw rows; preserve original '
                   'aggregate/T2A summaries as separately labelled legacy 4000-bootstrap seed17 output. '
                   'No new suite-cell CIs or selection; full raw files retained and hashed.',
            reference='T2A shared true-MD internal-distance centroids per trajectory; pooled samples are '
                      '96 per full-suite trajectory or 48 per matched8 trajectory; fixed k8/seed0.',
            bioemu2_t2a='Protected baseline directories may only be read and hashed, never modified. '
                        'Compare equal-protein means only if all trajectory/window keys, residue masks and '
                        'reference construction are verifiably matched; otherwise explicitly missing evidence. '
                        'Current seven-metric BioEmu JSON alone is insufficient. Any matched T2A comparison '
                        'is descriptive, without additional CIs or a new primary outcome.'),
        checkpoint_rule='both latest at exactly 500 optimizer steps; no best/validation selection',
        training=dict(seed=42, optimizer_steps=500, same_inputs_and_rng=True, grad_steps=64, n_samples=12,
            only_arm_arguments=['target', 'output'], frozen_encoder_decoder_component=True,
            target_choice_only=True, prediction_kind='endpoint', rmsf_weight=0., mode_weight=0.,
            mode_distribution_weight=0., distance_weight=0., latent_weight=.1,
            contact_objective='brier', contact_weight=20., clash_weight=40.),
        validation_loss='Arm-specific training targets differ; raw validation losses are retained separately '
                        'and never subtracted as a common forecast-loss comparison.',
        reused_H_gate=str(BASE_GATE), reused_gate_condition='Identical encoder, decoder and complete component '
            'configuration; no gate rerun or claim that frozen readout budget has improved.',
        legacy_geometry_and_contact_counts='descriptive means only, separate from canonical geometry',
        no_pass_or_noninferiority_threshold=True, test_opened=False, bioemu2_rerun=False,
        scientific_limits=['CI spanning zero is not equivalence.',
            'Standard200 temperature strata contain different proteins; not paired temperature response.',
            'All five temperatures were seen during training; suite is an input mechanism diagnostic.',
            'Target intervention does not identify S-memory benefit or improve frozen-H information.',
            'No matched-time BioEmu2 endpoint/path or T2B superiority claim.'])


def register(output=BASE / 'summary.protocol.json'):
    output = local(output)
    require(output == local(BASE / 'summary.protocol.json'), 'registration path is fixed')
    require(not output.exists(), 'refusing registration overwrite')
    for arm in ARMS:
        for kind in JOBS[1:]:
            require(not local(BASE / f'{arm}.{kind}.json').exists(), 'formal result exists before registration')
        require(not local(BASE / f'{arm}.execution.json').exists()
                and not local(ARTIFACTS / arm / 'checkpoint_latest.pt').exists(),
                'training/execution exists before registration')
    protocol = statistical_contract()
    protocol.update(registered_utc=datetime.now(timezone.utc).isoformat(),
        implementation_note='Statistical definitions frozen here before training. Summary implementation '
                            'may finish later; its source and dependencies are hashed in final output.',
        output=str(BASE / 'summary.json'))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(protocol, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return protocol


def groups_for(keys):
    keys = sorted(keys)
    groups = dict(overall=keys)
    groups.update({f'h{h}': [k for k in keys if k[4] == h] for h in HORIZONS})
    groups.update({f'T{int(t)}': [k for k in keys if k[2] == t] for t in TEMPERATURES})
    return groups


def paired(candidate, reference, keys, get_candidate, get_reference, direction=None):
    """Equal-protein pairing without dropping any missing or nonfinite observation."""
    require(len(keys) == len(set(keys)) and bool(keys), 'empty/duplicate pairing keys')
    require(direction in (None,'higher','lower'), 'invalid metric direction')
    require(all(k in candidate and k in reference for k in keys), 'incomplete paired support')
    for k in keys:
        number(get_candidate(candidate[k])); number(get_reference(reference[k]))
    result = shared.paired(candidate, reference, sorted(keys), get_candidate, get_reference,
        None if direction is None else (1 if direction == 'higher' else -1))
    result['better'] = direction
    return result


def comparisons(candidate, reference, groups, metrics, getters, directions=None):
    return {group: {metric: paired(candidate, reference, keys,
        lambda r, m=metric: getters[0](r, m), lambda r, m=metric: getters[1](r, m),
        None if directions is None else directions[metric]) for metric in metrics}
        for group, keys in groups.items()}


def analysis(ensemble, endpoint, baseline):
    keys = sorted(baseline)
    require(all(set(ensemble[a]) == set(endpoint[a]) == set(keys) for a in ARMS),
            'complete standard200 pairing required; no intersection fallback')
    groups = groups_for(keys)
    long_keys = [k for k in keys if k[4] in (16, 64)]
    counts = Counter(k[0] for k in long_keys)
    require(len(long_keys) == 100 and len(counts) == 50 and set(counts.values()) == {2},
            'primary requires both long horizons for all 50 proteins')
    get_score = lambda r, m: r['scores']['model'][m]
    get_seven = lambda r, m: r['lasp'][m]
    get_geometry = lambda r, m: r['mean_sample_geometry'][m]
    directions = {m: 'higher' if m in shared.HIGHER_BETTER else 'lower' for m in METRICS}
    primary = paired(endpoint[ARMS[0]], endpoint[ARMS[1]], long_keys,
        lambda r: get_score(r, SCORES[0]), lambda r: get_score(r, SCORES[0]), 'lower')
    primary.update(metric=SCORES[0], horizons=[16, 64], unit='angstrom', per_protein=[])
    for protein in sorted(counts):
        selected = [k for k in long_keys if k[0] == protein]
        values = [float(np.mean([get_score(endpoint[a][k], SCORES[0]) for k in selected])) for a in ARMS]
        primary['per_protein'].append(dict(protein=protein, candidate=values[0], reference=values[1],
            candidate_minus_reference=values[0]-values[1], n_windows=2))
    seven = dict(metric_version='empirical_w2_v2_pca_projection',
        reference_kind='future_path_mixture_including_anchor',
        units=dict(rmsf_mae='angstrom',rmsf_corr='dimensionless',root_mean_w2='angstrom',
            variance_w2='angstrom^2',md_pca_w2='angstrom',weak_contact_j='dimensionless',
            transient_contacts_j='dimensionless'),
        endpoint_vs_path_mixture=comparisons(ensemble[ARMS[0]], ensemble[ARMS[1]], groups,
            METRICS, (get_seven, get_seven), directions),
        arm_means={a: shared.means(ensemble[a], groups, METRICS, get_seven) for a in ARMS},
        bioemu2_comparisons={a: comparisons(ensemble[a], baseline, groups, METRICS,
            (get_seven, lambda r, m: r['bioemu2'][m]), directions) for a in ARMS},
        bioemu2_means=shared.means(baseline, groups, METRICS, lambda r, m: r['bioemu2'][m]),
        persistence_means=shared.means(ensemble[ARMS[0]], groups, METRICS, lambda r,m: r['persist'][m]))
    scores = dict(task_version='ca_conditional_endpoint_scores_v1',
        units=statistical_contract()['endpoint_units'],
        endpoint_vs_path_mixture=comparisons(endpoint[ARMS[0]], endpoint[ARMS[1]], groups,
            SCORES, (get_score, get_score), {m: 'lower' for m in SCORES}),
        arm_means={a: shared.means(endpoint[a], groups, SCORES, get_score) for a in ARMS},
        persistence_means=shared.means(endpoint[ARMS[0]], groups, SCORES, lambda r,m: r['scores']['persistence'][m]),
        persistence_comparisons={a: comparisons(endpoint[a], endpoint[a], groups, SCORES,
            (get_score, lambda r,m: r['scores']['persistence'][m]), {m:'lower' for m in SCORES}) for a in ARMS})
    geometry = dict(definition='Per-sample canonical CA geometry; bond interval [3.3,4.3] angstrom, '
        'nonlocal clash <2.5 angstrom. Equal-protein means, not pooled fractions.',
        units={m:'fraction' if 'fraction' in m else 'angstrom' for m in GEOMETRY},
        endpoint_vs_path_mixture=comparisons(endpoint[ARMS[0]], endpoint[ARMS[1]], groups,
            GEOMETRY, (get_geometry, get_geometry), statistical_contract()['geometry_directions']),
        arm_means={a: shared.means(endpoint[a], groups, GEOMETRY, get_geometry) for a in ARMS},
        target_means=shared.means(endpoint[ARMS[0]], groups, GEOMETRY, lambda r,m: r['target_geometry'][m]),
        persistence_means=shared.means(endpoint[ARMS[0]], groups, GEOMETRY, lambda r,m: r['persistence_geometry'][m]))
    descriptive = {}
    for name, metrics in (('geometry_vs_anchor', shared.LEGACY_GEOMETRY), ('contact_counts', shared.CONTACT_COUNTS)):
        descriptive[name] = {a: shared.means(ensemble[a], groups, metrics, lambda r,m,f=name: r[f][m]) for a in ARMS}
    descriptive['interpretation'] = ('Legacy anchor geometry is separate from canonical geometry: 3.0-angstrom '
        'clash threshold and angle change from observed anchor, not future-angle accuracy. Contact counts '
        'are means of per-window set sizes, not pooled counts. No better direction or CI is assigned.')
    return dict(primary_long_horizon_coordinate_UES=primary, seven_metrics=seven, endpoint_scores=scores,
        canonical_ca_geometry=geometry, legacy_descriptive=descriptive,
        group_sizes={g: dict(n_windows=len(ks), n_proteins=len({k[0] for k in ks})) for g,ks in groups.items()})


def audit_endpoint(data, arm, checkpoint):
    require(data['task_version'] == 'ca_conditional_endpoint_scores_v1'
        and data['dataset_role'] == 'development_validation' and data['prediction_kind'] == 'endpoint'
        and data['reference_kind'] == 'single_matched_future_endpoint'
        and data['future_information_given_to_sampler'] is False
        and data['checkpoint_step'] == 500 and data['checkpoint_training_target'] == arm,
        f'{arm}: endpoint task/target differs')
    require(data['model_tensor_sha256'] == checkpoint['inference_model_tensor_sha256'],
        f'{arm}: loaded inference tensors differ')
    shared.audit_eval_protocol(data['protocol'], arm)
    sampling = data['sampling']
    require((sampling['seed'], sampling['n_samples'], sampling['model_history_frames']) == (0, 12, 4),
        f'{arm}: endpoint sample budget differs')
    for name in ('s0_mode', 'noise_mix', 'noise_kernel', 'collective_mobility'):
        require(sampling[name] == checkpoint['args'][name], f'{arm}: sampling override: {name}')
    rows = shared.rows_for(data, arm+'/endpoint')
    for k, row in rows.items():
        seed = int(hashlib.sha256(f'{k[0]}|{k[1]}|{k[3]}|{k[4]}'.encode()).hexdigest()[:12],16) % (2**31-1)
        require(row['sampling_seed'] == seed and row['actual_n_samples'] == 12
            and row['observation_steps'] == [k[4]]*12, f'{arm}: endpoint draw contract differs')
        require(row['target_frame'] == k[3]+k[4] and row['context_frames'] == list(range(k[3]-7,k[3]+1))
            and row['horizon_ps'] == k[4]*checkpoint['step_ps'], f'{arm}: endpoint time differs')
        require(4 <= row['valid_residues'] <= 96 and row['nonlocal_pairs'] > 0, 'endpoint mask/pair count')
        for field in ('model', 'persistence'):
            shared.finite(row['scores'][field], SCORES, (arm,k,field))
        require(len(row['sample_geometry']) == 12, 'missing sample geometry')
        for values in [row['mean_sample_geometry'], row['target_geometry'], row['persistence_geometry'], *row['sample_geometry']]:
            shared.finite(values, GEOMETRY, (arm,k,'geometry'))
        for metric in GEOMETRY:
            require(math.isclose(row['mean_sample_geometry'][metric], np.mean([s[metric] for s in row['sample_geometry']]),
                rel_tol=1e-12, abs_tol=1e-12), 'geometry member average differs')
    return rows


def describe(rows, getters):
    """Descriptive equal-protein means, preserving support counts."""
    by_protein = defaultdict(list)
    for row in rows:
        by_protein[row['protein']].append(row)
    require(bool(by_protein), 'empty descriptive cell')
    return dict(n_windows=len(rows), n_proteins=len(by_protein),
        n_trajectories=len({(r['protein'],r.get('traj',r.get('trajectory'))) for r in rows}),
        means={m: float(np.mean([np.mean([number(getter(r)) for r in values])
            for _,values in sorted(by_protein.items())])) for m,getter in getters.items()})


def audit_suite(data, arm, kind):
    require(kind in ('suite', 'matched8'), 'unknown suite kind')
    n_proteins, n_anchors = (12,2) if kind == 'suite' else (8,1)
    modes = ['correct','wrong','none'] if kind == 'suite' else ['correct']
    args = data['args']
    expected = dict(temps=list(TEMPERATURES), modes=modes, wrong_temp=320., horizons=list(HORIZONS),
        proteins_per_temp=n_proteins, anchors_per_trajectory=n_anchors, n_samples=12,
        context_frames=8, max_residues=96, k_states=8, seed=0, t2a=True)
    require(all(args.get(k) == v for k,v in expected.items())
        and local(args['manifest']) == local(shared.VAL), f'{arm}/{kind}: suite arguments differ')
    require(data['noise_protocol']['version'] == 'shared-window-noise-v1'
        and data['noise_protocol']['shared_across_modes'] is True
        and data['noise_protocol']['seed_identity'] == ['protein_id','trajectory_id','anchor_frame','horizon'],
        f'{arm}/{kind}: shared-noise contract differs')
    indexed, cells = {}, defaultdict(list)
    for row in data['per_window']:
        k = shared.key(row)
        identity = (*k,row['mode'])
        require(identity not in indexed, 'duplicate suite key')
        require(row['mode'] in modes and k[2] in TEMPERATURES and k[4] in HORIZONS
            and float(row['T_star']) == k[2], 'suite row condition differs')
        expected_temp = k[2] if row['mode']=='correct' else (
            (450. if k[2]==320. else 320.) if row['mode']=='wrong' else None)
        require(row['temperature_used_k'] == expected_temp, 'wrong/none temperature is degenerate or differs')
        for field in ('model','persist'):
            shared.finite(row[field],METRICS,identity)
        shared.finite(row['geometry_vs_anchor'],shared.LEGACY_GEOMETRY,identity)
        shared.finite(row['position'],('kabsch_ca_rmsd','aligned_ca_rmsd',
            'persist_kabsch_ca_rmsd','persist_aligned_ca_rmsd'),identity)
        indexed[identity] = row; cells[k[2],row['mode']].append(row)
    expected_windows = n_proteins*5*n_anchors*4*5*len(modes)
    require(len(indexed) == data['n_windows'] == expected_windows
        and data['n_proteins'] == n_proteins and data['n_trajectories'] == n_proteins*25,
        f'{arm}/{kind}: incomplete suite panel')
    require(len({r['protein'] for r in indexed.values()})==data['n_proteins']
        and len({(r['protein'],r['traj']) for r in indexed.values()})==data['n_trajectories'],
        'suite raw distinct counts disagree')
    conditions = {}
    for T in TEMPERATURES:
        for mode in modes:
            rows = cells[T,mode]
            require(len(rows) == n_proteins*5*n_anchors*4
                and len({r['protein'] for r in rows}) == n_proteins, 'suite cell incomplete')
            proteins = {r['protein'] for r in rows}
            conditions[T,mode] = {shared.key(r) for r in rows}
            for protein in proteins:
                selected = [r for r in rows if r['protein']==protein]
                trajectories = {r['traj'] for r in selected}
                require(len(trajectories)==5, 'suite replica count differs')
                for trajectory in trajectories:
                    anchors = defaultdict(list)
                    for row in selected:
                        if row['traj']==trajectory: anchors[row['anchor_frame']].append(int(row['horizon']))
                    require(len(anchors)==n_anchors and all(sorted(hs)==list(HORIZONS) for hs in anchors.values()),
                        'suite anchor/horizon contract differs')
            require(conditions[T,mode] == conditions[T,'correct'], 'suite modes use different keys')
            require(proteins=={r['protein'] for r in cells[TEMPERATURES[0],'correct']},
                'suite temperatures use different proteins')
    t2a = data['t2a_per_protein']
    require(set(t2a)=={str(int(t)) for t in TEMPERATURES}, 'suite missing T2A temperatures')
    for T in TEMPERATURES:
        rows = t2a[str(int(T))]
        expected_traj = {(r['protein'],r['traj']) for r in cells[T,'correct']}
        require(len(rows)==n_proteins*5 and {(r['protein'],r['trajectory']) for r in rows}==expected_traj,
            'T2A trajectory support differs')
        for row in rows:
            shared.finite(row,T2A_METRICS,(arm,kind,T))
            require(row['n_model_samples']==n_anchors*4*12 and row['n_states']==8
                and row['n_ref_frames']>0, 'T2A sample/reference budget differs')
    getters = {m:(lambda r,m=m:r['model'][m]) for m in METRICS}
    getters.update({f'persist_{m}':(lambda r,m=m:r['persist'][m]) for m in METRICS})
    getters.update({m:(lambda r,m=m:r['position'][m]) for m in
        ('kabsch_ca_rmsd','aligned_ca_rmsd','persist_kabsch_ca_rmsd','persist_aligned_ca_rmsd')})
    getters.update(skill_rmsf=lambda r:r['persist']['rmsf_mae']-r['model']['rmsf_mae'],
        skill_position=lambda r:r['position']['persist_kabsch_ca_rmsd']-r['position']['kabsch_ca_rmsd'])
    result = dict(n_windows=expected_windows, n_proteins=n_proteins,
        by_temperature={str(int(T)):{mode:describe(cells[T,mode],getters) for mode in modes} for T in TEMPERATURES},
        by_horizon={str(h):{mode:describe([r for r in data['per_window'] if int(r['horizon'])==h
            and r['mode']==mode],getters) for mode in modes} for h in HORIZONS},
        overall={mode:describe([r for r in data['per_window'] if r['mode']==mode],getters) for mode in modes},
        t2a={T:describe(rows,{m:(lambda r,m=m:r[m]) for m in T2A_METRICS}) for T,rows in t2a.items()},
        original_report_summaries={k:data[k] for k in ('by_temperature','by_horizon','aggregate','t2a')},
        original_interval_note='Original suite CIs use 4000 protein-bootstrap draws, seed17; retained as '
            'legacy output, not the new 10000 paired comparisons. Descriptive means recomputed above.')
    for cell in result['t2a'].values():
        cell['n_trajectory_rows'] = cell.pop('n_windows')
        cell['row_unit'] = 'reference_trajectory'
    def check_cell(original, derived):
        require(all(original[k]==derived[k] for k in ('n_windows','n_proteins','n_trajectories')),
            'suite published counts differ')
        require(all(math.isclose(original[m]['mean'],v,rel_tol=1e-12,abs_tol=1e-12)
            for m,v in derived['means'].items()), 'suite published means differ')
    for T in TEMPERATURES:
        for mode in modes:
            original = data['by_temperature'][str(int(T))][mode]
            derived = result['by_temperature'][str(int(T))][mode]
            check_cell(original,derived)
        require(all(data['t2a'][str(int(T))][k]==result['t2a'][str(int(T))][k]
            for k in ('n_proteins','n_trajectories')), 'T2A published counts differ')
        for metric, value in result['t2a'][str(int(T))]['means'].items():
            require(math.isclose(data['t2a'][str(int(T))][metric]['mean'],value,rel_tol=1e-12,abs_tol=1e-12),
                'T2A published means differ')
    for mode in modes:
        check_cell(data['aggregate'][mode],result['overall'][mode])
        for h in HORIZONS:
            check_cell(data['by_horizon'][str(h)][mode],result['by_horizon'][str(h)][mode])
    return indexed, result


def expected_suite_keys(records, kind):
    """Replay only the frozen protocol's metadata selection, never coordinates."""
    require(kind in ('suite','matched8'), 'unknown suite kind')
    limit,anchors,modes = (12,(7,8),('correct','wrong','none')) if kind=='suite' else (8,(7,),('correct',))
    keys = set()
    for T in TEMPERATURES:
        selected = set()
        for record in records:
            if float(record['temperature_k'])!=T: continue
            protein = record['protein_id']
            if protein not in selected and len(selected)==limit: continue
            selected.add(protein)
            for anchor in anchors:
                require(record['num_frames']>anchor+max(HORIZONS), 'registered suite metadata too short')
                for h in HORIZONS:
                    for mode in modes:
                        key = (protein,record['trajectory_id'],T,anchor,h,mode)
                        require(key not in keys, 'duplicate manifest trajectory identity')
                        keys.add(key)
        require(len(selected)==limit, 'too few suite proteins in manifest')
    expected = 7200 if kind=='suite' else 800
    require(len(keys)==expected, 'suite manifest metadata count differs')
    return keys


def audit_protocol(protocol, registration, inputs):
    require(all(registration.get(k)==v for k,v in statistical_contract().items()),
        'statistical registration changed')
    require(protocol['schema_version']=='hs_target_pilot_v1'
        and tuple(protocol['arms'])==ARMS and protocol['optimizer_steps_per_arm']==500
        and protocol['prediction_kind']=='endpoint' and protocol['test_opened'] is False
        and protocol['bioemu2_rerun'] is False, 'unregistered experiment')
    require(protocol['checkpoint_selection']=='latest exactly500, no early stopping or validation selection',
        'checkpoint selection differs')
    require(local(protocol['statistics_protocol'])==local(BASE/'summary.protocol.json')
        and protocol['statistics_protocol_sha256']==inputs.hashes[local(BASE/'summary.protocol.json')],
        'runner did not pin statistical registration')
    require(datetime.fromisoformat(registration['registered_utc']) <= datetime.fromisoformat(protocol['registered_utc']),
        'statistics were registered after experiment')
    require(local(protocol['initial_checkpoint'])==local(INITIAL), 'initial checkpoint differs')
    required = {str(INITIAL), str(BIOEMU), str(BASE_GATE), shared.TRAIN, shared.VAL, str(BASE/'summary.protocol.json')}
    require(required <= set(protocol['input_hashes']), 'missing registered input hashes')
    for file,digest in protocol['input_hashes'].items(): inputs.track(file,digest)
    snapshot = local(protocol['source_snapshot'])
    require(snapshot==local(ARTIFACTS/'source_snapshot'), 'source snapshot path differs')
    for file,digest in protocol['source_hashes'].items():
        inputs.track(file,digest)
        inputs.track(snapshot/file,digest)
    for name,expected in dict(n_windows=200,n_proteins=50,n_samples=12,seed=0,horizons=list(HORIZONS)).items():
        require(protocol['evaluation'][name]==expected, 'registered standard200 budget differs')
    require(shared.BOOTSTRAPS==BOOTSTRAPS and shared.SEED==SEED, 'shared statistics implementation settings differ')


def audit_execution(arm, protocol, inputs):
    execution = inputs.read(BASE/f'{arm}.execution.json')
    require(execution.get('arm')==arm and execution.get('status')=='completed'
        and execution.get('completed_utc') and execution.get('frozen_inputs_verified') is True,
        f'{arm}: execution incomplete or failed')
    require([j['kind'] for j in execution['jobs']]==list(JOBS), f'{arm}: all five jobs required')
    immutable = dict(protocol['input_hashes'],**protocol['source_hashes'])
    latest = local(ARTIFACTS/arm/'checkpoint_latest.pt')
    inputs.track(latest,execution['latest_sha256'])
    for job in execution['jobs']:
        kind = job['kind']
        require(job.get('returncode')==0 and job.get('completed_utc')
            and isinstance(job.get('pid'),int) and not isinstance(job.get('pid'),bool) and job['pid']>0,
            f'{arm}/{kind}: missing real exit0 process')
        require(job['command']==protocol['per_arm_commands'][arm][kind], f'{arm}/{kind}: command changed')
        require(job['frozen_before']==immutable and job['frozen_after']==immutable,
            f'{arm}/{kind}: incomplete frozen-hash evidence')
        require(local(job['log'])==local(BASE/f'{arm}.{kind}.log'), 'log path differs')
        inputs.track(job['log'],job['log_sha256'])
        if kind!='train': inputs.track(BASE/f'{arm}.{kind}.json',job['output_sha256'])
    return execution


def component_identity(component):
    from protein_jepa_frame.utils.contracts import state_sha256
    config = {k:component[k] for k in ('checkpoint_kind','decoder_mode','calibration_args')}
    encoded = json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return dict(encoder=state_sha256(component['encoder']),decoder=state_sha256(component['delta_ca_decoder']),
        configuration_sha256=hashlib.sha256(encoded).hexdigest(),configuration=config,
        projection_steps=component['calibration_args']['bond_projection_steps'])


def audit_checkpoint(arm, checkpoint, initial, execution, protocol, inputs):
    import torch
    from protein_jepa_frame.utils.contracts import state_sha256
    require(checkpoint['checkpoint_kind']=='lasp_ca_dynamics' and checkpoint['step']==500,
        f'{arm}: expected fixed latest500')
    args = checkpoint['args']
    expected = dict(protocol['expected_checkpoint_arguments'],target=arm,output=str(ARTIFACTS/arm),
        component=str(INITIAL),init_kernel=str(INITIAL))
    require(all(args.get(k)==v for k,v in expected.items()), f'{arm}: checkpoint arguments differ')
    shared.check_cli_arguments(args,protocol['common_training_arguments'])
    contract = statistical_contract()['training']
    for field in ('seed','n_samples','grad_steps','rmsf_weight','mode_weight','mode_distribution_weight',
                  'distance_weight','latent_weight','contact_objective','contact_weight','clash_weight'):
        require(args[field]==contract[field], f'{arm}: preregistered loss/budget differs: {field}')
    require(args['steps']==500 and args['train_encoder'] is False and args['resume'] is False
        and args['stateless_data'] is True and args['horizons']==list(HORIZONS)
        and args['observed_readout_weight']==0 and checkpoint.get('target_encoder') is None,
        f'{arm}: unexpected training scope')
    require(local(args['manifest'])==local(shared.TRAIN) and local(args['val_manifest'])==local(shared.VAL),
        'training manifests differ')
    require(checkpoint['kernel_config']==initial['kernel_config'], f'{arm}: kernel physical configuration changed')
    require(shared.same_tree(checkpoint['component_checkpoint'],initial['component_checkpoint']),
        f'{arm}: frozen full component changed')
    identity = component_identity(checkpoint['component_checkpoint'])
    require(identity==protocol['initial_component_identity'], 'component identity differs')
    kernel_hash = state_sha256(checkpoint['kernel'])
    expected_audit = dict(step=500,actual_args=args,kernel_config=checkpoint['kernel_config'],
        kernel_tensor_sha256=kernel_hash,component_identity=identity,target_encoder_is_none=True,
        frozen_component_verified=True,h_gate_reused=str(BASE_GATE),h_gate_sha256=protocol['h_gate_reuse']['sha256'])
    require(execution['checkpoint_audit']==expected_audit, f'{arm}: execution checkpoint audit disagrees')
    combined = {f'bundle.encoder.{k}':v for k,v in checkpoint['component_checkpoint']['encoder'].items()}
    combined.update({f'bundle.decoder.{k}':v for k,v in checkpoint['component_checkpoint']['delta_ca_decoder'].items()})
    combined.update({f'kernel.{k}':v for k,v in checkpoint['kernel'].items()})
    require(all(bool(torch.isfinite(v).all()) for v in combined.values()), 'nonfinite model tensor')
    run = inputs.read(ARTIFACTS/arm/'run.json')
    require(run['args']==args, f'{arm}: run/checkpoint args differ')
    split = run['split_audit']
    require(all(split[k]==0 for k in ('protein_overlap','domain_overlap','source_file_overlap')), 'train/val overlap')
    for role,manifest in (('train',shared.TRAIN),('val',shared.VAL)):
        require(split[role+'_manifest_sha256']==protocol['input_hashes'][manifest], 'split manifest hash differs')
    records = [json.loads(line) for line in inputs.track(ARTIFACTS/arm/'metrics.jsonl').read_text().splitlines() if line]
    require(not any('resumed_from' in r for r in records), 'unexpected resume')
    validations = [r for r in records if r.get('step')==500 and 'val' in r]
    starts = [r for r in records if r.get('step')==0 and 'val' in r]
    require(len(validations)==len(starts)==1 and validations[0]['val']==checkpoint['validation'],
        'final checkpoint/log validation differs')
    require([r['step'] for r in records if 'train' in r]==[1,*range(10,501,10)], 'training log step budget differs')
    first = next(r for r in records if r.get('step')==1 and 'train' in r)
    require(first['horizon']==1, 'first update must have identical one-frame targets')
    # This validates each decomposition, but never treats different targets as a common loss.
    initial_val = shared.common_validation(starts[0]['val'],args)
    final_val = shared.common_validation(checkpoint['validation'],args)
    for summary in (initial_val,final_val):
        summary.pop('common_forecast_loss')
        for row in summary['by_horizon'].values(): row.pop('common_forecast_loss')
    return dict(checkpoint=str(ARTIFACTS/arm/'checkpoint_latest.pt'),checkpoint_sha256=execution['latest_sha256'],
        step=500,args=args,step_ps=checkpoint['kernel_config']['step_ps'],component_identity=identity,
        kernel_tensor_sha256=kernel_hash,inference_model_tensor_sha256=state_sha256(combined),
        first_training_step={k:first[k] for k in ('step','horizon','grad_norm','train')},
        initial_target_specific_validation=initial_val,final_target_specific_validation=final_val,
        frozen_component_and_kernel_configuration_verified=True)


def audit_bioemu(inputs):
    bio = inputs.read(BIOEMU)
    require(bio['baseline']=='bioemu2' and bio['metric_version']=='empirical_w2_v2_pca_projection'
        and bio['n_samples']==12 and local(bio['manifest'])==local(shared.VAL), 'BioEmu task/version differs')
    for name,expected in dict(seed=0,n_samples=12,max_residues=96,history_frames=8,
        max_anchors_per_trajectory=4,cutoff=8.,pca_k=3,weak_lo=.05,weak_hi=.5,trans_lo=.01,
        trans_hi=.99,horizons_ns=[1.,4.,16.,64.],manifest_order=False).items():
        require(bio['evaluation_args'][name]==expected, f'BioEmu adapter differs: {name}')
    rows = shared.rows_for(bio,'saved BioEmu2')
    for k,row in rows.items():
        shared.finite(row['bioemu2'],METRICS,k); shared.finite(row['persist'],METRICS,k)
    return bio,rows


def audit_pairing(ensemble, endpoint, baseline, documents):
    keys = set(baseline)
    require(all(set(ensemble[a])==set(endpoint[a])==keys for a in ARMS), 'unpaired standard200 keys')
    require(documents[ARMS[0]]['endpoint']['metric_semantics']==documents[ARMS[1]]['endpoint']['metric_semantics'],
        'endpoint score definitions differ')
    maximum = dict.fromkeys(METRICS,0.)
    for k in keys:
        first,second = [ensemble[a][k] for a in ARMS]
        pa,pb = [endpoint[a][k] for a in ARMS]
        require(first['persist']==second['persist'] and pa['scores']['persistence']==pb['scores']['persistence'],
            'reference persistence differs across arms')
        for field in ('target_frame','context_frames','horizon_ps','valid_residues','nonlocal_pairs',
                      'sampling_seed','observation_steps','target_geometry','persistence_geometry'):
            require(pa[field]==pb[field], f'endpoint reference/draw differs: {field}')
        for field in ('weak_n_true','transient_n_true'):
            require(first['contact_counts'][field]==second['contact_counts'][field], 'reference contact counts differ')
        for metric in METRICS:
            old,current = baseline[k]['persist'][metric],first['persist'][metric]
            require(math.isclose(old,current,abs_tol=2e-6,rel_tol=1e-6), 'saved BioEmu reference adapter differs')
            maximum[metric] = max(maximum[metric],abs(old-current))
    return maximum


def audit_reused_gate(protocol, initial, inputs):
    import torch
    gate = inputs.read(BASE_GATE)
    require(gate['purpose']=='true_H_readout_gate_only_not_forecast_evidence'
        and gate['test_opened'] is False, 'base H gate purpose differs')
    require(gate['encoder_tensor_sha256']==protocol['initial_component_identity']['encoder']
        and gate['decoder_tensor_sha256']==protocol['initial_component_identity']['decoder'],
        'base H gate component tensors differ')
    parent = torch.load(inputs.track(gate['checkpoint'],gate['checkpoint_sha256']),map_location='cpu',weights_only=False)
    require(shared.same_tree(parent['component_checkpoint'],initial['component_checkpoint']),
        'base H gate full component/configuration differs')
    require(inputs.hashes[local(BASE_GATE)]==protocol['h_gate_reuse']['sha256'], 'base H gate hash differs')
    return dict(path=str(BASE_GATE),sha256=inputs.hashes[local(BASE_GATE)],
        identical_full_component_verified=True,rerun=False,purpose=gate['purpose'],
        passed=gate['passed'],by_horizon=gate['by_horizon'],
        note='Original frozen readout gate retained unchanged; no new forecast, memory or gate-improvement claim.')


def bioemu_t2a_evidence(matched_rows, matched_documents, inputs):
    """Audit saved declarations without inventing absent historical reference hashes."""
    root = Path('results/lasp_full_v1/t2a_bioemu2')
    report = inputs.read(root/'bioemu2_t2a.json')
    validation = inputs.read('results/standard_eval/stopped_best/bioemu2_validation.json')
    require(report['source']=='bioemu2' and report['n_windows']==800 and report['n_trajectories']==200
        and report['n_proteins']==8 and report['completeness']['complete'] is True, 'saved BioEmu T2A incomplete')
    keys = set()
    for T in TEMPERATURES:
        data = inputs.read(root/f'bioemu2_T{int(T)}.json')
        require(data['n_windows']==160 and data['n_samples']==12 and data['n_proteins']==8, 'old T2A panel differs')
        selected = [shared.key(r) for r in data['per_window']]
        require(len(selected)==len(set(selected))==160 and all(k[2]==T for k in selected), 'duplicate/incorrect old keys')
        require({shared.key(r) for r in data['saved_sample_windows']}==set(selected), 'saved sample identities differ')
        keys.update(selected)
    paired_keys = {k[:-1] for k in matched_rows[ARMS[0]]}
    identity_match = keys==paired_keys and all({k[:-1] for k in matched_rows[a]}==keys for a in ARMS)
    require(identity_match, 'new matched8 and saved BioEmu 800 keys differ')
    declared = dict(max_residues=96,context_frames=8,max_frames=512,k_states=8,seed=0,
        proteins_per_temp=8,anchors_per_trajectory=1)
    require(all(report['args'][k]==v for k,v in declared.items()), 'saved T2A declared reference contract differs')
    for T,old_rows in report['per_protein'].items():
        old = {(r['protein'],r['trajectory']):r for r in old_rows}
        require(len(old)==len(old_rows)==40, 'saved T2A trajectory identities differ')
        for arm in ARMS:
            current = {(r['protein'],r['trajectory']):r for r in matched_documents[arm]['t2a_per_protein'][T]}
            require(set(old)==set(current), 'T2A trajectory pairing differs')
            for k in old:
                require(old[k]['n_model_samples']==current[k]['n_model_samples']==48
                    and old[k]['n_ref_frames']==current[k]['n_ref_frames']
                    and old[k]['n_states']==current[k]['n_states']==8,
                    'saved T2A reference/sample count differs')
    return dict(status='window_keys_and_declared_budget_match_reference_fingerprints_missing',
        paired_windows=800,paired_trajectories=200,unique_proteins=8,
        historical_reference_bitwise_verified=False,paired_metric_differences=None,
        missing_evidence=['historical per-trajectory residue-mask fingerprints',
            'historical reference coordinate/frame/centroid fingerprints and immutable input provenance'],
        old_validation_declaration=validation,declared_reference_contract=declared,
        historical_bioemu2_means_metadata_only=report['by_temperature'],
        reason='Same keys, budgets, n_ref_frames and an old masks_equal_reference assertion cannot certify '
            'the historical reference bitwise. No new BioEmu T2A paired superiority claim is made; '
            'model matched8 results are still reported completely. No baseline sampling or metric rerun.')


def build_summary(output=BASE/'summary.json'):
    import torch
    from protein_jepa_frame.utils.contracts import state_sha256
    output = local(output)
    require(output==local(BASE/'summary.json') and not output.exists(), 'refusing output overwrite or unregistered location')
    torch.set_num_threads(2)
    inputs = Inputs()
    inputs.track(SCRIPT); inputs.track(SHARED)
    registration = inputs.read(BASE/'summary.protocol.json')
    protocol = inputs.read(BASE/'protocol.json')
    # Fail before loading models/results if any arm still has unfinished jobs.
    executions = {a:audit_execution(a,protocol,inputs) for a in ARMS}
    audit_protocol(protocol,registration,inputs)
    initial = torch.load(inputs.track(INITIAL),map_location='cpu',weights_only=False)
    require(initial['checkpoint_kind']=='lasp_ca_dynamics' and initial['step']==0
        and initial['probe_provenance']['source_training_step']==protocol['source_training_step'], 'initial lineage differs')
    require(component_identity(initial['component_checkpoint'])==protocol['initial_component_identity']
        and state_sha256(initial['kernel'])==protocol['initial_kernel_sha256'], 'initial tensors differ')
    checkpoints = {a:torch.load(inputs.track(ARTIFACTS/a/'checkpoint_latest.pt'),map_location='cpu',weights_only=False) for a in ARMS}
    checked = {a:audit_checkpoint(a,checkpoints[a],initial,executions[a],protocol,inputs) for a in ARMS}
    normalized = [{k:v for k,v in checkpoints[a]['args'].items() if k not in ('target','output')} for a in ARMS]
    require(normalized[0]==normalized[1], 'training differs beyond target/output')
    require(checked[ARMS[0]]['first_training_step']==checked[ARMS[1]]['first_training_step'],
        'first h1 target-equivalent loss/gradient norm differs')
    require(torch.equal(checkpoints[ARMS[0]]['sampling_rng'],checkpoints[ARMS[1]]['sampling_rng']),
        'sampling RNG consumption differs')
    require(shared.same_tree(checkpoints[ARMS[0]]['rng_state'],checkpoints[ARMS[1]]['rng_state']),
        'training global RNG state differs')
    gate = audit_reused_gate(protocol,initial,inputs)
    documents,ensemble,endpoint,suite_rows,suite_reports = {},{},{},{},{}
    manifest_records = [json.loads(line) for line in inputs.track(shared.VAL).read_text().splitlines() if line]
    suite_keys = {kind:expected_suite_keys(manifest_records,kind) for kind in ('suite','matched8')}
    for arm in ARMS:
        documents[arm] = {kind:inputs.read(BASE/f'{arm}.{kind}.json') for kind in JOBS[1:]}
        for kind,data in documents[arm].items():
            require(local(data['checkpoint'])==local(ARTIFACTS/arm/'checkpoint_latest.pt'), 'evaluation checkpoint path differs')
            if kind in ('ensemble','endpoint'):
                require(data['checkpoint_sha256']==executions[arm]['latest_sha256']
                    and data['optimizer_steps']==0 and data['test_opened'] is False, 'evaluation identity differs')
                shared.audit_sources(data,protocol,inputs,arm+'/'+kind)
                if kind=='endpoint': shared.audit_provenance(data,protocol,inputs,arm+'/'+kind)
        ensemble[arm] = shared.audit_ensemble(documents[arm]['ensemble'],arm)
        endpoint[arm] = audit_endpoint(documents[arm]['endpoint'],arm,checked[arm])
        suite_rows[arm],suite_reports[arm] = {},{}
        for kind in ('suite','matched8'):
            suite_rows[arm][kind],suite_reports[arm][kind] = audit_suite(documents[arm][kind],arm,kind)
            require(set(suite_rows[arm][kind])==suite_keys[kind], 'suite raw keys differ from manifest protocol')
        for k,row in suite_rows[arm]['matched8'].items():
            require(row==suite_rows[arm]['suite'][k], 'same matched8/full-suite window is not reproduced exactly')
    for kind in ('suite','matched8'):
        require(set(suite_rows[ARMS[0]][kind])==set(suite_rows[ARMS[1]][kind]), 'suite arm keys differ')
        for k in suite_rows[ARMS[0]][kind]:
            a,b = [suite_rows[arm][kind][k] for arm in ARMS]
            require(a['persist']==b['persist'] and all(a['position'][m]==b['position'][m]
                for m in ('persist_kabsch_ca_rmsd','persist_aligned_ca_rmsd')), 'suite references differ across arms')
    bio,baseline = audit_bioemu(inputs)
    maximum = audit_pairing(ensemble,endpoint,baseline,documents)
    statistics = analysis(ensemble,endpoint,baseline)
    t2a_evidence = bioemu_t2a_evidence({a:suite_rows[a]['matched8'] for a in ARMS},
        {a:documents[a]['matched8'] for a in ARMS},inputs)
    report = dict(schema_version=VERSION,created_utc=datetime.now(timezone.utc).isoformat(),
        role=registration['role'],test_opened=False,bioemu2_rerun=False,
        n_windows=200,n_proteins=50,n_samples=12,
        statistical_registration=str(BASE/'summary.protocol.json'),experiment_protocol=str(BASE/'protocol.json'),
        bootstrap=registration['bootstrap'],scientific_limits=registration['scientific_limits'],
        checkpoint_selection=registration['checkpoint_rule'],checkpoint_audit=checked,
        paired_training_audit=dict(only_arguments_differ=['target','output'],same_final_sampling_rng=True,
            same_final_global_rng=True,same_stateless_training_data_configuration=True,
            first_h1_loss_terms_and_gradient_norm_bitwise_equal=True,
            input_identity_evidence='same train/val manifest hashes, stateless-data seed42, identical horizon '
                'cycle/500-step schedule and frozen loader source; no training-row identity list serialized'),
        execution_audit=dict(all_ten_jobs_exit0=True,frozen_sources_inputs_snapshots_verified=True),
        reused_H_gate=gate,endpoint_metric_semantics=documents[ARMS[0]]['endpoint']['metric_semantics'],
        suite_reports=suite_reports,bioemu2_matched8_reference_audit=t2a_evidence,
        bioemu2_adapter=dict(path=str(BIOEMU),sha256=inputs.hashes[local(BIOEMU)],
            evaluation_args=bio['evaluation_args'],original_sources_metadata_only=bio['sources'],
            persistence_max_abs_difference=maximum),
        validation_loss_note=registration['validation_loss'],**statistics)
    inputs.verify()
    report['input_sha256'] = {label(file):digest for file,digest in sorted(inputs.hashes.items())}
    with output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False); stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--out')
    args = parser.parse_args()
    if args.register:
        output = args.out or BASE / 'summary.protocol.json'
        register(output)
    else:
        output = args.out or BASE / 'summary.json'
        build_summary(output)
    print(json.dumps(dict(output=str(output), sha256=sha(output))))


if __name__ == '__main__':
    main()
