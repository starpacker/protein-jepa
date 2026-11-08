"""Registered TRAIN32 endpoint-leaf contact bias diagnostic; no sampling or fit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
from evaluation.ensemble_metrics_ext import _align_to
from protein_jepa_frame.training.contact_ensemble_objective_v1 import contact_ensemble_brier
from protein_jepa_frame.training.train_lasp_ca_dynamics import contact_probability_loss, energy_score

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'results/hs_joint_20261101/contact_bias_train32_v1'
HISTORY = 'results/hs_joint_20261101/microstep_train32_v1'
REPORT = HISTORY + '/diagnostic.json'
PROTOCOL = HISTORY + '/protocol.json'
REPORT_SHA = '0bce069c70ac415b88f261e7710e4d78ad26fd1f3bc3855969c6c2e7ad11eee7'
PROTOCOL_SHA = '12d033a62a21046cadda39ba7f3035753d09922ee50fdb2b8db3f245d4297c34'
VERSION = 'contact_ensemble_bias_train32_v1'
HORIZONS = (1, 4, 16, 64)
SOURCES = (
    'run_scripts/diagnose_contact_ensemble_bias_v1.py',
    'tests/test_contact_ensemble_bias_v1.py',
    'tests/test_contact_ensemble_objective_v1.py',
    'docs/CONTACT_ENSEMBLE_BIAS_V1_20261102.md',
    'protein_jepa_frame/training/contact_ensemble_objective_v1.py',
    'protein_jepa_frame/training/train_lasp_ca_dynamics.py',
    'evaluation/ensemble_metrics_ext.py',
)
TERMS = ('coordinate_u_es', 'contact_plugin_20', 'variance_penalty_20',
         'contact_u_20', 'clash_40', 'total', 'corrected_total')
COMPARISONS = ('dot', 'cosine', 'norm_ratio', 'change_right_along_minus_left',
               'change_right_along_unit_minus_left')
PARITY = ('decomposition_value_abs', 'decomposition_gradient_l2', 'cross_member_value_abs',
          'cross_member_gradient_l2', 'legacy_plugin_value_abs', 'legacy_plugin_gradient_l2',
          'coordinate_reference_value_abs', 'coordinate_reference_gradient_l2',
          'corrected_gradient_identity_l2')
SCALAR_NAMES = (TERMS + ('contact_plugin', 'variance_penalty', 'contact_u', 'clash',
    'pair_variance_unbiased_mean', 'pair_mean_occupancy_mean', 'variance_penalty_over_plugin')
    + tuple('gradient_norm/' + name for name in TERMS)
    + tuple('penalty_vs/' + name + '/' + metric for name in TERMS for metric in COMPARISONS)
    + tuple('total_vs_corrected/' + metric for metric in COMPARISONS)
    + tuple('parity/' + name for name in PARITY))


class NumericalFailure(RuntimeError):
    """A nonfinite arithmetic result retains its selected window with nulls."""


def failure_row(message):
    return dict(status='numerical_failure', failure=message, n_members=12,
                n_valid_residues=None, n_pairs=None,
                scalars={name: None for name in SCALAR_NAMES})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator != 0 else None


def gradient_comparison(left, right):
    a, b = left.flatten(), right.flatten()
    na, nb = float(a.norm()), float(b.norm())
    dot = float(a @ b)
    return dict(dot=dot, cosine=safe_ratio(dot, na * nb),
                norm_ratio=safe_ratio(na, nb),
                change_right_along_minus_left=-dot,
                change_right_along_unit_minus_left=safe_ratio(-dot, na))


def canonical_pairs(mask, residue_index, chain_id):
    require(mask.ndim == 1 and mask.dtype == torch.bool, 'mask must be bool [L]')
    require(residue_index.shape == chain_id.shape == mask.shape, 'residue metadata shape differs')
    separation = (residue_index[:, None] - residue_index[None, :]).abs()
    same = chain_id[:, None] == chain_id[None, :]
    pairs = mask[:, None] & mask[None, :] & ((separation >= 3) | ~same)
    pairs &= torch.triu(torch.ones_like(pairs), diagonal=1)
    require(bool(pairs.any()), 'no valid nonlocal pairs')
    return pairs


def cross_member_brier(z, q):
    """Independent explicit ordered distinct-member product formula."""
    residual = z - q
    m = len(z)
    return torch.stack([(residual[i] * residual[j]).mean()
                        for i in range(m) for j in range(m) if i != j]).mean()


def diagnose_endpoint(samples, target, anchor, mask, residue_index, chain_id):
    """Float64 gradients with respect to a new projected-CA endpoint leaf only."""
    require(samples.ndim == 3 and samples.shape[0] == 12 and samples.shape[-1] == 3,
            'expected 12 endpoint members [12,L,3]')
    require(target.shape == anchor.shape == samples.shape[1:], 'target/anchor shape differs')
    require(mask.shape == samples.shape[1:2] and int(mask.sum()) >= 4, 'invalid residue mask')
    require(all(t.device.type == 'cpu' for t in (samples, target, anchor, mask, residue_index, chain_id)),
            'CPU tensors required')
    if not all(bool(torch.isfinite(t).all()) for t in (samples, target, anchor)):
        raise NumericalFailure('nonfinite endpoint/target/anchor coordinates')
    x = samples.detach().double().clone().requires_grad_(True)
    qcoords, fixed_anchor = target.detach().double(), anchor.detach().double()
    pairs = canonical_pairs(mask, residue_index, chain_id)
    # The exact frozen trainer alignment affects coordinate ES only. Distances
    # are rigid-motion invariant; parity with aligned target is checked below.
    with torch.no_grad():
        aligned = _align_to(qcoords[None], fixed_anchor, mask)
    distances = torch.cdist(x[None], x[None])[0]
    target_distances = torch.cdist(qcoords[None, None], qcoords[None, None])[0]
    z = torch.sigmoid((8 - distances[:, pairs]) / .5)
    q = torch.sigmoid((8 - target_distances[:, pairs]) / .5).mean(0)
    result = contact_ensemble_brier(z, q)
    old_target = torch.sigmoid((8 - torch.cdist(aligned[None], aligned[None])[0]) / .5).mean(0)[pairs]
    old_plugin = contact_probability_loss(z.mean(0), old_target, 'brier')
    cross_u = cross_member_brier(z, q)
    coord = energy_score(x[:, mask].flatten(1), aligned[:, mask].flatten(1))
    # Independent explicit Euclidean U-energy formula, normalized per dimension.
    flat, truth = x[:, mask].flatten(1), aligned[:, mask].flatten(1)
    coord_reference = (flat - truth).norm(dim=1).mean() / math.sqrt(flat.shape[1])
    coord_reference = coord_reference - .5 * torch.stack([
        (flat[i] - flat[j]).norm() for i in range(12) for j in range(i+1, 12)
    ]).mean() / math.sqrt(flat.shape[1])
    clash = torch.relu(3.5 - distances[:, pairs]).square().mean()
    terms = dict(coordinate_u_es=coord, contact_plugin_20=20 * result.plug_in_brier,
        variance_penalty_20=20 * result.variance_penalty,
        contact_u_20=20 * result.population_brier_u, clash_40=40 * clash)
    terms['total'] = coord + terms['contact_plugin_20'] + terms['clash_40']
    terms['corrected_total'] = terms['total'] - terms['variance_penalty_20']
    grad = lambda value: torch.autograd.grad(value, x, retain_graph=True)[0]
    gradients = {name: grad(value) for name, value in terms.items()}
    g_plugin, g_penalty, g_u = [grad(v) for v in result]
    if not all(bool(torch.isfinite(value).all()) for value in
               (*terms.values(), *gradients.values(), grad(cross_u), grad(old_plugin), grad(coord_reference))):
        raise NumericalFailure('nonfinite derived objective or gradient')
    checks = dict(
        decomposition_value_abs=float((result.plug_in_brier-result.population_brier_u-result.variance_penalty).abs().detach()),
        decomposition_gradient_l2=float((g_plugin-g_u-g_penalty).norm()),
        cross_member_value_abs=float((result.population_brier_u-cross_u).abs().detach()),
        cross_member_gradient_l2=float((g_u-grad(cross_u)).norm()),
        legacy_plugin_value_abs=float((result.plug_in_brier-old_plugin).abs().detach()),
        legacy_plugin_gradient_l2=float((g_plugin-grad(old_plugin)).norm()),
        coordinate_reference_value_abs=float((coord-coord_reference).abs().detach()),
        coordinate_reference_gradient_l2=float((gradients['coordinate_u_es']-grad(coord_reference)).norm()),
        corrected_gradient_identity_l2=float((gradients['corrected_total']-gradients['total']+gradients['variance_penalty_20']).norm()),
    )
    for left, right in ((result.plug_in_brier, result.population_brier_u + result.variance_penalty),
                        (result.population_brier_u, cross_u), (result.plug_in_brier, old_plugin),
                        (coord, coord_reference)):
        torch.testing.assert_close(left, right, atol=1e-10, rtol=1e-9)
    for left, right in ((g_plugin, g_u+g_penalty), (g_u, grad(cross_u)),
                        (g_plugin, grad(old_plugin)), (grad(coord), grad(coord_reference))):
        torch.testing.assert_close(left, right, atol=1e-10, rtol=1e-9)
    scalars = {name: float(value.detach()) for name, value in terms.items()}
    scalars.update(contact_plugin=float(result.plug_in_brier.detach()),
        variance_penalty=float(result.variance_penalty.detach()),
        contact_u=float(result.population_brier_u.detach()), clash=float(clash.detach()),
        pair_variance_unbiased_mean=float(z.var(0, unbiased=True).mean().detach()),
        pair_mean_occupancy_mean=float(z.mean(0).mean().detach()),
        variance_penalty_over_plugin=safe_ratio(float(result.variance_penalty.detach()), float(result.plug_in_brier.detach())))
    for name, gradient in gradients.items():
        require(bool(torch.isfinite(gradient).all()), 'nonfinite gradient: ' + name)
        require(bool((gradient[:, ~mask] == 0).all()), 'masked gradient nonzero')
        scalars['gradient_norm/' + name] = float(gradient.norm())
        comparison = gradient_comparison(gradients['variance_penalty_20'], gradient)
        for metric, value in comparison.items():
            scalars['penalty_vs/' + name + '/' + metric] = value
    for metric, value in gradient_comparison(gradients['total'], gradients['corrected_total']).items():
        scalars['total_vs_corrected/' + metric] = value
    scalars.update({'parity/' + key: value for key, value in checks.items()})
    if not all(value is None or math.isfinite(value) for value in scalars.values()):
        raise NumericalFailure('nonfinite scalar')
    require(set(scalars) == set(SCALAR_NAMES), 'scalar schema differs')
    return dict(status='completed', failure=None, n_members=12, n_valid_residues=int(mask.sum()),
                n_pairs=int(pairs.sum()), scalars=scalars)


def validate_raw(raw, metadata):
    require(raw['metadata'] == metadata, 'raw metadata differs')
    observed, state, target = raw['observed'], raw['state'], raw['target']
    mask = observed['residue_mask'].all(0).bool()
    length = len(mask)
    require(observed['coords'].shape == (4, length, 4, 3), 'observed shape differs')
    require(state['mask'].shape == (12, length) and torch.equal(state['mask'], mask[None].expand(12, -1)),
            'saved mask differs from observed mask')
    require(state['anchor'].shape == (1, length, 3), 'anchor shape differs')
    require(torch.equal(state['anchor'][0], observed['coords'][-1, :, 1]), 'saved anchor differs')
    require(target['coords'].shape == (1, length, 4, 3), 'target shape differs')
    require(bool(target['atom_mask'][0, mask].all()), 'target missing observed-valid atoms')
    for key in ('aatype', 'residue_index', 'chain_id'):
        require(torch.equal(target[key], observed[key]), 'target metadata differs: ' + key)
    require(target['time_ps'].shape == (1,) and observed['time_ps'].shape == (4,), 'time shape differs')
    require(float(target['time_ps'][0]-observed['time_ps'][-1]) == 1000 * metadata['horizon'], 'target interval differs')
    ca = raw['branches']['1']['CA']
    require(ca.shape == (12, metadata['horizon']+1, length, 3), 'm1 CA shape differs')
    require(ca.dtype == target['coords'].dtype == state['anchor'].dtype == torch.float32, 'saved precision differs')
    return ca[:, -1], target['coords'][0, :, 1], state['anchor'][0], mask, observed['residue_index'], observed['chain_id']


def summarize(rows):
    names = sorted(set().union(*(row['scalars'] for row in rows)))
    output = {}
    for name in names:
        proteins = defaultdict(list)
        for row in rows:
            value = row['scalars'].get(name)
            if value is not None:
                proteins[row['metadata']['protein']].append(value)
        means = {key: sum(values)/len(values) for key, values in proteins.items()}
        count = sum(map(len, proteins.values()))
        mean = sum(means.values())/len(means) if means else None
        output[name] = dict(equal_protein_mean=mean if count == len(rows) else None,
            available_subset_mean=mean, complete=count == len(rows), n_windows=count,
            expected_windows=len(rows), n_proteins=len(means), per_protein=means)
    return output


def historical_metadata():
    require(sha(REPO/REPORT) == REPORT_SHA, 'historical report changed')
    require(sha(REPO/PROTOCOL) == PROTOCOL_SHA, 'historical protocol changed')
    report, previous = [json.loads((REPO/path).read_text()) for path in (REPORT, PROTOCOL)]
    require(report['protocol_sha256'] == PROTOCOL_SHA, 'historical report/protocol mismatch')
    rows = report['per_window']
    selection = [row['metadata'] for row in rows]
    require(selection == previous['selection'], 'historical selection differs')
    require(len(rows) == 32 and len({row['protein'] for row in selection}) == 32, 'fixed32 budget differs')
    require(Counter(row['horizon'] for row in selection) == Counter({h:8 for h in HORIZONS}), 'horizon budget differs')
    for index, row in enumerate(rows, 1):
        require(row['metadata']['step'] == index, 'historical row order differs')
        require(row['branches']['1']['status'] == 'completed', 'historical m1 branch incomplete')
        require(row['raw_tensors']['path'] == HISTORY + f'/window_{index:02d}.pt', 'unexpected raw path')
    # Verify historical sources without opening their datasets/checkpoints.
    for path, digest in previous['source_hashes'].items():
        require(sha(REPO/path) == digest, 'historical source changed: ' + path)
    return rows, previous


def cpu_setup():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', "require CUDA_VISIBLE_DEVICES='' explicitly")
    require(Path.cwd().resolve() == REPO, 'run from repository root')
    torch.set_num_threads(2)
    require(not torch.cuda.is_initialized(), 'CUDA initialized')


def register():
    cpu_setup()
    require(not OUT.exists(), 'refuse existing diagnostic directory')
    rows, previous = historical_metadata()
    inputs = {REPORT: REPORT_SHA, PROTOCOL: PROTOCOL_SHA}
    for row in rows:
        path, digest = row['raw_tensors']['path'], row['raw_tensors']['sha256']
        require(sha(REPO/path) == digest, 'historical raw changed: ' + path)
        inputs[path] = digest
    sources = {path: sha(REPO/path) for path in SOURCES}
    protocol = dict(schema_version=VERSION, registered_utc=now(), input_hashes=inputs,
        source_hashes=sources, historical_source_hashes=previous['source_hashes'],
        windows=[dict(metadata=row['metadata'], raw_tensors=row['raw_tensors']) for row in rows],
        n_windows=32, n_proteins=32, n_members=12, horizons=list(HORIZONS), branch='1',
        arithmetic='saved float32 coordinates unchanged; CPU float64 conversion before all loss/gradient calculations',
        gradient_scope='new projected endpoint CA leaf [12,L,3]; no decoder, projection, latent or parameter backpropagation',
        pair_mask='observed-valid both residues; abs residue_index separation >=3 OR different chain; strict upper triangle',
        contact='z=sigmoid((8-distance)/0.5); q=same soft transform of one saved target; pair means',
        objective='coordinate U-energy + 20*plug-in contact Brier + 40*clash; corrected subtracts 20*sample_var(z,unbiased=True)/12',
        latent_limit='historical latent weight 0.1 excluded: latent ES is not a function of the independent endpoint CA leaf; these are coordinate-space component gradients, not the full training parameter gradient',
        directional_convention='penalty_vs/t dot = grad(20*penalty) dot grad(t); minus/unit-minus penalty direction changes t to first order by -dot / -dot divided by penalty gradient norm',
        estimand='conditional population mean soft contact probability squared error; iid member assumption for expectation interpretation only; algebra exact for fixed ensemble',
        interpretation='diversity suppression is a hypothesis; local endpoint-coordinate gradients cannot establish training causality or forecast improvement',
        aggregation='per-window, then equal-protein all32/by-horizon; complete estimate null if any row null; available subset explicitly labelled; no bootstrap or validation inference',
        failure_policy='retain numerical failures as null rows; shape/provenance/parity violations abort and record execution failure; no restart or overwrite',
        adaptive_status='previously examined fixed TRAIN windows, no new sampling or data selection',
        optimizer_steps=0, model_instantiations=0, decoder_calls=0, sampling_calls=0,
        test_opened=False, validation_opened=False, bioemu2_rerun=False, evaluation_protocol_changed=False,
        versions=dict(torch=torch.__version__, python=sys.version))
    OUT.mkdir(parents=True, exist_ok=False)
    for path, digest in sources.items():
        content = (REPO/path).read_bytes()
        require(hashlib.sha256(content).hexdigest() == digest, 'source changed during registration')
        destination = OUT/'source_snapshot'/path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(content)
    write_new(OUT/'protocol.json', protocol)
    verify_registered(protocol, sha(OUT/'protocol.json'))
    print(json.dumps(dict(registered=str(OUT/'protocol.json'), sha256=sha(OUT/'protocol.json'))), flush=True)


def verify_registered(protocol, protocol_sha):
    require(sha(OUT/'protocol.json') == protocol_sha, 'registration changed')
    for field in ('input_hashes', 'source_hashes', 'historical_source_hashes'):
        for path, digest in protocol[field].items():
            require(sha(REPO/path) == digest, 'registered file changed: ' + path)
            if field == 'source_hashes':
                require(sha(OUT/'source_snapshot'/path) == digest, 'snapshot changed: ' + path)


def run():
    cpu_setup()
    protocol_bytes = (OUT/'protocol.json').read_bytes()
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = json.loads(protocol_bytes)
    require(protocol['schema_version'] == VERSION, 'wrong protocol version')
    require(not (OUT/'execution.started.json').exists(), 'no automatic restart')
    started = time.monotonic()
    write_new(OUT/'execution.started.json', dict(pid=os.getpid(), started_utc=now(),
        argv=sys.argv, command=[sys.executable, *sys.argv], protocol_sha256=protocol_sha,
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES']))
    try:
        verify_registered(protocol, protocol_sha)
        rows = []
        for window in protocol['windows']:
            raw = torch.load(REPO/window['raw_tensors']['path'], map_location='cpu', weights_only=True)
            arguments = validate_raw(raw, window['metadata'])
            # Numerical failures retain the window, never a subset of members.
            try:
                row = diagnose_endpoint(*arguments)
            except NumericalFailure as error:
                row = failure_row(str(error))
            row.update(metadata=window['metadata'], raw_tensors=window['raw_tensors'])
            rows.append(row)
            write_new(OUT/f"window_{window['metadata']['step']:02d}.json", row)
            print(json.dumps(dict(completed=len(rows), total=32, status=row['status'])), flush=True)
            del raw, arguments
        require(len(rows) == 32, 'incomplete panel')
        require(not torch.cuda.is_initialized(), 'CUDA initialized')
        verify_registered(protocol, protocol_sha)
        result = dict(schema_version=VERSION, protocol_sha256=protocol_sha, n_windows=32,
            n_proteins=32, per_window=rows, overall=summarize(rows),
            by_horizon={str(h):summarize([row for row in rows if row['metadata']['horizon'] == h]) for h in HORIZONS},
            numerical_failure_windows=sum(row['status'] != 'completed' for row in rows),
            source_inputs_before_after_verified=True, completed_utc=now(), elapsed_seconds=time.monotonic()-started)
        write_new(OUT/'diagnostic.json', result)
    except BaseException as error:
        try:
            verify_registered(protocol, protocol_sha)
            postcheck = dict(passed=True, error=None)
        except BaseException as verification_error:
            postcheck = dict(passed=False, error=f'{type(verification_error).__name__}: {verification_error}')
        write_new(OUT/'execution.json', dict(pid=os.getpid(), returncode=130 if isinstance(error, KeyboardInterrupt) else 1,
            completed_utc=now(), seconds=time.monotonic()-started,
            error=f'{type(error).__name__}: {error}', frozen_after_failure=postcheck))
        raise
    write_new(OUT/'execution.json', dict(pid=os.getpid(), returncode=0, completed_utc=now(),
        seconds=time.monotonic()-started, result_sha256=sha(OUT/'diagnostic.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('register', 'run'), required=True)
    args = parser.parse_args()
    register() if args.mode == 'register' else run()


if __name__ == '__main__':
    main()
