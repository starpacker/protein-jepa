"""Register, then preflight fixed TRAIN topology metadata; never load trajectories."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from protein_jepa_frame.data.manifest import TrajectoryRecord, load_manifest
from protein_jepa_frame.training.physical_distance_pair_support_v2 import (
    VERSION, canonical_sha256, file_stat, record_key, require, sha256_file,
    support_record, verify_provenance,
)

OUTPUT = Path('results/hs_joint_20261101/distance_pair_support_v2')
HISTORICAL = Path('results/hs_joint_20261101/ordered_path_v1/protocol.json')
HISTORICAL_SHA = 'a31b87128702cd796b25fbbef60c3bd1245304cfd369c668970e54c72c7235cb'
MANIFEST = Path('artifacts/mdcath_manifest_full/train.jsonl')
MANIFEST_SHA = '715532628f7f5795769be81643364cc309b345db9d32254b5e5fa692a83a91c0'
SOURCES = (
    'run_scripts/prepare_distance_pair_support_v2.py',
    'protein_jepa_frame/training/physical_distance_pair_support_v2.py',
    'protein_jepa_frame/data/backbone_topology_v2.py',
    'protein_jepa_frame/data/dataset.py', 'protein_jepa_frame/data/manifest.py',
    'protein_jepa_frame/config.py', 'protein_jepa_frame/utils/contracts.py',
    'protein_jepa_frame/vendored/openfold/np/residue_constants.py',
    'tests/test_distance_pair_support_v2.py', 'docs/DISTANCE_PAIR_SUPPORT_V2_20261102.md',
)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    encoded = json.dumps(value, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x') as stream:
        stream.write(encoded)


def plan_records(windows, records):
    """Resolve registered trajectory identities from TRAIN manifest metadata only."""
    lookup = {}
    for record in records:
        key = (str(record.path), record.protein_id, record.trajectory_id)
        require(key not in lookup, 'duplicate TRAIN manifest record')
        lookup[key] = record
    plans = {}
    for row in windows:
        record = lookup[(row['record_path'], row['protein'], row['trajectory'])]
        require(record.format == 'mdcath-hdf5' and record.temperature_k == row['temperature_k'],
                'registered TRAIN record format/temperature differs')
        metadata = record.metadata or {}
        plan = dict(record_key=record_key(record.path, record.protein_id),
            record_path=str(record.path.resolve()), protein=record.protein_id,
            domain_id=record.domain_id or record.protein_id,
            metadata={k: metadata[k] for k in ('protein_pdb_dataset','protein_psf_dataset','atom_count') if k in metadata})
        key = plan['record_key']
        if key in plans:
            require(plans[key] == plan, 'same record has conflicting topology metadata')
        plans[key] = plan
    return [plans[key] for key in sorted(plans)]


def register():
    require(sha256_file(HISTORICAL) == HISTORICAL_SHA, 'historical protocol changed')
    require(sha256_file(MANIFEST) == MANIFEST_SHA, 'TRAIN manifest changed')
    historical = json.loads(HISTORICAL.read_text())
    windows = historical['training_windows']
    require(len(windows) == 500 and [r['step'] for r in windows] == list(range(1,501)),
            'requires complete historical fixed500')
    require([r['horizon'] for r in windows] == [1,4,16,64]*125, 'historical horizon schedule differs')
    plans = plan_records(windows, load_manifest(MANIFEST))
    stats = {row['record_path']: file_stat(row['record_path']) for row in windows}
    require(stats == historical['selected_hdf5_stats'], 'historical HDF stats changed')
    inputs = {str(HISTORICAL): HISTORICAL_SHA, str(MANIFEST): MANIFEST_SHA}
    for arm in ('increment','absolute'):
        path = Path(f'results/hs_joint_20261101/distance_path_v1/{arm}.execution.json')
        failure = json.loads(path.read_text())
        require(failure['status'] == 'failed' and len(failure['jobs']) == 1
                and failure['jobs'][0]['kind'] == 'train' and failure['jobs'][0]['returncode'] == 1,
                'preserved v1 training failure required')
        inputs[str(path)] = sha256_file(path)
    sources = {path: sha256_file(path) for path in SOURCES}
    OUTPUT.mkdir(parents=True, exist_ok=False)
    snapshot = OUTPUT/'source_snapshot'
    for path, digest in sources.items():
        destination = snapshot/path
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = Path(path).read_bytes()
        with destination.open('xb') as stream:
            stream.write(data)
        require(sha256_file(destination) == digest, 'source changed during registration')
    protocol = dict(schema_version=VERSION, registered_utc=now(), max_residues=96,
        training_windows=windows, training_windows_sha256=canonical_sha256(windows),
        record_plans=plans, input_hashes=inputs, source_hashes=sources,
        source_snapshot=str(snapshot), selected_hdf5_stats=stats,
        policy='full PSF peptide graph; exclude undirected distances 0,1,2 before prefix96 crop or observed mask; disconnected pairs eligible',
        failure_policy='visit every registered unique record; retain every error; any error fails entire support; no replacement or fallback',
        changed_estimand='Topology-defined auxiliary pair support differs from v1 author-number rule; legacy model inputs and evaluation remain unchanged',
        trajectory_coordinates_loaded=False, model_loaded=False, validation_loaded=False, test_opened=False)
    verify_provenance(protocol)
    write_new(OUTPUT/'protocol.json', protocol)
    return protocol


def execute_records(protocol):
    """Continue after individual metadata failures; never substitute a record."""
    records, errors = {}, []
    for plan in protocol['record_plans']:
        try:
            record = TrajectoryRecord(path=Path(plan['record_path']), protein_id=plan['protein'],
                trajectory_id='metadata_only', format='mdcath-hdf5', domain_id=plan['domain_id'],
                metadata=plan['metadata'])
            records[plan['record_key']] = support_record(record, protocol['max_residues'])
        except Exception as error:
            errors.append(dict(record_key=plan['record_key'], record_path=plan['record_path'],
                               protein=plan['protein'], error=f'{type(error).__name__}: {error}'))
    return records, errors


def run():
    path = OUTPUT/'protocol.json'
    digest = sha256_file(path)
    protocol = json.loads(path.read_text())
    require(protocol['schema_version'] == VERSION, 'wrong support protocol')
    require(not any((OUTPUT/name).exists() for name in ('started.json','execution.json','support.json')),
            'support execution already attempted; no overwrite or retry')
    started = dict(pid=os.getpid(), command=sys.argv, started_utc=now(), protocol_sha256=digest)
    write_new(OUTPUT/'started.json', started)
    status = {**started, 'status':'running'}
    begin = time.monotonic()
    try:
        verify_provenance(protocol)
        records, errors = execute_records(protocol)
        # Preserve every record-level error even if a subsequent provenance
        # check detects drift and prevents writing a usable support artifact.
        status.update(records=len(records), errors=errors)
        require(sha256_file(path) == digest, 'protocol changed during execution')
        verify_provenance(protocol)
        document = dict(schema_version=VERSION, status='failed' if errors else 'completed',
            protocol_path=str(path), protocol_sha256=digest,
            training_windows=protocol['training_windows'], records=records, errors=errors,
            registered_record_count=len(protocol['record_plans']), completed_record_count=len(records),
            trajectory_coordinates_loaded=False, model_loaded=False, validation_loaded=False, test_opened=False)
        write_new(OUTPUT/'support.json', document)
        status.update(support_sha256=sha256_file(OUTPUT/'support.json'), records=len(records), errors=errors)
        require(not errors, 'metadata preflight failed; all individual errors preserved in support.json')
        status.update(status='completed', returncode=0)
    except BaseException as error:
        status.update(status='failed', returncode=1, error=f'{type(error).__name__}: {error}')
        raise
    finally:
        status.update(completed_utc=now(), seconds=time.monotonic()-begin)
        write_new(OUTPUT/'execution.json', status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('register','run'))
    args = parser.parse_args()
    result = register() if args.mode == 'register' else run()
    print(json.dumps(dict(mode=args.mode, status=result.get('status','registered')), allow_nan=False))


if __name__ == '__main__':
    main()
