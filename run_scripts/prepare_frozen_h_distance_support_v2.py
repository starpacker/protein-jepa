"""Register then preflight TRAIN-cache500 PSF metadata; no coordinate/model IO."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from protein_jepa_frame.data.manifest import TrajectoryRecord
from protein_jepa_frame.data.frozen_h_distance_support_v2 import (
    VERSION, canonical_sha256, full_file_stat, make_plan, registered_windows,
    require, sha256_file, verify_provenance,
)
from protein_jepa_frame.training.physical_distance_pair_support_v2 import support_record

OUTPUT = Path('results/hs_joint_20261101/frozen_h_distance_support_v2')
CACHE = Path('results/hs_joint_20261101/backbone_repair/readout_train500_v2.pt')
CACHE_SHA = 'd48faeb6e7385fa0b37a1daf440bf4708ed3472d83e2073b8a44d9073a216718'
SELECTION = CACHE.with_suffix('.selection.json')
SELECTION_SHA = '66d0613efc1302545a9a12105ffbf4af64101b824713495d6b24ea6bef4798e5'
CACHE_PROTOCOL = CACHE.with_suffix('.protocol.json')
CACHE_PROTOCOL_SHA = '0bb40561b9c37c1619f206cac92c0b324039e7524ad7368aa2fc2bd939ac0411'
FAILED_V1 = {
    'results/hs_joint_20261101/frozen_h_distance_support_v1/protocol.json':
        '3d5ee0296fec987587a9c51f1598f69b3444473cb1fb3316721e43454d412748',
    'results/hs_joint_20261101/frozen_h_distance_support_v1/support.json':
        'db270c9de07db9c8fdeb00a917989e8aa0c52c30c17ab1397d31fc71b5cc8cef',
    'results/hs_joint_20261101/frozen_h_distance_support_v1/execution.json':
        '82ba454b619eccbf7ac41c0df8016b9f0882b237b99ea608d3832507d27211c8',
}
SOURCES = (
    'run_scripts/prepare_frozen_h_distance_support_v2.py',
    'protein_jepa_frame/data/frozen_h_distance_support_v2.py',
    'tests/test_frozen_h_distance_support_v2.py', 'docs/FROZEN_H_DISTANCE_SUPPORT_V2_20261102.md',
    'protein_jepa_frame/training/physical_distance_pair_support_v2.py',
    'protein_jepa_frame/data/backbone_topology_v2.py',
    'protein_jepa_frame/data/dataset.py', 'protein_jepa_frame/data/manifest.py',
    'protein_jepa_frame/config.py', 'protein_jepa_frame/utils/contracts.py',
    'protein_jepa_frame/vendored/openfold/np/residue_constants.py',
)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    encoded = json.dumps(value,indent=2,allow_nan=False)+'\n'
    with Path(path).open('x') as stream:
        stream.write(encoded)


def register():
    inputs = {str(CACHE):CACHE_SHA,str(SELECTION):SELECTION_SHA,str(CACHE_PROTOCOL):CACHE_PROTOCOL_SHA,**FAILED_V1}
    for path,digest in inputs.items():
        require(sha256_file(path) == digest,'pinned input changed: '+path)
    selection = json.loads(SELECTION.read_text())
    historical = json.loads(CACHE_PROTOCOL.read_text())
    require(historical['protocol']['selection_file'] == str(SELECTION)
            and historical['protocol']['selection_sha256'] == SELECTION_SHA,
            'historical protocol/selection link differs')
    windows = registered_windows(selection,historical)
    stats = {w['record']['path']:full_file_stat(w['record']['path']) for w in windows}
    require(stats == historical['provenance']['data_file_stats'], 'historical HDF file stats changed')
    sources = {p:sha256_file(p) for p in SOURCES}
    OUTPUT.mkdir(parents=True,exist_ok=False)
    snapshot = OUTPUT/'source_snapshot'
    for path,digest in sources.items():
        destination = snapshot/path
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(Path(path).read_bytes())
        require(sha256_file(destination) == digest,'source changed during registration')
    protocol = dict(schema_version=VERSION,registered_utc=now(),windows=windows,
        windows_sha256=canonical_sha256(windows),max_residues=96,
        n_records=500,n_distinct_proteins=500,fit_indices=list(range(400)),
        readout_heldout_indices=list(range(400,500)),training_order=list(range(400))+list(range(100)),
        fit_horizon_counts={str(h):100 for h in (1,4,16,64)},
        readout_heldout_horizon_counts={str(h):25 for h in (1,4,16,64)},
        input_hashes=inputs,source_hashes=sources,source_snapshot=str(snapshot),selected_hdf5_stats=stats,
        pair_policy='full PSF graph excludes undirected hops0/1/2 before crop96 and observed mask; disconnected included; all lexicographic upper pairs',
        failure_policy='visit all500 registered metadata records; preserve every error; any error fails panel; no replacement or fallback',
        prior_failed_support=FAILED_V1,
        serialization_policy='normalize JSON tuple/list representation only; all fields, item order and exact historical topology SHA retained; no metadata value repair',
        boundary='readout-only protein holdout; encoders may have seen all500; no coordinates, cached H, checkpoints or models deserialized',
        trajectory_coordinates_loaded=False,cache_deserialized=False,model_loaded=False,
        validation_loaded=False,test_opened=False)
    verify_provenance(protocol)
    write_new(OUTPUT/'protocol.json',protocol)
    return protocol


def execute_records(protocol):
    plans,errors = [],[]
    for window in protocol['windows']:
        try:
            raw = dict(window['record']);raw['path'] = Path(raw['path'])
            support = support_record(TrajectoryRecord(**raw),protocol['max_residues'])
            plans.append(make_plan(window,support))
        except Exception as error:
            errors.append(dict(cache_row_index=window['cache_row_index'],metadata=window['metadata'],
                record_key=window['record_key'],error=f'{type(error).__name__}: {error}'))
    return plans,errors


def run():
    path = OUTPUT/'protocol.json';digest = sha256_file(path)
    protocol = json.loads(path.read_text())
    require(protocol['schema_version'] == VERSION,'wrong support schema')
    require(not any((OUTPUT/name).exists() for name in ('started.json','execution.json','support.json')),
            'support execution already attempted; no overwrite or retry')
    started = dict(pid=os.getpid(),command=sys.argv,started_utc=now(),protocol_sha256=digest)
    write_new(OUTPUT/'started.json',started)
    status = {**started,'status':'running'};begin=time.monotonic()
    try:
        verify_provenance(protocol)
        plans,errors = execute_records(protocol)
        status.update(completed_record_count=len(plans),errors=errors)
        require(sha256_file(path) == digest,'protocol changed during metadata execution')
        verify_provenance(protocol)
        doc = dict(schema_version=VERSION,status='failed' if errors else 'completed',
            protocol_path=str(path),protocol_sha256=digest,plans=plans,errors=errors,
            registered_record_count=len(protocol['windows']),completed_record_count=len(plans),
            trajectory_coordinates_loaded=False,cache_deserialized=False,model_loaded=False,
            validation_loaded=False,test_opened=False)
        write_new(OUTPUT/'support.json',doc)
        status.update(support_sha256=sha256_file(OUTPUT/'support.json'))
        require(not errors and len(plans)==500,'metadata preflight failed; all errors retained')
        status.update(status='completed',returncode=0)
    except BaseException as error:
        status.update(status='failed',returncode=1,error=f'{type(error).__name__}: {error}')
        raise
    finally:
        status.update(completed_utc=now(),seconds=time.monotonic()-begin)
        write_new(OUTPUT/'execution.json',status)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=('register','run'))
    args=parser.parse_args()
    result=register() if args.mode=='register' else run()
    print(json.dumps(dict(mode=args.mode,status=result.get('status','registered'))))


if __name__=='__main__':
    main()
