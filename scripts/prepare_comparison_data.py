"""Materialize small, whole-protein mdCATH partitions without temporal overlap.

Protein selection uses topology/identity only, never forecast errors. Physical
timestamps and original source-frame provenance survive NPZ materialization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.dataset import TrajectoryClipDataset
from protein_jepa_frame.data.manifest import load_manifest
from evaluation.splits import audit_manifest_pair


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--source', default='artifacts/mdcath_manifest_200dom')
    args = p.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'data_audit.json').exists():
        raise FileExistsError('data already locked; choose a new output')
    ds = TrajectoryClipDataset.__new__(TrajectoryClipDataset)
    ds.config = DataConfig(max_residues=None)
    ds._init_auxiliary_state(None)
    groups = {}
    selection = {}
    for role, n in [('train', 8), ('val', 4), ('test', 4)]:
        records = [r for r in load_manifest(Path(args.source) / f'{role}.jsonl')
                   if r.temperature_k == 320 and str(r.replica_id) in {'0', 'R0'}]
        # One physical trajectory per protein, ordered by complete topology size.
        unique = {r.protein_id: r for r in reversed(records)}
        ranked = sorted(unique.values(), key=lambda r: (len(ds._mdcath_topology(r).aatype), r.protein_id))
        chosen = ranked[:n]
        selection[role] = [dict(protein=r.protein_id, residues=len(ds._mdcath_topology(r).aatype)) for r in chosen]
        if len(chosen) != n:
            raise ValueError(f'insufficient {role} proteins')
        for r in chosen:
            F = r.num_frames or ds._read_num_frames(r)
            cuts = [('train', 0, int(.6 * F)), ('chrono_dev', int(.6 * F), int(.8 * F)),
                    ('chrono_test', int(.8 * F), F)] if role == 'train' else [(f'cross_{"dev" if role == "val" else "test"}', 0, F)]
            # Read only the topology-selected trajectory, no protein cropping.
            sample = ds._read_clip(r, np.arange(F))
            for name, start, stop in cuts:
                dest = out / 'npz' / name / (r.protein_id + '.npz')
                dest.parent.mkdir(parents=True, exist_ok=True)
                arrays = {k: sample[k].numpy() for k in
                          ('coords','atom_mask','residue_mask','aatype','residue_index','chain_id','time_ps')}
                for k in ('coords','atom_mask','residue_mask','time_ps'):
                    arrays[k] = arrays[k][start:stop]
                np.savez(dest, **arrays)
                stat = r.path.stat()
                metadata = dict(source_path=str(r.path.resolve()), source_protein=r.protein_id,
                                source_trajectory_id=r.trajectory_id, source_group=(r.metadata or {}).get('hdf5_group'),
                                original_frame_start=start, original_frame_stop=stop,
                                source_size_bytes=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
                                extracted_npz_sha256=sha(dest), split_role=name)
                row = dict(npz_path=str(dest), protein_id=r.protein_id, domain_id=r.domain_id,
                           trajectory_id=r.trajectory_id, format='npz', num_frames=stop-start,
                           frame_interval_ps=r.frame_interval_ps,
                           first_frame_time_ps=float(arrays['time_ps'][0]), temperature_k=r.temperature_k,
                           replica_id=r.replica_id, dataset=r.dataset, metadata=metadata)
                groups.setdefault(name, []).append(row)
    paths = {}
    for name, rows in groups.items():
        path = out / (name + '.jsonl')
        path.write_text(''.join(json.dumps(x) + '\n' for x in rows))
        paths[name] = str(path)
    # Combined development is explicitly a selection set, not a holdout claim.
    combined = out / 'development.jsonl'
    combined.write_text(Path(paths['chrono_dev']).read_text() + Path(paths['cross_dev']).read_text())
    audits = {}
    for a, b in [('train','chrono_dev'),('train','chrono_test'),('chrono_dev','chrono_test')]:
        audits[a + '__' + b] = audit_manifest_pair(paths[a], paths[b], 'chronological')
    for a, b in [('train','cross_dev'),('train','cross_test'),('cross_dev','cross_test')]:
        audits[a + '__' + b] = audit_manifest_pair(paths[a], paths[b], 'protein_holdout')
    report = dict(selection_rule='smallest complete topology then protein identity; 320K replica0 only',
                  selection=selection, partitions='train protein frames [0,264), dev[264,352), test[352,440)',
                  manifests={k: dict(path=v, sha256=sha(v)) for k, v in paths.items()}, audits=audits,
                  source_manifests={r: sha(Path(args.source)/f'{r}.jsonl') for r in ('train','val','test')},
                  limitations=['single temperature and replica; topology-size-selected small domains',
                               'raw HDF5 identity uses path/stat; extracted backbone NPZ content is hashed'])
    (out / 'data_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
