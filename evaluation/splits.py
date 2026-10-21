"""Audit materialized chronological partitions using original-frame provenance."""
from __future__ import annotations
from pathlib import Path
from protein_jepa_frame.data.manifest import load_manifest


def _source_interval(record):
    m=record.metadata or {}
    required=('source_path','source_protein','original_frame_start','original_frame_stop')
    if any(k not in m for k in required):
        raise ValueError('chronological split requires original source/frame provenance')
    start,stop=int(m['original_frame_start']),int(m['original_frame_stop'])
    if start<0 or stop<=start or (record.num_frames is not None and stop-start!=record.num_frames):
        raise ValueError('invalid chronological original-frame interval; stop is exclusive')
    key=(str(Path(m['source_path']).resolve()),str(m['source_protein']),
         record.temperature_k,record.replica_id)
    return key,start,stop


def validate_chronological_split(train_manifest,later_manifest):
    """Require known physical trajectories and strictly later whole partitions."""
    train,later=load_manifest(train_manifest),load_manifest(later_manifest)
    prior={}
    for r in train:
        key,start,stop=_source_interval(r)
        prior.setdefault(key,[]).append((start,stop))
    for r in later:
        key,start,stop=_source_interval(r)
        if key not in prior:
            raise ValueError('chronological evaluation must use a training-visible physical trajectory')
        if any(end>start for _,end in prior[key]):
            raise ValueError('chronological train/later intervals overlap or are reversed')
    return dict(protocol='materialized_original_frame_intervals_stop_exclusive',
                train_records=len(train),later_records=len(later),overlap=0,
                original_sources=len(prior))


def audit_manifest_pair(earlier,later,split):
    if split in {'chronological','within_protein_chronological'}:
        return validate_chronological_split(earlier,later)
    if split in {'protein_holdout','protein_and_temperature_holdout'}:
        from protein_jepa_frame.data.protocol import validate_manifest_split
        return validate_manifest_split(earlier,later)
    return None
