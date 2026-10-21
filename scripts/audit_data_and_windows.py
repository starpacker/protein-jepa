"""Read-only manifest/legacy-window auditing, retained for protocol regression tests."""
from __future__ import annotations

from pathlib import Path

from protein_jepa_frame.data.dataset import _paired_context_offsets
from protein_jepa_frame.data.manifest import load_manifest


def audit_manifest(path):
    records = load_manifest(path)
    return {
        'records': len(records),
        'proteins': len({r.protein_id for r in records}),
        'missing_checked_paths': sum(not r.path.exists() for r in records),
    }


def audit_paired_windows(config, frame_intervals_ps):
    offsets = set(map(int, _paired_context_offsets(config)))
    horizons = []
    for horizon in sorted(set(config.paired_horizons_frames)):
        future = {horizon+x for x in offsets}
        horizons.append({
            'horizon_frames': horizon,
            'overlap_count': len(offsets & future),
            'first_to_first_ps': {str(float(dt)): horizon*dt for dt in frame_intervals_ps},
            'context_to_future_gap_ps': {
                str(float(dt)): (min(future)-max(offsets))*dt for dt in frame_intervals_ps
            },
        })
    return {'zero_overlap': all(h['overlap_count']==0 for h in horizons), 'horizons':horizons}
