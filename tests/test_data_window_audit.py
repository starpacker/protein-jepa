from __future__ import annotations

import json
from pathlib import Path

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.dataset import validate_real_paired_window_protocol
from scripts.audit_data_and_windows import audit_manifest, audit_paired_windows


def test_paired_window_audit_detects_leapfrog_overlap():
    config = DataConfig(
        num_frames=8,
        frame_stride=4,
        paired_horizons_frames=(4, 8, 16),
        paired_nonoverlap_only=False,
    )

    audit = audit_paired_windows(config, frame_intervals_ps=[1000.0])

    assert [item["overlap_count"] for item in audit["horizons"]] == [7, 6, 4]
    assert [item["first_to_first_ps"]["1000.0"] for item in audit["horizons"]] == [
        4000.0,
        8000.0,
        16000.0,
    ]
    assert not audit["zero_overlap"]


def test_paired_window_audit_accepts_disjoint_windows():
    config = DataConfig(
        num_frames=8,
        frame_stride=4,
        paired_horizons_frames=(32, 48, 64),
        paired_nonoverlap_only=True,
    )

    audit = audit_paired_windows(config, frame_intervals_ps=[1000.0])

    assert audit["zero_overlap"]
    assert all(item["overlap_count"] == 0 for item in audit["horizons"])
    assert audit["horizons"][0]["context_to_future_gap_ps"]["1000.0"] == 4000.0


def test_manifest_audit_reports_missing_paths(tmp_path: Path):
    existing = tmp_path / "existing.h5"
    existing.touch()
    missing = tmp_path / "missing.h5"
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {
            "h5_path": str(existing),
            "protein_id": "p1",
            "trajectory_id": "t1",
            "format": "mdcath-hdf5",
            "dataset": "mdCATH",
            "frame_interval_ps": 1000.0,
        },
        {
            "h5_path": str(missing),
            "protein_id": "p2",
            "trajectory_id": "t2",
            "format": "mdcath-hdf5",
            "dataset": "mdCATH",
            "frame_interval_ps": 1000.0,
        },
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    audit = audit_manifest(manifest)

    assert audit["records"] == 2
    assert audit["proteins"] == 2
    assert audit["missing_checked_paths"] == 1


def test_real_training_protocol_rejects_shared_frames(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n")
    config = DataConfig(
        manifest_path=str(manifest),
        num_frames=8,
        frame_stride=4,
        paired_horizons_frames=(4, 8, 16),
    )

    try:
        validate_real_paired_window_protocol(config)
    except ValueError as exc:
        message = str(exc)
        assert "zero context/target frame overlap" in message
        assert "shares 7/8" in message
    else:
        raise AssertionError("overlapping real-data windows must be rejected")


def test_synthetic_smoke_protocol_may_overlap():
    config = DataConfig(
        manifest_path=None,
        num_frames=4,
        frame_stride=1,
        paired_horizons_frames=(2, 4),
    )

    validate_real_paired_window_protocol(config)
