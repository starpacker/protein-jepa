#!/usr/bin/env python3
"""Audit trajectory manifests and paired-window physical-time protocols.

This is a Gate-0 diagnostic.  It does not train a model or choose horizons.
It makes the exact frame indices, overlap, and physical time implied by a
configuration explicit so an invalid protocol can be rejected before a run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protein_jepa_frame.config import DataConfig, load_config
from protein_jepa_frame.data.dataset import _paired_context_offsets


PATH_FIELDS = (
    "h5_path",
    "npz_path",
    "trajectory_path",
    "pdb_path",
    "reference_pdb",
)


@dataclass(frozen=True)
class HorizonAudit:
    horizon_frames: int
    context_frames: list[int]
    future_frames: list[int]
    shared_frames: list[int]
    overlap_count: int
    overlap_ratio: float
    nonoverlap: bool
    first_to_first_ps: dict[str, float]
    endpoint_to_endpoint_ps: dict[str, float]
    context_to_future_gap_ps: dict[str, float]


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def audit_manifest(path: str | Path, max_path_checks: int = 10000) -> dict[str, Any]:
    """Return a compact, serializable audit of one JSONL manifest."""
    path = Path(path)
    records = 0
    proteins: set[str] = set()
    trajectories: set[str] = set()
    formats: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    intervals: Counter[str] = Counter()
    candidate_paths: set[Path] = set()

    for row in _iter_jsonl(path):
        records += 1
        protein = row.get("protein_id") or row.get("domain_id")
        trajectory = row.get("trajectory_id") or protein
        if protein is not None:
            proteins.add(str(protein))
        if trajectory is not None:
            trajectories.add(str(trajectory))
        formats[str(row.get("format", "unknown"))] += 1
        datasets[str(row.get("dataset", "unknown"))] += 1
        interval = row.get("frame_interval_ps")
        intervals["missing" if interval is None else str(float(interval))] += 1
        for field in PATH_FIELDS:
            value = row.get(field)
            if value:
                candidate_paths.add(Path(value))
                break

    checked_paths = sorted(candidate_paths, key=str)[:max_path_checks]
    missing_paths = [str(value) for value in checked_paths if not value.exists()]
    return {
        "path": str(path),
        "records": records,
        "proteins": len(proteins),
        "trajectories": len(trajectories),
        "datasets": dict(sorted(datasets.items())),
        "formats": dict(sorted(formats.items())),
        "frame_interval_ps": dict(sorted(intervals.items())),
        "unique_paths": len(candidate_paths),
        "checked_unique_paths": len(checked_paths),
        "missing_checked_paths": len(missing_paths),
        "missing_path_examples": missing_paths[:20],
    }


def _interval_values(manifest_audit: dict[str, Any]) -> list[float]:
    values = []
    for key in manifest_audit["frame_interval_ps"]:
        if key == "missing":
            continue
        values.append(float(key))
    return sorted(values)


def audit_paired_windows(
    data: DataConfig,
    frame_intervals_ps: Iterable[float],
) -> dict[str, Any]:
    """Expose the exact context/future indices and overlap for every horizon."""
    offsets = [int(value) for value in _paired_context_offsets(data).tolist()]
    intervals = sorted({float(value) for value in frame_intervals_ps})
    horizon_audits: list[HorizonAudit] = []

    for horizon in sorted({int(value) for value in data.paired_horizons_frames}):
        future = [horizon + value for value in offsets]
        shared = sorted(set(offsets).intersection(future))
        first_to_first = {
            str(interval): horizon * interval for interval in intervals
        }
        endpoint_to_endpoint = dict(first_to_first)
        context_to_future_gap = {
            str(interval): (future[0] - offsets[-1]) * interval
            for interval in intervals
        }
        horizon_audits.append(
            HorizonAudit(
                horizon_frames=horizon,
                context_frames=offsets,
                future_frames=future,
                shared_frames=shared,
                overlap_count=len(shared),
                overlap_ratio=len(shared) / max(1, len(offsets)),
                nonoverlap=not shared,
                first_to_first_ps=first_to_first,
                endpoint_to_endpoint_ps=endpoint_to_endpoint,
                context_to_future_gap_ps=context_to_future_gap,
            )
        )

    failures = [
        f"horizon={item.horizon_frames} shares {item.overlap_count}/{len(offsets)} frames"
        for item in horizon_audits
        if not item.nonoverlap
    ]
    if data.paired_nonoverlap_only and failures:
        failures.append("configuration declares paired_nonoverlap_only=true")

    return {
        "num_frames": data.num_frames,
        "frame_stride": data.frame_stride,
        "context_offsets": offsets,
        "context_raw_span_frames": offsets[-1] - offsets[0] if offsets else 0,
        "paired_nonoverlap_only": data.paired_nonoverlap_only,
        "horizons": [asdict(item) for item in horizon_audits],
        "zero_overlap": not any(failures),
        "failures": failures,
    }


def _default_output() -> Path:
    return Path("docs/audits/gate0_data_window_audit.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--manifest", action="append", default=[])
    parser.add_argument("--output", type=Path, default=_default_output())
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a non-zero status if any paired-window protocol overlaps",
    )
    args = parser.parse_args(argv)

    manifest_paths = [Path(value) for value in args.manifest]
    loaded_configs = []
    for config_path in args.config:
        config = load_config(config_path)
        if config.data.manifest_path:
            manifest_paths.append(Path(config.data.manifest_path))
        loaded_configs.append((config_path, config))

    unique_manifest_paths = list(dict.fromkeys(manifest_paths))
    manifests = [audit_manifest(path) for path in unique_manifest_paths]
    manifest_by_path = {str(Path(item["path"])): item for item in manifests}

    configurations = []
    strict_failures = []
    for config_path, config in loaded_configs:
        manifest = manifest_by_path.get(str(Path(config.data.manifest_path)))
        intervals = _interval_values(manifest) if manifest is not None else []
        paired = audit_paired_windows(config.data, intervals)
        configurations.append(
            {
                "path": str(config_path),
                "manifest_path": config.data.manifest_path,
                "training_mode": config.data.training_mode,
                "paired_window_audit": paired,
            }
        )
        strict_failures.extend(
            f"{config_path}: {failure}" for failure in paired["failures"]
        )

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifests": manifests,
        "configurations": configurations,
        "strict_failures": strict_failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2 if args.strict and strict_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
