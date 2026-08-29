from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrajectoryRecord:
    path: Path
    protein_id: str
    trajectory_id: str
    format: str = "npz"
    num_frames: int | None = None
    frame_interval_ps: float | None = None
    first_frame_time_ps: float = 0.0
    temperature_k: float | None = None
    replica_id: str = ""
    dataset: str = ""
    domain_id: str = ""
    metadata: dict[str, Any] | None = None


def load_manifest(path: str | Path) -> list[TrajectoryRecord]:
    path = Path(path)
    records: list[TrajectoryRecord] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        raw: dict[str, Any] = json.loads(line)
        data_path = raw.get("npz_path") or raw.get("h5_path") or raw.get("trajectory_path")
        if data_path is None:
            raise ValueError(f"{path}:{line_no} missing npz_path/h5_path/trajectory_path")
        dataset = str(raw.get("dataset", ""))
        temperature_k = raw.get("temperature_k")
        if temperature_k is None and dataset.lower() == "atlas":
            temperature_k = 300.0
        records.append(
            TrajectoryRecord(
                path=Path(data_path),
                protein_id=str(raw.get("protein_id") or raw.get("id") or data_path),
                trajectory_id=str(raw.get("trajectory_id") or raw.get("protein_id") or data_path),
                format=str(raw.get("format") or ("npz" if raw.get("npz_path") else "protein-jepa-hdf5")),
                num_frames=raw.get("num_frames"),
                frame_interval_ps=raw.get("frame_interval_ps"),
                first_frame_time_ps=float(raw.get("first_frame_time_ps") or 0.0),
                temperature_k=float(temperature_k) if temperature_k is not None else None,
                replica_id=str(raw.get("replica_id", "")),
                dataset=dataset,
                domain_id=str(raw.get("domain_id") or raw.get("protein_id") or ""),
                metadata=raw.get("metadata") or {},
            )
        )
    if not records:
        raise ValueError(f"manifest {path} contains no records")
    return records
