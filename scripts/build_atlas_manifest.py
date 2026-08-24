from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REPLICA_RE = re.compile(r"_prod_(R\d+)_fit\.xtc$")


def iter_records(
    root: Path,
    *,
    limit_proteins: int | None = None,
    limit_records: int | None = None,
    num_frames: int,
    frame_interval_ps: float,
) -> list[dict]:
    records: list[dict] = []
    protein_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if limit_proteins is not None:
        protein_dirs = protein_dirs[:limit_proteins]
    for protein_dir in protein_dirs:
        protein_id = protein_dir.name
        topology_path = protein_dir / f"{protein_id}.pdb"
        if not topology_path.exists():
            continue
        for xtc_path in sorted(protein_dir.glob(f"{protein_id}_prod_R*_fit.xtc")):
            match = REPLICA_RE.search(xtc_path.name)
            if match is None:
                continue
            replica_id = match.group(1)
            records.append(
                {
                    "dataset": "atlas",
                    "protein_id": protein_id,
                    "domain_id": protein_id,
                    "trajectory_id": f"{protein_id}_{replica_id}",
                    "replica_id": replica_id,
                    "trajectory_path": str(xtc_path.resolve()),
                    "topology_path": str(topology_path.resolve()),
                    "format": "xtc",
                    "license": "CC-BY-NC-4.0",
                    "num_frames": num_frames,
                    "frame_interval_ps": frame_interval_ps,
                    "first_frame_time_ps": 0.0,
                    "coordinate_unit": "Angstrom",
                    "metadata": {
                        "source": "ATLAS",
                        "trajectory_kind": "protein_only_fit_xtc",
                    },
                }
            )
            if limit_records is not None and len(records) >= limit_records:
                return records
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a raw ATLAS JSONL manifest.")
    parser.add_argument("--root", type=Path, default=Path("/liuyutian/protein_jepa/datasets/ATLAS"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-proteins", type=int, default=None)
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=10001)
    parser.add_argument("--frame-interval-ps", type=float, default=10.0)
    args = parser.parse_args()

    records = iter_records(
        args.root,
        limit_proteins=args.limit_proteins,
        limit_records=args.limit_records,
        num_frames=args.num_frames,
        frame_interval_ps=args.frame_interval_ps,
    )
    if not records:
        raise SystemExit(f"no ATLAS records found under {args.root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(records),
                "proteins": len({record["protein_id"] for record in records}),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
