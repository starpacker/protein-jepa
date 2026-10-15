from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from protein_jepa_frame.fastfolding.manifest import FastFoldingRecord, iter_frame_files, load_fastfolding_manifest
from protein_jepa_frame.fastfolding.pdb import (
    kabsch_rmsd,
    native_contact_fraction,
    native_contact_pairs,
    radius_of_gyration,
    read_ca_frame,
)


def _state_label(q_native: float, folded_q: float, unfolded_q: float) -> str:
    if q_native >= folded_q:
        return "folded"
    if q_native <= unfolded_q:
        return "unfolded"
    return "transition"


def derive_labels_for_record(
    record: FastFoldingRecord,
    stride: int,
    max_frames: int | None,
    max_scan_entries: int | None,
    folded_q: float,
    unfolded_q: float,
    direction_threshold: float,
    committor_horizon: int,
) -> list[dict]:
    reference = read_ca_frame(record.reference_pdb)
    contact_pairs = native_contact_pairs(reference.coords)
    frame_files = iter_frame_files(
        record,
        stride=stride,
        max_frames=max_frames,
        max_scan_entries=max_scan_entries,
    )
    rows = []
    for frame_index, path in frame_files:
        frame = read_ca_frame(path)
        q_native = native_contact_fraction(frame.coords, contact_pairs)
        rows.append(
            {
                "protein_id": record.protein_id,
                "trajectory_id": record.protein_id,
                "frame": int(frame_index),
                "pdb_path": str(path),
                "rmsd_to_ref": kabsch_rmsd(frame.coords, reference.coords),
                "q_native": q_native,
                "radius_of_gyration": radius_of_gyration(frame.coords),
                "state": _state_label(q_native, folded_q, unfolded_q),
            }
        )
    if not rows:
        return rows
    q = np.asarray([row["q_native"] for row in rows], dtype=np.float64)
    dq = np.zeros_like(q)
    if len(q) >= 2:
        dq[0] = q[1] - q[0]
        dq[-1] = q[-1] - q[-2]
    if len(q) >= 3:
        dq[1:-1] = 0.5 * (q[2:] - q[:-2])
    for i, row in enumerate(rows):
        row["dq_native"] = float(dq[i])
        if dq[i] > direction_threshold:
            row["direction"] = "folding"
        elif dq[i] < -direction_threshold:
            row["direction"] = "unfolding"
        else:
            row["direction"] = "stationary"
        future = q[i + 1 : i + 1 + committor_horizon]
        row["committor_proxy"] = None
        if future.size:
            folded_hits = np.nonzero(future >= folded_q)[0]
            unfolded_hits = np.nonzero(future <= unfolded_q)[0]
            first_folded = int(folded_hits[0]) if folded_hits.size else None
            first_unfolded = int(unfolded_hits[0]) if unfolded_hits.size else None
            if first_folded is not None and (first_unfolded is None or first_folded < first_unfolded):
                row["committor_proxy"] = 1
            elif first_unfolded is not None and (first_folded is None or first_unfolded < first_folded):
                row["committor_proxy"] = 0
    return rows


def derive_labels(
    manifest: str | Path,
    output: str | Path,
    stride: int = 100,
    max_frames_per_protein: int | None = 2000,
    max_scan_entries_per_protein: int | None = None,
    folded_q: float = 0.8,
    unfolded_q: float = 0.2,
    direction_threshold: float = 0.02,
    committor_horizon: int = 20,
) -> dict:
    records = load_fastfolding_manifest(manifest)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"proteins": 0, "frames": 0}
    with output.open("w") as handle:
        for record in records:
            rows = derive_labels_for_record(
                record,
                stride=stride,
                max_frames=max_frames_per_protein,
                max_scan_entries=max_scan_entries_per_protein,
                folded_q=folded_q,
                unfolded_q=unfolded_q,
                direction_threshold=direction_threshold,
                committor_horizon=committor_horizon,
            )
            counts["proteins"] += 1
            counts["frames"] += len(rows)
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {**counts, "output": str(output)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Derive read-only FastFolding downstream labels")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/fastfolding/manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/fastfolding/frame_labels.jsonl"))
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--max-frames-per-protein", type=int, default=2000)
    parser.add_argument("--max-scan-entries-per-protein", type=int, default=None)
    parser.add_argument("--folded-q", type=float, default=0.8)
    parser.add_argument("--unfolded-q", type=float, default=0.2)
    parser.add_argument("--direction-threshold", type=float, default=0.02)
    parser.add_argument("--committor-horizon", type=int, default=20)
    args = parser.parse_args(argv)
    summary = derive_labels(
        args.manifest,
        args.output,
        stride=args.stride,
        max_frames_per_protein=args.max_frames_per_protein,
        max_scan_entries_per_protein=args.max_scan_entries_per_protein,
        folded_q=args.folded_q,
        unfolded_q=args.unfolded_q,
        direction_threshold=args.direction_threshold,
        committor_horizon=args.committor_horizon,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
