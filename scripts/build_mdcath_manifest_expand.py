#!/usr/bin/env python3
"""Expand a mdCATH manifest to every domain on disk, PRESERVING its val/test.

Why: the 1000dom manifest covers 800 train domains, but mdCATH on disk has 5398.
Training used ``--max-train-samples 8000`` windows, i.e. ~0.1 % of the ~7.5M
available windows, and 4398 domains were never touched.  More data is the cheapest
axis with real headroom, but only if the evaluation split stays IDENTICAL -- so
this script copies the existing val/test files verbatim and appends every
remaining domain to train.

Usage:
  python scripts/build_mdcath_manifest_expand.py \
    --source artifacts/mdcath_manifest_1000dom \
    --output artifacts/mdcath_manifest_full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_mdcath_manifest import discover_domains, read_frame_counts  # noqa: E402

FRAME_INTERVAL_PS = 1000.0


def _ids(path: Path) -> set[str]:
    return {json.loads(line)["protein_id"] for line in path.read_text().splitlines() if line}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="artifacts/mdcath_manifest_1000dom")
    ap.add_argument("--output", default="artifacts/mdcath_manifest_full")
    ap.add_argument("--data-dir", default="/liuyutian/protein_jepa/datasets/mdCATH/repo/data")
    ap.add_argument("--max-new-domains", type=int, default=0,
                    help="Cap how many NEW domains are added (0 = all on disk)")
    ap.add_argument("--min-frames", type=int, default=10)
    args = ap.parse_args()

    src, out = Path(args.source), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    val_ids, test_ids = _ids(src / "val.jsonl"), _ids(src / "test.jsonl")
    held = val_ids | test_ids
    if held & _ids(src / "train.jsonl"):
        raise SystemExit("source manifest leaks domains across splits; refusing")

    # val/test are copied byte-for-byte: every existing number stays comparable.
    for split in ("val", "test"):
        (out / f"{split}.jsonl").write_text((src / f"{split}.jsonl").read_text())

    existing = _ids(src / "train.jsonl")
    disk = dict(discover_domains(Path(args.data_dir)))
    new_ids = sorted(set(disk) - held - existing)
    if args.max_new_domains > 0:
        new_ids = new_ids[: args.max_new_domains]

    n_new_traj = n_new_frames = 0
    skipped = []
    with (out / "train.jsonl").open("w") as fh:
        fh.write((src / "train.jsonl").read_text())
        for domain_id in new_ids:
            h5_path = disk[domain_id]
            try:
                counts = read_frame_counts(h5_path, domain_id)
            except Exception as exc:                       # noqa: BLE001
                skipped.append(f"{domain_id}: {exc}")
                continue
            if not counts:
                skipped.append(f"{domain_id}: no temperature/replica groups")
                continue
            for (temp, rep), n_frames in sorted(counts.items()):
                if n_frames < args.min_frames:
                    continue
                fh.write(json.dumps({
                    "h5_path": str(h5_path.resolve()),
                    "protein_id": domain_id,
                    "trajectory_id": f"{domain_id}_T{temp}_R{rep}",
                    "domain_id": domain_id,
                    "format": "mdcath-hdf5",
                    "dataset": "mdCATH",
                    "num_frames": n_frames,
                    "frame_interval_ps": FRAME_INTERVAL_PS,
                    "temperature_k": int(temp),
                    "replica_id": rep,
                    "metadata": {"hdf5_group": f"{domain_id}/{temp}/{rep}"},
                }) + "\n")
                n_new_traj += 1
                n_new_frames += n_frames

    print(f"source train domains : {len(existing)}")
    print(f"new train domains    : {len(new_ids) - len(skipped)}  (skipped {len(skipped)})")
    print(f"new trajectories     : {n_new_traj:,}")
    print(f"new frames           : {n_new_frames:,}")
    print(f"held-out (copied)    : {len(val_ids)} val / {len(test_ids)} test")
    print(f"written to           : {out}/{{train,val,test}}.jsonl")
    for s in skipped[:5]:
        print(f"  skipped {s}")


if __name__ == "__main__":
    main()
