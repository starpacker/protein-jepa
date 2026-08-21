from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path


FRAME_RE = re.compile(r"_frame(\d+)\.pdb$", re.IGNORECASE)


@dataclass(frozen=True)
class FastFoldingRecord:
    protein_id: str
    protein_dir: str
    complex_dir: str
    reference_pdb: str
    num_frame_files: int
    min_frame: int | None
    max_frame: int | None
    dataset: str = "FastFolding"
    format: str = "fastfolding-pdb-dir"


def _frame_index(name: str) -> int | None:
    match = FRAME_RE.search(name)
    return int(match.group(1)) if match else None


def scan_protein_dir(protein_dir: Path, max_scan: int | None = None) -> FastFoldingRecord | None:
    """Scan a FastFolding protein directory read-only."""
    complex_dir = protein_dir / "complex"
    reference = protein_dir / f"{protein_dir.name}_ref.pdb"
    if not complex_dir.is_dir() or not reference.is_file():
        return None
    count = 0
    min_frame: int | None = None
    max_frame: int | None = None
    with os.scandir(complex_dir) as entries:
        for entry in entries:
            if max_scan is not None and count >= max_scan:
                break
            frame = _frame_index(entry.name)
            if frame is None:
                continue
            count += 1
            min_frame = frame if min_frame is None else min(min_frame, frame)
            max_frame = frame if max_frame is None else max(max_frame, frame)
    return FastFoldingRecord(
        protein_id=protein_dir.name,
        protein_dir=str(protein_dir),
        complex_dir=str(complex_dir),
        reference_pdb=str(reference),
        num_frame_files=count,
        min_frame=min_frame,
        max_frame=max_frame,
    )


def build_manifest(
    root: str | Path,
    output: str | Path,
    max_scan_per_protein: int | None = None,
    protein_ids: set[str] | None = None,
) -> list[FastFoldingRecord]:
    root = Path(root)
    output = Path(output)
    records: list[FastFoldingRecord] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        if protein_ids is not None and entry.name not in protein_ids:
            continue
        record = scan_protein_dir(entry, max_scan=max_scan_per_protein)
        if record is not None and record.num_frame_files > 0:
            records.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    return records


def load_fastfolding_manifest(path: str | Path) -> list[FastFoldingRecord]:
    records = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            records.append(FastFoldingRecord(**json.loads(line)))
    if not records:
        raise ValueError(f"manifest {path} contains no FastFolding records")
    return records


def iter_frame_files(
    record: FastFoldingRecord,
    stride: int = 1,
    max_frames: int | None = None,
    max_scan_entries: int | None = None,
) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    seen = 0
    with os.scandir(record.complex_dir) as entries:
        for entry in entries:
            if max_scan_entries is not None and seen >= max_scan_entries:
                break
            seen += 1
            frame = _frame_index(entry.name)
            if frame is not None:
                files.append((frame, Path(record.complex_dir) / entry.name))
    files.sort(key=lambda item: item[0])
    if stride > 1:
        files = files[::stride]
    if max_frames is not None and len(files) > max_frames:
        if max_frames <= 0:
            return []
        step = max(1, len(files) // max_frames)
        files = files[::step][:max_frames]
    return files


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a read-only FastFolding manifest")
    parser.add_argument("--root", type=Path, default=Path("/liuyutian/folddata/FastFolding"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/fastfolding/manifest.jsonl"))
    parser.add_argument("--max-scan-per-protein", type=int, default=None)
    parser.add_argument("--protein", action="append", default=None)
    args = parser.parse_args(argv)
    records = build_manifest(
        args.root,
        args.output,
        args.max_scan_per_protein,
        protein_ids=set(args.protein) if args.protein else None,
    )
    print(json.dumps({"output": str(args.output), "num_records": len(records)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
