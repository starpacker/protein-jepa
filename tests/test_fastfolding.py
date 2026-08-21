from pathlib import Path

import json

from protein_jepa_frame.fastfolding.manifest import build_manifest, load_fastfolding_manifest
from protein_jepa_frame.fastfolding.tasks import derive_labels
from protein_jepa_frame.fastfolding.downstream import build_window_samples, load_label_rows, read_backbone_frame


def _pdb_text(offset: float = 0.0) -> str:
    rows = []
    atom_id = 1
    for i in range(1, 7):
        x = float(i * 3) + offset
        for atom, dx in [("N", -1.0), ("CA", 0.0), ("C", 1.0), ("O", 1.5)]:
            rows.append(
                f"ATOM  {atom_id:5d} {atom:<4s} ALA A{i:4d}    "
                f"{x + dx:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n"
            )
            atom_id += 1
    rows.append("END\n")
    return "".join(rows)


def test_fastfolding_manifest_and_labels_are_derived_outside_source(tmp_path: Path):
    source = tmp_path / "FastFolding"
    protein = source / "Toy-0-protein"
    complex_dir = protein / "complex"
    complex_dir.mkdir(parents=True)
    (protein / "Toy-0-protein_ref.pdb").write_text(_pdb_text(0.0))
    for frame, offset in [(0, 0.0), (10, 0.2), (20, 4.0), (30, 0.1)]:
        (complex_dir / f"Toy-0-protein_frame{frame}.pdb").write_text(_pdb_text(offset))

    output_dir = tmp_path / "artifacts" / "fastfolding"
    manifest_path = output_dir / "manifest.jsonl"
    labels_path = output_dir / "labels.jsonl"

    records = build_manifest(source, manifest_path)
    loaded = load_fastfolding_manifest(manifest_path)
    summary = derive_labels(manifest_path, labels_path, stride=1, max_frames_per_protein=10, committor_horizon=2)

    assert len(records) == 1
    assert loaded[0].protein_id == "Toy-0-protein"
    assert loaded[0].num_frame_files == 4
    assert summary["frames"] == 4
    rows = [json.loads(line) for line in labels_path.read_text().splitlines()]
    assert {row["frame"] for row in rows} == {0, 10, 20, 30}
    assert {"rmsd_to_ref", "q_native", "direction", "committor_proxy"} <= set(rows[0])
    assert not (source / "manifest.jsonl").exists()
    assert not (source / "labels.jsonl").exists()


def test_fastfolding_downstream_builds_context_windows(tmp_path: Path):
    source = tmp_path / "FastFolding"
    protein = source / "Toy-0-protein"
    complex_dir = protein / "complex"
    complex_dir.mkdir(parents=True)
    (protein / "Toy-0-protein_ref.pdb").write_text(_pdb_text(0.0))
    for frame, offset in [(0, 0.0), (10, 0.2), (20, 4.0), (30, 0.1), (40, 0.0)]:
        (complex_dir / f"Toy-0-protein_frame{frame}.pdb").write_text(_pdb_text(offset))
    manifest_path = tmp_path / "artifacts" / "manifest.jsonl"
    labels_path = tmp_path / "artifacts" / "labels.jsonl"
    build_manifest(source, manifest_path)
    derive_labels(manifest_path, labels_path, stride=1, max_frames_per_protein=10, committor_horizon=2)

    rows = load_label_rows(labels_path)
    windows = build_window_samples(rows, num_frames=3)
    frame = read_backbone_frame(windows[0]["window"][0]["pdb_path"])

    assert len(rows) == 5
    assert len(windows) == 3
    assert [row["frame"] for row in windows[0]["window"]] == [0, 10, 20]
    assert frame.coords.shape == (6, 4, 3)
    assert frame.atom_mask.all()
