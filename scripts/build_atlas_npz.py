#!/usr/bin/env python3
"""ATLAS XTC -> per-replica NPZ, resampled to the mdCATH time axis.

Design decisions (see STAGE_ABC_REDESIGN_PLAN.md §4.1/§4.2):

  * mdCATH stores one frame per 1000 ps (1 ns).  ATLAS stores one frame per
    10 ps.  To mix the two datasets under a *single* absolute-time contract we
    resample ATLAS with stride=100 so each stored frame is 1000 ps apart --
    i.e. every dataset frame is "1 ns".  The downstream history/horizon
    arithmetic (4/8/16/32 ns ...) then means the same thing for both.

  * Each replica (R1/R2/R3) becomes its OWN npz record.  The old converter
    blindly concatenated the three replicas into one fake 30003-frame
    trajectory, which (a) creates a non-monotonic / discontinuous time axis at
    the replica boundary and (b) lets paired-window sampling draw physically
    meaningless cross-replica pairs (§4.2: replica boundaries are not ordinary
    adjacent transitions).

  * aatype is written in the dataset loader's AA_ALPHABET ordering
    (A=0, C=1, D=2, ...), NOT the 3-letter order used by the old converter.
    Mixing the two orderings silently scrambles the identity embedding.

  * time_ps is materialised per frame so the loader never has to guess the
    interval and can never fall back to "frame index == picoseconds".

Usage:
  python scripts/build_atlas_npz.py \
      --atlas-dir  /liuyutian/protein_jepa/datasets/ATLAS \
      --output-dir /liuyutian/protein_jepa/datasets/ATLAS_NPZ_1ns \
      --stride 100 --max-proteins 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Loader ordering: ACDEFGHIKLMNPQRSTVWY (+X).  MUST match
# protein_jepa_frame.data.dataset.AA_ALPHABET / AA_TO_INDEX.
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
AA_TO_INDEX = {aa: i for i, aa in enumerate(AA_ALPHABET)}

BACKBONE_ATOMS = ("N", "CA", "C", "O")

# ATLAS native sampling interval, verified from the xtc time stamps
# (t.time = 0,10,20,...,100000 ps; unique(diff)=10 ps).
ATLAS_NATIVE_INTERVAL_PS = 10.0
MDCATH_INTERVAL_PS = 1000.0


def _aatype_from_names(residue_names) -> np.ndarray:
    out = np.empty(len(residue_names), dtype=np.int64)
    for i, name in enumerate(residue_names):
        one = THREE_TO_ONE.get(str(name).strip().upper()[:3], "X")
        out[i] = AA_TO_INDEX[one]
    return out


def _extract_backbone(traj, stride: int):
    """Return coords [T,L,4,3], residue_names, residue_numbers from an mdtraj traj."""
    if stride > 1:
        traj = traj[::stride]

    top = traj.topology
    n_idx, ca_idx, c_idx, o_idx, keep = [], [], [], [], []
    for ri, residue in enumerate(top.residues):
        found = {}
        for atom in residue.atoms:
            name = atom.name.strip().upper()
            if name in BACKBONE_ATOMS:
                found.setdefault(name, atom.index)
        if "CA" not in found:
            continue  # residue without CA (rare) is dropped
        keep.append(residue)
        n_idx.append(found.get("N", -1))
        ca_idx.append(found["CA"])
        c_idx.append(found.get("C", -1))
        o_idx.append(found.get("O", -1))

    L = len(ca_idx)
    if L == 0:
        raise ValueError("no CA atoms found")

    xyz = traj.xyz  # [T, N_atoms, 3]
    T = xyz.shape[0]
    coords = np.zeros((T, L, 4, 3), dtype=np.float32)
    atom_mask = np.zeros((T, L, 4), dtype=bool)
    for i in range(L):
        for slot, idx in enumerate((n_idx[i], ca_idx[i], c_idx[i], o_idx[i])):
            if idx >= 0:
                coords[:, i, slot] = xyz[:, idx]
                atom_mask[:, i, slot] = True
            elif slot == 1:
                atom_mask[:, i, slot] = True  # CA always present

    # Fill any missing N/C/O by the CA position so the geometry features see no
    # NaN; the atom_mask still records which slots were real.
    for i in range(L):
        for slot in (0, 2, 3):
            if not atom_mask[0, i, slot]:
                coords[:, i, slot] = coords[:, i, 1]

    residue_names = [r.name for r in keep]
    residue_numbers = [r.resSeq for r in keep]
    return coords, residue_names, residue_numbers, T, L


def process_protein(protein_dir: Path, output_dir: Path, stride: int,
                    force: bool, time_offset_ps: float = 0.0) -> list[dict]:
    """Convert all replicas of one ATLAS protein to per-replica npz files."""
    import mdtraj

    protein_id = protein_dir.name
    pdb_files = sorted(protein_dir.glob("*.pdb"))
    xtc_files = sorted(protein_dir.glob("*_prod_R*_fit.xtc"))
    if not pdb_files or not xtc_files:
        print(f"  SKIP {protein_id}: no PDB/XTC")
        return []
    pdb_path = str(pdb_files[0])

    interval_ps = ATLAS_NATIVE_INTERVAL_PS * stride
    records: list[dict] = []

    for xtc_file in xtc_files:
        replica_id = xtc_file.stem.split("_prod_")[-1].replace("_fit", "")  # R1/R2/R3
        out_path = output_dir / f"{protein_id}_{replica_id}.npz"
        if out_path.exists() and not force:
            records.append({
                "protein_id": protein_id, "replica_id": replica_id,
                "npz_path": str(out_path), "status": "skipped",
                "frame_interval_ps": interval_ps,
            })
            continue
        try:
            traj = mdtraj.load(str(xtc_file), top=pdb_path)
            coords, res_names, res_nums, T, L = _extract_backbone(traj, stride)
            native_times = np.asarray(traj.time, dtype=np.float64)[::stride]
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {protein_id}/{replica_id}: {exc}")
            continue

        if native_times.shape[0] != coords.shape[0]:
            native_times = np.arange(coords.shape[0], dtype=np.float64) * interval_ps
        time_ps = native_times.astype(np.float32) + np.float32(time_offset_ps)

        aatype = _aatype_from_names(res_names)
        chain_id = np.zeros(L, dtype=np.int64)
        residue_index = np.asarray(res_nums, dtype=np.int64)
        # Sanitise residue_index into the loader's embedding range [0, max_res).
        if residue_index.min() < 0:
            residue_index = residue_index - residue_index.min()

        np.savez_compressed(
            str(out_path),
            coords=coords,
            atom_mask=np.broadcast_to(
                np.isfinite(coords).all(axis=-1), coords.shape[:-1]
            ).copy(),
            aatype=aatype,
            residue_index=residue_index,
            chain_id=chain_id,
            time_ps=time_ps,
            frame_interval_ps=np.float32(interval_ps),
            first_frame_time_ps=np.float32(time_ps[0]),
            temperature_k=np.float32(300.0),
        )
        records.append({
            "protein_id": protein_id,
            "replica_id": replica_id,
            "npz_path": str(out_path),
            "num_frames": int(coords.shape[0]),
            "num_residues": int(L),
            "frame_interval_ps": interval_ps,
            "temperature_k": 300.0,
            "status": "ok",
        })
        print(f"    {protein_id}/{replica_id}: {coords.shape} "
              f"({coords.shape[0]} frames, L={L}, dt={interval_ps} ps)")
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="ATLAS XTC -> 1 ns/frame NPZ")
    ap.add_argument("--atlas-dir", default="/liuyutian/protein_jepa/datasets/ATLAS")
    ap.add_argument("--output-dir", default="/liuyutian/protein_jepa/datasets/ATLAS_NPZ_1ns")
    ap.add_argument("--stride", type=int, default=100,
                    help="Frame stride; 100 makes ATLAS match mdCATH (1 ns/frame).")
    ap.add_argument("--max-proteins", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    atlas_dir = Path(args.atlas_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    protein_dirs = sorted(d for d in atlas_dir.iterdir()
                          if d.is_dir() and list(d.glob("*.pdb")))
    print(f"Found {len(protein_dirs)} ATLAS protein directories")
    if args.max_proteins > 0:
        protein_dirs = protein_dirs[: args.max_proteins]
        print(f"  limiting to {args.max_proteins}")

    all_records: list[dict] = []
    for i, pdir in enumerate(protein_dirs):
        print(f"\n[{i+1}/{len(protein_dirs)}] {pdir.name}")
        all_records.extend(process_protein(
            pdir, output_dir, stride=args.stride, force=args.force))

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    n_ok = sum(1 for r in all_records if r.get("status") == "ok")
    total_frames = sum(r.get("num_frames", 0) for r in all_records)
    print(f"\n{'='*60}")
    print(f"records: {len(all_records)} ({n_ok} freshly written)")
    print(f"total frames: {total_frames}")
    print(f"output: {output_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
