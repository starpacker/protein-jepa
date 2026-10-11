#!/usr/bin/env python3
"""Phase 0: ATLAS XTC trajectories → NPZ backbone coordinates.

Converts GROMACS XTC trajectories to a format the protein_jepa dataset loader
expects: [T, L, 4, 3] backbone coords (N, CA, C, O) + residue metadata.

Usage:
  python scripts/preprocess_xtc_to_npz.py \
    --atlas-dir /liuyutian/protein_jepa/datasets/ATLAS \
    --output-dir /liuyutian/protein_jepa/datasets/ATLAS_NPZ \
    --max-proteins 10  # for smoke test; remove for full dataset
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _find_backbone_atom_indices(topology) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return lists of (N, CA, C, O) atom indices per residue from mdtraj topology."""
    n_idx, ca_idx, c_idx, o_idx = [], [], [], []
    for residue in topology.residues:
        n_found = ca_found = c_found = o_found = -1
        for atom in residue.atoms:
            name = atom.name.strip().upper()
            if name == "N":   n_found = atom.index
            if name == "CA":  ca_found = atom.index
            if name == "C":   c_found = atom.index
            if name == "O":   o_found = atom.index
        if ca_found >= 0:
            n_idx.append(n_found if n_found >= 0 else -1)
            ca_idx.append(ca_found)
            c_idx.append(c_found if c_found >= 0 else -1)
            o_idx.append(o_found if o_found >= 0 else -1)
        # skip residues without CA (typically N-terminal/water)
    # Filter to residues that have valid CA
    valid = [i for i, ca in enumerate(ca_idx) if ca >= 0]
    return (
        [n_idx[i] for i in valid],
        [ca_idx[i] for i in valid],
        [c_idx[i] for i in valid],
        [o_idx[i] for i in valid],
    )


def _extract_backbone_coords(xtc_path: str, pdb_path: str, stride: int = 1):
    """Load XTC+PDB, extract backbone coordinates [T, L, 4, 3].

    Returns:
        coords: [T, L_residues, 4_atoms, 3] float32 array
        residue_names: list[str] of 3-letter codes
        residue_indices: list[int] PDB residue numbers
    """
    import mdtraj

    traj = mdtraj.load(xtc_path, top=pdb_path)
    if stride > 1:
        traj = traj[::stride]

    # Find backbone atoms
    top = traj.topology
    n_atoms_per_res = top.n_residues
    n_idx, ca_idx, c_idx, o_idx = _find_backbone_atom_indices(top)
    L = len(ca_idx)
    T = traj.n_frames

    if L == 0:
        raise ValueError(f"No CA atoms found in {pdb_path}")

    # Extract
    xyz = traj.xyz  # [T, N_atoms, 3]
    coords = np.zeros((T, L, 4, 3), dtype=np.float32)

    for i in range(L):
        if n_idx[i] >= 0:
            coords[:, i, 0] = xyz[:, n_idx[i]]
        else:
            coords[:, i, 0] = np.nan  # N missing — will be filtered
        coords[:, i, 1] = xyz[:, ca_idx[i]]  # CA
        if c_idx[i] >= 0:
            coords[:, i, 2] = xyz[:, c_idx[i]]
        else:
            coords[:, i, 2] = np.nan
        if o_idx[i] >= 0:
            coords[:, i, 3] = xyz[:, o_idx[i]]
        else:
            coords[:, i, 3] = np.nan

    # Check for NaN residues (missing N/O)
    nan_mask = np.any(np.isnan(coords), axis=(0, 2, 3))  # [L]
    if nan_mask.any():
        print(f"  WARNING: {nan_mask.sum()}/{L} residues have NaN atoms — filling with CA")
        for i in range(L):
            if nan_mask[i]:
                coords[:, i, 0] = coords[:, i, 1]  # N ← CA
                coords[:, i, 2] = coords[:, i, 1]  # C ← CA

    # Residue info
    res_codes = []
    res_nums = []
    for i in range(L):
        res = list(top.residues)[i] if i < n_atoms_per_res else None
        if res is not None:
            res_codes.append(res.name[:3])
            res_nums.append(res.resSeq)
        else:
            res_codes.append("UNK")
            res_nums.append(i + 1)

    return coords, res_codes, res_nums, T, L


def _get_aatype_indices(residue_names: list[str]) -> np.ndarray:
    """Convert 3-letter residue names to integer aatype indices (0-20).

    Index 20 = unknown/other.
    """
    STANDARD = {
        "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
        "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
        "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
        "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
    }
    result = []
    for name in residue_names:
        code = name.strip().upper()[:3]
        result.append(STANDARD.get(code, 20))
    return np.array(result, dtype=np.int64)


def process_protein(
    protein_dir: Path,
    output_dir: Path,
    stride: int = 1,
    force: bool = False,
) -> dict | None:
    """Process one ATLAS protein directory.

    Each protein has 3 replicas (R1, R2, R3). We concatenate them into one
    trajectory and save a single NPZ per protein.

    Returns processing metadata or None if skipped.
    """
    pdb_files = sorted(protein_dir.glob("*.pdb"))
    xtc_files = sorted(protein_dir.glob("*_fit.xtc"))
    if not pdb_files or not xtc_files:
        print(f"  SKIP {protein_dir.name}: no PDB or XTC files")
        return None

    pdb_path = str(pdb_files[0])
    protein_id = protein_dir.name
    output_path = output_dir / f"{protein_id}.npz"

    if output_path.exists() and not force:
        print(f"  SKIP {protein_id}: already exists")
        return {"protein_id": protein_id, "status": "skipped", "path": str(output_path)}

    # Load all replicas
    all_coords = []
    total_frames = 0
    res_names = None
    res_nums = None

    for xtc_file in xtc_files:
        replica_id = xtc_file.stem.replace("_fit", "").replace(protein_id + "_", "")
        print(f"    Loading {xtc_file.name}...", end=" ", flush=True)
        try:
            coords, rn, ri, T, L = _extract_backbone_coords(str(xtc_file), pdb_path, stride)
            all_coords.append(coords)
            total_frames += T
            res_names = rn
            res_nums = ri
            print(f"{T} frames, L={L}")
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    if not all_coords:
        print(f"  FAIL {protein_id}: no replicas loaded")
        return None

    # Concatenate replicas
    L = all_coords[0].shape[1]
    # Verify consistent L
    for i, c in enumerate(all_coords):
        if c.shape[1] != L:
            print(f"  WARNING: replica {i} has L={c.shape[1]} vs expected {L}, trimming")
            all_coords[i] = c[:, :L]

    coords = np.concatenate(all_coords, axis=0).astype(np.float32)
    aatype = _get_aatype_indices(res_names)

    # Save
    print(f"    Saving {protein_id}: {coords.shape} ({total_frames} frames, {L} residues)")
    np.savez_compressed(
        str(output_path),
        coords=coords,
        aatype=aatype.astype(np.int64),
        residue_index=np.array(res_nums, dtype=np.int64),
        chain_id=np.zeros(L, dtype=np.int64),
        # No temperature info in ATLAS; fill with default
        temperature_k=300.0,
    )

    return {
        "protein_id": protein_id,
        "npz_path": str(output_path),
        "num_frames": total_frames,
        "num_residues": L,
        "num_replicas": len(xtc_files),
        "residue_names": res_names,
    }


def main():
    ap = argparse.ArgumentParser(description="ATLAS XTC → NPZ preprocessor")
    ap.add_argument("--atlas-dir", default="/liuyutian/protein_jepa/datasets/ATLAS")
    ap.add_argument("--output-dir", default="/liuyutian/protein_jepa/datasets/ATLAS_NPZ")
    ap.add_argument("--stride", type=int, default=1, help="Frame stride (1 = all frames)")
    ap.add_argument("--max-proteins", type=int, default=0, help="Limit for smoke test")
    ap.add_argument("--force", action="store_true", help="Overwrite existing")
    ap.add_argument("--manifest", default=None, help="Write JSONL manifest")
    args = ap.parse_args()

    atlas_dir = Path(args.atlas_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Protein directories are those that contain PDB files
    protein_dirs = sorted(
        d for d in atlas_dir.iterdir()
        if d.is_dir() and list(d.glob("*.pdb"))
    )
    print(f"Found {len(protein_dirs)} protein directories")

    if args.max_proteins > 0:
        protein_dirs = protein_dirs[: args.max_proteins]
        print(f"  Limiting to {len(protein_dirs)} for smoke test")

    # Process
    manifests = []
    n_ok = 0
    n_skip = 0
    n_fail = 0
    total_frames = 0

    for i, pdir in enumerate(protein_dirs):
        print(f"\n[{i+1}/{len(protein_dirs)}] {pdir.name}")

        try:
            meta = process_protein(pdir, output_dir, stride=args.stride, force=args.force)
            if meta is None:
                n_fail += 1
            elif meta.get("status") == "skipped":
                n_skip += 1
                manifests.append(meta)
            else:
                n_ok += 1
                total_frames += meta["num_frames"]
                manifests.append(meta)
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            n_fail += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {n_ok} processed, {n_skip} skipped, {n_fail} failed")
    print(f"Total frames: {total_frames}")
    print(f"Output: {output_dir}")

    # Write manifest
    manifest_path = args.manifest or str(output_dir / "manifest.jsonl")
    with open(manifest_path, "w") as f:
        for m in manifests:
            f.write(json.dumps(m) + "\n")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
