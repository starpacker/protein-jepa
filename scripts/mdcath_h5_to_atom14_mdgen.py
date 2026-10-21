#!/usr/bin/env python3
"""mdCATH h5 -> MDGen atom14 .npy  (NEW file, 2026-10-20).

Converts the mdCATH h5 files referenced by *our* manifests into the
``[F, L, 14, 3]`` atom14 numpy arrays that MDGen's ``MDGenDataset`` expects
(``mdgen/dataset.py`` + ``scripts/prep_sims.py``).  This lets MDGen be trained
and evaluated on **exactly the same train/val split** as our Protein-JEPA model.

Key facts reproduced from ``protein_jepa_frame/data/dataset.py``:
  * the h5 stores all-atom coords under ``<domain>/<temperature>/<replica>/coords``
  * the atom ordering / residue names come from the PDB stored as
    ``<domain>/pdbProteinAtoms`` (atom count must match ``coords.shape[1]``)
  * time axis is one frame per ``frame_interval_ps`` (1000 ps = 1 ns for mdCATH)

We keep ALL atoms of the PDB (not just backbone), map them onto the 14 atom14
slots per residue using ``mdgen.residue_constants.restype_name_to_atom14_names``,
and write one ``.npy`` per manifest line:
``<out>/<domain>_<temp>_<replica>.npy`` (this triple IS our trajectory_id).

Usage:
    python scripts/mdcath_h5_to_atom14_mdgen.py \
        --manifest artifacts/mdcath_manifest_200dom/train.jsonl \
        --outdir   /liuyutian/mdgen_run/data_mdcath \
        --num-workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# CHARMM-style HIS protonation states / other aliases -> canonical 3-letter name.
_RESNAME_ALIASES = {
    "HSE": "HIS", "HSD": "HIS", "HSP": "HIS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS",
    "MSE": "MET", "SEC": "CYS", "SEP": "SER", "TPO": "THR", "PTR": "TYR",
}


def _canonical_resname(rn: str) -> str:
    return _RESNAME_ALIASES.get(rn.upper(), rn.upper())


def _restype_name_to_atom14_names():
    """atom14 slot names per residue (imported lazily to avoid hard dep at parse)."""
    from mdgen.residue_constants import restype_name_to_atom14_names
    return restype_name_to_atom14_names


def _read_pdb_atoms(pdb_text: str):
    """Return list of (atom_name, resname, chain, resseq, icode) in file order."""
    atoms = []
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_name = line[12:16].strip()
        resname = line[17:20].strip().upper()
        chain = line[21:22].strip() or "_"
        resseq = int(line[22:26])
        icode = line[26:27].strip()
        atoms.append((atom_name, resname, chain, resseq, icode))
    return atoms


def convert_one(record: dict, outdir: str, overwrite: bool = False) -> str:
    import h5py

    tid = record["trajectory_id"]
    out = os.path.join(outdir, f"{tid}.npy")
    if os.path.exists(out) and not overwrite:
        return f"skip {tid}"

    with h5py.File(record["h5_path"], "r") as handle:
        domain = handle[record["domain_id"]]
        grp = handle[record["metadata"]["hdf5_group"]]
        coords = np.asarray(grp["coords"], dtype=np.float32)     # [F, N, 3] Angstrom
        pdb_text = domain["pdbProteinAtoms"][()]
        if isinstance(pdb_text, bytes):
            pdb_text = pdb_text.decode("utf-8", errors="replace")

    atoms = _read_pdb_atoms(str(pdb_text))
    if len(atoms) != coords.shape[1]:
        return f"ERR {tid}: pdb atoms {len(atoms)} != coords {coords.shape[1]}"

    # group atoms by residue key (preserving order)
    res_keys, res_atoms = [], {}
    for i, (an, rn, ch, rs, ic) in enumerate(atoms):
        key = (ch, rs, ic)
        if key not in res_atoms:
            res_atoms[key] = {"name": rn, "atoms": []}
            res_keys.append(key)
        res_atoms[key]["atoms"].append((an, i))

    rc = _restype_name_to_atom14_names()
    L = len(res_keys)
    F = coords.shape[0]
    arr = np.zeros((F, L, 14, 3), dtype=np.float16)
    for r, key in enumerate(res_keys):
        info = res_atoms[key]
        names = rc.get(_canonical_resname(info["name"]))
        if names is None:
            continue
        for an, idx in info["atoms"]:
            if an in names:
                j = names.index(an)
                arr[:, r, j] = coords[:, idx]
    tmp = out + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, out)
    return f"ok {tid} {arr.shape}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    records = [json.loads(l) for l in open(args.manifest) if l.strip()]
    # dedupe by trajectory_id (manifest may list a traj once per horizon)
    seen, uniq = set(), []
    for r in records:
        if r["trajectory_id"] in seen:
            continue
        seen.add(r["trajectory_id"])
        uniq.append(r)
    print(f"[convert] {len(uniq)} unique trajectories from {args.manifest}")

    tasks = [(r, args.outdir, args.overwrite) for r in uniq]
    if args.num_workers > 1:
        with Pool(args.num_workers) as p:
            results = list(p.starmap(convert_one, tasks, chunksize=1))
    else:
        results = [convert_one(*t) for t in tasks]
    errs = [r for r in results if r.startswith("ERR")]
    print(f"[convert] done: {len(results)-len(errs)} ok, {len(errs)} err")
    for e in errs[:10]:
        print("  ", e)


if __name__ == "__main__":
    main()
