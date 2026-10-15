"""Stage C evaluation adapter: MDTraj RMSD + AlphaFlow ensemble metrics.

Gate 5 canonical implementation:
- RMSD via mdtraj.rmsd (locked to MDTraj >= 1.9.9)
- Six ensemble metrics via AlphaFlow official scripts
  (commit 0408d7c89dac444a43a9089d7427ce470b0a5e67)
- Adapter only: export PDB, call subprocess, parse output
- No mathematical reimplementation of RMSF, W2, PCA, or contact metrics

Gate 5 invariants (from audit plan):
1. Fixed AlphaFlow commit, script SHA256, environment lock
2. Reference MD ensemble (multi-frame), NOT single target structure
3. Predicted/reference ensemble size, atom set, alignment unified
4. Six metrics saved per protein, then protein-level bootstrap
5. Subprocess failures NEVER silently propagate NaN into aggregates

Official output mapping:
| Paper Metric           | Official Source          |
|------------------------|--------------------------|
| RMSD                   | MDTraj rmsd              |
| Per-target RMSF r      | AlphaFlow Per target RMSF r |
| Root-mean W2           | AlphaFlow RMWD           |
| Variance W2            | AlphaFlow RMWD var       |
| MD-PCA W2              | AlphaFlow MD PCA W2      |
| Weak contacts J        | AlphaFlow Weak contacts J |
| Transient contacts J   | AlphaFlow Transient contacts J |
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ═══════════════════════════════════════════════════════════════════════
# Gate 5.1: Tool version locks
# ═══════════════════════════════════════════════════════════════════════

MDTRAJ_MIN_VERSION = (1, 9, 9)
ALPHAFLOW_COMMIT = "0408d7c89dac444a43a9089d7427ce470b0a5e67"
ALPHAFLOW_REPO = "bjing2016/alphaflow"

# Optional SHA256 for scripts — set when environment is locked
_SCRIPT_SHA256: dict[str, str] = {}


def register_script_hash(script_name: str, sha256: str) -> None:
    """Register expected SHA256 for an AlphaFlow script (Gate 5.1 lock)."""
    _SCRIPT_SHA256[script_name] = sha256


def verify_script_hash(script_path: str | Path) -> str:
    """Compute SHA256 of a script file. Returns hex digest."""
    script_path = Path(script_path)
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    return hashlib.sha256(script_path.read_bytes()).hexdigest()


def verify_mdtraj_version() -> str:
    """Verify MDTraj version >= minimum. Returns version string or "not_installed"."""
    try:
        import mdtraj as md
    except ImportError:
        return "not_installed"

    ver = tuple(int(x) for x in md.__version__.split(".")[:3])
    if ver < MDTRAJ_MIN_VERSION:
        raise RuntimeError(
            f"MDTraj version {md.__version__} < {'.'.join(map(str, MDTRAJ_MIN_VERSION))}. "
            f"Install with: pip install 'mdtraj>={'.'.join(map(str, MDTRAJ_MIN_VERSION))}'"
        )
    return md.__version__


@dataclass(frozen=True)
class StageCMetrics:
    """Container for all eight Stage C metrics (Gate 5)."""
    # MDTraj metrics
    rmsd_ca: float = float("nan")
    rmsd_backbone: float = float("nan")

    # AlphaFlow metrics
    per_target_rmsf_r: float = float("nan")
    rmwd: float = float("nan")              # Root-mean W2 distance
    rmwd_var: float = float("nan")          # Variance W2
    md_pca_w2: float = float("nan")         # MD-PCA W2 distance
    weak_contacts_j: float = float("nan")   # Weak contacts Jaccard
    transient_contacts_j: float = float("nan")  # Transient contacts Jaccard

    # Metadata
    nan_contact_count: int = 0
    nan_contact_ratio: float = 0.0
    evaluation_notes: str = ""

    # Gate 5.3: eval status (replaces silent NaN propagation)
    status: str = ""  # "ok", "mdtraj_failed", "alphaflow_failed", "skipped"
    alphaflow_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rmsd_ca": self.rmsd_ca,
            "rmsd_backbone": self.rmsd_backbone,
            "per_target_rmsf_r": self.per_target_rmsf_r,
            "rmwd": self.rmwd,
            "rmwd_var": self.rmwd_var,
            "md_pca_w2": self.md_pca_w2,
            "weak_contacts_j": self.weak_contacts_j,
            "transient_contacts_j": self.transient_contacts_j,
            "nan_contact_count": self.nan_contact_count,
            "nan_contact_ratio": self.nan_contact_ratio,
            "status": self.status,
            "alphaflow_error": self.alphaflow_error,
        }


# ═══════════════════════════════════════════════════════════════════════
# PDB export
# ═══════════════════════════════════════════════════════════════════════

def _atom37_to_protein(
    coords: np.ndarray,
    atom_mask: np.ndarray,
    aatype: np.ndarray,
    residue_index: np.ndarray,
    chain_id: np.ndarray,
):
    """Build an official OpenFold ``Protein`` from atom37 arrays.

    Coordinates/atom ordering follow OpenFold's own atom37 convention (the
    ordering produced by ``stage_c.openfold_atom37``), so the structure is
    written with OpenFold's authoritative ``protein.to_pdb``.

    Args:
        coords: [L, 37, 3] atom37 coordinates (Angstrom), OpenFold ordering.
        atom_mask: [L, 37] which atoms are present.
        aatype: [L] residue type indices (0-19).
        residue_index: [L] sequential residue indices.
        chain_id: [L] chain identifiers.

    Returns:
        ``openfold.np.protein.Protein`` instance.
    """
    from protein_jepa_frame import vendored as _vendored  # noqa: F401
    from openfold.np import protein as of_protein

    coords = np.asarray(coords, dtype=np.float32)
    atom_mask = np.asarray(atom_mask, dtype=np.float32)
    aatype = np.asarray(aatype, dtype=np.int64)
    residue_index = np.asarray(residue_index, dtype=np.int64)
    chain_id = np.asarray(chain_id, dtype=np.int64)
    n = aatype.shape[0]
    return of_protein.Protein(
        atom_positions=coords,
        aatype=aatype,
        atom_mask=atom_mask,
        residue_index=residue_index,
        b_factors=np.zeros((n, coords.shape[1]), dtype=np.float32),
        chain_index=chain_id,
    )


def _to_openfold_ordering(
    coords: np.ndarray, atom_mask: np.ndarray | None, atom37_order: str,
):
    """Permute atom37 arrays from ``atom37_order`` into OpenFold ordering.

    ``atom37_order="openfold"`` is a no-op; ``atom37_order="project"`` applies
    the audited permutation from the project schema ordering to OpenFold's
    authoritative ordering.
    """
    if atom37_order == "openfold":
        return coords, atom_mask
    if atom37_order != "project":
        raise ValueError(f"unknown atom37_order: {atom37_order!r}")
    from protein_jepa_frame.models.stage_c.openfold_atom37 import (
        PROJECT_ATOM37_TO_OPENFOLD,
    )

    # Scatter: openfold[PROJECT_ATOM37_TO_OPENFOLD[k]] = project[k].
    idx = np.asarray(PROJECT_ATOM37_TO_OPENFOLD, dtype=np.int64)
    coords = np.asarray(coords)
    out = np.zeros_like(coords)
    out[..., idx, :] = coords
    if atom_mask is not None:
        atom_mask = np.asarray(atom_mask)
        out_mask = np.zeros_like(atom_mask)
        out_mask[..., idx] = atom_mask
        atom_mask = out_mask
    return out, atom_mask


def _atom37_to_pdb_lines(
    coords: np.ndarray,
    atom_mask: np.ndarray,
    aatype: np.ndarray,
    residue_index: np.ndarray,
    chain_id: np.ndarray,
    model_num: int = 1,
    atom37_order: str = "openfold",
) -> str:
    """Convert atom37 coordinates to a PDB model block via official OpenFold.

    Wraps ``openfold.np.protein.to_pdb`` (no hand-written PDB formatting).

    Args:
        coords: [L, 37, 3] atom37 coordinates (Angstrom).
        atom_mask: [L, 37] which atoms are present.
        aatype: [L] residue type indices (0-19).
        residue_index: [L] sequential residue indices.
        chain_id: [L] chain identifiers.
        model_num: PDB model number.
        atom37_order: "openfold" (default) or "project" — the ordering of the
            input ``coords``/``atom_mask`` along the atom axis.

    Returns:
        str: complete PDB model text.
    """
    from protein_jepa_frame import vendored as _vendored  # noqa: F401
    from openfold.np import protein as of_protein

    coords, atom_mask = _to_openfold_ordering(coords, atom_mask, atom37_order)
    prot = _atom37_to_protein(coords, atom_mask, aatype, residue_index, chain_id)
    pdb_text = of_protein.to_pdb(prot)

    # Strip OpenFold's trailing END so the ensemble exporter can join models,
    # and wrap the block with explicit MODEL/ENDMDL records.
    body_lines = [
        line for line in pdb_text.splitlines()
        if not line.startswith("END")
    ]
    lines = [f"MODEL     {model_num:4d}"] + body_lines + ["ENDMDL"]
    return "\n".join(lines)


def export_ensemble_pdb(
    samples: list[np.ndarray],
    atom_masks: list[np.ndarray],
    aatype: np.ndarray,
    residue_index: np.ndarray,
    chain_id: np.ndarray,
    output_path: str | Path,
    atom37_order: str = "openfold",
) -> Path:
    """Export an ensemble of structures as a multi-model PDB file.

    Args:
        samples: list of [L, 37, 3] atom37 coordinates
        atom_masks: list of [L, 37] atom masks
        aatype: [L] residue type indices
        residue_index: [L] sequential residue indices
        chain_id: [L] chain identifiers
        output_path: where to save the PDB file
        atom37_order: "openfold" (default; Stage C decoder output) or "project"
            (schema/``atom37_layer`` ordering).

    Returns:
        Path to generated PDB file
    """
    output_path = Path(output_path)
    models = []
    for i, (coords, mask) in enumerate(zip(samples, atom_masks)):
        models.append(_atom37_to_pdb_lines(
            coords, mask, aatype, residue_index, chain_id,
            model_num=i + 1, atom37_order=atom37_order,
        ))

    pdb_text = "\n".join(models) + "\nEND\n"
    output_path.write_text(pdb_text)
    return output_path


# ═══════════════════════════════════════════════════════════════════════
# MDTraj RMSD adapter
# ═══════════════════════════════════════════════════════════════════════

def compute_rmsd_mdtraj(
    pred_pdb_path: str | Path,
    target_pdb_path: str | Path,
    atom_selection: str = "name CA",
) -> float:
    """Compute RMSD using mdtraj with version check.

    Note: mdtraj.rmsd() returns nanometers. This function converts to Angstroms.

    If mdtraj is not installed, raises ImportError (never silently returns NaN).
    """
    import mdtraj as md

    ver = tuple(int(x) for x in md.__version__.split(".")[:3])
    if ver < MDTRAJ_MIN_VERSION:
        raise RuntimeError(
            f"MDTraj {md.__version__} < {'.'.join(map(str, MDTRAJ_MIN_VERSION))}"
        )

    pred_traj = md.load(str(pred_pdb_path))
    target_traj = md.load(str(target_pdb_path))

    pred_selected = pred_traj.topology.select(atom_selection)
    target_selected = target_traj.topology.select(atom_selection)

    # PDB writers omit atoms whose presence mask is false.  Integer atom
    # indices therefore need not agree between prediction and reference.
    # Match by identity and pass both index arrays to MDTraj; using only the
    # prediction indices can silently compare different atoms.
    def atom_key(atom):
        residue = atom.residue
        return (
            atom.name,
            int(atom.residue.chain.index),
            int(residue.resSeq),
            str(getattr(residue, "insertion_code", "")),
        )

    target_by_key = {
        atom_key(target_traj.topology.atom(i)): int(i)
        for i in target_selected
    }
    pred_atoms, target_atoms = [], []
    for i in pred_selected:
        j = target_by_key.get(atom_key(pred_traj.topology.atom(i)))
        if j is not None:
            pred_atoms.append(int(i))
            target_atoms.append(j)

    if len(pred_atoms) < 3:
        raise ValueError(
            f"Only {len(pred_atoms)} common atoms match '{atom_selection}'"
        )

    if pred_traj.n_frames > 1 and target_traj.n_frames == 1:
        rmsds = []
        ref_frame = target_traj[0]
        for i in range(pred_traj.n_frames):
            r = md.rmsd(
                pred_traj[i], ref_frame,
                atom_indices=pred_atoms, ref_atom_indices=target_atoms,
            )
            rmsds.append(float(r[0]))
        return float(np.mean(rmsds)) * 10.0
    else:
        r = md.rmsd(
            pred_traj, target_traj,
            atom_indices=pred_atoms, ref_atom_indices=target_atoms,
        )
        return float(r.mean()) * 10.0


# ═══════════════════════════════════════════════════════════════════════
# AlphaFlow adapter
# ═══════════════════════════════════════════════════════════════════════

class AlphaFlowAdapter:
    """Adapter for calling AlphaFlow official evaluation scripts.

    Fixed commit: 0408d7c89dac444a43a9089d7427ce470b0a5e67
    Repo: bjing2016/alphaflow

    This adapter ONLY:
    - Exports PDB files from model predictions
    - Calls alphaflow/scripts/analyze_ensembles.py as subprocess
    - Calls alphaflow/scripts/print_analysis.py as subprocess
    - Parses official pickle/stdout output
    - Does NOT implement any metric mathematics

    Gate 5.4: Ensemble size, atom set, and alignment must be unified
    between predicted and reference before calling AlphaFlow.

    Required setup:
    - AlphaFlow installed in separate conda/pip environment
    - Path to alphaflow scripts directory configured
    """

    def __init__(
        self,
        alphaflow_scripts_dir: str | Path,
        analysis_script: str = "scripts/analyze_ensembles.py",
        print_script: str = "scripts/print_analysis.py",
        conda_env: str | None = None,
        verify_sha256: bool = False,
    ):
        self.scripts_dir = Path(alphaflow_scripts_dir)
        self.analysis_script = self.scripts_dir / analysis_script
        self.print_script = self.scripts_dir / print_script
        self.conda_env = conda_env

        # Gate 5.1: verify script SHA256 if locked
        if verify_sha256:
            for name, path in [
                ("analyze_ensembles", self.analysis_script),
                ("print_analysis", self.print_script),
            ]:
                if name in _SCRIPT_SHA256:
                    actual = verify_script_hash(path)
                    if actual != _SCRIPT_SHA256[name]:
                        raise RuntimeError(
                            f"AlphaFlow script '{name}' SHA256 mismatch. "
                            f"Expected {_SCRIPT_SHA256[name]}, got {actual}. "
                            f"Script at {path}"
                        )

    def _run_alphaflow(
        self,
        args: list[str],
        timeout: int = 3600,
    ) -> tuple[bool, str, str]:
        """Run AlphaFlow script as subprocess.

        Returns:
            (success, stdout, stderr)
        """
        python_cmd = ["python", "-u"]
        cmd = python_cmd + [str(arg) for arg in args]

        if self.conda_env:
            cmd = [
                "conda", "run", "-n", self.conda_env,
                "python", "-u",
            ] + [str(arg) for arg in args]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return False, "", "TIMEOUT"
        except FileNotFoundError:
            return False, "", f"AlphaFlow script not found"

    def analyze_ensemble(
        self,
        predicted_pdb: str | Path,
        target_ensemble_dir: str | Path,
        output_dir: str | Path,
        protein_id: str = "unknown",
        ensemble_size: int | None = None,
    ) -> dict[str, Any]:
        """Run AlphaFlow analyze_ensembles.py on predicted ensemble.

        Gate 5.3: target_ensemble_dir must contain a multi-frame MD ensemble
        (not a single target structure). The AlphaFlow script internally reads
        all PDB models from the target directory.

        Args:
            predicted_pdb: multi-model PDB with predicted ensemble
            target_ensemble_dir: directory containing reference MD ensemble PDB(s)
            output_dir: where to write results
            protein_id: protein identifier
            ensemble_size: if set, validate predicted/reference sizes match

        Returns:
            dict with "success", "metrics" (StageCMetrics), "stdout", "stderr"
        """
        predicted_pdb = Path(predicted_pdb)
        target_ensemble_dir = Path(target_ensemble_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pdbdir = output_dir / "pdbs"
        pdbdir.mkdir(parents=True, exist_ok=True)

        # Copy predicted PDB to pdbdir
        import shutil
        shutil.copy(predicted_pdb, pdbdir / f"{protein_id}_pred.pdb")

        # Gate 5.4: Verify ensemble sizes match if requested
        if ensemble_size is not None:
            try:
                import mdtraj as md
                pred_traj = md.load(str(predicted_pdb))
                n_pred = pred_traj.n_frames
                if n_pred != ensemble_size:
                    raise ValueError(
                        f"Predicted ensemble size {n_pred} != expected {ensemble_size}"
                    )
                reference_pdbs = sorted(target_ensemble_dir.glob("*.pdb"))
                if not reference_pdbs:
                    raise ValueError(f"No reference PDBs found in {target_ensemble_dir}")
                n_ref = sum(md.load(str(path)).n_frames for path in reference_pdbs)
                if n_ref != ensemble_size:
                    raise ValueError(
                        f"Reference ensemble size {n_ref} != expected {ensemble_size}"
                    )
            except Exception:
                # This check is a protocol invariant, not optional metadata.
                raise

        # Verify scripts exist
        if not self.analysis_script.exists():
            return {
                "success": False,
                "error": f"Analysis script not found: {self.analysis_script}",
                "metrics": StageCMetrics(status="alphaflow_failed"),
            }

        # Build command for AlphaFlow
        results_path = pdbdir / "out.pkl"

        args = [
            str(self.analysis_script),
            "--atlas_dir", str(target_ensemble_dir),
            "--pdbdir", str(pdbdir),
            "--pdb_id", protein_id,
        ]

        success, stdout, stderr = self._run_alphaflow(args)

        metrics = StageCMetrics(status="ok" if success else "alphaflow_failed")

        if success and results_path.exists():
            try:
                with open(results_path, "rb") as f:
                    results = pickle.load(f)

                if protein_id in results:
                    r = results[protein_id]
                elif isinstance(results, dict):
                    r = next(iter(results.values())) if results else {}
                else:
                    r = {}

                metrics = StageCMetrics(
                    status="ok",
                    per_target_rmsf_r=float(r.get("Per target RMSF r", float("nan"))),
                    rmwd=float(r.get("RMWD", float("nan"))),
                    rmwd_var=float(r.get("RMWD var", float("nan"))),
                    md_pca_w2=float(r.get("MD PCA W2", float("nan"))),
                    weak_contacts_j=float(r.get("Weak contacts J", float("nan"))),
                    transient_contacts_j=float(r.get("Transient contacts J", float("nan"))),
                    nan_contact_count=int(r.get("nan_contact_count", 0)),
                    nan_contact_ratio=float(r.get("nan_contact_ratio", 0.0)),
                )

            except Exception as e:
                import traceback
                metrics = StageCMetrics(
                    status="alphaflow_failed",
                    evaluation_notes=f"Parse error: {e}",
                    alphaflow_error=f"{e}\n{traceback.format_exc()}"[:1000],
                )

        return {
            "success": success,
            "metrics": metrics,
            "stdout": stdout[:2000] if stdout else "",
            "stderr": stderr[:2000] if stderr else "",
        }

    def print_analysis(
        self,
        results_path: str | Path,
    ) -> str:
        """Run AlphaFlow print_analysis.py to get human-readable summary."""
        if not self.print_script.exists():
            return f"Print script not found: {self.print_script}"

        args = [str(self.print_script), str(results_path)]
        success, stdout, stderr = self._run_alphaflow(args, timeout=300)
        return stdout if success else f"ERROR: {stderr}"


# ═══════════════════════════════════════════════════════════════════════
# Main evaluation entry point
# ═══════════════════════════════════════════════════════════════════════

def evaluate_stage_c(
    predicted_samples: list[np.ndarray],
    reference_ensemble_samples: list[np.ndarray],
    atom_masks: list[np.ndarray],
    ref_atom_masks: list[np.ndarray],
    aatype: np.ndarray,
    residue_index: np.ndarray,
    chain_id: np.ndarray,
    output_dir: str | Path,
    protein_id: str = "unknown",
    alphaflow_scripts_dir: str | None = None,
    mode: str = "direct",
    horizon_label: str = "",
    dataset_label: str = "",
    pred_atom37_order: str = "openfold",
    ref_atom37_order: str = "project",
) -> dict[str, Any]:
    """Full Stage C evaluation pipeline (Gate 5).

    Gate 5.3: reference_ensemble_samples is a LIST of [L, 37, 3] arrays
    representing the MD reference ensemble (not a single structure).

    Args:
        predicted_samples: list of [L, 37, 3] atom37 coordinate arrays
        reference_ensemble_samples: list of [L, 37, 3] reference MD ensemble
        atom_masks: list of [L, 37] atom masks for predicted samples
        ref_atom_masks: list of [L, 37] atom masks for reference ensemble
        aatype: [L] residue type indices
        residue_index: [L] sequential indices
        chain_id: [L] chain identifiers
        output_dir: directory for PDB exports and analysis results
        protein_id: protein identifier
        alphaflow_scripts_dir: path to AlphaFlow scripts directory
        mode: prediction mode ("oracle", "direct", "recursive") for record-keeping
        horizon_label: horizon identifier for record-keeping
        dataset_label: dataset name for record-keeping

    Returns:
        dict with all metrics and metadata, including evaluation status.
        Gate 5.5: NaN metrics are ALWAYS accompanied by a non-"ok" status.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export predictions as multi-model PDB
    pred_pdb = output_dir / f"{protein_id}_predicted.pdb"
    export_ensemble_pdb(
        predicted_samples, atom_masks,
        aatype, residue_index, chain_id,
        pred_pdb, atom37_order=pred_atom37_order,
    )

    results = {
        "protein_id": protein_id,
        "predicted_pdb": str(pred_pdb),
        "ensemble_size_pred": len(predicted_samples),
        "ensemble_size_ref": len(reference_ensemble_samples),
        "mode": mode,
        "horizon_label": horizon_label,
        "dataset_label": dataset_label,
        "status": "ok",
    }

    # 1. MDTraj RMSD
    try:
        # Export first frame of reference as single-model PDB for RMSD
        ref_pdb = output_dir / f"{protein_id}_ref_frame.pdb"
        export_ensemble_pdb(
            [reference_ensemble_samples[0]], [ref_atom_masks[0]],
            aatype, residue_index, chain_id,
            ref_pdb, atom37_order=ref_atom37_order,
        )
        results["rmsd_ca"] = compute_rmsd_mdtraj(pred_pdb, ref_pdb, "name CA")
        results["rmsd_backbone"] = compute_rmsd_mdtraj(
            pred_pdb, ref_pdb, "name N or name CA or name C"
        )
    except Exception as e:
        results["status"] = "mdtraj_failed"
        results["rmsd_ca"] = float("nan")
        results["rmsd_backbone"] = float("nan")
        results["mdtraj_error"] = str(e)[:500]

    # 2. AlphaFlow (if available)
    if alphaflow_scripts_dir:
        # Export reference ensemble as multi-model PDB in a dedicated directory
        ref_ensemble_dir = output_dir / "reference_ensemble"
        ref_ensemble_dir.mkdir(exist_ok=True)
        ref_ensemble_pdb = ref_ensemble_dir / f"{protein_id}_ref_ensemble.pdb"
        export_ensemble_pdb(
            reference_ensemble_samples, ref_atom_masks,
            aatype, residue_index, chain_id,
            ref_ensemble_pdb, atom37_order=ref_atom37_order,
        )

        adapter = AlphaFlowAdapter(
            alphaflow_scripts_dir,
            verify_sha256=bool(_SCRIPT_SHA256),
        )
        af_result = adapter.analyze_ensemble(
            pred_pdb,
            ref_ensemble_dir,
            output_dir / "alphaflow",
            protein_id,
            ensemble_size=len(reference_ensemble_samples),
        )
        if "metrics" in af_result:
            results.update(af_result["metrics"].to_dict())
        if af_result.get("error"):
            results["alphaflow_error"] = af_result["error"][:500]
            if results.get("status") == "ok":
                results["status"] = "alphaflow_failed"
    else:
        # AlphaFlow metrics are ensemble metrics.  Leaving status="ok" with
        # their default NaNs would make aggregate_metrics treat an incomplete
        # evaluation as complete.
        results["status"] = "alphaflow_skipped"
        results["alphaflow_error"] = "alphaflow_scripts_dir was not supplied"

    # A successful subprocess is not sufficient: every requested metric must
    # be present and finite before a per-protein result can be marked ``ok``.
    metric_keys = [
        "rmsd_ca", "rmsd_backbone", "per_target_rmsf_r", "rmwd",
        "rmwd_var", "md_pca_w2", "weak_contacts_j", "transient_contacts_j",
    ]
    nonfinite = [
        key for key in metric_keys
        if key in results and not np.isfinite(float(results[key]))
    ]
    if nonfinite and results.get("status") == "ok":
        results["status"] = "metrics_incomplete"
        results["alphaflow_error"] = (
            "non-finite metrics: " + ", ".join(nonfinite)
        )

    # Save results (Gate 5.5: per-protein save)
    results_path = output_dir / f"{protein_id}_evaluation.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


# ═══════════════════════════════════════════════════════════════════════
# Protein-level bootstrap aggregation (Gate 5.5)
# ═══════════════════════════════════════════════════════════════════════

def aggregate_metrics(
    protein_results: list[dict[str, Any]],
    metric_names: list[str] | None = None,
    bootstrap_samples: int = 1000,
    ci_alpha: float = 0.05,
    require_status: str | None = "ok",
) -> dict[str, Any]:
    """Aggregate metrics across proteins with bootstrap CI.

    Gate 5.5: Only proteins with status=="ok" (or matching require_status)
    are included in the aggregate. Proteins with evaluation failures are
    counted but excluded from statistics.

    Uses protein-level bootstrap (not frame-level), as required by audit plan.

    Args:
        protein_results: list of per-protein evaluation results
        metric_names: which metrics to aggregate (default: all)
        bootstrap_samples: number of bootstrap resamples
        ci_alpha: significance level for CI (0.05 = 95% CI)
        require_status: only include proteins with this status (None = all)

    Returns:
        dict with mean, std, CI for each metric, plus exclusion counts
    """
    if metric_names is None:
        metric_names = [
            "rmsd_ca", "rmsd_backbone", "per_target_rmsf_r",
            "rmwd", "rmwd_var", "md_pca_w2",
            "weak_contacts_j", "transient_contacts_j",
        ]

    # Filter by status
    if require_status is not None:
        valid_results = [r for r in protein_results if r.get("status") == require_status]
        excluded = len(protein_results) - len(valid_results)
    else:
        valid_results = protein_results
        excluded = 0

    rng = np.random.RandomState(42)

    aggregate = {}
    for name in metric_names:
        values = np.array([
            r.get(name, float("nan")) for r in valid_results
        ])
        valid = values[~np.isnan(values)]

        if len(valid) == 0:
            aggregate[name] = {
                "mean": float("nan"), "std": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "n": 0,
            }
            continue

        # Bootstrap at protein level
        boot_means = []
        for _ in range(bootstrap_samples):
            idx = rng.choice(len(valid), size=len(valid), replace=True)
            boot_means.append(valid[idx].mean())

        boot_means = np.array(boot_means)
        mean = float(valid.mean())
        std = float(valid.std())
        ci_low = float(np.percentile(boot_means, 100 * ci_alpha / 2))
        ci_high = float(np.percentile(boot_means, 100 * (1 - ci_alpha / 2)))

        aggregate[name] = {
            "mean": mean,
            "std": std,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n": len(valid),
        }

    # Also report overall counts
    aggregate["metadata"] = {
        "n_proteins_total": len(protein_results),
        "n_valid": len(valid_results),
        "n_excluded": excluded,
        "exclusion_reason": f"status != '{require_status}'" if require_status else "none",
        "bootstrap_samples": bootstrap_samples,
        "ci_level": 1.0 - ci_alpha,
    }

    return aggregate


# ═══════════════════════════════════════════════════════════════════════
# Gate 5.6: Split table helper
# ═══════════════════════════════════════════════════════════════════════

def build_split_table(
    all_protein_results: list[dict[str, Any]],
    split_dims: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Build per-split aggregate tables.

    Args:
        all_protein_results: list of per-protein evaluation dicts with
                             "mode", "horizon_label", "dataset_label" fields
        split_dims: dimension labels to group by
                    e.g. {"mode": "*", "horizon_label": "*", "dataset_label": "*"}

    Returns:
        dict mapping split_key -> aggregate metrics
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for r in all_protein_results:
        key_parts = []
        for dim in split_dims:
            key_parts.append(f"{dim}={r.get(dim, '?')}")
        key = "|".join(key_parts)
        groups[key].append(r)

    tables = {}
    for key, results in groups.items():
        tables[key] = aggregate_metrics(results, require_status="ok")
    return tables


__all__ = [
    "StageCMetrics",
    "AlphaFlowAdapter",
    "export_ensemble_pdb",
    "compute_rmsd_mdtraj",
    "evaluate_stage_c",
    "aggregate_metrics",
    "build_split_table",
    "register_script_hash",
    "verify_script_hash",
    "verify_mdtraj_version",
    "MDTRAJ_MIN_VERSION",
    "ALPHAFLOW_COMMIT",
]
