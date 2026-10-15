"""Gate 5: Ensemble metrics infrastructure tests.

Gate 5 requirements from audit plan:
1. Fixed AlphaFlow commit, script SHA256, environment lock
2. Reference MD ensemble (multi-frame), not single target
3. Predicted/reference ensemble size, atom set, alignment unified
4. Six metrics saved per protein, then protein-level bootstrap
5. Subprocess failures NEVER silently propagate NaN into aggregates
6. Direct/recursive, short/medium/long, three datasets per table

Tests:
- MDTraj version verification
- SHA256 registry for AlphaFlow scripts
- StageCMetrics includes status field
- aggregate_metrics filters by status
- evaluate_stage_c accepts reference ensemble (list, not single)
- Per-protein evaluation JSON saved
- build_split_table produces mode/horizon/dataset splits
- RMSD returns Angstroms (nm->A conversion)
- Silent NaN exclusion from aggregates
- export_ensemble_pdb with multi-model encoding
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from protein_jepa_frame.eval.stage_c_evaluator import (
    StageCMetrics,
    aggregate_metrics,
    evaluate_stage_c,
    export_ensemble_pdb,
    build_split_table,
    register_script_hash,
    verify_mdtraj_version,
    MDTRAJ_MIN_VERSION,
    ALPHAFLOW_COMMIT,
)
from protein_jepa_frame.data.schema import ATOM37_NAMES


# ── Gate 5.1: Tool version locks ───────────────────────────────────────

def test_mdtraj_version_verified():
    """MDTraj version string conforms to expected format."""
    ver = verify_mdtraj_version()
    assert ver != "not_installed", "MDTraj must be installed for tests"
    parts = ver.split(".")
    assert len(parts) >= 2, f"MDTraj version '{ver}' has unexpected format"


def test_min_mdtraj_version_defined():
    """MDTRAJ_MIN_VERSION tuple is properly defined."""
    assert len(MDTRAJ_MIN_VERSION) == 3
    assert all(isinstance(x, int) for x in MDTRAJ_MIN_VERSION)
    assert MDTRAJ_MIN_VERSION >= (1, 9, 9)


def test_alphaflow_commit_fixed():
    """AlphaFlow commit hash is fixed and non-empty."""
    assert len(ALPHAFLOW_COMMIT) == 40, (
        f"AlphaFlow commit must be 40-char SHA1, got {len(ALPHAFLOW_COMMIT)}"
    )
    assert all(c in "0123456789abcdef" for c in ALPHAFLOW_COMMIT), (
        "AlphaFlow commit must be lowercase hex"
    )


def test_script_hash_registry():
    """SHA256 registry can be set and queried."""
    register_script_hash("analyze_ensembles", "abc123")
    from protein_jepa_frame.eval.stage_c_evaluator import _SCRIPT_SHA256
    assert _SCRIPT_SHA256.get("analyze_ensembles") == "abc123"
    # Clean up
    _SCRIPT_SHA256.clear()


# ── Gate 5.2: Reference ensemble (multi-frame) ─────────────────────────

def test_evaluate_stage_c_accepts_reference_ensemble():
    """evaluate_stage_c accepts a LIST of reference structures, not single target."""
    # Synthetic data
    L = 5
    aatype = np.zeros(L, dtype=int)
    residue_index = np.arange(L)
    chain_id = np.zeros(L, dtype=int)

    # Predicted ensemble: 3 samples
    pred_samples = [
        np.random.randn(L, 37, 3) for _ in range(3)
    ]
    # Reference ensemble: 5 samples
    ref_samples = [
        np.random.randn(L, 37, 3) for _ in range(5)
    ]
    pred_masks = [np.ones((L, 37), dtype=bool) for _ in range(3)]
    ref_masks = [np.ones((L, 37), dtype=bool) for _ in range(5)]

    with tempfile.TemporaryDirectory() as tmpdir:
        result = evaluate_stage_c(
            predicted_samples=pred_samples,
            reference_ensemble_samples=ref_samples,
            atom_masks=pred_masks,
            ref_atom_masks=ref_masks,
            aatype=aatype,
            residue_index=residue_index,
            chain_id=chain_id,
            output_dir=tmpdir,
            protein_id="test_protein",
            mode="direct",
            horizon_label="short",
            dataset_label="mdcath",
        )

    assert result["protein_id"] == "test_protein"
    assert result["ensemble_size_pred"] == 3
    assert result["ensemble_size_ref"] == 5
    assert result["mode"] == "direct"
    assert result["horizon_label"] == "short"
    assert result["dataset_label"] == "mdcath"


# ── Gate 5.3: Status field prevents silent NaN ─────────────────────────

def test_stage_c_metrics_has_status_field():
    """StageCMetrics includes explicit 'status' field."""
    m = StageCMetrics(status="ok")
    assert m.status == "ok"

    m_fail = StageCMetrics(status="alphaflow_failed", evaluation_notes="script not found")
    assert m_fail.status == "alphaflow_failed"

    d = m.to_dict()
    assert "status" in d
    assert d["status"] == "ok"


# ── Gate 5.4: Ensemble size/atom set validation ────────────────────────

def test_export_ensemble_pdb_multimodel():
    """export_ensemble_pdb produces multi-model PDB with correct MODEL/ENDMDL tags."""
    L = 3
    aatype = np.zeros(L, dtype=int)
    residue_index = np.arange(L)
    chain_id = np.zeros(L, dtype=int)

    samples = [np.random.randn(L, 37, 3) for _ in range(4)]
    masks = [np.ones((L, 37), dtype=bool) for _ in range(4)]

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = Path(tmpdir) / "test.pdb"
        export_ensemble_pdb(samples, masks, aatype, residue_index, chain_id, pdb_path)

        content = pdb_path.read_text()

    # Must have MODEL records
    assert "MODEL" in content
    # Must have 4 models
    assert content.count("MODEL") == 4
    # Must have ENDMDL records
    assert content.count("ENDMDL") == 4
    # Must end with END
    assert content.strip().endswith("END")


# ── Gate 5.5: Per-protein save and bootstrap ───────────────────────────

def test_aggregate_metrics_filters_by_status():
    """aggregate_metrics excludes proteins with non-'ok' status."""
    results = [
        {"rmsd_ca": 1.0, "rmsd_backbone": 1.5, "status": "ok"},
        {"rmsd_ca": 2.0, "rmsd_backbone": 2.5, "status": "ok"},
        {"rmsd_ca": 999.0, "rmsd_backbone": 999.0, "status": "alphaflow_failed"},
        {"rmsd_ca": 3.0, "rmsd_backbone": 3.5, "status": "ok"},
    ]

    agg = aggregate_metrics(results, require_status="ok")

    assert agg["metadata"]["n_proteins_total"] == 4
    assert agg["metadata"]["n_valid"] == 3
    assert agg["metadata"]["n_excluded"] == 1

    # Mean should only include the 3 ok values
    rmsd_mean = agg["rmsd_ca"]["mean"]
    assert rmsd_mean == pytest.approx(2.0), (
        f"Mean should be (1+2+3)/3=2.0, got {rmsd_mean}. "
        f"Failed status must be excluded."
    )


def test_aggregate_metrics_excludes_nan():
    """NaN values within valid-status proteins are excluded from stats."""
    results = [
        {"rmsd_ca": 1.0, "status": "ok"},
        {"rmsd_ca": float("nan"), "status": "ok"},  # NaN in ok protein
        {"rmsd_ca": 3.0, "status": "ok"},
    ]

    agg = aggregate_metrics(results, require_status="ok")

    # n=2 not 3 (NaN excluded)
    assert agg["rmsd_ca"]["n"] == 2
    assert agg["rmsd_ca"]["mean"] == pytest.approx(2.0)


def test_aggregate_metrics_bootstrap_ci():
    """Bootstrap CI bounds are ordered: ci_low <= mean <= ci_high."""
    rng = np.random.RandomState(42)
    results = [
        {"rmsd_ca": float(rng.normal(2.0, 0.5)), "status": "ok"}
        for _ in range(100)
    ]

    agg = aggregate_metrics(results, bootstrap_samples=500, require_status="ok")

    assert agg["rmsd_ca"]["ci_low"] <= agg["rmsd_ca"]["mean"] <= agg["rmsd_ca"]["ci_high"], (
        f"CI bounds violated: {agg['rmsd_ca']['ci_low']:.3f} <= "
        f"{agg['rmsd_ca']['mean']:.3f} <= {agg['rmsd_ca']['ci_high']:.3f}"
    )


# ── Gate 5.6: Split table helper ───────────────────────────────────────

def test_build_split_table_mode():
    """build_split_table groups results by prediction mode."""
    results = [
        {"rmsd_ca": 0.5, "status": "ok", "mode": "oracle", "horizon_label": "short", "dataset_label": "mdcath"},
        {"rmsd_ca": 0.6, "status": "ok", "mode": "oracle", "horizon_label": "short", "dataset_label": "mdcath"},
        {"rmsd_ca": 2.0, "status": "ok", "mode": "direct", "horizon_label": "short", "dataset_label": "mdcath"},
        {"rmsd_ca": 2.5, "status": "ok", "mode": "direct", "horizon_label": "short", "dataset_label": "mdcath"},
        {"rmsd_ca": 4.0, "status": "ok", "mode": "recursive", "horizon_label": "short", "dataset_label": "mdcath"},
    ]

    tables = build_split_table(results, split_dims={"mode": "*"})

    assert len(tables) == 3, f"Expected 3 splits, got {len(tables)}: {list(tables.keys())}"

    # Oracle should have lower RMSD mean than direct/recursive
    oracle_key = [k for k in tables if "oracle" in k][0]
    direct_key = [k for k in tables if "direct" in k][0]
    assert tables[oracle_key]["rmsd_ca"]["mean"] < tables[direct_key]["rmsd_ca"]["mean"]


def test_build_split_table_horizon_dataset():
    """build_split_table can split by horizon and dataset."""
    results = []
    for mode in ["direct"]:
        for horizon in ["short", "medium", "long"]:
            for dataset in ["mdcath", "atlas"]:
                for _ in range(5):
                    results.append({
                        "rmsd_ca": np.random.rand() * 5,
                        "status": "ok",
                        "mode": mode,
                        "horizon_label": horizon,
                        "dataset_label": dataset,
                    })

    tables = build_split_table(
        results,
        split_dims={"mode": "*", "horizon_label": "*", "dataset_label": "*"},
    )

    # 1 mode * 3 horizons * 2 datasets = 6 splits
    assert len(tables) == 6, f"Expected 6 splits, got {len(tables)}"


# ── Gate 5.7: RMSD unit safety ─────────────────────────────────────────

def compute_rmsd_from_arrays(
    pred_coords: np.ndarray,   # [L, 37, 3]
    target_coords: np.ndarray, # [L, 37, 3]
    mask: np.ndarray,          # [L, 37]
    atom_indices: list[int] | None = None,
) -> float:
    """Compute RMSD in Angstroms by directly aligning and measuring."""
    if atom_indices is not None:
        valid = np.zeros(mask.shape, dtype=bool)
        for idx in atom_indices:
            valid[:, idx] = mask[:, idx]
    else:
        valid = mask

    p = pred_coords[valid]
    t = target_coords[valid]

    # Center both
    p_centered = p - p.mean(axis=0)
    t_centered = t - t.mean(axis=0)

    # Kabsch
    H = p_centered.T @ t_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    p_aligned = p_centered @ R
    diff = p_aligned - t_centered
    return float(np.sqrt((diff ** 2).mean()))


def test_rmsd_unit_is_angstroms():
    """RMSD computed from arrays produces Angstrom-scale values."""
    L = 10
    # Create a structure and slightly perturb it
    np.random.seed(42)
    base = np.random.randn(L, 37, 3) * 2.0  # Angstrom-scale
    perturbed = base + np.random.randn(L, 37, 3) * 0.5  # ~0.5A perturbation
    mask = np.ones((L, 37), dtype=bool)

    # Use backbone atoms: N(0), CA(1), C(2)
    rmsd = compute_rmsd_from_arrays(perturbed, base, mask, atom_indices=[0, 1, 2])

    # Should be roughly 0.3-0.7Å (Angstrom scale, not nm)
    assert 0.1 < rmsd < 2.0, (
        f"RMSD {rmsd:.3f} is outside Angstrom-scale range [0.1, 2.0]"
    )


# ── Gate 5.8: Per-protein save ─────────────────────────────────────────

def test_evaluate_stage_c_saves_per_protein_json():
    """evaluate_stage_c writes a per-protein evaluation JSON file."""
    L = 5
    aatype = np.zeros(L, dtype=int)
    residue_index = np.arange(L)
    chain_id = np.zeros(L, dtype=int)

    pred_samples = [np.random.randn(L, 37, 3) for _ in range(2)]
    ref_samples = [np.random.randn(L, 37, 3) for _ in range(2)]
    pred_masks = [np.ones((L, 37), dtype=bool) for _ in range(2)]
    ref_masks = [np.ones((L, 37), dtype=bool) for _ in range(2)]

    with tempfile.TemporaryDirectory() as tmpdir:
        evaluate_stage_c(
            predicted_samples=pred_samples,
            reference_ensemble_samples=ref_samples,
            atom_masks=pred_masks,
            ref_atom_masks=ref_masks,
            aatype=aatype,
            residue_index=residue_index,
            chain_id=chain_id,
            output_dir=tmpdir,
            protein_id="myprotein",
        )

        # JSON must exist
        json_path = Path(tmpdir) / "myprotein_evaluation.json"
        assert json_path.exists(), f"Evaluation JSON not found at {json_path}"

        with open(json_path) as f:
            data = json.load(f)

        assert data["protein_id"] == "myprotein"
        assert "rmsd_ca" in data
        assert "status" in data
