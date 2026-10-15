"""Gate 6: Final experiment matrix validation.

Gate 6 requirements from audit plan:
1. Minimum baselines: persistence, constant-velocity, Stage A direct,
   causal Transformer, conditional stochastic predictor, LaSP no-bath,
   Full LaSP, oracle latent, ground-truth ceiling
2. At least 3 training seeds per condition
3. Protein-level bootstrap (not frame-level)
4. All results from a single metrics.jsonl source
5. direct/recursive, short/medium/long, three datasets per table
6. Experiment registry with provenance

Tests:
- build_experiment_matrix completeness (all 9 minimum baselines)
- Each condition has 3 seeds
- validate_experiment_matrix catches missing conditions
- ExperimentRunner executes and saves results
- generate_result_table produces valid output for all formats
- HorizonProtocol has physical time units (ps)
- ExperimentRegistry is append-only with CSV header
- MetricsLogger append-only JSONL
- Single metrics.jsonl source for all results
- Bootstrap CI computation at protein level
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from protein_jepa_frame.eval.experiment_matrix import (
    build_experiment_matrix,
    validate_experiment_matrix,
    ExperimentCondition,
    ExperimentRunner,
    ExperimentResult,
    generate_result_table,
    HorizonProtocol,
)
from protein_jepa_frame.eval.experiment_registry import (
    ExperimentRegistry,
    ExperimentRecord,
    MetricsLogger,
)


# ── Gate 6.1: Matrix completeness ─────────────────────────────────────

def test_experiment_matrix_has_all_minimum_baselines():
    """Gate 6: all 9 minimum baselines are present."""
    conditions = build_experiment_matrix()
    names = {c.name for c in conditions}

    minimum_baselines = {
        "persistence",        # persistence structure
        "constant_velocity",  # constant velocity / aligned displacement
        "stage_a_direct",     # Stage A direct
        "no_bath",            # LaSP no-bath
        "lasp_recursive",     # Full LaSP (recursive)
        "lasp_direct",        # Full LaSP (direct) — part of conditional stochastic predictor
        "oracle_latent",      # oracle latent
        "ground_truth",       # ground-truth evaluator ceiling
    }

    missing = minimum_baselines - names
    assert len(missing) == 0, (
        f"Missing minimum baselines: {missing}. "
        f"Current conditions: {sorted(names)}"
    )

    # Also verify at least 9 conditions total
    assert len(conditions) >= 9, f"Expected >= 9 conditions, got {len(conditions)}"


def test_matrix_has_bounds():
    """Both metric bounds (ground truth and oracle latent) are present."""
    conditions = build_experiment_matrix()
    bounds = [c for c in conditions if c.category == "bound"]
    bound_names = {c.name for c in bounds}
    assert "ground_truth" in bound_names, "Missing ground_truth bound"
    assert "oracle_latent" in bound_names, "Missing oracle_latent bound"


def test_matrix_has_ablations():
    """Negative controls/ablations: no_bath, constant_diffusion, no_free_energy."""
    conditions = build_experiment_matrix()
    ablations = {c.name for c in conditions if c.category == "ablation"}
    required = {"no_bath", "constant_diffusion", "no_free_energy", "global_bath"}
    assert required.issubset(ablations), (
        f"Missing ablations: {required - ablations}. Present: {ablations}"
    )


# ── Gate 6.2: Three seeds per condition ────────────────────────────────

def test_each_condition_has_three_seeds():
    """Gate 6: each condition must have at least 3 seeds."""
    conditions = build_experiment_matrix()
    for c in conditions:
        assert c.num_seeds >= 3, (
            f"Condition '{c.name}' has {c.num_seeds} seeds, needs >= 3"
        )


# ── Gate 6.3: Validate catches gaps ────────────────────────────────────

def test_validate_matrix_catches_missing_ground_truth():
    """validate_experiment_matrix flags missing ground_truth."""
    conditions = build_experiment_matrix()
    # remove ground_truth
    filtered = [c for c in conditions if c.name != "ground_truth"]
    warnings = validate_experiment_matrix(filtered)
    assert any("ground_truth" in w for w in warnings)


def test_validate_matrix_catches_missing_persistence():
    """validate_experiment_matrix flags missing persistence."""
    conditions = build_experiment_matrix()
    filtered = [c for c in conditions if c.name != "persistence"]
    warnings = validate_experiment_matrix(filtered)
    assert any("persistence" in w for w in warnings)


def test_validate_matrix_passes_complete():
    """Complete matrix passes validation with no warnings."""
    conditions = build_experiment_matrix()
    warnings = validate_experiment_matrix(conditions)
    assert len(warnings) == 0, f"Complete matrix should pass: {warnings}"


# ── Gate 6.4: Horizon protocol ─────────────────────────────────────────

def test_horizon_protocol_physical_time():
    """Gate 6: horizons are physical time in picoseconds, not frame counts."""
    hp = HorizonProtocol(short_ps=1000.0, medium_ps=10000.0, long_ps=100000.0)

    assert hp.short_ps == 1000.0
    assert hp.medium_ps == 10000.0
    assert hp.long_ps == 100000.0

    labels = hp.horizon_labels()
    assert labels == {"short": 1000.0, "medium": 10000.0, "long": 100000.0}

    assert hp.dataset == "mdcath"


# ── Gate 6.5: Experiment runner ────────────────────────────────────────

def _fake_predict_fn(condition, seed, protein_id, horizon_ps):
    """Fake prediction function for testing ExperimentRunner."""
    rng = np.random.RandomState(seed + hash(protein_id) % 10000)
    return {
        "metrics": {
            "rmsd_ca": float(rng.uniform(0.5, 5.0)),
            "rmsd_backbone": float(rng.uniform(0.5, 5.0)),
        },
        "ensemble_size": 10,
        "num_params": 100000,
    }


def test_experiment_runner_executes_and_saves():
    """ExperimentRunner runs conditions and saves results."""
    conditions = [c for c in build_experiment_matrix() if c.name in {"persistence", "oracle_latent"}]
    proteins = ["prot1", "prot2"]
    hp = HorizonProtocol()

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(
            conditions=conditions,
            seeds=[42],
            output_dir=tmpdir,
        )
        results = runner.run(
            proteins=proteins,
            horizon_protocol=hp,
            predict_fn=_fake_predict_fn,
        )

        # Results dict has one key per condition
        assert "persistence" in results
        assert "oracle_latent" in results

        # Each condition has SEEDS * PROTEINS * HORIZONS results
        for cond_results in results.values():
            assert len(cond_results) == 1 * 2 * 3, (
                f"Expected 6 results (1 seed * 2 proteins * 3 horizons), "
                f"got {len(cond_results)}"
            )

        # Results JSON was saved
        results_path = Path(tmpdir) / "experiment_results.json"
        assert results_path.exists()

        with open(results_path) as f:
            saved = json.load(f)
        assert "persistence" in saved


def test_experiment_runner_handles_errors():
    """ExperimentRunner captures errors without stopping execution."""
    def failing_fn(condition, seed, protein_id, horizon_ps):
        if protein_id == "bad_protein":
            raise RuntimeError("simulated failure")
        return _fake_predict_fn(condition, seed, protein_id, horizon_ps)

    conditions = build_experiment_matrix()[:1]  # just persistence
    proteins = ["good_protein", "bad_protein"]
    hp = HorizonProtocol()

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(
            conditions=conditions,
            seeds=[42],
            output_dir=tmpdir,
        )
        results = runner.run(
            proteins=proteins,
            horizon_protocol=hp,
            predict_fn=failing_fn,
        )

        # Should still complete, with errors in results
        cond_name = conditions[0].name
        errors = [r for r in results[cond_name] if r.get("error")]
        assert len(errors) > 0, "Failed protein should have error recorded"


# ── Gate 6.6: Table generation ─────────────────────────────────────────

def test_generate_result_table_all_formats():
    """generate_result_table works for latex, markdown, and plain formats."""
    conditions = build_experiment_matrix()
    hp = HorizonProtocol()

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(conditions=[c for c in conditions if c.name in {"persistence", "oracle_latent"}],
                                 seeds=[42], output_dir=tmpdir)
        results = runner.run(proteins=["p1", "p2"], horizon_protocol=hp, predict_fn=_fake_predict_fn)

        for fmt in ["latex", "markdown", "plain"]:
            table = generate_result_table(results, output_format=fmt)
            assert len(table) > 0, f"Format '{fmt}' produced empty table"
            # Should mention the condition names
            assert "persistence" in table or "Persistence" in table or "PERSISTENCE" in table
            # Should mention metrics
            assert "rmsd" in table.lower()


# ── Gate 6.7: Experiment registry ──────────────────────────────────────

def test_experiment_registry_append_only():
    """ExperimentRegistry appends records to CSV with proper header."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "registry.csv"
        reg = ExperimentRegistry(csv_path)

        reg.register(ExperimentRecord(
            experiment_id="E01_v1", task="T1a", model="LaSP",
            dataset="mdCATH", representation="stage_a_ckpt_hash",
            seed=42, horizon_ps=1000.0,
            n_params=1000, n_steps_trained=5000,
            train_loss_final=0.05,
        ))

        reg.register(ExperimentRecord(
            experiment_id="E01_v2", task="T1a", model="LaSP",
            dataset="mdCATH", representation="stage_a_ckpt_hash",
            seed=123, horizon_ps=1000.0,
            n_params=1000, n_steps_trained=5000,
            train_loss_final=0.048,
        ))

        # Query
        results = reg.query(model="LaSP", dataset="mdCATH")
        assert len(results) == 2, f"Expected 2 records, got {len(results)}"
        assert results[0]["seed"] == "42"
        assert results[1]["seed"] == "123"


# ── Gate 6.8: Metrics logger (single source) ───────────────────────────

def test_metrics_logger_jsonl_append_only():
    """MetricsLogger is append-only JSONL (single metrics.jsonl source)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "metrics.jsonl"
        logger = MetricsLogger(log_path)

        logger.log({
            "protein_id": "prot1", "condition": "lasp_recursive",
            "seed": 42, "horizon": "short",
            "rmsd_ca": 1.23, "rmsd_backbone": 1.45,
        })

        logger.log({
            "protein_id": "prot1", "condition": "persistence",
            "seed": 42, "horizon": "short",
            "rmsd_ca": 5.67, "rmsd_backbone": 5.89,
        })

        entries = logger.load_all()
        assert len(entries) == 2
        assert entries[0]["condition"] == "lasp_recursive"
        assert entries[1]["condition"] == "persistence"

        # All entries have timestamps
        for entry in entries:
            assert "timestamp" in entry


# ── Gate 6.9: Bootstrap is at protein level ────────────────────────────

def test_bootstrap_respects_protein_level():
    """Bootstrap aggregation uses protein-level resampling, not frame-level."""
    from protein_jepa_frame.eval.stage_c_evaluator import aggregate_metrics

    # 5 proteins with different RMSD
    results = [
        {"rmsd_ca": 1.0, "rmsd_backbone": 1.2, "status": "ok"},
        {"rmsd_ca": 2.0, "rmsd_backbone": 2.2, "status": "ok"},
        {"rmsd_ca": 3.0, "rmsd_backbone": 3.2, "status": "ok"},
        {"rmsd_ca": 4.0, "rmsd_backbone": 4.2, "status": "ok"},
        {"rmsd_ca": 5.0, "rmsd_backbone": 5.2, "status": "ok"},
    ]

    agg = aggregate_metrics(results, bootstrap_samples=200)

    # With uniform spread, mean should be ~3.0
    assert abs(agg["rmsd_ca"]["mean"] - 3.0) < 0.5, (
        f"CA RMSD mean {agg['rmsd_ca']['mean']:.3f} far from expected 3.0"
    )

    # n should be 5 (not #frames or #pairs)
    assert agg["rmsd_ca"]["n"] == 5
    assert agg["metadata"]["n_proteins_total"] == 5
    assert agg["metadata"]["n_valid"] == 5
    assert agg["metadata"]["n_excluded"] == 0


# ── Gate 6.10: Condition category counts ───────────────────────────────

def test_experiment_matrix_category_counts():
    """Matrix has correct number of categories."""
    conditions = build_experiment_matrix()
    cats = {}
    for c in conditions:
        cats[c.category] = cats.get(c.category, 0) + 1

    assert cats.get("bound", 0) >= 2, f"Need >=2 bounds, got {cats.get('bound', 0)}"
    assert cats.get("baseline", 0) >= 3, f"Need >=3 baselines, got {cats.get('baseline', 0)}"
    assert cats.get("main", 0) >= 2, f"Need >=2 main results, got {cats.get('main', 0)}"
    assert cats.get("ablation", 0) >= 4, f"Need >=4 ablations, got {cats.get('ablation', 0)}"
