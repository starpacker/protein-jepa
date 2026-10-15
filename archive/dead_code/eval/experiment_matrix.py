"""Gate 8 experiment matrix: final evaluation with all baselines and controls.

Canonical experiment matrix (from audit plan Gate 8):
1. Persistence structure (last observed frame replicated)
2. Constant-velocity / aligned displacement
3. Stage A direct latent + Stage C decoder
4. Canonical LaSP recursive + Stage C decoder
5. Negative controls: no bath, constant diffusion, no free-energy, global bath
6. Oracle future latent + decoder (decoding upper bound)
7. Ground-truth structure passed through evaluator (metric ceiling)

Requirements:
- 3 random seeds per condition
- Blind protein test set (only run once after protocol lock)
- Paired bootstrap CI at protein level
- Pre-registered multiple comparison correction
- Report: parameters, training/inference cost, ensemble size, step count, failure rate
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Experiment definitions
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ExperimentCondition:
    """A single experimental condition in the Gate 8 matrix."""
    name: str                                        # short identifier
    description: str                                  # one-line description
    category: str                                     # "main", "baseline", "ablation", "bound"
    prediction_mode: str                              # "direct_endpoint" or "autonomous_recursive"
    use_dynamics: bool = True                         # whether to use LaSP dynamics
    use_bath: bool = True                             # whether to use Mori bath
    use_free_energy: bool = True                      # whether to use energy-driven forces
    use_state_dependent_diffusion: bool = True        # whether D depends on state
    use_oracle_latent: bool = False                   # use ground-truth future Q
    use_persistence: bool = False                     # just copy last observed structure
    num_seeds: int = 3                                # number of random seeds
    horizon_weights: dict[str, float] = field(default_factory=lambda: {"short": 1.0, "medium": 1.0, "long": 1.0})

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass(frozen=True)
class HorizonProtocol:
    """Physical horizon definitions for short/medium/long evaluation.

    Gate 6: all horizons must be physical time in picoseconds,
    not frame counts. These are set per-dataset based on ACF analysis.
    """
    short_ps: float = 1000.0      # 1 ns
    medium_ps: float = 10000.0    # 10 ns
    long_ps: float = 100000.0     # 100 ns
    dataset: str = "mdcath"

    def horizon_labels(self) -> dict[str, float]:
        return {"short": self.short_ps, "medium": self.medium_ps, "long": self.long_ps}


# ═══════════════════════════════════════════════════════════════════════
# Standard experiment matrix
# ═══════════════════════════════════════════════════════════════════════

def build_experiment_matrix() -> list[ExperimentCondition]:
    """Build the Gate 6 evaluation matrix.

    Gate 6 minimum baselines (from audit plan):
    - persistence structure
    - constant velocity / aligned displacement
    - Stage A direct
    - causal Transformer
    - conditional stochastic predictor
    - LaSP no-bath
    - Full LaSP
    - oracle latent
    - ground-truth evaluator ceiling

    Each condition is tested with 3 random seeds and at
    short, medium, long physical horizons.
    """
    conditions = []

    # ── Bounds ──────────────────────────────────────────────────────

    # Ground truth (metric ceiling)
    conditions.append(ExperimentCondition(
        name="ground_truth",
        description="Ground-truth structure passed through export/evaluator",
        category="bound",
        prediction_mode="direct_endpoint",
        use_dynamics=False,
        use_bath=False,
        use_free_energy=False,
        use_state_dependent_diffusion=False,
        use_oracle_latent=True,
        use_persistence=False,
    ))

    # Oracle latent (decoding upper bound)
    conditions.append(ExperimentCondition(
        name="oracle_latent",
        description="Oracle future Q_t + decoder (perfect latent)",
        category="bound",
        prediction_mode="direct_endpoint",
        use_dynamics=False,
        use_bath=False,
        use_free_energy=False,
        use_state_dependent_diffusion=False,
        use_oracle_latent=True,
        use_persistence=False,
    ))

    # ── Baselines ───────────────────────────────────────────────────

    # Persistence (no movement)
    conditions.append(ExperimentCondition(
        name="persistence",
        description="Last observed structure replicated as prediction",
        category="baseline",
        prediction_mode="direct_endpoint",
        use_dynamics=False,
        use_bath=False,
        use_free_energy=False,
        use_state_dependent_diffusion=False,
        use_persistence=True,
    ))

    # Constant-velocity / aligned displacement
    conditions.append(ExperimentCondition(
        name="constant_velocity",
        description="Linear extrapolation from last observed displacement",
        category="baseline",
        prediction_mode="direct_endpoint",
        use_dynamics=False,
        use_bath=False,
        use_free_energy=False,
        use_state_dependent_diffusion=False,
        use_persistence=False,
    ))

    # Stage A direct latent + Stage C decoder (no dynamics)
    conditions.append(ExperimentCondition(
        name="stage_a_direct",
        description="Stage A direct latent prediction + Stage C decoder (no LaSP)",
        category="baseline",
        prediction_mode="direct_endpoint",
        use_dynamics=False,
        use_bath=False,
        use_free_energy=False,
        use_state_dependent_diffusion=False,
    ))

    # ── Main ────────────────────────────────────────────────────────

    # Canonical LaSP recursive + Stage C decoder
    conditions.append(ExperimentCondition(
        name="lasp_recursive",
        description="Canonical LaSP autonomous recursive + Stage C decoder",
        category="main",
        prediction_mode="autonomous_recursive",
        use_dynamics=True,
        use_bath=True,
        use_free_energy=True,
        use_state_dependent_diffusion=True,
    ))

    # Canonical LaSP direct endpoint + Stage C decoder
    conditions.append(ExperimentCondition(
        name="lasp_direct",
        description="Canonical LaSP direct endpoint + Stage C decoder",
        category="main",
        prediction_mode="direct_endpoint",
        use_dynamics=True,
        use_bath=True,
        use_free_energy=True,
        use_state_dependent_diffusion=True,
    ))

    # ── Ablations (negative controls) ───────────────────────────────

    # No bath
    conditions.append(ExperimentCondition(
        name="no_bath",
        description="Canonical LaSP without Mori bath (only overdamped dynamics)",
        category="ablation",
        prediction_mode="autonomous_recursive",
        use_dynamics=True,
        use_bath=False,
        use_free_energy=True,
        use_state_dependent_diffusion=True,
    ))

    # Constant diffusion
    conditions.append(ExperimentCondition(
        name="constant_diffusion",
        description="LaSP with constant (state-independent) diffusion",
        category="ablation",
        prediction_mode="autonomous_recursive",
        use_dynamics=True,
        use_bath=True,
        use_free_energy=True,
        use_state_dependent_diffusion=False,
    ))

    # No free energy (zero force)
    conditions.append(ExperimentCondition(
        name="no_free_energy",
        description="LaSP with zero free-energy force (pure diffusion)",
        category="ablation",
        prediction_mode="autonomous_recursive",
        use_dynamics=True,
        use_bath=True,
        use_free_energy=False,
        use_state_dependent_diffusion=True,
    ))

    # Global bath (wrong structure — per-protein, not per-residue)
    conditions.append(ExperimentCondition(
        name="global_bath",
        description="LaSP with global (per-protein) bath instead of per-residue",
        category="ablation",
        prediction_mode="autonomous_recursive",
        use_dynamics=True,
        use_bath=True,
        use_free_energy=True,
        use_state_dependent_diffusion=True,
    ))

    return conditions


def validate_experiment_matrix(conditions: list[ExperimentCondition]) -> list[str]:
    """Validate experiment matrix completeness.

    Returns:
        list of warnings (empty if complete)
    """
    warnings = []
    categories = set(c.name for c in conditions)
    names = set(c.name for c in conditions)

    if "ground_truth" not in names:
        warnings.append("Missing ground_truth (metric ceiling)")
    if "oracle_latent" not in names:
        warnings.append("Missing oracle_latent (decoding bound)")
    if "persistence" not in names:
        warnings.append("Missing persistence (trivial baseline)")
    if "lasp_recursive" not in names:
        warnings.append("Missing lasp_recursive (main result)")
    if "no_bath" not in names:
        warnings.append("Missing no_bath ablation")
    if "no_free_energy" not in names:
        warnings.append("Missing no_free_energy ablation")

    return warnings


# ═══════════════════════════════════════════════════════════════════════
# Experiment runner
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentResult:
    """Result from one experiment condition."""
    condition: str
    seed: int
    protein_id: str
    horizon: str                             # "short", "medium", or "long"
    horizon_ps: float
    prediction_mode: str
    metrics: dict[str, float] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    ensemble_size: int = 0
    model_parameters: int = 0
    error: str = ""


class ExperimentRunner:
    """Runner for the Gate 8 experiment matrix.

    Executes each condition × seed × horizon and collects results.
    """

    def __init__(
        self,
        conditions: list[ExperimentCondition],
        seeds: list[int] | None = None,
        output_dir: str | Path = "./experiments",
    ):
        self.conditions = conditions
        self.seeds = seeds or [42, 123, 456]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save experiment matrix
        matrix_path = self.output_dir / "experiment_matrix.json"
        matrix_path.write_text(json.dumps(
            [c.to_dict() for c in conditions],
            indent=2,
        ))

    def run(
        self,
        proteins: list[str],
        horizon_protocol: Any,  # HorizonProtocol
        predict_fn: Callable | None = None,
        progress_callback: Callable | None = None,
    ) -> dict[str, Any]:
        """Run all experiments.

        Args:
            proteins: list of protein IDs in test set
            horizon_protocol: HorizonProtocol with short/medium/long values
            predict_fn: (condition, seed, protein_id, horizon_ps) -> (samples, target, mask, aatype, ...)
            progress_callback: (completed, total) -> None

        Returns:
            dict with all results organized by condition
        """
        if predict_fn is None:
            raise ValueError("predict_fn is required for running experiments")

        results: dict[str, list[ExperimentResult]] = {}
        total_tasks = len(self.conditions) * len(self.seeds) * len(proteins)
        completed = 0

        horizons = {
            "short": horizon_protocol.short_ps,
            "medium": horizon_protocol.medium_ps,
            "long": horizon_protocol.long_ps,
        }

        for condition in self.conditions:
            cond_results: list[ExperimentResult] = []
            for seed in self.seeds:
                for protein_id in proteins:
                    for horizon_name, horizon_ps in horizons.items():

                        t_start = time.time()
                        try:
                            result_data = predict_fn(
                                condition.name, seed, protein_id, horizon_ps
                            )

                            result = ExperimentResult(
                                condition=condition.name,
                                seed=seed,
                                protein_id=protein_id,
                                horizon=horizon_name,
                                horizon_ps=horizon_ps,
                                prediction_mode=condition.prediction_mode,
                                metrics=result_data.get("metrics", {}),
                                runtime_seconds=time.time() - t_start,
                                ensemble_size=result_data.get("ensemble_size", 0),
                                model_parameters=result_data.get("num_params", 0),
                            )
                        except Exception as e:
                            result = ExperimentResult(
                                condition=condition.name,
                                seed=seed,
                                protein_id=protein_id,
                                horizon=horizon_name,
                                horizon_ps=horizon_ps,
                                prediction_mode=condition.prediction_mode,
                                error=str(e),
                                runtime_seconds=time.time() - t_start,
                            )
                            import traceback
                            traceback.print_exc()

                        cond_results.append(result)
                        completed += 1

                        if progress_callback:
                            progress_callback(completed, total_tasks)

            results[condition.name] = cond_results

        # Save all results
        serializable = {}
        for cond_name, res_list in results.items():
            serializable[cond_name] = [
                {
                    "protein_id": r.protein_id,
                    "seed": r.seed,
                    "horizon": r.horizon,
                    "horizon_ps": r.horizon_ps,
                    "prediction_mode": r.prediction_mode,
                    "metrics": r.metrics,
                    "runtime_seconds": r.runtime_seconds,
                    "ensemble_size": r.ensemble_size,
                    "model_parameters": r.model_parameters,
                    "error": r.error,
                }
                for r in res_list
            ]

        results_path = self.output_dir / "experiment_results.json"
        results_path.write_text(json.dumps(serializable, indent=2, default=str))

        return serializable


# ═══════════════════════════════════════════════════════════════════════
# Result table generator
# ═══════════════════════════════════════════════════════════════════════

def generate_result_table(
    results: dict[str, Any],
    metrics: list[str] | None = None,
    horizons: list[str] | None = None,
    output_format: str = "latex",
) -> str:
    """Generate paper-ready result table from experiment results.

    Args:
        results: dict from ExperimentRunner.run()
        metrics: which metrics to include (default: all 7)
        horizons: which horizons to report (default: short, medium, long)
        output_format: "latex", "markdown", or "plain"

    Returns:
        formatted table string
    """
    if metrics is None:
        metrics = [
            "rmsd_ca", "rmsd_backbone", "per_target_rmsf_r",
            "rmwd", "rmwd_var", "md_pca_w2",
            "weak_contacts_j", "transient_contacts_j",
        ]

    if horizons is None:
        horizons = ["short", "medium", "long"]


    # Condition ordering
    condition_order = [
        "ground_truth", "oracle_latent",
        "persistence", "constant_velocity", "stage_a_direct",
        "lasp_recursive", "lasp_direct",
        "no_bath", "constant_diffusion", "no_free_energy", "global_bath",
    ]

    if output_format == "latex":
        return _format_latex(results, condition_order, metrics, horizons)
    elif output_format == "markdown":
        return _format_markdown(results, condition_order, metrics, horizons)
    else:
        return _format_plain(results, condition_order, metrics, horizons)


def _aggregate_condition(
    results: dict[str, Any],
    condition: str,
    horizon: str,
    metric: str,
) -> tuple[float, float]:
    """Compute mean ± std for one condition × horizon × metric."""
    if condition not in results:
        return float("nan"), float("nan")

    values = []
    for r in results[condition]:
        if r.get("horizon") == horizon and not r.get("error"):
            v = r.get("metrics", {}).get(metric, float("nan"))
            if not (np.isnan(v) if isinstance(v, float) else False):
                values.append(float(v))

    if not values:
        return float("nan"), float("nan")

    return float(np.mean(values)), float(np.std(values)) if len(values) > 1 else 0.0


def _format_latex(results, conditions, metrics, horizons):
    """Generate LaTeX table."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Stage C evaluation results (mean $\pm$ std over 3 seeds).}")
    lines.append(r"\begin{tabular}{l" + "c" * (len(metrics) * len(horizons)) + "}")
    lines.append(r"\toprule")

    # Header
    header = "Condition"
    for h in horizons:
        for m in metrics:
            header += f" & {m} ({h})"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for cond in conditions:
        if cond not in results:
            continue
        row = cond.replace("_", r"\_")
        for h in horizons:
            for m in metrics:
                mean_val, std_val = _aggregate_condition(results, cond, h, m)
                if np.isnan(mean_val):
                    row += " & --"
                else:
                    row += f" & {mean_val:.3f}$\\pm${std_val:.3f}"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _format_markdown(results, conditions, metrics, horizons):
    """Generate markdown table."""
    lines = []
    header = "| Condition " + " | ".join(f"{m} ({h})" for h in horizons for m in metrics) + " |"
    lines.append(header)
    sep = "|" + "|".join([" --- " for _ in range(len(metrics) * len(horizons))]) + " |"
    lines.append(sep)

    for cond in conditions:
        if cond not in results:
            continue
        row = f"| {cond} "
        for h in horizons:
            for m in metrics:
                mean_val, std_val = _aggregate_condition(results, cond, h, m)
                if np.isnan(mean_val):
                    row += "| -- "
                else:
                    row += f"| {mean_val:.3f}±{std_val:.3f} "
        row += "|"
        lines.append(row)

    return "\n".join(lines)


def _format_plain(results, conditions, metrics, horizons):
    """Generate plain-text table."""
    lines = []

    for cond in conditions:
        if cond not in results:
            continue
        lines.append(f"\n{cond}:")
        for h in horizons:
            line = f"  {h}:"
            for m in metrics:
                mean_val, std_val = _aggregate_condition(results, cond, h, m)
                if not np.isnan(mean_val):
                    line += f"  {m}={mean_val:.4f}±{std_val:.4f}"
            lines.append(line)

    return "\n".join(lines)


__all__ = [
    "HorizonProtocol",
    "ExperimentCondition",
    "ExperimentResult",
    "ExperimentRunner",
    "build_experiment_matrix",
    "validate_experiment_matrix",
    "generate_result_table",
]