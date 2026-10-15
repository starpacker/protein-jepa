from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot V4-D rate calibration results.")
    parser.add_argument(
        "--coarse",
        type=Path,
        default=ROOT / "artifacts/paper_eval/v4d2_rate_sweep_validation.json",
    )
    parser.add_argument(
        "--refined",
        type=Path,
        default=ROOT / "artifacts/paper_eval/v4d2_rate_sweep_validation_refined.json",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=ROOT / "docs/results/v4d2_rate_scale_025_test_accuracy.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/assets/v4d_rate_pareto.png",
    )
    args = parser.parse_args()
    records = {}
    for path in (args.coarse, args.refined):
        data = json.loads(path.read_text())
        for scale, values in data["scales"].items():
            records[float(scale)] = values["aggregate"]
    scales = sorted(records)
    skill = [100.0 * records[scale]["persistence_relative_skill"] for scale in scales]
    history = [100.0 * records[scale]["history_advantage"] for scale in scales]
    horizon = [100.0 * records[scale]["horizon_advantage"] for scale in scales]
    fixed = [100.0 * records[scale]["correct_vs_best_fixed_skill"] for scale in scales]
    lambda2 = [records[scale]["mean_lambda2"] for scale in scales]
    test = json.loads(args.test.read_text())["scales"]["0.25"]["aggregate"]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    left, right = axes
    left.plot(scales, skill, marker="o", label="Brier skill")
    left.plot(scales, history, marker="s", label="History advantage")
    left.plot(scales, horizon, marker="^", label="Horizon advantage")
    left.plot(scales, fixed, marker="d", label="Correct vs fixed")
    left.axhline(0.0, color="black", linewidth=0.8)
    left.axvline(0.25, color="#b23a48", linestyle="--", linewidth=1.2)
    left.set(xlabel="Generator rate scale", ylabel="Held-out metric (%)", title="Validation rate calibration")
    left.legend(frameon=False, fontsize=8)

    right.plot(scales, lambda2, marker="o", color="#2a6f97", label=r"Base $\lambda_2$")
    right.axvline(0.25, color="#b23a48", linestyle="--", linewidth=1.2, label="Selected on validation")
    right.scatter([0.25], [test["mean_lambda2"]], color="#b23a48", s=45, zorder=3)
    right.text(
        0.27,
        test["mean_lambda2"] - 0.03,
        f"Test: Brier {100 * test['persistence_relative_skill']:.1f}%\n"
        f"History {100 * test['history_advantage']:.1f}%",
        fontsize=8,
        va="top",
    )
    right.set(
        xlabel="Generator rate scale",
        ylabel=r"Second eigenvalue $\lambda_2$",
        title="Skill-timescale tradeoff",
        ylim=(0.35, 1.02),
    )
    right.legend(frameon=False, fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
