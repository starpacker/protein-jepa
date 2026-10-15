from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot V4-D hard-state test metrics.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "docs/results/v4d2_rate_scale_025_test_accuracy.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/assets/v4d_test_accuracy.png",
    )
    args = parser.parse_args()
    metrics = json.loads(args.input.read_text())["scales"]["0.25"]["aggregate"]
    random_top1 = 1.0 / 6.0
    random_top2 = 2.0 / 6.0
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True)

    labels = ["Top-1\n(persistence ref.)", "Top-2\n(random ref.)", "Balanced\n(random ref.)"]
    model = [metrics["top1_accuracy"], metrics["top2_accuracy"], metrics["balanced_accuracy"]]
    baseline = [metrics["persistence_accuracy"], random_top2, random_top1]
    x = range(len(labels))
    axes[0].bar([value - 0.18 for value in x], [100 * value for value in model], width=0.36, label="V4-D")
    axes[0].bar([value + 0.18 for value in x], [100 * value for value in baseline], width=0.36, label="Reference")
    axes[0].set_xticks(list(x), labels)
    axes[0].set(ylabel="Accuracy (%)", title="Six-state forecast accuracy", ylim=(0, 45))
    axes[0].legend(frameon=False)

    transition_labels = ["Precision", "Recall", "Changed-state\naccuracy", "Unchanged-state\naccuracy"]
    transition_values = [
        metrics["transition_precision"], metrics["transition_recall"],
        metrics["changed_state_accuracy"], metrics["unchanged_state_accuracy"],
    ]
    colors = ["#2a6f97", "#2a6f97", "#b23a48", "#4f772d"]
    axes[1].bar(transition_labels, [100 * value for value in transition_values], color=colors)
    axes[1].set(ylabel="Score (%)", title="Transition-event diagnostics", ylim=(0, 85))
    axes[1].tick_params(axis="x", labelsize=8)
    for index, value in enumerate(transition_values):
        axes[1].text(index, 100 * value + 1.5, f"{100 * value:.1f}", ha="center", fontsize=8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
