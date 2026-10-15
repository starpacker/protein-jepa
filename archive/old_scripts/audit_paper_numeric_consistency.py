#!/usr/bin/env python3
"""Audit easily-checkable numeric inconsistencies in the paper draft.

The goal is not to parse all LaTeX. This script records arithmetic checks that
reviewers can verify directly from visible tables, especially Table 3 and
Table 5 in ``paper/experiment.tex``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


NUMBER = r"[-+]?(?:\d+\.\d+|\d+)"


def _extract_table3(tex: str) -> dict:
    labels = {
        "Causal Transformer": r"Causal Transformer\s*&\s*\$?\\mathbf\{?(" + NUMBER + r")\}?\$?\s*&\s*\$?(" + NUMBER + r")\$?\s*&\s*\$?(" + NUMBER + r")\$?",
        "GRU": r"GRU\s*&\s*\$?(" + NUMBER + r")\$?\s*&\s*\$?(" + NUMBER + r")\$?\s*&\s*\$?(" + NUMBER + r")\$?",
        "Generic Markov": r"Generic Markov\s*&\s*\$?(" + NUMBER + r")\$?\s*&\s*\$?(" + NUMBER + r")\$?\s*&\s*\$?(" + NUMBER + r")\$?",
        "LaSP": r"\\textbf\{LaSP\}\s*&\s*\$?(" + NUMBER + r")\$?\s*&\s*\$?\\mathbf\{?(" + NUMBER + r")\}?\$?\s*&\s*\$?\\mathbf\{?(" + NUMBER + r")\}?\$?",
    }
    rows = {}
    for name, pattern in labels.items():
        match = re.search(pattern, tex)
        if not match:
            continue
        mse, dw, jsd = (float(match.group(i)) for i in range(1, 4))
        rows[name] = {"mse": mse, "dw": dw, "stationary_jsd": jsd}
    if "LaSP" not in rows:
        raise RuntimeError("could not extract LaSP row from Table 3")
    lasp_jsd = rows["LaSP"]["stationary_jsd"]
    deltas = {
        name: lasp_jsd - row["stationary_jsd"]
        for name, row in rows.items()
        if name != "LaSP"
    }
    best_jsd = min(rows.items(), key=lambda item: item[1]["stationary_jsd"])[0]
    return {"rows": rows, "lasp_minus_baseline_jsd": deltas, "best_stationary_jsd": best_jsd}


def _extract_table5(tex: str) -> dict:
    label = tex.find(r"\label{tab:k1_its}")
    if label < 0:
        raise RuntimeError("could not find Table 5 label tab:k1_its")
    begin = tex.rfind(r"\begin{table", 0, label)
    end = tex.find(r"\end{table", label)
    if begin < 0 or end < 0:
        raise RuntimeError("could not isolate Table 5 block")
    block = tex[begin:end]

    def extract_row_numbers(name_pattern: str) -> list[float]:
        match = re.search(r"^[ \t]*" + name_pattern + r"[ \t]*(.*?)\\\\", block, flags=re.S | re.M)
        if not match:
            return []
        return [float(value) for value in re.findall(NUMBER, match.group(1))]

    row_match = re.search(r"\\textbf\{LaSP \(ours\)\}(.*?)\\\\", block, flags=re.S)
    if not row_match:
        raise RuntimeError("could not extract LaSP row from Table 5")
    lasp_numbers = [float(value) for value in re.findall(NUMBER, row_match.group(1))]
    if len(lasp_numbers) < 9:
        raise RuntimeError(f"could not extract all LaSP Table 5 numbers, got {lasp_numbers}")
    values = lasp_numbers[:8]
    arithmetic_mean = sum(values) / len(values)
    displayed_mean = lasp_numbers[8]
    baseline_means = {}
    for name, pattern in [
        ("tICA+MSM", re.escape("tICA+MSM")),
        ("MSM", r"MSM"),
        ("Persistence", r"Persistence"),
    ]:
        values_for_row = extract_row_numbers(pattern)
        if len(values_for_row) >= 9:
            baseline_means[name] = values_for_row[8]
    improvements = {
        name: (value - arithmetic_mean) / value
        for name, value in baseline_means.items()
        if value != 0.0
    }
    return {
        "lasp_values": values,
        "lasp_arithmetic_mean": arithmetic_mean,
        "lasp_displayed_mean": displayed_mean,
        "baseline_means": baseline_means,
        "relative_error_reductions": improvements,
    }


def run(args: argparse.Namespace) -> dict:
    tex = Path(args.input).read_text()
    return {
        "input": args.input,
        "table3": _extract_table3(tex),
        "table5": _extract_table5(tex),
        "notes": [
            "Table 3 deltas are LaSP minus baseline; negative means lower/better JSD for LaSP.",
            "Table 5 mean is the arithmetic mean of the eight visible LaSP horizon values.",
        ],
    }


def write_summary(result: dict, path: Path) -> None:
    t3 = result["table3"]
    t5 = result["table5"]
    lines = [
        "# Paper Numeric Consistency Audit",
        "",
        f"Input: `{result['input']}`",
        "",
        "## Table 3 Stationary JSD",
        "",
        "| Method | MSE | DW | Stationary JSD | LaSP - method JSD |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in t3["rows"].items():
        delta = t3["lasp_minus_baseline_jsd"].get(name)
        delta_text = "--" if delta is None else f"{delta:.3f}"
        lines.append(
            f"| {name} | {row['mse']:.5f} | {row['dw']:.2f} | {row['stationary_jsd']:.3f} | {delta_text} |"
        )
    lines.extend([
        "",
        f"Best stationary JSD according to the table: **{t3['best_stationary_jsd']}**.",
        "",
        "Correct prose deltas:",
    ])
    for name, delta in t3["lasp_minus_baseline_jsd"].items():
        lines.append(f"- LaSP - {name}: {delta:.3f}")
    lines.extend([
        "",
        "## Table 5 ITS Mean",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| LaSP displayed mean | {t5['lasp_displayed_mean']:.6f} |",
        f"| LaSP arithmetic mean from visible horizons | {t5['lasp_arithmetic_mean']:.6f} |",
    ])
    for name, value in t5["baseline_means"].items():
        lines.append(f"| {name} displayed mean | {value:.6f} |")
    lines.extend(["", "Relative error reductions using arithmetic LaSP mean:"])
    for name, value in t5["relative_error_reductions"].items():
        lines.append(f"- vs {name}: {100.0 * value:.2f}%")
    lines.append("")
    rounded = round(float(t5["lasp_arithmetic_mean"]), 3)
    if abs(float(t5["lasp_displayed_mean"]) - rounded) < 5.0e-4:
        lines.append("Status: displayed LaSP mean matches the arithmetic mean after rounding to three decimals.")
    else:
        lines.append("Action: either change the displayed LaSP mean to 0.459 or document a weighted mean with weights.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="paper/experiment.tex")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default="")
    args = parser.parse_args()
    result = run(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    if args.summary:
        write_summary(result, Path(args.summary))
    print(json.dumps({
        "output": str(out),
        "summary": args.summary,
        "table3_best_jsd": result["table3"]["best_stationary_jsd"],
        "table5_lasp_arithmetic_mean": result["table5"]["lasp_arithmetic_mean"],
    }, indent=2))


if __name__ == "__main__":
    main()
