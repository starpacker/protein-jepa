#!/usr/bin/env python3
"""Run independent CA-coordinate baselines for R1 history closure.

Each protein is aligned and represented independently because mdCATH proteins
have different residue counts.  PCA/tICA/VAMP/SRV are fitted on the first
80% of each validation trajectory and evaluated on the held-out tail.  The
same current-vs-history probe protocol is then applied to every method.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baselines.data import extract_features_for_protein, load_all_proteins
from artifacts.reviewer_experiments._shared.baseline_representations import (
    PCAExtractor, TICAExtractor, VAMPExtractor, SRVExtractor,
)


def _make_extractor(name, dim, lag):
    cls = {"pca": PCAExtractor, "tica": TICAExtractor,
           "vamp": VAMPExtractor, "srv": SRVExtractor}[name]
    return cls(n_components=dim, lag=lag)


def _fit_predict_gap(z, horizon, history_len, seed=6303, epochs=80):
    """Return held-out MSE current, history, and delta for one trajectory."""
    n = len(z) - horizon - history_len + 1
    if n < 40:
        return None
    # Build aligned samples. The first 80% is probe training, the tail is test.
    Xc, Xh, Y = [], [], []
    for t in range(history_len - 1, len(z) - horizon):
        Xc.append(z[t]); Xh.append(z[t-history_len+1:t+1].reshape(-1)); Y.append(z[t+horizon])
    Xc, Xh, Y = map(np.asarray, (Xc, Xh, Y))
    cut = max(20, int(len(Y) * 0.8))
    if len(Y) - cut < 10: return None

    # A deterministic ridge probe keeps the capacity and regularization
    # protocol identical while avoiding expensive per-protein neural training.
    def train(x, y):
        xtr = np.concatenate([x[:cut], np.ones((cut, 1))], axis=1)
        xte = np.concatenate([x[cut:], np.ones((len(x)-cut, 1))], axis=1)
        reg = 1e-3 * np.eye(xtr.shape[1])
        reg[-1, -1] = 0.0
        weights = np.linalg.solve(xtr.T @ xtr + reg, xtr.T @ y[:cut])
        pred = xte @ weights
        return float(np.mean((pred - y[cut:]) ** 2))
    current = train(Xc, Y)
    history = train(Xh, Y)
    return {"current_mse": current, "history_mse": history,
            "delta_hist": current - history, "n_test": int(len(Y)-cut)}


def _bootstrap_mean(values, n_resamples=2000, seed=6303):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return [float(values.mean()), float(values.mean())]
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(n_resamples, len(values)))]
    return [float(np.quantile(samples.mean(axis=1), 0.025)),
            float(np.quantile(samples.mean(axis=1), 0.975))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="/liuyutian/protein_jepa_frame/manifests/tmjepa_v3.validation.jsonl")
    p.add_argument("--output", default="/liuyutian/protein_jepa_frame/artifacts/reviewer_experiments/representation_closure/coordinate_baselines.json")
    p.add_argument("--representations", nargs="+", default=["pca", "tica", "vamp", "srv"])
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--lag", type=int, default=1)
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--history-lengths", nargs="+", type=int, default=[1, 2, 4])
    p.add_argument("--max-proteins", type=int, default=None)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--probe-epochs", type=int, default=80)
    args = p.parse_args()
    groups = load_all_proteins(args.manifest, frame_stride=args.frame_stride,
                               max_proteins=args.max_proteins)
    print(f"Loaded {len(groups)} proteins")
    result = {"protocol": {"manifest": args.manifest, "dim": args.dim,
                            "fit_split": "first_80_percent_per_protein",
                            "coordinate_source": "aligned_CA"}, "by_representation": {}}
    for name in args.representations:
        result["by_representation"][name] = {}
        print(f"=== {name} ===")
        # Coordinate extraction and dimensionality reduction are independent
        # of probe horizon/history; cache them once per representation.
        latent_by_protein = {}
        for pid, records in groups.items():
            try:
                features = extract_features_for_protein(records, args.frame_stride)
                seqs = list(features.values())
                if not seqs: continue
                extractor = _make_extractor(name, args.dim, args.lag)
                # Fit the unsupervised representation on the training prefix
                # only; the held-out temporal suffix is evaluation data.
                fit_seqs = [x[:max(2, int(len(x) * 0.8))] for x in seqs]
                extractor.fit({str(i): x for i, x in enumerate(fit_seqs)})
                latent_by_protein[pid] = [(tid, extractor.transform({tid: x}))
                                          for tid, x in features.items()]
            except Exception as exc:
                print(f"  skip {pid}: {type(exc).__name__}: {exc}")
        for horizon in args.horizons:
            for H in args.history_lengths:
                rows = []
                for pid, sequences in latent_by_protein.items():
                    for tid, z in sequences:
                        row = _fit_predict_gap(z, horizon, H, epochs=args.probe_epochs)
                        if row is not None:
                            row.update({"protein_id": pid, "trajectory_id": tid})
                            rows.append(row)
                if rows:
                    # The inferential unit is protein, not trajectory. First
                    # average trajectories within each protein, then bootstrap
                    # those protein means as required by the protocol.
                    by_protein = {}
                    for r in rows:
                        by_protein.setdefault(r["protein_id"], []).append(r)
                    protein_rows = []
                    for pid, prs in by_protein.items():
                        protein_rows.append({
                            "protein_id": pid,
                            "current_mse": np.mean([r["current_mse"] for r in prs]),
                            "history_mse": np.mean([r["history_mse"] for r in prs]),
                            "delta_hist": np.mean([r["delta_hist"] for r in prs]),
                        })
                    delta = np.asarray([r["delta_hist"] for r in protein_rows])
                    key = f"h={horizon}/H={H}"
                    result["by_representation"][name][key] = {
                        "mean_current_mse": float(np.mean([r["current_mse"] for r in protein_rows])),
                        "mean_history_mse": float(np.mean([r["history_mse"] for r in protein_rows])),
                        "mean_delta_hist": float(delta.mean()),
                        "delta_hist_std": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                        "delta_hist_ci_95": _bootstrap_mean(delta),
                        "n_trajectories": len(rows),
                        "n_proteins": len(protein_rows),
                    }
                    print(key, result["by_representation"][name][key])
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__": main()
