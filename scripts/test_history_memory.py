"""Model-free test: does the observed history carry memory about the future
beyond the anchor frame H_0?

For every evaluation window we know the CA trace of the context and of the
true future.  Fit a ridge regression from history features to the future
displacement (relative to the anchor) and compare held-out errors for two
feature sets:

  (a) anchor-only : intercept, i.e. no memory at all (predict zero drift);
  (b) full history: the last K-1 anchor-relative displacements
                    (H_{-j} - H_0), j = 1..K-1.

Features are per-residue and anchor-relative, so a rigid-body translation is
removed and the comparison is about internal motion.  Proteins are split into
two folds so the test set contains proteins never seen in fitting; if (b) is
materially better than (a) on held-out proteins, the history holds genuine
predictable memory.  A linear map is a LOWER bound on what a nonlinear model
could extract, so a null result here is evidence, and a positive one is
conclusive in the direction the question asks.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.data import EvalProtocol, iter_windows


def ridge_fit(X, Y, lam):
    # X [N,D], Y [N,O]; return W [D,O], b [O]
    mu, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - mu, Y - ym
    A = Xc.T @ Xc + lam * np.eye(X.shape[1])
    W = np.linalg.solve(A, Xc.T @ Yc)
    return W, mu, ym


def ridge_pred(X, W, mu, ym):
    return (X - mu) @ W + ym


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="artifacts/mdcath_manifest_full/val.jsonl")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--max-windows", type=int, default=200)
    p.add_argument("--max-residues", type=int, default=96)
    p.add_argument("--context-frames", type=int, default=8)
    p.add_argument("--lam", type=float, nargs="+", default=[0.01, 0.1, 1., 10.])
    p.add_argument("--out", required=True)
    a = p.parse_args()

    proto = EvalProtocol(
        manifest=a.manifest, horizons_frames=tuple(a.horizons),
        context_frames=a.context_frames, frame_stride=1,
        max_residues=a.max_residues, max_anchors_per_trajectory=4,
        temperatures=(320., 348., 379., 413., 450.), seed=0,
        round_robin_proteins=True, round_robin_conditions=True)

    # Collect per-window residue-level rows, keyed by protein for the split.
    data = []   # dict(protein, horizon, X_hist, Y)
    n = 0
    for w in iter_windows(proto):
        if n >= a.max_windows:
            break
        mask = w.context["residue_mask"].all(0).bool().numpy()
        if mask.sum() < 4:
            continue
        ctx = w.context["coords"][:, :, 1].numpy()          # [K,L,3] CA
        fut = w.future_path["coords"][:, :, 1].numpy()       # [h,L,3]
        h0 = ctx[-1]                                         # [L,3]
        disp_hist = ctx[:-1][::-1] - h0[None]                # [K-1,L,3] newest first
        target = fut - h0[None]                              # [h,L,3]
        L = h0.shape[0]
        for l in range(L):
            if not mask[l]:
                continue
            Xh = disp_hist[:, l, :].reshape(-1)              # [3(K-1)]
            data.append(dict(protein=w.protein_id, horizon=int(w.horizon),
                             Xh=Xh.astype(np.float64),
                             Y=target[:, l, :].reshape(-1).astype(np.float64)))
        n += 1

    proteins = sorted({d["protein"] for d in data})
    rng = np.random.default_rng(20261024)
    rng.shuffle(proteins)
    fold = {p_: i % 2 for i, p_ in enumerate(proteins)}

    result = dict(scope="model-free history memory test on val split",
                  test_opened=False, n_windows=n, n_proteins=len(proteins),
                  n_rows=len(data), by_horizon={})
    for h in a.horizons:
        rows = [d for d in data if d["horizon"] == h]
        if not rows:
            continue
        Xh = np.stack([r["Xh"] for r in rows])
        Y = np.stack([r["Y"] for r in rows])
        foldid = np.array([fold[r["protein"]] for r in rows])
        tr, te = foldid == 0, foldid == 1
        best = {}
        for name, feat in (("anchor_only", np.zeros((len(rows), 1))),
                           ("history", Xh)):
            errs = []
            for lam in a.lam:
                W, mu, ym = ridge_fit(feat[tr], Y[tr], lam)
                pred = ridge_pred(feat[te], W, mu, ym)
                errs.append(((pred - Y[te]) ** 2).mean())
            best[name] = float(min(errs))
            best[name + "_lam"] = float(a.lam[int(np.argmin(errs))])
        base, hist = best["anchor_only"], best["history"]
        result["by_horizon"][h] = dict(
            mse_anchor_only=base, mse_history=hist,
            mse_reduction=float((base - hist) / base),
            lam_history=best["history_lam"],
            n_rows=int(len(rows)),
            n_test=int(te.sum()))
        print(f"h={h:3d}  anchor_only={base:.6f}  history={hist:.6f}  "
              f"reduction={100 * (base - hist) / base:6.2f}%  (n_test={int(te.sum())})")

    Path(a.out).write_text(json.dumps(result, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
