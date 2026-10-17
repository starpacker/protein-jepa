"""THE evaluation entrypoint.

Loops variants × (manifest/split) × horizons × temperature, computes anchor-aligned
CA RMSD, and writes results.json / results.md / per_protein.jsonl / meta.json.

Usage:
  python -m evaluation.run_eval --config evaluation/configs/mdcath_holdout_protein.yaml
  python -m evaluation.run_eval --config evaluation/configs/smoke.yaml --max-proteins 2

Design: this file only orchestrates.  Geometry lives in geometry.py, window
construction in data.py, model loading in loaders.py, variant logic in
variants.py, aggregation in metrics.py.  Nothing model-specific is re-implemented.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.data import EvalProtocol, iter_windows
from evaluation.geometry import aligned_rmsd, ca_from_backbone
from evaluation.loaders import build_bundle, sha256_file
from evaluation.metrics import PerProteinRow, aggregate, paired_bootstrap_ci
from evaluation import variants as V


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).decode().strip()
    except Exception:
        return "unknown"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _encoder_key(variant: str, models: dict) -> str:
    """Which encoder checkpoint a variant uses (for encode-once sharing).

    h_only  -> stage_a_ckpt
    h_ft / h_plus_s -> the fine-tuned encoder inside b1a_ckpt
    """
    if variant == "h_only":
        return "sa:" + str(models.get("stage_a_ckpt"))
    return "b1a:" + str(models.get("b1a_ckpt"))


def run(cfg: dict, cli_overrides: dict) -> dict:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizons = tuple(cfg["horizons_frames"])
    K = cfg.get("context_frames", 8)
    variants = cfg["variants"]
    n_reps = cfg.get("n_rollout_reps", 1)
    seed = cfg.get("seed", 0)
    max_proteins = cli_overrides.get("max_proteins", cfg.get("max_proteins"))
    oracle_max_horizon = cfg.get("oracle_max_horizon", max(horizons))

    # ------------------------------------------------------------------ bundles
    bundles = {}
    for v in variants:
        if v == "persist":
            bundles[v] = None
            continue
        spec = cfg["models"]
        kwargs = dict(
            variant=v,
            stage_a_ckpt=spec.get("stage_a_ckpt"),
            stage_a_predictor_ckpt=spec.get("stage_a_predictor_ckpt"),
            b1a_ckpt=spec.get("b1a_ckpt"),
            b1b_ckpt=spec.get("b1b_ckpt"),
            device=dev,
            mock_encoder=cfg.get("mock_encoder", False),
        )
        bundles[v] = build_bundle(**kwargs)
        print(f"[bundle] {v}: decoder={bundles[v].decoder_kind} "
              f"b1a={bundles[v].provenance.get('b1a_decoder_version')} "
              f"enc_flags={bundles[v].provenance.get('encoder_flags')}", flush=True)

    # ------------------------------------------------------------------ windows
    proto = EvalProtocol(
        manifest=cfg["manifest"],
        horizons_frames=horizons,
        context_frames=K,
        context_stride=cfg.get("context_stride", 1),
        frame_stride=cfg.get("frame_stride", 1),
        max_proteins=max_proteins,
        temperatures=tuple(cfg["temperatures"]) if cfg.get("temperatures") else None,
        anchor_stride=cfg.get("anchor_stride", 1),
        skip_head_frames=cfg.get("skip_head_frames", 0),
        seed=seed,
    )

    rows: list[PerProteinRow] = []
    n_seen = 0
    t0 = time.time()

    # Distinct encoders: group variants by which encoder checkpoint they use so
    # we encode the context ONCE per distinct encoder per anchor.
    enc_groups: dict[str, list[str]] = {}
    for v in variants:
        if v == "persist":
            continue
        key = _encoder_key(v, cfg["models"])
        enc_groups.setdefault(key, []).append(v)

    def _ctx_tensors(clip):
        return {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v) for k, v in clip.items()}

    rep_variant = next((v for v in variants if v in ("h_ft", "h_only")), None)
    if rep_variant is None and "h_plus_s" in variants:
        rep_variant = "h_plus_s"

    # Anchor-major: all horizons of one anchor share the context encode and the
    # anchor frame.  Direct predictor calls remain per-horizon because tau is an
    # explicit predictor input; h_plus_s also remains per-horizon because its
    # rollout length is tau.
    # iter_windows yields horizons grouped per anchor, so we buffer by key.
    def _anchor_key(w):
        return (w.trajectory_id, w.anchor_frame)

    def _flush(anchor_windows):
        nonlocal n_seen
        w0 = anchor_windows[0]
        n_seen += len(anchor_windows)
        ctx = _ctx_tensors(w0.context)
        rmask = ctx["residue_mask"][:, -1]
        if int(rmask.sum()) < 3:
            return
        X_anchor = ca_from_backbone(ctx["coords"][:, -1])[0]
        R_anchor, t_anchor = V.anchor_frames(ctx)

        H_cache: dict[str, torch.Tensor] = {}
        for key, vs in enc_groups.items():
            b = bundles[vs[0]]
            if b is None:
                continue
            with torch.no_grad():
                H_cache[key] = V.encode_clip(b, ctx)

        for w in anchor_windows:
            X_true = ca_from_backbone(w.target["coords"].unsqueeze(0).to(dev))[0, 0]
            p_rmsd = aligned_rmsd(X_anchor, X_true, rmask[0])

            oracle_rmsd = None
            if rep_variant is not None and w.horizon <= oracle_max_horizon:
                b = bundles[rep_variant]
                with torch.no_grad():
                    tgt = _ctx_tensors(w.target)
                    H_true = V.encode_clip(b, tgt)[:, 0]
                    H_anchor = H_cache[_encoder_key(rep_variant, cfg["models"])][:, -1]
                    ca_oracle = V.decode(
                        b, H_true, H_anchor, ctx, R_anchor, t_anchor, rmask,
                    )[0]
                oracle_rmsd = aligned_rmsd(ca_oracle, X_true, rmask[0])

            for v in variants:
                if v == "persist":
                    ca_pred = X_anchor
                    msd = 0.0
                else:
                    b = bundles[v]
                    res = V.produce(v, b, ctx, H_cache[_encoder_key(v, cfg["models"])],
                                    w.horizon, horizon_ps=w.horizon_ps,
                                    seed=seed, n_rollout_reps=n_reps)
                    ca_pred = res["ca_pred"]
                    msd = res["meta"].get("msd")
                r = aligned_rmsd(ca_pred, X_true, rmask[0])
                rows.append(PerProteinRow(
                    protein_id=w.protein_id, trajectory_id=w.trajectory_id,
                    temperature_k=w.temperature_k, horizon=w.horizon, variant=v,
                    rmsd=r, persist_rmsd=p_rmsd, oracle_rmsd=oracle_rmsd,
                    msd=msd, n_residues=int(rmask.sum()),
                ))

    prev_key = None
    cur: list = []
    last_report = [0]
    last_rows = [0]
    for w in iter_windows(proto):
        k = _anchor_key(w)
        if k != prev_key:
            if cur:
                _flush(cur)
                if len(rows) - last_rows[0] >= 200:
                    last_rows[0] = len(rows)
                    print(f"  {len(rows)} rows, {n_seen} windows, {time.time()-t0:.0f}s", flush=True)
            cur = []
            prev_key = k
        cur.append(w)
    if cur:
        _flush(cur)

    agg = aggregate(rows)
    return {"config": cfg, "n_windows": n_seen, "aggregate": agg, "rows": rows,
            "meta": {
                "git_sha": _git_sha(), "device": str(dev), "timestamp": datetime.now().isoformat(),
                "manifest_sha256": sha256_file(cfg["manifest"]),
                "models": {v: (bundles[v].provenance if bundles[v] else {"variant": "persist"})
                           for v in variants},
                "decoder_protocol": {
                    "direct": "h_only and h_ft share the B1a oracle decoder",
                    "h_plus_s": "uses the B1b ResidualCADecoder; not decoder-controlled against direct variants",
                    "persist": "identity X_anchor",
                },
            }}


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "  --  "
    return f"{x:.{nd}f}"


def write_reports(out: dict, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    cfg = out["config"]
    variants = cfg["variants"]
    horizons = cfg["horizons_frames"]

    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump({"n_windows": out["n_windows"],
                   "aggregate": {str(T): {str(h): blk for h, blk in byh.items()}
                                 for T, byh in out["aggregate"].items()},
                   "meta": out["meta"]}, f, indent=2)

    with open(os.path.join(outdir, "per_protein.jsonl"), "w") as f:
        for r in out["rows"]:
            f.write(json.dumps(asdict(r)) + "\n")

    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(out["meta"], f, indent=2)

    lines = [f"# Evaluation — {cfg.get('name', 'run')}", "",
             f"- n_windows: {out['n_windows']}",
             f"- manifest: `{cfg['manifest']}` (sha256 `{out['meta']['manifest_sha256'][:16]}`)",
             f"- git: `{out['meta']['git_sha']}`  device: `{out['meta']['device']}`",
             f"- horizons (frames/ns): {list(horizons)}  context K={cfg.get('context_frames',8)}",
             f"- variants: {variants}", ""]
    header = "| horizon | variant | RMSD median | mean±SEM | persist | oracle | Δ vs persist | skill | MSD | sig? |"
    sep = "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|"
    for T in sorted(out["aggregate"].keys(), key=lambda x: (x != x, x)):
        lines.append(f"\n## Temperature {T} K\n")
        lines.append(header)
        lines.append(sep)
        for h in sorted(out["aggregate"][T].keys()):
            blk = out["aggregate"][T][h]
            for v in variants:
                if v not in blk:
                    continue
                b = blk[v]
                lines.append(
                    f"| {h} | {v} | {_fmt(b['rmsd_median'])} | "
                    f"{_fmt(b['rmsd_mean'])}±{_fmt(b['rmsd_sem'])} | "
                    f"{_fmt(b['persist_median'])} | {_fmt(b.get('oracle_median'))} | "
                    f"{_fmt(b.get('delta_vs_persist'))} | {_fmt(b.get('skill'))} | "
                    f"{_fmt(b.get('msd_median'))} | "
                    f"{'YES' if b.get('delta_significant') else 'no'} |")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(outdir, "results.md"), "w") as f:
        f.write(md)
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--max-proteins", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--horizons", type=int, nargs="*", default=None)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--anchor-stride", type=int, default=None)
    ap.add_argument("--manifest-override", default=None)
    ap.add_argument("--oracle-max-horizon", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.name:
        cfg["name"] = args.name
    if args.manifest_override:
        cfg["manifest"] = args.manifest_override
    if args.oracle_max_horizon is not None:
        cfg["oracle_max_horizon"] = args.oracle_max_horizon
    if args.horizons:
        cfg["horizons_frames"] = args.horizons
    if args.variants:
        cfg["variants"] = args.variants
    if args.anchor_stride:
        cfg["anchor_stride"] = args.anchor_stride
    overrides = {"max_proteins": args.max_proteins}
    out = run(cfg, overrides)

    run_name = cfg.get("name") or os.path.basename(args.config).replace(".yaml", "")
    outdir = args.outdir or os.path.join("artifacts", "eval", run_name)
    md = write_reports(out, outdir)
    print("\n" + md)
    print(f"\nWrote reports to {outdir}/")


if __name__ == "__main__":
    main()
