"""One sweep, five methods, identical windows.

Every method is scored on exactly the same ``plan_windows`` set (same anchors,
same true futures, same residue masks, same persistence), and every metric comes
from ``evaluation.*`` — no method gets its own metric implementation.

The sweep is anchored on a single long walk (``--horizons`` max = 64 ns), so one
run produces

  * the multi-scale ensemble table  (terminate the walk at 1 / 4 / 16 / 64 ns),
  * the position table             (common-anchor + aligned CA RMSD),
  * the order-aware rollout table  (path length, ACF, walk RMSF, ...),

from the *same* trajectories, which is the only way those three views stay
consistent with each other.

Output is JSONL, one record per window, flushed as we go: a long sweep that
dies at window 900 keeps the first 899.

Usage
-----
    python -m scripts.five_method.run_five_method --suite unseen --methods lasp model1
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.five_method import adapters_external as EXT   # noqa: E402
from scripts.five_method import adapters_local as LOCAL    # noqa: E402
from scripts.five_method.common import (                   # noqa: E402
    METHODS, POSITION_METRICS, PRETTY, WALK_METRICS, dump, fork_seed,
    make_protocol, plan_windows, score_ensemble, score_position, score_walk,
    set_linalg_backend, window_seed,
)

# Bump whenever the scoring protocol changes so old records can never be
# silently mixed with new ones.
#   p1: endpoint drawn per horizon (published protocol) + one ordered walk for
#       the order-aware metrics.
PROTOCOL_VERSION = 1

# --------------------------------------------------------------------------- #
# experiment definitions
# --------------------------------------------------------------------------- #
# ``suite`` -> (manifest, temperature, description).  The three temperature
# cells the task asks for are all here, plus the seen-protein/seen-temperature
# reference row the other three are read against.
# Every cell the report needs is a (protein split x temperature) point, so the
# suite name *is* the cell and the table generator just cross-tabulates them.
# mdCATH temperatures: 320/348/379/413 K are seen; 450 K is held out.
MDCATH_TEMPS = (320.0, 348.0, 379.0, 413.0, 450.0)
SEEN_TEMPS = (320.0, 348.0, 379.0, 413.0)

SUITES: dict[str, dict] = {}
for _T in MDCATH_TEMPS:
    _t = int(_T)
    # unseen protein (val split) at this temperature
    SUITES[f"val_t{_t}"] = dict(suite=f"val_t{_t}",
                                manifest="artifacts/mdcath_manifest_1000dom/val.jsonl",
                                temperature=_T, protein_set="unseen",
                                temperature_seen=_T in SEEN_TEMPS,
                                split="val")
    # seen protein (train split) at this temperature
    SUITES[f"train_t{_t}"] = dict(suite=f"train_t{_t}",
                                  manifest="artifacts/mdcath_manifest_1000dom/train.jsonl",
                                  temperature=_T, protein_set="seen",
                                  temperature_seen=_T in SEEN_TEMPS,
                                  split="train")

# Rollout-focused cell: the ordered-walk table lives here.  Fewer windows and
# fewer proteins because bioemu2's ordered rollout costs one denoiser pass per
# nanosecond of horizon.
SUITES["rollout"] = dict(suite="rollout",
                         manifest="artifacts/mdcath_manifest_1000dom/val.jsonl",
                         temperature=348.0, protein_set="unseen",
                         temperature_seen=True, split="val")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--suite", nargs="+", required=True, choices=sorted(SUITES),
                   help="one or more cells; several cells run in one process so "
                        "the model and the per-sequence bioemu2 MSA/embedding "
                        "caches are reused")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--device", default="cuda")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 16, 64],
                   help="horizons (ns == frames in mdCATH) read off ONE walk")
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--max-proteins", type=int, default=20)
    p.add_argument("--max-anchors-per-trajectory", type=int, default=2)
    # mdCATH has 5 replicas per (protein, temperature); one is enough, otherwise
    # a "12 protein" cell silently becomes 60 trajectories.
    p.add_argument("--max-trajectories-per-protein", type=int, default=1)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--shard", type=int, default=0,
                   help="shard index; with --shards, this run scores only "
                        "proteins whose rank in the sorted protein list is "
                        "shard, shard+shards, ... so N parallel workers cover "
                        "disjoint windows instead of redoing the first few")
    p.add_argument("--shards", type=int, default=1,
                   help="number of disjoint shards (default 1 = no sharding)")
    p.add_argument("--history-frames", type=int, default=8)
    p.add_argument("--max-residues", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature-mode", default="correct",
                   choices=["correct", "wrong", "none"])
    p.add_argument("--wrong-temp", type=float, default=320.0)
    p.add_argument("--out", default=None)
    p.add_argument("--torch-threads", type=int, default=4,
                   help="CPU threads for the metric code (keep small when "
                        "running methods in parallel)")
    p.add_argument("--tag", default=None,
                   help="suffix for the output file (e.g. bioemu2_walk)")
    p.add_argument("--manifests-root", default=None,
                   help="override the manifest root (e.g. the T=450 holdout split)")

    # checkpoints
    p.add_argument("--stage-a-ckpt", default="artifacts/stage_a_disp_relpos/checkpoint_latest.pt")
    p.add_argument("--b1a-ckpt",
                   default="artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt")
    p.add_argument("--ensemble-ckpt",
                   default="artifacts/stage_ensemble_1000dom/checkpoint_best.pt")
    p.add_argument("--ensemble-temp-ckpt",
                   default="artifacts/stage_ensemble_1000dom_temp_cond/checkpoint_best.pt")
    p.add_argument("--model2-temp-ckpt",
                   default="artifacts/fromnone/stage_cb1b_temp_v2/checkpoint_latest.pt")
    p.add_argument("--model2-ckpt",
                   default="artifacts/fromnone/stage_cb1b_v4g/checkpoint_latest.pt")
    p.add_argument("--lasp-ckpt",
                   default="artifacts/plan_a_unfrozen_dec_1000dom/checkpoint_best.pt")
    p.add_argument("--mdgen-ckpt", default="/liuyutian/mdgen_run/hf_mdgen/atlas.ckpt")
    p.add_argument("--mdgen-sampler", default="euler")
    p.add_argument("--mdgen-steps", type=int, default=50)
    p.add_argument("--bioemu2-ckpt",
                   default="baselines/bioemu2/checkpoints/bioemu2-preview/checkpoint.ckpt")
    p.add_argument("--bioemu2-name", default="bioemu2-preview")
    p.add_argument("--bioemu2-cache", default="/octopus-runtime/cache/huggingface/hub")
    p.add_argument("--bioemu2-msa-cache",
                   default="baselines/bioemu2/msa_cache")
    p.add_argument("--bioemu2-sampler", default="edm")
    p.add_argument("--bioemu2-steps", type=int, default=128)
    p.add_argument("--bioemu2-walk", action="store_true",
                   help="also roll bioemu2 out into an ordered path (slow)")
    p.add_argument("--temp-split", action="store_true",
                   help="use the T=450-holdout checkpoints for Model 1 / Model 2, so "
                        "T=450 is a genuinely unseen temperature for them")
    p.add_argument("--no-walk", action="store_true",
                   help="skip ordered rollouts entirely (ensemble table only)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# per-method plumbing
# --------------------------------------------------------------------------- #
def _horizon_view(win, h):
    """Shallow copy of a window re-targeted at horizon ``h``."""
    w = dict(win)
    w["horizon"] = int(h)
    w["horizon_ns"] = float(h) * (float(win["horizon_ns"]) / max(int(win["horizon"]), 1))
    return w


def _ckpt_for(method, args):
    return {"model1": args.ensemble_temp_ckpt if args.temp_split else args.ensemble_ckpt,
            "model2": args.model2_temp_ckpt if args.temp_split else args.model2_ckpt,
            "lasp": args.lasp_ckpt, "mdgen": args.mdgen_ckpt,
            "bioemu2": args.bioemu2_ckpt}[method]


def build(method, args, device):
    if method == "model1":
        args.ensemble_ckpt = (args.ensemble_temp_ckpt if args.temp_split
                              else args.ensemble_ckpt)
        return LOCAL.build_model1(args, device)
    if method == "model2":
        args.model2_ckpt = args.model2_temp_ckpt if args.temp_split else args.model2_ckpt
        return LOCAL.build_model2(args, device)
    if method == "lasp":
        return LOCAL.build_lasp(args, device)
    if method == "mdgen":
        args.mdgen_frames = max(args.horizons) + 1
        return EXT.build_mdgen(args, device)
    if method == "bioemu2":
        return EXT.build_bioemu2(args, device)
    raise KeyError(method)


def sample(method, bundle, win, args, device, force_no_walk=False):
    """Return (samples [N,L,3], walk [N,T,L,3] | None) for a window."""
    kw = dict(walk=not (args.no_walk or force_no_walk))
    if method == "model1":
        return LOCAL.sample_model1(bundle[0], bundle[1], win, args.n_samples, device,
                                   temperature_mode=args.temperature_mode,
                                   wrong_temp=args.wrong_temp, **kw)
    if method == "model2":
        return LOCAL.sample_model2(bundle[0], bundle[1], win, args.n_samples, device,
                                   temperature_mode=args.temperature_mode,
                                   wrong_temp=args.wrong_temp, **kw)
    if method == "lasp":
        return LOCAL.sample_lasp(bundle[0], bundle[1], win, args.n_samples, device,
                                 temperature_mode=args.temperature_mode,
                                 wrong_temp=args.wrong_temp, **kw)
    if method == "mdgen":
        return EXT.sample_mdgen(bundle, win, args.n_samples, device, **kw)
    if method == "bioemu2":
        kw["walk"] = kw["walk"] and args.bioemu2_walk
        return EXT.sample_bioemu2(bundle, win, args.n_samples, device, **kw)
    raise KeyError(method)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score_window(method, out, win, args, device, t_anchor, t_true_path):
    """Per-horizon metric blocks for one window.

    Two *different* protocols, deliberately:

    ``ens`` / ``pos``  — the endpoint at horizon h is drawn by conditioning the
        method on horizon h directly (one call per h).  This is the published
        protocol (``scripts/eval_lasp_final.py``, ``run_bioemu2_zeroshot.py``),
        so these numbers are directly comparable to the numbers already in
        ``EVALUATION_METHODS_AND_COMMANDS_20261021.md``.  Reading frame h off a 64-step
        rollout instead would silently change what "1 ns prediction" means:
        for Model 1 the frame at step 1 of a 64-step autoregressive walk has
        already been polluted by 63 later feedback steps, and its Kabsch error
        grows from ~4 A to ~145 A.

    ``walk``           — the order-aware metrics are computed on ONE ordered
        rollout of the full horizon, sliced to h.  Those metrics are about path
        shape, so they must come from a single self-consistent trajectory.

    A method that cannot produce an ordered path (bioemu2 without
    ``--bioemu2-walk``) gets ``walk = None`` and the walk block is omitted.
    """
    mask = torch.ones(win["n_residues"], dtype=torch.bool, device=device)
    walk = out.get("walk")
    blocks = {}

    for h in args.horizons:
        h = int(h)
        if h > t_true_path.shape[0] - 1:
            continue
        true_ens = t_true_path[:h + 1]      # observed frames anchor..horizon
        true_h = t_true_path[h]
        w_h = _horizon_view(win, h)

        pred = out["samples"] if h == int(win["horizon"]) else None
        if pred is None:
            with fork_seed(window_seed(w_h, salt=method)):
                res = sample(method, args._bundle, w_h, args, device,
                             force_no_walk=True)
            pred = res["samples"]

        block = {"n_samples": int(pred.shape[0])}
        block["ens"] = score_ensemble(pred, true_ens, None)
        block["pos"] = score_position(pred, t_anchor, true_h, mask)
        if walk is not None and walk.shape[1] >= h + 1:
            # ``walk_summary`` wants FUTURE frames only: it prepends the anchor
            # itself, so walk[0] (= the anchor) must be dropped here.
            wsub = walk[:, 1:h + 1]
            block["walk"] = score_walk(wsub, t_true_path[1:h + 1], t_anchor, mask)
            block["walk_m"] = int(wsub.shape[0])
        blocks[str(h)] = block
    return blocks


def main(argv=None):
    args = parse_args(argv)
    if "bioemu2" in args.methods:
        # PYTHONHASHSEED must be set before the interpreter starts; bioemu2 uses
        # hash()/set iteration in its bond bookkeeping, so without this reruns
        # are not bit-identical.  Re-exec *this* module (not the baseline's).
        if os.environ.get("PYTHONHASHSEED") != "0":
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = "0"
            env["_FIVE_REEXEC"] = "1"
            print("[five] re-exec with PYTHONHASHSEED=0", flush=True)
            os.execve(sys.executable, [sys.executable, "-m",
                                       "scripts.five_method.run_five_method"] + sys.argv[1:],
                      env)
        # reproducibility knobs, identical to run_bioemu2_zeroshot
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    # Metric code is CPU-heavy (per-frame Kabsch loops, eigh on 3L x 3L), and we
    # run five methods in parallel on one host.  Without this cap every process
    # spawns a full thread pool and they fight each other: the same window went
    # from 10 s to 105 s purely from oversubscription.
    torch.set_num_threads(int(args.torch_threads))
    torch.set_num_interop_threads(1)

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    specs = [SUITES[s] for s in args.suite]

    bundles = {}
    for m in args.methods:
        t0 = time.time()
        set_linalg_backend(m)     # MDGen needs MAGMA, everything else cuSOLVER
        bundles[m] = build(m, args, device)
        print(f"[five] built {m} in {time.time()-t0:.1f}s", flush=True)

    t_start = time.time()
    for spec in specs:
        run_cell(spec, args, device, bundles)
    print(f"[five] all cells done in {time.time()-t_start:.0f}s", flush=True)


def run_cell(spec, args, device, bundles):
    """Score every requested method on one (protein-split x temperature) cell."""
    suite = spec["suite"]
    manifest = spec["manifest"]
    if args.manifests_root:
        manifest = str(Path(args.manifests_root) / Path(manifest).name)

    proto = make_protocol(args, horizons_frames=(max(args.horizons),),
                          manifest=manifest, temperature=spec["temperature"])
    proto.max_proteins = args.max_proteins
    proto.max_anchors_per_trajectory = args.max_anchors_per_trajectory
    proto.max_trajectories_per_protein = args.max_trajectories_per_protein
    wins = plan_windows(proto, args.max_windows)
    if args.shards > 1:
        pids = sorted({w["protein_id"] for w in wins})
        keep = {pid for i, pid in enumerate(pids) if i % args.shards == args.shard}
        wins = [w for w in wins if w["protein_id"] in keep]
    print(f"[five] suite={suite} manifest={manifest} "
          f"T={spec['temperature']} windows={len(wins)} "
          f"proteins={len({w['protein_id'] for w in wins})} "
          f"horizons={args.horizons}"
          + (f" shard={args.shard}/{args.shards}" if args.shards > 1 else ""),
          flush=True)

    # Protocol version is part of the file name: the endpoint protocol and the
    # walk protocol are both baked into each record, so mixing records produced
    # by two different protocols in one table would be silently wrong.
    name = f"{suite}_{args.temperature_mode}_p{PROTOCOL_VERSION}"
    if args.tag:
        name = f"{name}_{args.tag}"
    out_path = args.out or f"results/five_method/{name}.jsonl"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if Path(out_path).exists():
        for line in open(out_path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("protocol") != PROTOCOL_VERSION:
                continue
            if r.get("method") in args.methods:
                done.add((r["method"], r["protein_id"], r["trajectory_id"],
                          r["anchor_frame"], r["temperature_mode"]))
        print(f"[five] resuming: {len(done)} (method,window) records already scored",
              flush=True)

    sink = open(out_path, "a")
    t_cell = time.time()
    try:
        for i, win in enumerate(wins):
            anchor_ca = win["context"]["coords"][-1][:, 1, :][win["residue_mask"]]
            anchor_ca = anchor_ca.to(device).float()
            true_path = win["true_ca_path"].to(device).float()
            for method in args.methods:
                key = (method, win["protein_id"], win["trajectory_id"],
                       win["anchor_frame"], args.temperature_mode)
                if key in done:
                    continue
                args._bundle = bundles[method]
                t0 = time.time()
                try:
                    set_linalg_backend(method)
                    with fork_seed(window_seed(win, salt=method)):
                        out = sample(method, bundles[method], win, args, device)
                    set_linalg_backend("metrics")   # cuSOLVER for all metrics
                    blocks = score_window(method, out, win, args, device,
                                          anchor_ca, true_path)
                except Exception as exc:                    # noqa: BLE001
                    rec = {"suite": suite, "method": method,
                           "protein_id": win["protein_id"],
                           "trajectory_id": win["trajectory_id"],
                           "temperature_k": win["temperature_k"],
                           "anchor_frame": win["anchor_frame"],
                           "temperature_mode": args.temperature_mode,
                           "protocol": PROTOCOL_VERSION,
                           "error": f"{type(exc).__name__}: {exc}"}
                    sink.write(json.dumps(rec) + "\n")
                    sink.flush()
                    print(f"[five] {method} {win['protein_id']}@{win['anchor_frame']} "
                          f"FAILED {type(exc).__name__}: {exc}", flush=True)
                    continue
                rec = {"suite": suite, "method": method, "pretty": PRETTY[method],
                       "protein_id": win["protein_id"],
                       "trajectory_id": win["trajectory_id"],
                       "temperature_k": float(win["temperature_k"]),
                       "protein_set": spec["protein_set"],
                       "temperature_seen": spec["temperature_seen"],
                       "anchor_frame": int(win["anchor_frame"]),
                       "n_residues": int(win["n_residues"]),
                       "temperature_mode": args.temperature_mode,
                       "protocol": PROTOCOL_VERSION,
                       "n_samples": int(args.n_samples),
                       "has_walk": bool(out.get("walk") is not None),
                       "seconds": round(time.time() - t0, 2),
                       "horizons": blocks,
                       "temp_split": bool(args.temp_split),
                       "ckpt": _ckpt_for(method, args),
                       "config": {"horizons": args.horizons,
                                  "max_proteins": args.max_proteins,
                                  "max_anchors_per_trajectory": args.max_anchors_per_trajectory,
                                  "max_trajectories_per_protein": args.max_trajectories_per_protein,
                                  "history_frames": args.history_frames}}
                sink.write(json.dumps(rec) + "\n")
                sink.flush()
                ka = blocks[str(args.horizons[-1])]["pos"]["kabsch_ca_rmsd"] \
                    if str(args.horizons[-1]) in blocks else float("nan")
                print(f"[five] ({i+1}/{len(wins)}) {method:8s} "
                      f"{win['protein_id']:8s} T{int(win['temperature_k'])} "
                      f"{time.time()-t0:5.1f}s  kabsch@max={ka:.2f} "
                      f"elapsed={time.time()-t_cell:.0f}s", flush=True)
    finally:
        sink.close()
    print(f"[five] wrote {out_path} in {time.time()-t_cell:.0f}s", flush=True)


if __name__ == "__main__":
    main()
