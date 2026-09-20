#!/usr/bin/env python3
"""
R1: History Gap — Representation Closure

Measures whether future states can be predicted as accurately from a
history of past latent states as from the current latent state alone.

Protocol:
  1. Extract latent endpoints (z_t, z_{t+tau}) and per-frame states.
  2. Train LinearProbe: z_t -> z_{t+tau}
  3. Train HistoryProbe: [z_{t-H+1}..z_t] -> z_{t+tau}
  4. Per-protein Delta_hist = MSE_current - MSE_history
  5. Bootstrap 95% CI across proteins
  6. Gate: CI.lower > 0.02
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from artifacts.reviewer_experiments._shared.model_interface import (
    DynamicsModel, NullModel,
)
from artifacts.reviewer_experiments._shared.lspwm_adapter import LSPWMDynamicsModel
from artifacts.reviewer_experiments._shared.metrics import paired_bootstrap_ci
from artifacts.reviewer_experiments._shared.result_schema import (
    ExperimentResult, ProvenanceBlock,
)
from artifacts.reviewer_experiments._shared.baseline_representations import EXTRACTOR_REGISTRY

# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    """z_t -> z_{t+tau}"""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Linear(latent_dim, latent_dim)
        nn.init.xavier_uniform_(self.net.weight, gain=0.01)
        nn.init.zeros_(self.net.bias)
    def forward(self, x): return self.net(x)


class HistoryProbe(nn.Module):
    """[z_{t-H+1}..z_t] -> z_{t+tau}. MLP: flatten -> 256 -> ReLU -> 256 -> ReLU -> D"""
    def __init__(self, latent_dim: int, history_len: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * history_len, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, latent_dim),
        )
    def forward(self, x):
        if x.dim() == 3: x = x.reshape(x.shape[0], -1)
        return self.net(x)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_probe(probe, Xtr, Ytr, Xva, Yva, lr=0.001, max_epochs=200, patience=20):
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    best_loss, best_state, no_improve = float("inf"), None, 0
    for _ in range(max_epochs):
        probe.train(); opt.zero_grad()
        F.mse_loss(probe(Xtr), Ytr).backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        opt.step()
        probe.eval()
        with torch.no_grad(): vl = F.mse_loss(probe(Xva), Yva).item()
        if vl < best_loss - 1e-7:
            best_loss = vl
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience: break
    if best_state is not None: probe.load_state_dict(best_state)
    return probe, best_loss


@torch.no_grad()
def eval_probe(probe, data, src, tgt):
    """Per-protein MSE. Returns {pid: loss}."""
    probe.eval(); losses = {}
    for pid, d in data.items():
        X, y = d.get(src), d.get(tgt)
        if X is None or y is None or len(X) < 5: continue
        losses[pid] = F.mse_loss(probe(torch.from_numpy(X).float()),
                                 torch.from_numpy(y).float()).item()
    return losses

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def hist_inputs(data, H):
    """dict[pid, ndarray (N, H*D)] from frame_states."""
    out = {}
    for pid, d in data.items():
        fs = d.get("frame_states")
        if fs is None or fs.shape[1] < H: continue
        out[pid] = fs[:, -H:, :].reshape(len(fs), -1)
    return out


def make_representation_data(data, name, latent_dim=16):
    """Project the extracted frame/endpoints into a baseline representation.

    The model adapter currently exposes latent features rather than raw CA
    coordinates, so this also gives a reproducible fallback for existing
    latent dumps.  A future coordinate-bearing adapter can pass its features
    through the same extractor interface.
    """
    if name == "lspwm":
        return data, next(iter(data.values()))["context_endpoint"].shape[-1]
    if name not in EXTRACTOR_REGISTRY:
        raise ValueError(f"Unknown representation {name}")
    features = {pid: d["frame_states"] for pid, d in data.items() if "frame_states" in d}
    extractor = EXTRACTOR_REGISTRY[name](n_components=latent_dim)
    extractor.fit(features)
    out = {}
    for pid, d in data.items():
        fs = np.asarray(d["frame_states"])
        shape = fs.shape
        projected = extractor.transform({pid: fs}).reshape(shape[0], shape[1], -1)
        # Endpoints are projected with the same fitted transform.
        current = extractor.transform({pid: d["context_endpoint"]})
        future = extractor.transform({pid: d["future_endpoint"]})
        out[pid] = {**d, "frame_states": projected,
                    "context_endpoint": current, "future_endpoint": future}
    return out, extractor.latent_dim()


def run_history_gap(data, horizons, H_list, latent_dim,
                    probe_lr=0.001, probe_epochs=200, probe_patience=20,
                    verbose=True):
    all_res, rep_gate = {}, False
    gate_details = {}
    pids = sorted(data.keys())
    if verbose:
        print(f"  Proteins: {len(pids)}  Horizons: {horizons}  History lengths: {H_list}")

    for hf in horizons:
        if verbose: print(f"\n  -- Horizon {hf} frames --")
        Zs, Ys = [], []
        for pid in pids:
            z, y = data[pid].get("context_endpoint"), data[pid].get("future_endpoint")
            if z is None or y is None or len(z) < 10: continue
            Zs.append(z); Ys.append(y)
        if len(Zs) < 2:
            if verbose: print("    SKIP (too few data)"); continue

        Zp, Yp = np.concatenate(Zs, 0), np.concatenate(Ys, 0)
        n = len(Zp)
        rng = np.random.default_rng(42)
        perm = rng.permutation(n)
        tr, va = perm[:int(n*0.8)], perm[int(n*0.8):]
        Ztr = torch.from_numpy(Zp[tr]).float()
        Ytr = torch.from_numpy(Yp[tr]).float()
        Zva = torch.from_numpy(Zp[va]).float()
        Yva = torch.from_numpy(Yp[va]).float()

        curr = LinearProbe(latent_dim)
        _, curr_vl = train_probe(curr, Ztr, Ytr, Zva, Yva, probe_lr, probe_epochs, probe_patience)
        curr_losses = eval_probe(curr, data, "context_endpoint", "future_endpoint")
        mc = float(np.mean(list(curr_losses.values())))
        if verbose: print(f"    Current probe: val={curr_vl:.6f}  mean_per_protein={mc:.6f}")

        by_H = {}
        for H in H_list:
            hi = hist_inputs(data, H)
            if len(hi) < 2:
                if verbose: print(f"    H={H}: SKIP (<2 proteins)"); continue
            all_h = np.concatenate([hi[pid] for pid in pids if pid in hi], 0)
            Htr = torch.from_numpy(all_h[tr]).float()
            Hva = torch.from_numpy(all_h[va]).float()

            hist = HistoryProbe(latent_dim, H)
            _, hist_vl = train_probe(hist, Htr, Ytr, Hva, Yva, probe_lr, probe_epochs, probe_patience)

            hist_losses = {}
            for pid in pids:
                if pid not in hi: continue
                hist_losses[pid] = F.mse_loss(
                    hist(torch.from_numpy(hi[pid]).float()),
                    torch.from_numpy(data[pid]["future_endpoint"]).float()
                ).item()
            mh = float(np.mean(list(hist_losses.values())))

            dh = {pid: curr_losses[pid] - hist_losses.get(pid, 0.)
                  for pid in curr_losses if pid in hist_losses}
            dv = np.array(list(dh.values()))
            if len(dv) < 3:
                if verbose: print(f"    H={H}: SKIP (<3 proteins)"); continue

            ci = paired_bootstrap_ci(dh, n_resamples=2000, seed=42)
            lo, hi = ci["lower"], ci["upper"]
            md = float(np.mean(dv))
            pos = lo > 0.
            by_H[str(H)] = {
                "mean_current_loss": mc, "mean_history_loss": mh,
                "mean_delta_hist": md,
                "delta_hist_ci_95": [lo, hi], "delta_hist_positive": pos,
                "n_proteins": len(dh), "history_val_loss": hist_vl,
            }
            if verbose:
                print(f"    H={H}: Delta={md:.6f} [{lo:.6f}, {hi:.6f}]  pos={pos}  (n={len(dh)})")
            if lo > 0.02:
                rep_gate = True; gate_details[f"h={hf}_H={H}"] = "PASS"
            else:
                gate_details[f"h={hf}_H={H}"] = "FAIL"

        all_res[f"h={hf}"] = {
            "horizon_frames": hf, "by_history_length": by_H,
            "n_proteins_current": len(curr_losses), "current_val_loss": curr_vl,
        }
    return all_res, rep_gate, gate_details


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="R1: History Gap")
    p.add_argument("--model", choices=["lspwm","null"], default="lspwm")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--config", default="")
    p.add_argument("--manifest", default="/liuyutian/protein_jepa_frame/manifests/tmjepa_v3.validation.jsonl")
    p.add_argument("--output-dir", default="/liuyutian/protein_jepa_frame/artifacts/reviewer_experiments/representation_closure")
    p.add_argument("--horizons", default="1,2,4,8,16,32,64")
    p.add_argument("--history-lengths", default="1,2,4,8")
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-batches", type=int, default=100)
    p.add_argument("--probe-lr", type=float, default=0.001)
    p.add_argument("--probe-epochs", type=int, default=200)
    p.add_argument("--probe-patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=6303)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--representations", nargs="+", default=["lspwm"],
                   help="Representations separated by spaces or commas")
    args = p.parse_args()
    horizons = [int(h) for h in args.horizons.split(",")]
    H_list   = [int(h) for h in args.history_lengths.split(",")]

    print("="*60)
    print("R1: History Gap -- Representation Closure")
    print("="*60)

    if args.dummy or not args.checkpoint:
        print("[Dummy mode]")
        model = NullModel()
        data = model.extract_latents()
        latent_dim = model.latent_dim
    elif args.model == "lspwm":
        print(f"[LSP-WM] {args.checkpoint}")
        model = LSPWMDynamicsModel()
        model.load(checkpoint_path=args.checkpoint, config_path=args.config,
                   manifest=args.manifest, device=args.device,
                   batch_size=args.batch_size, max_batches=args.max_batches, seed=args.seed)
        data = model.extract_latents(max_batches=args.max_batches)
        latent_dim = model.latent_dim
    else:
        raise ValueError(f"Unknown model: {args.model}")

    representations = [r for item in args.representations for r in item.split(",") if r]
    print(f"  Proteins: {len(data)}  Latent dim: {latent_dim}  Representations: {representations}")
    if data:
        k = next(iter(data))
        for kk, vv in data[k].items(): print(f"    {kk}: {vv.shape}")

    all_res, rep_gate, gate_details = {}, False, {}
    for representation in representations:
        print(f"\n  === Representation: {representation} ===")
        rep_data, rep_dim = make_representation_data(data, representation, min(16, latent_dim))
        rep_res, rep_pass, rep_details = run_history_gap(
            rep_data, horizons, H_list, rep_dim,
            probe_lr=args.probe_lr, probe_epochs=args.probe_epochs,
            probe_patience=args.probe_patience, verbose=True)
        all_res[representation] = {"latent_dim": rep_dim, "by_horizon": rep_res}
        rep_gate = rep_gate or rep_pass
        gate_details.update({f"{representation}/{k}": v for k, v in rep_details.items()})

    print(f"\n  Representation gate: {rep_gate}")
    for k, v in gate_details.items():
        print(f"    {k}: {v}")

    prov = ProvenanceBlock(
        protocol="reviewer_experiment_v1", experiment_id="R1",
        config_path=args.config, checkpoint_path=args.checkpoint,
        manifest_path=args.manifest, random_seed=args.seed,
        device=args.device, batch_size=args.batch_size,
        max_batches=args.max_batches, sampler_mode="hierarchical",
        horizons=horizons, history_lengths=H_list,
        output_dir=args.output_dir,
        result_path=str(Path(args.output_dir)/"results.json"),
        train_only_fitted_objects={
            "probe_type": "linear_current + mlp_history",
            "probe_lr": args.probe_lr,
            "probe_max_epochs": args.probe_epochs,
            "probe_patience": args.probe_patience,
        })
    result = ExperimentResult(
        experiment_id="R1", provenance=prov,
        metrics={"by_horizon": all_res, "horizons": horizons,
                 "history_lengths": H_list, "n_proteins": len(data),
                 "latent_dim": latent_dim, "representations": representations},
        gates={"representation_gate": rep_gate, "gate_details": gate_details})
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    result.save(out/"results.json")
    print(f"\n  -> {out/'results.json'}")
    model.close()


if __name__ == "__main__":
    main()
