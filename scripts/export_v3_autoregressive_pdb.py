"""Export v3 (SOTA) autoregressive CA trajectories as multi-MODEL PDBs.

This is the CORRECT autoregressive rollout for the v3 model
(``HSEnsembleProducer`` from ``train_stage_b_hs``): encode the anchor ONCE,
run a single GLE ``kernel.rollout(n_steps)`` in H-space, and decode every one
of the ``n_steps + 1`` H frames back to CA.  There is NO coordinate-space
self-feedback (no decode-CA -> re-encode -> feed-back loop).

Why this matters: the old ``rollout_ours_hs`` in
``scripts/export_eval_trajectories_pdb.py`` decoded the terminal CA, re-encoded
it, and fed it back each frame -- a self-feedback loop that is NOT what the
SOTA metrics used, and it diverges (radius of gyration 13.9 -> 43.3 A over 16
frames for 1ka1A01).  This script mirrors the current-model's
``sample_ca_trajectory`` (single H-space rollout + per-frame decode) so v3 and
current are sampled on the same footing.

Only the val manifest is read; test.jsonl is never opened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluation.data import EvalProtocol, iter_windows
from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig
from protein_jepa_frame.training.train_stage_b_hs import HSEnsembleProducer
from protein_jepa_frame.training.train_stage_ensemble import (
    _extract_backbone_frames, load_frozen_stack,
)
from scripts.export_eval_trajectories_pdb import write_ca_pdb
from scripts.diag_lasp_ca_ceiling import _development_manifest, _window_seed


def load_v3(args, dev):
    """Load the v3 HSEnsembleProducer (encoder + decoder + StageB1b kernel)."""
    encoder, decoder, dH = load_frozen_stack(args.stage_a_ckpt, args.b1a_ckpt, dev)
    kck = torch.load(args.kernel_ckpt, map_location=dev, weights_only=False)
    k_cfg = kck.get("kernel_config") or {}
    k_args = kck.get("args", {})
    k_cfg["step_ps"] = k_args.get("step_ps", k_cfg.get("step_ps", 1000.0))
    k_cfg["rollout_grad_steps"] = None
    kernel = StageB1b(StageB1bConfig(**k_cfg)).to(dev)
    kernel.load_state_dict(kck["kernel"])
    kernel.eval()
    producer = HSEnsembleProducer(encoder, decoder, kernel).to(dev).eval()
    for p in producer.parameters():
        p.requires_grad_(False)
    return producer


@torch.no_grad()
def rollout_v3_hspace(producer, context, mask, aa, ri, ci, temperature_k,
                      n_steps, generator, dev):
    """Single H-space GLE rollout + per-frame decode. Returns [T, L_all, 3] CA.

    Mirrors ``sample_ca_trajectory`` (used for the current model): ONE
    ``kernel.rollout``, then decode every H frame.  No coordinate feedback.
    """
    B, K, L, _, _ = context.shape
    H_anchor = producer.encode_frame(context[:, -1], mask, aa, ri, ci)  # [B,L,dH]
    flat_H = H_anchor  # B == 1
    temp = torch.as_tensor(temperature_k, device=dev, dtype=torch.float32).reshape(B)
    traj = producer.kernel.rollout(
        flat_H, mask, temp, n_steps,
        seq_feature=flat_H, generator=generator, detach=False,
    )  # [B, n_steps+1, L, dH]

    R_anchor, t_anchor = _extract_backbone_frames(context[:, -1])  # [B,L,3,3],[B,L,3]
    ca_path = torch.empty(B, n_steps + 1, L, 3, device=dev)
    for t in range(n_steps + 1):
        Ht = traj[:, t]
        out = producer.decoder.decode_structure(Ht, aa, R_anchor, t_anchor, mask)
        ca_path[:, t] = out["backbone_coords"][:, :, 1, :]
    return ca_path[0]  # [n_steps+1, L_all, 3]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage-a-ckpt",
                   default="artifacts/stage_a_disp_relpos/checkpoint_latest.pt")
    p.add_argument("--b1a-ckpt",
                   default="artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt")
    p.add_argument("--kernel-ckpt",
                   default="artifacts/lasp_v3_velocity/checkpoint_best.pt")
    p.add_argument("--outdir", required=True)
    p.add_argument("--manifest", default="artifacts/mdcath_manifest_full/val.jsonl")
    p.add_argument("--n-windows", type=int, default=3)
    p.add_argument("--n-frames", type=int, default=64,
                   help="number of GLE rollout steps (= frames in the trajectory)")
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--max-residues", type=int, default=96)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(0)
    dev = torch.device(a.device)
    manifest = _development_manifest(a.manifest)
    producer = load_v3(a, dev)

    protocol = EvalProtocol(
        manifest=manifest, context_frames=8, horizons_frames=(a.n_frames,),
        max_residues=a.max_residues, max_anchors_per_trajectory=4,
        temperatures=(320., 348., 379., 413., 450.),
        seed=0, round_robin_proteins=True, round_robin_conditions=True,
    )

    n_windows_done = 0
    for window in iter_windows(protocol):
        if n_windows_done >= a.n_windows:
            break
        n_windows_done += 1
        context = window.context
        observed = context["coords"].to(dev)
        mask = context["residue_mask"].to(dev).all(0).bool()
        aa, ri, ci = [context[key].to(dev) for key in ("aatype", "residue_index", "chain_id")]
        idx = mask.cpu().numpy()
        seed = _window_seed(0, window)
        for s in range(a.n_samples):
            gen = torch.Generator(device=dev).manual_seed(seed + s)
            frames = rollout_v3_hspace(
                producer, observed[None], mask[None], aa[None], ri[None], ci[None],
                window.temperature_k, a.n_frames, gen, dev,
            )  # [n_frames+1, L_all, 3]
            frames = frames[:, idx].detach().cpu().double().numpy()
            path = Path(a.outdir) / (
                f"{window.protein_id}_{window.trajectory_id}_h{window.horizon}_s{s}.pdb")
            write_ca_pdb(path, frames, aa.cpu(), ri.cpu(), ci.cpu(),
                         origin=f"v3 HSEnsembleProducer single-H-space rollout "
                                f"n_frames={a.n_frames} temp={window.temperature_k}K")
            print(f"wrote {path.name}", flush=True)
    print(json.dumps(dict(n_windows=n_windows_done)), flush=True)


if __name__ == "__main__":
    main()
