"""Adapters for the two zero-shot external methods: MDGen and bioemu2.

Both are used *as published* (no mdCATH fine-tuning for bioemu2; MDGen uses
its official ATLAS weights, with an optional mdCATH-trained checkpoint), and
both are scored on exactly the same windows as the three in-repo methods.

MDGen
-----
MDGen is a forward-simulation model: conditioned on the start frame it samples
a whole frame sequence.  We therefore run it once per window and read the
ensemble of terminal frames at every requested horizon off the SAME trajectory
(the correct, self-consistent way to use it), which also gives a genuine walk
for the order-aware metrics.

bioemu2
-------
bioemu2-preview is a *propagation* model: given an anchor it draws one endpoint
`lag_time_ns` into the future.  Endpoints are independent draws, so there is no
ordered path and the walk metrics are reported "n/a" for it (documented, not
hidden).  Score endpoints for the horizon's `lag_time_ns` exactly as the
existing baseline does.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import torch

from .common import window_seed


# =========================================================================== #
# MDGen
# =========================================================================== #
_MDGEN_CACHE = {}


def build_mdgen(args, device):
    """Load MDGen once and cache the wrapper + config on ``args``."""
    import sys
    sys.path.insert(0, "/liuyutian/mdgen_src/mdgen-master")
    import mdgen_baseline.mdgen_compat as mdgen_compat  # shims must come first

    from mdgen.wrapper import NewMDGenWrapper
    ckpt = args.mdgen_ckpt
    model = NewMDGenWrapper.load_from_checkpoint(ckpt, map_location=str(device))
    model = model.to(device).eval()
    m = model.args
    # Score *conditioned on the anchor* (sim_condition) for every method parity.
    m.sim_condition = True
    m.num_frames = int(args.mdgen_frames)
    m.sampling_method = args.mdgen_sampler
    if args.mdgen_steps:
        mdgen_compat.set_ode_steps(int(args.mdgen_steps))
        m.inference_steps = int(args.mdgen_steps)
    crop = int(getattr(m, "crop", 256))
    return {"model": model, "args": m, "crop": crop, "T": int(args.mdgen_frames),
            "restype_order": __import__("mdgen.residue_constants",
                                        fromlist=["restype_order"]).restype_order}


def _mdgen_atom14(anchor_bb: np.ndarray, crop: int):
    """[L,4,3] backbone (N,CA,C,O) -> padded [Lc,14,3] atom14 (side chains zero)."""
    L = anchor_bb.shape[0]
    a14 = np.zeros((L, 14, 3), dtype=np.float32)
    a14[:, :4] = anchor_bb
    if L < crop:
        a14 = np.pad(a14, ((0, crop - L), (0, 0), (0, 0)))
    else:
        a14 = a14[:crop]
    return a14, L


@torch.no_grad()
def sample_mdgen(bundle, win, n_samples, device, *, walk=True, **_ignored):
    """Sample one MDGen trajectory per replicate; returns terminal cloud + walk.

    ``n_samples`` replicates are drawn in a single batch.  The terminal frame at
    any horizon h is simply frame h of the trajectory (frames are 1 ns/step in
    mdCATH), so a single MDGen call serves *all* horizons.
    """
    from mdgen.geometry import atom14_to_frames
    model, margs, crop, T = (bundle["model"], bundle["args"], bundle["crop"], bundle["T"])
    ro_map = bundle["restype_order"]

    ctx = win["context"]
    rmask = win["residue_mask"]
    # MDGen samples a whole trajectory in ONE call, and every horizon is just a
    # frame of it.  Cache the call per window identity; without this a 4-horizon
    # sweep pays 4x the (expensive) ODE solve to compute the same trajectory.
    ck = (win["protein_id"], win["trajectory_id"], win["anchor_frame"],
          int(round(win["temperature_k"])))
    if ck in bundle.setdefault("cache", {}):
        a0, ca = bundle["cache"][ck]
        h = int(win["horizon"])
        return {"samples": ca[:, h],
                "walk": torch.cat([a0, ca[:, 1:h + 1]], dim=1) if walk else None,
                "walk_steps_ms": 1.0}

    anchor_bb = ctx["coords"][-1][rmask].numpy()               # [Lv,4,3]
    aatype = ctx["aatype"]
    aatype = aatype[-1] if aatype.dim() == 2 else aatype
    letters = "ACDEFGHIKLMNPQRSTVWYX"
    seq = "".join(letters[int(a)] if 0 <= int(a) < len(letters) else "X"
                  for a in aatype[rmask].tolist())
    ro = np.array([ro_map.get(c, 20) for c in seq], dtype=np.int64)

    a14, Lv = _mdgen_atom14(anchor_bb, crop)
    a14 = np.repeat(a14[None], n_samples, axis=0)              # [N,Lc,14,3]
    r = np.repeat(ro[None], n_samples, axis=0)
    if Lv < crop:
        pad = crop - Lv
        r = np.pad(r, ((0, 0), (0, pad)))
    x = torch.from_numpy(a14).float()
    flat = x[:, None].expand(-1, T, -1, -1, -1).contiguous().reshape(-1, crop, 14, 3)
    frames = atom14_to_frames(flat)
    batch = {
        "name": [f"{win['trajectory_id']}_{i}" for i in range(n_samples)],
        "frame_start": [0] * n_samples,
        "trans": frames._trans.reshape(n_samples, T, crop, 3).to(device),
        "rots": frames._rots._rot_mats.reshape(n_samples, T, crop, 3, 3).to(device),
        "torsions": torch.zeros(n_samples * T, crop, 7, 2, device=device),
        "torsion_mask": torch.ones(n_samples, crop, 7, device=device),
        "seqres": torch.from_numpy(r).long().to(device),
        "mask": torch.ones(n_samples, crop, device=device),
    }
    atom14, _ = model.inference(batch)                         # [N,T,Lc,14,3]
    ca = atom14[:, :, :Lv, 1, :].detach().float()               # [N,T,Lv,3]
    # MDGen frame t is t ns after the conditioned anchor (frame 0 ~= anchor),
    # so the walk is [anchor, frame1, ..., frame_horizon].
    h = int(win["horizon"])
    if h > T - 1:
        raise ValueError(f"MDGen was run with {T} frames; horizon {h} ns needs "
                         f"num_frames >= {h + 1}")
    a0 = torch.as_tensor(anchor_bb[None, :, 1, :], dtype=ca.dtype,
                         device=ca.device).expand(n_samples, 1, -1, -1)
    bundle["cache"][ck] = (a0, ca)
    walk_out = torch.cat([a0, ca[:, 1:h + 1]], dim=1) if walk else None
    return {"samples": ca[:, h], "walk": walk_out, "walk_steps_ms": 1.0}


# =========================================================================== #
# bioemu2
# =========================================================================== #
def build_bioemu2(args, device):
    """Load bioemu2-preview (published weights, no fine-tuning)."""
    os.environ.setdefault("BIOEMU2_MSA_CACHE", str(args.bioemu2_msa_cache))
    from bioemu2 import load_model as _load
    from bioemu2.sampling import sampler_from_preset
    ckpt = str(args.bioemu2_ckpt) if args.bioemu2_ckpt else None
    extra = {} if ckpt else {"checkpoint": args.bioemu2_name,
                             "cache_dir": str(args.bioemu2_cache)}
    model = _load(checkpoint_path=ckpt, device=str(device), precision="32-true",
                  matmul_precision="highest", **extra)
    sampler = sampler_from_preset(args.bioemu2_sampler, mode="equilibrium",
                                  num_steps=int(args.bioemu2_steps))
    return {"model": model, "sampler": sampler, "embeds": {}, "msa": {}}


def _bioemu2_embed(bundle, win):
    """Cached (system, trunk embeddings, CA indices) per (protein, T)."""
    from bioemu2 import get_msa
    from evaluation.baselines.bioemu2_io import ca_indices, make_system

    model = bundle["model"]
    rmask = win["residue_mask"]
    # system / MSA / trunk embeddings depend only on the SEQUENCE, not on the
    # temperature or the anchor, so cache them once per sequence: one embed per
    # protein serves all five temperature cells.
    system = make_system(win["window"], rmask)
    key = "".join(r.name for r in system.topology.residues)
    hit = bundle["embeds"].get(key)
    if hit is None:
        msa = bundle["msa"].get(key)
        if msa is None:
            msa = get_msa(system)
            bundle["msa"][key] = msa
        trunk = model.embed(system, msa=msa, mode="equilibrium")
        hit = (system, trunk, ca_indices(system))
        bundle["embeds"][key] = hit
    return hit


@torch.no_grad()
def sample_bioemu2(bundle, win, n_samples, device, *, walk=False, **_ignored):
    """bioemu2 endpoints (independent draws).

    ``walk=True`` asks bioemu2 for an ordered path instead: it propagates one
    chain of ``horizon`` unit-lag steps, which costs ``horizon`` denoiser calls.
    The cost is real, so the runner only requests it when walk metrics are
    actually being reported for bioemu2.
    """
    from bioemu2.propagation import (propagate_with_embeddings,
                                    simulate_trajectory_with_embeddings)
    from evaluation.baselines.bioemu2_io import window_anchor_x0
    from scripts.five_method.common import set_linalg_backend
    set_linalg_backend("bioemu2")

    model, sampler = bundle["model"], bundle["sampler"]
    rmask = win["residue_mask"]
    L_val = int(rmask.sum())
    system, trunk, ca_idx = _bioemu2_embed(bundle, win)
    if system.topology.n_residues != L_val:
        raise ValueError(f"sequence length {system.topology.n_residues} != {L_val}")
    x0 = window_anchor_x0(system, win["window"], rmask)
    seed = window_seed(_WinView(win), run=0, salt="bioemu2")

    if walk:
        # Ordered rollout at the window's *unit* lag (1 frame = 1 ns in mdCATH),
        # one independent trajectory per replicate -> [N, horizon+1, L, 3].
        n_steps = max(1, int(win["horizon"]))
        lag = float(win["horizon_ns"]) / n_steps
        with _seeded(seed):
            trajs = simulate_trajectory_with_embeddings(
                system, trunk, x0=x0[[0] * int(n_samples)], lag_time_ns=lag,
                num_trajectory_steps=n_steps, model=model, sampler=sampler,
                batch_size=None, seed=None, display_propagation_progress=False)
        arr = torch.stack([torch.as_tensor(t.xyz[:, ca_idx] * 10.0) for t in trajs])
        w = arr.to(device).float()                              # [N,T,L,3]
        return {"samples": w[:, -1], "walk": w, "walk_steps_ms": lag}

    with _seeded(seed):
        traj = propagate_with_embeddings(
            system, trunk, x0=x0[[0] * int(n_samples)],
            lag_time_ns=float(win["horizon_ns"]), model=model,
            sampler=sampler, batch_size=None, seed=None, progress=False)
    samples = torch.as_tensor(traj.xyz[:, ca_idx] * 10.0, dtype=torch.float32,
                              device=device)                    # [N,Lv,3] Angstrom
    return {"samples": samples, "walk": None, "walk_steps_ms": None}


class _seeded:
    """Seed the *global* torch RNG (bioemu2's denoiser reads it directly)."""

    def __init__(self, seed):
        self.seed = int(seed)

    def __enter__(self):
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        self._ctx = torch.random.fork_rng(devices=devices)
        self._ctx.__enter__()
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed % (2 ** 32 - 1))

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


class _WinView:
    """Minimal duck-typed window for ``common.window_seed``."""

    def __init__(self, w):
        self.protein_id = w["protein_id"]
        self.trajectory_id = w["trajectory_id"]
        self.anchor_frame = w["anchor_frame"]
        self.horizon = w["horizon"]
        self.temperature_k = w["temperature_k"]
