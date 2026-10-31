"""Post-training mechanism check for the S_0 arms.

If `exact` and `prior` converge to the same model there are two very different
explanations, and the experiment is only interpretable if they are told apart:

  (a) the history really carries little memory beyond H_0, so the optimizer
      shrinks the coupling C and the two arms legitimately become equivalent
      -- this is a RESULT;
  (b) the mechanism never engaged (C collapsed to ~0 for some other reason,
      or the bath stayed inert), so the arms are equivalent because S has no
      influence at all -- this is a BUG.

The discriminator is the trained coupling magnitude and the resulting leverage
of S_0 on the forecast, measured on each arm's OWN checkpoint.

Reports, per checkpoint:
  * effective coupling size: std(c_net output weight) * coupling_scale,
  * the per-step bath force |C S_0| relative to the energy force |f|,
  * the per-step bath mean displacement relative to the diffusion scale,
  * the no-update leverage of each S_0 mode at the checkpoint's own scale
    (RMS CA shift of the ensemble mean vs the prior, identical noise seed).

Usage
    python -m scripts.diagnose_s0_mechanism --checkpoint <ckpt> [--checkpoint ...]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.data import EvalProtocol, iter_windows
from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics

MODES = ("prior", "gru", "exact")


def effective_coupling(model):
    """std of the coupling head's output layer, times the coupling scale."""
    w = model.kernel.bath.c_net[-1].weight.detach()
    return float(w.std()) * float(model.kernel.cfg.coupling_scale)


def scalar_diagnostics(model, H_hist, dt, temp, mask, seq):
    """|C m0| vs |f| and bath displacement vs diffusion, in latent units."""
    dS = model.kernel.cfg.bath_dim
    model.kernel.s0_mode = "exact"
    m0, _ = model.kernel.initial_bath_state(H_hist, dt, temp, mask, seq_feature=seq)
    D, C, Gamma = model.kernel.bath(seq, temp)
    f = model.kernel.compute_forces(H_hist[:, 0], mask, temp, create_graph=False)
    cs = torch.einsum("blij,blj->bli", C, m0)
    h = float(model.kernel.cfg.step_ps)
    out = {}
    out["abs_Cm0_over_abs_f"] = float(cs.norm() / f.norm().clamp(min=1e-12))
    out["bath_disp_per_step"] = float(h * D.mean() * cs.norm())
    out["diffusion_per_step"] = float((2.0 * h * D.mean()).sqrt())
    out["abs_m0"] = float(m0.norm())
    out["mean_D"] = float(D.mean())
    out["mean_Gamma"] = float(Gamma.mean())
    out["abs_C"] = float(C.norm())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--manifest", default="artifacts/mdcath_manifest_full/val.jsonl")
    p.add_argument("--max-windows", type=int, default=40)
    p.add_argument("--max-residues", type=int, default=96)
    p.add_argument("--n-samples", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 16, 64])
    a = p.parse_args()
    torch.set_num_threads(4)

    protocol = EvalProtocol(
        manifest=a.manifest, horizons_frames=tuple(a.horizons),
        context_frames=8, frame_stride=1, max_residues=a.max_residues,
        max_anchors_per_trajectory=4,
        temperatures=(320., 348., 379., 413., 450.),
        seed=a.seed, round_robin_proteins=True, round_robin_conditions=True)
    windows = []
    for w in iter_windows(protocol):
        if len(windows) >= a.max_windows:
            break
        mask = w.context["residue_mask"].all(0).bool().cuda()
        if int(mask.sum()) < 4:
            continue
        windows.append((w, mask))

    result = {"scope": "S_0 mechanism diagnostics", "test_opened": False,
              "max_windows": len(windows), "checkpoints": {}}
    for ckpt in a.checkpoint:
        name = Path(ckpt).parent.name
        component = Path(a.out).with_suffix(f".{name}.component.pt")
        component.parent.mkdir(parents=True, exist_ok=True)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        torch.save(ck["component_checkpoint"], component)

        model = CADynamics(
            str(component), "cuda", ckpt,
            history_conditioning=ck["args"].get("history_conditioning", False),
            noise_mix=ck["args"].get("noise_mix", 0.),
            noise_length=ck["args"].get("noise_length", 8.),
            collective_mobility=ck["args"].get("collective_mobility", False),
            noise_kernel=ck["args"].get("noise_kernel", "gaussian"),
            noise_cutoff=ck["args"].get("noise_cutoff", 10.)).eval()
        trained_mode = ck["args"].get("s0_mode", "gru")

        entry = {"trained_s0_mode": trained_mode,
                 "coupling_scale": float(model.kernel.cfg.coupling_scale),
                 "gamma_scale": float(model.kernel.cfg.gamma_scale),
                 "c_init_std": float(model.kernel.cfg.c_init_std),
                 "effective_coupling": effective_coupling(model),
                 "step": int(ck.get("step", -1)),
                 "scalars": [], "leverage_A": {}}

        # Scalars on a handful of real windows (cheap; no sampling).
        with torch.no_grad():
            for w, mask in windows[:8]:
                args = [w.context[k][None].cuda() for k in
                        ("aatype", "residue_index", "chain_id")]
                coords = w.context["coords"][None].cuda()   # [1,K,L,4,3]
                K = coords.shape[1]
                h = model.bundle.encode(coords[:, -1], mask[None], *args)
                dt = torch.full((1, 4), model.kernel.cfg.step_ps, device="cuda")
                Hh = torch.stack([model.bundle.encode(coords[:, i], mask[None], *args)
                                  for i in range(K - 4, K)], dim=1)
                entry["scalars"].append(scalar_diagnostics(
                    model, Hh, dt, torch.tensor([w.temperature_k], device="cuda"),
                    mask[None], h))

        # Leverage per S_0 mode at the checkpoint's own scale.
        means = {m: {} for m in MODES}
        with torch.no_grad():
            for idx, (w, mask) in enumerate(windows):
                key = f"{w.protein_id}|{w.trajectory_id}|{w.anchor_frame}|{w.horizon}"
                seed = (a.seed * 1000003 +
                        int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)) % (2**31 - 1)
                for mode in MODES:
                    model.kernel.s0_mode = mode
                    gen = torch.Generator(device="cuda").manual_seed(seed)
                    ca, _ = model.sample(
                        w.context["coords"][None].cuda(), mask[None],
                        *[w.context[k][None].cuda() for k in
                          ("aatype", "residue_index", "chain_id")],
                        torch.tensor([w.temperature_k], device="cuda"),
                        w.horizon, a.n_samples, gen)
                    means[mode][idx] = ca[0][:, mask].mean(0).double().cpu()
        for mode in MODES:
            if mode == "prior":
                continue
            per_h = {}
            for i, (w, _) in enumerate(windows):
                d = (means[mode][i] - means["prior"][i]).norm(dim=-1)
                per_h.setdefault(int(w.horizon), []).append(float((d ** 2).mean().sqrt()))
            entry["leverage_A"][mode] = {h: sum(v) / len(v) for h, v in sorted(per_h.items())}

        result["checkpoints"][name] = entry
        sc = entry["scalars"]
        mean = lambda k: sum(s[k] for s in sc) / max(len(sc), 1)
        print(f"\n=== {name}  (trained s0_mode={trained_mode}, step={entry['step']}) ===")
        print(f"  coupling_scale        {entry['coupling_scale']:.3e}")
        print(f"  effective coupling    {entry['effective_coupling']:.3e} "
              f"(init would be {entry['c_init_std'] * entry['coupling_scale']:.3e})")
        print(f"  |C m0| / |f|          {mean('abs_Cm0_over_abs_f'):.3e}")
        print(f"  bath disp / step      {mean('bath_disp_per_step'):.3e}")
        print(f"  diffusion / step      {mean('diffusion_per_step'):.3e}")
        print(f"  |m0|                  {mean('abs_m0'):.3e}")
        print(f"  leverage (A)          " + "  ".join(
            f"{m}:" + ",".join(f"h{k}={v:.4f}" for k, v in d.items())
            for m, d in entry["leverage_A"].items()))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
