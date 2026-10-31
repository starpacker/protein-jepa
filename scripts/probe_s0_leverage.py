"""No-update probe: how much does the S_0 source move a trained LASP forecast?

For one checkpoint, sample the same windows/horizons under three S_0 sources
('prior', 'gru', 'exact') with an IDENTICAL random seed, so any difference is
attributable to S_0 and not to sampling noise.  Report:

  * leverage: RMS CA displacement of each mode's ensemble mean from 'prior';
  * the seven harmonized ensemble metrics against the true MD future.

Two kernel scales are probed because the deployed SOTA kernel is inert (the
bath's variance contribution is ~1e-8 of R), so at the trained scale all three
modes MUST agree; the 'active' scale (x100 coupling, Gamma x0.1) is the setting
where S is a real memory variable, and is where S_0 could plausibly matter.

This is a diagnostic on the val split.  No test split is opened.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.data import EvalProtocol, iter_windows
from evaluation.ensemble_metrics_ext import extended_ensemble_summary
from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics
from scripts.eval_lasp_final import METRICS

MODES = ("prior", "gru", "exact")


# Absolute bath scales the probe can put a checkpoint at.  These are ABSOLUTE,
# not multipliers of whatever the checkpoint was trained under: an arm trained
# at the active scale must not have the active change applied twice.
ABSOLUTE_SCALES = {
    # name: (coupling_scale, gamma_scale, gamma_floor)
    "sota":     (1e-3, 2e-4, 2e-5),   # the frozen SOTA kernel (inert)
    "active":   (1e-1, 2e-5, 2e-6),   # x100 coupling, Gamma x0.1
}


def build_model(ckpt, component, scale_name):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = CADynamics(
        str(component), "cuda", ckpt,
        history_conditioning=True,           # enable the history path for the probe
        noise_mix=ck["args"].get("noise_mix", 0.),
        noise_length=ck["args"].get("noise_length", 8.),
        collective_mobility=ck["args"].get("collective_mobility", False),
        noise_kernel=ck["args"].get("noise_kernel", "gaussian"),
        noise_cutoff=ck["args"].get("noise_cutoff", 10.)).eval()
    # Keep the S_0 source the checkpoint was trained with as the reported
    # default; the probe then overrides it per mode on purpose.
    model.kernel.s0_mode = ck["args"].get("s0_mode", "gru")
    if scale_name != "checkpoint":
        coupling, gamma, floor = ABSOLUTE_SCALES[scale_name]
        for obj in (model.kernel.cfg, model.kernel.bath):
            obj.coupling_scale = coupling
            obj.gamma_scale = gamma
            obj.gamma_floor = floor
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--manifest", default="artifacts/mdcath_manifest_full/val.jsonl")
    p.add_argument("--max-windows", type=int, default=60)
    p.add_argument("--max-residues", type=int, default=96)
    p.add_argument("--n-samples", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--scale", nargs="+", default=["checkpoint", "active"],
                   choices=["checkpoint", "sota", "active"],
                   help="'checkpoint' = the scale recorded in the checkpoint; "
                        "'sota'/'active' = those ABSOLUTE scales")
    a = p.parse_args()
    torch.set_num_threads(4)

    component = Path(a.out).with_suffix(".component.pt")
    component.parent.mkdir(parents=True, exist_ok=True)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    torch.save(ck["component_checkpoint"], component)

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

    result = dict(scope="S_0 no-update probe (optimizer_steps=0)",
                  checkpoint=a.checkpoint, test_opened=False,
                  max_windows=len(windows), n_samples=a.n_samples,
                  scales={})
    for scale_name in a.scale:
        model = build_model(a.checkpoint, component, scale_name)
        per_mode = {m: [] for m in MODES}
        means = {m: {} for m in MODES}
        print(f"=== scale {scale_name} ===", flush=True)
        with torch.no_grad():
            for idx, (w, mask) in enumerate(windows):
                key = f"{w.protein_id}|{w.trajectory_id}|{w.anchor_frame}|{w.horizon}"
                # Same seed for every S_0 mode: the shared noise isolates S_0.
                seed = (a.seed * 1000003 +
                        int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)) % (2**31 - 1)
                anchor = w.context["coords"][-1, :, 1].cuda()[mask]
                truth = w.future_path["coords"][:, :, 1].cuda()[:, mask]
                for mode in MODES:
                    model.kernel.s0_mode = mode
                    gen = torch.Generator(device="cuda").manual_seed(seed)
                    ca, _ = model.sample(
                        w.context["coords"][None].cuda(), mask[None],
                        *[w.context[k][None].cuda() for k in
                          ("aatype", "residue_index", "chain_id")],
                        torch.tensor([w.temperature_k], device="cuda"),
                        w.horizon, a.n_samples, gen)
                    pred = ca[0][:, mask]
                    means[mode][idx] = pred.mean(0).double().cpu()
                    met = extended_ensemble_summary(pred.float(), truth.float(),
                                                    None, cutoff=8., pca_k=3)
                    per_mode[mode].append(dict(
                        protein=w.protein_id, horizon=float(w.horizon),
                        **{k: float(met[k]) for k in METRICS}))
                if (idx + 1) % 10 == 0:
                    print(f"  window {idx + 1}/{len(windows)}", flush=True)

        # Leverage: RMS CA displacement of the ensemble mean from 'prior'.
        lev = {}
        for mode in MODES:
            if mode == "prior":
                continue
            per_h = {}
            for row in per_mode["prior"]:
                pass
            for i, (w, _) in enumerate(windows):
                d = (means[mode][i] - means["prior"][i]).norm(dim=-1)
                per_h.setdefault(int(w.horizon), []).append(float((d ** 2).mean().sqrt()))
            lev[mode] = {h: sum(v) / len(v) for h, v in sorted(per_h.items())}
        result["scales"][scale_name] = dict(
            scale=scale_name,
            absolute_scale=ABSOLUTE_SCALES.get(scale_name),
            leverage_A=lev,
            metrics={m: per_mode[m] for m in MODES})

    Path(a.out).write_text(json.dumps(result, indent=2))
    for scale_name in a.scale:
        s = result["scales"][scale_name]
        print(f"\n--- {scale_name} leverage (A rms of ensemble-mean shift vs prior) ---")
        for mode, v in s["leverage_A"].items():
            print(f"  {mode:6s} " + "  ".join(f"h{h}={d:.4f}" for h, d in v.items()))
        print(f"--- {scale_name} metrics (mean over windows) ---")
        for h in a.horizons:
            print(f"  h={h}")
            for mode in MODES:
                rows = [r for r in s["metrics"][mode] if int(r["horizon"]) == h]
                if not rows:
                    continue
                vals = {k: sum(r[k] for r in rows) / len(rows) for k in METRICS}
                print(f"    {mode:6s} " + "  ".join(f"{k}={vals[k]:.4f}" for k in METRICS))


if __name__ == "__main__":
    main()
