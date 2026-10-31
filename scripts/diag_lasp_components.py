"""Controlled encoder/decoder/GLE diagnostics; oracle branches use future truth.

Latent optimization is an attainable reconstruction probe, not a formal decoder
capacity bound. Temporal pair-distance correlation measures representation
sensitivity, not an information-theoretic test of encoder sufficiency.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.data import EvalProtocol, iter_windows
from evaluation.ensemble_metrics import align_ensemble, rmsf_profile
from evaluation.ensemble_metrics_ext import extended_ensemble_summary, _align_to, w2_1d
from protein_jepa_frame.training.train_stage_ensemble import _extract_backbone_frames
from scripts.eval_lasp_final import build


def correlation(x, y):
    x, y = x.double().flatten(), y.double().flatten()
    x, y = x - x.mean(), y - y.mean()
    den = x.norm() * y.norm()
    return float((x @ y) / den) if den > 1e-12 else None


def rmsd(pred, true):
    # Pairwise corresponding-frame superposition, independent of gauge drift.
    errors = [(_align_to(p[None].double(), t.double(), None)[0] - t).square().sum(-1).mean()
              for p, t in zip(pred, true)]
    return float(torch.stack(errors).mean().sqrt())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kernel-ckpt', required=True)
    p.add_argument('--ensemble-ckpt', default='artifacts/stage_ensemble_1000dom/checkpoint_best.pt')
    p.add_argument('--stage-a-ckpt', default='artifacts/stage_a_disp_relpos/checkpoint_latest.pt')
    p.add_argument('--b1a-ckpt', default='artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt')
    p.add_argument('--manifest', default='artifacts/mdcath_manifest_full/val.jsonl')
    p.add_argument('--max-windows', type=int, default=40)
    p.add_argument('--max-residues', type=int, default=96)
    p.add_argument('--history-frames', type=int, default=4)
    p.add_argument('--n-samples', type=int, default=6)
    p.add_argument('--oracle-steps', type=int, default=30)
    p.add_argument('--oracle-frames', type=int, default=5)
    p.add_argument('--oracle-lr', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    _, hs, _, _ = build(a, dev)
    hs.eval()
    for param in hs.parameters():
        param.requires_grad_(False)
    protocol = EvalProtocol(manifest=a.manifest, horizons_frames=(1,4,16,64),
                            context_frames=a.history_frames, max_residues=a.max_residues,
                            context_stride=1, frame_stride=1, max_anchors_per_trajectory=4,
                            anchor_stride=1, seed=a.seed, round_robin_proteins=True,
                            round_robin_conditions=True, max_windows_per_protein=4)
    rows = []
    for w in iter_windows(protocol):
        if len(rows) >= a.max_windows:
            break
        ctx = w.context
        valid = ctx['residue_mask'].bool().all(0) & w.future_path['residue_mask'].bool().all(0)
        if valid.sum() < 4:
            continue
        # Keep one consistent, explicitly cropped residue index set everywhere.
        x = ctx['coords'][:, valid].to(dev).float()[None]
        future = w.future_path['coords'][:, valid].to(dev).float()
        aa, ri, ci = [ctx[k][valid].to(dev)[None] for k in ('aatype','residue_index','chain_id')]
        n, length = future.shape[:2]
        mask = torch.ones(1, length, device=dev, dtype=torch.bool)
        def decode(h, R, t):
            count = h.shape[0]
            return hs.decoder.decode_structure(h, aa.expand(count,-1),
                    R.expand(count,-1,-1,-1), t.expand(count,-1,-1),
                    mask.expand(count,-1))['backbone_coords'][:,:,1]
        with torch.no_grad():
            R, t = _extract_backbone_frames(x[:,-1])
            H = hs.encoder(future[:,None], mask[:,None].expand(n,1,length),
                    aa.expand(n,-1), ri.expand(n,-1), ci.expand(n,-1),
                    token_mask=mask[:,None].expand(n,1,length))[:,0]
            Ha = hs.encode_frame(x[:,-1], mask, aa, ri, ci)
            reconstructed = decode(H, R, t)
            constant = decode(Ha.expand(n,-1,-1), R, t)
            self_R, self_t = _extract_backbone_frames(future)
            self_reconstructed = decode(H, self_R, self_t)
            hs._rollout_steps = int(w.horizon)
            rollout = hs.sample_future_ca(x, mask, aa, ri, ci, R, t,
                    n_samples=a.n_samples, temperature_k=float(w.temperature_k),
                    context_time_ps=ctx['time_ps'].to(dev)[None])
            end = rollout['ca_samples'][0]
            true = future[:,:,1]
            target = _align_to(true, x[0,-1,:,1], None)
            true_aligned = align_ensemble(true)
            frame_pairs = torch.triu_indices(n,n,1,device=dev)
            def distances(v):
                return torch.pdist(v.flatten(1)) / v[0].numel()**0.5
            hdist, xdist = distances(H), distances(true_aligned)
            h_roll = rollout['H_traj'][:,-1]
            # Match distributions against encoded true path; not single-realization endpoint MSE.
            anchor_radius_true = (H-Ha).square().mean((-1,-2)).sqrt()
            anchor_radius_roll = (h_roll-Ha).square().mean((-1,-2)).sqrt()
            rec_profile = rmsf_profile(align_ensemble(reconstructed))
            true_profile = rmsf_profile(true_aligned)
            idx = torch.linspace(0,n-1,min(n,a.oracle_frames),device=dev).round().long().unique()
            initial = H[idx].detach().clone()
            initial_pred = decode(initial,R,t)
            initial_mse = (initial_pred-target[idx]).square().mean().item()
        latent = torch.nn.Parameter(initial.clone())
        opt = torch.optim.Adam([latent], lr=a.oracle_lr)
        best = initial_pred.detach().clone()
        best_loss = (best-target[idx]).square().mean((-1,-2))
        for _ in range(a.oracle_steps):
            opt.zero_grad()
            pred = decode(latent,R,t)
            per_frame = (pred-target[idx]).square().mean((-1,-2))
            if not torch.isfinite(per_frame).all():
                raise RuntimeError('Nonfinite latent reconstruction probe')
            better = per_frame.detach() < best_loss
            best[better] = pred.detach()[better]
            best_loss = torch.minimum(best_loss,per_frame.detach())
            per_frame.mean().backward()
            if latent.grad is None or not torch.isfinite(latent.grad).all():
                raise RuntimeError('Nonfinite latent gradient')
            opt.step()
        with torch.no_grad():
            pred = decode(latent,R,t)
            per_frame = (pred-target[idx]).square().mean((-1,-2))
            better = per_frame < best_loss
            best[better] = pred[better]
            best_loss = torch.minimum(best_loss,per_frame)
            scores = {name: extended_ensemble_summary(cloud, true, None)
                      for name,cloud in [('encoded_future',reconstructed),('gle_endpoint',end),
                                        ('constant_latent',constant)]}
            row = dict(protein=w.protein_id,traj=w.trajectory_id,anchor_frame=int(w.anchor_frame),
                       horizon=float(w.horizon),temperature_k=float(w.temperature_k),
                       frames=n,residues=length,scores=scores,
                       encdec_ca_rmsd=rmsd(reconstructed,true),
                       constant_ca_rmsd=rmsd(constant,true),
                       self_anchor_ca_rmsd=rmsd(self_reconstructed,true),
                       encdec_rmsf_ratio=float(rec_profile.mean()/true_profile.mean().clamp_min(1e-8)),
                       latent_geometry_pair_corr=correlation(hdist,xdist),
                       latent_true_path_radius=float(anchor_radius_true.mean()),
                       latent_roll_radius=float(anchor_radius_roll.mean()),
                       latent_radius_w2=float(w2_1d(anchor_radius_roll,anchor_radius_true)),
                       oracle_initial_ca_rmsd=rmsd(initial_pred,true[idx]),
                       oracle_optimized_ca_rmsd=rmsd(best,true[idx]),
                       oracle_initial_gauge_mse=initial_mse,
                       oracle_best_gauge_mse=float(best_loss.mean()),
                       oracle_latent_change_rms=float((latent-initial).square().mean().sqrt()))
        rows.append(row)
        print(f"{len(rows):3d} H={w.horizon:2d} {w.protein_id} "
              f"encdec={row['encdec_ca_rmsd']:.3f} constant={row['constant_ca_rmsd']:.3f} "
              f"self-anchor={row['self_anchor_ca_rmsd']:.3f} "
              f"latent-opt={row['oracle_initial_ca_rmsd']:.3f}->{row['oracle_optimized_ca_rmsd']:.3f}",flush=True)
        result = dict(checkpoint=a.kernel_ckpt,args=vars(a),n_windows=len(rows),
                      n_proteins=len({r['protein'] for r in rows}),
                      interpretation='Future-informed diagnostic branches are not forecasts or formal capacity bounds.',
                      windows=rows)
        Path(a.out).parent.mkdir(parents=True,exist_ok=True)
        Path(a.out).write_text(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
