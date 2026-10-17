"""Stage B1b training: H-space latent dynamics with Kalman innovation NLL.

Implements STAGE_ABC_REDESIGN_PLAN.md §2.4-2.6:
  - encoder frozen (Stage A V2 context encoder)
  - H normalisation frozen (Stage B1a HNormalizer)
  - H is stop_grad'ed; only dynamics params (energy U, static bath D/C/Gamma,
    causal-GRU history posterior r_psi) train
  - per-sample objective: Kalman innovation NLL over a continuous trajectory
    segment [H_0..H_N] conditioned on the K-frame history H_{-K+1:0}
  - after es_warmup_steps an energy-score term is added from short rollouts

Usage:
    python -m protein_jepa_frame.training.train_stage_b1b \
        --manifest artifacts/mdcath_manifest_200dom/train.jsonl \
        --stage-a-ckpt artifacts/full_abc_v2/stage_a/checkpoint_latest.pt \
        --b1a-ckpt     artifacts/full_abc_v2/stage_b1a/checkpoint_latest.pt \
        --output       artifacts/full_abc_v2/stage_b1b
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.stage_a_v2 import StageAV2, StageAV2Config
from protein_jepa_frame.models.stage_b1a import StageBOracleDecoder, StageBOracleConfig
from protein_jepa_frame.models.stage_b1b import (
    StageB1b, StageB1bConfig, StageB1bLossWeights, compute_stage_b1b_loss,
    ResidualCADecoder, ResidualDecoderConfig,
)
from protein_jepa_frame.models.stage_b1b.objectives import energy_score, kabsch_rmsd_loss
from protein_jepa_frame.training.train_utils import (
    _device, _lr_scale, _move_batch, _set_reproducibility,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage B1b (H-space dynamics)")
    p.add_argument("--manifest", type=str, default="artifacts/mdcath_manifest_200dom/train.jsonl")
    p.add_argument("--stage-a-ckpt", type=str, required=True)
    p.add_argument("--b1a-ckpt", type=str, default=None,
                   help="Stage B1a checkpoint providing the frozen HNormalizer")
    p.add_argument("--output", type=str, default="artifacts/full_abc_v2/stage_b1b")
    p.add_argument("--resume", type=str, default=None)

    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)

    # sequence construction: history K frames for r_psi, then N supervised steps
    p.add_argument("--history-frames", type=int, default=8,
                   help="K: frames fed to the history posterior (2.3)")
    p.add_argument("--horizon-steps", type=int, default=8,
                   help="N: supervised Kalman steps per sample")

    # Model
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--bath-dim", type=int, default=8,
                   help="dS; 0 disables the bath (H-only ablation)")
    p.add_argument("--energy-depth", type=int, default=3)
    p.add_argument("--step-ps", type=float, default=1000.0,
                   help="integration interval h (ps); MDcath = 1 ns/frame")

    # Loss
    p.add_argument("--nll-weight", type=float, default=1.0)
    p.add_argument("--es-weight", type=float, default=0.1)
    p.add_argument("--es-warmup-steps", type=int, default=2000)
    p.add_argument("--es-num-samples", type=int, default=4)
    p.add_argument("--es-horizon-steps", type=int, default=8)

    # End-to-end RMSD head (residual CA decoder anchored on the context frame).
    # This decodes rollout H into CA displacements relative to the last context
    # frame and supervises Kabsch RMSD to the true future.  Zero-output => the
    # persistence baseline, so it can only improve on persistence.
    p.add_argument("--rmsd-weight", type=float, default=1.0,
                   help="weight on the end-to-end Kabsch-RMSD loss (0 disables)")
    p.add_argument("--rmsd-horizon-steps", type=int, default=4,
                   help="number of rolled-out steps supervised by RMSD")
    p.add_argument("--decoder-hidden", type=int, default=256)
    p.add_argument("--decoder-depth", type=int, default=3)
    p.add_argument("--decoder-lr", type=float, default=3e-4,
                   help="separate LR for the residual decoder")

    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _take_last_k(batch: dict, k: int) -> dict:
    out = dict(batch)
    for key in ("context_coords", "context_residue_mask", "context_time_ps"):
        v = out.get(key)
        if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] > k:
            out[key] = v[:, -k:].contiguous()
    return out


def build_paired_batch(samples, total_frames: int):
    return _take_last_k(collate_paired_windows(samples), total_frames)


@torch.no_grad()
def encode_frames(encoder, coords, rmask, aatype, ridx, cid, normalizer):
    """coords [B,T,L,4,3] -> H [B,T,L,dH], encoder frozen + stop_grad, normalised."""
    B, T, L = coords.shape[:3]
    H = encoder(
        coords.reshape(B * T, 1, L, 4, 3),
        rmask.reshape(B * T, 1, L),
        aatype[:, None].expand(B, T, L).reshape(B * T, L),
        ridx[:, None].expand(B, T, L).reshape(B * T, L),
        cid[:, None].expand(B, T, L).reshape(B * T, L),
        token_mask=rmask.reshape(B * T, 1, L),
    )[:, 0].reshape(B, T, L, -1)
    return normalizer(H)


def main() -> None:
    args = parse_args()
    _set_reproducibility(args.seed, False)
    device = _device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Frozen Stage A encoder ----
    ck = torch.load(args.stage_a_ckpt, map_location=device)
    a_cfg = StageAV2Config(**ck.get("stage_a_v2_config", {}))
    stage_a = StageAV2(a_cfg).to(device)
    missing, unexpected = stage_a.load_state_dict(ck["model"], strict=False)
    if missing:
        print(f"[B1b] Stage A ckpt missing keys (ok if predictor-only): {missing[:4]}", flush=True)
    stage_a.eval()
    encoder = stage_a.context_encoder
    for p in encoder.parameters():
        p.requires_grad_(False)

    # ---- Frozen H normaliser from B1a (else identity) ----
    if args.b1a_ckpt:
        b1a = torch.load(args.b1a_ckpt, map_location=device)
        o_cfg = StageBOracleConfig.from_saved(b1a.get("stage_b1a_config"))
        oracle = StageBOracleDecoder(o_cfg)
        missing, unexpected = oracle.load_state_dict(b1a["oracle"], strict=False)
        if missing:
            print(f"WARNING: {len(missing)} missing keys when loading B1a oracle "
                  f"(only the HNormalizer is required here): {missing[:4]}...", flush=True)
        normalizer = oracle.h_normalizer.to(device).eval()
        for p in normalizer.parameters():
            p.requires_grad_(False)
        print(f"loaded H normalizer from {args.b1a_ckpt}", flush=True)
    else:
        normalizer = torch.nn.Identity()
        print("WARNING: no B1a normalizer; using identity", flush=True)

    cfg = StageB1bConfig(
        latent_dim=args.latent_dim,
        bath_dim=args.bath_dim,
        energy_depth=args.energy_depth,
        step_ps=args.step_ps,
    )
    model = StageB1b(cfg).to(device)
    weights = StageB1bLossWeights(
        nll_weight=args.nll_weight,
        es_weight=args.es_weight,
        es_warmup_steps=args.es_warmup_steps,
        es_num_samples=args.es_num_samples,
        es_horizon_steps=args.es_horizon_steps,
    )

    # ---- Residual CA decoder for end-to-end RMSD supervision ----
    decoder = None
    if args.rmsd_weight > 0.0:
        dec_cfg = ResidualDecoderConfig(
            latent_dim=args.latent_dim,
            hidden=args.decoder_hidden,
            depth=args.decoder_depth,
        )
        decoder = ResidualCADecoder(dec_cfg).to(device)
        print(f"residual CA decoder: hidden={args.decoder_hidden} depth={args.decoder_depth}", flush=True)

    # ---- Data: K history frames + (N+1) supervised frames in one window ----
    total = args.history_frames + args.horizon_steps + 1
    dcfg = DataConfig(
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        training_mode="paired_window",
        num_frames=total,
        paired_all_horizons=False,
    )
    ds = build_dataset(dcfg)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda s: build_paired_batch(s, total),
        drop_last=True, pin_memory=(device.type == "cuda"),
    )

    param_groups = [{"params": list(model.parameters()), "lr": args.lr}]
    if decoder is not None:
        param_groups.append({"params": list(decoder.parameters()), "lr": args.decoder_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    for pg in optimizer.param_groups:
        pg["base_lr"] = pg["lr"]

    start_step = 0
    if args.resume:
        r = torch.load(args.resume, map_location=device)
        model.load_state_dict(r["model"])
        optimizer.load_state_dict(r["optimizer"])
        start_step = int(r.get("step", 0))
        print(f"resumed at {start_step}", flush=True)

    with (out_dir / "config.json").open("w") as f:
        json.dump({"stage_b1b_config": cfg.__dict__,
                   "loss_weights": weights.__dict__,
                   "train_args": vars(args)}, f, indent=2)

    model.train()
    step = start_step
    t0 = time.time()
    running: dict[str, float] = {}
    seen = 0

    while step < args.steps:
        for paired in loader:
            if step >= args.steps:
                break
            paired = _move_batch(paired, device)
            scale = _lr_scale(step, args.steps, args.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = pg["base_lr"] * scale

            ctx_c = paired["context_coords"]        # [B,K,L,4,3]
            ctx_m = paired["context_residue_mask"]
            fut_c = paired["future_coords"]         # [B,Tf,L,4,3]
            fut_m = paired["future_residue_mask"]
            aat = paired["future_aatype"]
            ridx = paired["future_residue_index"]
            cid = paired["future_chain_id"]

            K = ctx_c.shape[1]
            Np1 = min(args.horizon_steps + 1, fut_c.shape[1])

            # ---- Encode history and supervised windows (frozen, no grad) ----
            H_hist = encode_frames(encoder, ctx_c, ctx_m, aat, ridx, cid, normalizer)  # [B,K,L,dH]
            H_seq = encode_frames(
                encoder, fut_c[:, :Np1], fut_m[:, :Np1], aat, ridx, cid, normalizer,
            )                                                                          # [B,N+1,L,dH]

            rmask = fut_m[:, 0].bool()            # [B,L]; residues valid across the window
            dt_hist = torch.full((H_hist.shape[0], K), args.step_ps, device=device)
            temperature_k = paired.get("temperature_k")

            # The NLL depends on f = -grad_H U, so backward requires second-order
            # autograd through attention.  The flash/mem-efficient kernels do not
            # implement double backward; force kernels that do.
            # The NLL force f = -grad_H U requires double-backward through the
            # energy's attention.  Only the MATH kernel implements the needed
            # second-order derivative; the flash/efficient kernels do not.
            backends = [SDPBackend.MATH]

            with sdpa_kernel(backends):
                # ---- Kalman innovation NLL (§2.4) ----
                _want_diag = (step % args.log_every == 0)
                if _want_diag:
                    nll_sum, _diag = model.kalman_nll(
                        H_hist, dt_hist, H_seq, rmask, temperature_k,
                        return_diagnostics=True,
                    )
                    running["diag/bath_ratio"] = _diag.get("bath_ratio", float("nan"))
                    running["diag/jitter_hits"] = float(_diag.get("jitter_hits", 0))
                else:
                    nll_sum = model.kalman_nll(
                        H_hist, dt_hist, H_seq, rmask, temperature_k,
                    )
                n_valid = int(rmask.sum().item()) * (Np1 - 1) if Np1 > 1 else 1
                n_valid = max(n_valid, 1)

                # ---- Energy score (§2.5), added after warmup ----
                es_val = None
                lam = weights.lambda_es_at(step)
                nll_finite = torch.isfinite(nll_sum).all().item()
                if lam > 0.0 and args.es_horizon_steps > 0 and nll_finite:
                    es_val = _energy_score_term(
                        model, H_hist, dt_hist, H_seq, rmask, temperature_k, weights,
                    )
                    # Guard: if ES itself produced NaN/Inf, discard it
                    if es_val is not None and not torch.isfinite(es_val).all():
                        es_val = None

                comp = compute_stage_b1b_loss(nll_sum, n_valid, weights, step=step, es_value=es_val)

                # ---- End-to-end RMSD on teacher-forced one-step predictions ----
                rmsd_loss = torch.zeros((), device=device)
                pers_rmsd_val = 0.0
                if decoder is not None and args.rmsd_weight > 0.0:
                    rmsd_loss, pers_rmsd_val = _rmsd_term(
                        decoder, H_seq, fut_c[:, :Np1], rmask,
                        n_steps=args.rmsd_horizon_steps,
                    )

                total = comp.total + args.rmsd_weight * rmsd_loss

                optimizer.zero_grad(set_to_none=True)
                total.backward()
            if args.grad_clip > 0:
                clip_params = list(model.parameters())
                if decoder is not None:
                    clip_params = clip_params + list(decoder.parameters())
                torch.nn.utils.clip_grad_norm_(clip_params, args.grad_clip)
            optimizer.step()

            seen += 1
            for k, v in comp.as_loggable_dict().items():
                running[k] = running.get(k, 0.0) + v
            running["loss/rmsd"] = running.get("loss/rmsd", 0.0) + float(rmsd_loss.item())
            running["loss/persist_rmsd"] = running.get("loss/persist_rmsd", 0.0) + pers_rmsd_val
            if step % args.log_every == 0:
                avg = {k: v / seen for k, v in running.items()}
                # diag/* are instantaneous, not accumulated averages.
                for k in [k for k in avg if k.startswith("diag/")]:
                    avg[k] = running[k]
                print(f"step {step:6d} | lr {optimizer.param_groups[0]['lr']:.2e} | "
                      f"{time.time()-t0:.1f}s | " +
                      " ".join(f"{k.replace('b1b/','')}={v:.3g}" for k, v in avg.items()),
                      flush=True)
                running = {}
                seen = 0
                t0 = time.time()

            if step > 0 and step % args.save_every == 0:
                _save(out_dir, model, optimizer, step, cfg, weights, decoder)
            if args.dry_run and step >= 3:
                print("dry-run complete", flush=True)
                return
            step += 1

    _save(out_dir, model, optimizer, step, cfg, weights, decoder)
    print(f"training complete at step {step}", flush=True)


def _rmsd_term(decoder, H_seq, fut_coords, rmask, n_steps):
    """Kabsch-RMSD using oracle (true) future H as decoder input.

    For each supervised step n (1..n_steps):
        H_future = H_seq[:, n]        (true encoded future frame — oracle)
        H_anchor = H_seq[:, 0]        (true encoded anchor frame)
        X_hat    = decoder(H_future, H_anchor, X_anchor)
        loss     = Kabsch_RMSD(X_hat, X_true[:, n])

    Using true H gives the decoder a real signal: delta = H_future - H_anchor
    has measurable variance tied to actual conformational change.  Training
    the decoder with GLE-predicted mu (which is tiny and untrained) gives
    delta ≈ 0 and no gradient signal.

    Once the decoder learns to map H-deltas → displacements we can optionally
    couple it back to GLE rollout in a second phase.

    Returns (mean_rmsd_loss, mean_persistence_rmsd_float).
    """
    B, T, L, dH = H_seq.shape
    N = min(n_steps, T - 1)
    if N < 1:
        return H_seq.new_zeros(()), 0.0

    # CA coordinates [B, T, L, 3] from backbone atom index 1
    ca = fut_coords[:, :, :, 1, :]                  # [B, T, L, 3]

    H_anchor = H_seq[:, 0]                           # [B, L, dH]

    # Center anchor CA over valid residues.
    mask_f = rmask.unsqueeze(-1).to(ca.dtype)        # [B, L, 1]
    def _center(x):
        s = (x * mask_f).sum(1, keepdim=True)
        n = mask_f.sum(1, keepdim=True).clamp_min(1.0)
        return (x - s / n) * mask_f

    X_anchor = _center(ca[:, 0])                      # [B, L, 3]

    rmsd_losses = []
    pers_vals = []
    for n in range(1, N + 1):
        H_future = H_seq[:, n]                        # oracle true H
        out = decoder(H_future, H_anchor, X_anchor, rmask)
        X_pred = out["ca_pred"]
        X_true = _center(ca[:, n])

        rmsd_losses.append(kabsch_rmsd_loss(X_pred, X_true, rmask))
        with torch.no_grad():
            pers_vals.append(kabsch_rmsd_loss(X_anchor, X_true, rmask).item())

    loss = torch.stack(rmsd_losses).mean()
    pers = float(sum(pers_vals) / len(pers_vals)) if pers_vals else 0.0
    return loss, pers


def _energy_score_term(model, H_hist, dt_hist, H_seq, rmask, temperature_k, weights):
    """Short rollout from H_0; energy score against the observed trajectory (§2.5).

    Gradients flow through r_psi and the dynamics (detach=False), so ES trains
    both the history posterior and the kernel.
    """
    cfg = model.cfg
    B, T, L, dH = H_seq.shape
    M = max(2, weights.es_num_samples)
    Ns = min(weights.es_horizon_steps, T - 1)
    if Ns < 1:
        return None

    m = rmask.unsqueeze(-1).to(H_seq.dtype)                          # [B,L,1]
    # Observed trajectory, invalid residues zeroed to match the masked sampler.
    v_star = (H_seq[:, :Ns + 1] * m.unsqueeze(1)).reshape(B, -1)     # [B,(Ns+1)*L*dH]

    # Baseline history so r_psi contributes to the ES gradient (2.3).
    H_hist_rep = H_hist.unsqueeze(0).expand(M, *H_hist.shape).reshape(M * B, *H_hist.shape[1:])
    dt_rep = dt_hist.unsqueeze(0).expand(M, *dt_hist.shape).reshape(M * B, *dt_hist.shape[1:])
    rmask_rep = rmask.unsqueeze(0).expand(M, B, L).reshape(M * B, L)
    temp_rep = None
    if temperature_k is not None:
        t = temperature_k if torch.is_tensor(temperature_k) else torch.tensor(temperature_k)
        temp_rep = t.reshape(-1).expand(M * B) if t.numel() == 1 else t.unsqueeze(0).expand(M, B).reshape(-1)

    m0, chol0 = model.history_posterior(H_hist_rep, dt_rep, temp_rep, rmask_rep)
    eps = torch.randn(m0.shape, device=m0.device, dtype=m0.dtype)
    S0 = m0 + torch.einsum("blij,blj->bli", chol0, eps)              # [M*B,L,dS]

    H0 = H_seq[:, 0].unsqueeze(0).expand(M, *H_seq[:, 0].shape).reshape(M * B, L, dH)
    seq_feature = H0
    rollout = model.rollout(
        H0, rmask_rep, temp_rep, n_steps=Ns,
        seq_feature=seq_feature, S0=S0, detach=False,
    )                                                                # [M*B,Ns+1,L,dH]
    v_samp = rollout.reshape(M, B, (Ns + 1) * L * dH)

    es = torch.zeros((), device=H_seq.device, dtype=H_seq.dtype)
    for b in range(B):
        es = es + energy_score(v_samp[:, b], v_star[b])
    return es / B


def _save(out_dir, model, optimizer, step, cfg, weights, decoder=None):
    payload = {"model": model.state_dict(),
               "optimizer": optimizer.state_dict(),
               "step": step,
               "stage_b1b_config": cfg.__dict__,
               "loss_weights": weights.__dict__}
    if decoder is not None:
        payload["decoder"] = decoder.state_dict()
        payload["decoder_config"] = decoder.cfg.__dict__
    torch.save(payload, out_dir / "checkpoint_latest.pt")
    print(f"checkpoint saved at step {step}", flush=True)


if __name__ == "__main__":
    main()
