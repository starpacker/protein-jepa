#!/usr/bin/env python3
"""Quick train M04 readout on synthetic data to make E01 metrics meaningful.

Trains PhysicalObservableReadout to map Q:[B,L,8] → CA distances, contacts, RMSF.
Uses frozen Stage B checkpoint to compute Qs from synthetic coordinates.
No future data access — supervised on static observables only.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import torch, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def make_synthetic_dataset(N=32, L=16, T=6):
    device = torch.device("cpu")
    r_base = torch.arange(L, dtype=torch.float32)
    base = torch.stack([1.5*r_base, 4.0*torch.sin(r_base/5.0), 4.0*torch.cos(r_base/5.0)], dim=-1)
    offsets = torch.tensor([[-1.2,0.1,0.0],[0.0,0.0,0.0],[1.3,0.1,0.0],[1.8,1.0,0.0]])
    all_coords = []
    for b in range(N):
        phase = float(b % 13) / 13.0 * 2.0 * np.pi
        amplitude = 0.5 + 0.5 * (b % 7) / 7.0
        sample = []
        for t in range(T):
            time_v = phase + 0.4 * t
            bend = torch.zeros_like(base)
            bend[:,1] = amplitude * torch.sin(r_base/7.0 + time_v) * 0.8
            bend[:,2] = amplitude * torch.cos(r_base/11.0 - time_v) * 0.6
            sample.append(base[:,None,:] + offsets[None,:,:] + bend[:,None,:])
        all_coords.append(torch.stack(sample, dim=0))
    coords = torch.stack(all_coords, dim=0)  # [N, T, L, 4, 3]
    mask_t = torch.ones(N, T, L, dtype=torch.bool)
    return coords, mask_t


def compute_targets(coords, mask_2d):
    N, T, L = coords.shape[:3]
    ca = coords[..., 1, :]
    target_ca = ca[:, -1]
    target_d = torch.cdist(target_ca, target_ca)
    target_contacts = (target_d < 8.0).float()
    m2d = mask_2d.unsqueeze(1).float() * mask_2d.unsqueeze(2).float()
    # RMSF
    target_mean = ca.mean(dim=1)
    target_rmsf = torch.sqrt(((ca - target_mean.unsqueeze(1))**2).mean(dim=1).sum(dim=-1).clamp(min=1e-8))
    return {"ca_d": target_d, "contacts": target_contacts, "rmsf": target_rmsf}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/smoke_stage_b/checkpoint_latest.pt")
    ap.add_argument("--output-dir", default="artifacts/m04_readout")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load frozen encoder
    print("Loading frozen Stage B encoder...")
    from protein_jepa_frame.config import ModelConfig
    from protein_jepa_frame.models.stage_b.model import StageBLASP
    cfg = ModelConfig(latent_dim=128, encoder_depth=2, temporal_depth=1,
        num_heads=2, num_rbf=8, max_frames=64, max_residues=128,
        dropout=0.0, orientation_features=False,
        time_conditioning="none", thermodynamic_conditioning=True)
    lasp = StageBLASP(cfg, state_dim=8, bath_dim=16, base_step_ps=100.0)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model", ckpt)
    remapped = {}
    for k, v in state.items():
        if k.startswith("context_encoder."):
            remapped["spatial_encoder." + k.removeprefix("context_encoder.")] = v
        elif k.startswith("temporal_encoder."):
            remapped[k] = v
    if remapped: state = remapped
    lasp.load_state_dict(state, strict=False)
    lasp.to(device); lasp.eval()
    for p in lasp.parameters(): p.requires_grad = False
    print(f"  Loaded: {args.checkpoint}")

    # Build readout
    from protein_jepa_frame.models.stage_c.readout import (
        ReadoutConfig, PhysicalObservableReadout,
        readout_distance_loss, readout_contact_loss, readout_rmsf_loss,
    )
    rcfg = ReadoutConfig(q_dim=8, hidden_dim=64, depth=2, num_rbf=16)
    readout = PhysicalObservableReadout(rcfg).to(device)
    readout.train()
    print(f"  Readout params: {sum(p.numel() for p in readout.parameters()):,}")

    # Data
    coords, mask_t = make_synthetic_dataset(N=32, L=16, T=6)
    coords = coords.to(device); mask_t = mask_t.to(device)
    B_total, T, L = coords.shape[:3]
    aa = torch.full((B_total, L), 20, dtype=torch.long, device=device)
    ri = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0).expand(B_total, -1)
    ch = torch.zeros(B_total, L, dtype=torch.long, device=device)
    tk = torch.full((B_total,), 348.0, device=device)

    # Pre-compute all Qs (frozen encoder)
    print("Pre-computing Qs from frozen encoder...")
    all_qs = []
    for b_start in range(0, B_total, args.batch_size):
        b_end = min(b_start + args.batch_size, B_total)
        with torch.no_grad():
            state = lasp.encode_state(
                coords[b_start:b_end], mask_t[b_start:b_end],
                aa[b_start:b_end], ri[b_start:b_end], ch[b_start:b_end],
                time_ps=None,
                temperature_k=None,
                use_temporal=True,
                return_bath=False,
                return_ca_distances=False,
                return_forces=False,
            )
            q = state["q"].detach()
        all_qs.append(q.cpu())
    all_qs = torch.cat(all_qs, dim=0)
    print(f"  Q shape: {tuple(all_qs.shape)}")

    # Compute targets
    print("Computing ground-truth observables...")
    all_targets = {}
    for b_start in range(0, B_total, args.batch_size):
        b_end = min(b_start + args.batch_size, B_total)
        targets = compute_targets(coords[b_start:b_end], mask_t[b_start:b_end, 0])
        for k in targets:
            if k not in all_targets:
                all_targets[k] = []
            all_targets[k].append(targets[k].cpu())
    for k in all_targets:
        all_targets[k] = torch.cat(all_targets[k], dim=0)
    print(f"  Target shapes: { {k: tuple(v.shape) for k,v in all_targets.items()} }")

    # Train/test split
    indices = torch.randperm(B_total)
    n_train = int(0.8 * B_total)
    train_idx = indices[:n_train].tolist()
    test_idx = indices[n_train:].tolist()

    optimizer = torch.optim.Adam(readout.parameters(), lr=args.lr)
    log_path = out_dir / "train.jsonl"
    best_loss = float("inf")
    best_weights = None

    print(f"\nTraining ({n_train} train, {B_total - n_train} test, {args.epochs} epochs)...")
    t0 = time.time()

    for epoch in range(args.epochs):
        # Shuffle
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        epoch_d_loss = 0.0
        epoch_c_loss = 0.0
        epoch_r_loss = 0.0

        for b_start in range(0, n_train, args.batch_size):
            b_end = min(b_start + args.batch_size, n_train)
            batch_idx = [train_idx[i] for i in perm[b_start:b_end].tolist()]
            B_cur = len(batch_idx)

            q_batch = all_qs[batch_idx].to(device)
            d_target = all_targets["ca_d"][batch_idx].to(device)
            c_target = all_targets["contacts"][batch_idx].to(device)
            r_target = all_targets["rmsf"][batch_idx].to(device)
            mask_b = torch.ones(B_cur, L, dtype=torch.bool, device=device)

            pred = readout(q_batch, mask_b)
            l_d = readout_distance_loss(pred["ca_distances"], d_target, mask_b)
            l_c = readout_contact_loss(pred["contact_prob"], c_target, mask_b)
            l_r = readout_rmsf_loss(pred["rmsf"], r_target, mask_b)
            loss = l_d + 0.5 * l_c + 0.1 * l_r

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_d_loss += l_d.item()
            epoch_c_loss += l_c.item()
            epoch_r_loss += l_r.item()

        n_batches = max(1, (n_train + args.batch_size - 1) // args.batch_size)
        avg_loss = epoch_loss / n_batches

        # Validation
        readout.eval()
        with torch.no_grad():
            val_q = all_qs[test_idx].to(device)
            val_d = all_targets["ca_d"][test_idx].to(device)
            val_c = all_targets["contacts"][test_idx].to(device)
            val_r = all_targets["rmsf"][test_idx].to(device)
            val_mask = torch.ones(len(test_idx), L, dtype=torch.bool, device=device)
            val_pred = readout(val_q, val_mask)
            val_loss = (
                readout_distance_loss(val_pred["ca_distances"], val_d, val_mask) +
                0.5 * readout_contact_loss(val_pred["contact_prob"], val_c, val_mask) +
                0.1 * readout_rmsf_loss(val_pred["rmsf"], val_r, val_mask)
            ).item()
        readout.train()

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in readout.state_dict().items()}

        record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_d": epoch_d_loss / n_batches,
            "train_c": epoch_c_loss / n_batches,
            "train_r": epoch_r_loss / n_batches,
            "val_loss": val_loss,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}: train={avg_loss:.4f} "
                  f"(d={record['train_d']:.3f} c={record['train_c']:.3f} r={record['train_r']:.3f}) "
                  f"val={val_loss:.4f}")

    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed:.1f}s, best_val={best_loss:.4f}")

    # Save best weights
    torch.save({"model": best_weights, "config": rcfg}, out_dir / "readout_checkpoint.pt")
    print(f"  Saved to {out_dir / 'readout_checkpoint.pt'}")

    # Final validation metrics
    readout.load_state_dict(best_weights)
    readout.eval()
    n_val = min(len(test_idx), val_q.shape[0])
    with torch.no_grad():
        val_pred = readout(val_q[:n_val], torch.ones(n_val, L, dtype=torch.bool, device=device))
        d_mae = (val_pred["ca_distances"] - val_d[:n_val]).abs().mean().item()
        c_acc = (val_pred["contact_prob"] > 0.5).eq(val_c[:n_val] > 0.5).float().mean().item()
        r_mae = (val_pred["rmsf"] - val_r[:n_val]).abs().mean().item()
        print(f"  Final val metrics: d_MAE={d_mae:.3f}Å, contact_acc={c_acc:.3f}, rmsf_MAE={r_mae:.4f}Å")

