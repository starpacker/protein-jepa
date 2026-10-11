import torch, json, sys, os


from protein_jepa_frame.config import load_config
config = load_config('configs/full_abc_stage_c_v3.yaml')
device = 'cuda:0'
torch.cuda.empty_cache()

from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from torch.utils.data import DataLoader
from protein_jepa_frame.training.train_utils import _set_reproducibility, _move_batch
from protein_jepa_frame.models.stage_c import StructureDecoder, DecoderConfig, DirectEndpointHead, DirectHeadConfig, stage_c_total_loss
from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame
from protein_jepa_frame.models.stage_b.model import StageBLASP
from protein_jepa_frame.training.stage_c import _compute_target_torsions
import torch.nn.functional as F

_set_reproducibility(42, False)
dataset = build_dataset(config.data, msm_labels_path=None)
loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False, collate_fn=collate_paired_windows, drop_last=True)

stage_b_model = StageBLASP(config.model).to(device)
ckpt = torch.load(config.init_checkpoint, map_location=device)
state = ckpt.get('model', ckpt)
remapped = {}
for k, v in state.items():
    if k.startswith('context_encoder.'):
        remapped['spatial_encoder.' + k.removeprefix('context_encoder.')] = v
    else:
        remapped[k] = v
stage_b_model.load_state_dict(remapped, strict=False)
stage_b_model.eval()
for p in stage_b_model.parameters():
    p.requires_grad = False
print("Stage B loaded.", flush=True)

def get_batch_data(batch, device):
    B = batch['context_coords'].shape[0]; L = batch['context_coords'].shape[2]
    aatype = batch.get('future_aatype', batch.get('aatype'))
    if aatype.dim() == 1: aatype = aatype.unsqueeze(0).expand(B, -1)
    target_coords_raw = batch['future_coords']
    target_backbone = target_coords_raw[:, -1]
    target_mask = batch['future_residue_mask'][:, -1]
    with torch.no_grad():
        t_res_idx = batch.get('context_residue_index', torch.arange(L, device=device).unsqueeze(0).expand(B,-1))
        t_chain = batch.get('context_chain_id', torch.zeros(B, L, dtype=torch.long, device=device))
        temperature_k = batch.get('context_temperature_k', torch.ones(B, device=device)*348.0)
        if temperature_k.dim() > 1: temperature_k = temperature_k[:, -1]
        ctx_state = stage_b_model.encode_state(
            batch['context_coords'], batch['context_residue_mask'],
            aatype, t_res_idx, t_chain,
            time_ps=batch.get('context_time_ps'),
            temperature_k=temperature_k, use_temporal=True, return_bath=False,
            return_ca_distances=False, return_forces=False)
        h_frozen = ctx_state['h_last'].detach()
        fut_state = stage_b_model.encode_state(
            batch['future_coords'], batch['future_residue_mask'],
            aatype, t_res_idx, t_chain,
            time_ps=batch.get('future_time_ps'),
            temperature_k=temperature_k, use_temporal=True, return_bath=False,
            return_ca_distances=False, return_forces=False)
        q_oracle = fut_state['q']
    anchor_backbone = batch['context_coords'][:, -1]
    R_anchor, t_anchor = RigidBodyFrame.from_backbone(
        anchor_backbone[...,0,:], anchor_backbone[...,1,:], anchor_backbone[...,2,:])
    tau_ps = batch.get('horizon_time_ps', torch.tensor([1000.0], device=device))
    if tau_ps.dim() == 0: tau_ps = tau_ps.unsqueeze(0)
    return aatype, h_frozen, q_oracle, R_anchor, t_anchor, target_backbone, target_mask, temperature_k, tau_ps

def run_train(decoder, direct_head, q_mode, steps=30):
    all_params = list(decoder.parameters()) + list(direct_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=5e-5, weight_decay=0.01)
    for train_step in range(steps):
        try:
            batch = next(iter(loader))
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = _move_batch(batch, device)
        aatype, h_frozen, q_oracle, R_anchor, t_anchor, target_backbone, target_mask, temperature_k, tau_ps = get_batch_data(batch, device)
        
        optimizer.zero_grad()
        q_pred_mean, _ = direct_head(h_frozen, None, tau_ps, temperature_k, target_mask)
        q_decode = q_oracle if q_mode == 'oracle' else q_pred_mean
        q_pn = q_pred_mean.norm(dim=-1).mean().item()
        q_on = q_oracle.norm(dim=-1).mean().item()
        
        if torch.isnan(q_decode).any():
            print(f"[{q_mode}] step {train_step}: NaN in q_decode", flush=True); return False
        
        decoder_outputs = decoder(aatype, h_frozen, q_decode, R_anchor, t_anchor, target_mask)
        for key, val in decoder_outputs.items():
            if isinstance(val, torch.Tensor) and (torch.isnan(val).any() or torch.isinf(val).any()):
                print(f"[{q_mode}] step {train_step}: NaN/Inf in decoder[{key}]", flush=True)
                if q_mode == 'predicted':
                    print(f"  q_pred [{q_pred_mean.min():.4f}, {q_pred_mean.max():.4f}] norm={q_pn:.2f}", flush=True)
                    print(f"  q_orc  [{q_oracle.min():.4f}, {q_oracle.max():.4f}] norm={q_on:.2f}", flush=True)
                return False
        
        R_target, t_target = RigidBodyFrame.from_backbone(
            target_backbone[...,0,:], target_backbone[...,1,:], target_backbone[...,2,:])
        target_torsions = _compute_target_torsions(target_backbone)
        target = {'backbone_coords': target_backbone, 'CA': target_backbone[...,1,:],
                  'R': R_target, 't': t_target, 'torsion_angles': target_torsions}
        
        total_loss, components = stage_c_total_loss(decoder_outputs, target, target_mask, use_frame_aligned=False)
        direct_mse = F.mse_loss(q_pred_mean, q_oracle)
        total_loss = total_loss + direct_mse * 0.5
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"[{q_mode}] step {train_step}: NaN/Inf loss", flush=True)
            for n, v in components.items(): print(f"  loss_{n}: {v.item():.6f}", flush=True)
            return False
        
        total_loss.backward()
        
        max_g = 0.0
        for name, param in all_params if True else []:
            pass
        for name, param in list(decoder.named_parameters()) + list(direct_head.named_parameters()):
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"[{q_mode}] step {train_step}: NaN grad in {name}", flush=True); return False
                if torch.isinf(param.grad).any():
                    print(f"[{q_mode}] step {train_step}: Inf grad in {name}", flush=True); return False
                gmax = param.grad.abs().max().item()
                if gmax > max_g: max_g = gmax
        
        torch.nn.utils.clip_grad_norm_(all_params, 10.0)
        optimizer.step()
        
        if train_step < 5 or train_step % 5 == 0:
            cs = ' '.join([f'{n}={v.item():.3f}' for n,v in components.items()])
            print(f"[{q_mode}] step {train_step}: loss={total_loss.item():.4f} ({cs}) dmse={direct_mse.item():.3f} |qp|={q_pn:.2f} |qo|={q_on:.2f} max_g={max_g:.2e}", flush=True)
    return True

# Test 1: Oracle Q
print("\n=== Oracle Q ===", flush=True)
d1 = StructureDecoder(DecoderConfig(context_dim=128, q_dim=8, c_hidden=128)).to(device)
dh1 = DirectEndpointHead(DirectHeadConfig(context_dim=128, condition_dim=0, latent_dim=8, hidden_dim=128)).to(device)
ok1 = run_train(d1, dh1, 'oracle', 20)
print(f"Oracle: {'PASS' if ok1 else 'FAIL'}", flush=True)

# Test 2: Predicted Q
print("\n=== Predicted Q ===", flush=True)
d2 = StructureDecoder(DecoderConfig(context_dim=128, q_dim=8, c_hidden=128)).to(device)
dh2 = DirectEndpointHead(DirectHeadConfig(context_dim=128, condition_dim=0, latent_dim=8, hidden_dim=128)).to(device)
ok2 = run_train(d2, dh2, 'predicted', 20)
print(f"Predicted: {'PASS' if ok2 else 'FAIL'}", flush=True)

# Test 3: Extended oracle training
print("\n=== Extended Oracle Q (30 steps) ===", flush=True)
ok3 = run_train(d1, dh1, 'oracle', 30)
print(f"Extended Oracle: {'PASS' if ok3 else 'FAIL'}", flush=True)

torch.cuda.empty_cache()
print("\nDone.", flush=True)
