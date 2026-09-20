import torch, sys
sys.path.insert(0, '/liuyutian/protein_jepa_frame')
from protein_jepa_frame.config import load_config
from protein_jepa_frame.models.jepa import ProteinJEPA
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.collate import collate_paired_windows
from torch.utils.data import DataLoader

cfg = load_config('configs/lspwm_v44_stage_b_v18_dmc_fix.yaml')
dev = torch.device('cuda:0')
model = ProteinJEPA(cfg.model).to(dev)
ck = torch.load('artifacts/lspwm_v44_stage_b_v18_dmc_fix/checkpoint_latest.pt', map_location=dev)
sd = ck.get('model', ck.get('state_dict', ck))
model.load_state_dict(sd, strict=False); model.eval()
ds = build_dataset(cfg.data, msm_labels_path=cfg.data.msm_labels_path)
loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_paired_windows, drop_last=True)
batch = next(iter(loader))
batch = {k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in batch.items()}
with torch.no_grad():
    out = model.forward_paired(batch)
cq = out['lspwm_context_sequence']; cm = out['lspwm_context_mask']
print('context_q', tuple(cq.shape), 'mask', tuple(cm.shape))
print('mask per-row valid frames:', cm.sum(dim=1).tolist())
wm = model.lspwm.world_model
print('auxiliary_dim', wm.config.auxiliary_dim, 'state_dim', wm.config.state_dim)
# check step can run on a window slice
q0 = cq[:, 0, :]
aux0 = q0.new_zeros(q0.shape[0], wm.config.auxiliary_dim)
temp = out['lspwm_temperature_k']
print('temp shape', None if temp is None else tuple(temp.shape))
with torch.no_grad():
    st = wm.step(q0, aux0, out.get('lspwm_condition'), 8000.0, temp, stochastic=False)
print('step ok. next_q', tuple(st['q'].shape), 'aux', tuple(st['auxiliary'].shape))
print('q_drift norm mean', st['q_drift'].norm(dim=-1).mean().item())
print('coupling shape', tuple(st['coupling'].shape))
# does coupling·aux contribute? with aux0=0 it is 0; check second step
with torch.no_grad():
    st2 = wm.step(st['q'], st['auxiliary'], out.get('lspwm_condition'), 8000.0, temp, stochastic=False)
    bath_contrib = torch.einsum('bqa,ba->bq', st2['coupling'].expand(q0.shape[0],-1,-1), st['auxiliary'])
    grad2 = -wm.potential.mean_force(st['q'], out.get('lspwm_condition'), temp)
    D2 = wm.coefficients(st['q'], temp)['diffusion']
    consv = -torch.einsum('bij,bj->bi', D2, grad2)
print('step2 conservative-drift norm', consv.norm(dim=-1).mean().item())
print('step2 bath-contrib   norm', bath_contrib.norm(dim=-1).mean().item())
print('ratio bath/consv', (bath_contrib.norm(dim=-1).mean()/consv.norm(dim=-1).mean().clamp(min=1e-12)).item())
