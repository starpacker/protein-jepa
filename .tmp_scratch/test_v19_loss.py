import torch, sys
sys.path.insert(0,'/liuyutian/protein_jepa_frame')
from protein_jepa_frame.config import load_config
from protein_jepa_frame.models.jepa import ProteinJEPA
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.collate import collate_paired_windows
from torch.utils.data import DataLoader

cfg = load_config('configs/lspwm_v44_stage_b_v19_bath_aware.yaml')
dev = torch.device('cuda:1')
model = ProteinJEPA(cfg.model).to(dev)
ck = torch.load('artifacts/lspwm_v44_stage_b_v18_dmc_fix/checkpoint_latest.pt', map_location=dev)
sd = ck.get('model', ck.get('state_dict', ck))
miss,unexp = model.load_state_dict(sd, strict=False)
print('loaded miss=%d unexp=%d'%(len(miss),len(unexp)))
model.train()
model._curriculum_step = 3500  # dynamics phase so drift losses active
ds = build_dataset(cfg.data, msm_labels_path=cfg.data.msm_labels_path)
loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_paired_windows, drop_last=True)
batch = next(iter(loader))
batch = {k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in batch.items()}
out = model.forward_paired(batch)
for k in ['lspwm_bath_aware_drift_matching_loss','lspwm_bath_aware_conservative_only_loss','lspwm_bath_drift_fraction','lspwm_force_structure_loss','lspwm_per_frame_drift_matching_loss']:
    v = out.get(k)
    print(k, '=', float(v) if torch.is_tensor(v) and v.numel()==1 else v)
# check gradient flows into potential from bath-aware loss
loss = out['lspwm_bath_aware_drift_matching_loss'] + 0.05*out['lspwm_force_structure_loss']
model.zero_grad()
loss.backward()
pot = model.lspwm.world_model.potential
gnorm = sum(p.grad.norm().item() for p in pot.parameters() if p.grad is not None)
print('potential grad norm from v19 losses =', gnorm)
print('PASS' if gnorm>0 and torch.isfinite(loss) else 'FAIL')
