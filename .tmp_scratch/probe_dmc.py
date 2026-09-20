import torch, sys
sys.path.insert(0, '/liuyutian/protein_jepa_frame')
from protein_jepa_frame.config import load_config
from protein_jepa_frame.models.jepa import ProteinJEPA
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.collate import collate_clips, collate_paired_windows
from torch.utils.data import DataLoader

cfg = load_config('configs/lspwm_v44_stage_b_v17_force_fix.yaml')
dev = torch.device('cuda:0')
ds = build_dataset(cfg.data, msm_labels_path=cfg.data.msm_labels_path)
loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_paired_windows, drop_last=True)
model = ProteinJEPA(cfg.model).to(dev)
model._curriculum_step = 5000  # rollout phase

print('weight=', model.lspwm.drift_magnitude_consistency_weight)
print('calib=', model.lspwm.physical_calibration_enabled)

batch = next(iter(loader))
batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k,v in batch.items()}
out = model.forward_paired(batch)
drift_keys = [k for k in out if 'drift' in k or 'per_frame' in k or 'magnitude' in k]
for k in sorted(drift_keys):
    v = out[k]
    print(k, '=', float(v) if torch.is_tensor(v) and v.numel()==1 else type(v))
print('has magnitude key:', 'lspwm_drift_magnitude_consistency_loss' in out)
# also check context_q shape
cq = out.get('lspwm_context_sequence')
cm = out.get('lspwm_context_mask')
if cq is not None:
    print('context_q shape:', cq.shape, 'shape[1]>=2:', cq.shape[1]>=2)
if cm is not None:
    print('context_mask valid pairs:', int((cm[:, :-1] & cm[:, 1:]).sum()) if cm.shape[1]>=2 else 0)
