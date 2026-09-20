import torch, sys, math
sys.path.insert(0, '/liuyutian/protein_jepa_frame')
from protein_jepa_frame.models.lspwm import calibrated_drift_matching_loss

ckpt = torch.load('/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v6_kernel_fix/checkpoint_latest.pt', map_location='cpu', weights_only=False)
from protein_jepa_frame.config import LSPWMConfig
from protein_jepa_frame.models.lspwm import LSPWM

cfg = LSPWMConfig()
lspwm = LSPWM(cfg)
lspwm_state = {}
for k, v in ckpt['model'].items():
    if 'lspwm.' in k:
        lspwm_state[k.replace('lspwm.', '')] = v
lspwm.load_state_dict(lspwm_state, strict=False)
wm = lspwm.world_model
for p in wm.potential.parameters():
    p.requires_grad_(True)
for p in wm.parameters():
    if p not in wm.potential.parameters():
        p.requires_grad_(False)

z = torch.randn(4, 256)
t = z + 0.03 * torch.randn(4, 256)
cond = torch.randn(4, 64)
temp = torch.full((4,), 348.0)

loss = calibrated_drift_matching_loss(wm, z, t, cond, 8000.0, temp)
print(f'Loss: {loss.item():.6f}')
loss.backward()
for name, p in wm.potential.named_parameters():
    if p.grad is not None:
        gn = p.grad.norm().item()
        print(f'{name}: grad_norm={gn:.4e}')
print('ALL OK')