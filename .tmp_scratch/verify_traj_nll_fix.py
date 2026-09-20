"""Numerical verification: trajectory NLL dt-mismatch fix.

Loads the v12 checkpoint and computes rollout_trajectory_nll with:
  (a) the buggy full-span horizon_time_ps as per-step dt
  (b) the fixed per-frame dt (horizon_time_ps / horizon_frames)

Confirms (a) explodes and (b) is bounded.
"""
import sys
sys.path.insert(0, "/liuyutian/protein_jepa_frame")
import torch
from protein_jepa_frame.config import load_config
from protein_jepa_frame.models.jepa import ProteinJEPA

CKPT = "/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v12_improved/checkpoint_latest.pt"
CFG = "/liuyutian/protein_jepa_frame/configs/lspwm_v44_stage_b_v12_improved.yaml"

cfg = load_config(CFG)
model = ProteinJEPA(cfg.model).to("cuda:0")
state = torch.load(CKPT, map_location="cuda:0")
sd = state.get("model", state.get("model_state_dict", state))
missing, unexpected = model.load_state_dict(sd, strict=False)
model.eval()

wm = model.lspwm.world_model
torch.manual_seed(0)
batch, seq_len, d = 8, 7, cfg.model.latent_dim
# Simulate realistic consecutive frames: small per-frame drift
q0 = torch.randn(batch, d, device="cuda:0") * 0.5
q_seq = [q0]
for _ in range(seq_len - 1):
    q_seq.append(q_seq[-1] + torch.randn(batch, d, device="cuda:0") * 0.05)
q_seq = torch.stack(q_seq, dim=1)  # [batch, seq_len, d]
condition = torch.randn(batch, cfg.model.lspwm_condition_dim, device="cuda:0")
temp = torch.full((batch,), 348.0, device="cuda:0")

frame_interval_ps = 1000.0
horizon_frames = 96  # a typical horizon
full_span = torch.full((batch,), frame_interval_ps * horizon_frames, device="cuda:0")
per_frame = torch.full((batch,), frame_interval_ps, device="cuda:0")

with torch.no_grad():
    nll_bug = wm.rollout_trajectory_nll(q_seq, condition, full_span, temp)
    nll_fix = wm.rollout_trajectory_nll(q_seq, condition, per_frame, temp)

print(f"BUGGY (dt=full_span {full_span[0].item():.0f}ps):  trajectory_nll = {nll_bug.item():.3e}")
print(f"FIXED (dt=per_frame {per_frame[0].item():.0f}ps): trajectory_nll = {nll_fix.item():.3e}")
print(f"Ratio (buggy/fixed): {(nll_bug/nll_fix).item():.1f}x")
