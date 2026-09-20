import torch, sys
sys.path.insert(0,'.')
from protein_jepa_frame.models.lspwm import (
    LSPWMWorldModel, LSPWMConfig,
    calibrated_drift_matching_loss, overdamped_transition_nll,
)
import dataclasses
# small model, calibration ENABLED, tiny diffusion like v6 (5e-5)
cfg=LSPWMConfig(state_dim=16, condition_dim=0, auxiliary_dim=4,
                relaxation_times_ps=(1000.,2000.,4000.,8000.),
                diffusion_scale=5e-5, coupling_scale=0.1,
                physical_calibration_enabled=True)
wm=LSPWMWorldModel(cfg)
torch.manual_seed(0)
B=64
q=torch.randn(B,16)
# observed increment with real O(1) magnitude & state-dependent drift
A=0.3*torch.randn(16,16)
target=q + (q@A)*0.5 + 0.05*torch.randn(B,16)
dt=torch.full((B,),8000.0)

def grad_norms(loss):
    wm.zero_grad()
    loss.backward()
    fg=sum(p.grad.abs().sum().item() for p in wm.potential.parameters() if p.grad is not None)
    mg=wm.physical_calibration.log_mobility_scale.grad
    mg=abs(mg.item()) if mg is not None else 0.0
    return fg, mg

# 1) OLD path: dynamic NLL (dead-force mechanism)
nll=overdamped_transition_nll(wm,q,target,None,dt,None,history_features=None)
fg_nll,_=grad_norms(nll)
# 2) NEW path: calibrated drift matching
dm=calibrated_drift_matching_loss(wm,q,target,None,dt,None,history_features=None,exclude_history=True)
fg_dm,mg_dm=grad_norms(dm)
print("dynamic NLL   -> force-grad L1 = %.3e"%fg_nll)
print("drift match   -> force-grad L1 = %.3e ,  mobility-grad = %.3e"%(fg_dm,mg_dm))
print("drift-match force-grad is %.0fx the NLL force-grad"%(fg_dm/max(fg_nll,1e-30)))
# 3) does optimizing drift-match actually reduce increment error & grow drift?
opt=torch.optim.Adam(wm.parameters(),1e-2)
with torch.no_grad():
    d0=(wm.calibrated_transition(q,None,dt,None,history_features=None)["mean"]-q)
    err0=((d0-(target-q))**2).mean().item(); mag0=d0.norm(dim=1).mean().item()
for i in range(300):
    opt.zero_grad()
    l=calibrated_drift_matching_loss(wm,q,target,None,dt,None,history_features=None,exclude_history=True)
    l.backward(); opt.step()
with torch.no_grad():
    d1=(wm.calibrated_transition(q,None,dt,None,history_features=None)["mean"]-q)
    err1=((d1-(target-q))**2).mean().item(); mag1=d1.norm(dim=1).mean().item()
    obs=(target-q).norm(dim=1).mean().item()
print("increment MSE  %.4e -> %.4e"%(err0,err1))
print("drift ||.||    %.4e -> %.4e   (observed %.4e)"%(mag0,mag1,obs))
print("mobility_scale now %.3e"%wm.physical_calibration.mobility_scale.item())
