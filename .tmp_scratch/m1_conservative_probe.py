import numpy as np, torch, sys
sys.path.insert(0,'.')
from artifacts.reviewer_experiments._shared.lspwm_adapter import LSPWMDynamicsModel
from artifacts.reviewer_experiments._shared.latent_trajectories import extract_ordered_latent_trajectories, split_temporal
ck='artifacts/lspwm_v44_stage_b_v6_kernel_fix/checkpoint_latest.pt'
cfg='artifacts/lspwm_v44_stage_b_v6_kernel_fix/eval_improved/eval_config.json'
man='manifests/tmjepa_v3.validation.jsonl'
ad=LSPWMDynamicsModel(); ad.load(checkpoint_path=ck,config_path=cfg,manifest=man,device='cuda',batch_size=32,max_batches=1,seed=6303)
traj,meta,_=extract_ordered_latent_trajectories(ad,man,batch_size=32,clip_frames=24,window_stride=8,max_batches=None)
ad.close()
tr,te=split_temporal(traj)
Ztr=np.concatenate([s[:-1] for s in tr.values()]); DZtr=np.concatenate([s[1:]-s[:-1] for s in tr.values()])
Zte=np.concatenate([s[:-1] for s in te.values()]); DZte=np.concatenate([s[1:]-s[:-1] for s in te.values()])
dev='cuda'; d=Ztr.shape[1]
c=torch.tensor(Ztr.mean(0),dtype=torch.float32,device=dev)
xt=torch.tensor(Ztr,dtype=torch.float32,device=dev); yt=torch.tensor(DZtr,dtype=torch.float32,device=dev)
xv=torch.tensor(Zte,dtype=torch.float32,device=dev); yv=torch.tensor(DZte,dtype=torch.float32,device=dev)
import torch.nn as nn
# Conservative field: drift = -grad U(z), U scalar MLP. drift is exact gradient (curl-free).
class Pot(nn.Module):
    def __init__(s):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(d,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,1))
    def forward(s,z): return s.net(z-c).squeeze(-1)
    def grad(s,z):
        z=z.detach().requires_grad_(True)
        U=s(z).sum()
        g,=torch.autograd.grad(U,z,create_graph=True)
        return g
pot=Pot().to(dev)
opt=torch.optim.Adam(pot.parameters(),1e-3)
for e in range(500):
    opt.zero_grad()
    drift=-pot.grad(xt)
    loss=((drift-yt)**2).mean()
    loss.backward(); opt.step()
with torch.no_grad():
    pass
drift_v=(-pot.grad(xv)).detach()
r2=1-((yv-drift_v)**2).sum().item()/((yv-yv.mean(0))**2).sum().item()
cos=(drift_v*yv).sum(1)/(drift_v.norm(dim=1)*yv.norm(dim=1)+1e-12)
print('CONSERVATIVE (gradient-of-scalar) drift held-out R2 = %.4f  cos=%.4f'%(r2,cos.mean().item()))
# Helmholtz: how much of free-MLP drift is conservative vs rotational?
free=nn.Sequential(nn.Linear(d,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,d)).to(dev)
opt2=torch.optim.Adam(free.parameters(),1e-3)
for e in range(500):
    opt2.zero_grad(); loss=((free(xt-c)-yt)**2).mean(); loss.backward(); opt2.step()
with torch.no_grad():
    fv=free(xv-c); r2f=1-((yv-fv)**2).sum().item()/((yv-yv.mean(0))**2).sum().item()
print('FREE-MLP drift held-out R2 = %.4f'%r2f)
print('conservative captures %.0f%% of free-MLP predictable variance'%(100*max(0,r2)/max(1e-9,r2f)))
