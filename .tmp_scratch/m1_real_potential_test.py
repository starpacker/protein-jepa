import numpy as np, torch, sys
sys.path.insert(0,'.')
from artifacts.reviewer_experiments._shared.lspwm_adapter import LSPWMDynamicsModel
from artifacts.reviewer_experiments._shared.latent_trajectories import extract_ordered_latent_trajectories, split_temporal
from protein_jepa_frame.models.lspwm import calibrated_drift_matching_loss
ck='artifacts/lspwm_v44_stage_b_v6_kernel_fix/checkpoint_latest.pt'
cfg='artifacts/lspwm_v44_stage_b_v6_kernel_fix/eval_improved/eval_config.json'
man='manifests/tmjepa_v3.validation.jsonl'
ad=LSPWMDynamicsModel(); ad.load(checkpoint_path=ck,config_path=cfg,manifest=man,device='cuda',batch_size=32,max_batches=1,seed=6303)
traj,meta,_=extract_ordered_latent_trajectories(ad,man,batch_size=32,clip_frames=24,window_stride=8,max_batches=None)
# grab the real world model
wm=ad.model.lspwm.world_model
dev=next(wm.parameters()).device
# enable calibration on the real module, set mobility high
pc=wm.physical_calibration
import math
pc.enabled=True
with torch.no_grad():
    pc.log_mobility_scale.fill_(math.log(1e4))
ad_close=ad.close
tr,te=split_temporal(traj)
Ztr=np.concatenate([s[:-1] for s in tr.values()]); Ttr=np.concatenate([s[1:] for s in tr.values()])
Zte=np.concatenate([s[:-1] for s in te.values()]); Tte=np.concatenate([s[1:] for s in te.values()])
zt=torch.tensor(Ztr,dtype=torch.float32,device=dev); tt=torch.tensor(Ttr,dtype=torch.float32,device=dev)
zv=torch.tensor(Zte,dtype=torch.float32,device=dev); tv=torch.tensor(Tte,dtype=torch.float32,device=dev)
dt_tr=torch.full((zt.shape[0],),8000.0,device=dev); dt_te=torch.full((zv.shape[0],),8000.0,device=dev)
# optimize ONLY potential + calibration via drift matching
params=list(wm.potential.parameters())+[pc.log_mobility_scale,pc.log_free_energy_scale]
opt=torch.optim.Adam(params,3e-3)
def drift_mag_and_err(z,t,dt):
    with torch.no_grad():
        m=wm.calibrated_transition(z,None,dt,None,history_features=None)["mean"]
        d=m-z; obs=t-z
        err=((d-obs)**2).mean().item(); 
        r2=1-((obs-d)**2).sum().item()/((obs-obs.mean(0))**2).sum().item()
        return d.norm(dim=1).mean().item(), err, r2, obs.norm(dim=1).mean().item()
mag0,err0,r20,obsmag=drift_mag_and_err(zv,tv,dt_te)
print("BEFORE: drift||=%.4e  incMSE=%.4e  heldout_R2=%.4f  (obs||=%.4e)"%(mag0,err0,r20,obsmag))
N=zt.shape[0]; bs=256
for ep in range(60):
    perm=torch.randperm(N,device=dev)
    for i in range(0,N,bs):
        idx=perm[i:i+bs]
        opt.zero_grad()
        loss=calibrated_drift_matching_loss(wm,zt[idx],tt[idx],None,dt_tr[idx],None,history_features=None,exclude_history=True)
        loss.backward(); opt.step()
    if ep%15==0 or ep==59:
        mag,err,r2,_=drift_mag_and_err(zv,tv,dt_te)
        print("ep%2d: drift||=%.4e incMSE=%.4e heldout_R2=%.4f mobility=%.3e"%(ep,mag,err,r2,pc.mobility_scale.item()))
ad_close()
