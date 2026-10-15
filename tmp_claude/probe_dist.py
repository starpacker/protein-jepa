"""Measure whether frozen Stage A H encodes CA-CA geometry.
Baseline = predict mean distance (per split). Compare to the model's
struct_recon_head reconstruction RMSE, plus a quick held-out MLP probe.
"""
import sys, math
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.stage_a_v2 import StageAV2, StageAV2Config

CKPT="artifacts/abc_v3/stage_a_abs/checkpoint_latest.pt"
MANIFEST="artifacts/mdcath_manifest_200dom/train.jsonl"
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")

ck=torch.load(CKPT,map_location=dev)
cfg=StageAV2Config(**ck.get("stage_a_v2_config",{}))
sa=StageAV2(cfg).to(dev)
sa.load_state_dict(ck["model"],strict=False); sa.eval()
enc=sa.context_encoder
for p in enc.parameters(): p.requires_grad_(False)
print("step", ck.get("step"), "delta_target", cfg.delta_target)

dc=DataConfig(manifest_path=MANIFEST)
ds=build_dataset(dc)
import random; random.seed(0)
idxs=random.sample(range(len(ds)), 24)

@torch.no_grad()
def encode(sample):
    # dataset returns flat tensors: coords [T,L,4,3], residue_mask [T,L], aatype [L], etc.
    coords=sample["coords"][:1].to(dev)   # [1,L,4,3]
    rmask=sample["residue_mask"][:1].to(dev)  # [1,L]
    aa=sample["aatype"].to(dev); ri=sample["residue_index"].to(dev); ci=sample["chain_id"].to(dev)
    H=enc(coords[None],rmask[None],aa[None],ri[None],ci[None],token_mask=rmask[None])[0,0]
    v=rmask[0].bool()
    ca=coords[0,:,1,:][v]
    return H[v], ca

# collect pairs
feats=[]; dists=[]
recon_sq=[]; 
allpairs_d=[]
for k,i in enumerate(idxs):
    H,ca=encode(ds[i])
    L=H.shape[0]
    if L<4: continue
    npair=min(400, L*L)
    ii=torch.randint(L,(npair,),device=dev); jj=torch.randint(L,(npair,),device=dev)
    Hi=H[ii]; Hj=H[jj]
    d=(ca[ii]-ca[jj]).norm(dim=-1)
    # model recon head
    fij=torch.cat([Hi,Hj],-1); fji=torch.cat([Hj,Hi],-1)
    with torch.no_grad():
        pr=0.5*(sa.struct_recon_head(fij)+sa.struct_recon_head(fji)).squeeze(-1)
    recon_sq.append(((pr-d)**2).cpu())
    feat=torch.cat([Hi*Hj, (Hi-Hj).abs()],-1)
    split = 0 if k<16 else 1
    feats.append((feat.cpu(), split))
    dists.append((d.cpu(), split))
    allpairs_d.append(d.cpu())

allp=torch.cat(allpairs_d)
print(f"dist stats: mean {allp.mean():.2f} std {allp.std():.2f} Å  -> baseline(predict mean) RMSE {allp.std():.2f} Å")
recon=torch.cat(recon_sq)
print(f"model struct_recon_head RMSE = {recon.mean().sqrt():.2f} Å  (MSE {recon.mean():.1f} Å²)")

# quick MLP probe fit on split0, eval split1
Xtr=torch.cat([f for f,s in feats if s==0]).to(dev)
ytr=torch.cat([d for d,s in dists if s==0]).to(dev)
Xte=torch.cat([f for f,s in feats if s==1]).to(dev)
yte=torch.cat([d for d,s in dists if s==1]).to(dev)
din=Xtr.shape[1]
probe=torch.nn.Sequential(torch.nn.Linear(din,256),torch.nn.GELU(),torch.nn.Linear(256,1)).to(dev)
opt=torch.optim.Adam(probe.parameters(),lr=1e-3)
for ep in range(400):
    opt.zero_grad()
    p=probe(Xtr).squeeze(-1)
    loss=F.mse_loss(p,ytr); loss.backward(); opt.step()
with torch.no_grad():
    pte=probe(Xte).squeeze(-1)
    rmse=(pte-yte).pow(2).mean().sqrt()
    ss_res=(pte-yte).pow(2).sum(); ss_tot=(yte-yte.mean()).pow(2).sum()
    r2=1-ss_res/ss_tot
print(f"held-out MLP probe: RMSE {rmse:.2f} Å  R2 {r2:.3f}")
