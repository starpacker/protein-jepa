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
missing, unexpected = model.load_state_dict(sd, strict=False)
print('loaded. missing=%d unexpected=%d' % (len(missing), len(unexpected)))
model.eval()

ds = build_dataset(cfg.data, msm_labels_path=cfg.data.msm_labels_path)
loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_paired_windows, drop_last=True)
batch = next(iter(loader))
batch = {k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in batch.items()}

wm = model.lspwm.world_model
# encode a context to get q states
with torch.no_grad():
    out = model.forward_paired(batch)
cq = out.get('lspwm_context_sequence')
cm = out.get('lspwm_context_mask')
print('context_q', None if cq is None else tuple(cq.shape))
if cq is not None:
    q = cq.reshape(-1, cq.shape[-1])
    if cm is not None:
        q = cq[cm]
    q = q[:256]
    cond = None
    temp = torch.full((q.shape[0],), float(cfg.model.lspwm_reference_temperature_k) if hasattr(cfg.model,'lspwm_reference_temperature_k') else 348.0, device=dev)
    # mean force = -grad U
    with torch.no_grad():
        g = wm.potential.mean_force(q, cond, temp, create_graph=False)
        coef = wm.coefficients(q, temp)
        D = coef['diffusion']
    gnorm = g.norm(dim=-1)
    # drift = -D grad U ; here g already = mean_force = -gradU, so drift ~ D @ g
    drift = torch.einsum('bij,bj->bi', D, g)
    dnorm = drift.norm(dim=-1)
    print('||mean_force|| (=-gradU): mean=%.4e  std=%.4e  min=%.4e max=%.4e' % (gnorm.mean(), gnorm.std(), gnorm.min(), gnorm.max()))
    print('||D||_fro mean=%.4e' % D.reshape(D.shape[0],-1).norm(dim=-1).mean())
    print('||drift=D·(-gradU)|| mean=%.4e' % dnorm.mean())
    # variation across states: if force is identifiable, gnorm should VARY across q
    print('force-norm coefficient of variation (std/mean)=%.4f  <- ~0 means degenerate/constant' % (gnorm.std()/gnorm.mean().clamp(min=1e-12)))
