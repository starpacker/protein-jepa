"""Fresh bounded-capacity distance readout of two frozen raw-H encoders.

V2 repairs the explicit configuration lineage after the preserved v1 failure.
No dynamics, historical reconstruction head, H normalizer or cached H is used.
The caller must register inputs before invoking the explicit checkpoint loader.
Topology-defined pairs are supplied by the separately audited support helper.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn

from protein_jepa_frame.models.encoder import ProteinGeometryEncoder
from protein_jepa_frame.utils.contracts import state_sha256

VERSION = 'frozen_h_distance_probe_v2'
LATENT_DIM = 128
HIDDEN_DIM = 256


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def encoder_payload(checkpoint, kind, *, configuration_checkpoint=None):
    """Extract only the named context/embedded encoder, with no fallback."""
    require(kind in ('original','current'), 'unknown encoder lineage')
    if kind=='original':
        require(configuration_checkpoint is None, 'original encoder uses its own explicit configuration')
        prefix='context_encoder.'
        state={k[len(prefix):]:v for k,v in checkpoint['model'].items() if k.startswith(prefix)}
        config=checkpoint['stage_a_v2_config']
        source='model/context_encoder.*'
    else:
        component=checkpoint['component_checkpoint']
        state=dict(component['encoder'])
        require(configuration_checkpoint is not None, 'current encoder requires registered upstream configuration')
        require('stage_a_v2_config' not in component, 'unexpected alternate embedded architecture config')
        config=configuration_checkpoint['stage_a_v2_config']
        source='component_checkpoint.encoder'
    require(bool(state), 'empty explicitly selected encoder state')
    require(all(isinstance(v,torch.Tensor) and v.dtype==torch.float32
                and v.device.type=='cpu' and bool(torch.isfinite(v).all()) for v in state.values()),
            'encoder state must be finite CPU float32 tensors')
    require(not any(k.startswith('anchor_pos_proj.') for k in state)
            and not config.get('anchor_conditioned',False), 'undeclared anchor-conditioned encoder')
    required=('latent_dim','encoder_depth','num_heads','num_rbf','dropout','max_residues','identity_embed_scale')
    require(all(k in config for k in required), 'incomplete explicit encoder construction config')
    require(config['latent_dim']==LATENT_DIM and state['geom_proj.weight'].shape[0]==LATENT_DIM,
            'requires128-dimensional encoder')
    width=state['geom_proj.weight'].shape[1]
    require(width in (6,12), 'unknown encoder geometric feature width')
    actual_flags=dict(orientation_features=width==12,relative_position_features='relpos_proj.weight' in state)
    kwargs=dict(latent_dim=config['latent_dim'],depth=config['encoder_depth'],num_heads=config['num_heads'],
        num_rbf=config['num_rbf'],dropout=config['dropout'],max_residues=config['max_residues'],
        identity_embed_scale=config['identity_embed_scale'],anchor_conditioned=False,**actual_flags)
    require(all(type(kwargs[k]) is int and kwargs[k]>0 for k in ('latent_dim','depth','num_heads','num_rbf','max_residues')),
            'encoder dimensions must be positive integers')
    require(kwargs['num_rbf']>=2 and 0<=kwargs['dropout']<1
            and math.isfinite(float(kwargs['identity_embed_scale'])), 'invalid encoder numeric config')
    audit=dict(lineage=kind,state_source=source,actual_constructor=kwargs,
        saved_flags={k:config.get(k) for k in actual_flags},actual_flags=actual_flags,
        flag_rule='geom_proj input width12=>orientation,6=>no orientation; relpos_proj.weight presence=>relpos',
        encoder_state_sha256=state_sha256(state),historical_head_loaded=False,
        historical_struct_recon_dim=config.get('struct_recon_dim'),fresh_probe_hidden_dim=HIDDEN_DIM)
    return state,kwargs,audit


def load_frozen_encoder(path, *, kind, expected_sha256, configuration_checkpoint_path=None,
                        expected_configuration_sha256=None):
    """Run-stage only: exact upstream config lineage, current-only strict weights.

    Current component weights do not contain their architecture configuration.
    The frozen deployment loader constructs from calibration_args.stage_a_ckpt,
    then strictly replaces encoder weights. This function loads that explicitly
    supplied, hashed config checkpoint without constructing its model/decoder.
    """
    require(kind in ('original','current'), 'unknown encoder lineage')
    require(isinstance(expected_sha256,str) and len(expected_sha256)==64, 'explicit checkpoint SHA required')
    require(sha256_file(path)==expected_sha256, 'checkpoint hash differs before deserialization')
    if kind=='current':
        require(configuration_checkpoint_path is not None
                and isinstance(expected_configuration_sha256,str) and len(expected_configuration_sha256)==64,
                'current requires explicit configuration checkpoint path and SHA')
        require(sha256_file(configuration_checkpoint_path)==expected_configuration_sha256,
                'configuration checkpoint hash differs before deserialization')
    else:
        require(configuration_checkpoint_path is None and expected_configuration_sha256 is None,
                'original encoder configuration must come from its own checkpoint')
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    upstream=None;recorded={}
    if kind=='current':
        component=checkpoint['component_checkpoint']
        require(component.get('checkpoint_kind')=='ca_component_probe', 'current component kind differs')
        declared=component['calibration_args']['stage_a_ckpt']
        require(isinstance(declared,str) and Path(declared).resolve()==Path(configuration_checkpoint_path).resolve(),
                'calibration configuration path differs from registered lineage')
        recorded['calibration_args.stage_a_ckpt']=declared
        if 'stage_a_ckpt' in component:
            top=component['stage_a_ckpt']
            require(isinstance(top,str) and top==declared,
                    'component stage_a_ckpt disagrees with calibration lineage')
            recorded['component_checkpoint.stage_a_ckpt']=top
        upstream=torch.load(configuration_checkpoint_path,map_location='cpu',weights_only=False)
    state,kwargs,audit=encoder_payload(checkpoint,kind,configuration_checkpoint=upstream)
    config_path=path if kind=='original' else configuration_checkpoint_path
    config_sha=expected_sha256 if kind=='original' else expected_configuration_sha256
    audit.update(configuration_source='stage_a_v2_config' if kind=='original' else 'registered_upstream_stage_a.stage_a_v2_config',
        configuration_checkpoint_path=str(Path(config_path).resolve()),configuration_checkpoint_sha256=config_sha,
        recorded_config_lineage=recorded)
    # Instantiate only ProteinGeometryEncoder; preserve the caller's CPU RNG.
    with torch.random.fork_rng(devices=[]):
        encoder=ProteinGeometryEncoder(**kwargs)
    encoder.load_state_dict(state,strict=True)
    encoder.eval().requires_grad_(False)
    require(sha256_file(path)==expected_sha256, 'checkpoint changed during encoder loading')
    require(sha256_file(config_path)==config_sha, 'configuration checkpoint changed during encoder loading')
    require(state_sha256(encoder)==audit['encoder_state_sha256'], 'strict encoder tensor identity differs')
    audit.update(checkpoint_sha256=expected_sha256,encoder_tensor_sha256=state_sha256(encoder),
                 frozen=True,evaluation_mode=True,device='cpu',dtype='float32')
    return encoder,audit


def _mask(mask, length):
    require(isinstance(mask,torch.Tensor) and mask.dtype==torch.bool and mask.device.type=='cpu',
            'fixed observed mask must be CPU bool')
    if mask.shape==(1,length):
        mask=mask[0]
    require(mask.shape==(length,) and bool(mask.any()), 'invalid or empty fixed observed support')
    return mask


def _row_inputs(row):
    observed,target=row['observed_backbone'],row['target_backbone']
    require(isinstance(observed,torch.Tensor) and observed.ndim==4 and observed.shape[0]==1
            and observed.shape[-2:]==(4,3), 'observed backbone must be[1,L,4,3]')
    length=observed.shape[1]
    mask=_mask(row['observed_mask'],length)
    for name,coords,atom_key in [('observed',observed,'atom_mask'),('target',target,'target_atom_mask')]:
        require(isinstance(coords,torch.Tensor) and coords.shape==observed.shape
                and coords.dtype==torch.float32 and coords.device.type=='cpu', 'invalid '+name+' coordinate shape/dtype')
        atoms=row[atom_key]
        require(atoms.shape==(length,4) and atoms.dtype==torch.bool and atoms.device.type=='cpu'
                and bool(atoms[mask].all()), 'missing fixed observed-support atoms in '+name)
        require(bool(torch.isfinite(coords[:,mask]).all()), 'nonfinite valid '+name+' coordinates')
    ids=[]
    for field,padding in [('embedding_aatype',20),('embedding_residue_index',0),('embedding_chain_id',0)]:
        value=row[field]
        require(value.shape==(length,) and value.dtype==torch.long and value.device.type=='cpu', 'invalid '+field)
        if field=='embedding_aatype':
            require(bool(((value[mask]>=0)&(value[mask]<=20)).all()), 'invalid valid amino-acid embedding IDs')
        # Retain valid legacy IDs exactly; geometry IDs never replace them.
        ids.append(torch.where(mask,value,padding)[None])
    safe=[torch.where(mask[None,:,None,None],coords,0.) for coords in (observed,target)]
    return safe,mask,ids


@torch.no_grad()
def encode_pair(encoder,row):
    """Re-encode observed and ALIGNED target atoms; return two[1,L,128] tensors.

    Both single-frame calls share identical masking/sanitization. No cached
    oracle_H/anchor_H, target-dependent mask, H normalization or anchor feature
    is consulted. Missing target support fails rather than being intersected.
    """
    require(isinstance(encoder,ProteinGeometryEncoder) and not encoder.training
            and not any(m.training for m in encoder.modules())
            and not any(p.requires_grad for p in encoder.parameters()), 'encoder must be strictly frozen/eval')
    require(encoder.latent_dim==LATENT_DIM and not encoder.anchor_conditioned,
            'requires declared unanchored128D encoder')
    require(all(p.device.type=='cpu' and p.dtype==torch.float32 for p in encoder.parameters()),
            'encoder must be CPU float32')
    coords,mask,(aa,ri,ci)=_row_inputs(row)
    outputs=[]
    for frame in coords:
        value=encoder(frame[:,None],mask[None,None],aa,ri,ci,token_mask=mask[None,None])[:,0]
        require(value.shape==(1,len(mask),LATENT_DIM) and value.dtype==torch.float32
                and bool(torch.isfinite(value).all()), 'nonfinite or invalid raw encoder output')
        require(bool((value[:,~mask]==0).all()), 'encoder returned nonzero masked tokens')
        outputs.append(value.detach().clone())
    return tuple(outputs)


def _pairs(pair_indices,mask):
    require(isinstance(pair_indices,torch.Tensor) and pair_indices.dtype==torch.long
            and pair_indices.device.type=='cpu' and pair_indices.ndim==2 and pair_indices.shape[1]==2
            and len(pair_indices)>0, 'pairs must be nonempty CPU int64[P,2]')
    require(bool(((pair_indices>=0)&(pair_indices<len(mask))).all()), 'pair index out of bounds')
    i,j=pair_indices.T
    require(bool((i<j).all()) and bool(mask[pair_indices].all()), 'pairs must use distinct upper-triangle observed rows')
    codes=i*len(mask)+j
    require(bool((codes[1:]>codes[:-1]).all()), 'pairs must be unique lexicographic full-support indices')
    return i,j


class SymmetricDistanceProbe(nn.Module):
    """Fresh384→256→256→1 GELU MLP; average both pair orders in log space.

    Hidden256 is predeclared and differs from the historical hidden384 head.
    Every pair uses its own H_i,H_j and the same observed-mask H mean.
    """
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(3*LATENT_DIM,HIDDEN_DIM),nn.GELU(),
            nn.Linear(HIDDEN_DIM,HIDDEN_DIM),nn.GELU(),nn.Linear(HIDDEN_DIM,1))

    def forward(self,H,mask,pair_indices):
        require(H.ndim==3 and H.shape[0]==1 and H.shape[-1]==LATENT_DIM
                and H.dtype==torch.float32 and H.device.type=='cpu', 'H must be CPU float32[1,L,128]')
        mask=_mask(mask,H.shape[1]);i,j=_pairs(pair_indices,mask)
        require(bool(torch.isfinite(H[:,mask]).all()), 'nonfinite valid H')
        clean=torch.where(mask[None,:,None],H,0.)[0]
        pooled=clean.sum(0)/mask.sum()
        mean=pooled.expand(len(i),-1)
        first=self.net(torch.cat((clean[i],clean[j],mean),-1)).squeeze(-1)
        reverse=self.net(torch.cat((clean[j],clean[i],mean),-1)).squeeze(-1)
        prediction=(first+reverse)*.5
        require(bool(torch.isfinite(prediction).all()), 'nonfinite log-distance prediction')
        return prediction


def new_probe(seed=42):
    require(type(seed) is int and seed==42, 'only fixed seed42 probe initialization allowed')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        probe=SymmetricDistanceProbe()
    return probe.float().cpu()


def pair_log_targets(row,pair_indices):
    coords,mask,_=_row_inputs(row);i,j=_pairs(pair_indices,mask)
    distances=[torch.linalg.vector_norm(frame[0,i,1]-frame[0,j,1],dim=-1) for frame in coords]
    observed,target=distances
    require(bool(torch.isfinite(observed).all()) and bool(torch.isfinite(target).all()), 'nonfinite CA pair distance')
    return dict(target_distance=target.detach().clone(),target_log=torch.log1p(target).detach(),
                observed_distance=observed.detach().clone())


def probe_loss(probe,H,mask,pair_indices,target_log):
    prediction=probe(H,mask,pair_indices)
    require(target_log.shape==prediction.shape and target_log.dtype==torch.float32
            and target_log.device.type=='cpu' and bool(torch.isfinite(target_log).all())
            and bool((target_log>=0).all()), 'invalid fixed log1p-distance target')
    loss=(prediction-target_log).square().mean()
    require(bool(torch.isfinite(loss)), 'nonfinite per-protein log-MSE')
    return loss,prediction


def distance_metrics(predicted_distance,target_distance):
    """Float64 pair means, retaining negative predictions and rejecting overflow."""
    require(isinstance(predicted_distance,torch.Tensor) and predicted_distance.ndim==1
            and predicted_distance.numel()>0 and predicted_distance.is_floating_point()
            and target_distance.shape==predicted_distance.shape and target_distance.is_floating_point()
            and predicted_distance.device.type==target_distance.device.type=='cpu', 'invalid distance score vectors')
    pred,target=predicted_distance.detach().double(),target_distance.detach().double()
    require(bool(torch.isfinite(pred).all()) and bool(torch.isfinite(target).all())
            and bool((target>=0).all()), 'nonfinite distances or negative physical target')
    error=pred-target
    mse=error.square().mean();mae=error.abs().mean()
    require(bool(torch.isfinite(mse)) and bool(torch.isfinite(mae)), 'nonfinite distance errors')
    return dict(n_pairs=pred.numel(),mae_angstrom=float(mae),mse_angstrom2=float(mse),
                rmse_angstrom=float(mse.sqrt()),negative_count=int((pred<0).sum()))


def score_predictions(pred_log,target_log,*,target_distance):
    """Raw float32 log outputs→float64 expm1; raw physical labels, no clipping."""
    require(pred_log.ndim==1 and pred_log.numel()>0 and pred_log.dtype==torch.float32
            and target_log.shape==pred_log.shape and target_log.dtype==torch.float32
            and pred_log.device.type==target_log.device.type=='cpu'
            and bool(torch.isfinite(pred_log).all()) and bool(torch.isfinite(target_log).all()),
            'invalid raw float32 log prediction/target')
    require(target_distance.dtype==torch.float32 and target_distance.shape==target_log.shape
            and target_distance.device.type=='cpu' and torch.equal(torch.log1p(target_distance),target_log),
            'raw distance/log target linkage differs')
    metrics=distance_metrics(torch.expm1(pred_log.detach().double()),target_distance)
    log_mse=(pred_log.detach().double()-target_log.detach().double()).square().mean()
    require(bool(torch.isfinite(log_mse)), 'nonfinite log error')
    return dict(metrics,log_mse=float(log_mse))
