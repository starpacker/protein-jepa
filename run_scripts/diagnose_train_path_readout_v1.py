"""Registered CPU first-four TRAIN oracle/raw/P32 attribution; zero updates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO/'results/hs_joint_20261101/train_path_readout_v1'
VERSION = 'train_path_readout_v1'
HISTORY = 'results/hs_joint_20261101/ordered_path_v1/protocol.json'
HISTORY_SHA = 'a31b87128702cd796b25fbbef60c3bd1245304cfd369c668970e54c72c7235cb'
EXECUTION = 'results/hs_joint_20261101/ordered_path_v1/marginal_path.execution.json'
EXECUTION_SHA = '97a4242b99b69b34699d92514c4b8a8ea2b6a311cc0fd81383c057d5c8355edf'
CHECKPOINT = 'artifacts/hs_joint_20261101/ordered_path_v1/marginal_path/checkpoint_latest.pt'
CHECKPOINT_SHA = '0f45200e331df6dc50c75cd0aec870dc965948fbbd4a5b430bd5d7ef2147886f'
WINDOW_LOG = 'artifacts/hs_joint_20261101/ordered_path_v1/marginal_path/training_windows.jsonl'
WINDOW_LOG_SHA = 'cbcad706d02645312d3a8209f0b18210a375998bb3b532a2f78788f24b8a71cb'
TRAIN_INPUTS = ('artifacts/mdcath_manifest_full/train.jsonl',
    'artifacts/stage_a_disp_relpos/checkpoint_latest.pt',
    'artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt')
SOURCES = ('run_scripts/diagnose_train_path_readout_v1.py',
    'tests/test_train_path_readout_v1.py', 'docs/TRAIN_PATH_READOUT_DIAGNOSTIC_V1_20261102.md')
ROLES = ('true_MD', 'true_MD_P32', 'oracle_raw', 'oracle_P32', 'forecast_raw', 'forecast_P32')
LAGS = (1, 2, 4, 8, 16, 32)
MAX_PAIRS = 512
DECODE_CHUNK = 64


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def now():
    return datetime.now(timezone.utc).isoformat()


def path(value):
    p = Path(value)
    p = (p if p.is_absolute() else REPO/p).resolve()
    require(p.is_relative_to(REPO) and p.name != 'test.jsonl', 'outside repository or sealed test input')
    return p


def sha(value):
    digest = hashlib.sha256()
    with path(value).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(file, value):
    text = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with Path(file).open('x') as stream:
        stream.write(text)


def file_stat(file):
    stat = Path(file).stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def snapshot_rng(generator):
    from protein_jepa_frame.training.lasp_run_state import rng_state
    source = rng_state()
    numpy = source['numpy']
    return dict(python=source['python'], numpy_metadata=[numpy[0], *numpy[2:]],
        numpy_keys=torch.from_numpy(numpy[1].copy()), torch=source['torch'],
        cuda=source['cuda'], sampling=generator.get_state().clone())


def tree_sha(value):
    """Hash nested replay tensors by dtype/shape/bytes and scalar metadata."""
    from protein_jepa_frame.utils.contracts import state_sha256
    flat = {}
    def visit(item, keys):
        if isinstance(item, dict):
            flat[json.dumps(keys+["@dict"])] = sorted(item)
            for key, child in item.items():
                visit(child, keys+[key])
        elif isinstance(item, (tuple,list)):
            flat[json.dumps(keys+["@sequence"])] = len(item)
            for i, child in enumerate(item):
                visit(child, keys+[i])
        else:
            flat[json.dumps(keys)] = item
    visit(value, [])
    return state_sha256(flat)


def verify_files(protocol):
    for name, digest in {**protocol['source_hashes'], **protocol['input_hashes']}.items():
        require(sha(name) == digest, 'frozen file changed: '+name)
    for name, digest in protocol['source_hashes'].items():
        require(sha(path(protocol['source_snapshot'])/name) == digest, 'source snapshot changed: '+name)
    for name, digest in protocol['metadata_snapshots'].items():
        require(sha(path(protocol['metadata_snapshot'])/name) == digest, 'metadata snapshot changed: '+name)
    for name, expected in protocol['selected_hdf5_stats'].items():
        require(file_stat(name) == expected, 'selected HDF identity changed: '+name)


def register():
    require(not OUT.exists(), 'refusing existing diagnostic directory')
    for name, digest in ((HISTORY,HISTORY_SHA), (EXECUTION,EXECUTION_SHA), (WINDOW_LOG,WINDOW_LOG_SHA)):
        require(sha(name) == digest, 'historical metadata changed: '+name)
    old = json.loads(path(HISTORY).read_text())
    execution = json.loads(path(EXECUTION).read_text())
    jobs = [j for j in execution['jobs'] if j['kind'] == 'train']
    audit = execution['checkpoint_audit']
    require(execution['status'] == 'completed' and len(jobs) == 1 and jobs[0]['returncode'] == 0
        and audit['step'] == 500 and audit['optimizer_steps'] == 500
        and execution['latest_sha256'] == CHECKPOINT_SHA, 'old completed training identity differs')
    selected = old['training_windows'][:4]
    require([r['step'] for r in selected] == [1,2,3,4] and
        [r['horizon'] for r in selected] == [1,4,16,64], 'first-four metadata differs')
    with path(WINDOW_LOG).open() as stream:
        historical = [json.loads(next(stream)) for _ in range(4)]
    require(all(all(actual[k] == v for k,v in row.items()) and len(actual['actual_batch_sha256']) == 64
        for row,actual in zip(selected,historical)), 'historical actual batch identity differs')
    inputs = {HISTORY:HISTORY_SHA, EXECUTION:EXECUTION_SHA, WINDOW_LOG:WINDOW_LOG_SHA,
              CHECKPOINT:CHECKPOINT_SHA}
    inputs.update({name:old['input_hashes'][name] for name in TRAIN_INPUTS})
    for name,digest in inputs.items():
        require(sha(name) == digest, 'input changed: '+name)
    sources = dict(old['source_hashes'])
    for name,digest in sources.items():
        require(sha(name) == digest and sha(path(old['source_snapshot'])/name) == digest,
                'historical source changed: '+name)
    sources.update({name:sha(name) for name in SOURCES})
    metadata = {name:inputs[name] for name in (HISTORY,EXECUTION,WINDOW_LOG)}
    stats = {r['record_path']:old['selected_hdf5_stats'][r['record_path']] for r in selected}
    for name,expected in stats.items():
        require(file_stat(name) == expected, 'selected HDF stat changed')
    protocol = dict(schema_version=VERSION,registered_utc=now(),source_hashes=sources,input_hashes=inputs,
        source_snapshot=str((OUT/'source_snapshot').relative_to(REPO)),
        metadata_snapshot=str((OUT/'metadata_snapshot').relative_to(REPO)),metadata_snapshots=metadata,
        selected_hdf5_stats=stats,selected_training_windows=selected,historical_actual_batches=historical,
        training_arguments=old['training_arguments']['marginal_path'],checkpoint=CHECKPOINT,
        checkpoint_sha256=CHECKPOINT_SHA,checkpoint_audit=audit,roles=list(ROLES),device='cpu',torch_threads=2,
        n_samples=12,seed=42,projection_steps=32,decode_chunk_size=DECODE_CHUNK,
        lags=list(LAGS),max_pairs=MAX_PAIRS,optimizer_steps=0,validation_loaded=False,test_opened=False,
        bioemu2_rerun=False,source_history=dict(protocol=HISTORY,sha256=HISTORY_SHA),
        hdf_evidence='size/mtime before and after plus exact selected batch tensor hash; no full-HDF digest',
        inference='four fixed TRAIN cases, one per horizon; no CI, across-horizon headline, information bound or timescale claim')
    OUT.mkdir(parents=True,exist_ok=False)
    for directory,files in (('source_snapshot',sources),('metadata_snapshot',metadata)):
        for name,digest in files.items():
            target = OUT/directory/name
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:
                stream.write(path(name).read_bytes())
            require(sha(target) == digest, 'file changed during snapshot')
    verify_files(protocol)
    write_new(OUT/'protocol.json',protocol)
    print(json.dumps(dict(protocol=str(OUT/'protocol.json'),sha256=sha(OUT/'protocol.json'))),flush=True)


@torch.no_grad()
def project_path(coordinates, anchor, mask, ri, ci, iterations=32):
    """Explicit projection, retaining supplied dtype and exact anchor at t0."""
    from protein_jepa_frame.models.ca_geometry import project_ca_bonds
    require(coordinates.ndim == 4 and coordinates.shape[-2:] == anchor.shape,
            'coordinate path shape differs')
    require(torch.isfinite(coordinates[...,mask,:]).all(), 'nonfinite coordinate path')
    members,frames,length,_ = coordinates.shape
    safe = torch.where(mask[None,None,:,None],coordinates,torch.zeros_like(coordinates))
    flat = safe[:,1:].reshape(-1,length,3)
    repeat = lambda x: x[None].expand(len(flat),*x.shape)
    output = safe.clone()
    output[:,1:] = project_ca_bonds(flat,repeat(anchor),repeat(mask),repeat(ri),repeat(ci),iterations).reshape(
        members,frames-1,length,3)
    output[:,0] = anchor
    return output


@torch.no_grad()
def decode_latents(bundle, latent, aa, ri, ci, rotation, anchor, mask):
    """Use adapter's batch order, including its separate endpoint batch."""
    require(latent.ndim == 4 and latent.shape[2] == len(mask) and latent.shape[1] >= 2,
            'latent path shape differs')
    require(torch.isfinite(latent[...,mask,:]).all(), 'nonfinite latent path')
    require(bundle.projection_steps == 32, 'requires frozen P32 component')
    members,frames,length,_ = latent.shape
    steps = frames-1
    raw = anchor.new_empty(members,frames,length,3)
    raw[:,0] = anchor
    def decode(member, frame):
        repeat = lambda x: x[None].expand(len(member),*x.shape)
        return bundle.decoder(latent[member,frame],latent[member,0],repeat(aa),repeat(ri),repeat(ci),
                              repeat(rotation),repeat(anchor),repeat(mask))
    raw[:,-1] = decode(torch.arange(members),torch.full((members,),steps,dtype=torch.long))
    for offset in range(0,members*(steps-1),DECODE_CHUNK):
        index = torch.arange(offset,min(offset+DECODE_CHUNK,members*(steps-1)))
        member,frame = index//(steps-1),index%(steps-1)+1
        raw[member,frame] = decode(member,frame)
    require(torch.isfinite(raw[...,mask,:]).all(), 'nonfinite raw decode')
    return raw,project_path(raw,anchor,mask,ri,ci)


def latent_descriptives(latent, mask):
    require(latent.ndim == 4 and latent.shape[2] == len(mask) and mask.dtype == torch.bool
            and mask.any(), 'latent support shape differs')
    require(torch.isfinite(latent[...,mask,:]).all(), 'nonfinite latent description')
    value = latent.double()
    mean = value.mean(-1)
    centered = value-mean[...,None]
    tensors = dict(channel_mean=mean,channel_variance=centered.square().mean(-1),
        centered_norm=centered.norm(dim=-1),increment_norm=value.diff(dim=1).norm(dim=-1))
    curves = {name:dict(mean_over_members_and_valid_residues=tensor[...,mask].mean((0,2)).tolist(),
        min_over_members_and_valid_residues=tensor[...,mask].amin((0,2)).tolist(),
        max_over_members_and_valid_residues=tensor[...,mask].amax((0,2)).tolist()) for name,tensor in tensors.items()}
    return tensors,dict(latent_dim=latent.shape[-1],valid_residues=int(mask.sum()),n_paths=len(latent),
        curves=curves,increment_time_axis='t=1..h; other curves t=0..h',
        interpretation='encoder LayerNorm support diagnostic only; no hard constraint in rollout, no information bound')


@torch.no_grad()
def score_roles(roles, reference, anchor, mask, ri, ci, times):
    from evaluation.geometry import ca_geometry_summary
    from evaluation.trajectory_metrics_v2 import ordered_path_summary
    require(tuple(roles) == ROLES, 'six ordered roles required')
    output = {}
    for name,coordinates in roles.items():
        require(torch.isfinite(coordinates[...,mask,:]).all(), 'nonfinite scored role: '+name)
        metrics = ordered_path_summary(coordinates,reference,anchor,mask,predicted_times_ps=times,
            reference_times_ps=times,reference_mask=mask[None].expand(len(times),-1),
            residue_index=ri,chain_id=ci,lags=LAGS,max_pairs=MAX_PAIRS)
        increments = coordinates.double().diff(dim=1)[...,mask,:]
        lengths = increments.norm(dim=-1).mean(-1).sum(-1)
        rmsd = increments.square().sum(-1).mean(-1).sqrt()
        geometry = [[ca_geometry_summary(frame.double(),mask,residue_index=ri,chain_id=ci)
                     for frame in member] for member in coordinates]
        output[name] = dict(n_paths=len(coordinates),coordinate_dtype=str(coordinates.dtype),
            ordered_metrics=metrics,anchor_gauge=dict(path_length_per_member=lengths.tolist(),
                increment_rmsd_per_member=rmsd.tolist(),increment_rmsd_mean_curve=rmsd.mean(0).tolist(),
                convention='no prediction alignment; t0 included; length=sum time mean residue displacement norm'),
            geometry_per_member_per_time=geometry,
            geometry_support='finite invalid geometry retained; no frame/residue removal; includes t0')
    return output


@torch.no_grad()
def diagnose_batch(model, batch, horizon, args, generator):
    from evaluation.ca_trajectory_adapter import sample_ca_trajectory
    from evaluation.geometry import kabsch_align
    from protein_jepa_frame.training.ordered_path_objective_v1 import _validate_batch
    from protein_jepa_frame.training.train_stage_b1a import _extract_backbone_frames
    from protein_jepa_frame.utils.contracts import state_sha256
    require(not any(module.training for module in model.modules()), 'requires eval mode')
    require(all(p.device.type == 'cpu' and p.grad is None for p in model.parameters()), 'CPU gradient-free model required')
    safe,mask_batch = _validate_batch(model,batch,horizon,args)
    require(len(mask_batch) == 1, 'one historical training window required')
    mask = mask_batch[0]
    aa,ri,ci = [safe[key][0] for key in ('aatype','residue_index','chain_id')]
    rotation,anchor_batch = _extract_backbone_frames(safe['coords'][:,3])
    anchor,rotation = anchor_batch[0],rotation[0]
    rng_before = snapshot_rng(generator)
    # One observed-only rollout is complete before any oracle target encoding.
    sampled = sample_ca_trajectory(model,safe['coords'][:,:4],mask_batch,
        *[safe[key] for key in ('aatype','residue_index','chain_id')],safe['temperature_k'],
        horizon,12,generator,decode_chunk_size=DECODE_CHUNK,context_dt_ps=float(model.kernel.cfg.step_ps))
    rng_forecast = snapshot_rng(generator)
    forecast_h = sampled.latent[0]
    require(forecast_h.shape[:3] == (12,horizon+1,len(mask)), 'forecast H shape differs')
    forecast_raw,forecast_p32 = decode_latents(model.bundle,forecast_h,aa,ri,ci,rotation,anchor,mask)
    # Matching raw-decoder batch order makes this an exact identity, not a fitted tolerance.
    require(torch.equal(forecast_p32[...,mask,:],sampled.ca[0,...,mask,:]), 'explicit P32 differs from frozen adapter')
    oracle_h = torch.stack([model.bundle.encode(safe['coords'][:,t],mask_batch,
        *[safe[key] for key in ('aatype','residue_index','chain_id')])[0] for t in range(3,4+horizon)])[None]
    require(torch.equal(oracle_h[0,0],forecast_h[0,0]) and
            torch.equal(forecast_h[:,0],oracle_h[:,0].expand_as(forecast_h[:,0])), 'oracle/forecast initial H differs')
    oracle_raw,oracle_p32 = decode_latents(model.bundle,oracle_h,aa,ri,ci,rotation,anchor,mask)
    reference = safe['coords'][0,3:,:,1].double().clone()
    reference[0] = anchor.double()
    aligned = torch.stack([kabsch_align(frame,anchor.double(),mask) for frame in reference])
    aligned[0] = anchor.double()
    # MD projection is a deterministic control in production float32, recorded explicitly.
    md_p32 = project_path(aligned.float()[None],anchor,mask,ri,ci)
    roles = dict(zip(ROLES,(aligned[None],md_p32,oracle_raw,oracle_p32,forecast_raw,forecast_p32)))
    times = safe['time_ps'][0,3:].double()-safe['time_ps'][0,3].double()
    rng_after = snapshot_rng(generator)
    require(tree_sha(rng_forecast) == tree_sha(rng_after), 'oracle/decoding consumed RNG')
    descriptors,latent_report = {},{}
    for name,value in (('true_H',oracle_h),('forecast_H',forecast_h)):
        descriptors[name],latent_report[name] = latent_descriptives(value,mask)
    saved = dict(schema_version=VERSION,batch={k:v.detach().cpu().clone() if torch.is_tensor(v) else v
        for k,v in batch.items()},roles=roles,true_H=oracle_h,forecast_H=forecast_h,
        forecast_S=None if sampled.bath is None else sampled.bath[0],reference_unaligned=reference,
        mask=mask,aatype=aa,residue_index=ri,chain_id=ci,anchor=anchor,rotation=rotation,
        time_ps=times,absolute_time_ps=safe['time_ps'][0],latent_descriptives=descriptors,
        rng_before=rng_before,rng_after_forecast=rng_forecast,rng_after=rng_after)
    return saved,dict(roles=score_roles(roles,reference,anchor.double(),mask,ri,ci,times),
        latent=latent_report,forecast_P32_exact_adapter_match=True,
        rng_sha256={key:tree_sha(saved[key]) for key in ('rng_before','rng_after_forecast','rng_after')},
        reference_alignment_dtype='torch.float64',MD_P32_projection_dtype='torch.float32')


def execute(protocol):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU diagnostic requires CUDA_VISIBLE_DEVICES empty")
    from dataclasses import asdict
    from protein_jepa_frame.training import train_lasp_ca_ordered_v1 as training
    from protein_jepa_frame.training.train_utils import _set_reproducibility
    from protein_jepa_frame.utils.contracts import state_sha256
    from scripts.eval_ca_endpoint_scores import load_model
    torch.set_num_threads(2)
    _set_reproducibility(42,False)
    args = SimpleNamespace(**{**protocol['training_arguments'],'output':str(OUT),'device':'cpu','torch_threads':2})
    checkpoint = torch.load(path(protocol['checkpoint']),map_location='cpu',weights_only=False)
    require(checkpoint['step'] == 500 and checkpoint['args']['coord_objective'] == 'marginal_path',
            'requires frozen old marginal latest500')
    model = load_model(str(path(protocol['checkpoint'])),checkpoint,'cpu').eval()
    audit = protocol['checkpoint_audit']
    require(state_sha256(model.kernel) == audit['kernel_tensor_sha256'] and
        asdict(model.kernel.cfg) == audit['kernel_config'], 'loaded kernel identity differs')
    require(model.bundle.projection_steps == 32 and not any(p.requires_grad for p in model.bundle.parameters()),
            'frozen P32 component required')
    initial = state_sha256(model)
    write_new(OUT/'initial.json',dict(model_sha256=initial,kernel_sha256=state_sha256(model.kernel),
        checkpoint_sha256=protocol['checkpoint_sha256'],device='cpu',torch_threads=torch.get_num_threads()))
    del checkpoint
    loaders = {h:training.make_loader(args,args.manifest,h,True) for h in args.horizons}
    generator = torch.Generator(device='cpu').manual_seed(42)
    rows = []
    for selected,historical in zip(protocol['selected_training_windows'],protocol['historical_actual_batches']):
        verify_files(protocol)
        step,horizon = selected['step'],selected['horizon']
        require(training.training_identity(loaders[horizon],step,42) == selected, 'selected training metadata differs')
        batch,actual = training.load_training_batch(loaders[horizon],step,42)
        require(actual == historical, 'actual TRAIN batch differs from old run')
        write_new(OUT/f'step{step:03d}.started.json',dict(step=step,horizon=horizon,batch=actual,started_utc=now()))
        started = time.monotonic()
        saved,report = diagnose_batch(model,batch,horizon,args,generator)
        sidecar = OUT/f'step{step:03d}.paths.pt'
        with sidecar.open('xb') as stream:
            torch.save(saved,stream)
        row = dict(step=step,horizon=horizon,batch=actual,seconds=time.monotonic()-started,**report,
            sidecar=str(sidecar.relative_to(REPO)),sidecar_sha256=sha(sidecar),
            replay_tensor_sha256=tree_sha(saved),model_sha256=state_sha256(model),optimizer_steps=0)
        write_new(OUT/f'step{step:03d}.json',row)
        require(row['model_sha256'] == initial and all(p.grad is None for p in model.parameters()),
                'model tensors or parameter gradients changed')
        verify_files(protocol)
        rows.append(row)
        print(json.dumps(dict(step=step,horizon=horizon,seconds=row['seconds'],status='completed')),flush=True)
    return dict(schema_version=VERSION,completed_utc=now(),rows=rows,optimizer_steps=0,
        initial_model_sha256=initial,final_model_sha256=state_sha256(model),device='cpu',torch_threads=2,
        torch=torch.__version__,inference=protocol['inference'],validation_loaded=False,test_opened=False,
        bioemu2_rerun=False,source_hashes=protocol['source_hashes'],input_hashes=protocol['input_hashes'])


def run():
    require((OUT/'protocol.json').exists(), 'register before run')
    require(not any((OUT/name).exists() for name in ('started.json','execution.json','result.json')), 'refusing rerun')
    started = time.monotonic()
    identity = dict(pid=os.getpid(),command=sys.argv,started_utc=now(),protocol_sha256=sha(OUT/'protocol.json'))
    write_new(OUT/'started.json',identity)
    try:
        protocol = json.loads((OUT/'protocol.json').read_text())
        require(protocol['schema_version'] == VERSION, 'protocol version differs')
        verify_files(protocol)  # Before any model/checkpoint deserialization or trajectory read.
        result = execute(protocol)
        verify_files(protocol)
        require(sha(OUT/'protocol.json') == identity['protocol_sha256'], 'protocol changed during run')
        write_new(OUT/'result.json',result)
    except BaseException as error:
        write_new(OUT/'execution.json',dict(identity,status='failed',exit_code=1,seconds=time.monotonic()-started,
            completed_utc=now(),error=f'{type(error).__name__}: {error}'))
        raise
    write_new(OUT/'execution.json',dict(identity,status='completed',exit_code=0,seconds=time.monotonic()-started,
        completed_utc=now(),output_sha256=sha(OUT/'result.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('register','run'),required=True)
    arguments = parser.parse_args()
    register() if arguments.mode == 'register' else run()
