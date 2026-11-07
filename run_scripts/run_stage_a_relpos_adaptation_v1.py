"""Registered, single-attempt PCA/zero/local Stage A adaptation and readout.

Every phase has its own exclusive directory and linked actual execution record.
No phase resumes. All terminal adaptations and readout fits precede heldout
native-history access. Existing scientific inputs and source snapshots are read
only. The historical cache is a whole pickle (including unused H); only its
explicit geometry/identity fields are selected after deserialization.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data import frozen_h_distance_support_v2 as support_api
from protein_jepa_frame.data import stage_a_relpos_support_v1 as adapter
from protein_jepa_frame.data.backbone_topology_v2 import load_mdcath_backbone_topology
from protein_jepa_frame.data.dataset import TrajectoryClipDataset
from protein_jepa_frame.models.stage_a_relpos_checkpoint_v1 import (
    build_versioned_payload, load_versioned_payload, load_warm_start)
from protein_jepa_frame.models.stage_a_relpos_prediction_v1 import predict_endpoint
from protein_jepa_frame.models.stage_a_v2 import StageAV2Config
from protein_jepa_frame.utils.contracts import state_sha256
from scripts.prepare_backbone_readout_cache import clip_fingerprint, verify_clip

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'results/hs_joint_20261101/stage_a_relpos_adaptation_v1'
VERSION = 'stage_a_relpos_adaptation_trial_v1'
ARMS = ('pca', 'zero', 'local')
HORIZONS = (1, 4, 16, 64)
ORIGINAL = 'artifacts/stage_a_disp_relpos/checkpoint_latest.pt'
ORIGINAL_SHA = 'b2de4fed497c62cb196d908aa68a3a0d64a59e8c8179b9a91558797384b78689'
CURRENT = 'artifacts/hs_joint_20261101/ordered_path_v1/marginal_path/checkpoint_latest.pt'
CURRENT_SHA = '0f45200e331df6dc50c75cd0aec870dc965948fbbd4a5b430bd5d7ef2147886f'
ENCODER_SHA = '741d8fbf712981d8aefd4a4841d75ea201663c247dca92396e69f4bc43bbcc27'
CONFIG = 'artifacts/stage_a_disp_relpos/config.json'
CONFIG_SHA = '3de59859b19ced3f47867730bbfe082a32e01c39054c8bf63ec15b670acc5ea4'
CACHE = 'results/hs_joint_20261101/backbone_repair/readout_train500_v2.pt'
CACHE_SHA = 'd48faeb6e7385fa0b37a1daf440bf4708ed3472d83e2073b8a44d9073a216718'
SUPPORT = 'results/hs_joint_20261101/frozen_h_distance_support_v2/support.json'
SUPPORT_SHA = '262a9de49b9de46a514f165201c52681db6e56b936a6cd8cc20f4fb4a5fad1a3'
BIOEMU = 'results/lasp_v19/bioemu2_complete.json'
BIOEMU_SHA = 'fa5bb2d5b9831ca927d2d4aaae769727eddb9e51e4f8471f3f9fb609247ca4b5'
DOCUMENT = 'docs/STAGE_A_RELPOS_ADAPTATION_TRIAL_V1_20261102.md'
TEST = 'tests/test_stage_a_relpos_trial_v1.py'
CACHE_KEYS = ('metadata','observed_mask','embedding_aatype','embedding_residue_index',
    'embedding_chain_id','chemical_aatype','geometry_row_ids','peptide_edges','atom_mask',
    'target_atom_mask','observed_backbone','target_backbone','topology','fingerprints','boundary')
require = support_api.require
sha = support_api.sha256_file


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    raw = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with Path(path).open('x') as stream:
        stream.write(raw)


def save_tensor(path, value):
    with Path(path).open('xb') as stream:
        torch.save(value, stream)


def ref(path):
    return dict(path=str(Path(path).relative_to(ROOT)), sha256=sha(path))


def checked_ref(value):
    path = (ROOT/value['path']).resolve()
    require(path.is_relative_to(ROOT) and path.name != 'test.jsonl', 'invalid artifact path')
    require(sha(path) == value['sha256'], 'artifact bytes changed: '+str(path))
    return path


def training_api():
    return importlib.import_module('protein_jepa_frame.training.stage_a_relpos_adaptation_v1')


def readout_api():
    return importlib.import_module('protein_jepa_frame.training.stage_a_relpos_readout_v1')


def contract():
    return dict(schema_version=VERSION, arms=list(ARMS), seed=42,
        adaptation=dict(proteins=400, indices=list(range(400)), batch_size=4, updates=500,
            row_order=list(range(400))*5, accumulation=1, history=8, future_frames=1,
            horizons=list(HORIZONS), lags=[0], recent_frac=.5, optimizer='AdamW',
            lr=.0003, encoder_lr_scale=.3, weight_decay=.02, gradient_clip=1., warmup_steps=100,
            lr_schedule='frozen _lr_scale, zero-based steps0..499; last rate positive',
            weights=dict(future=1.,masked=.5,orth=.01,factor_activity=.01,encoder_variance=.01,
                temporal_variance=.05,temporal_min_std=.1,temporal_contrast=2.,
                temporal_contrast_rmsd_scale=2.,factor_min_std=.1,encoder_min_std=.1,
                lag_balanced=False,eps=1e-6,struct_recon=.3,displacement=1.,direction=1.),
            device='cuda:0', dtype='float32', amp=False, tf32=False,
            teacher='arm-local; current-online hard sync after full original load; EMAstep reset0',
            ema='one update after each optimizer update; all500 momentum.998',
            structural_pairs='historical512 requested ->510 actual; unchanged tensor-index sampler',
            ranking='unchanged margin.5Angstrom; log actual eligible adjacent-history pairs',
            selection='terminal online at update500 only; no resume, validation or early stopping'),
        readout=dict(proteins=400, updates=500, order=list(range(400))+list(range(100)),
            seed=42, optimizer='Adam', lr=.001, weight_decay=0., gradient_clip=1.,
            device='cpu',dtype='float32',hidden_dim=256, input='aligned cached target re-encoded online H',
            historical_auxiliary_head_used=False, normalize_H=False),
        score=dict(indices=list(range(400,500)),primary='future_H equal-protein log-MSE',
            contrasts=['zero-pca','local-pca'], secondary_roles=['observed_H_substitution','predicted_H','CA_persistence'],
            secondary_metrics=['mae_angstrom','mse_angstrom2','rmse_angstrom'],
            groups=['overall','h1','h4','h16','h64'],
            bootstrap=dict(replicates=10000,seed=20261102,stratified_by_horizon=True,
                paired=True,percentiles=[2.5,97.5],interpretation='descriptive; no multiplicity-adjusted claim'),
            predicted_H='native observed8 only; online anchor plus predictor delta through fixed OPF; no normalization/rebasing',
            predicted_H_alignment='none; no target coordinates or teacher in prediction',
            observed_H_batch_boundary='single-frame readout and 8-frame predictor may differ by FP32 batching; retain max difference'),
        phase_order=['smoke','prepare','train_pca','train_zero','train_local','fit_pca','fit_zero','fit_local','score'],
        budgets=dict(smoke_seconds=600,prepare_seconds=1800,train_seconds_per_arm=7200,
            fit_seconds_per_arm=1800,score_seconds=1800,host_rss_bytes=16*2**30,
            gpu_peak_bytes=70*2**30,output_bytes=8*2**30),
        limits='Single seed paired warm-start adaptation; PCA-pretrained predictor/projection/heads privilege PCA. '
            'Last100 already inspected TRAIN proteins, possibly seen in pretraining; held out only from new adaptation and head fitting. '
            'Arm-local latent losses cannot rank arms. Static/predicted endpoint readability is not StageB kinetics or BioEmu superiority. '
            'No assumed compatibility with existing StageB or decoders.',
        cache_policy='whole historical cache deserialized including unused H; only listed geometry/identity fields consumed',
        test_opened=False,bioemu2_rerun=False,stage_b_or_decoder_constructed=False)


def load_support():
    doc = support_api.load_support(ROOT/SUPPORT, SUPPORT_SHA)
    path = ROOT/SUPPORT
    protocol_path = path.parent/'protocol.json'
    require(sha(protocol_path) == doc['protocol_sha256'], 'support protocol bytes differ before parsing')
    protocol = json.loads(protocol_path.read_text())
    require(sha(protocol_path) == doc['protocol_sha256'], 'support protocol changed while parsing')
    return doc, protocol


def source_files():
    training_api(); readout_api()
    paths = {DOCUMENT,TEST,'run_scripts/run_stage_a_relpos_adaptation_v1.py',
        'tests/test_stage_a_relpos_adaptation_v1.py','tests/test_stage_a_relpos_readout_v1.py',
        'tests/test_stage_a_relpos_prediction_v1.py','tests/test_stage_a_relpos_support_v1.py',
        'tests/test_stage_a_relpos_checkpoint_v1.py','tests/test_stage_a_relpos_v1.py',
        'tests/test_relpos_mode_encoder_v1.py'}
    for module in tuple(sys.modules.values()):
        filename = getattr(module,'__file__',None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and path.is_relative_to(ROOT) and path.is_file():
                paths.add(str(path.relative_to(ROOT)))
    return sorted(paths)


def verify(protocol):
    require({k:protocol[k] for k in contract()} == contract(), 'trial contract differs')
    for path,digest in protocol['source_hashes'].items():
        require(sha(ROOT/path) == digest == sha(ROOT/protocol['source_snapshot']/path), 'source changed: '+path)
    for path,digest in protocol['input_hashes'].items():
        require(sha(ROOT/path) == digest, 'input changed: '+path)
    for path,stat in protocol['selected_hdf5_stats'].items():
        require(support_api.full_file_stat(path) == stat, 'HDF file stat changed: '+path)


def register():
    require(Path.cwd().resolve() == ROOT, 'run from repository root')
    OUTPUT.mkdir(parents=True,exist_ok=False)
    start = dict(pid=os.getpid(),command=sys.argv,started_utc=now()); begin=time.monotonic()
    status = dict(start)
    try:
        fixed = {ORIGINAL:ORIGINAL_SHA,CURRENT:CURRENT_SHA,CONFIG:CONFIG_SHA,
                 CACHE:CACHE_SHA,SUPPORT:SUPPORT_SHA,BIOEMU:BIOEMU_SHA}
        for path,digest in fixed.items(): require(sha(ROOT/path)==digest,'fixed input differs: '+path)
        support, support_protocol = load_support()
        plans = adapter.select_adaptation_plans(support,support_protocol)
        metadata_config = json.loads((ROOT/CONFIG).read_text())['stage_a_v2_config']
        require(set(metadata_config) == set(asdict(StageAV2Config())), 'incomplete original config metadata')
        inputs = dict(fixed)
        inputs.update({p:sha(ROOT/p) for p in support_protocol['input_hashes']})
        for name in ('protocol','started','execution'):
            path = str(Path(SUPPORT).with_name(name+'.json'));inputs[path]=sha(ROOT/path)
        snapshot = OUTPUT/'source_snapshot'
        sources={path:sha(ROOT/path) for path in source_files()}
        for path,digest in sources.items():
            target=snapshot/path;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream: stream.write((ROOT/path).read_bytes())
            require(sha(target)==digest,'source changed during snapshot')
        protocol=dict(contract(),registered_utc=now(),input_hashes=inputs,source_hashes=sources,
            source_snapshot=str(snapshot.relative_to(ROOT)),selected_hdf5_stats=support_protocol['selected_hdf5_stats'],
            stage_a_v2_config=metadata_config,fit_plans=plans,
            selected_metadata=[p['metadata'] for p in support['plans']],
            support_protocol_sha256=support['protocol_sha256'],
            checkpoint_deserialized_at_registration=False,coordinate_arrays_read_at_registration=False)
        verify(protocol);write_json(OUTPUT/'protocol.json',protocol)
        status.update(status='completed',returncode=0,protocol_sha256=sha(OUTPUT/'protocol.json'))
    except BaseException as error:
        status.update(status='failed',returncode=1,error=f'{type(error).__name__}: {error}');raise
    finally:
        status.update(completed_utc=now(),seconds=time.monotonic()-begin)
        write_json(OUTPUT/'registration.execution.json',status)
    return status


def completed(phase, protocol_sha):
    folder=OUTPUT/phase
    execution=json.loads((folder/'execution.json').read_text())
    started=json.loads((folder/'started.json').read_text())
    require(execution['status']=='completed' and execution['returncode']==0
        and execution['protocol_sha256']==protocol_sha
        and all(execution[k]==started[k] for k in ('pid','command','started_utc','protocol_sha256','phase')),
        'prerequisite phase did not complete: '+phase)
    require(execution['phase']==phase and execution['pid']>0, 'prerequisite phase identity differs')
    require(datetime.fromisoformat(execution['started_utc'])<=datetime.fromisoformat(execution['completed_utc'])
        and execution['seconds']>=0,'prerequisite execution time differs')
    result_path=checked_ref(execution['result'])
    require(result_path==folder/'result.json','prerequisite result path differs')
    result=json.loads(result_path.read_text())
    require(result['phase']==phase and result['protocol_sha256']==protocol_sha
        and result['status']=='completed','result linkage differs')
    for value in result.get('artifacts',{}).values(): checked_ref(value)
    if phase.startswith('train_'):
        manifest=json.loads(checked_ref(result['artifacts']['draw_manifest']).read_text())
        require(len(manifest)==500 and len({v['path'] for v in manifest})==500,'draw tape inventory differs')
        for step,value in enumerate(manifest,1):
            require(checked_ref(value)==folder/'draws'/f'{step:04d}.pt','draw tape path differs')
    if phase=='score':
        for row in result['per_protein']: checked_ref(row['predictions'])
    return result


def prerequisites(phase, protocol_sha, *, phase_started=None):
    order=contract()['phase_order']; require(phase in order,'unknown phase')
    result={};previous_end=None
    for name in order[:order.index(phase)]:
        result[name]=completed(name,protocol_sha)
        execution=json.loads((OUTPUT/name/'execution.json').read_text())
        start=datetime.fromisoformat(execution['started_utc']);end=datetime.fromisoformat(execution['completed_utc'])
        require(previous_end is None or previous_end<=start,'phase completion order differs')
        previous_end=end
    if phase_started is not None and previous_end is not None:
        require(previous_end<=datetime.fromisoformat(phase_started),'phase started before prerequisites completed')
    return result


def budget(protocol,phase,begin):
    b=protocol['budgets'];kind=phase.split('_')[0]
    seconds=b.get(kind+'_seconds_per_arm',b.get(kind+'_seconds'))
    require(time.monotonic()-begin <= seconds,'phase wall-clock budget exceeded')
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    require(rss <= b['host_rss_bytes'],'host memory budget exceeded')
    gpu=0
    if torch.cuda.is_initialized():
        gpu=torch.cuda.max_memory_allocated(0)
        require(gpu <= b['gpu_peak_bytes'],'GPU memory budget exceeded')
    return dict(elapsed_seconds=time.monotonic()-begin,host_peak_rss_bytes=rss,gpu_peak_allocated_bytes=gpu)


def backend(gpu=False):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    require(not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('cuda'), 'AMP forbidden')
    if gpu:
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and torch.cuda.device_count()==1,
            'GPU phases require physicalGPU0 exclusively visible')
        torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats(0)
    else:
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU phases require hidden GPUs')


def fresh_model(arm):
    model,audit=load_warm_start(ROOT/ORIGINAL,original_sha256=ORIGINAL_SHA,
        current_path=ROOT/CURRENT,current_sha256=CURRENT_SHA,feature_mode=arm)
    require(state_sha256(model.context_encoder)==ENCODER_SHA,'current encoder identity differs')
    require(asdict(model.config)==json.loads((OUTPUT/'protocol.json').read_text())['stage_a_v2_config'],
        'checkpoint config differs from preregistered metadata')
    return model,audit


def load_cache(protocol):
    require(sha(ROOT/CACHE)==CACHE_SHA,'cache changed before load')
    cache=torch.load(ROOT/CACHE,map_location='cpu',weights_only=False)
    require(sha(ROOT/CACHE)==CACHE_SHA,'cache changed during load')
    expected=json.loads((ROOT/Path(CACHE).with_suffix('.protocol.json')).read_text())['protocol']
    require(cache['schema_version']=='backbone_readout_cache_v1' and cache['protocol']==expected,
        'cache protocol differs')
    require(expected['mode']=='train' and expected['dataset_role']=='training_readout_calibration'
        and expected['n_windows']==500 and expected['n_forecast_samples']==expected['optimizer_steps']==0
        and expected['test_opened'] is False,'cache scope differs')
    completion=cache['completion']
    require(completion['frozen_inputs_verified'] is True and completion['model_sha256_before']==
        completion['model_sha256_after']==expected['model_tensor_sha256'],'cache incomplete')
    require(len(cache['rows'])==500 and [row['metadata'] for row in cache['rows']]==protocol['selected_metadata'],
        'cache panel/order differs')
    rows=[{k:row[k] for k in CACHE_KEYS} for row in cache['rows']]
    del cache
    return rows


def read_native(window, *, endpoint):
    spec=adapter._window_spec(window)
    topology=load_mdcath_backbone_topology(spec.record)
    topology=topology.subset(torch.arange(min(96,topology.n_residues)))
    reader=TrajectoryClipDataset([spec.record],DataConfig(num_frames=8,max_residues=96))
    context=reader._read_clip(spec.record,spec.context_frames)
    verify_clip(context,topology,spec)
    target=None
    if endpoint:
        target=reader._read_clip(spec.record,np.asarray([spec.target_frame],dtype=np.int64))
        verify_clip(target,topology,spec,endpoint=True)
    return context,target,topology,spec


def prepare(protocol, folder, prior, check):
    support,_=load_support();cached=load_cache(protocol);rows=[];audits=[]
    for index,plan in enumerate(protocol['fit_plans']):
        require(index<400 and plan['cache_plan']['cache_row_index']==index,'fit400 order differs')
        context,target,topology,spec=read_native(plan['window'],endpoint=True)
        require(clip_fingerprint(context)==cached[index]['fingerprints']['observation']
            and clip_fingerprint(target)==cached[index]['fingerprints']['raw_label'],
            'actual native clips differ from registered cache fingerprints')
        support_api.validate_cached_row(cached[index],support['plans'][index])
        row=adapter.prepare_adaptation_row(plan,context,target,topology)
        rows.append(row);audits.append(row['adaptation_audit'])
        write_json(folder/f'row_{index:03d}.json',dict(metadata=spec.metadata('train'),**row['adaptation_audit']))
        check()
        if (index+1)%20==0:print(f'prepare {index+1}/400',flush=True)
    require(len(rows)==400,'incomplete adaptation panel')
    path=folder/'fit400.pt';save_tensor(path,dict(schema_version=VERSION,rows=rows,rows_sha256=adapter.fingerprint(rows)))
    return dict(artifacts=dict(rows=ref(path)),n_fit_rows=400,heldout_history_loaded=False,
        audits_sha256=support_api.canonical_sha256(audits),cached_H_deserialized_but_unused=True)


def smoke(protocol,folder,prior,check):
    audits={}
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    def factory(arm):
        model,audits[arm]=fresh_model(arm)
        return model.to('cuda:0')
    result=training_api().smoke_all_arms(factory)
    check();path=folder/'smoke.pt';save_tensor(path,result)
    return dict(artifacts=dict(smoke=ref(path)),warm_starts=audits,
        updates=0,actual_training_rows_read=False,smoke_state_reused=False)


def train(protocol,folder,prior,check,arm):
    data=torch.load(checked_ref(prior['prepare']['artifacts']['rows']),map_location='cpu',weights_only=False)
    rows=data['rows'];require(len(rows)==400 and adapter.fingerprint(rows)==data['rows_sha256'],'fit data changed')
    model,audit=fresh_model(arm);model=model.to('cuda:0')
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    initial_state=state_sha256(model);initial_buffers=state_sha256(dict(model.named_buffers()))
    logpath=folder/'updates.jsonl';draws=folder/'draws';draws.mkdir()
    audit_hashes=[];draw_manifest=[]
    with logpath.open('x') as stream:
        def emit(record):
            record=dict(record);tensors=record.pop('tensor_audit')
            step=record['step'];path=draws/f'{step:04d}.pt';save_tensor(path,tensors)
            digest=adapter.fingerprint(tensors);audit_hashes.append(digest)
            record['tensor_audit']=ref(path);record['tensor_audit_sha256']=digest
            draw_manifest.append(record['tensor_audit'])
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();check()
            if len(audit_hashes)%25==0:print(f'{arm} updates {len(audit_hashes)}/500',flush=True)
        def batch(indices):
            require(len(indices)==4 and all(type(i)is int and 0<=i<400 for i in indices),'adaptation batch escapes400')
            paired=adapter.collate_adaptation_rows([rows[i] for i in indices])
            return {key:value.to('cuda:0') if torch.is_tensor(value) else value for key,value in paired.items()}
        try:
            optimizer=training_api().run_fixed_training(model,batch,emit)
        except BaseException as error:
            # Raw failure snapshots deliberately accept nonfinite tensors; do
            # not run the successful-checkpoint validator on a failed model.
            evidence_errors=[]
            try:
                save_tensor(folder/'failed_weights.pt',dict(feature_mode=arm,config=asdict(model.config),
                    model_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    named_buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()}))
            except BaseException as save_error:
                evidence_errors.append('weights: '+repr(save_error))
            try:
                if hasattr(error,'adaptation_failure_payload'):
                    save_tensor(folder/'failed_optimizer.pt',error.adaptation_failure_payload)
            except BaseException as save_error:
                evidence_errors.append('optimizer: '+repr(save_error))
            if evidence_errors:
                print(json.dumps(dict(failure_evidence_errors=evidence_errors)),flush=True)
            raise
    require(len(audit_hashes)==500 and adapter.fingerprint(rows)==data['rows_sha256'],'incomplete updates or mutated inputs')
    require(int(model._ema_step)==500,'EMA count differs')
    path=folder/'terminal500.pt';save_tensor(path,build_versioned_payload(model,readout_encoder='online'))
    optimizer_path=folder/'optimizer500.pt';save_tensor(optimizer_path,optimizer.state_dict())
    manifest_path=folder/'draws.manifest.json';write_json(manifest_path,draw_manifest)
    require(all(float(state['step'])==500 for state in optimizer.state.values()),'optimizer count differs')
    training=dict(optimizer='AdamW',updates=500,ema_updates=500,optimizer_state_entries=len(optimizer.state),
        config=training_api().contract(),terminal_state_sha256=state_sha256(model))
    # Exact stochastic tape equality is tested across the fully executed arms.
    for previous in ARMS[:ARMS.index(arm)]:
        require(prior['train_'+previous]['draw_sha256_by_step']==audit_hashes,'matched stochastic draws/RNG differ across arms')
        require(prior['train_'+previous]['initial_state_sha256']==initial_state
            and prior['train_'+previous]['initial_buffers_sha256']==initial_buffers,'warm-start tensors differ across arms')
    return dict(artifacts=dict(checkpoint=ref(path),updates=ref(logpath),optimizer=ref(optimizer_path),
        draw_manifest=ref(manifest_path)),warm_start=audit,training=training,
        initial_state_sha256=initial_state,initial_buffers_sha256=initial_buffers,
        draw_sha256_by_step=audit_hashes,terminal_encoder_sha256=state_sha256(model.context_encoder))


def terminal_model(protocol,prior,arm):
    artifact=prior['train_'+arm]['artifacts']['checkpoint'];path=checked_ref(artifact)
    model,audit=load_versioned_payload(path,expected_sha256=artifact['sha256'],expected_feature_mode=arm,
        expected_config=StageAV2Config(**protocol['stage_a_v2_config']),expected_readout_encoder='online')
    require(int(model._ema_step)==500,'terminal model update differs')
    model.eval().requires_grad_(False);return model,audit


def fit(protocol,folder,prior,check,arm):
    api=readout_api();support,_=load_support()
    prepared=api.prepare_readout_rows(load_cache(protocol),support)
    model,model_audit=terminal_model(protocol,prior,arm)
    features,encoding_audit=api.encode_readout_rows(model,prepared[:400])
    logpath=folder/'updates.jsonl'
    with logpath.open('x') as stream:
        def emit(record):
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();check()
        try:
            head,audit,payload=api.fit_readout(prepared[:400],features,emit)
        except BaseException as error:
            if hasattr(error,'readout_failure_payload'):
                try: save_tensor(folder/'failed_readout.pt',error.readout_failure_payload)
                except BaseException as save_error:
                    print(json.dumps(dict(failure_evidence_errors=[repr(save_error)])),flush=True)
            raise
    path=folder/'readout500.pt';save_tensor(path,payload)
    for previous in ARMS[:ARMS.index(arm)]:
        require(prior['fit_'+previous]['readout']['initial_head_sha256']==audit['initial_head_sha256'],
            'fresh head initialization differs')
    return dict(artifacts=dict(head=ref(path),updates=ref(logpath)),model=model_audit,
        encoding=encoding_audit,readout=audit,heldout_history_loaded=False)


def observed_history(index, support_protocol, plan, cached):
    require(400<=index<500 and plan['split']=='readout_heldout','score history must be last100 only')
    window=support_protocol['windows'][index]
    require(window['cache_row_index']==index and window['split']=='readout_heldout','score history identity differs')
    context,target,topology,spec=read_native(window,endpoint=False)
    require(target is None and clip_fingerprint(context)==cached['fingerprints']['observation'],
        'observed history differs from fixed original cache')
    require(support_api.canonical_metadata(topology.as_metadata())==plan['cache_topology'],'heldout topology differs')
    mask=cached['observed_mask']
    require(torch.equal(context['residue_mask'],mask[None].expand(8,-1)),'heldout history support differs')
    coords=torch.where(mask[None,:,None,None],context['coords'],0.)
    inputs=dict(history_coords=coords[None],history_mask=context['residue_mask'][None],
        aatype=context['aatype'][None],residue_index=context['residue_index'][None],chain_id=context['chain_id'][None],
        history_time_ps=context['time_ps'][None],
        endpoint_time_ps=torch.tensor([spec.record.first_frame_time_ps+spec.target_frame*1000.],dtype=torch.float32),
        horizon_frames=torch.tensor([spec.horizon]),temperature_k=context['temperature_k'][None])
    return inputs,dict(cache_row_index=index,observation_sha256=clip_fingerprint(context),
        input_sha256=adapter.fingerprint(inputs),context_frames=spec.context_frames.tolist(),
        endpoint_coordinates_read=False,topology_sha256=topology.topology_sha256)


def score(protocol,folder,prior,check):
    # prerequisites() has verified every fit completion and artifact before entry.
    api=readout_api();support,support_protocol=load_support();cached=load_cache(protocol)
    prepared=api.prepare_readout_rows(cached,support);models={};heads={};features={};identities={}
    from protein_jepa_frame.models.stage_c.frozen_h_distance_probe_v2 import new_probe
    for arm in ARMS:
        models[arm],model_audit=terminal_model(protocol,prior,arm)
        features[arm],encoding=api.encode_readout_rows(models[arm],prepared[400:])
        payload=torch.load(checked_ref(prior['fit_'+arm]['artifacts']['head']),map_location='cpu',weights_only=False)
        heads[arm]=new_probe(seed=42);heads[arm].load_state_dict(payload['head'],strict=True);heads[arm].eval()
        require(state_sha256(heads[arm])==prior['fit_'+arm]['readout']['final_head_sha256'],'readout state differs')
        identities[arm]=dict(model=model_audit,encoding=encoding,model_state=state_sha256(models[arm]),
            model_buffers=state_sha256(dict(models[arm].named_buffers())),head_state=state_sha256(heads[arm]))
    records=[];history_audits=[]
    for index in range(400,500):
        inputs,history_audit=observed_history(index,support_protocol,support['plans'][index],cached[index])
        scores={};arrays={};pred_audits={}
        for arm in ARMS:
            prediction=predict_endpoint(models[arm],feature_mode=arm,**inputs)
            scored=api.score_readout(heads[arm],features[arm][index-400],prepared[index],prediction['H_predicted'])
            for role,value in scored['scores'].items():
                key='CA_persistence' if role=='CA_persistence' else arm+'/'+role
                if key in scores: require(scores[key]==value,'persistence changed across arms')
                scores[key]=value
            for key,value in scored['arrays'].items():
                arrays[arm+'__'+key]=np.asarray(value)
            for key in ('H_predicted','H_observed','pred_factors'):
                arrays[arm+'__'+key]=prediction[key].detach().numpy().copy()
            valid=prepared[index]['mask']
            anchor_difference=(prediction['H_observed']-features[arm][index-400]['observed_H'])[:,valid]
            pred_audits[arm]=dict(prediction['audit'],single_vs_history_observed_H_max_abs=float(anchor_difference.abs().max()))
        require(adapter.fingerprint(inputs)==history_audit['input_sha256'],'prediction changed observed inputs')
        path=folder/f'heldout_{index:03d}.npz'
        require(all(not value.dtype.hasobject for value in arrays.values()),'object arrays prohibited')
        with path.open('xb') as stream: np.savez(stream,**arrays)
        record=dict(cache_row_index=index,metadata=cached[index]['metadata'],n_pairs=len(prepared[index]['pairs']),scores=scores,
            predictions=ref(path),history=history_audit,prediction_audits=pred_audits)
        write_json(folder/f'heldout_{index:03d}.json',record);records.append(record);history_audits.append(history_audit)
        check()
        if (index+1)%20==0: print(f'score {index-399}/100',flush=True)
    for arm in ARMS:
        require(state_sha256(models[arm])==identities[arm]['model_state']
            and state_sha256(dict(models[arm].named_buffers()))==identities[arm]['model_buffers']
            and state_sha256(heads[arm])==identities[arm]['head_state'],'scoring changed learned tensors')
    return dict(artifacts={},identities=identities,per_protein=records,
        aggregates=api.aggregate_readouts(records),paired_bootstrap=api.paired_bootstrap(records),
        heldout_history_read_after_all_fits=True,history_audit_sha256=support_api.canonical_sha256(history_audits),
        no_cached_H_used=True,no_stage_b_or_decoder_execution=True)


def run_phase(phase):
    protocol_path=OUTPUT/'protocol.json';protocol_sha=sha(protocol_path)
    protocol=json.loads(protocol_path.read_text())
    registration=json.loads((OUTPUT/'registration.execution.json').read_text())
    require(registration['status']=='completed' and registration['returncode']==0
        and registration['protocol_sha256']==protocol_sha,'registration incomplete')
    folder=OUTPUT/phase;folder.mkdir(exist_ok=False)
    started=dict(pid=os.getpid(),command=sys.argv,started_utc=now(),protocol_sha256=protocol_sha,phase=phase)
    write_json(folder/'started.json',started);status=dict(started);begin=time.monotonic()
    try:
        verify(protocol);prior=prerequisites(phase,protocol_sha,phase_started=started['started_utc'])
        backend(gpu=phase=='smoke' or phase.startswith('train_'))
        check=lambda:budget(protocol,phase,begin)
        if phase in ('prepare','smoke','score'):result=globals()[phase](protocol,folder,prior,check)
        else:
            kind,arm=phase.split('_');result=globals()[kind](protocol,folder,prior,check,arm)
        resources=check();verify(protocol);require(sha(protocol_path)==protocol_sha,'protocol changed')
        require(prerequisites(phase,protocol_sha,phase_started=started['started_utc'])==prior,
            'prerequisite phase evidence changed during execution')
        resources=check()
        output_bytes=sum(path.stat().st_size for path in OUTPUT.rglob('*') if path.is_file())
        require(output_bytes<=protocol['budgets']['output_bytes'],'output budget exceeded')
        result=dict(result,phase=phase,protocol_sha256=protocol_sha,resources=resources,
            test_opened=False,bioemu2_rerun=False,status='completed')
        write_json(folder/'result.json',result)
        status.update(status='completed',returncode=0,result=ref(folder/'result.json'))
    except BaseException as error:
        status.update(status='failed',returncode=1,error=f'{type(error).__name__}: {error}');raise
    finally:
        status.update(completed_utc=now(),seconds=time.monotonic()-begin)
        write_json(folder/'execution.json',status)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=('register',*contract()['phase_order']))
    args=parser.parse_args()
    result=register() if args.phase=='register' else run_phase(args.phase)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()
