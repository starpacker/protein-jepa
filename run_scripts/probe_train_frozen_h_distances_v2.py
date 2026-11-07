"""Fixed TRAIN500 frozen-encoder distance readability probe; register before run.

Only the fresh readout is fitted. The last100 proteins are held out from this
readout fit, not necessarily from either encoder's training. No dynamics run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from protein_jepa_frame.utils.contracts import state_sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = 'results/hs_joint_20261101/frozen_h_distance_probe_v2'
VERSION = 'train_frozen_h_distance_probe_v2'
CACHE = 'results/hs_joint_20261101/backbone_repair/readout_train500_v2.pt'
CACHE_SHA = 'd48faeb6e7385fa0b37a1daf440bf4708ed3472d83e2073b8a44d9073a216718'
SELECTION = str(Path(CACHE).with_suffix('.selection.json'))
CACHE_PROTOCOL = str(Path(CACHE).with_suffix('.protocol.json'))
CHECKPOINTS = dict(original='artifacts/stage_a_disp_relpos/checkpoint_latest.pt',
    current='artifacts/hs_joint_20261101/spatial_probe_v1/spatial_exact.pt')
CHECKPOINT_SHAS = dict(original='b2de4fed497c62cb196d908aa68a3a0d64a59e8c8179b9a91558797384b78689',
    current='396e6e4df35019140324484bf8d560a292cf2688a11946f2e0ccde5cd6ea8dda')
SUPPORT = 'results/hs_joint_20261101/frozen_h_distance_support_v2/support.json'
SUPPORT_SHA = '262a9de49b9de46a514f165201c52681db6e56b936a6cd8cc20f4fb4a5fad1a3'
PRIOR_FAILURE_INPUTS = {'results/hs_joint_20261101/frozen_h_distance_probe_v1/protocol.json': 'c64575df00410f8c4d2386438d5d13212597d6578219545b0ebf2c0c01b2e257',
 'results/hs_joint_20261101/frozen_h_distance_probe_v1/started.json': '4a44ebda5eff1e9cfc7f9d879d172dd9bf6437374c4cff777ebc322fc24b2fa5',
 'results/hs_joint_20261101/frozen_h_distance_probe_v1/execution.json': '86b245407699ce17284285d09ef224afb6b18262f586cfb4e3834a03b90aa200',
 'run_scripts/probe_train_frozen_h_distances_v1.py': '2439a315f69d4f676ba3f65bbee6e04ec24a589ec472cc5eee211581b9108f5d',
 'tests/test_train_frozen_h_distance_probe_v1.py': 'a66bf091689a4c2f2d13db8d821315272f2bf74497e2bec3e32c2a0d9981bd63',
 'docs/TRAIN_FROZEN_H_DISTANCE_PROBE_V1_20261102.md': '200adf5475bb8e7b78bc40e6eb20ecc0b067c68175dbd8cd2f2604fb9d38af50',
 'protein_jepa_frame/models/stage_c/frozen_h_distance_probe_v1.py': '34c53424f04f40128ae0d28a462286c69689efa28f9183cff4c9cea5c2ed2de1',
 'tests/test_frozen_h_distance_probe_v1.py': '5277a8ff05b080945f92f1d62f08ed10b49e39a0ff69e478ec935d658c122488',
 'results/hs_joint_20261101/frozen_h_distance_probe_v1/all500_support_audit.json': 'b41f526ea5af582010fadf09070eb2fa9c3a31a871afa9ae91cbb3b8a05b2b59',
 'results/hs_joint_20261101/frozen_h_distance_probe_v1/original.encoded_identity.json': '025c70790cb8ddf845add99ff329378013aa0a827b5735b7500bbc3e583201a2',
 'results/hs_joint_20261101/frozen_h_distance_probe_v1/original.training.jsonl': 'ce462141e4b9a8ac63127f8cb3d05cd1b987b9b377f74eee8a9a6467363ac51d',
 'results/hs_joint_20261101/frozen_h_distance_probe_v1/original.readout_latest500.pt': '90f0f989f208c08841d8254fce8483abd4117150f7f744a4e2400c0294b6a0ed'}
ARMS = ('original','current')
INPUT_ROLES = ('target_H','observed_H_substitution')
HORIZONS = (1,4,16,64)
TRAIN_ORDER = tuple(range(400))+tuple(range(100))
SCRIPT = 'run_scripts/probe_train_frozen_h_distances_v2.py'
TEST = 'tests/test_train_frozen_h_distance_probe_v2.py'
DOCUMENT = 'docs/TRAIN_FROZEN_H_DISTANCE_PROBE_V2_20261102.md'
PROBE_MODULE = 'protein_jepa_frame.models.stage_c.frozen_h_distance_probe_v2'
SUPPORT_MODULE = 'protein_jepa_frame.data.frozen_h_distance_support_v2'
SOURCES = (SCRIPT,TEST,DOCUMENT,
    'protein_jepa_frame/models/stage_c/frozen_h_distance_probe_v2.py',
    'tests/test_frozen_h_distance_probe_v2.py',
    'protein_jepa_frame/data/frozen_h_distance_support_v2.py',
    'run_scripts/prepare_frozen_h_distance_support_v2.py',
    'tests/test_frozen_h_distance_support_v2.py',
    'docs/FROZEN_H_DISTANCE_SUPPORT_V2_20261102.md',
    'protein_jepa_frame/utils/contracts.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def local(value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT/path).resolve()
    require(path.is_relative_to(ROOT) and path.name!='test.jsonl', 'outside repository or sealed test input')
    return path


def sha(value):
    digest=hashlib.sha256()
    with local(value).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):digest.update(chunk)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_json(path,value):
    encoded=json.dumps(value,indent=2,allow_nan=False)+'\n'
    with local(path).open('x') as stream:stream.write(encoded)


def save_tensor(path,value):
    with local(path).open('xb') as stream:torch.save(value,stream)


def save_npz(path,values):
    require(all(isinstance(v,np.ndarray) and not v.dtype.hasobject for v in values.values()), 'pickle/object arrays prohibited')
    with local(path).open('xb') as stream:np.savez(stream,**values)


def probe_module():
    return importlib.import_module(PROBE_MODULE)


def support_module():
    return importlib.import_module(SUPPORT_MODULE)


def cpu_guard():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'requires explicit CUDA_VISIBLE_DEVICES empty')
    require(not torch.is_autocast_enabled('cpu'), 'CPU autocast must be disabled')


def source_files():
    # Imports define classes only; no checkpoint/cache deserialization or model construction.
    probe_module();support_module()
    result=set(SOURCES)
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.suffix=='.py' and path.is_relative_to(ROOT) and path.is_file():
                result.add(str(path.relative_to(ROOT)))
    return sorted(result)


def split_metadata(selection,cached_protocol,support):
    require(selection.get('schema_version')=='backbone_readout_cache_v1'
        and selection.get('mode')=='train' and selection.get('seed')==42
        and selection.get('n_windows')==selection.get('selected_prefix')==500, 'fixed TRAIN500 selection required')
    rows=[entry['metadata'] for entry in selection['full_selection']]
    require(len(rows)==500 and len({row['protein'] for row in rows})==500
        and [row['horizon'] for row in rows]==list(HORIZONS)*125, 'distinct500 protein identities or schedule differ')
    p=cached_protocol['protocol']
    require(p.get('mode')=='train' and p.get('dataset_role')=='training_readout_calibration'
        and p.get('n_windows')==500 and p.get('selection_file')==SELECTION
        and p.get('selection_sha256')==sha(SELECTION), 'TRAIN cache selection linkage differs')
    plans=support['plans']
    require(len(plans)==500, 'complete metadata support required')
    for index,(row,plan) in enumerate(zip(rows,plans)):
        split='fit' if index<400 else 'readout_heldout'
        require(plan['cache_row_index']==index and plan['metadata']==row and plan['split']==split,
                'support row/order/split differs')
    require(Counter(row['horizon'] for row in rows[400:])=={h:25 for h in HORIZONS}, 'heldout horizon allocation differs')
    return rows


def contract():
    return dict(schema_version=VERSION,arms=list(ARMS),input_roles=list(INPUT_ROLES),
        versioned_repair=dict(prior_failure_inputs=PRIOR_FAILURE_INPUTS,
            issue='v1 current component has no embedded stage_a_v2_config; exact upstream original config required',
            original_v1_completed_updates=500,current_v1_completed_updates=0,v1_heldout_scored=False,
            both_arms_fresh=True,prior_head_reused=False,
            ordering='validate and encode both frozen encoders before either fresh head is fitted'),
        encoder_reconstruction=dict(
            state_sources=dict(original='model/context_encoder.*',current='component_checkpoint.encoder'),
            orientation_rule='geom_proj.weight input width12 enables orientation features; width6 disables them; other widths reject',
            relative_position_rule='relpos_proj.weight presence enables relative-position features; absence disables them',
            rationale='Checkpoint tensor structure determines the instantiated feature branches; saved flags and actual inferred flags are both reported, without modifying weights.',
            configuration_sources=dict(original='own stage_a_v2_config',current='registered original checkpoint stage_a_v2_config'),
            current_configuration_checkpoint=CHECKPOINTS['original'],current_configuration_sha256=CHECKPOINT_SHAS['original'],
            current_lineage_rule='calibration_args.stage_a_ckpt must resolve to registered original checkpoint; component stage_a_ckpt if present must equal it; no embedded-config fallback',
            strict_load=True,latent_dim=128,
            anchor_rule='Reject anchor_pos_proj.* state keys or saved anchor_conditioned=True; instantiate unanchored encoder only'),
        split=dict(fit_rows=list(range(400)),readout_heldout_rows=list(range(400,500)),
            n_fit_proteins=400,n_readout_heldout_proteins=100,heldout_proteins_per_horizon=25,
            meaning='Held out only from the fresh readout fit; either frozen encoder may have seen these TRAIN proteins.'),
        training=dict(row_order=list(TRAIN_ORDER),updates=500,seed=42,optimizer='Adam',lr=.001,
            weight_decay=0.,gradient_clip=1.,schedule='constant',selection='latest500 only',
            architecture='fresh symmetric [Hi,Hj,masked_H_mean] 384->256->256->1 GELU; average pair orders in log space',
            target='float32 per-protein all-pair log1p-distance MSE',input='aligned target backbone re-encoded with frozen encoder',
            identical_initial_readout=True,validation_during_training=False,resume=False),
        evaluation=dict(rows=list(range(400,500)),primary='equal-protein log-distance MSE',
            secondary='equal-protein mean pair absolute error; sqrt(equal-protein mean pair squared error) in angstrom',
            aggregation_dtype='float64',prediction_storage_dtype='float32',
            exponentiation='raw float32 predicted log cast to float64 then expm1, without clipping; reject nonfinite predictions/errors',
            target='raw float32 CA pair distance cast to float64; never expm1(logtarget) substitute',
            controls='same fitted head applied to observed H; separate observed-CA physical distance persistence, no extra fitted predictor',
            groups=['overall','h1','h4','h16','h64'],confidence_intervals=False),
        device='cpu',torch_threads=2,amp=False,tf32=False,
        support='all observed-valid full-PSF peptide-graph-distance>2 pairs, full graph before crop/mask; identical for both encoders/controls',
        feature_boundary='Recompute observed and aligned target raw H with identical masks and sanitization. Never use cached oracle_H or anchor_H.',
        limits='Fixed-budget static structural readability, not JEPA causal benefit, an H information bound, forecast quality, kinetics or complete-backbone feasibility. Shared static fold signal may dominate.',
        test_opened=False,bioemu2_rerun=False,dynamics_sampled=False)


def verify_prior_failure():
    for source,digest in PRIOR_FAILURE_INPUTS.items():require(sha(source)==digest,'preserved v1 failure changed: '+source)
    files={Path(source).name:source for source in PRIOR_FAILURE_INPUTS if source.endswith('.json')}
    failure=json.loads(local(files['execution.json']).read_text())
    started=json.loads(local(files['started.json']).read_text())
    require(failure['status']=='failed' and failure['returncode']==1
        and failure['failed_phase']=='current/encoding' and failure['error']=="KeyError: 'stage_a_v2_config'"
        and failure['protocol_sha256']==sha(files['protocol.json'])
        and all(failure[k]==started[k] for k in ('pid','command','started_utc','protocol_sha256')),
        'v1 failed execution identity differs')


def verify(protocol):
    for source,digest in protocol['source_hashes'].items():
        require(sha(source)==digest and sha(Path(protocol['source_snapshot'])/source)==digest, 'source/snapshot changed: '+source)
    for source,digest in protocol['input_hashes'].items():require(sha(source)==digest, 'input changed: '+source)


def register():
    cpu_guard()
    out=local(OUTPUT);out.mkdir(parents=True,exist_ok=False)
    begin=time.monotonic();status=dict(pid=os.getpid(),command=sys.argv,started_utc=now())
    try:
        verify_prior_failure()
        support=support_module().load_support(local(SUPPORT),SUPPORT_SHA)
        require(sha(CACHE)==CACHE_SHA, 'fixed cache bytes differ')
        for arm in ARMS:require(sha(CHECKPOINTS[arm])==CHECKPOINT_SHAS[arm], 'fixed '+arm+' encoder checkpoint bytes differ')
        selection=json.loads(local(SELECTION).read_text());cached_protocol=json.loads(local(CACHE_PROTOCOL).read_text())
        rows=split_metadata(selection,cached_protocol,support)
        source_hashes={source:sha(source) for source in source_files()}
        snapshot=Path(OUTPUT)/'source_snapshot'
        for source,digest in source_hashes.items():
            target=local(snapshot/source);target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write(local(source).read_bytes())
            require(sha(target)==digest, 'source changed during snapshot')
        support_files=[str(Path(SUPPORT).with_name(name+'.json')) for name in ('protocol','started','execution')]
        input_hashes={source:sha(source) for source in (CACHE,SELECTION,CACHE_PROTOCOL,*CHECKPOINTS.values(),SUPPORT,*support_files,*PRIOR_FAILURE_INPUTS)}
        protocol=dict(contract(),registered_utc=now(),input_hashes=input_hashes,source_hashes=source_hashes,
            source_snapshot=str(snapshot),selected_metadata=rows,selected_metadata_sha256=json_sha(rows),
            pair_support=SUPPORT,pair_support_sha256=SUPPORT_SHA,cache=CACHE,cache_sha256=CACHE_SHA,
            checkpoints={arm:dict(path=CHECKPOINTS[arm],sha256=CHECKPOINT_SHAS[arm]) for arm in ARMS},
            cache_deserialized_at_registration=False,checkpoint_deserialized_at_registration=False)
        verify(protocol);write_json(out/'protocol.json',protocol)
        status.update(status='completed',returncode=0,protocol_sha256=sha(out/'protocol.json'))
    except BaseException as error:
        status.update(status='failed',returncode=1,error=f'{type(error).__name__}: {error}');raise
    finally:
        status.update(completed_utc=now(),seconds=time.monotonic()-begin);write_json(out/'registration.execution.json',status)
    return protocol


def validate_cache(cache,protocol):
    expected=dict(mode='train',dataset_role='training_readout_calibration',seed=42,n_windows=500,
        context_frames=8,max_residues=96,horizons=list(HORIZONS),n_forecast_samples=0,optimizer_steps=0,
        test_opened=False,bioemu2_rerun=False)
    p=cache['protocol']
    require(cache['schema_version']=='backbone_readout_cache_v1' and all(p.get(k)==v for k,v in expected.items()), 'TRAIN cache scope differs')
    require(p==json.loads(local(CACHE_PROTOCOL).read_text())['protocol'], 'embedded cache protocol differs')
    completion=cache['completion']
    require(completion['frozen_inputs_verified'] is True
        and completion['model_sha256_before']==completion['model_sha256_after']==p['model_tensor_sha256'], 'cache incomplete/model changed')
    rows=cache['rows']
    require(len(rows)==500 and [row['metadata'] for row in rows]==protocol['selected_metadata'], 'actual cache identities differ')
    return rows


def prepare_rows(rows,support):
    prepared=[];api=probe_module()
    for index,row in enumerate(rows):
        physical=support_module().attach_cached_physical_support(row,index,support)
        pairs,mask=physical['pair_indices'],physical['observed_mask']
        require(torch.equal(mask,row['observed_mask']), 'physical support changed observed mask')
        labels=api.pair_log_targets(row,pairs)
        require(set(labels)=={'target_distance','target_log','observed_distance'}, 'distance label schema differs')
        require(all(value.shape==(len(pairs),) and value.dtype==torch.float32 and value.device.type=='cpu'
                    and bool(torch.isfinite(value).all()) for value in labels.values()), 'distance label shape/dtype/finiteness differs')
        prepared.append(dict(row=row,pairs=pairs,mask=mask,labels=labels,physical_support_audit=physical['audit'],
            labels_sha256=state_sha256(dict(pair_indices=pairs,**labels))))
    require(len(prepared)==500, 'incomplete cache support validation')
    return prepared


def encode_rows(encoder,prepared):
    result=[];api=probe_module()
    for item in prepared:
        observed,target=api.encode_pair(encoder,item['row'])
        require(all(value.shape==(1,len(item['mask']),128) and value.dtype==torch.float32
            and not value.requires_grad and bool(torch.isfinite(value).all()) for value in (observed,target)), 'raw encoded feature contract differs')
        result.append(dict(observed_H=observed,target_H=target))
    return result


def fit_readout(arm,prepared,features,out):
    """Exactly500 fixed updates; scoring rows/features never enter a fit call."""
    api=probe_module();head=api.new_probe(seed=42).train()
    initial={key:value.detach().clone() for key,value in head.state_dict().items()}
    initial_sha=state_sha256(initial)
    optimizer=torch.optim.Adam(head.parameters(),lr=.001,weight_decay=0.)
    completed=0;metrics=out/f'{arm}.training.jsonl'
    try:
        with local(metrics).open('x') as stream:
            for step,index in enumerate(TRAIN_ORDER,1):
                item=prepared[index]
                optimizer.zero_grad(set_to_none=True)
                loss,prediction=api.probe_loss(head,features[index]['target_H'],item['mask'],item['pairs'],item['labels']['target_log'])
                require(loss.ndim==0 and loss.dtype==torch.float32 and bool(torch.isfinite(loss)), 'invalid training loss')
                loss.backward()
                require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()), 'nonfinite/missing readout gradient')
                norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
                require(bool(torch.isfinite(norm)), 'nonfinite gradient norm')
                optimizer.step();completed=step
                require(all(bool(torch.isfinite(p).all()) for p in head.parameters()), 'nonfinite updated readout')
                record=dict(step=step,cache_row_index=index,protein=item['row']['metadata']['protein'],
                    horizon=item['row']['metadata']['horizon'],n_pairs=len(item['pairs']),log_mse=float(loss.detach()),
                    gradient_norm_before_clip=float(norm),lr=.001,labels_sha256=item['labels_sha256'])
                stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
                del loss,prediction,norm
        require(completed==500 and all(float(state['step'])==500 for state in optimizer.state.values()), 'readout update count differs')
        head.eval()
        output=out/f'{arm}.readout_latest500.pt'
        save_tensor(output,dict(schema_version=VERSION,arm=arm,step=500,initial_head=initial,
            initial_head_sha256=initial_sha,head=head.state_dict(),head_sha256=state_sha256(head),
            optimizer=optimizer.state_dict(),training_order=list(TRAIN_ORDER),selection='latest500 only'))
        return head,dict(initial_head_sha256=initial_sha,final_head_sha256=state_sha256(head),
            optimizer_steps=500,training_log=str(metrics.relative_to(ROOT)),training_log_sha256=sha(metrics),
            checkpoint=str(output.relative_to(ROOT)),checkpoint_sha256=sha(output))
    except BaseException as error:
        partial=out/f'{arm}.readout_failed.pt'
        save_tensor(partial,dict(arm=arm,completed_updates=completed,initial_head=initial,
            head=head.state_dict(),optimizer=optimizer.state_dict(),error=f'{type(error).__name__}: {error}'))
        raise


def aggregate(records):
    require(len(records)==100 and len({row['metadata']['protein'] for row in records})==100
        and Counter(row['metadata']['horizon'] for row in records)=={h:25 for h in HORIZONS}, 'heldout100 support differs')
    groups={'overall':records,**{'h'+str(h):[row for row in records if row['metadata']['horizon']==h] for h in HORIZONS}}
    output={}
    roles=[arm+'/'+role for arm in ARMS for role in INPUT_ROLES]+['observed_CA_persistence']
    for name,rows in groups.items():
        result={}
        for role in roles:
            values=[row['scores'][role] for row in rows]
            means={key:float(np.mean(np.asarray([v[key] for v in values],dtype=np.float64)))
                   for key in ('log_mse','mae_angstrom','mse_angstrom2')}
            require(all(math.isfinite(value) for value in means.values()), 'nonfinite aggregate')
            result[role]=dict(**means,rmse_angstrom=math.sqrt(means['mse_angstrom2']),
                negative_count=sum(v['negative_count'] for v in values),n_pairs=sum(v['n_pairs'] for v in values),
                mean_protein_negative_fraction=float(np.mean([v['negative_count']/v['n_pairs'] for v in values])),
                n_proteins=len(rows))
        output[name]=result
    return output


@torch.no_grad()
def score_heldout(heads,features,prepared,out):
    api=probe_module();records=[]
    for index in range(400,500):
        item=prepared[index];labels=item['labels']
        arrays={name:value.numpy().copy() for name,value in labels.items()}
        arrays['pair_indices']=item['pairs'].numpy().copy()
        scores={}
        for arm in ARMS:
            require(not heads[arm].training, 'readout must be eval for heldout scoring')
            for role,key in [('target_H','target_H'),('observed_H_substitution','observed_H')]:
                prediction=heads[arm](features[arm][index][key],item['mask'],item['pairs'])
                scores[arm+'/'+role]=api.score_predictions(prediction,labels['target_log'],target_distance=labels['target_distance'])
                arrays[arm+'__'+role+'__predicted_log']=prediction.detach().numpy().copy()
        persistent_log=torch.log1p(labels['observed_distance'])
        persistent=api.distance_metrics(labels['observed_distance'],labels['target_distance'])
        persistent['log_mse']=float((persistent_log.double()-labels['target_log'].double()).square().mean())
        scores['observed_CA_persistence']=persistent
        arrays['persistence_log']=persistent_log.numpy().copy()
        file=out/f'heldout_{index:03d}.npz';save_npz(file,arrays)
        record=dict(cache_row_index=index,metadata=item['row']['metadata'],n_pairs=len(item['pairs']),
            labels_sha256=item['labels_sha256'],physical_support_audit=item['physical_support_audit'],
            scores=scores,pair_predictions=dict(path=str(file.relative_to(ROOT)),sha256=sha(file)))
        write_json(out/f'heldout_{index:03d}.json',record);records.append(record)
    return records


def run():
    cpu_guard();out=local(OUTPUT);protocol_file=out/'protocol.json'
    require(not any((out/name).exists() for name in ('started.json','execution.json','result.json')), 'execution already attempted')
    digest=sha(protocol_file);begin=time.monotonic()
    started=dict(pid=os.getpid(),command=sys.argv,started_utc=now(),protocol_sha256=digest)
    write_json(out/'started.json',started);status=dict(started);phase='verify'
    try:
        protocol=json.loads(protocol_file.read_text())
        require(all(protocol.get(k)==v for k,v in contract().items()), 'registered contract differs')
        registered=json.loads((out/'registration.execution.json').read_text())
        require(registered['status']=='completed' and registered['returncode']==0 and registered['protocol_sha256']==digest, 'registration incomplete')
        verify(protocol)
        verify_prior_failure()
        support=support_module().load_support(local(SUPPORT),SUPPORT_SHA) # before cache/encoder loads
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        rng_before=torch.get_rng_state().clone()
        phase='cache';cache=torch.load(local(CACHE),map_location='cpu',weights_only=False)
        prepared=prepare_rows(validate_cache(cache,protocol),support)
        write_json(out/'all500_support_audit.json',[dict(cache_row_index=i,metadata=item['row']['metadata'],
            labels_sha256=item['labels_sha256'],physical_support_audit=item['physical_support_audit']) for i,item in enumerate(prepared)])
        api=probe_module();features={};heads={};arm_audits={}
        for arm in ARMS:
            phase=arm+'/encoding'
            config_args={} if arm=='original' else dict(configuration_checkpoint_path=local(CHECKPOINTS['original']),
                expected_configuration_sha256=CHECKPOINT_SHAS['original'])
            encoder,encoder_audit=api.load_frozen_encoder(local(CHECKPOINTS[arm]),kind=arm,expected_sha256=CHECKPOINT_SHAS[arm],**config_args)
            before=state_sha256(encoder);features[arm]=encode_rows(encoder,prepared)
            require(state_sha256(encoder)==before and not encoder.training
                and all(p.grad is None and not p.requires_grad for p in encoder.parameters()), 'frozen encoder changed')
            write_json(out/f'{arm}.encoded_identity.json',dict(encoder=encoder_audit,
                before_sha256=before,after_sha256=state_sha256(encoder),
                rows=[dict(cache_row_index=i,sha256=state_sha256(value)) for i,value in enumerate(features[arm])]))
            arm_audits[arm]=dict(encoder=encoder_audit)
            del encoder
        # Both lineages and all frozen encodings must succeed before any fit.
        for arm in ARMS:
            phase=arm+'/fit';heads[arm],fitting=fit_readout(arm,prepared,features[arm],out)
            arm_audits[arm]['readout']=fitting
        require(arm_audits['original']['readout']['initial_head_sha256']==arm_audits['current']['readout']['initial_head_sha256'], 'readout initial tensors differ')
        head_hashes={arm:state_sha256(heads[arm]) for arm in ARMS}
        phase='readout_heldout';records=score_heldout(heads,features,prepared,out)
        require(all(state_sha256(heads[arm])==head_hashes[arm] for arm in ARMS), 'scoring changed fitted readout')
        require(torch.equal(torch.get_rng_state(),rng_before), 'readout diagnostic consumed global CPU RNG')
        require(sha(protocol_file)==digest, 'protocol changed during execution');verify(protocol)
        support_module().load_support(local(SUPPORT),SUPPORT_SHA)
        result=dict(schema_version=VERSION,status='completed',protocol_sha256=digest,
            arm_audits=arm_audits,n_fit_proteins=400,n_readout_heldout_proteins=100,
            per_protein=records,aggregates=aggregate(records),
            tensor_identity_evidence=dict(readout_initial_tensors_equal=True,encoders_frozen=True,
                scoring_readouts_unchanged=True,cpu_rng_unchanged=True,
                cpu_rng_sha256_before=state_sha256(dict(cpu_rng=rng_before)),
                cpu_rng_sha256_after=state_sha256(dict(cpu_rng=torch.get_rng_state()))),
            no_cached_H_used=True,no_dynamics_sampling=True,test_opened=False,bioemu2_rerun=False,
            scientific_limits=protocol['limits'])
        write_json(out/'result.json',result)
        status.update(status='completed',returncode=0,result_sha256=sha(out/'result.json'))
    except BaseException as error:
        status.update(status='failed',returncode=1,failed_phase=phase,error=f'{type(error).__name__}: {error}');raise
    finally:
        status.update(completed_utc=now(),seconds=time.monotonic()-begin);write_json(out/'execution.json',status)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=('register','run'))
    args=parser.parse_args();result=register() if args.mode=='register' else run()
    print(json.dumps(dict(mode=args.mode,status=result.get('status','registered')),allow_nan=False),flush=True)


if __name__=='__main__':main()
