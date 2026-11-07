"""Read-only provenance and scalar audits for the fixed-cache CA readout probe.

The independent numerical estimands remain owned by the frozen
ca_local_residual_statistics_v1 module. This wrapper does not select a model,
train, sample dynamics, open MD trajectories, or extend the task claims.
"""
from __future__ import annotations

from collections.abc import Mapping
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from protein_jepa_frame.utils.contracts import state_sha256
from scripts import ca_local_residual_statistics_v1 as numeric
from scripts import register_ca_local_residual_statistics_v1 as statistics_registration
from run_scripts import run_ca_local_residual_probe_v1 as producer

REPO = Path(__file__).resolve().parents[1]
BASE = Path('results/hs_joint_20261101/ca_local_residual_v1')
ARTIFACTS = Path('artifacts/hs_joint_20261101/ca_local_residual_v1')
IMPLEMENTATION = Path('results/hs_joint_20261101/ca_local_residual_summary_v1')
SCRIPT = 'scripts/summarize_ca_local_residual_probe_v1.py'
TEST = 'tests/test_ca_local_residual_summary_v1.py'
STATISTICS_SHA = '6019841688e0e39fb5bf52b27d9f04d05771e7db96d69fb4091e68f8746d87cf'
VERSION = 'ca_local_residual_probe_summary_v1'
EXTRA = 'local_residual.weight'
require = numeric.require


def now():
    return datetime.now(timezone.utc).isoformat()


def local(value):
    path = Path(value)
    path = (REPO/path).resolve() if not path.is_absolute() else path.resolve()
    require(path.is_relative_to(REPO.resolve()) and path.name != 'test.jsonl', 'outside repository or sealed test path')
    require('bioemu' not in str(path).lower(), 'BioEmu input is outside this probe')
    return path


def sha(value):
    digest = hashlib.sha256()
    with local(value).open('rb') as stream:
        for block in iter(lambda: stream.read(1<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path,value):
    encoded = json.dumps(value,indent=2,allow_nan=False)+'\n'
    path = local(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        stream.write(encoded)


class Inputs:
    def __init__(self):
        self.hashes = {}

    def track(self,value,expected=None):
        path = local(value);digest = sha(path)
        require(expected is None or digest == expected, 'SHA mismatch: '+str(path))
        require(path not in self.hashes or self.hashes[path] == digest, 'input changed: '+str(path))
        self.hashes[path] = digest
        return path

    def read(self,value,expected=None):
        return json.loads(self.track(value,expected).read_text())

    def verify(self):
        for path,digest in self.hashes.items():
            require(sha(path) == digest, 'input changed during summary: '+str(path))


def close(actual,expected,where='scalar'):
    if expected is None:
        require(actual is None, where+': undefined value was replaced')
    else:
        require(math.isclose(numeric.number(actual),numeric.number(expected),rel_tol=1e-10,abs_tol=1e-12),
                where+': scalar arithmetic differs')


def complete_mean(values):
    require(bool(values), 'empty member support')
    if any(value is None for value in values):
        for value in values:
            if value is not None:numeric.number(value)
        return None
    return float(np.mean([numeric.number(value) for value in values]))


def audit_leaf(leaf,role,persistence_rmsd):
    """Rebuild member means, preserving null support and mean-of-RMSD order."""
    require(role in numeric.ROLES, 'unknown decoding role')
    count = 12 if role == 'forecast' else 1
    require(type(leaf.get('n_samples')) is int and leaf['n_samples'] == count
        and type(leaf.get('n_members')) is int and leaf['n_members'] == count, 'member budget differs')
    members = leaf['per_member']
    require(isinstance(members,list) and len(members) == count, 'missing member scalar records')
    metrics = numeric.FORECAST_SCALARS if role == 'forecast' else numeric.SCALARS
    require(set(leaf['scalars']) == set(metrics), 'leaf scalar schema differs')
    persistence = numeric.number(persistence_rmsd)
    require(persistence >= 0, 'negative persistence RMSD')
    for member in members:
        require(set(numeric.SCALARS) <= set(member), 'member scalar schema incomplete')
        mse,rmsd = (numeric.number(member[name]) for name in ('coordinate_mse','ca_rmsd'))
        require(mse >= 0 and rmsd >= 0, 'negative coordinate error')
        close(rmsd,math.sqrt(3*mse),'per-member sqrt(3*MSE)')
        close(member['rmsd_to_persistence_ratio'],rmsd/persistence if persistence else None,'persistence ratio')
        for metric in numeric.GEOMETRY:
            value = member[metric]
            if value is not None:
                value = numeric.number(value)
                require(value >= 0,'negative geometry scalar')
                if metric.endswith('fraction'):require(value <= 1,'geometry fraction outside [0,1]')
    for metric in numeric.SCALARS:
        close(leaf['scalars'][metric],complete_mean([m[metric] for m in members]),metric)
    if role == 'forecast':
        for metric in set(numeric.FORECAST_SCALARS)-set(numeric.SCALARS):
            numeric.number(leaf['scalars'][metric])
        require(0 <= leaf['scalars']['contact_brier'] <= 1,'contact Brier outside [0,1]')
    return count


def finite_state(state):
    require(isinstance(state,Mapping) and bool(state),'missing decoder state')
    for name,value in state.items():
        require(isinstance(name,str) and isinstance(value,torch.Tensor)
            and value.is_floating_point() and bool(torch.isfinite(value).all()), 'invalid decoder state tensor')


def same_state(first,second):
    require(set(first) == set(second),'decoder state key mismatch')
    for name,value in first.items():
        other = second[name]
        require(value.shape == other.shape and value.dtype == other.dtype and torch.equal(value,other),
                'decoder state differs: '+name)


def audit_initial_states(states,frozen_d0):
    require(set(states) == set(numeric.ARMS),'initial decoder arms differ')
    finite_state(frozen_d0)
    for state in states.values():finite_state(state)
    same_state(states['baseline'],frozen_d0)
    same_state(states['base_tuned'],frozen_d0)
    residual = states['local_residual']
    require(set(residual) == set(frozen_d0)|{EXTRA},'residual state is not the exact nested architecture')
    same_state({k:v for k,v in residual.items() if k != EXTRA},frozen_d0)
    weight = residual[EXTRA]
    require(weight.shape == (3,128) and weight.numel() == 384 and not bool(weight.any()),'initial local W must be exactly zero384')
    return dict(old_parameters_identical=True,extra_parameters=384,extra_bias=False,
        extra_initially_zero=True,baseline_state_sha256=state_sha256(frozen_d0))


def audit_final_state(initial,final,arm):
    require(arm in ('base_tuned','local_residual'),'unexpected trained arm')
    finite_state(final)
    require(set(initial) == set(final),'final decoder architecture differs')
    require(torch.equal(initial['centers'],final['centers']),'fixed centers buffer changed')
    changed = []
    for name,value in final.items():
        require(value.shape == initial[name].shape and value.dtype == initial[name].dtype,'final decoder tensor metadata differs')
        if not torch.equal(value,initial[name]):changed.append(name)
    require(any(name not in (EXTRA,'centers') for name in changed),'no old decoder parameter changed after training')
    if arm == 'local_residual':require(EXTRA in changed,'local residual remained identically zero after training')
    return dict(finite=True,changed_tensor_names=changed,final_state_sha256=state_sha256(final))


def audit_optimizer(optimizer,parameter_names,final_state,hyperparameters):
    """Audit actual Adam state support/steps, rather than trusting checkpoint.step."""
    expected_names = [name for name in final_state if name != 'centers']
    require(isinstance(parameter_names,list) and parameter_names == expected_names,
            'optimizer parameter names/order differ from decoder schema')
    groups = optimizer['param_groups']
    require(len(groups) == 1,'registered decoder optimizer requires one parameter group')
    group = groups[0]
    ids = group['params']
    require(isinstance(ids,list) and len(ids) == len(parameter_names) and len(set(ids)) == len(ids)
        and all(type(i) is int for i in ids),'optimizer parameter ID support differs')
    require(set(optimizer['state']) == set(ids),'optimizer state is missing a trainable parameter')
    # JSON canonicalization is explicit: Adam stores betas as a tuple while
    # the registered configuration is JSON. Numbers and key membership remain
    # exact; this only normalizes container representation.
    actual_hyper = json.loads(json.dumps({k:v for k,v in group.items() if k != 'params'},allow_nan=False))
    for key,value in hyperparameters.items():
        require(key in actual_hyper and actual_hyper[key] == value,'optimizer hyperparameter differs: '+key)
    for identifier,name in zip(ids,parameter_names):
        state = optimizer['state'][identifier]
        require({'step','exp_avg','exp_avg_sq'} <= set(state),'Adam moment fields missing')
        step = state['step']
        require(isinstance(step,torch.Tensor) and step.numel() == 1 and bool(torch.isfinite(step))
            and float(step) == 500.,'optimizer must record exactly500 updates for every trainable parameter')
        for field in ('exp_avg','exp_avg_sq'):
            value = state[field]
            require(isinstance(value,torch.Tensor) and value.shape == final_state[name].shape
                and value.dtype == final_state[name].dtype and bool(torch.isfinite(value).all()),
                'optimizer moment shape/dtype/finite differs: '+name)
        require(bool((state['exp_avg_sq'] >= 0).all()),'negative Adam second moment')
    return dict(active_parameter_tensors=len(parameter_names),optimizer_steps=500,
        parameter_order_verified=True,registered_hyperparameters=hyperparameters)


def audit_cache(cache,split):
    """Tensor/metadata checks only; no HDF5 or reconstruction operations."""
    require(split in ('train','val') and cache['schema_version'] == 'backbone_readout_cache_v1','cache kind differs')
    count = 500 if split == 'train' else 200
    p = cache['protocol']
    expected = dict(mode=split,dataset_role='training_readout_calibration' if split == 'train' else 'development_validation',
        n_windows=count,complete_panel_windows=count,full_panel=True,seed=42 if split == 'train' else 0,
        horizons=[1,4,16,64],max_residues=96,context_frames=8,n_forecast_samples=0 if split == 'train' else 12,
        forecast_observed_only=True,before_target_read=True,optimizer_steps=0,test_opened=False,bioemu2_rerun=False,
        frame_interval_ps=1000.)
    require(all(p.get(k) == value for k,value in expected.items()),'cache protocol differs')
    completion = cache['completion']
    require(completion['frozen_inputs_verified'] is True and completion.get('completed_utc')
        and completion['model_sha256_before'] == completion['model_sha256_after'] == p['model_tensor_sha256'],
        'cache completion/model identity differs')
    rows = cache['rows']
    require(len(rows) == count,'cache row budget differs')
    numeric.indexed([row['metadata'] for row in rows],split)
    if split == 'train':
        require([row['metadata']['horizon'] for row in rows] == [numeric.HORIZONS[i%4] for i in range(count)],
                'training cache horizon order differs')
    for row in rows:
        mask = row['observed_mask'];length = len(mask);m = row['metadata']
        require(mask.dtype == torch.bool and mask.shape == (length,) and 4 <= int(mask.sum()) <= length <= 96,'invalid cache mask')
        require(m['target_frame'] == m['anchor_frame']+m['horizon'] and m['horizon_ps'] == m['horizon']*1000
            and m['context_frames'] == list(range(m['anchor_frame']-7,m['anchor_frame']+1)),'cache physical window contract differs')
        boundary = row['boundary']
        require(boundary['forecast_before_target_read'] is True and boundary['future_reencoding_for_forecast'] == 0
            and boundary['observed_frames'] == m['context_frames'] and boundary['label_frames'] == [m['target_frame']],
            'cache observation/target boundary differs')
        shapes = dict(observed_R=(1,length,3,3),observed_backbone=(1,length,4,3),
            target_backbone=(1,length,4,3),anchor_H=(1,length,128),oracle_H=(1,length,128),oracle_CA=(1,length,3))
        if split == 'val':shapes.update(forecast_H=(12,length,128),forecast_CA=(12,length,3))
        else:require('forecast_H' not in row and 'forecast_CA' not in row,'training cache unexpectedly contains forecast samples')
        for name,shape in shapes.items():
            value = row[name]
            require(value.shape == shape and value.dtype == torch.float32 and value.device.type == 'cpu'
                and value.requires_grad is False
                and bool(torch.isfinite(value[:,mask]).all()),'cache tensor contract differs: '+name)
        for name in ('embedding_aatype','embedding_residue_index','embedding_chain_id','chemical_aatype','geometry_row_ids'):
            value = row[name]
            require(value.shape == (length,) and value.dtype == torch.long,'cache integer metadata differs')
            require(value.tolist() == row['topology'][name],'topology tensor/list metadata differs')
        require(row['peptide_edges'].tolist() == row['topology']['peptide_edges'],'topology edge metadata differs')
        require(row['atom_mask'].tolist() == row['topology']['atom_mask'],'topology atom metadata differs')
        for name in ('atom_mask','target_atom_mask'):
            value = row[name]
            require(value.shape == (length,4) and value.dtype == torch.bool and bool(value[mask].all()),'incomplete cached backbone')
        fps = row['fingerprints']
        require(fps['aligned_label'] == state_sha256({'backbone':row['target_backbone']})
            and fps['oracle_features'] == state_sha256({'H':row['oracle_H'],'CA':row['oracle_CA']}), 'cache label/oracle fingerprint differs')
        forecast = None if split == 'train' else state_sha256({name:row[name] for name in ('forecast_H','forecast_CA')})
        require(fps['forecast'] == forecast,'cache forecast fingerprint differs')
    return rows


def tree_equal(first,second):
    if isinstance(first,torch.Tensor):
        return isinstance(second,torch.Tensor) and first.shape == second.shape and first.dtype == second.dtype and torch.equal(first,second)
    if isinstance(first,np.ndarray):
        return isinstance(second,np.ndarray) and first.dtype == second.dtype and np.array_equal(first,second)
    if isinstance(first,Mapping):
        return isinstance(second,Mapping) and set(first) == set(second) and all(tree_equal(v,second[k]) for k,v in first.items())
    if isinstance(first,(tuple,list)):
        return isinstance(second,(tuple,list)) and len(first) == len(second) and all(tree_equal(a,b) for a,b in zip(first,second))
    return type(first) is type(second) and first == second


def json_equal(first,second):
    return json.dumps(first,sort_keys=True,allow_nan=False) == json.dumps(second,sort_keys=True,allow_nan=False)


def audit_execution(record,phase,protocol_sha,source_sha):
    require(record.get('phase') == phase and record.get('status') == 'completed'
        and type(record.get('exit_code')) is int and record['exit_code'] == 0,'probe '+phase+' is incomplete or failed')
    require(type(record.get('pid')) is int and record['pid'] > 0 and record.get('started_utc') and record.get('completed_utc'),
            'missing actual process evidence')
    require(datetime.fromisoformat(record['completed_utc']) >= datetime.fromisoformat(record['started_utc']), 'execution timestamps reversed')
    require(numeric.number(record['seconds']) >= 0 and record['protocol_sha256'] == protocol_sha
        and record['source_sha256'] == source_sha and record['frozen_inputs_sources_verified'] is True,
        'execution provenance differs')
    require(record['cpu_threads'] == 2 and record['process_kind'] == 'single_actual_process_with_sequential_stages',
            'execution backend/stage convention differs')
    argv = record['command']
    require(isinstance(argv,list) and len(argv) >= 3 and argv[-2:] == ['--mode',phase]
        and Path(argv[0]).name == Path(producer.SCRIPT).name,'execution command differs')
    require(record['output'] == str(BASE/('smoke.json' if phase == 'smoke' else 'raw.json')),'execution output differs')
    if phase == 'run':
        stages = record['stages']
        require(len(stages) == 3,'expected two in-process training stages and one evaluation')
        for index,stage in enumerate(stages):
            require(stage['status'] == 'completed' and type(stage['exit_code']) is int and stage['exit_code'] == 0
                and stage['pid'] == record['pid'] and stage['process_kind'] == 'in_process_sequential_stage',
                'missing successful in-process stage')
            require(datetime.fromisoformat(record['started_utc']) <= datetime.fromisoformat(stage['started_utc'])
                <= datetime.fromisoformat(stage['completed_utc']) <= datetime.fromisoformat(record['completed_utc']),
                'stage timestamps outside process execution')
            if index:
                require(datetime.fromisoformat(stages[index-1]['completed_utc']) <= datetime.fromisoformat(stage['started_utc']),
                        'stages are not sequential')
            if index < 2:
                arm = producer.ARMS[index]
                require(stage['arm'] == arm and stage['steps'] == stage['all_optimizer_steps'] == 500
                    and stage['checkpoint'] == str(ARTIFACTS/arm/'checkpoint_latest.pt'), 'training stage budget/path differs')
            else:
                require(stage['stage'] == 'evaluate' and stage['raw'] == str(BASE/'raw.json')
                    and stage['raw_sha256'] == record['output_sha256']
                    and stage['path'] == str(BASE/'score_coordinates.pt'),'evaluation evidence differs')
    return record


def audit_frozen(protocol,inputs,*,current=True):
    for name,digest in protocol['input_hashes'].items():inputs.track(name,digest)
    for name,digest in protocol['source_hashes'].items():
        if current:inputs.track(name,digest)
        inputs.track(Path(protocol['source_snapshot'])/name,digest)


def audit_cache_provenance(cache,path,inputs,initial):
    path = Path(path)
    registration = inputs.read(path.with_suffix('.protocol.json'))
    for field in ('protocol','source_hashes','input_hashes','provenance'):
        require(json_equal(cache[field],registration[field]),'cache embedded registration differs: '+field)
    for name,digest in cache['input_hashes'].items():inputs.track(name,digest)
    # Historical source bytes live in their historical snapshot; intentionally
    # versioned current source files need not have the old implementation hash.
    for name,digest in cache['source_hashes'].items():inputs.track(path.with_suffix('.source_snapshot')/name,digest)
    execution = inputs.read(path.with_suffix('.execution.json'))
    require(type(execution['exit_code']) is int and execution['exit_code'] == 0 and execution.get('completed_utc')
        and execution.get('started_utc'),'historical cache build did not exit0')
    argv = execution['argv']
    require(argv[1:3] == ['-m','scripts.prepare_backbone_readout_cache']
        and argv[argv.index('--out')+1] == str(path)
        and argv[argv.index('--mode')+1] == cache['protocol']['mode'],'cache execution identity differs')
    selection = inputs.read(cache['protocol']['selection_file'],cache['protocol']['selection_sha256'])
    rows = audit_cache(cache,cache['protocol']['mode'])
    producer.validate_cache(cache,cache['protocol']['mode'],initial,selection)
    require(cache['protocol']['model_tensor_sha256'] == producer.inference_hash(initial),'frozen representation/kernel differs')
    return rows


def audit_training_records(metrics,batches,rows):
    require(len(metrics) == len(batches) == len(rows) == 500,'training must consume exactly500 fixed rows')
    for step,(metric,batch,row) in enumerate(zip(metrics,batches,rows),1):
        require(type(metric['step']) is int and metric['step'] == batch['step'] == step
            and metric['metadata'] == row['metadata'] and batch['key'] == list(producer.key(row))
            and batch['input_tensor_sha256'] == producer.row_fingerprint(row),'actual training stream differs')
        require(numeric.number(metric['loss']) >= 0 and numeric.number(metric['gradient_norm']) >= 0,
                'training loss/gradient invalid')
        require(metric['lr'] == producer.learning_rate(step),'training learning-rate schedule differs')
    return dict(n_steps=500,n_unique_proteins=len({r['metadata']['protein'] for r in rows}),same_order=True)


def audit_checkpoint(checkpoint,initial,arm,protocol,protocol_sha):
    require(checkpoint['schema_version'] == producer.VERSION and checkpoint['arm'] == arm
        and checkpoint['step'] == 500 and checkpoint['selection'] == 'latest500'
        and checkpoint['validation'] is None,'checkpoint selection/budget differs')
    require(checkpoint['model_config'] == producer.CONFIG
        and checkpoint['hyperparameters'] == protocol['hyperparameters'] == producer.HYPER,'checkpoint hyperparameters differ')
    require(checkpoint['encoder_instantiated'] is False and checkpoint['kernel_instantiated'] is False
        and checkpoint['protocol_sha256'] == protocol_sha,'checkpoint frozen-input contract differs')
    require(checkpoint['input_hashes'] == protocol['input_hashes'] and checkpoint['source_hashes'] == protocol['source_hashes'],
            'checkpoint provenance differs')
    require(checkpoint['initial_model_sha256'] == state_sha256(initial)
        and checkpoint['initial_decoder_sha256'] == state_sha256({k:v for k,v in initial.items() if k != EXTRA}),
        'checkpoint initial decoder differs')
    final = audit_final_state(initial,checkpoint['head'],arm)
    require(checkpoint['final_model_sha256'] == final['final_state_sha256'],'checkpoint final state hash differs')
    optimizer = audit_optimizer(checkpoint['optimizer'],checkpoint['parameter_names'],checkpoint['head'],
        dict(lr=producer.learning_rate(500),weight_decay=.01,betas=[.9,.999],eps=1e-8,
             amsgrad=False,maximize=False,foreach=None,capturable=False,differentiable=False,fused=None))
    return dict(final=final,optimizer=optimizer)


def reference_tensors(row):
    return dict(target_ca=row['target_backbone'][0,:,1,:],anchor_ca=row['observed_backbone'][0,:,1,:],
        **{name:row[name] for name in ('observed_mask','embedding_residue_index','embedding_chain_id','peptide_edges')})


def audit_raw(raw,sidecar,caches,identity,protocol,protocol_sha):
    require(raw['schema_version'] == sidecar['schema_version'] == producer.VERSION
        and raw['protocol_sha256'] == sidecar['protocol_sha256'] == protocol_sha,'raw/sidecar protocol differs')
    require(raw['initial_identity'] == identity and raw['cache_sha256'] == producer.CACHE_SHA
        and raw['source_hashes'] == protocol['source_hashes'] and raw['input_hashes'] == protocol['input_hashes'],
        'raw identity/provenance differs')
    for name in ('seven_metrics_run','encoder_instantiated','kernel_instantiated','hdf_opened','test_opened','new_sampling'):
        require(raw[name] is False,'out-of-scope operation declared: '+name)
    require(raw['controls'] == protocol['controls'] and raw['completed_utc'],'raw controls/completion differ')
    require(set(raw['per_window']) == set(sidecar['per_window']) == {'train','val'},'raw split support differs')
    donors = producer.wrong_horizon_donors(caches['val']['rows'])
    leaves = members = 0
    mismatch = {}
    for split in ('train','val'):
        rows = caches[split]['rows'];records = raw['per_window'][split];stored = sidecar['per_window'][split]
        require(len(records) == len(stored) == len(rows),'raw/coordinate panel count differs')
        numeric.indexed(records,split)
        mismatch[split] = []
        for index,(row,record,coords) in enumerate(zip(rows,records,stored)):
            require(all(record[k] == v for k,v in row['metadata'].items()) and record['row_index'] == coords['row_index'] == index
                and coords['key'] == list(producer.key(row)),'raw coordinate/cache key or order differs')
            require(tree_equal(coords['reference_tensors'],reference_tensors(row)),'sidecar reference tensors differ from frozen cache')
            mask = row['observed_mask'];target = row['target_backbone'][0,:,1,:].double();anchor = row['observed_backbone'][0,:,1,:].double()
            support = producer.topology_support(row)
            require(record['topology_audit'] == support[2] and record['valid_residues'] == int(mask.sum()),'raw topology/support differs')
            if not support[2]['legacy_psf_equal']:mismatch[split].append(list(producer.key(row)))
            require(record['input_fingerprints'] == dict(row_tensor_sha256=producer.row_fingerprint(row),**row['fingerprints']),
                    'raw input fingerprints differ')
            persistence = float((anchor[mask]-target[mask]).square().sum(-1).mean().sqrt())
            require(set(record['references']) == {'persistence_ca_rmsd','persistence_geometry','target_geometry'},'reference fields differ')
            close(record['references']['persistence_ca_rmsd'],persistence,'cached persistence RMSD')
            for field,value in (('persistence_geometry',anchor),('target_geometry',target)):
                expected = producer.geometry(value,row,support)
                require(set(record['references'][field]) == set(expected),'reference geometry fields differ')
                for metric,value in expected.items():close(record['references'][field][metric],value,field+'/'+metric)
            expected_donor = None
            if split == 'val':
                donor = rows[donors[index]]
                expected_donor = dict(row_index=donors[index],key=list(producer.key(donor)),oracle_H_sha256=state_sha256({'H':donor['oracle_H']}))
            require(record['wrong_horizon_donor'] == expected_donor,'wrong-horizon donor identity differs')
            require(set(record['routes']) == set(coords['routes']) == set(numeric.ARMS),'raw route set differs')
            for arm in numeric.ARMS:
                require(set(record['routes'][arm]) == set(coords['routes'][arm]) == set(numeric.ROLES),'role set differs')
                for role in numeric.ROLES:
                    leaf_set = record['routes'][arm][role];outputs = coords['routes'][arm][role]
                    if split == 'train' and role != 'oracle':
                        require(leaf_set is None and outputs is None,'unsupported training role must be explicit null')
                        continue
                    require(set(leaf_set) == set(outputs) == set(numeric.STAGES),'stage set differs')
                    for stage in numeric.STAGES:
                        leaf = leaf_set[stage];value = outputs[stage]
                        n_members = audit_leaf(leaf,role,persistence)
                        require(isinstance(value,torch.Tensor) and value.dtype == torch.float32 and value.device.type == 'cpu'
                            and value.requires_grad is False and value.shape == (n_members,len(mask),3)
                            and bool(torch.isfinite(value[:,mask]).all()),'coordinate member tensor invalid')
                        require(leaf['output_tensor_sha256'] == state_sha256({'CA':value}),'coordinate output hash differs')
                        require(leaf['geometry_counts'] == {k:support[2][k] for k in
                            ('n_legacy_edges','n_psf_edges','n_canonical_nonlocal_pairs','n_psf_nonlocal_pairs')},'geometry denominator counts differ')
                        # Frozen scorer is replayed on saved coordinates. This is an
                        # arithmetic/provenance audit, not a new model inference.
                        expected = producer.score_leaf(value,row,support,forecast=role == 'forecast')
                        require(set(leaf) == set(expected),'raw leaf schema differs')
                        for metric,v in expected['scalars'].items():close(leaf['scalars'][metric],v,'coordinate rescore/'+metric)
                        for actual_member,expected_member in zip(leaf['per_member'],expected['per_member']):
                            require(set(actual_member) == set(expected_member),'member fields differ')
                            for metric,v in expected_member.items():close(actual_member[metric],v,'member coordinate rescore/'+metric)
                        if role == 'anchor_identity':
                            require(torch.equal(value[0,mask],row['observed_backbone'][0,mask,1,:]),'anchor-H identity is not preserved')
                        leaves += 1;members += n_members
    require([len(mismatch[k]) for k in ('train','val')] == [7,0],'legacy/PSF mismatch count changed')
    require(leaves == 7800 and members == 21000,'raw total leaf/member support differs')
    return dict(window_counts={'train':500,'val':200},leaf_count=leaves,member_coordinate_sets=members,
        legacy_psf_mismatch_windows=mismatch,coordinate_rescoring='all members and all three forecast proper scores; frozen scorer reused',
        sidecar_reference_tensors_bitwise_equal_cache=True)


def verify_statistics(inputs):
    registered = inputs.read(BASE/'summary.protocol.json',STATISTICS_SHA)
    require(all(registered.get(k) == v for k,v in numeric.contract().items()),'frozen numerical contract differs')
    for path,digest in registered['source_hashes'].items():inputs.track(path,digest)
    for path,digest in registered['input_hashes'].items():inputs.track(path,digest)
    return registered


def register():
    require(not local(IMPLEMENTATION).exists(),'summary implementation registration already exists')
    inputs = Inputs();verify_statistics(inputs)
    protocol = inputs.read(BASE/'protocol.json')
    require(protocol['schema_version'] == producer.VERSION and protocol['hyperparameters'] == producer.HYPER,
            'producer protocol differs from imported frozen implementation')
    audit_frozen(protocol,inputs)
    # Only file identities/existence are observed here. No score JSON, cache
    # tensors, or decoder checkpoints are deserialized by registration.
    outputs = [BASE/name for name in ('smoke.json','zero_update_full.json','raw.json','score_coordinates.pt','summary.json')]
    presence = {str(path):local(path).exists() for path in outputs}
    require(not presence[str(BASE/'summary.json')],'formal summary already exists')
    sources = dict(protocol['source_hashes'])
    for name in (SCRIPT,TEST,*statistics_registration.SOURCES):sources[name] = sha(name)
    snapshot = IMPLEMENTATION/'source_snapshot'
    producer.snapshot(snapshot,sources)
    record = dict(schema_version=VERSION,registered_utc=now(),statistics_protocol=str(BASE/'summary.protocol.json'),
        statistics_protocol_sha256=STATISTICS_SHA,producer_protocol=str(BASE/'protocol.json'),
        producer_protocol_sha256=sha(BASE/'protocol.json'),output=str(BASE/'summary.json'),
        source_hashes=sources,source_snapshot=str(snapshot),
        input_hashes={str(path.relative_to(REPO)):digest for path,digest in inputs.hashes.items()},
        output_existence_at_registration=presence,
        registration_context='Implementation fixed before inspecting new probe scores. Frozen numerical estimands '
            'were registered earlier. This register invocation only reads protocols and hashes existing source/input bytes; '
            'it does not deserialize cache/model tensors or performance outputs.',
        numerical_implementation='Reuse the unchanged ca_local_residual_statistics_v1.analysis:1501 requested cells, '
            '3480 descriptive cells; same finite-support and explicit-null rules.',
        audit_scope=dict(actual_processes='smoke and run; run contains two sequential training stages and evaluation, not three child processes',
            parameter_state='exact D0 initialization; zero384 residual; centers buffer unchanged; full optimizer500 moments and parameter names',
            stream='exact ordered500 cache rows and tensor fingerprints in both arms; 500 optimizer calls,499 nonzero learning rates',
            coordinates='all7800 leaves/21000 member coordinate sets, output hashes, cache reference bitwise equality, frozen scalar rescoring',
            initial_proof='full700 CPU initial identity records match baseline output hashes; first4 CPU three-head zero-update replay',
            source_scope='current producer/wrapper/metric source plus immutable historical cache source snapshots; no MD data reopened'),
        limitations=['oracle reconstruction uses future H, not a forecast',
            'forecast12 H draws remain from the original frozen sampler; only decoder compatibility is measured',
            'single adaptive development probe; no seven-metric, full T1, T2A, T3, or BioEmu comparison',
            'historical HDF provenance is cached tensor fingerprints and metadata/stat records, not a retrospective full-file digest'],
        test_opened=False,bioemu2_rerun=False,hdf_opened=False,new_sampling=False)
    inputs.verify()
    write_new(IMPLEMENTATION/'protocol.json',record)
    return record


def audit_implementation(inputs):
    record = inputs.read(IMPLEMENTATION/'protocol.json')
    require(record['schema_version'] == VERSION and record['output'] == str(BASE/'summary.json')
        and record['statistics_protocol_sha256'] == STATISTICS_SHA,'summary implementation registration differs')
    audit_frozen(record,inputs)
    protocol = inputs.read(BASE/'protocol.json',record['producer_protocol_sha256'])
    require(protocol['schema_version'] == producer.VERSION and protocol['hyperparameters'] == producer.HYPER
        and protocol['model_config'] == producer.CONFIG and protocol['statistics_sha256'] == STATISTICS_SHA,
        'producer semantics differ')
    require(protocol['arms'] == list(producer.ARMS) and protocol['routes'] == list(numeric.ARMS)
        and protocol['tasks'] == list(numeric.ROLES) and protocol['stages'] == list(numeric.STAGES), 'producer route/role contract differs')
    for field in ('test_opened','hdf_opened','new_sampling','encoder_instantiated','kernel_instantiated'):
        require(protocol[field] is False,'forbidden producer scope: '+field)
    require(protocol['forbidden_forward_arguments'] == ['target_backbone','target_atom_mask','chemical_aatype','peptide_edges'],
            'future-coordinate/chemical forward boundary differs')
    audit_frozen(protocol,inputs)
    verify_statistics(inputs)
    return record,protocol


def execution_gate(inputs,protocol,protocol_sha):
    records = {}
    immutable = {**protocol['input_hashes'],**protocol['source_hashes']}
    for phase in ('smoke','run'):
        record = inputs.read(BASE/f'{phase}.execution.json')
        records[phase] = audit_execution(record,phase,protocol_sha,protocol['source_hashes'][producer.SCRIPT])
        for item in [record]+record.get('stages',[]):
            require(item['frozen_before'] == item['frozen_after'] == immutable,'execution before/after source/input identities differ')
        started = inputs.read(BASE/f'{phase}.started.json')
        require(all(record.get(k) == v for k,v in started.items()),'started/final execution evidence differs')
    require(datetime.fromisoformat(records['smoke']['completed_utc']) <= datetime.fromisoformat(records['run']['started_utc']),
            'training started before smoke completed')
    return records


def json_lines(inputs,path,expected):
    lines = inputs.track(path,expected).read_text().splitlines()
    require(all(line.strip() for line in lines),'empty training evidence line')
    return [json.loads(line) for line in lines]


def initial_identity(initial,d0):
    component = initial['component_checkpoint']
    require(component['decoder_mode'] == 'geometric_ca' and component['calibration_args']['bond_projection_steps'] == 32,
            'frozen component decoder/projection differs')
    return dict(initial_checkpoint_sha256=producer.INITIAL_SHA,inference_tensor_sha256=producer.inference_hash(initial),
        encoder_tensor_sha256=state_sha256(component['encoder']),kernel_tensor_sha256=state_sha256(initial['kernel']),
        decoder_tensor_sha256=state_sha256(d0),complete_component_config_sha256=hashlib.sha256(producer.canonical({
            k:component[k] for k in ('checkpoint_kind','decoder_mode','calibration_args')}).encode()).hexdigest(),
        projection_steps=32,encoder_instantiated=False,kernel_instantiated=False)


def audit_zero_update(evidence,rows,initial_states,*,sidecar=None):
    require(evidence['all_three_raw_and_projected_bitwise_equal'] is True and evidence['zero_additional_parameters'] == 384,
            'initial zero-update equivalence missing')
    require(evidence['model_sha256'] == {a:state_sha256(s) for a,s in initial_states.items()},'initial model state hashes differ')
    require(len(evidence['rows']) == len(rows),'zero-update proof is incomplete')
    for index,(record,row) in enumerate(zip(evidence['rows'],rows)):
        require(record['key'] == list(producer.key(row)) and record['input_tensor_sha256'] == producer.row_fingerprint(row)
            and set(record['output_sha256']) == set(numeric.STAGES),'zero-update proof row differs')
        if sidecar is not None:
            for stage in numeric.STAGES:
                value = sidecar[index]['routes']['baseline']['oracle'][stage]
                require(record['output_sha256'][stage] == state_sha256({'CA':value}),'baseline output differs from original zero-update proof')


def build_summary():
    require(not local(BASE/'summary.json').exists(),'refusing to overwrite summary')
    require(not torch.cuda.is_initialized(),'summary is CPU-only')
    torch.set_num_threads(2)
    inputs = Inputs();registration,protocol = audit_implementation(inputs)
    protocol_sha = registration['producer_protocol_sha256']
    # Hard gate precedes any performance JSON or cache/checkpoint tensor read.
    execution = execution_gate(inputs,protocol,protocol_sha)
    initial = torch.load(inputs.track(producer.INITIAL,producer.INITIAL_SHA),map_location='cpu',weights_only=False)
    d0 = initial['component_checkpoint']['delta_ca_decoder'];identity = initial_identity(initial,d0)
    caches = {split:torch.load(inputs.track(path,producer.CACHE_SHA[split]),map_location='cpu',weights_only=False)
        for split,path in producer.CACHES.items()}
    for split,path in producer.CACHES.items():audit_cache_provenance(caches[split],path,inputs,initial)
    require(not ({r['metadata']['protein'] for r in caches['train']['rows']}
                 & {r['metadata']['protein'] for r in caches['val']['rows']}),'training/validation proteins overlap')
    models,zero_replay = producer.zero_update_check(d0,caches['train']['rows'][:4])
    initial_states = {a:model.state_dict() for a,model in models.items()}
    initial_audit = audit_initial_states(initial_states,d0)
    smoke = inputs.read(BASE/'smoke.json',execution['smoke']['output_sha256'])
    require(smoke['schema_version'] == producer.VERSION and smoke['phase'] == 'smoke' and smoke['optimizer_steps'] == 0
        and smoke['optimizer_instantiated'] is False and smoke['model_state_unchanged'] is True
        and smoke['input_caches_unchanged'] is True and smoke['cuda_initialized'] is False
        and smoke['initial_identity'] == identity and smoke['first4_rows'] is True,'smoke scope/identity differs')
    require(smoke['zero_update'] == zero_replay,'actual first4 CPU zero-update replay differs from smoke')
    require(len(smoke['checks']) == 8,'smoke gradient support differs')
    for index,check in enumerate(smoke['checks']):
        arm = producer.ARMS[index//4];row_index = index%4
        require(check['arm'] == arm and check['row_index'] == row_index
            and check['key'] == list(producer.key(caches['train']['rows'][row_index])),'smoke row identity differs')
        require(numeric.number(check['loss']) >= 0 and numeric.number(check['gradient_norm']) >= 0,'invalid smoke gradient/loss')
        require(type(check['residual_gradient_nonzero']) is bool if arm == 'local_residual'
            else check['residual_gradient_nonzero'] is None,'smoke residual gradient evidence differs')
    require(any(c['residual_gradient_nonzero'] for c in smoke['checks'] if c['arm'] == 'local_residual'),
            'smoke never observed a nonzero gradient to the added residual')
    checkpoints = {};training_audits = {};batch_paths = []
    for arm,stage in zip(producer.ARMS,execution['run']['stages'][:2]):
        require(inputs.read(BASE/f'{arm}.training.json') == stage,'training sidecar differs from final execution')
        checkpoint = torch.load(inputs.track(stage['checkpoint'],stage['checkpoint_sha256']),map_location='cpu',weights_only=False)
        checkpoints[arm] = checkpoint
        evidence = audit_checkpoint(checkpoint,initial_states[arm],arm,protocol,protocol_sha)
        require(stage['active_optimizer_parameters'] == evidence['optimizer']['active_parameter_tensors'],'optimizer support self-report differs')
        directory = ARTIFACTS/arm
        metrics = json_lines(inputs,directory/'metrics.jsonl',stage['metrics_sha256'])
        batches = json_lines(inputs,directory/'batches.jsonl',stage['batches_sha256'])
        evidence['training_stream'] = audit_training_records(metrics,batches,caches['train']['rows'])
        training_audits[arm] = evidence;batch_paths.append(local(directory/'batches.jsonl'))
    require(batch_paths[0].read_bytes() == batch_paths[1].read_bytes(),'arms consumed different ordered inputs')
    require(tree_equal(checkpoints['base_tuned']['rng_state'],checkpoints['local_residual']['rng_state']),
            'final arm RNG states differ under deterministic cached training')
    raw = inputs.read(BASE/'raw.json',execution['run']['output_sha256'])
    evaluation_stage = execution['run']['stages'][-1]
    require(raw['coordinate_sidecar'] == {k:evaluation_stage[k] for k in ('path','sha256')},'coordinate sidecar provenance differs')
    sidecar = torch.load(inputs.track(raw['coordinate_sidecar']['path'],raw['coordinate_sidecar']['sha256']),map_location='cpu',weights_only=False)
    raw_audit = audit_raw(raw,sidecar,caches,identity,protocol,protocol_sha)
    zero_meta = execution['run']['zero_update_full']
    require(zero_meta['path'] == str(BASE/'zero_update_full.json'),'full initial proof path differs')
    full_zero = inputs.read(zero_meta['path'],zero_meta['sha256'])
    require(full_zero['schema_version'] == producer.VERSION and full_zero['n_train'] == 500 and full_zero['n_val'] == 200
        and full_zero['optimizer_steps'] == 0 and full_zero['initial_identity'] == identity,'full initial proof budget/identity differs')
    audit_zero_update(full_zero['zero_update'],caches['train']['rows']+caches['val']['rows'],initial_states,
        sidecar=sidecar['per_window']['train']+sidecar['per_window']['val'])
    # All scientific estimands are delegated unchanged to the previously
    # registered module after provenance and arithmetic gates pass.
    statistics = numeric.analysis(raw)
    inputs.verify()
    require(not torch.cuda.is_initialized(),'CUDA initialized during CPU summary')
    return dict(schema_version=VERSION,completed_utc=now(),statistics_protocol=str(BASE/'summary.protocol.json'),
        statistics_protocol_sha256=STATISTICS_SHA,implementation_protocol=str(IMPLEMENTATION/'protocol.json'),
        implementation_protocol_sha256=sha(IMPLEMENTATION/'protocol.json'),
        provenance=dict(input_hashes={str(p.relative_to(REPO)):h for p,h in inputs.hashes.items()},
            frozen_inputs_sources_verified=True,executions=execution,initial_identity=identity,
            initial_state=initial_audit,initial700_outputs_match_baseline=True,first4_zero_update_replayed=True,
            training=training_audits,raw_coordinates=raw_audit,final_rng_states_equal=True,
            reused_numeric_source_sha256=sha('scripts/ca_local_residual_statistics_v1.py')),
        limitations=registration['limitations'],**statistics)


def execute():
    require(not local(BASE/'summary.json').exists(),'refusing to overwrite summary')
    started = dict(pid=os.getpid(),started_utc=now(),command=sys.argv,source_sha256=sha(SCRIPT),
        protocol_sha256=sha(IMPLEMENTATION/'protocol.json'),output=str(BASE/'summary.json'),cpu_threads=2)
    write_new(IMPLEMENTATION/'started.json',started);start = time.monotonic()
    try:
        result = build_summary();write_new(BASE/'summary.json',result)
        write_new(IMPLEMENTATION/'execution.json',dict(started,status='completed',exit_code=0,
            completed_utc=now(),seconds=time.monotonic()-start,output_sha256=sha(BASE/'summary.json')))
    except BaseException as error:
        write_new(IMPLEMENTATION/'execution.json',dict(started,status='failed',exit_code=1,
            completed_utc=now(),seconds=time.monotonic()-start,error=f'{type(error).__name__}: {error}'))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('register','run'),required=True)
    args = parser.parse_args()
    require(Path.cwd().resolve() == REPO,'run from repository root')
    if args.mode == 'register':
        register();print(json.dumps(dict(protocol=str(IMPLEMENTATION/'protocol.json'),sha256=sha(IMPLEMENTATION/'protocol.json'))))
    else:execute()


if __name__ == '__main__':main()
