"""Independent NumPy audit of frozen-H predictor calibration (prospective).

Reuse only frozen NumPy/stdlib scalar and byte-audit utilities. No producer
scoring imports, Torch, checkpoint deserialization, encoder or optimizer replay.
Registration is before score, and may follow predictor training; disclose the
actual boundary. Sources and prior artifacts remain immutable.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
COMMON_PATH=ROOT/'run_scripts/verify_stage_a_relpos_adaptation_v2.py'
COMMON_SHA='86ae6b3938f98897a5f5e630cac5182055741a2287da9feb99d8856f8d4167d0'
if hashlib.sha256(COMMON_PATH.read_bytes()).hexdigest()!=COMMON_SHA:
    raise ValueError('frozen NumPy utility source changed')
_spec=importlib.util.spec_from_file_location('_frozen_relpos_numpy_v2',COMMON_PATH)
common=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(common)
Audit=common.Audit;need=common.need;sha=common.sha;path=common.path;write=common.write
canonical=common.canonical;timestamp=common.timestamp;digest=common.digest
metrics=common.metrics;log_link=common.log_link
PRODUCER=ROOT/'results/hs_joint_20261101/frozen_h_predictor_calibration_v1'
OUTPUT=ROOT/'results/hs_joint_20261101/frozen_h_predictor_calibration_verification_v1'
MAIN_PROTOCOL_SHA='PENDING'
VERSION='frozen_h_predictor_calibration_independent_verification_v1'
SEEDS=(42,43)
ROLES=('future_H','observed_H_substitution','raw_initial','latent','physical')
HORIZONS=(1,4,16,64)
PHASES=('smoke','prepare','fit42','fit43','train_latent','train_physical','score')
METRICS=('log_mse','mae_angstrom','mse_angstrom2','rmse_angstrom')
LIMITS=('Independent saved FP32 score-array to FP64 metric/bootstrap replay and hash-linked recorded lifecycle. '
        'No model, encoder, OPF, predictor, optimizer or random-draw numerical replay; no HDF reads. '
        'Frozen tensors/RNG equality is checked as producer records and artifact byte hashes. '
        'One seed, inspected TRAIN panel and fixed readout transfer; intervals are descriptive, not multiplicity-adjusted selection.')


def now():return datetime.now(timezone.utc).isoformat()


def selected_arrays(file):
    # Do not materialize the saved H/feature members.
    keys=['mask','pair_indices','target_log','target_distance','observed_distance','persistence_log']
    keys += [f'seed{seed}__{role}__predicted_log' for seed in SEEDS for role in ROLES]
    with np.load(file,allow_pickle=False) as saved:return {k:saved[k].copy() for k in keys}


def score_arrays(arrays):
    mask=np.asarray(arrays['mask']);pairs=np.asarray(arrays['pair_indices']);n=len(pairs)
    need(mask.dtype==np.bool_ and mask.ndim==1 and mask.any(),'invalid fixed observed support')
    need(pairs.dtype==np.int64 and pairs.shape==(n,2) and n>0 and (pairs>=0).all()
         and (pairs<len(mask)).all() and (pairs[:,0]<pairs[:,1]).all(),'invalid pair support')
    need(mask[pairs].all() and len(np.unique(pairs,axis=0))==n
         and np.array_equal(np.lexsort((pairs[:,1],pairs[:,0])),np.arange(n)),'pairs invalid, duplicated or unordered')
    target=arrays['target_distance'];logs=arrays['target_log'];observed=arrays['observed_distance'];persistent=arrays['persistence_log']
    need(all(np.asarray(x).shape==(n,) for x in (target,logs,observed,persistent)),'label pair count differs')
    log_link(logs,target);log_link(persistent,observed)
    result={}
    for seed in SEEDS:
        for role in ROLES:result[f'{seed}/{role}']=metrics(arrays[f'seed{seed}__{role}__predicted_log'],logs,target)
    result['CA_persistence']=metrics(persistent,logs,target,pred_distance=observed)
    return result


def panel(records):
    need(len(records)==100 and [r['cache_row_index'] for r in records]==list(range(400,500)),'exact heldout100 required')
    need(len({r['metadata']['protein'] for r in records})==100
         and [r['metadata']['horizon'] for r in records]==list(HORIZONS)*25,'protein/horizon panel differs')
    keys={f'{seed}/{role}' for seed in SEEDS for role in ROLES}|{'CA_persistence'}
    for r in records:
        need(set(r['scores'])==keys,'two-head score roles differ')
        for value in r['scores'].values():
            need(type(value['n_pairs']) is int and value['n_pairs']==r['n_pairs']>0,'pair counts differ')
            need(type(value['negative_count']) is int and 0<=value['negative_count']<=value['n_pairs'],'invalid negative count')
            need(all(math.isfinite(value[k]) and value[k]>=0 for k in METRICS),'invalid score')
    return sorted(keys)


def aggregate(records):
    keys=panel(records);out={}
    for group in ('overall',*(f'h{h}' for h in HORIZONS)):
        rows=records if group=='overall' else [r for r in records if r['metadata']['horizon']==int(group[1:])]
        out[group]={}
        for key in keys:
            values=[r['scores'][key] for r in rows]
            means={k:float(np.mean([v[k] for v in values],dtype=np.float64)) for k in METRICS[:3]}
            out[group][key]=dict(**means,rmse_angstrom=math.sqrt(means['mse_angstrom2']),n_proteins=len(rows),
                n_pairs=sum(v['n_pairs'] for v in values),negative_count=sum(v['negative_count'] for v in values),
                mean_protein_negative_fraction=float(np.mean([v['negative_count']/v['n_pairs'] for v in values])))
    return out


def contrast_pairs():
    result={}
    for seed in SEEDS:
        result[f'{seed}/physical-minus-latent']=(f'{seed}/physical',f'{seed}/latent')
        for role in ('latent','physical','raw_initial'):
            result[f'{seed}/{role}-minus-observed_H_substitution']=(f'{seed}/{role}',f'{seed}/observed_H_substitution')
            result[f'{seed}/{role}-minus-CA_persistence']=(f'{seed}/{role}','CA_persistence')
    return result


def paired_contrasts(records,contrasts):
    """Same fixed stratified draws; explicit score-key pairs, no fitted targets."""
    keys=panel(records);rng=np.random.default_rng(20261102);draws=[]
    for h in HORIZONS:
        stratum=np.flatnonzero([r['metadata']['horizon']==h for r in records])
        draws.append(stratum[rng.integers(25,size=(10000,25))])
    indices=np.concatenate(draws,axis=1);result={}
    for name,(left,right) in contrasts.items():
        need(left in keys and right in keys,'contrast score keys differ');result[name]={}
        for metric in METRICS:
            field='mse_angstrom2' if metric=='rmse_angstrom' else metric
            a=np.asarray([r['scores'][left][field] for r in records],dtype=np.float64)
            b=np.asarray([r['scores'][right][field] for r in records],dtype=np.float64)
            av=a[indices].mean(1);bv=b[indices].mean(1);point=float(a.mean()-b.mean())
            if metric=='rmse_angstrom':av=np.sqrt(av);bv=np.sqrt(bv);point=math.sqrt(a.mean())-math.sqrt(b.mean())
            bounds=np.percentile(av-bv,[2.5,97.5],method='linear')
            result[name][metric]=dict(difference=point,interval95=[float(x) for x in bounds])
    return result


# Lifecycle/schema adapter is completed against the author's frozen producer
# fields before registration. No main() exists in this draft, so it cannot run.
