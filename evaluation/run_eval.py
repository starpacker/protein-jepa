"""Causal autonomous/direct evaluation with a fixed real-anchor coordinate gauge.

Headline results require verified representation bundles. Explicit legacy/mock
runs write diagnostics only and cannot appear in the headline aggregate.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time
import numpy as np
import torch
import yaml
from evaluation.data import EvalProtocol, iter_windows
from evaluation.geometry import (aligned_rmsd, ca_from_backbone, common_anchor_rmsd,
    displacement_msd, physical_distance_energy_score,ca_geometry_summary,ca_observable_errors)
from evaluation.loaders import build_bundle, sha256_file
from evaluation.metrics import WindowResult, aggregate, protein_summaries,paired_variant_comparisons
from evaluation import variants as V


def load_config(path):
    with open(path) as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError('evaluation config must be a mapping')
    return cfg


def _git(*args):
    try:
        return subprocess.check_output(['git', *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def source_provenance():
    root = Path(__file__).resolve().parents[1]
    paths = sorted([*root.joinpath('evaluation').rglob('*.py'),
                    *root.joinpath('protein_jepa_frame').rglob('*.py')])
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode()+b'\0')
        digest.update(path.read_bytes())
    return dict(git_sha=_git('rev-parse','HEAD'), git_status=_git('status','--porcelain'),
                source_sha256=digest.hexdigest())


def set_determinism(seed, deterministic=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic


def window_seed(seed, window):
    key = (seed,window.protein_id,window.trajectory_id,window.temperature_k,
           window.anchor_frame,window.target_frame)
    return int.from_bytes(hashlib.sha256(json.dumps(key).encode()).digest()[:8],'little') % (2**31-1)


def _effective_config(cfg, overrides):
    out = dict(cfg)
    for key,value in overrides.items():
        if value is not None:
            out[key] = value
    return out


def audit_split(cfg):
    """A temperature filter is a subset, unless training exclusion is verified."""
    from protein_jepa_frame.data.manifest import load_manifest
    from protein_jepa_frame.data.protocol import validate_manifest_split
    val = load_manifest(cfg['manifest'])
    trains = cfg.get('train_manifests', [])
    if isinstance(trains,str): trains = [trains]
    train_records = [r for p in trains for r in load_manifest(p)]
    val_proteins = {r.protein_id for r in val}
    train_proteins = {r.protein_id for r in train_records}
    val_temps = {float(r.temperature_k) for r in val if r.temperature_k is not None}
    if cfg.get('temperatures'):
        val_temps &= set(map(float,cfg['temperatures']))
    train_temps = {float(r.temperature_k) for r in train_records if r.temperature_k is not None}
    requested = cfg.get('split','diagnostic')
    protein_split_audits = []
    chronological_split_audits=[]
    if requested in {'chronological','within_protein_chronological'}:
        if not trains:
            raise ValueError('chronological evaluation requires actual upstream training manifests')
        from evaluation.splits import validate_chronological_split
        chronological_split_audits=[dict(manifest=p,**validate_chronological_split(p,cfg['manifest'])) for p in trains]
    if requested in {'protein_holdout','protein_and_temperature_holdout'}:
        if not trains:
            raise ValueError('protein holdout requires declared disjoint training manifests')
        # Apply the same physical-identity audit as the training and H gate
        # paths: changing manifest labels cannot hide a reused domain or file.
        protein_split_audits = [dict(manifest=p, **validate_manifest_split(p,cfg['manifest']))
                                for p in trains]
    if requested in {'temperature_holdout','protein_and_temperature_holdout'}:
        if not trains or val_temps & train_temps:
            raise ValueError('temperature holdout requires exclusion from every declared training manifest; a val filter is only a temperature subset')
    return dict(name=requested, train_manifests={p:sha256_file(p) for p in trains},
                protein_overlap=len(val_proteins&train_proteins),
                protein_split_audits=protein_split_audits,
                chronological_split_audits=chronological_split_audits,
                train_temperatures=sorted(train_temps), evaluation_temperatures=sorted(val_temps),
                training_exclusion_verified=bool(trains))


def _batch(clip, device):
    return {k:v.unsqueeze(0).to(device) if torch.is_tensor(v) else v for k,v in clip.items()}


def run(cfg, cli_overrides=None):
    cfg = _effective_config(cfg, cli_overrides or {})
    seed = int(cfg.get('seed',0))
    set_determinism(seed,cfg.get('deterministic',True))
    device = torch.device(cfg.get('device','cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda':
        import os
        if cfg.get('deterministic',True) and os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in {':4096:8',':16:8'}:
            raise ValueError('deterministic CUDA evaluation requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before process launch')
    source_before = source_provenance()
    split_audit = audit_split(cfg)
    from evaluation.preregister import verify_registration
    registration=verify_registration(cfg)
    names = cfg.get('variants',['persist'])
    if not names or len(set(names)) != len(names):
        raise ValueError('variants must be nonempty and unique')
    model_cfg = cfg.get('models',{})
    bundles = {}
    for name in names:
        if name == 'persist':
            bundles[name] = None
            continue
        spec = dict(model_cfg)
        spec.update(cfg.get('variant_models',{}).get(name,{}))
        bundles[name] = build_bundle(variant=name,device=device,
            **{k:spec.get(k) for k in ['stage_a_ckpt','b1a_ckpt','b1b_ckpt','direct_ckpt','stage_a_predictor_ckpt',
                                        'legacy_encoder_ckpt','legacy_predictor_output']},
            allow_legacy=bool(cfg.get('allow_legacy',False)), mock_encoder=bool(cfg.get('mock_encoder',False)),
            diagnostic_comparison=bool(cfg.get('diagnostic_comparison',False)))
    diagnostic = bool(cfg.get('diagnostic_comparison',False)) or cfg.get('split','diagnostic') == 'diagnostic' or any(
        b.diagnostic_only for b in bundles.values() if b is not None)
    if not diagnostic or cfg.get('diagnostic_comparison',False):
        declared_paths = cfg.get('train_manifests',[])
        if isinstance(declared_paths,str): declared_paths = [declared_paths]
        declared = {str(Path(p).resolve()) for p in declared_paths}
        for b in bundles.values():
            if b is None: continue
            records = b.provenance.get('training_manifests',[])
            if (not records or b.provenance.get('synthetic_lineage')
                    or not b.provenance.get('training_lineage_complete')
                    or not b.provenance.get('upstream_source_hash_verified')
                    or not all(r['saved_hash_verified'] for r in records)):
                raise ValueError('comparison evaluation requires hash-locked real training lineage from every stage')
            if any(r['manifest'] not in declared for r in records):
                raise ValueError('declare every actual upstream training manifest with --train-manifest')
        ids = {b.representation_contract['representation_id'] for b in bundles.values() if b is not None}
        if len(ids) > 1:
            raise ValueError('comparison variants must share the identical encoder/normalizer/decoder representation')
    proto = EvalProtocol(manifest=cfg['manifest'],horizons_frames=tuple(cfg['horizons_frames']),
        context_frames=cfg.get('context_frames',8),context_stride=cfg.get('context_stride',1),
        frame_stride=cfg.get('frame_stride',1),max_proteins=cfg.get('max_proteins'),
        max_residues=cfg.get('max_residues'),
        max_trajectories_per_protein=cfg.get('max_trajectories_per_protein'),
        max_anchors_per_trajectory=cfg.get('max_anchors_per_trajectory'),
        temperatures=tuple(cfg['temperatures']) if cfg.get('temperatures') else None,
        anchor_stride=cfg.get('anchor_stride',1),skip_head_frames=cfg.get('skip_head_frames',0),
        seed=seed,temporal_lags=tuple(cfg.get('temporal_lags',[])),emit_multi_lag=cfg.get('emit_multi_lag',False))
    rows = []
    n_windows = 0
    previous = None
    contexts = {}
    started = time.monotonic()
    scale = float(cfg.get('physical_distance_scale_angstrom',1.))
    max_pairs = int(cfg.get('physical_distance_max_pairs',512))
    oracle_limit = cfg.get('oracle_max_horizon',max(cfg['horizons_frames']))
    for w in iter_windows(proto):
        n_windows += 1
        key = (w.protein_id,w.trajectory_id,w.anchor_frame)
        ctx = _batch(w.context,device)
        target = _batch(w.target,device)
        # A residue must exist throughout the history and in the target.
        valid = ctx['residue_mask'].all(dim=1)[0].bool() & target['residue_mask'][0,0].bool()
        if int(valid.sum()) < 3:
            raise ValueError(f'fewer than 3 common valid residues in window {key}')
        anchor = ca_from_backbone(ctx['coords'][:,-1])[0]
        truth = ca_from_backbone(target['coords'])[0,0]
        p_rmsd = common_anchor_rmsd(anchor,anchor,truth,valid)
        p_score = physical_distance_energy_score(anchor[None],truth,valid,scale_angstrom=scale,max_pairs=max_pairs)
        geometry_args=dict(chain_id=ctx['chain_id'][0],residue_index=ctx['residue_index'][0])
        target_geometry=ca_geometry_summary(truth,valid,**geometry_args)
        if key != previous:
            contexts = {}
            previous = key
        R,t = V.anchor_frames(ctx)
        for name in names:
            bundle = bundles[name]
            oracle = anchor_decode = None
            dynamics_stats={}
            if bundle is None:
                predicted = anchor
                samples = anchor[None]
                latent_msds = []
            else:
                # Key by content-verified representation, never assumed lineage labels.
                cache_key = bundle.representation_contract.get('representation_id') if bundle.representation_contract else name
                if cache_key not in contexts:
                    contexts[cache_key] = V.encode_clip(bundle,ctx)
                history = contexts[cache_key]
                h_anchor = history[:,-1]
                if w.horizon <= oracle_limit:
                    true_h = V.encode_clip(bundle,target)[:,0]
                    ca_oracle = V.decode(bundle,true_h,h_anchor,ctx,R,t,valid[None])[0]
                    oracle = common_anchor_rmsd(ca_oracle,anchor,truth,valid)
                    ca_anchor = V.decode(bundle,h_anchor,h_anchor,ctx,R,t,valid[None])[0]
                    anchor_decode = common_anchor_rmsd(ca_anchor,anchor,truth,valid)
                result = V.produce(name,bundle,ctx,history,w.horizon,horizon_ps=w.horizon_ps,
                                   seed=window_seed(seed,w),n_rollout_reps=cfg.get('n_rollout_reps',4))
                predicted,samples = result['ca_pred'],result['ca_samples']
                latent_msds = result['meta'].get('latent_msd_reps',[])
                dynamics_stats=result['meta'].get('dynamics_stats',{})
            if not torch.isfinite(samples).all():
                raise FloatingPointError(f'nonfinite prediction for {name}, window {key}')
            row = WindowResult(protein_id=w.protein_id,trajectory_id=w.trajectory_id,
                temperature_k=w.temperature_k,horizon=w.horizon,horizon_ps=w.horizon_ps,
                anchor_frame=w.anchor_frame,target_frame=w.target_frame,variant=name,
                rmsd=common_anchor_rmsd(predicted,anchor,truth,valid),persist_rmsd=p_rmsd,
                oracle_rmsd=oracle,anchor_decode_rmsd=anchor_decode,
                shape_rmsd=aligned_rmsd(predicted,truth,valid),
                msd=displacement_msd(predicted,anchor,valid),
                sample_rmsds=[common_anchor_rmsd(x,anchor,truth,valid) for x in samples],
                sample_shape_rmsds=[aligned_rmsd(x,truth,valid) for x in samples],
                sample_msds=[displacement_msd(x,anchor,valid) for x in samples],
                physical_energy_score=physical_distance_energy_score(samples,truth,valid,scale_angstrom=scale,max_pairs=max_pairs),
                persist_physical_energy_score=p_score,n_residues=int(valid.sum()),diagnostic_only=diagnostic,
                sampling_seed=window_seed(seed,w),
                geometry=ca_geometry_summary(predicted,valid,**geometry_args),
                sample_geometry=[ca_geometry_summary(x,valid,**geometry_args) for x in samples],
                target_geometry=target_geometry,observables=ca_observable_errors(predicted,truth,valid,**geometry_args),
                sample_observables=[ca_observable_errors(x,truth,valid,**geometry_args) for x in samples],
                dynamics_stats=dynamics_stats)
            row.latent_msds = latent_msds
            rows.append(row)
        if n_windows % 100 == 0:
            print(f'{n_windows} windows, {len(rows)} variant rows, {time.monotonic()-started:.1f}s',flush=True)
    if not rows:
        raise ValueError('evaluation produced no valid windows')
    source_after = source_provenance()
    if source_after['source_sha256'] != source_before['source_sha256']:
        raise RuntimeError('source code changed during evaluation; refuse mixed-version result')
    meta = dict(**source_before,timestamp=datetime.now(timezone.utc).isoformat(),device=str(device),
        python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        cuda=torch.version.cuda,deterministic=cfg.get('deterministic',True),seed=seed,
        effective_config=cfg,window_protocol=asdict(proto),split_audit=split_audit,
        manifest=str(Path(cfg['manifest']).resolve()),manifest_sha256=sha256_file(cfg['manifest']),
        models={v:b.provenance if b else {'variant':'persist'} for v,b in bundles.items()},
        diagnostic_only=diagnostic,headline_eligible=not diagnostic,
        dataset_role=cfg.get('dataset_role','development_validation'),
        diagnostic_comparison=bool(cfg.get('diagnostic_comparison',False)),
        limitations=cfg.get('limitations',[]),
        preregistration=registration,
        metric_protocol={'primary':'common_real_anchor_to_target_CA_RMSD',
            'secondary':'independent_prediction_to_target_shape_RMSD',
            'decoder_pose':'one decoded-current-H to observed-anchor history-only fit; fixed for all futures',
            'uncertainty_unit':'protein; equal-window means within protein',
            'ensemble_mean':'coordinate mean scored separately from samples',
            'proper_score':'empirical energy score of nonlocal CA distances; V statistic',
            'physical_distance_scale_angstrom':scale,'physical_distance_max_pairs':max_pairs})
    return dict(config=cfg,n_windows=n_windows,aggregate=aggregate(rows,seed=seed),
                pairwise_comparisons=paired_variant_comparisons(rows,seed=seed),
                rows=rows,per_protein=protein_summaries(rows),meta=meta)


def _jsonable(value):
    if isinstance(value,dict): return {str(k):_jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_jsonable(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value): return None
    return value


def write_reports(out,outdir):
    path = Path(outdir); path.mkdir(parents=True,exist_ok=True)
    diag = out['meta']['diagnostic_only']
    aggregate_key = 'diagnostic_aggregate' if diag else 'aggregate'
    payload = dict(n_windows=out['n_windows'],meta=out['meta'],**{aggregate_key:out['aggregate']},
                   paired_variant_comparisons=out.get('pairwise_comparisons',[]))
    def dump(name,obj):
        (path/name).write_text(json.dumps(_jsonable(obj),indent=2,sort_keys=True,allow_nan=False)+'\n')
    dump('results.json',payload); dump('meta.json',out['meta'])
    dump('paired_comparisons.json',out.get('pairwise_comparisons',[]))
    with (path/'per_window.jsonl').open('w') as handle:
        for row in out['rows']:
            handle.write(json.dumps(_jsonable(asdict(row)),sort_keys=True,allow_nan=False)+'\n')
    with (path/'per_protein.jsonl').open('w') as handle:
        for row in out['per_protein']:
            handle.write(json.dumps(_jsonable(row),sort_keys=True,allow_nan=False)+'\n')
    with (path/'samples.jsonl').open('w') as handle:
        for row in out['rows']:
            for rep,(rmsd,shape,msd) in enumerate(zip(row.sample_rmsds,row.sample_shape_rmsds,row.sample_msds)):
                sample = dict(protein_id=row.protein_id,trajectory_id=row.trajectory_id,
                    temperature_k=row.temperature_k,anchor_frame=row.anchor_frame,target_frame=row.target_frame,
                    horizon=row.horizon,horizon_ps=row.horizon_ps,variant=row.variant,model_rep=rep,
                    rmsd=rmsd,shape_rmsd=shape,msd=msd,diagnostic_only=row.diagnostic_only)
                sample['sampling_seed']=row.sampling_seed+rep
                if row.sample_geometry: sample['geometry']=row.sample_geometry[rep]
                if row.sample_observables: sample['observables']=row.sample_observables[rep]
                latent = getattr(row,'latent_msds',[])
                if latent: sample['normalized_latent_msd'] = latent[rep]
                handle.write(json.dumps(_jsonable(sample),sort_keys=True,allow_nan=False)+'\n')
    def fmt(x): return '--' if x is None or not np.isfinite(x) else f'{x:.4f}'
    lines = [f"# {'Diagnostic only — ' if diag else ''}Evaluation: {out['config'].get('name','run')}",'',
        f"Windows: {out['n_windows']}; seed: {out['meta']['seed']}; split: {out['meta']['split_audit']['name']}.",
        'Primary error uses one real anchor→target transform; shape RMSD is secondary.',
        'Means/SEM and paired bootstrap use proteins; single-protein runs cannot establish significance.','',
        '| T/K | gap/raw frames | variant | proteins | mean±SEM/Å | Δ mean [95% CI] | oracle/Å | sample RMSD/Å | distance ES |',
        '|---:|---:|---|---:|---:|---|---:|---:|---:|']
    for temp,byh in out['aggregate'].items():
        for horizon,byv in byh.items():
            for variant,stats in byv.items():
                delta,lo,hi = stats['delta_ci']
                lines.append(f"| {temp} | {horizon} | {variant} | {stats['n_proteins']} | {fmt(stats['rmsd_mean'])}±{fmt(stats['rmsd_sem'])} | {fmt(delta)} [{fmt(lo)}, {fmt(hi)}] | {fmt(stats['oracle_median'])} | {fmt(stats['sample_rmsd_mean'])} | {fmt(stats['physical_energy_score_mean'])} |")
    if out.get('pairwise_comparisons'):
        lines.extend(['','Paired model effects (A minus B; lower is better).',
            '', '| T/K | gap | A | B | metric | effect [95% CI] | proteins |',
            '|---:|---:|---|---|---|---|---:|'])
        for r in out['pairwise_comparisons']:
            if r['metric'] not in {'rmsd','physical_energy_score'}: continue
            lo,hi=r['ci95']
            lines.append(f"| {r['temperature_k']} | {r['horizon']} | {r['variant_a']} | {r['variant_b']} | {r['metric']} | {fmt(r['delta_a_minus_b'])} [{fmt(lo)}, {fmt(hi)}] | {r['n_proteins']} |")
    if diag:
        lines.extend(['','These are diagnostic comparisons. A failed/unestablished readout gate remains a limitation; numerical gains do not establish a validated dynamics model.'])
    for limitation in out['meta'].get('limitations',[]):
        lines.append(f'- {limitation}')
    md = '\n'.join(lines)+'\n'; (path/'results.md').write_text(md)
    return md


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True)
    for name in ['outdir','name','manifest-override','device','stage-a-ckpt',
                 'stage-a-predictor-ckpt','b1a-ckpt','b1b-ckpt','direct-ckpt','legacy-encoder-ckpt',
                 'legacy-predictor-output']: ap.add_argument('--'+name)
    ap.add_argument('--train-manifest',action='append',default=None)
    ap.add_argument('--split',choices=['diagnostic','protein_holdout','temperature_subset',
                                      'temperature_holdout','protein_and_temperature_holdout',
                                      'chronological','within_protein_chronological'])
    for name in ['max-proteins','max-residues','anchor-stride','oracle-max-horizon','max-anchors-per-trajectory']:
        ap.add_argument('--'+name,type=int)
    ap.add_argument('--horizons',nargs='+',type=int)
    ap.add_argument('--variants',nargs='+')
    ap.add_argument('--allow-legacy',action='store_true',default=None,help='diagnostic only; never headline')
    args = ap.parse_args()
    cfg = load_config(args.config)
    overrides = dict(max_proteins=args.max_proteins,max_residues=args.max_residues,anchor_stride=args.anchor_stride,
        oracle_max_horizon=args.oracle_max_horizon,max_anchors_per_trajectory=args.max_anchors_per_trajectory,
        name=args.name,manifest=args.manifest_override,device=args.device,
        horizons_frames=args.horizons,variants=args.variants,allow_legacy=args.allow_legacy,
        train_manifests=args.train_manifest,split=args.split)
    cfg = _effective_config(cfg,overrides)
    cfg['models'] = dict(cfg.get('models',{}))
    for key in ['stage_a_ckpt','stage_a_predictor_ckpt','b1a_ckpt','b1b_ckpt','direct_ckpt',
                'legacy_encoder_ckpt','legacy_predictor_output']:
        if getattr(args,key) is not None:
            cfg['models'][key] = getattr(args,key)
    cfg['config_source'] = str(Path(args.config).resolve())
    cfg['config_sha256'] = sha256_file(args.config)
    out = run(cfg)
    directory = args.outdir or str(Path('artifacts/eval')/cfg.get('name',Path(args.config).stem))
    print(write_reports(out,directory))
    print(f'Wrote {directory}')


if __name__ == '__main__':
    main()
