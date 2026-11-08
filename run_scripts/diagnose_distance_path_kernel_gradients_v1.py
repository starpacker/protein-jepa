"""CPU first-four TRAIN five-gradient distance-target audit; no updates."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

import torch

REPO=Path(__file__).resolve().parents[1]
OUT=REPO/'results/hs_joint_20261101/distance_path_kernel_gradients_v1'
VERSION='distance_path_kernel_gradients_v1'
HISTORY='results/hs_joint_20261101/ordered_path_v1/protocol.json'
HISTORY_SHA='a31b87128702cd796b25fbbef60c3bd1245304cfd369c668970e54c72c7235cb'
WINDOW_LOG='artifacts/hs_joint_20261101/ordered_path_v1/marginal_path/training_windows.jsonl'
WINDOW_LOG_SHA='cbcad706d02645312d3a8209f0b18210a375998bb3b532a2f78788f24b8a71cb'
INITIAL='artifacts/hs_joint_20261101/spatial_probe_v1/spatial_exact.pt'
INITIAL_SHA='396e6e4df35019140324484bf8d560a292cf2688a11946f2e0ccde5cd6ea8dda'
TRAIN_INPUTS=(INITIAL,'artifacts/mdcath_manifest_full/train.jsonl',
    'artifacts/stage_a_disp_relpos/checkpoint_latest.pt','artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt')
SOURCES=('run_scripts/diagnose_distance_path_kernel_gradients_v1.py',
    'tests/test_distance_path_kernel_gradients_v1.py','docs/DISTANCE_PATH_KERNEL_GRADIENT_DIAGNOSTIC_V1_20261102.md',
    'protein_jepa_frame/training/physical_distance_path_objective_v1.py','tests/test_physical_distance_path_objective_v1.py',
    'protein_jepa_frame/training/physical_distance_augmented_objective_v1.py','tests/test_physical_distance_augmented_objective_v1.py')
TERMS=('base','absolute_aux','increment_aux','absolute_total','increment_total')
LOSS_KEYS=('loss_base','distance_absolute_es','distance_increment_es','loss_absolute','loss_increment')
PAIRS=(('absolute_aux','base'),('increment_aux','base'),('absolute_aux','increment_aux'),
       ('base','absolute_total'),('base','increment_total'),('absolute_total','increment_total'))
ATOL,RTOL=2e-6,5e-4
KERNEL_TENSORS,KERNEL_ELEMENTS=73,669429


def require(ok,reason):
    if not ok:raise ValueError(reason)


def now():return datetime.now(timezone.utc).isoformat()


def path(value):
    p=Path(value);p=(p if p.is_absolute() else REPO/p).resolve()
    require(p.is_relative_to(REPO) and p.name!='test.jsonl','outside repository or sealed test input')
    return p


def sha(value):
    digest=hashlib.sha256()
    with path(value).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def write_new(file,value):
    encoded=json.dumps(value,indent=2,allow_nan=False)+'\n'
    with Path(file).open('x') as stream:stream.write(encoded)


def file_stat(file):
    stat=Path(file).stat();return dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)


def cpu_guard():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU diagnostic requires CUDA_VISIBLE_DEVICES empty')


def rng_state(generator):
    import random
    import numpy as np
    value=np.random.get_state()
    return dict(python=random.getstate(),numpy_metadata=[value[0],*value[2:]],
        numpy_keys=torch.from_numpy(value[1].copy()),torch=torch.get_rng_state().clone(),
        sampling=generator.get_state().clone())


def rng_sha(value):
    from protein_jepa_frame.utils.contracts import state_sha256
    return state_sha256(value)


def memory_snapshot():
    return dict(process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
                convention='Linux process high-water RSS, not current RSS or isolated-window peak')


def inventory(kernel):
    return [dict(name=n,shape=list(p.shape),dtype=str(p.dtype),numel=p.numel(),requires_grad=p.requires_grad)
            for n,p in kernel.named_parameters()]


def ratio(a,b):return a/b if b else None


def comparison(left,right):
    left,right=left.double(),right.double()
    a,b=float(left.norm()),float(right.norm());dot=float(left@right)
    return dict(dot=dot,cosine=ratio(dot,a*b),left_over_right_norm=ratio(a,b))


def vector_statistics(vectors):
    require(tuple(vectors)==TERMS,'five ordered gradient terms required')
    base=vectors['base']
    require(base.ndim==1 and base.numel()>0 and all(v.shape==base.shape and v.ndim==1
        and bool(torch.isfinite(v).all()) for v in vectors.values()),'gradient shape/nonfinite/empty support')
    return dict(elements=base.numel(),norms={name:float(value.double().norm()) for name,value in vectors.items()},
        comparisons={a+'_vs_'+b:comparison(vectors[a],vectors[b]) for a,b in PAIRS})


def identity(actual,expected):
    actual,expected=actual.double().reshape(-1),expected.double().reshape(-1)
    require(actual.shape==expected.shape and actual.numel()>0 and
        bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),'nonfinite identity or shape mismatch')
    error=actual-expected
    return dict(passed=bool((error.abs()<=ATOL+RTOL*expected.abs()).all()),
        max_absolute_error=float(error.abs().max()),l2_error=float(error.norm()),
        relative_l2_error=ratio(float(error.norm()),float(expected.norm())),atol=ATOL,rtol=RTOL,
        convention='elementwise abs(actual-expected)<=2e-6+5e-4*abs(expected), float64 audit of saved gradients')


def collect_gradients(model,terms,generator):
    """Five grad calls on one graph; unused None stored as flags plus audit zeros."""
    named=list(model.kernel.named_parameters())
    require(named and all(p.requires_grad for _,p in named),'all kernel parameters must be trainable')
    require(not any(p.requires_grad or p.grad is not None for p in model.bundle.parameters()),'component not frozen/gradient-free')
    require(all(p.grad is None and p.device.type=='cpu' for p in model.parameters()),'CPU gradient-free model required')
    losses={name:terms[key] for name,key in zip(TERMS,LOSS_KEYS)}
    require(all(v.ndim==0 and bool(torch.isfinite(v)) and v.requires_grad for v in losses.values()),
            'five finite differentiable scalars required')
    gradients,unused,rng_after={},{},{}
    memory_before=memory_snapshot()
    for index,(name,loss) in enumerate(losses.items()):
        values=torch.autograd.grad(loss,[p for _,p in named],retain_graph=index<len(TERMS)-1,allow_unused=True)
        unused[name]=[n for (n,_),g in zip(named,values) if g is None]
        gradients[name]={n:(torch.zeros_like(p) if g is None else g).detach().cpu().clone()
                         for (n,p),g in zip(named,values)}
        rng_after[name]=rng_state(generator)
    return dict(parameter_names=[n for n,_ in named],shapes={n:list(p.shape) for n,p in named},
        dtypes={n:str(p.dtype) for n,p in named},gradients=gradients,unused=unused,rng_after_gradients=rng_after,
        parameter_gradients_present=any(p.grad is not None for p in model.parameters()),
        memory_before_gradients=memory_before,memory_after_gradients=memory_snapshot())


def summarize_gradients(saved,losses,horizon):
    names=saved['parameter_names']
    require(names and len(names)==len(set(names)),'unique full kernel names required')
    for term in TERMS:
        require(set(saved['gradients'][term])==set(names) and set(saved['unused'][term]).issubset(names),
                'saved gradient inventory differs')
        for name in names:
            value=saved['gradients'][term][name]
            require(list(value.shape)==saved['shapes'][name] and str(value.dtype)==saved['dtypes'][name],
                    'saved gradient shape/dtype differs')
            if name in saved['unused'][term]:require(not bool(value.count_nonzero()),'unused audit zero differs')
    def flatten(selected):
        return {term:torch.cat([saved['gradients'][term][n].reshape(-1) for n in selected]) for term in TERMS}
    vectors=flatten(names);report=vector_statistics(vectors)
    report['identities']={arm:identity(vectors[arm+'_total'],vectors['base'].double()+vectors[arm+'_aux'].double())
                          for arm in ('absolute','increment')}
    if horizon==1:
        report['h1_identity']=dict(auxiliary_gradient=identity(vectors['increment_aux'],vectors['absolute_aux']),
            total_gradient=identity(vectors['increment_total'],vectors['absolute_total']),
            auxiliary_value=identity(torch.tensor(losses['distance_increment_es'],dtype=torch.float64),
                                     torch.tensor(losses['distance_absolute_es'],dtype=torch.float64)),
            total_value=identity(torch.tensor(losses['loss_increment'],dtype=torch.float64),
                                 torch.tensor(losses['loss_absolute'],dtype=torch.float64)))
    else:report['h1_identity']=None
    report['parameters']={term:dict(total=len(names),unused=len(saved['unused'][term]),unused_names=saved['unused'][term],
        nonzero=sum(bool(saved['gradients'][term][n].count_nonzero()) for n in names)) for term in TERMS}
    report['module_groups']={group:dict(parameter_names=[n for n in names if n.split('.')[0]==group],
        **vector_statistics(flatten([n for n in names if n.split('.')[0]==group]))) for group in sorted({n.split('.')[0] for n in names})}
    report['per_parameter']={name:vector_statistics(flatten([name])) for name in names}
    return report


def verify_files(protocol):
    for name,digest in {**protocol['source_hashes'],**protocol['input_hashes']}.items():
        require(sha(name)==digest,'frozen file changed: '+name)
    for field,files in (('source_snapshot',protocol['source_hashes']),('metadata_snapshot',protocol['metadata_snapshots'])):
        for name,digest in files.items():require(sha(path(protocol[field])/name)==digest,'snapshot changed: '+name)
    for name,expected in protocol['selected_hdf5_stats'].items():require(file_stat(name)==expected,'selected HDF identity changed: '+name)


def register():
    require(not OUT.exists(),'refusing existing diagnostic directory')
    require(sha(HISTORY)==HISTORY_SHA and sha(WINDOW_LOG)==WINDOW_LOG_SHA,'historical protocol/window log changed')
    old=json.loads(path(HISTORY).read_text());selected=old['training_windows'][:4]
    require([r['step'] for r in selected]==[1,2,3,4] and [r['horizon'] for r in selected]==[1,4,16,64],
            'first-four historical metadata differs')
    with path(WINDOW_LOG).open() as stream:historical=[json.loads(next(stream)) for _ in range(4)]
    require(all(all(actual[k]==v for k,v in row.items()) and len(actual['actual_batch_sha256'])==64
        for row,actual in zip(selected,historical)),'historical actual batch identities differ')
    metadata={HISTORY:HISTORY_SHA,WINDOW_LOG:WINDOW_LOG_SHA}
    inputs={**metadata,**{name:old['input_hashes'][name] for name in TRAIN_INPUTS}}
    require(inputs[INITIAL]==INITIAL_SHA,'initial checkpoint identity differs')
    for name,digest in inputs.items():require(sha(name)==digest,'input changed: '+name)
    sources=dict(old['source_hashes'])
    for name,digest in sources.items():
        require(sha(name)==digest and sha(path(old['source_snapshot'])/name)==digest,'historical source changed: '+name)
    sources.update({name:sha(name) for name in SOURCES})
    stats={r['record_path']:old['selected_hdf5_stats'][r['record_path']] for r in selected}
    for name,expected in stats.items():require(file_stat(name)==expected,'selected HDF stat changed')
    protocol=dict(schema_version=VERSION,registered_utc=now(),source_hashes=sources,input_hashes=inputs,
        source_snapshot=str((OUT/'source_snapshot').relative_to(REPO)),metadata_snapshots=metadata,
        metadata_snapshot=str((OUT/'metadata_snapshot').relative_to(REPO)),selected_hdf5_stats=stats,
        selected_training_windows=selected,historical_actual_batches=historical,
        training_arguments=old['training_arguments']['marginal_path'],initial_kernel_sha256=old['initial_kernel_sha256'],
        initial_kernel_config=old['initial_kernel_config'],initial_component_identity=old['initial_component_identity'],
        kernel_parameter_tensors=KERNEL_TENSORS,kernel_parameter_elements=KERNEL_ELEMENTS,
        terms=list(TERMS),loss_keys=list(LOSS_KEYS),device='cpu',torch_threads=2,seed=42,n_samples=12,
        contact_estimator='plugin',distance_path_weight=1.,distance_path_objective='absolute',
        gradient_identity_tolerance=dict(atol=ATOL,rtol=RTOL),optimizer_steps=0,
        hdf_evidence='size/mtime before and after plus exact selected batch tensor hash; no full-HDF digest',
        inference='four fixed TRAIN cases, unchanged initial weights, no across-horizon headline/CI/lambda selection/update prediction',
        validation_loaded=False,test_opened=False,bioemu2_rerun=False)
    OUT.mkdir(parents=True,exist_ok=False)
    for directory,files in (('source_snapshot',sources),('metadata_snapshot',metadata)):
        for name,digest in files.items():
            target=OUT/directory/name;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write(path(name).read_bytes())
            require(sha(target)==digest,'file changed during snapshot')
    verify_files(protocol);write_new(OUT/'protocol.json',protocol)
    print(json.dumps(dict(protocol=str(OUT/'protocol.json'),sha256=sha(OUT/'protocol.json'))),flush=True)


def check_audit(report,saved,rng_after_forward):
    require(not saved['parameter_gradients_present'],'autograd populated parameter .grad')
    require(all(rng_sha(value)==rng_sha(rng_after_forward) for value in saved['rng_after_gradients'].values()),
            'gradient recomputation consumed RNG')
    require(all(value['passed'] for value in report['identities'].values()),'additive gradient identity outside fixed tolerance')
    require(report['h1_identity'] is None or all(value['passed'] for value in report['h1_identity'].values()),
            'h1 value/gradient identity outside fixed tolerance')


def execute(protocol):
    cpu_guard()
    from protein_jepa_frame.training import train_lasp_ca_ordered_v1 as training
    from protein_jepa_frame.training.physical_distance_augmented_objective_v1 import physical_distance_augmented_path_losses
    from protein_jepa_frame.training.train_utils import _set_reproducibility
    from protein_jepa_frame.utils.contracts import state_sha256
    from run_scripts.run_hs_target_pilot import component_identity
    torch.set_num_threads(2);_set_reproducibility(42,False)
    args=SimpleNamespace(**{**protocol['training_arguments'],'device':'cpu','torch_threads':2,'output':str(OUT),
        'contact_estimator':'plugin','distance_path_objective':'absolute','distance_path_weight':1.})
    model=training.build_model(args)
    require(state_sha256(model.kernel)==protocol['initial_kernel_sha256'] and
        asdict(model.kernel.cfg)==protocol['initial_kernel_config'],'initial kernel identity differs')
    require(component_identity(model.component_source)==protocol['initial_component_identity'] and
        state_sha256(model.bundle.encoder)==protocol['initial_component_identity']['encoder'] and
        state_sha256(model.bundle.decoder)==protocol['initial_component_identity']['decoder'],
        'initial encoder/decoder metadata or tensors differ')
    parameter_inventory=inventory(model.kernel)
    require(len(parameter_inventory)==KERNEL_TENSORS==protocol['kernel_parameter_tensors'] and
        sum(r['numel'] for r in parameter_inventory)==KERNEL_ELEMENTS==protocol['kernel_parameter_elements'] and
        all(r['requires_grad'] for r in parameter_inventory),'kernel inventory differs')
    initial=state_sha256(model);initial_component=state_sha256(model.bundle)
    write_new(OUT/'initial.json',dict(model_sha256=initial,component_sha256=initial_component,
        kernel_sha256=state_sha256(model.kernel),kernel_config=asdict(model.kernel.cfg),
        component_identity=component_identity(model.component_source),kernel_inventory=parameter_inventory,
        device='cpu',torch_threads=2,memory=memory_snapshot()))
    loaders={h:training.make_loader(args,args.manifest,h,True) for h in args.horizons}
    generator=torch.Generator(device='cpu').manual_seed(42);rows=[]
    for selected,historical in zip(protocol['selected_training_windows'],protocol['historical_actual_batches']):
        verify_files(protocol);step,horizon=selected['step'],selected['horizon']
        require(training.training_identity(loaders[horizon],step,42)==selected,'selected TRAIN metadata differs')
        batch,actual=training.load_training_batch(loaders[horizon],step,42)
        require(actual==historical,'actual TRAIN batch differs from original run')
        write_new(OUT/f'step{step:03d}.started.json',dict(step=step,horizon=horizon,batch=actual,started_utc=now()))
        require(all(p.grad is None for p in model.parameters()),'stale parameter gradients')
        model.train();model.bundle.eval();before=rng_state(generator);memory_before=memory_snapshot();started=time.monotonic()
        terms=physical_distance_augmented_path_losses(model,batch,horizon,args,generator)
        after_forward=rng_state(generator)
        losses={key:float(value.detach()) for key,value in terms.items()}
        saved=collect_gradients(model,terms,generator)
        saved.update(schema_version=VERSION,step=step,horizon=horizon,batch_identity=actual,losses=losses,
            rng_before=before,rng_after_forward=after_forward,memory_before_forward=memory_before)
        # Save actual gradients before numerical or RNG acceptance, including failure evidence.
        sidecar=OUT/f'step{step:03d}.gradients.pt'
        with sidecar.open('xb') as stream:torch.save(saved,stream)
        report=summarize_gradients(saved,losses,horizon)
        row=dict(step=step,horizon=horizon,batch=actual,losses=losses,gradient_statistics=report,
            sidecar=str(sidecar.relative_to(REPO)),sidecar_sha256=sha(sidecar),
            rng_before_sha256=rng_sha(before),rng_after_forward_sha256=rng_sha(after_forward),
            rng_after_gradient_sha256={k:rng_sha(v) for k,v in saved['rng_after_gradients'].items()},
            seconds=time.monotonic()-started,memory_before_forward=memory_before,memory_after_gradients=memory_snapshot(),
            model_sha256=state_sha256(model),component_sha256=state_sha256(model.bundle),optimizer_steps=0)
        write_new(OUT/f'step{step:03d}.json',row)
        check_audit(report,saved,after_forward)
        require(row['model_sha256']==initial and row['component_sha256']==initial_component
            and all(p.grad is None for p in model.parameters()),'model/component tensors or gradients changed')
        verify_files(protocol);rows.append(row)
        print(json.dumps(dict(step=step,horizon=horizon,seconds=row['seconds'],status='completed')),flush=True)
        del terms,saved,batch,report
    return dict(schema_version=VERSION,completed_utc=now(),rows=rows,initial_model_sha256=initial,
        final_model_sha256=state_sha256(model),initial_component_sha256=initial_component,
        final_component_sha256=state_sha256(model.bundle),kernel_inventory=parameter_inventory,
        device='cpu',torch_threads=2,torch=torch.__version__,optimizer_steps=0,
        inference=protocol['inference'],validation_loaded=False,test_opened=False,bioemu2_rerun=False,
        source_hashes=protocol['source_hashes'],input_hashes=protocol['input_hashes'])


def run():
    require((OUT/'protocol.json').exists(),'register before run')
    require(not any((OUT/n).exists() for n in ('started.json','execution.json','result.json')),'refusing rerun')
    started=time.monotonic()
    identity_record=dict(pid=os.getpid(),command=sys.argv,started_utc=now(),protocol_sha256=sha(OUT/'protocol.json'))
    write_new(OUT/'started.json',identity_record)
    try:
        protocol=json.loads((OUT/'protocol.json').read_text());require(protocol['schema_version']==VERSION,'protocol version differs')
        verify_files(protocol);result=execute(protocol);verify_files(protocol)
        require(sha(OUT/'protocol.json')==identity_record['protocol_sha256'],'protocol changed during run')
        write_new(OUT/'result.json',result)
    except BaseException as error:
        write_new(OUT/'execution.json',dict(identity_record,status='failed',exit_code=1,seconds=time.monotonic()-started,
            completed_utc=now(),error=f'{type(error).__name__}: {error}'));raise
    write_new(OUT/'execution.json',dict(identity_record,status='completed',exit_code=0,seconds=time.monotonic()-started,
        completed_utc=now(),output_sha256=sha(OUT/'result.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('register','run'),required=True)
    args=parser.parse_args();register() if args.mode=='register' else run()
