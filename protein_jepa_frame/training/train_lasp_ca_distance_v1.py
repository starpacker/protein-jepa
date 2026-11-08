"""Fixed latest500 marginal-path training with one of two distance additions.

The frozen ordered trainer supplies arguments, model and batch construction,
and checkpoint packaging. Only the selected fixed-weight physical-distance addition differs.
No validation, resume, early stopping, or best-checkpoint selection occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import random
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from protein_jepa_frame.training import train_lasp_ca_ordered_v1 as ordered
from protein_jepa_frame.training.physical_distance_augmented_objective_v1 import physical_distance_augmented_path_losses
from protein_jepa_frame.training.train_utils import _lr_scale, _move_batch, _set_reproducibility
from protein_jepa_frame.utils.contracts import state_sha256

VERSION = 'lasp_ca_distance_training_v1'
ARMS = ('absolute', 'increment')
INITIAL = ordered.INITIAL
make_loader = ordered.make_loader
load_training_batch = ordered.load_training_batch


def arguments(output, arm, *, device='cuda:0', torch_threads=4, run_kind='train'):
    if arm not in ARMS:
        raise ValueError('unregistered distance path objective')
    args = ordered.arguments(output,'marginal_path',device=device,torch_threads=torch_threads,run_kind=run_kind)
    args.contact_estimator = 'plugin'
    args.distance_path_objective = arm
    args.distance_path_weight = 1.
    args.training_version = VERSION
    return args


def write_new(path, value):
    encoded = json.dumps(value,indent=2,allow_nan=False)+'\n'
    with Path(path).open('x') as stream:
        stream.write(encoded)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def build_model(args):
    model = ordered.build_model(args)
    parameters = list(model.kernel.parameters())
    if len(parameters) != 73 or sum(p.numel() for p in parameters) != 669429 or not all(p.requires_grad for p in parameters):
        raise ValueError('expected all 73 trainable kernel tensors and 669429 elements')
    return model


def rng_hashes(generator):
    """Hash global and explicit sampling streams without consuming either."""
    numpy_state = np.random.get_state()
    state = dict(python=random.getstate(),numpy_kind=numpy_state[0],
        numpy_keys=torch.from_numpy(numpy_state[1].copy()),numpy_tail=numpy_state[2:],
        torch=torch.get_rng_state())
    if torch.cuda.is_available():
        state.update({f'cuda_{i}':value for i,value in enumerate(torch.cuda.get_rng_state_all())})
    return dict(global_sha256=state_sha256(state),sampling_sha256=state_sha256({'state':generator.get_state()}))


def loss_backward(model, batch, horizon, args, generator, *, rng_audit=None):
    model.train()
    model.bundle.eval()
    if any(p.requires_grad for p in model.bundle.parameters()):
        raise ValueError('component must remain frozen')
    model.zero_grad(set_to_none=True)
    before_forward = rng_hashes(generator)
    terms = physical_distance_augmented_path_losses(model,batch,horizon,args,generator)
    after_forward = rng_hashes(generator)
    if any(value.ndim != 0 or not bool(torch.isfinite(value)) for value in terms.values()):
        raise FloatingPointError('nonfinite or nonscalar distance-augmented loss term')
    if terms['loss'] is not terms['loss_'+args.distance_path_objective]:
        raise ValueError('selected loss must alias the predeclared distance total')
    for arm in ARMS:
        if not torch.equal(terms['loss_'+arm], terms['loss_base']+terms['distance_'+arm+'_es']):
            raise ValueError('fixed weight-one distance loss identity differs')
    terms['loss'].backward()
    after_backward = rng_hashes(generator)
    if after_backward != after_forward:
        raise ValueError('backward changed global or sampling RNG')
    if rng_audit is not None:
        rng_audit.update(before_forward=before_forward,after_forward=after_forward,after_backward=after_backward)
    parameters = list(model.kernel.parameters())
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError('missing or nonfinite kernel gradients')
    if any(p.grad is not None for p in model.bundle.parameters()):
        raise RuntimeError('frozen component acquired gradients')
    # All 73 tensors are supplied. Unused .grad stays None, so AdamW neither
    # initializes moments nor applies weight decay to those unused tensors.
    norm = torch.nn.utils.clip_grad_norm_(parameters,1.)
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError('nonfinite total gradient norm')
    return terms,norm


def checkpoint_dict(model, optimizer, args, generator, step, component_before):
    checkpoint = ordered.checkpoint_dict(model,optimizer,args,generator,step,component_before)
    checkpoint['training_version'] = VERSION
    return checkpoint


def train(model, args, out):
    if args.steps != 500 or args.horizons != [1,4,16,64] or args.seed != 42:
        raise ValueError('only the fixed500 seed42 training stream is supported')
    loaders = {h:make_loader(args,args.manifest,h,True) for h in args.horizons}
    parameters = list(model.kernel.parameters())
    if len(parameters) != 73 or not all(p.requires_grad for p in parameters):
        raise ValueError('expected all 73 trainable kernel parameter tensors')
    optimizer = torch.optim.AdamW([dict(params=parameters,lr=args.lr,lr_scale=1.)],
                                 lr=args.lr,weight_decay=.01)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    frozen = state_sha256(model.bundle)
    started = time.monotonic()
    with (out/'metrics.jsonl').open('x') as log, (out/'training_windows.jsonl').open('x') as windows:
        for step in range(1,args.steps+1):
            horizon = args.horizons[(step-1)%len(args.horizons)]
            batch,identity = load_training_batch(loaders[horizon],step,args.seed)
            windows.write(json.dumps(identity,allow_nan=False)+'\n'); windows.flush()
            scale = _lr_scale(step-1,args.steps,args.warmup)
            for group in optimizer.param_groups:
                group['lr'] = args.lr*scale*group.get('lr_scale',1.)
            terms,norm = loss_backward(model,_move_batch(batch,args.device),horizon,args,generator)
            optimizer.step()
            scalars = {key:float(value.detach()) for key,value in terms.items()}
            gradient_norm = float(norm.detach())
            # The unselected auxiliary branch may still own saved tensors.
            # Release every graph-bearing return before the next forward.
            del terms,norm
            row = dict(step=step,horizon=horizon,seconds=time.monotonic()-started,grad_norm=gradient_norm,
                lr=args.lr*scale,kernel_parameters=len(parameters),
                gradient_parameters=sum(p.grad is not None for p in parameters),
                train=scalars)
            log.write(json.dumps(row,allow_nan=False)+'\n'); log.flush()
            if step == 1 or step%10 == 0:
                print(json.dumps(row,allow_nan=False),flush=True)
    checkpoint = checkpoint_dict(model,optimizer,args,generator,args.steps,frozen)
    with (out/'checkpoint_latest.pt').open('xb') as stream:
        torch.save(checkpoint,stream)
    return dict(status='completed',optimizer_steps=500,seconds=time.monotonic()-started,
        frozen_component_unchanged=True,kernel_parameter_tensors=len(parameters),
        active_optimizer_states=len(optimizer.state),checkpoint=str(out/'checkpoint_latest.pt'))


def smoke(model, args, out):
    """Fixed train-step4/h64/L96, both distance arms, one selected backward each, zero updates."""
    if not str(args.device).startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('memory smoke requires the selected CUDA device')
    loader = make_loader(args,args.manifest,64,True)
    batch,identity = load_training_batch(loader,4,42)
    if batch['coords'].shape[2] != 96:
        raise ValueError('fixed smoke requires L96; no replacement window')
    before = state_sha256(model)
    batch = _move_batch(batch,args.device)
    rows = {}
    for arm in ARMS:
        current = SimpleNamespace(**{**vars(args),'distance_path_objective':arm})
        generator = torch.Generator(device=args.device).manual_seed(42)
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        started = time.monotonic()
        rng_audit = {}
        terms,norm = loss_backward(model,batch,64,current,generator,rng_audit=rng_audit)
        torch.cuda.synchronize(args.device)
        rows[arm] = dict(seconds=time.monotonic()-started,rng_audit=rng_audit,
            max_memory_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
            max_memory_reserved_bytes=torch.cuda.max_memory_reserved(args.device),
            gradient_norm=float(norm),terms={k:float(v.detach()) for k,v in terms.items()},
            sampling_rng_sha256=state_sha256({'state':generator.get_state()}),
            kernel_parameter_tensors=sum(1 for _ in model.kernel.parameters()),
            gradient_parameters=sum(p.grad is not None for p in model.kernel.parameters()),
            component_gradients_present=any(p.grad is not None for p in model.bundle.parameters()))
        if state_sha256(model) != before:
            raise ValueError('zero-update smoke changed model weights')
        del terms,norm
    if rows['absolute']['sampling_rng_sha256'] != rows['increment']['sampling_rng_sha256']:
        raise ValueError('distance objective changed sampling RNG')
    if rows['absolute']['rng_audit'] != rows['increment']['rng_audit']:
        raise ValueError('matched smoke global or sampling RNG evidence differs')
    shared = [key for key in rows['absolute']['terms'] if key != 'loss']
    if any(rows['absolute']['terms'][key] != rows['increment']['terms'][key] for key in shared):
        raise ValueError('matched smoke shared terms differ')
    report = dict(schema_version='distance_training_h64_smoke_v1',status='completed',optimizer_steps=0,
        selected_training_step=4,horizon=64,n_samples=12,n_residues=96,batch=identity,arms=rows,
        model_before_sha256=before,model_after_sha256=state_sha256(model),
        purpose='fixed feasibility only; no performance or coefficient selection',
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(torch.device(args.device)),
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32))
    write_new(out/'smoke.json',report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--distance-path-objective',choices=ARMS,required=True)
    parser.add_argument('--run-kind',choices=('train','smoke'),default='train')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--torch-threads',type=int,default=4)
    parsed = vars(parser.parse_args(argv))
    arm = parsed.pop('distance_path_objective')
    return arguments(arm=arm,**parsed)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    identity = dict(pid=os.getpid(),command=sys.argv,started_utc=now(),source_sha256=sha(__file__),
                    run_kind=args.run_kind,distance_path_objective=args.distance_path_objective)
    write_new(out/'started.json',identity)
    try:
        torch.set_num_threads(args.torch_threads)
        _set_reproducibility(args.seed,False)
        write_new(out/'run.json',dict(args=vars(args),train_only=True,test_opened=False,
            validation_loaded=False,no_validation_selection=True))
        model = build_model(args)
        write_new(out/'initial_identity.json',dict(model_sha256=state_sha256(model),
            component_sha256=state_sha256(model.bundle),kernel_sha256=state_sha256(model.kernel),
            kernel_parameter_names=[n for n,_ in model.kernel.named_parameters()],
            kernel_parameter_elements=sum(p.numel() for p in model.kernel.parameters())))
        result = smoke(model,args,out) if args.run_kind == 'smoke' else train(model,args,out)
        output = out/('smoke.json' if args.run_kind == 'smoke' else 'checkpoint_latest.pt')
    except BaseException as error:
        write_new(out/'execution.json',dict(identity,status='failed',exit_code=1,seconds=time.monotonic()-started,
            completed_utc=now(),error=f'{type(error).__name__}: {error}'))
        raise
    write_new(out/'execution.json',dict(identity,status='completed',exit_code=0,seconds=time.monotonic()-started,
        completed_utc=now(),result=result,output=str(output),output_sha256=sha(output)))
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
