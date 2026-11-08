"""Independent, fixed-budget ordered CA training; frozen endpoint trainer unchanged.

This entry point has no validation, early stopping, resume, or best checkpoint.
The two new arms differ only in the coordinate objective. A separate h64 smoke
does one forward/backward per arm and never calls optimizer.step().
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import torch

from protein_jepa_frame.data.protocol import validate_manifest_split
from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics, make_loader
from protein_jepa_frame.training.lasp_run_state import rng_state, step_batch
from protein_jepa_frame.training.train_utils import _lr_scale, _move_batch, _set_reproducibility
from protein_jepa_frame.utils.contracts import state_sha256


VERSION = 'lasp_ca_ordered_training_v1'
INITIAL = 'artifacts/hs_joint_20261101/spatial_probe_v1/spatial_exact.pt'
ARMS = ('joint_path', 'marginal_path')
FIXED = dict(component=INITIAL, init_kernel=INITIAL, bath_dim=8,
    manifest='artifacts/mdcath_manifest_full/train.jsonl',
    val_manifest='artifacts/mdcath_manifest_full/val.jsonl',
    steps=500, seed=42, horizons=[1, 4, 16, 64], n_samples=12, grad_steps=64,
    max_residues=96, train_windows=0, stateless_data=True, lr=1e-4, warmup=50,
    cosine=True, train_encoder=False, resume=False, history_conditioning=True,
    collective_mobility=True, s0_mode='exact_spatial', noise_kernel='gaussian',
    noise_mix=.8, noise_length=8., noise_cutoff=10., coupling_scale=.1,
    gamma_scale=2e-5, gamma_floor=2e-6, target='endpoint', prediction_kind='endpoint',
    latent_weight=.1, contact_objective='brier', contact_weight=20., clash_weight=40.,
    distance_weight=0., rmsf_weight=0., mode_weight=0., mode_distribution_weight=0.,
    observed_readout_weight=0., teacher_weight=1., teacher_delta_weight=.1,
    path_weight=.5, checkpoint_decoder=True, validation_during_training=False)


def arguments(output, coord_objective, *, device='cuda:0', torch_threads=4, run_kind='train'):
    if coord_objective not in ARMS or run_kind not in ('train', 'smoke'):
        raise ValueError('unregistered objective or run kind')
    if torch_threads != 4:
        raise ValueError('this fixed pilot uses four CPU threads')
    return SimpleNamespace(**{**FIXED, 'horizons': list(FIXED['horizons']),
        'output': str(output), 'coord_objective': coord_objective, 'device': device,
        'torch_threads': torch_threads, 'run_kind': run_kind, 'training_version': VERSION})


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def training_identity(loader, step, seed):
    """Resolve exactly step_batch's single-item stream without reading coordinates."""
    if loader.batch_size != 1:
        raise ValueError('fixed pilot requires batch size one')
    dataset = loader.dataset
    index = random.Random(seed + 1000003 * step).randrange(len(dataset))
    record, frames = dataset._locate(index)
    return dict(step=step, dataset_index=index, protein=record.protein_id,
        trajectory=record.trajectory_id, temperature_k=float(record.temperature_k),
        record_path=str(record.path), context_frames=frames[:4].tolist(),
        future_frames=frames[4:].tolist(), anchor_frame=int(frames[3]),
        target_frame=int(frames[-1]), horizon=len(frames)-4)


def file_stat(path):
    value = Path(path).stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns)


def load_training_batch(loader, step, seed):
    identity = training_identity(loader, step, seed)
    before = file_stat(identity['record_path'])
    batch = step_batch(loader, step, seed)
    if file_stat(identity['record_path']) != before:
        raise ValueError('training trajectory changed during read')
    if (batch['protein_id'] != [identity['protein']]
            or batch['trajectory_id'] != [identity['trajectory']]
            or batch['coords'].shape[1] != 4 + identity['horizon']):
        raise ValueError('actual batch differs from stateless metadata selection')
    keys = ('coords', 'atom_mask', 'residue_mask', 'aatype', 'residue_index',
            'chain_id', 'time_ps', 'temperature_k')
    identity.update(file_stat=before,
        actual_batch_sha256=state_sha256({key: batch[key] for key in keys}))
    return batch, identity


def build_model(args):
    model = CADynamics(args.component, args.device, args.init_kernel, args.bath_dim,
        args.history_conditioning, args.noise_mix, args.noise_length,
        args.collective_mobility, args.noise_kernel, args.noise_cutoff,
        s0_mode=args.s0_mode).to(args.device)
    for name in ('coupling_scale', 'gamma_scale', 'gamma_floor'):
        value = getattr(args, name)
        setattr(model.kernel.cfg, name, value)
        setattr(model.kernel.bath, name, value)
    if model.train_encoder or any(p.requires_grad for p in model.bundle.parameters()):
        raise ValueError('ordered pilot requires a completely frozen CA component')
    if model.component_source is None:
        raise ValueError('warm start must embed the complete component')
    return model


def loss_backward(model, batch, horizon, args, generator):
    from protein_jepa_frame.training.ordered_path_objective_v1 import ordered_path_losses
    model.train()
    model.bundle.eval()
    model.zero_grad(set_to_none=True)
    terms = ordered_path_losses(model, batch, horizon, args, generator)
    if any(value.numel() != 1 or not bool(torch.isfinite(value)) for value in terms.values()):
        raise FloatingPointError('nonfinite or non-scalar ordered loss term')
    terms['loss'].backward()
    parameters = list(model.kernel.parameters())
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError('missing or nonfinite kernel gradients')
    if any(p.grad is not None for p in model.bundle.parameters()):
        raise RuntimeError('frozen component acquired gradients')
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError('nonfinite total gradient norm')
    return terms, norm


def checkpoint_dict(model, optimizer, args, generator, step, component_before):
    if state_sha256(model.bundle) != component_before:
        raise ValueError('frozen component changed during training')
    return dict(checkpoint_kind='lasp_ca_dynamics', training_version=VERSION,
        kernel=model.kernel.state_dict(), kernel_config=asdict(model.kernel.cfg),
        component_checkpoint=model.component_checkpoint(model.component_source),
        args=vars(args), step=step, validation=None, target_encoder=None,
        optimizer=optimizer.state_dict(), rng_state=rng_state(),
        sampling_rng=generator.get_state(), component_tensor_sha256=component_before,
        selection='latest exactly500; no validation or best selection')


def train(model, args, out):
    loaders = {h: make_loader(args, args.manifest, h, True) for h in args.horizons}
    optimizer = torch.optim.AdamW([dict(params=list(model.kernel.parameters()),
        lr=args.lr, lr_scale=1.)], lr=args.lr, weight_decay=.01)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    frozen = state_sha256(model.bundle)
    start = time.monotonic()
    with (out/'metrics.jsonl').open('x') as log, (out/'training_windows.jsonl').open('x') as windows:
        for step in range(1, args.steps+1):
            horizon = args.horizons[(step-1) % len(args.horizons)]
            batch, identity = load_training_batch(loaders[horizon], step, args.seed)
            windows.write(json.dumps(identity, allow_nan=False)+'\n'); windows.flush()
            scale = _lr_scale(step-1, args.steps, args.warmup)
            for group in optimizer.param_groups:
                group['lr'] = args.lr * scale * group.get('lr_scale', 1.)
            terms, norm = loss_backward(model, _move_batch(batch, args.device),
                                       horizon, args, generator)
            optimizer.step()
            row = dict(step=step, horizon=horizon, seconds=time.monotonic()-start,
                grad_norm=float(norm), lr=args.lr*scale,
                train={key: float(value.detach()) for key, value in terms.items()})
            log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
            if step == 1 or step % 10 == 0:
                print(json.dumps(row, allow_nan=False), flush=True)
    checkpoint = checkpoint_dict(model, optimizer, args, generator, args.steps, frozen)
    with (out/'checkpoint_latest.pt').open('xb') as stream:
        torch.save(checkpoint, stream)
    return dict(status='completed', optimizer_steps=args.steps, seconds=time.monotonic()-start,
        frozen_component_unchanged=True, checkpoint=str(out/'checkpoint_latest.pt'))


def smoke(model, args, out):
    """Fixed train-stream step4/h64, both objectives, zero optimizer updates."""
    if not str(args.device).startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('memory smoke requires the explicitly selected CUDA device')
    loader = make_loader(args, args.manifest, 64, True)
    batch, identity = load_training_batch(loader, 4, args.seed)
    if batch['coords'].shape[2] != 96:
        raise ValueError('registered step4 smoke must exercise L=96; no automatic window replacement')
    before = state_sha256(model)
    batch = _move_batch(batch, args.device)
    rows = {}
    for arm in ARMS:
        current = SimpleNamespace(**{**vars(args), 'coord_objective': arm})
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        started = time.monotonic()
        terms, norm = loss_backward(model, batch, 64, current, generator)
        torch.cuda.synchronize(args.device)
        rows[arm] = dict(seconds=time.monotonic()-started,
            max_memory_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
            max_memory_reserved_bytes=torch.cuda.max_memory_reserved(args.device),
            gradient_norm=float(norm), terms={k: float(v.detach()) for k,v in terms.items()},
            sampling_rng_sha256=state_sha256({'state': generator.get_state()}),
            gradient_parameters=sum(p.grad is not None for p in model.kernel.parameters()),
            component_gradients_present=any(p.grad is not None for p in model.bundle.parameters()))
        if state_sha256(model) != before:
            raise ValueError('zero-update smoke changed model weights')
        del terms
    if rows[ARMS[0]]['sampling_rng_sha256'] != rows[ARMS[1]]['sampling_rng_sha256']:
        raise ValueError('coordinate objective changed sampling RNG')
    report = dict(schema_version='ordered_path_h64_smoke_v1', status='completed',
        optimizer_steps=0, selected_training_step=4, horizon=64, n_samples=12,
        n_residues=96, batch=identity, arms=rows, model_before_sha256=before,
        model_after_sha256=state_sha256(model), purpose='feasibility only; no coefficient or model selection')
    report['environment'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(torch.device(args.device)),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32)
    write_new(out/'smoke.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--coord-objective', choices=ARMS, required=True)
    parser.add_argument('--run-kind', choices=('train', 'smoke'), default='train')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--torch-threads', type=int, default=4)
    parsed = parser.parse_args()
    args = arguments(**vars(parsed))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.torch_threads)
    _set_reproducibility(args.seed, False)
    audit = validate_manifest_split(args.manifest, args.val_manifest)
    write_new(out/'run.json', dict(args=vars(args), split_audit=audit,
        actual_pid=os.getpid(), train_only=True, test_opened=False, no_validation_selection=True))
    model = build_model(args)
    write_new(out/'initial_identity.json', dict(model_sha256=state_sha256(model),
        component_sha256=state_sha256(model.bundle), kernel_sha256=state_sha256(model.kernel),
        kernel_config=asdict(model.kernel.cfg)))
    result = smoke(model, args, out) if args.run_kind == 'smoke' else train(model, args, out)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
