"""Fixed predictor-only calibration on caller-authenticated native frozen H.

No readers, encoder/teacher forwards, EMA updates, OPF hooks or resumption.
The production loop is CUDA:0 FP32, 500 B4 updates over first400 rows five
times. CPU execution of one_step/smoke is for synthetic checks. Metadata and
hashes here authenticate unchanged supplied tensors, not their data lineage.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict

import torch

from protein_jepa_frame.models.stage_a_relpos_v1 import StageARelposV1
from protein_jepa_frame.models.stage_a_v2.model import CausalTransformerPredictor
from protein_jepa_frame.training.frozen_h_predictor_readout_v1 import SymmetricDistanceHead
from protein_jepa_frame.training.train_utils import _lr_scale
from protein_jepa_frame.utils.contracts import state_sha256

VERSION = 'frozen_h_predictor_calibration_v1'
ARMS = ('latent', 'physical')
STEPS, BATCH_SIZE, FIT_ROWS, SEED = 500, 4, 400, 42
_MARKER = '_frozen_h_predictor_calibration_v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def contract():
    return dict(schema_version=VERSION, arms=list(ARMS), steps=STEPS, batch_size=4,
        fit_rows=400, passes=5, seed=42, lr=3e-4, weight_decay=.02, warmup_steps=100,
        grad_clip=1., physical_weight=dict(latent=0., physical=1.), device='cuda:0',
        dtype='float32', amp=False, tf32=False, dropout=.1, optimizer='AdamW',
        latent_reduction='per-protein valid-residue factor-coordinate MSE, then B mean',
        physical_reduction='per-protein all supplied physical-pair log-MSE, then B mean',
        synthesis='compose(decompose(last native history H)+predicted delta factors)',
        frozen_head_seed=42, audit_head43_consumed=False, feature_mode='pca',
        teacher_executed=False, encoder_executed=False, ema_updates=0,
        opf_after_step=False, latest_only=True, resume=False)


def fixed_batch_indices(step):
    require(type(step) is int and 0 <= step < STEPS, 'step must be integer0..499')
    return [(4*step+j) % FIT_ROWS for j in range(4)]


def _snapshot(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {copy.deepcopy(k): _snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(v) for v in value)
    return copy.deepcopy(value)


def fingerprint(value):
    leaves = {}
    def walk(x, path):
        if isinstance(x, dict):
            require(all(type(k) is str for k in x), 'input metadata keys must be strings')
            leaves[path+'/dict'] = sorted(x)
            for key in sorted(x):
                walk(x[key], path+'/'+repr(key))
        elif isinstance(x, (list, tuple)):
            leaves[path+'/'+type(x).__name__] = len(x)
            for index, item in enumerate(x):
                walk(item, path+'/'+str(index))
        else:
            leaves[path] = x
    walk(value, '')
    return state_sha256(leaves)


def model_identity(model):
    return state_sha256({**{'parameter/'+n: p for n, p in model.named_parameters()},
                         **{'buffer/'+n: b for n, b in model.named_buffers()}})


def frozen_identity(model, *, include_predictor_buffers=True):
    return state_sha256({**{'parameter/'+n: p for n, p in model.named_parameters()
                           if not n.startswith('predictor.')},
                        **{'buffer/'+n: b for n, b in model.named_buffers()
                           if include_predictor_buffers or not n.startswith('predictor.')}})


def rng_snapshot(device):
    states = {'cpu': torch.get_rng_state().clone()}
    if torch.device(device).type == 'cuda':
        states['cuda:0'] = torch.cuda.get_rng_state(0).cpu().clone()
    return dict(states=states, sha256=state_sha256(states))


def restore_rng(snapshot):
    torch.set_rng_state(snapshot['states']['cpu'])
    if 'cuda:0' in snapshot['states']:
        torch.cuda.set_rng_state(snapshot['states']['cuda:0'], 0)


def initialize_predictor(model, seed=42):
    """Replace only the predictor, preserving caller RNG and all other bytes.

    Requires the already-transplanted PCA encoder and its exactly synced
    teacher. This does not silently sync/change a supplied teacher. Requires
    CPU construction; the caller may subsequently move the whole model/head.
    """
    require(type(seed) is int and seed == 42, 'fixed predictor seed42 required')
    require(torch.get_default_device().type == 'cpu' and torch.get_default_dtype() == torch.float32
            and not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('cuda'),
            'initialization requires default CPU FP32 without AMP')
    require(isinstance(model, StageARelposV1) and model.feature_mode == 'pca', 'PCA StageARelposV1 required')
    model._runtime_contract()
    require(not hasattr(model, _MARKER), 'predictor already initialized; no restart')
    require(model.config.latent_dim == 128 and model.config.opf_mode == 'fixed_orthogonal'
            and model.config.opf_learnable is False and model.config.dropout == .1
            and int(model._ema_step) == 0, 'fixed128/OPF/dropout/fresh EMA0 required')
    require(all(p.device.type == 'cpu' and p.dtype == torch.float32 and p.grad is None
                and bool(torch.isfinite(p).all()) for p in model.parameters()), 'finite CPU FP32 model/clear gradients required')
    require(all(b.device.type == 'cpu' and (not b.is_floating_point() or b.dtype == torch.float32)
                and bool(torch.isfinite(b).all()) for b in model.buffers()), 'finite CPU buffers required')
    require(model_identity(model.context_encoder) == model_identity(model.teacher_encoder), 'teacher must already equal frozen online encoder')
    old_full = model_identity(model)
    old_frozen = frozen_identity(model, include_predictor_buffers=False)
    rng = torch.get_rng_state().clone()
    c = model.config
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        predictor = CausalTransformerPredictor(latent_dim=c.latent_dim, num_factors=c.num_factors,
            factor_dim=c.factor_dim, depth=c.predictor_depth, num_heads=c.predictor_heads,
            max_frames=c.max_frames, dropout=c.dropout, spatial_depth=c.predictor_spatial_depth)
        with torch.no_grad():
            predictor.out_head.weight.zero_()
            predictor.out_head.bias.zero_()
    model.eval().requires_grad_(False)
    model.predictor = predictor.train().requires_grad_(True)
    require(torch.equal(rng, torch.get_rng_state()), 'predictor initialization changed CPU RNG')
    require(old_frozen == frozen_identity(model, include_predictor_buffers=False), 'initialization changed nonpredictor state')
    audit = dict(schema_version=VERSION, seed=seed, feature_mode='pca',
        original_full_model_sha256=old_full, initial_full_model_sha256=model_identity(model),
        initial_predictor_sha256=model_identity(model.predictor),
        frozen_state_sha256=frozen_identity(model),
        named_buffers_sha256=state_sha256(dict(model.named_buffers())),
        teacher_online_identical=True, cpu_rng_unchanged=True, final_linear_zero=True,
        config=asdict(c), ema_step=0, completed_updates=0, optimizer_steps_returned=0)
    setattr(model, _MARKER, copy.deepcopy(audit))
    return audit


def check_frozen(model, head, *, require_training=True):
    """Fail closed on mutable frozen weights/buffers, mode or gradient drift."""
    require(isinstance(model, StageARelposV1) and hasattr(model, _MARKER), 'initialized versioned model required')
    model._runtime_contract()
    require(model.feature_mode == 'pca' and int(model._ema_step) == 0, 'PCA/EMA0 contract changed')
    marker = getattr(model, _MARKER)
    require(frozen_identity(model) == marker['frozen_state_sha256'], 'frozen model parameters/buffers changed')
    device = next(model.parameters()).device
    require(device.type == 'cpu' or device == torch.device('cuda:0'), 'only CPU synthetic or CUDA:0 supported')
    require(not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('cuda'), 'AMP forbidden')
    if device.type == 'cuda':
        require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32, 'TF32 forbidden')
    require(all(p.dtype == torch.float32 and p.device == device and bool(torch.isfinite(p).all())
                for p in model.parameters()), 'finite single-device FP32 model required')
    require(all(b.device == device and (not b.is_floating_point() or b.dtype == torch.float32)
                and bool(torch.isfinite(b).all()) for b in model.buffers()), 'finite colocated buffers required')
    for name, module in model.named_modules():
        if name == 'predictor' or name.startswith('predictor.'):
            require(not require_training or module.training, 'predictor must remain train/dropout enabled')
        else:
            require(not module.training, 'nonpredictor module must remain eval')
    for name, parameter in model.named_parameters():
        require(parameter.requires_grad == name.startswith('predictor.'), 'only predictor parameters may be trainable')
        require(name.startswith('predictor.') or parameter.grad is None, 'frozen parameter acquired gradient')
    require(isinstance(head, SymmetricDistanceHead) and getattr(head, 'calibration_seed', None) == 42,
            'only explicitly labelled optimization head42 is allowed')
    require(all(not m.training for m in head.modules())
            and all(not p.requires_grad and p.grad is None and p.device == device and p.dtype == torch.float32
                    and bool(torch.isfinite(p).all()) for p in head.parameters()), 'head42 must be finite frozen/eval/colocated FP32')
    require(all(b.device == device and (not b.is_floating_point() or b.dtype == torch.float32)
                and bool(torch.isfinite(b).all()) for b in head.buffers()), 'head buffers invalid')
    return dict(frozen_state_sha256=marker['frozen_state_sha256'], head_sha256=model_identity(head),
                named_buffers_sha256=state_sha256(dict(model.named_buffers())), ema_step=0)


def fresh_optimizer(model):
    require(hasattr(model, _MARKER) and getattr(model, _MARKER)['completed_updates'] == 0
            and getattr(model, _MARKER)['optimizer_steps_returned'] == 0, 'fresh initialized predictor required')
    optimizer = torch.optim.AdamW(model.predictor.parameters(), lr=3e-4, weight_decay=.02)
    optimizer.param_groups[0]['base_lr'] = 3e-4
    return optimizer


def _optimizer_contract(model, optimizer, step):
    require(type(optimizer) is torch.optim.AdamW and len(optimizer.param_groups) == 1, 'one fresh AdamW group required')
    group = optimizer.param_groups[0]
    require([id(p) for p in group['params']] == [id(p) for p in model.predictor.parameters()], 'optimizer must contain exactly predictor parameters in order')
    require(group['base_lr'] == 3e-4 and group['weight_decay'] == .02 and group['betas'] == (.9, .999)
            and group['eps'] == 1e-8 and not group['amsgrad'] and not group['maximize'], 'fixed AdamW settings differ')
    marker = getattr(model, _MARKER)
    require(marker['completed_updates'] == marker['optimizer_steps_returned'] == step, 'optimizer/update sequence differs')
    require((not optimizer.state if step == 0 else bool(optimizer.state))
            and all(float(s['step']) == step for s in optimizer.state.values()), 'optimizer step state differs')


def _collate(batch, device):
    require(isinstance(batch, list) and len(batch) == 4, 'list of four fixed protein rows required')
    for row in batch:
        require(isinstance(row, dict), 'row dictionary required')
        index = row['cache_row_index']; meta = row['metadata']
        require(type(index) is int and 0 <= index < 400 and isinstance(meta, dict)
                and type(meta.get('protein')) is str and bool(meta['protein'])
                and meta.get('horizon') == (1, 4, 16, 64)[index % 4]
                and row.get('split', 'fit') == 'fit', 'fixed fit400 identity/horizon differs')
        for key in ('lag_ps', 'temp'):
            require(type(row[key]) in (float, int) and math.isfinite(row[key]), 'finite scalar '+key+' required')
        require(row['lag_ps'] == 1000.*meta['horizon'] and row['temp'] > 0, 'physical horizon/temperature differs')
        mask = row['mask']; H = row['history_H']; target = row['target_H']
        require(torch.is_tensor(mask) and mask.dtype == torch.bool and mask.ndim == 1
                and 1 <= len(mask) <= 96 and bool(mask.any()), 'nonempty fixed mask required')
        require(mask.device.type == 'cpu' or mask.device == device, 'batch device differs')
        L = len(mask)
        for value, shape in ((H, (1, 8, L, 128)), (target, (1, L, 128))):
            require(torch.is_tensor(value) and value.shape == shape and value.dtype == torch.float32
                    and value.device == mask.device and not value.requires_grad
                    and bool(torch.isfinite(value).all()) and bool((value[..., ~mask, :] == 0).all()),
                    'detached finite native FP32 H with zero masked rows required')
        pairs, labels = row['pairs'], row['target_log']
        require(torch.is_tensor(pairs) and pairs.dtype == torch.long and pairs.device == mask.device
                and pairs.ndim == 2 and pairs.shape[1] == 2 and len(pairs) > 0
                and bool(((pairs >= 0) & (pairs < L)).all()), 'nonempty physical pair indices required')
        i, j = pairs.T; code = i*L+j
        require(bool((i < j).all()) and bool(mask[pairs].all()) and bool((code[1:] > code[:-1]).all()), 'pairs must be valid unique lexicographic upper triangle')
        require(torch.is_tensor(labels) and labels.shape == (len(pairs),) and labels.dtype == torch.float32
                and labels.device == mask.device and not labels.requires_grad
                and bool(torch.isfinite(labels).all()) and bool((labels >= 0).all()), 'fixed finite FP32 target logs required')
    require(len({row['cache_row_index'] for row in batch}) == 4
            and len({row['metadata']['protein'] for row in batch}) == 4, 'four distinct fit proteins required')
    L = max(len(row['mask']) for row in batch)
    history = torch.zeros((4, 8, L, 128), dtype=torch.float32, device=device)
    target = torch.zeros((4, L, 128), dtype=torch.float32, device=device)
    mask = torch.zeros((4, L), dtype=torch.bool, device=device)
    for index, row in enumerate(batch):
        n = len(row['mask'])
        history[index, :, :n] = row['history_H'][0].to(device)
        target[index, :n] = row['target_H'][0].to(device)
        mask[index, :n] = row['mask'].to(device)
    lag = torch.tensor([r['lag_ps'] for r in batch], dtype=torch.float32, device=device)
    temp = torch.tensor([r['temp'] for r in batch], dtype=torch.float32, device=device)
    return history, target, mask, lag, temp


def _grad_norm(parameters):
    values = [p.grad.detach().norm() for p in parameters if p.grad is not None]
    return float(torch.stack(values).norm()) if values else 0.


def one_step(model, batch, head, arm, optimizer=None, step=0):
    """Compute both objectives, backward/clip, and optionally one predictor update.

    With optimizer=None gradients remain for a zero-update smoke inspection.
    No forward output graphs are retained. Both arms execute the physical head
    and multiply its scalar loss by 0 or 1 only after the same forward work.
    """
    fixed_batch_indices(step)
    require(arm in ARMS, 'unknown arm')
    frozen = check_frozen(model, head)
    require(all(p.grad is None for p in model.parameters()), 'clear model gradients required at entry')
    device = next(model.parameters()).device
    before_input = fingerprint(batch)
    history, target, valid, lag, temp = _collate(batch, device)
    lrs = []
    if optimizer is not None:
        _optimizer_contract(model, optimizer, step)
        optimizer.param_groups[0]['lr'] = 3e-4*_lr_scale(step, 500, 100)
        lrs = [optimizer.param_groups[0]['lr']]
    before_predictor = model_identity(model.predictor)
    rng = {'before': rng_snapshot(device)}
    predicted = model.predictor(history, lag, temp, valid_mask=valid, task_id=1)
    target_factors = model.opf.decompose(target-history[:, -1])
    anchor_factors = model.opf.decompose(history[:, -1])
    endpoint = model.opf.compose(anchor_factors+predicted)
    require(bool(torch.isfinite(predicted).all()) and bool(torch.isfinite(endpoint).all()), 'nonfinite predicted endpoint')
    latent = torch.stack([(predicted[i, valid[i]]-target_factors[i, valid[i]]).square().mean()
                          for i in range(4)])
    physical = torch.stack([(head(endpoint[i:i+1, :len(row['mask'])], row['mask'].to(device),
                                 row['pairs'].to(device))-row['target_log'].to(device)).square().mean()
                            for i, row in enumerate(batch)])
    latent_mean, physical_mean = latent.mean(), physical.mean()
    total = latent_mean + (0. if arm == 'latent' else 1.)*physical_mean
    require(bool(torch.isfinite(latent).all()) and bool(torch.isfinite(physical).all())
            and bool(torch.isfinite(total)), 'nonfinite loss')
    rng['after_forward'] = rng_snapshot(device)
    total.backward()
    rng['after_backward'] = rng_snapshot(device)
    require(rng['after_forward']['sha256'] == rng['after_backward']['sha256'], 'backward consumed RNG')
    active = {n: p for n, p in model.predictor.named_parameters() if p.grad is not None}
    require(bool(active) and all(bool(torch.isfinite(p.grad).all()) for p in active.values()), 'missing/nonfinite predictor gradients')
    out_norm = _grad_norm(model.predictor.out_head.parameters())
    body_norm = _grad_norm(p for n, p in model.predictor.named_parameters() if not n.startswith('out_head.'))
    norm = float(torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 1., error_if_nonfinite=True))
    if optimizer is not None:
        optimizer.step()
        getattr(model, _MARKER)['optimizer_steps_returned'] += 1
        optimizer.zero_grad(set_to_none=True)
    rng['after_update'] = rng_snapshot(device)
    require(rng['after_backward']['sha256'] == rng['after_update']['sha256'], 'optimizer consumed RNG')
    require(check_frozen(model, head) == frozen, 'frozen model/readout changed during step')
    require(fingerprint(batch) == before_input, 'training input mutated')
    if optimizer is not None:
        getattr(model, _MARKER)['completed_updates'] += 1
    after_predictor = model_identity(model.predictor)
    require(optimizer is not None or before_predictor == after_predictor, 'zero-update predictor changed')
    return dict(step=step+1, arm=arm, cache_row_indices=[r['cache_row_index'] for r in batch],
        losses=dict(latent=float(latent_mean.detach()), physical=float(physical_mean.detach()), total=float(total.detach())),
        per_protein_losses=[dict(cache_row_index=r['cache_row_index'], valid_residues=int(r['mask'].sum()),
            n_pairs=len(r['pairs']), latent=float(latent[i].detach()), physical=float(physical[i].detach()))
            for i, r in enumerate(batch)], lr=lrs, global_grad_norm_before_clip=norm,
        out_head_grad_norm_before_clip=out_norm, body_grad_norm_before_clip=body_norm,
        grad_active_names=list(active), grad_none_names=[n for n, p in model.predictor.named_parameters() if n not in active],
        ema_before=0, ema_after=0, optimizer_updated=optimizer is not None,
        input_sha256=before_input, **frozen, predictor_sha256_before=before_predictor,
        predictor_sha256_after=after_predictor, rng_sha256={k: v['sha256'] for k, v in rng.items()},
        tensor_audit=dict(rng=rng, per_protein_latent=latent.detach().cpu().clone(),
                         per_protein_physical=physical.detach().cpu().clone()))


def training_state_payload(model, head, arm, optimizer=None):
    """Detached current state, also valid for a failed/nonfinite partial step.

    This is a predictor-calibration payload, not a resumed Stage A checkpoint.
    A returned Adam step and a verified completed update are separate counts.
    """
    marker = getattr(model, _MARKER)
    return dict(schema_version=VERSION, arm=arm, feature_mode='pca',
        completed_updates=marker['completed_updates'], optimizer_steps_returned=marker['optimizer_steps_returned'],
        ema_step=int(model._ema_step), predictor=_snapshot(model.predictor.state_dict()),
        predictor_named_buffers=_snapshot(dict(model.predictor.named_buffers())),
        optimizer_state_dict=None if optimizer is None else _snapshot(optimizer.state_dict()),
        frozen_state_sha256=frozen_identity(model), head_sha256=model_identity(head),
        initial_audit=copy.deepcopy(marker), selection='latest only; no optimizer resume')


def _fixed_run_contract(model, head):
    device = next(model.parameters()).device
    require(device == torch.device('cuda:0'), 'fixed training requires CUDA:0')
    require(model.config.latent_dim == 128 and model.config.dropout == .1, 'production predictor configuration differs')
    require([tuple(head.net[i].weight.shape) for i in (0, 2, 4)] == [(256, 384), (256, 256), (1, 256)], 'production head width differs')
    check_frozen(model, head)
    marker = getattr(model, _MARKER)
    require(marker['completed_updates'] == marker['optimizer_steps_returned'] == 0
            and model_identity(model.predictor) == marker['initial_predictor_sha256'], 'fresh zero-initialized predictor required')
    require(all(p.grad is None for p in model.parameters()), 'clear gradients required before run')
    return device


def run_fixed_training(model, batch_for_indices, head, arm, emit):
    """Exactly500 CUDA:0 updates; stop on first failure without retry/resume.

    Inputs may remain CPU; collating copies them to the model's device. Returned
    optimizer belongs only to the predictor. Each emitted tensor_audit includes
    full CPU/CUDA RNG states. Failure carries calibration_failure_payload with
    exact current predictor/optimizer, attempted step and emitted-record count.
    """
    require(arm in ARMS and callable(batch_for_indices) and callable(emit), 'arm/callback contract differs')
    model.eval(); model.predictor.train()
    device = _fixed_run_contract(model, head)
    optimizer = fresh_optimizer(model)
    torch.random.default_generator.manual_seed(42)
    torch.cuda.manual_seed(42)
    emitted = 0; attempted = None
    try:
        for step in range(500):
            attempted = step+1
            indices = fixed_batch_indices(step)
            callback_rng = rng_snapshot(device)
            batch = batch_for_indices(indices)
            require(rng_snapshot(device)['sha256'] == callback_rng['sha256'], 'batch callback consumed RNG')
            require([row['cache_row_index'] for row in batch] == indices, 'fixed TRAIN400 batch order differs')
            record = one_step(model, batch, head, arm, optimizer=optimizer, step=step)
            emit(record)
            require(rng_snapshot(device)['sha256'] == record['rng_sha256']['after_update'], 'emit consumed RNG')
            emitted += 1
        require(getattr(model, _MARKER)['completed_updates'] == 500, 'incomplete predictor calibration')
        check_frozen(model, head)
    except BaseException as error:
        error.calibration_failure_payload = training_state_payload(model, head, arm, optimizer)
        error.calibration_failure_payload.update(emitted_records=emitted, attempted_step=attempted,
            error=f'{type(error).__name__}: {error}')
        raise
    return optimizer


def synthetic_smoke_batch(device='cpu'):
    """Synthetic B4K8L96 raw H; local generator, no global RNG or data reads."""
    generator = torch.Generator(device='cpu').manual_seed(90210)
    rows = []
    for index, horizon in enumerate((1, 4, 16, 64)):
        mask = torch.ones(96, dtype=torch.bool)
        history = torch.randn((1, 8, 96, 128), generator=generator)
        target = history[:, -1] + .2*torch.randn((1, 96, 128), generator=generator)
        pairs = torch.triu_indices(96, 96, offset=3).T.contiguous()
        rows.append(dict(cache_row_index=index, metadata=dict(protein=f'synthetic{index}', horizon=horizon),
            split='fit', history_H=history.to(device), target_H=target.to(device), mask=mask.to(device),
            pairs=pairs.to(device), target_log=torch.full((len(pairs),), 2., device=device),
            lag_ps=float(horizon*1000), temp=300.))
    return rows


def smoke(model, head, arm):
    """Real zero-update B4K8L96 backward; restore RNG/modes and clear gradients."""
    require(hasattr(model, _MARKER) and getattr(model, _MARKER)['completed_updates'] == 0, 'fresh calibration model required')
    before = model_identity(model), model_identity(head)
    device = next(model.parameters()).device
    rng = rng_snapshot(device)
    modes = {name: module.training for name, module in model.named_modules()}
    try:
        model.eval(); model.predictor.train()
        record = one_step(model, synthetic_smoke_batch(), head, arm)
        require(record['out_head_grad_norm_before_clip'] > 0.
                and record['body_grad_norm_before_clip'] == 0., 'zero-head first-step gradient contract differs')
        require((model_identity(model), model_identity(head)) == before, 'zero-update smoke changed weights/buffers')
        return record
    finally:
        model.zero_grad(set_to_none=True)
        for name, module in model.named_modules():
            module.training = modes[name]
        restore_rng(rng)
