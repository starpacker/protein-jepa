"""Registered CPU-only common-Brownian forward integration diagnostic.

No training, posterior refitting, projection of H, or sampler source changes.
Use --mode register first; --mode run consumes that immutable registration.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from evaluation.geometry import ca_geometry_summary, kabsch_align
from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.dataset import TrajectoryClipDataset
from protein_jepa_frame.data.manifest import load_manifest
from protein_jepa_frame.training.train_lasp_ca_dynamics import energy_score
from protein_jepa_frame.training.train_stage_b1a import _extract_backbone_frames
from protein_jepa_frame.utils.contracts import state_sha256
from scripts.eval_ca_endpoint_scores import load_model

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'results/hs_joint_20261101/microstep_train32_v1'
CHECKPOINT = 'artifacts/hs_joint_20261101/spatial_probe_v1/spatial_exact.pt'
CHECKPOINT_SHA = '396e6e4df35019140324484bf8d560a292cf2688a11946f2e0ccde5cd6ea8dda'
SELECTION_PROTOCOL = 'results/hs_joint_20261101/latent_manifold_v1/train32_manifold_v2.protocol.json'
MANIFEST = 'artifacts/mdcath_manifest_full/train.jsonl'
DOCUMENT = 'docs/HS_MICROSTEP_DIAGNOSTIC_V1_20261101.md'
TEST_SOURCE = 'tests/test_hs_microsteps_v1.py'
VERSION = 'hs_fixed_initial_posterior_microsteps_v1'
HORIZONS = (1, 4, 16, 64)
MICROSTEPS = (1, 4, 8)
MEMBERS = 12
BUDGET_KEYS = ('force', 'bath', 'drift_squared', 'noise_trace', 'expected',
               'actual', 'stochastic_residual', 'noise_cross', 'noise_squared_minus_trace',
               'rounding_closure')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def plain(value):
    if isinstance(value, torch.Tensor):
        return plain(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(plain(value), stream, indent=2, allow_nan=False)
        stream.write('\n')


def save_new(path, value):
    with Path(path).open('xb') as stream:
        torch.save(value, stream)


def file_stat(path):
    s = Path(path).stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns,
                inode=s.st_ino, device=s.st_dev)


def coarsen(fine, count):
    require(fine.ndim == 5 and count in MICROSTEPS and fine.shape[2] == 8,
            'expected [members,macro,8,residue,channel] fine increments')
    group = 8 // count
    return fine.reshape(*fine.shape[:2], count, group, *fine.shape[3:]).sum(3) / math.sqrt(group)


def mobility(value, diffusion, factor):
    root = diffusion.sqrt()
    return root * (factor @ (factor.transpose(-1, -2) @ (root * value)))


def norm_budget(before, after, force, bath_state, diffusion, coupling, factor, dt,
                noise_standard, mask):
    """Float64 norm accounting, conditional on current H AND S.

    Returned [member] terms use total squared norm, not a per-dimension mean.
    Noise is independently reconstructed from the supplied Brownian increment.
    The centered projection is per residue, across latent channels; its trace is
    (1-1/d) times the raw trace because M has no cross-channel covariance.
    """
    require(before.shape == after.shape == force.shape == diffusion.shape,
            'budget state/force/diffusion shapes differ')
    require(mask.shape == before.shape[:2] and mask.dtype == torch.bool, 'budget mask differs')
    h, a, f, s, d, c, fac, z = [x.double() for x in
        (before, after, force, bath_state, diffusion, coupling, factor, noise_standard)]
    valid = mask[..., None]
    h, a = h * valid, a * valid
    mf = mobility(f, d, fac) * valid
    mcs = mobility(torch.einsum('blij,blj->bli', c, s), d, fac) * valid
    b = mf - mcs
    eta = torch.sqrt(2 * dt * d) * (fac @ z) * valid
    kii = fac.square().sum(-1)
    trace = 2 * dt * (d * kii[..., None] * valid).sum((-1, -2))
    result = {}
    for name, centered in (('raw', False), ('centered', True)):
        project = (lambda x: x - x.mean(-1, keepdim=True)) if centered else (lambda x: x)
        hp, ap, fp, cp, bp, ep = [project(x) for x in (h, a, mf, mcs, b, eta)]
        sumprod = lambda x, y: (x * y).sum((-1, -2))
        ft = 2 * dt * sumprod(hp, fp)
        bt = -2 * dt * sumprod(hp, cp)
        ds = dt * dt * sumprod(bp, bp)
        nt = trace * (1 - 1 / h.shape[-1]) if centered else trace
        expected = ft + bt + ds + nt
        actual = sumprod(ap, ap) - sumprod(hp, hp)
        cross = 2 * sumprod(hp + dt * bp, ep)
        sq = sumprod(ep, ep) - nt
        result[name] = dict(force=ft, bath=bt, drift_squared=ds, noise_trace=nt,
            expected=expected, actual=actual, stochastic_residual=actual-expected,
            noise_cross=cross, noise_squared_minus_trace=sq,
            rounding_closure=actual-expected-cross-sq)
    return result


class NumericalBranchFailure(RuntimeError):
    """A numerical failure retained in the branch; interface errors propagate."""


class BudgetSink:
    """Streaming callback: preserve budgets and macro states, not all microstates."""
    def __init__(self, kernel, h0, s0, mask, temp, diffusion, coupling, gamma,
                 factor, brownian_h, count):
        self.kernel, self.mask, self.temp = kernel, mask, temp
        self.diffusion, self.coupling, self.gamma = diffusion, coupling, gamma
        self.factor, self.brownian_h, self.count = factor, brownian_h, count
        self.dt = float(kernel.cfg.step_ps) / count
        self.budgets = []
        self.macro_h, self.macro_s = [h0.detach().clone()], [s0.detach().clone()]
        self.failure = None
        self.failure_tensors = None
        self.next_position = dict(macro_step=0, micro_step=0)

    def append(self, event):
        i, j = int(event['macro_step']), int(event['micro_step'])
        require(self.next_position == dict(macro_step=i, micro_step=j), 'trace order differs')
        fields = ('H_before', 'S_before', 'H_after', 'S_after')
        bad = {key: (~torch.isfinite(event[key])).flatten(1).sum(1).tolist() for key in fields}
        if any(any(counts) for counts in bad.values()):
            self.failure = dict(macro_step=i, micro_step=j, stage='transition', nonfinite_per_member=bad)
            self.failure_tensors = {key: event[key].clone() for key in fields}
            raise NumericalBranchFailure('nonfinite transition')
        # Deliberate recomputation, with no RNG or parameter update; avoids
        # monkeypatching a frozen sampler. It approximately doubles force cost.
        force = self.kernel.compute_forces(event['H_before'], self.mask, self.temp, create_graph=False).detach()
        if not bool(torch.isfinite(force).all()):
            self.failure = dict(macro_step=i, micro_step=j, stage='budget_force',
                                nonfinite_per_member=(~torch.isfinite(force)).flatten(1).sum(1).tolist())
            self.failure_tensors = {key: event[key].clone() for key in fields} | {'force': force}
            raise NumericalBranchFailure('nonfinite force')
        budget = norm_budget(event['H_before'], event['H_after'], force, event['S_before'],
            self.diffusion, self.coupling, self.factor, self.dt, self.brownian_h[:, i, j], self.mask)
        if not all(bool(torch.isfinite(value).all()) for terms in budget.values() for value in terms.values()):
            self.failure = dict(macro_step=i, micro_step=j, stage='float64_budget')
            self.failure_tensors = {key: event[key].clone() for key in fields} | {'force': force}
            raise NumericalBranchFailure('nonfinite norm budget')
        self.budgets.append(budget)
        if j + 1 == self.count:
            self.macro_h.append(event['H_after'].clone())
            self.macro_s.append(event['S_after'].clone())
        self.next_position = dict(macro_step=i+1, micro_step=0) if j+1 == self.count else dict(macro_step=i, micro_step=j+1)

    def raw(self):
        budget = {space: {key: torch.stack([row[space][key] for row in self.budgets])
                          if self.budgets else torch.empty(0, len(self.mask), dtype=torch.float64)
                          for key in BUDGET_KEYS} for space in ('raw', 'centered')}
        return dict(H=torch.stack(self.macro_h, 1), S=torch.stack(self.macro_s, 1),
                    budgets=budget, failure_tensors=self.failure_tensors)


@torch.no_grad()
def initial_state(model, context, horizon, seed, members=MEMBERS):
    """Only the four observed frames enter this function. No endpoint argument."""
    require(context['coords'].shape[0] == 4, 'exactly four observed frames required')
    mask = context['residue_mask'].all(0)[None].bool()
    require(int(mask.sum()) >= 4, 'too few observed residues')
    require(bool(torch.isfinite(context['coords'][:, mask[0]]).all()), 'nonfinite observed coordinates')
    meta = [context[k][None] for k in ('aatype', 'residue_index', 'chain_id')]
    temp = torch.tensor([float(context['temperature_k'])], dtype=context['coords'].dtype)
    h0 = model.bundle.encode(context['coords'][-1:], mask, *meta)
    expand4 = lambda x: x.expand(4, *x.shape[1:])
    history = model.bundle.encode(context['coords'], expand4(mask), *[expand4(x) for x in meta])[None]
    dt = torch.full((1, 4), float(model.kernel.cfg.step_ps), dtype=h0.dtype)
    factor = model.anchor_kernel(context['coords'][-1:, :, 1], mask)
    with sdpa_kernel(SDPBackend.MATH):
        mean, chol = model.kernel.initial_bath_state(history, dt, temp, mask, seq_feature=h0,
                                                    noise_factor=factor, create_graph=False)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    rng_initial = generator.get_state().clone()
    eps = torch.randn((1, members, mean.shape[1]*mean.shape[2]), generator=generator, dtype=h0.dtype)
    s0 = (mean[:, None] + torch.einsum('bij,bmj->bmi', chol, eps).reshape(1, members, *mean.shape[1:]))[0]
    repeat = lambda x: x.expand(members, *x.shape[1:]).contiguous()
    h, rm, t, fac = repeat(h0), repeat(mask), temp.expand(members).contiguous(), repeat(factor)
    d, c, gamma = model.kernel.bath(h, t)
    fine_h = torch.randn((members, horizon, 8, h.shape[1], h.shape[2]), generator=generator, dtype=h.dtype)
    fine_s = torch.randn((members, horizon, 8, h.shape[1], s0.shape[2]), generator=generator, dtype=h.dtype)
    rotation, anchor = _extract_backbone_frames(context['coords'][-1:])
    return dict(H0=h, S0=s0, mask=rm, temp=t, factor=fac, D=d, C=c, Gamma=gamma,
        history=history, dt_history=dt, posterior_mean=mean, posterior_cholesky=chol,
        posterior_standard_normal=eps, fine_H=fine_h, fine_S=fine_s,
        rng_initial=rng_initial, rng_final=generator.get_state().clone(),
        metadata=meta, rotation=rotation, anchor=anchor, anchor_H=h0)


@torch.no_grad()
def run_branch(model, state, horizon, count):
    h, s, mask, temp, fac = [state[k] for k in ('H0', 'S0', 'mask', 'temp', 'factor')]
    d, c, gamma = model.kernel.bath(h, temp)
    require(all(torch.equal(x, state[k]) for x,k in ((d,'D'),(c,'C'),(gamma,'Gamma'))),
            'anchor-fixed coefficients differ across branches')
    zh, zs = coarsen(state['fine_H'], count), coarsen(state['fine_S'], count)
    sink = BudgetSink(model.kernel, h, s, mask, temp, d, c, gamma, fac, zh, count)
    started = time.monotonic()
    try:
        with sdpa_kernel(SDPBackend.MATH):
            trajectory, bath = model.kernel.rollout_microstep_v1(h, mask, temp, horizon,
                S0=s, seq_feature=h, detach=True, return_bath=True, noise_factor=fac,
                collective_mobility=True, microsteps=count, brownian_H=zh, brownian_S=zs,
                integrator='explicit', microstate_trace=sink)
        raw = sink.raw()
        require(torch.equal(raw['H'], trajectory) and torch.equal(raw['S'], bath), 'macro trace differs from primitive')
        for space in ('raw', 'centered'):
            project = (lambda x: x-x.mean(-1,keepdim=True)) if space=='centered' else (lambda x:x)
            first=project(h.double())*mask[...,None]
            last=project(trajectory[:,-1].double())*mask[...,None]
            endpoint_change=(last.square()-first.square()).sum((-1,-2))
            torch.testing.assert_close(raw['budgets'][space]['actual'].sum(0),endpoint_change,
                                       rtol=5e-11,atol=1e-8)
            trace=2*float(model.kernel.cfg.step_ps)*horizon*(d.double()*fac.double().square().sum(-1)[...,None]*mask[...,None]).sum((-1,-2))
            if space=='centered':trace=trace*(1-1/h.shape[-1])
            torch.testing.assert_close(raw['budgets'][space]['noise_trace'].sum(0),trace,
                                       rtol=5e-12,atol=1e-8)
        status = 'completed'
    except NumericalBranchFailure:
        raw, status = sink.raw(), 'numerical_failure'
    return dict(status=status, failure=sink.failure, seconds=time.monotonic()-started,
                completed_macro_steps=raw['H'].shape[1]-1, completed_micro_steps=len(sink.budgets)), raw


def macro_statistics(h, s, mask):
    """Input [member,time,residue,channel]; retain member and macro axes."""
    h, s = h.double()[:, :, mask], s.double()[:, :, mask]
    mean, variance = h.mean(-1), h.var(-1, unbiased=False)
    return dict(channel_mean_abs=mean.abs().mean(-1), channel_mean_squared=mean.square().mean(-1),
        channel_variance_mean=variance.mean(-1), centered_radius_rms=variance.mean(-1).sqrt(),
        norm_squared_per_dim=h.square().mean((-1,-2)), bath_norm_squared_per_dim=s.square().mean((-1,-2)),
        bath_l2_norm=s.square().sum((-1,-2)).sqrt(), n_valid_residues=int(mask.sum()))


@torch.no_grad()
def decode_macro(model, state, h):
    members, times, length, dim = h.shape
    parts = []
    for k in range(times):
        repeat = lambda x: x.expand(members, *x.shape[1:])
        parts.append(model.bundle.decode_ca(h[:,k], repeat(state['anchor_H']),
            *[repeat(x) for x in state['metadata']], repeat(state['rotation']),
            repeat(state['anchor']), state['mask']))
    return torch.stack(parts, 1)


def coordinate_scores(samples, target, anchor, mask, ri, ci):
    y = kabsch_align(target.double(), anchor.double(), mask)
    x = samples.double()
    mse = (x[:,mask]-y[mask]).square().sum(-1).mean(-1)
    geometry = [ca_geometry_summary(frame, mask, residue_index=ri, chain_id=ci) for frame in x]
    result = dict(coordinate_energy_score=float(energy_score(x[:,mask].flatten(1),y[None,mask].flatten(1))),
        member_rmsd=float(mse.sqrt().mean()), per_sample_rmsd=mse.sqrt(),
        geometry_per_sample=geometry, geometry_mean={key: float(np.mean([g[key] for g in geometry]))
            if all(g[key] is not None for g in geometry) else None for key in geometry[0]})
    if (not math.isfinite(result['coordinate_energy_score']) or not bool(torch.isfinite(mse).all())
            or any(v is not None and not math.isfinite(v) for g in geometry for v in g.values())):
        raise NumericalBranchFailure('nonfinite endpoint metric arithmetic')
    return result


def coupled_difference(a, b, mask, *, coordinates):
    require(a.shape == b.shape, 'coupled paths have different shapes')
    squared = (a.double()[:,:,mask] - b.double()[:,:,mask]).square()
    # CA is vector RMSD per residue; H is RMS per latent component.
    per_member_time = squared.sum(-1).mean(-1).sqrt() if coordinates else squared.mean((-1,-2)).sqrt()
    return dict(per_member_per_macro=per_member_time, mean_per_macro=per_member_time.mean(0),
                endpoint_mean=float(per_member_time[:,-1].mean()),
                units='angstrom RMSD per residue' if coordinates else 'raw H RMS per component')


def registered_sources(previous):
    paths = set(previous['sources']) | {str(Path(__file__).resolve().relative_to(REPO)), DOCUMENT, TEST_SOURCE}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.is_relative_to(REPO) and path.suffix == '.py' and path.is_file():
                paths.add(str(path.relative_to(REPO)))
    return {p: sha(REPO/p) for p in sorted(paths)}


def validate_selection(selection, records):
    require(len(selection)==32 and len({s['protein'] for s in selection})==32, 'fixed32 protein budget differs')
    require(Counter(s['horizon'] for s in selection)==Counter({h:8 for h in HORIZONS}), 'horizon allocation differs')
    by_id = {r.trajectory_id:r for r in records}
    require(len(by_id)==len(records),'ambiguous trajectory identity in training manifest')
    datasets = {h:TrajectoryClipDataset(records, DataConfig(num_frames=4+h,frame_stride=1,max_residues=96)) for h in HORIZONS}
    for index, row in enumerate(selection, 1):
        require(row['step']==index and row['horizon']==HORIZONS[(index-1)%4], 'step/horizon schedule changed')
        ds = datasets[row['horizon']]
        idx = random.Random(42+1000003*index).randrange(len(ds))
        record, frames = ds._locate(idx)
        actual = dict(step=index,horizon=row['horizon'],dataset_index=idx,protein=record.protein_id,
            trajectory=record.trajectory_id,temperature_k=record.temperature_k,context_frames=frames[:4].tolist(),
            anchor_frame=int(frames[3]),target_frame=int(frames[-1]),record_path=str(record.path),
            sampling_seed=42+1000003*index)
        require(actual==row, 'original stateless train selection changed')
        require(record.format=='mdcath-hdf5' and record.frame_interval_ps==1000., 'unexpected dataset/time grid')
    return by_id


def cpu_setup():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='', "require CUDA_VISIBLE_DEVICES='' explicitly")
    require(Path.cwd().resolve()==REPO, 'run from repository root')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    require(not torch.cuda.is_initialized(), 'CUDA initialized in CPU-only diagnostic')


def register():
    cpu_setup()
    require(not OUT.exists(), 'refuse existing diagnostic directory')
    previous = json.loads((REPO/SELECTION_PROTOCOL).read_text())
    require(previous['checkpoint']==CHECKPOINT and previous['inputs'][CHECKPOINT]==CHECKPOINT_SHA,
            'original selection checkpoint differs')
    for path,digest in previous['inputs'].items():
        require(sha(REPO/path)==digest, 'original frozen input changed: '+path)
    for path,digest in previous['sources'].items():
        require(sha(REPO/path)==digest, 'original frozen source changed: '+path)
    records = load_manifest(REPO/MANIFEST)
    validate_selection(previous['selection'], records)
    stats = {s['record_path']:file_stat(s['record_path']) for s in previous['selection']}
    for path, stat in stats.items():
        old = previous['hdf5_file_stats'][path]
        require(stat['size']==old['size'] and stat['mtime_ns']==old['mtime_ns'], 'HDF5 changed since selection')
    checkpoint = torch.load(REPO/CHECKPOINT,map_location='cpu',weights_only=False)
    model = load_model(str(REPO/CHECKPOINT), checkpoint, 'cpu').eval().requires_grad_(False)
    require(model.kernel.s0_mode=='exact_spatial' and model.collective_mobility and model.history_conditioning,
            'required conditional collective sampler differs')
    require(model.kernel.cfg.step_ps==1000. and model.kernel.cfg.force_clip is None, 'kernel semantics differ')
    inputs = dict(previous['inputs'], **{SELECTION_PROTOCOL:sha(REPO/SELECTION_PROTOCOL)})
    sources = registered_sources(previous)
    protocol = dict(schema_version=VERSION,registered_utc=now(),device='cpu',torch_threads=2,
        checkpoint=CHECKPOINT,checkpoint_sha256=CHECKPOINT_SHA,selection_protocol=SELECTION_PROTOCOL,
        selection=previous['selection'],source_hashes=sources,input_hashes=inputs,hdf5_file_stats=stats,
        n_windows=32,n_proteins=32,n_samples=12,horizons=list(HORIZONS),microsteps=list(MICROSTEPS),
        model_tensor_sha256=state_sha256(model),kernel_config=checkpoint['kernel_config'],
        posterior='original 1000ps exact_spatial posterior from four observations, computed once per window; same sampled S0 for all branches',
        rng='CPU generator seed=42+1000003*step; draw full-posterior eps[1,12,L*dS], then fineH[12,h,8,L,128], then fineS[12,h,8,L,8]; coarsen by sum/sqrt(group)',
        rng_caveat='same seed does not recreate prior production sample paths: fine-array draw order differs; branches share one explicitly stored Brownian path',
        initial_and_coefficients='same H0,S0,T,anchor factor and repeated-anchor D/C/Gamma; force reevaluated at each state',
        arithmetic='frozen sampler/encoder/decoder float32; budgets and summary arithmetic float64; extra force recomputation for accounting',
        norm_budget='conditional on current H,S: force + bath + drift_squared + noise_trace; stochastic residual is actual minus conditional expectation, not required zero',
        posterior_marginalization='If S0 is integrated out, first-step covariance additionally has dt^2 M C P0 C^T M. This conditional-on-sampled-S diagnostic does not add that term again at every step.',
        centered_trace='Q per-residue channel centering; tr(Q M Q)=(1-1/dH)*sum_valid_i,c K_ii D_ic; normalization remains Nvalid*dH',
        label_use='read four observed frames first; endpoint is read only after all three branches and their CA paths finish',
        temporal_scope='same total physical horizons; 8 microsteps is a comparison reference, not exact truth or a matched high-frequency posterior',
        failure_policy='retain every selected row and every branch; any member nonfinite fails full branch, no 12-member score from subset; programming/interface failures abort job',
        aggregation='within-window member means; within-protein means then equal-protein, with explicit complete/support counts; by horizon uses different eight proteins',
        adaptive_status='new diagnostic registered before its sampling; these train32 windows and earlier readout outcomes were already examined, not unseen data',
        no_projection=True,optimizer_steps=0,test_opened=False,validation_opened=False,bioemu2_rerun=False,
        no_selection='no best grid, checkpoint, score, or tolerance selection; no stability/forecast-improvement claim',
        versions=dict(torch=torch.__version__,numpy=np.__version__,python=sys.version))
    OUT.mkdir(parents=True,exist_ok=False)
    snapshot=OUT/'source_snapshot';snapshot.mkdir()
    for path,digest in sources.items():
        data=(REPO/path).read_bytes();require(hashlib.sha256(data).hexdigest()==digest,'source changed during registration')
        dest=snapshot/path;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as stream:stream.write(data)
    write_new(OUT/'protocol.json',protocol)
    print(json.dumps(dict(registered=str(OUT/'protocol.json'),sha256=sha(OUT/'protocol.json'))),flush=True)


def verify_registered(protocol, expected_protocol_sha=None):
    if expected_protocol_sha is not None:
        require(sha(OUT/'protocol.json')==expected_protocol_sha,'registration protocol changed')
    for field in ('input_hashes','source_hashes'):
        for path,digest in protocol[field].items():
            require(sha(REPO/path)==digest, 'registered file changed: '+path)
            if field=='source_hashes':require(sha(OUT/'source_snapshot'/path)==digest,'snapshot changed: '+path)
    for path,expected in protocol['hdf5_file_stats'].items():
        require(file_stat(path)==expected,'registered HDF5 stat changed: '+path)


def validate_clip(clip, frames, record):
    require(clip['coords'].dtype==torch.float32 and clip['coords'].shape[0]==len(frames), 'clip shape/dtype differs')
    expected=torch.tensor(record.first_frame_time_ps+np.asarray(frames)*record.frame_interval_ps,dtype=torch.float32)
    require(torch.equal(clip['time_ps'],expected),'actual physical timestamps differ')


def summarize(rows):
    """Null overall estimate if any registered window lacks the metric."""
    names=sorted(set().union(*(r['scalars'] for r in rows)))
    result={}
    for name in names:
        proteins=defaultdict(list)
        for row in rows:
            value=row['scalars'].get(name)
            if value is not None:proteins[row['metadata']['protein']].append(value)
        per_protein={p:float(np.mean(v)) for p,v in proteins.items()}
        complete=sum(map(len,proteins.values()))==len(rows)
        result[name]=dict(equal_protein_mean=float(np.mean(list(per_protein.values()))) if complete else None,
            available_subset_mean=float(np.mean(list(per_protein.values()))) if per_protein else None,
            complete=complete,n_windows=sum(map(len,proteins.values())),expected_windows=len(rows),
            n_proteins=len(per_protein),per_protein=per_protein)
    return result


def run():
    cpu_setup()
    protocol_bytes=(OUT/'protocol.json').read_bytes()
    protocol_sha=hashlib.sha256(protocol_bytes).hexdigest()
    protocol=json.loads(protocol_bytes)
    require(protocol['schema_version']==VERSION,'wrong registration version')
    require(not (OUT/'execution.started.json').exists(),'no automatic restart')
    verify_registered(protocol,protocol_sha)
    write_new(OUT/'execution.started.json',dict(pid=os.getpid(),started_utc=now(),argv=sys.argv,
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],protocol_sha256=protocol_sha))
    started=time.monotonic()
    try:
        result=execute(protocol,started,protocol_sha)
    except BaseException as error:
        exit_code=130 if isinstance(error,KeyboardInterrupt) else (error.code if isinstance(error,SystemExit) and isinstance(error.code,int) else 1)
        write_new(OUT/'execution.json',dict(pid=os.getpid(),returncode=exit_code,completed_utc=now(),
            seconds=time.monotonic()-started,error=f'{type(error).__name__}: {error}'))
        raise
    write_new(OUT/'execution.json',dict(pid=os.getpid(),returncode=0,completed_utc=now(),
        seconds=time.monotonic()-started,result_sha256=sha(OUT/'diagnostic.json'),
        numerical_failure_branches=result['numerical_failure_branches']))


def execute(protocol,started,protocol_sha):
    checkpoint=torch.load(REPO/CHECKPOINT,map_location='cpu',weights_only=False)
    model=load_model(str(REPO/CHECKPOINT),checkpoint,'cpu').eval().requires_grad_(False)
    require(state_sha256(model)==protocol['model_tensor_sha256'],'loaded model identity differs')
    records=load_manifest(REPO/MANIFEST)
    by_id=validate_selection(protocol['selection'],records)
    rows=[]
    for selection in protocol['selection']:
        record=by_id[selection['trajectory']]
        reader=TrajectoryClipDataset([record],DataConfig(num_frames=4,max_residues=96))
        context=reader._read_clip(record,np.array(selection['context_frames'],dtype=np.int64))
        validate_clip(context,selection['context_frames'],record)
        state=initial_state(model,context,selection['horizon'],selection['sampling_seed'])
        m=state['mask'][0]; count_dim=int(m.sum())*state['H0'].shape[-1]
        raw=dict(metadata=selection,observed=context,state=state,branches={})
        row=dict(metadata=selection,branches={},differences={},scalars={},
                 observed_sha256=state_sha256({k:context[k] for k in ('coords','atom_mask','residue_mask','time_ps','aatype','residue_index','chain_id')}))
        for count in MICROSTEPS:
            branch,tensors=run_branch(model,state,selection['horizon'],count)
            branch['macro_times_ps']=[i*1000 for i in range(tensors['H'].shape[1])]
            branch['macro_statistics']=macro_statistics(tensors['H'],tensors['S'],m)
            branch['budget_totals']={space:{key:float(value.sum(0).mean()/count_dim)
                for key,value in terms.items()} for space,terms in tensors['budgets'].items()}
            if branch['status']=='completed':
                ca=decode_macro(model,state,tensors['H'])
                tensors['CA']=ca
                if not bool(torch.isfinite(ca[:,:,m]).all()):
                    branch['status']='numerical_failure'
                    bad=(~torch.isfinite(ca[:,:,m])).flatten(2).sum(-1)
                    branch['failure']=dict(stage='CA_decode',nonfinite_per_member_macro=bad.tolist())
            raw['branches'][str(count)]=tensors;row['branches'][str(count)]=branch
        # Endpoint labels are unavailable until all forward branches finish.
        target=reader._read_clip(record,np.array([selection['target_frame']],dtype=np.int64))
        validate_clip(target,[selection['target_frame']],record)
        require(bool(target['atom_mask'][0,m].all()),'endpoint missing observed-valid atom')
        require(all(torch.equal(target[k],context[k]) for k in ('aatype','residue_index','chain_id')),
                'endpoint metadata differs')
        raw['target']=target
        row['label_sha256']=state_sha256({k:target[k] for k in ('coords','atom_mask','time_ps')})
        for count in MICROSTEPS:
            key=str(count);branch=row['branches'][key]
            if branch['status']=='completed':
                try:
                    branch['endpoint_scores']=coordinate_scores(raw['branches'][key]['CA'][:,-1],
                        target['coords'][0,:,1],state['anchor'][0],m,context['residue_index'],context['chain_id'])
                except NumericalBranchFailure as error:
                    branch.update(status='numerical_failure',failure=dict(stage='endpoint_metrics',error=str(error)),endpoint_scores=None)
            else:branch['endpoint_scores']=None
            for metric in ('coordinate_energy_score','member_rmsd'):
                row['scalars'][f'm{count}/{metric}']=branch['endpoint_scores'][metric] if branch['endpoint_scores'] else None
            # Use the actual canonical metric keys, including null quantities;
            # a failed branch inherits the same key set from persistence below.
            if branch['endpoint_scores']:
                for metric,value in branch['endpoint_scores']['geometry_mean'].items():
                    row['scalars'][f'm{count}/geometry/{metric}']=value
            for metric in ('channel_mean_squared','channel_variance_mean','norm_squared_per_dim','bath_norm_squared_per_dim'):
                row['scalars'][f'm{count}/{metric}']=float(branch['macro_statistics'][metric][:,-1].mean()) if branch['status']=='completed' else None
            for space in ('raw','centered'):
                for metric in BUDGET_KEYS:
                    row['scalars'][f'm{count}/{space}/{metric}']=branch['budget_totals'][space][metric] if branch['status']=='completed' else None
        row['persistence']=coordinate_scores(state['anchor'].expand(12,-1,-1),target['coords'][0,:,1],
            state['anchor'][0],m,context['residue_index'],context['chain_id'])
        for metric in ('coordinate_energy_score','member_rmsd'):row['scalars'][f'persistence/{metric}']=row['persistence'][metric]
        for metric,value in row['persistence']['geometry_mean'].items():
            row['scalars'][f'persistence/geometry/{metric}']=value
            for count in MICROSTEPS:
                row['scalars'].setdefault(f'm{count}/geometry/{metric}',None)
        for left,right in ((1,4),(4,8)):
            key=f'{left}_vs_{right}'
            if all(row['branches'][str(k)]['status']=='completed' for k in (left,right)):
                row['differences'][key]={kind:coupled_difference(raw['branches'][str(left)][kind],
                    raw['branches'][str(right)][kind],m,coordinates=kind=='CA') for kind in ('H','CA')}
            else:row['differences'][key]=None
            for kind in ('H','CA'):
                row['scalars'][f'{key}/{kind}_endpoint']=row['differences'][key][kind]['endpoint_mean'] if row['differences'][key] else None
        raw_path=OUT/f"window_{selection['step']:02d}.pt"
        save_new(raw_path,raw)
        row['raw_tensors']=dict(path=str(raw_path.relative_to(REPO)),sha256=sha(raw_path))
        write_new(OUT/f"window_{selection['step']:02d}.json",row)
        rows.append(plain(row))
        print(json.dumps(dict(completed=len(rows),total=32,horizon=selection['horizon'],
            statuses={k:v['status'] for k,v in row['branches'].items()},seconds=time.monotonic()-started)),flush=True)
        del raw,state
    require(len(rows)==32,'incomplete panel')
    require(state_sha256(model)==protocol['model_tensor_sha256'],'model changed during diagnostic')
    require(all(p.grad is None for p in model.parameters()),'unexpected parameter gradients')
    require(not torch.cuda.is_initialized(),'CUDA initialized')
    verify_registered(protocol,protocol_sha)
    result=dict(schema_version=VERSION,protocol_sha256=protocol_sha,n_windows=32,
        n_proteins=len({r['metadata']['protein'] for r in rows}),per_window=rows,
        numerical_failure_branches=sum(b['status']!='completed' for r in rows for b in r['branches'].values()),
        source_input_hdf5_model_before_after_verified=True,overall=summarize(rows),
        by_horizon={str(h):summarize([r for r in rows if r['metadata']['horizon']==h]) for h in HORIZONS},
        completed_utc=now(),elapsed_seconds=time.monotonic()-started)
    write_new(OUT/'diagnostic.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['register','run'],required=True)
    args=parser.parse_args()
    register() if args.mode=='register' else run()


if __name__=='__main__':
    main()
