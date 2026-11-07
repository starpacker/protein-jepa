"""Registered first-four TRAIN kernel gradients; no optimizer updates or evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO/'results/hs_joint_20261101/contact_kernel_gradients_v2'
HISTORY = 'results/hs_joint_20261101/ordered_path_v1/protocol.json'
HISTORY_SHA = 'a31b87128702cd796b25fbbef60c3bd1245304cfd369c668970e54c72c7235cb'
WINDOW_LOG = 'artifacts/hs_joint_20261101/ordered_path_v1/marginal_path/training_windows.jsonl'
WINDOW_LOG_SHA = 'cbcad706d02645312d3a8209f0b18210a375998bb3b532a2f78788f24b8a71cb'
INITIAL = 'artifacts/hs_joint_20261101/spatial_probe_v1/spatial_exact.pt'
INITIAL_SHA = '396e6e4df35019140324484bf8d560a292cf2688a11946f2e0ccde5cd6ea8dda'
VERSION = 'contact_kernel_gradients_v2'
TERMS = ('plugin', 'variance_penalty_20', 'corrected')
ATOL, RTOL = 2e-6, 5e-4
SOURCES = (
    'run_scripts/diagnose_contact_kernel_gradients_v2.py',
    'tests/test_contact_kernel_gradients_v2.py',
    'docs/CONTACT_KERNEL_GRADIENTS_V2_20261102.md',
    'protein_jepa_frame/training/contact_corrected_path_objective_v1.py',
    'tests/test_contact_corrected_path_objective_v1.py',
    'protein_jepa_frame/training/contact_ensemble_objective_v1.py',
    'tests/test_contact_ensemble_objective_v1.py',
)
PRIOR_FILES = {
    'run_scripts/diagnose_contact_kernel_gradients_v1.py': '83fcc18924ac44641a5b93d3fb654be8c73da3b3395d06303fd8cc5859b59651',
    'tests/test_contact_kernel_gradients_v1.py': '1cc8a67832b8d8fd86a00b90dcb3d28b27f2c8d320523a871ca2ebe6f36fd523',
    'docs/CONTACT_KERNEL_GRADIENTS_V1_20261102.md': 'e0ddd1001f2aeb71c9f2b10bb27887b29bea2a8588b83595a99427ff25f67454',
    'results/hs_joint_20261101/contact_kernel_gradients_v1/protocol.json': '34631634365b2dfc65df91a008b7903f599c77e4118e3e3d7b6ce3ef9e5192b9',
    'results/hs_joint_20261101/contact_kernel_gradients_v1/started.json': 'fd9091e4f52baafb682d709d629513beeee4bd01994473d634bb17baa8ec3fda',
    'results/hs_joint_20261101/contact_kernel_gradients_v1/execution.json': '15c650043da1ca89f0b8c957bd8a4061c1936d826fc075d08a3deb921c46f847',
    'results/hs_joint_20261101/contact_kernel_gradients_run_v1.execution.json': 'dea6db325645e8e62d682a178ab5a2679f3eaac4e5b13b39a516270f006a540e',
    'results/hs_joint_20261101/contact_kernel_gradients_run_v1.log': '67444cded19933270ca1aa39118d41ff177980a3597980e3e2d6449584bcef95',
}
PRIOR_EXECUTION = 'results/hs_joint_20261101/contact_kernel_gradients_v1/execution.json'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def now():
    return datetime.now(timezone.utc).isoformat()


def path(value):
    p = Path(value)
    p = (p if p.is_absolute() else REPO/p).resolve()
    require(p.is_relative_to(REPO) and p.name != 'test.jsonl', 'outside repository or sealed test input')
    return p


def sha(value):
    digest = hashlib.sha256()
    with path(value).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(file, value):
    encoded = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with Path(file).open('x') as stream:
        stream.write(encoded)


def file_stat(file):
    value = Path(file).stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns)


def prior_failure():
    for name,digest in PRIOR_FILES.items():
        require(sha(name) == digest, 'prior failure evidence changed: '+name)
    failed = json.loads(path(PRIOR_EXECUTION).read_text())
    require(failed['status'] == 'failed' and failed['exit_code'] == 1 and
            failed['error'] == 'ValueError: kernel tensor count differs' and
            failed['completed_utc'] == '2026-11-02T05:29:20.042491+00:00', 'prior failure identity differs')
    return dict(prior_version='v1', actual_exit_code=1, completed_utc=failed['completed_utc'],
        defect='64 gradient-bearing tensors in the old smoke were mistaken for all 73 trainable kernel tensors.',
        repair='Keep all 73 named tensors, including unused tensors with explicit zero gradients and counts.',
        prior_stage='Model instantiated and initial kernel verified; count assertion failed before loaders, trajectory reads or rollout.',
        prior_rollouts=0, prior_optimizer_steps=0, numerical_definitions_changed=False,
        gradient_tolerances_changed=False, preserved_files=PRIOR_FILES)


def kernel_inventory(kernel):
    return [dict(name=name,shape=list(p.shape),dtype=str(p.dtype),requires_grad=p.requires_grad)
            for name,p in kernel.named_parameters()]


def rng_fingerprint(generator):
    """Hash all global streams plus the continuous explicit sampling stream."""
    from protein_jepa_frame.training.lasp_run_state import rng_state
    from protein_jepa_frame.utils.contracts import state_sha256
    state = rng_state()
    numpy = state['numpy']
    flat = dict(python=state['python'], numpy_metadata=[numpy[0], *numpy[2:]],
                numpy_keys=torch.from_numpy(numpy[1].copy()), torch=state['torch'])
    flat.update({f'cuda_{i}': value for i, value in enumerate(state['cuda'])})
    return dict(global_sha256=state_sha256(flat),
                sampling_sha256=state_sha256({'state': generator.get_state()}))


def ratio(a, b):
    return a/b if b else None


def compare_vectors(left, right):
    require(left.shape == right.shape and left.ndim == 1, 'gradient comparison shape')
    require(bool(torch.isfinite(left).all() and torch.isfinite(right).all()), 'nonfinite gradient')
    a, b = left.double(), right.double()
    na, nb = float(a.norm()), float(b.norm())
    dot = float(a @ b)
    return dict(dot=dot, cosine=ratio(dot, na*nb), left_over_right_norm=ratio(na, nb),
                change_right_along_minus_left=-dot,
                change_right_along_unit_minus_left=ratio(-dot, na))


def vector_statistics(vectors):
    require(set(vectors) == set(TERMS), 'three gradient terms required')
    require(all(v.ndim == 1 and v.shape == vectors['plugin'].shape for v in vectors.values()),
            'gradient term shape mismatch')
    require(vectors['plugin'].numel() > 0, 'empty gradient vector')
    require(all(bool(torch.isfinite(v).all()) for v in vectors.values()), 'nonfinite gradient')
    v = {k: x.double() for k, x in vectors.items()}
    norms = {k: float(x.norm()) for k, x in v.items()}
    # Same scalar coefficient as clip_grad_norm_(max_norm=1), without mutation.
    scales = {k: min(1., 1./(n+1e-6)) for k, n in norms.items()}
    return dict(norms=norms, finite_elements={k: x.numel() for k,x in v.items()},
        unit_clip_scale=scales, unit_clipped_norms={k: norms[k]*scales[k] for k in TERMS},
        penalty_vs_plugin=compare_vectors(v['variance_penalty_20'], v['plugin']),
        plugin_vs_corrected=compare_vectors(v['plugin'], v['corrected']),
        unit_clipped_plugin_vs_corrected=compare_vectors(
            v['plugin']*scales['plugin'], v['corrected']*scales['corrected']))


def collect_gradients(model, terms):
    """Three derivatives of one forward graph; never populate .grad or step."""
    named = list(model.kernel.named_parameters())
    require(named and all(p.requires_grad for _,p in named), 'kernel trainable parameter support')
    require(not any(p.requires_grad or p.grad is not None for p in model.bundle.parameters()),
            'component must be frozen and gradient-free')
    require(all(p.grad is None for _,p in named), 'clear kernel gradients before forward')
    losses = dict(plugin=terms['loss_plugin'], variance_penalty_20=20*terms['contact_variance'],
                  corrected=terms['loss'])
    require(all(x.ndim == 0 and bool(torch.isfinite(x)) and x.requires_grad for x in losses.values()),
            'loss must be finite differentiable scalar')
    parameters = [p for _,p in named]
    gradients, unused = {}, {}
    for i, (name, loss) in enumerate(losses.items()):
        values = torch.autograd.grad(loss, parameters, retain_graph=i < 2, allow_unused=True)
        unused[name] = [n for (n,_),g in zip(named,values) if g is None]
        gradients[name] = {n: (torch.zeros_like(p) if g is None else g).detach().cpu().clone()
                           for (n,p),g in zip(named,values)}
    require(not any(p.grad is not None for p in model.parameters()), 'autograd populated parameter .grad')
    return dict(parameter_names=[n for n,_ in named], gradients=gradients, unused=unused,
                shapes={n:list(p.shape) for n,p in named}, dtypes={n:str(p.dtype) for n,p in named})


def summarize_gradients(saved):
    names = saved['parameter_names']
    require(names and len(names) == len(set(names)), 'unique kernel names required')
    for term in TERMS:
        require(set(saved['gradients'][term]) == set(names), 'kernel name support differs')
        for name in names:
            value = saved['gradients'][term][name]
            require(list(value.shape) == saved['shapes'][name] and str(value.dtype) == saved['dtypes'][name],
                    'saved gradient shape/dtype mismatch')
    def flatten(selected):
        return {term:torch.cat([saved['gradients'][term][n].reshape(-1) for n in selected]) for term in TERMS}
    vectors = flatten(names)
    expected = vectors['plugin'].double()-vectors['variance_penalty_20'].double()
    error = vectors['corrected'].double()-expected
    tolerance = ATOL+RTOL*expected.abs()
    report = vector_statistics(vectors)
    report['identity'] = dict(max_absolute_error=float(error.abs().max()), l2_error=float(error.norm()),
        relative_l2_error=ratio(float(error.norm()), float(expected.norm())), atol=ATOL, rtol=RTOL,
        passed=bool((error.abs() <= tolerance).all()),
        convention='elementwise abs(corrected-(plugin-penalty)) <= atol+rtol*abs(plugin-penalty)')
    report['parameters'] = {term:dict(total=len(names), unused=len(saved['unused'][term]),
        unused_names=saved['unused'][term], nonzero=sum(bool(saved['gradients'][term][n].count_nonzero()) for n in names))
        for term in TERMS}
    groups = sorted({name.split('.')[0] for name in names})
    report['module_groups'] = {group:dict(parameter_names=[n for n in names if n.split('.')[0] == group],
        **vector_statistics(flatten([n for n in names if n.split('.')[0] == group]))) for group in groups}
    return report


def verify_files(protocol):
    for name,digest in {**protocol['source_hashes'], **protocol['input_hashes']}.items():
        require(sha(name) == digest, 'frozen file changed: '+name)
    for name,digest in protocol['source_hashes'].items():
        require(sha(path(protocol['source_snapshot'])/name) == digest, 'source snapshot changed: '+name)
    for name,expected in protocol['selected_hdf5_stats'].items():
        require(file_stat(name) == expected, 'selected HDF identity changed: '+name)


def register():
    require(not OUT.exists(), 'refusing existing diagnostic directory')
    repair = prior_failure()
    require(sha(HISTORY) == HISTORY_SHA, 'ordered protocol changed')
    old = json.loads(path(HISTORY).read_text())
    selected = old['training_windows'][:4]
    require([r['step'] for r in selected] == [1,2,3,4] and
            [r['horizon'] for r in selected] == [1,4,16,64], 'first-four metadata changed')
    require(sha(WINDOW_LOG) == WINDOW_LOG_SHA, 'historical training-window log changed')
    historical = []
    with path(WINDOW_LOG).open() as stream:
        for _ in range(4):
            historical.append(json.loads(next(stream)))
    require(all(all(actual[k] == value for k,value in row.items()) for row,actual in zip(selected,historical)),
            'historical actual batch metadata differs')
    inputs = {HISTORY:HISTORY_SHA, WINDOW_LOG:WINDOW_LOG_SHA, **PRIOR_FILES}
    # Only training dependencies: no validation/test/baseline result access.
    for name in (INITIAL, 'artifacts/mdcath_manifest_full/train.jsonl',
                 'artifacts/stage_a_disp_relpos/checkpoint_latest.pt',
                 'artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt'):
        require(sha(name) == old['input_hashes'][name], 'training input changed: '+name)
        inputs[name] = old['input_hashes'][name]
    require(inputs[INITIAL] == INITIAL_SHA, 'initial checkpoint identity changed')
    sources = dict(old['source_hashes'])
    for name,digest in sources.items():
        require(sha(name) == digest and sha(path(old['source_snapshot'])/name) == digest,
                'historical source changed: '+name)
    sources.update({name:sha(name) for name in SOURCES})
    stats = {r['record_path']:old['selected_hdf5_stats'][r['record_path']] for r in selected}
    for name,expected in stats.items():
        require(file_stat(name) == expected, 'selected HDF stat changed')
    protocol = dict(schema_version=VERSION, registered_utc=now(), versioned_repair=repair, source_hashes=sources, input_hashes=inputs,
        source_snapshot=str((OUT/'source_snapshot').relative_to(REPO)), selected_hdf5_stats=stats,
        selected_training_windows=selected, historical_actual_batches=historical,
        training_arguments=old['training_arguments']['marginal_path'],
        initial_kernel_sha256=old['initial_kernel_sha256'], initial_component_identity=old['initial_component_identity'],
        initial_kernel_config=old['initial_kernel_config'], kernel_parameter_tensors=73,
        contact_estimator='population_u', contact_weight=20., seed=42, device='cuda:0', torch_threads=4,
        optimizer_steps=0, inference='descriptive first-four TRAIN gradient audit only; no CI or optimizer-update prediction',
        gradient_identity_tolerance=dict(atol=ATOL,rtol=RTOL),
        source_history=dict(protocol=HISTORY,sha256=HISTORY_SHA,source_snapshot=old['source_snapshot']),
        hdf_evidence='size/mtime identity before and after plus actual selected batch tensor hash; no full-HDF digest',
        validation_loaded=False,test_opened=False,bioemu2_rerun=False)
    OUT.mkdir(parents=True, exist_ok=False)
    for name,digest in sources.items():
        target = OUT/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(path(name).read_bytes())
        require(sha(target) == digest, 'source changed during snapshot')
    verify_files(protocol)
    write_new(OUT/'protocol.json',protocol)
    print(json.dumps(dict(protocol=str(OUT/'protocol.json'),sha256=sha(OUT/'protocol.json'))),flush=True)


def execute(protocol):
    """Only entered after full registered-file validation; training inputs only."""
    from dataclasses import asdict
    from protein_jepa_frame.training import train_lasp_ca_ordered_v1 as training
    from protein_jepa_frame.training.contact_corrected_path_objective_v1 import contact_corrected_path_losses
    from protein_jepa_frame.training.train_utils import _set_reproducibility, _move_batch
    from protein_jepa_frame.utils.contracts import state_sha256
    require(torch.cuda.is_available(), 'one visible CUDA device required')
    torch.set_num_threads(4)
    _set_reproducibility(42,False)
    args = training.arguments(str(OUT),'marginal_path',device='cuda:0',torch_threads=4)
    require({k:v for k,v in vars(args).items() if k != 'output'} ==
            {k:v for k,v in protocol['training_arguments'].items() if k != 'output'}, 'training arguments changed')
    args.contact_estimator = 'population_u'
    model = training.build_model(args)
    require(state_sha256(model.kernel) == protocol['initial_kernel_sha256'] and
            asdict(model.kernel.cfg) == protocol['initial_kernel_config'], 'initial kernel identity differs')
    initial = state_sha256(model)
    inventory = kernel_inventory(model.kernel)
    write_new(OUT/'initial.json',dict(model_sha256=initial,kernel_sha256=state_sha256(model.kernel),
        kernel_config=asdict(model.kernel.cfg),kernel_parameter_inventory=inventory))
    require(len(inventory) == protocol['kernel_parameter_tensors'] == 73 and
            all(row['requires_grad'] for row in inventory), 'kernel tensor count or trainability differs')
    loaders = {h:training.make_loader(args,args.manifest,h,True) for h in args.horizons}
    generator = torch.Generator(device=args.device).manual_seed(42)
    rows = []
    for selected,historical in zip(protocol['selected_training_windows'],protocol['historical_actual_batches']):
        verify_files(protocol)
        step,horizon = selected['step'],selected['horizon']
        require(training.training_identity(loaders[horizon],step,42) == selected, 'selected training metadata differs')
        batch,actual = training.load_training_batch(loaders[horizon],step,42)
        require(actual == historical, 'actual TRAIN batch differs from old run')
        model.train(); model.bundle.eval(); model.zero_grad(set_to_none=True)
        batch = _move_batch(batch,args.device)
        before = rng_fingerprint(generator)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        terms = contact_corrected_path_losses(model,batch,horizon,args,generator)
        require(torch.equal(terms['loss'],terms['loss_plugin']-20*terms['contact_variance']), 'corrected loss identity')
        losses = {k:float(v.detach()) for k,v in terms.items()}
        after_forward = rng_fingerprint(generator)
        saved = collect_gradients(model,terms)
        after_gradients = rng_fingerprint(generator)
        torch.cuda.synchronize()
        elapsed = time.monotonic()-started
        # Preserve actual vectors before applying numerical identity acceptance.
        sidecar = OUT/f'step{step:03d}.gradients.pt'
        with sidecar.open('xb') as stream:
            torch.save(dict(schema_version=VERSION,step=step,horizon=horizon,**saved),stream)
        statistics = summarize_gradients(saved)
        row = dict(step=step,horizon=horizon,batch=actual,losses=losses,gradient_statistics=statistics,
            gradient_sidecar=str(sidecar.relative_to(REPO)),gradient_sidecar_sha256=sha(sidecar),
            rng_before=before,rng_after_forward=after_forward,rng_after_gradients=after_gradients,
            seconds=elapsed,max_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
            max_memory_reserved_bytes=torch.cuda.max_memory_reserved(),model_sha256=state_sha256(model),
            optimizer_steps=0,parameter_gradients_present=any(p.grad is not None for p in model.parameters()))
        write_new(OUT/f'step{step:03d}.json',row)
        require(after_forward == after_gradients, 'gradient recomputation consumed RNG')
        require(statistics['identity']['passed'], 'corrected gradient identity outside registered tolerance')
        require(row['model_sha256'] == initial and not row['parameter_gradients_present'], 'model or .grad changed')
        verify_files(protocol)
        rows.append(row)
        print(json.dumps(dict(step=step,horizon=horizon,seconds=elapsed,gradient_norms=statistics['norms'])),flush=True)
        del terms,saved,batch
    return dict(schema_version=VERSION,completed_utc=now(),rows=rows,optimizer_steps=0,
        initial_model_sha256=initial,final_model_sha256=state_sha256(model),initial_kernel_parameter_inventory=inventory,
        versioned_repair=protocol['versioned_repair'],final_rng=rng_fingerprint(generator),
        device=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
        inference=protocol['inference'],validation_loaded=False,test_opened=False,bioemu2_rerun=False,
        source_hashes=protocol['source_hashes'],input_hashes=protocol['input_hashes'])


def run():
    require((OUT/'protocol.json').exists(), 'register before run')
    require(not any((OUT/name).exists() for name in ('started.json','execution.json','result.json')), 'refusing rerun')
    started = time.monotonic()
    identity = dict(pid=os.getpid(),command=sys.argv,started_utc=now(),protocol_sha256=sha(OUT/'protocol.json'))
    write_new(OUT/'started.json',identity)
    try:
        protocol = json.loads((OUT/'protocol.json').read_text())
        require(protocol['schema_version'] == VERSION, 'protocol version differs')
        require(prior_failure() == protocol['versioned_repair'], 'versioned failure evidence differs')
        verify_files(protocol)  # Must precede model instantiation and raw trajectory reads.
        result = execute(protocol)
        verify_files(protocol)
        require(sha(OUT/'protocol.json') == identity['protocol_sha256'], 'protocol changed during run')
        write_new(OUT/'result.json',result)
    except BaseException as error:
        write_new(OUT/'execution.json',dict(identity,status='failed',exit_code=1,seconds=time.monotonic()-started,
                  completed_utc=now(),error=f'{type(error).__name__}: {error}'))
        raise
    write_new(OUT/'execution.json',dict(identity,status='completed',exit_code=0,seconds=time.monotonic()-started,
              completed_utc=now(),output_sha256=sha(OUT/'result.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('register','run'),required=True)
    args = parser.parse_args()
    register() if args.mode == 'register' else run()
