from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from scripts import summarize_hs_ordered_pilot_v1 as s


def execution_fixture():
    protocol = dict(input_hashes={'input': 'a'}, source_hashes={'source': 'b'}, per_arm_commands={})
    executions = {}
    for arm in s.ARMS:
        commands = {kind: ['python', kind, arm] for kind in s.JOBS}
        protocol['per_arm_commands'][arm] = commands
        executions[arm] = dict(arm=arm, status='completed', runner_pid=11, completed_utc='done',
            successful_logical_jobs=6, jobs=[dict(kind=kind, pid=12, returncode=0,
                started_utc='start', completed_utc='done', command=commands[kind],
                frozen_before={'input': 'a', 'source': 'b'}, frozen_after={'input': 'a', 'source': 'b'})
                for kind in s.JOBS])
    return executions, protocol


def test_twelve_real_jobs_are_required_before_score_or_checkpoint_access():
    executions, protocol = execution_fixture()
    s.execution_headers(executions, protocol)
    executions[s.ARMS[1]]['status'] = 'running'
    with pytest.raises(ValueError, match='both arms'):
        s.execution_headers(executions, protocol)


@pytest.mark.parametrize('mutation', ['missing_job', 'exit2', 'bool_exit', 'no_pid', 'command', 'frozen'])
def test_execution_rejects_partial_or_modified_evidence(mutation):
    executions, protocol = execution_fixture()
    record = executions[s.ARMS[0]]
    job = record['jobs'][2]
    if mutation == 'missing_job': record['jobs'].pop()
    elif mutation == 'exit2': job['returncode'] = 2
    elif mutation == 'bool_exit': job['returncode'] = False
    elif mutation == 'no_pid': job['pid'] = 0
    elif mutation == 'command': job['command'] = ['different']
    else: job['frozen_after']['input'] = 'changed'
    with pytest.raises(ValueError):
        s.execution_headers(executions, protocol)


def test_build_summary_stops_at_headers_before_any_model_or_score(monkeypatch):
    executions, protocol = execution_fixture()
    executions[s.ARMS[1]]['status'] = 'running'
    reads = []
    class Inputs:
        def read(self, path):
            path = str(path); reads.append(path)
            if path.endswith('/joint_path.execution.json'): return executions[s.ARMS[0]]
            if path.endswith('/marginal_path.execution.json'): return executions[s.ARMS[1]]
            if path == str(s.BASE/'protocol.json'): return protocol
            return {}
    monkeypatch.setattr(s.old, 'Inputs', Inputs)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('checkpoint loaded before complete jobs'))
    with pytest.raises(ValueError, match='both arms'):
        s.build_summary()
    assert not any(any(path.endswith(f'.{kind}.json') for kind in s.JOBS[1:]) for path in reads)


def optimizer():
    return dict(param_groups=[{'params': [0], 'lr': .001}], state={0: dict(step=torch.tensor(500.),
        exp_avg=torch.tensor([1., 2.]), exp_avg_sq=torch.tensor([3., 4.]))})


def test_optimizer_support_and_step_equal_but_moments_may_differ():
    a, b = optimizer(), optimizer()
    b['state'][0]['exp_avg'] *= 2
    assert s.optimizer_pair(a, b) == dict(active_states=1, steps=500, moments_may_differ=True)


@pytest.mark.parametrize('mutation', ['step', 'support', 'group', 'shape', 'nan'])
def test_optimizer_metadata_changes_fail(mutation):
    a, b = optimizer(), optimizer()
    if mutation == 'step': b['state'][0]['step'] = torch.tensor(499.)
    elif mutation == 'support': b['state'][1] = b['state'].pop(0)
    elif mutation == 'group': b['param_groups'][0]['lr'] = .002
    elif mutation == 'shape': b['state'][0]['exp_avg'] = torch.ones(3)
    else: b['state'][0]['exp_avg'][0] = float('nan')
    with pytest.raises(ValueError): s.optimizer_pair(a, b)


def panel_rows():
    rows = {}
    for i in range(50):
        for h in (1, 4, 16, 64):
            key = (f'p{i:02}', f't{i:02}', (320., 348., 379., 413., 450.)[i//10], 7, h)
            rows[key] = dict(target_frame=7+h, context_frames=list(range(8)), horizon_ps=h*1000,
                valid_residues=8, sampling_seed=i*100+h,
                time_grid={'step_ps': 1000}, n_forecast_paths=12, n_reference_paths=1,
                temporal_controls={}, input_fingerprints={'observed': 'o', 'reference': 'r'}, arms={})
            for role in s.paths.ARMS:
                rows[key]['arms'][role] = dict(metrics=dict(conventions={'p': 1}, lag_statistics={'reference': {}}, scalars={}))
    return rows


def test_cross_task_pairing_requires_all_200_not_an_intersection():
    base = panel_rows()
    maps = [{a: deepcopy(base) for a in s.ARMS} for _ in range(3)]
    s.cross_task_pairing(*maps, base)
    maps[2][s.ARMS[1]].pop(next(iter(base)))
    with pytest.raises(ValueError, match='keys differ'):
        s.cross_task_pairing(*maps, base)


def test_cross_task_endpoint_mask_and_cross_arm_reference_hash_must_match():
    base = panel_rows()
    maps = [{a: deepcopy(base) for a in s.ARMS} for _ in range(3)]
    key = next(iter(base))
    maps[2][s.ARMS[0]][key]['valid_residues'] = 7
    with pytest.raises(ValueError, match='observation contract'):
        s.cross_task_pairing(*maps, base)
    maps[2][s.ARMS[0]][key]['valid_residues'] = 8
    maps[2][s.ARMS[1]][key]['input_fingerprints']['reference'] = 'changed'
    with pytest.raises(ValueError, match='input tensor'):
        s.cross_task_pairing(*maps, base)


def suite_fixture():
    row = dict(persist={'score': 1.}, model={'score': 2.}, position={
        'persist_kabsch_ca_rmsd': 1., 'persist_aligned_ca_rmsd': 2.})
    return {a: {'suite': {'key': deepcopy(row)}, 'matched8': {'key': deepcopy(row)}} for a in s.ARMS}


def test_suite_requires_exact_full_matched_rows_but_predictions_can_differ_between_arms():
    rows = suite_fixture()
    for kind in ('suite', 'matched8'): rows[s.ARMS[1]][kind]['key']['model']['score'] = 3.
    s.suite_pairing(rows)
    rows[s.ARMS[1]]['matched8']['key']['model']['score'] = 4.
    with pytest.raises(ValueError, match='matched8/full'):
        s.suite_pairing(rows)


def test_suite_reference_mismatch_is_not_tolerated():
    rows = suite_fixture()
    for kind in ('suite', 'matched8'): rows[s.ARMS[1]][kind]['key']['persist']['score'] += 1e-12
    with pytest.raises(ValueError, match='reference differs'):
        s.suite_pairing(rows)


def test_legacy_slots_are_explicit_and_do_not_mutate_input():
    values = dict(joint_path={'x': 1}, marginal_path={'x': 2})
    mapped = s.slot(values)
    assert mapped == {'endpoint': {'x': 1}, 'path_mixture': {'x': 2}}
    assert set(values) == set(s.ARMS)
    with pytest.raises(ValueError): s.slot({'endpoint': {}, 'path_mixture': {}})


def test_registered_count_includes_one_new_primary_and_60_ordered_intervals():
    assert s.contract()['n_paired_intervals'] == 402
    nested = dict(standard=[{'ci95': [0., 1.]} for _ in range(341)],
        primary={'ci95': [0., 1.]}, ordered=[{'ci95': [0., 1.]} for _ in range(60)])
    assert s.count_intervals(nested) == 402


def test_register_refuses_score_before_reading_score_or_sources(monkeypatch, tmp_path):
    monkeypatch.setattr(s, 'BASE', tmp_path/'base')
    monkeypatch.setattr(s, 'IMPLEMENTATION', tmp_path/'wrapper')
    monkeypatch.setattr(s.old, 'local', lambda p: Path(p))
    s.BASE.mkdir()
    (s.BASE/'joint_path.ordered.json').write_text('THIS MUST NOT BE PARSED')
    monkeypatch.setattr(s.old, 'Inputs', lambda: pytest.fail('score preexistence must be checked first'))
    with pytest.raises(ValueError, match='new score exists'):
        s.register()


def test_exclusive_json_write_never_overwrites(monkeypatch, tmp_path):
    monkeypatch.setattr(s.old, 'local', lambda p: Path(p))
    output = tmp_path/'new.json'
    s.write_new(output, {'a': 1})
    with pytest.raises(FileExistsError): s.write_new(output, {'a': 2})
    assert json.loads(output.read_text()) == {'a': 1}
    with pytest.raises(ValueError): s.write_new(tmp_path/'nonfinite.json', {'a': float('nan')})
    assert not (tmp_path/'nonfinite.json').exists()


def test_initial_inference_hash_uses_real_model_prefixes():
    from protein_jepa_frame.utils.contracts import state_sha256
    ck = dict(component_checkpoint={'encoder': {'x': torch.tensor([1.])}, 'delta_ca_decoder': {'y': torch.tensor([2.])}},
        kernel={'z': torch.tensor([3.])})
    assert s.inference_hash(ck) == state_sha256({'bundle.encoder.x': torch.tensor([1.]),
        'bundle.decoder.y': torch.tensor([2.]), 'kernel.z': torch.tensor([3.])})
    assert s.inference_hash(ck, component_only=True) == state_sha256({'encoder.x': torch.tensor([1.]), 'decoder.y': torch.tensor([2.])})


def test_complete_training_audit_synthetic_500_rows(monkeypatch, tmp_path):
    from protein_jepa_frame.utils.contracts import state_sha256
    monkeypatch.setattr(s, 'ARTIFACTS', tmp_path/'artifacts')
    monkeypatch.setattr(s, 'BASE', tmp_path/'results')
    monkeypatch.setattr(s.old, 'component_identity', lambda c: {'fixed': 'component'})
    monkeypatch.setattr(s.runner, 'checkpoint_audit', lambda ck, arm, p: {'checked': arm})
    class Inputs:
        def track(self, path, expected=None):
            return Path(path)
        def read(self, path):
            return json.loads(Path(path).read_text())
    def write(path, value, lines=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(json.dumps(r) for r in value)+'\n' if lines else json.dumps(value))
    initial = dict(component_checkpoint={'encoder': {'x': torch.tensor([1.])}, 'delta_ca_decoder': {'y': torch.tensor([2.])}},
        kernel={'z': torch.tensor([3.])}, kernel_config={'step_ps': 1000})
    expected_windows = [dict(step=i, horizon=(1, 4, 16, 64)[(i-1)%4], record_path='/synthetic.h5') for i in range(1, 501)]
    protocol = dict(initial_kernel_sha256=state_sha256(initial['kernel']), initial_kernel_config=initial['kernel_config'],
        initial_component_identity={'fixed': 'component'}, training_windows=expected_windows,
        selected_hdf5_stats={'/synthetic.h5': {'size': 10}},
        input_hashes={s.old.shared.TRAIN: 'train', s.old.shared.VAL: 'val'})
    split = dict(protein_overlap=0, domain_overlap=0, source_file_overlap=0,
        train_manifest_sha256='train', val_manifest_sha256='val')
    identities = dict(model_sha256=s.inference_hash(initial), component_sha256=s.inference_hash(initial, component_only=True),
        kernel_sha256=state_sha256(initial['kernel']), kernel_config=initial['kernel_config'])
    checkpoints, executions = {}, {}
    for arm in s.ARMS:
        directory = s.ARTIFACTS/arm
        args = dict(output=str(directory), coord_objective=arm, seed=42)
        ck = dict(initial, args=args, optimizer=optimizer(), sampling_rng=torch.tensor([1, 2], dtype=torch.uint8), rng_state={'global': [1, 2]})
        checkpoints[arm] = ck
        write(directory/'run.json', dict(args=args, split_audit=split, train_only=True, test_opened=False,
            no_validation_selection=True, actual_pid=12))
        write(directory/'initial_identity.json', identities)
        rows = [dict(row, actual_batch_sha256='a'*64, file_stat={'size': 10}) for row in expected_windows]
        write(directory/'training_windows.jsonl', rows, True)
        metrics = [dict(step=row['step'], horizon=row['horizon'], grad_norm=1., lr=.001, train={'loss': 1.}) for row in rows]
        write(directory/'metrics.jsonl', metrics, True)
        executions[arm] = dict(checkpoint_audit={'checked': arm}, latest_sha256='checkpoint', jobs=[dict(pid=12,
            training_windows_audit=s.runner.audit_training_windows(directory/'training_windows.jsonl', expected_windows))])
    pair = dict(initial_model_equal=True, actual_training_window_bytes_equal=True, final_sampling_and_global_rng_equal=True,
        only_argument_differences=['output', 'coord_objective'], compared_with_historical_endpoint_rng=False)
    write(s.BASE/'paired_training_audit.json', pair)
    checked, audited = s.audit_training(checkpoints, initial, executions, protocol, Inputs())
    assert set(checked) == set(s.ARMS) and audited['n_actual_training_windows'] == 500
    assert audited['first_h1_loss_and_gradient_equal'] is True
    path = s.ARTIFACTS/s.ARMS[1]/'training_windows.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]['actual_batch_sha256'] = 'b'*64
    write(path, rows, True)
    executions[s.ARMS[1]]['jobs'][0]['training_windows_audit'] = s.runner.audit_training_windows(path, expected_windows)
    with pytest.raises(ValueError, match='actual 500 training batches differ'):
        s.audit_training(checkpoints, initial, executions, protocol, Inputs())


def test_ordered_wrapper_calls_full_panel_auditor(monkeypatch):
    from scripts import eval_ca_ordered_paths as producer
    calls = []
    data = dict(task_version='ca_ordered_paths_development_v1', dataset_role='development_validation',
        optimizer_steps=0, model_selection=False, test_opened=False, bioemu2_rerun=False,
        future_information_given_to_sampler=False, frozen_inputs_verified=True, completed_utc='done',
        panel=producer.panel_spec([1, 4, 16, 64], 200), n_windows=200, n_proteins=50,
        args={'checkpoint': 'synthetic.pt'}, input_hashes={'synthetic.pt': 'sha'},
        model_tensor_sha256='tensor', sampling=dict(seed=0, n_paths=12, model_history_frames=4,
            prediction_kind='endpoint', s0_mode='exact_spatial', one_continuous_rollout_per_member=True,
            no_future_reencoding=True, no_bath_reset=True), protocol={},
        source_hashes={p: 'source' for p in producer.SOURCES}, per_window=['raw'], by_horizon={'h': 1})
    class Inputs:
        def read(self, p): return {}
        def track(self, *a): return None
    def audit(rows, panel):
        calls.append((rows, panel))
        assert panel == producer.panel_spec([1, 4, 16, 64], 200)
        return rows
    monkeypatch.setattr(s.old.shared, 'audit_eval_protocol', lambda *a: None)
    monkeypatch.setattr(s.old.shared, 'audit_provenance', lambda *a: None)
    monkeypatch.setattr(s.paths, 'audit_rows', audit)
    monkeypatch.setattr(producer, 'summarize', lambda rows: {'h': 1})
    monkeypatch.setattr(s.old.shared, 'rows_for', lambda data, where: {'checked': 1})
    checked = dict(checkpoint='synthetic.pt', checkpoint_sha256='sha', inference_model_tensor_sha256='tensor')
    protocol = dict(source_hashes=data['source_hashes'])
    assert s.audit_ordered(data, s.ARMS[0], checked, protocol, Inputs()) == {'checked': 1}
    assert len(calls) == 1 and calls[0][0] == ['raw']
    data['source_hashes'] = {}
    with pytest.raises(ValueError, match='source coverage'):
        s.audit_ordered(data, s.ARMS[0], checked, protocol, Inputs())
