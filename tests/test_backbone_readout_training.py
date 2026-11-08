"""CPU-only synthetic training contracts for the separate fixed-CA head."""
import copy
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch

from protein_jepa_frame.utils.contracts import state_sha256
from run_scripts import run_backbone_readout_pilot as runner


def fixtures():
    fixture = runpy.run_path(str(Path(__file__).with_name('test_backbone_readout_pilot.py')))['fixture']
    head = fixture()[0]['init']
    rows = []
    for index, horizon in enumerate((1,4,16,64)):
        row = fixture(horizon)[1]
        row['metadata'].update(protein=f'p{index}',trajectory=f'p{index}_T320_R0',sampling_seed=None)
        row.pop('forecast_H'); row.pop('forecast_CA')
        rows.append(row)
    return head, rows


def test_actual_NCO_objective_and_geometry_backpropagate_only_to_head():
    head, rows = fixtures()
    row = rows[0]
    for name in ('oracle_H','oracle_CA','observed_R','target_backbone'):
        row[name].requires_grad_()
    before = {k:v.detach().clone() for k,v in row.items() if torch.is_tensor(v)}
    coordinate, geometry, details = runner.actual_losses(head,row)
    mask = row['observed_mask']
    with torch.no_grad():
        output = head(row['oracle_H'],row['observed_R'],row['oracle_CA'],
            row['embedding_aatype'][None],row['chemical_aatype'][None],mask[None])
        error = output['backbone'][...,[0,2,3],:] - row['target_backbone'][...,[0,2,3],:]
        expected = error.square().sum(-1).mean()
    assert torch.allclose(coordinate,expected,rtol=2e-7,atol=1e-7)
    assert torch.equal(geometry,sum(runner.WEIGHTS[k]*details['losses'][k] for k in runner.WEIGHTS))
    (coordinate+geometry).backward()
    assert all(row[name].grad is None for name in ('oracle_H','oracle_CA','observed_R','target_backbone'))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters() if p.requires_grad)
    assert any(p.grad.abs().sum()>0 for p in head.parameters() if p.requires_grad)
    for name,value in before.items(): assert torch.equal(value,row[name])


def test_CA_target_error_is_excluded_and_no_loss_realignment():
    head,rows = fixtures()
    first=runner.actual_losses(head,rows[0])
    changed=copy.deepcopy(rows[0]);changed['target_backbone'][:,:,1]+=100.
    second=runner.actual_losses(head,changed)
    assert torch.equal(first[0],second[0]) and torch.equal(first[1],second[1])
    changed['target_backbone'][...,[0,2,3],:]+=100.
    assert runner.actual_losses(head,changed)[0]>first[0]+1000


def test_training_geometry_uses_explicit_edges_despite_duplicate_legacy_numbers():
    head,rows=fixtures();row=rows[0]
    row['embedding_residue_index'].fill_(1)
    coordinate,geometry,details=runner.actual_losses(head,row)
    assert details['counts']['n_connection_candidates'].tolist()==[4]
    assert torch.isfinite(coordinate) and torch.isfinite(geometry)


def test_gradient_calibration_is_read_only_and_learning_rate_schedule_is_fixed():
    head,rows=fixtures();head.eval()
    before=state_sha256(head)
    parameters=[p for p in head.parameters() if p.requires_grad]
    coordinate,geometry,_=runner.actual_losses(head,rows[0])
    assert runner.grad_norm(coordinate,parameters,retain_graph=True)>0
    assert runner.grad_norm(geometry,parameters)>0
    assert state_sha256(head)==before and all(p.grad is None for p in head.parameters())
    assert runner.lr_at(1,500)==pytest.approx(4e-6)
    assert runner.lr_at(25,500)==pytest.approx(1e-4)
    assert runner.lr_at(500,500)==pytest.approx(1e-5)
    assert runner.lr_at(4,4)==pytest.approx(1e-4)


def synthetic_inputs(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'REPO',tmp_path)
    monkeypatch.setattr(runner,'SOURCES',('source.py',))
    monkeypatch.chdir(tmp_path)
    source=tmp_path/'source.py';source.write_text('# frozen synthetic source\n')
    source_hashes={'source.py':runner.sha(source)}
    head,rows=fixtures()
    cache=dict(schema_version='backbone_readout_cache_v1',
        protocol=dict(mode='train',dataset_role='training_readout_calibration',context_frames=8,
            max_residues=96,seed=42,horizons=[1,4,16,64],n_windows=4,n_forecast_samples=0,
            optimizer_steps=0,test_opened=False,bioemu2_rerun=False,model_tensor_sha256='a'*64),rows=rows,
        input_hashes=source_hashes,source_hashes=source_hashes,
        completion=dict(completed_utc='synthetic',frozen_inputs_verified=True,
            model_sha256_before='a'*64,model_sha256_after='a'*64))
    cache_path=tmp_path/'cache.pt';torch.save(cache,cache_path)
    initial=tmp_path/'initial.pt'
    payload=head.checkpoint_dict();payload.update(step=0,protocol={'mode':'init'},
        input_hashes=source_hashes,source_hashes=source_hashes)
    torch.save(payload,initial)
    return cache_path,initial,head


def test_cache_refuses_validation_or_nonregistered_budget(tmp_path,monkeypatch):
    cache,_,_=synthetic_inputs(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='500'): runner.load_training_cache(cache,runner.Inputs())
    good=runner.load_training_cache(cache,runner.Inputs(),allow_smoke=True)
    assert len(good['rows'])==4
    altered=copy.deepcopy(good);altered['protocol']['mode']='val';torch.save(altered,cache)
    with pytest.raises(ValueError,match='training cache protocol'):
        runner.load_training_cache(cache,runner.Inputs(),allow_smoke=True)


@pytest.mark.parametrize('tamper',('coefficient','mean','raw','negative_norm','nonfinite_norm','row_order','budget'))
def test_gradient_scale_rejects_tampering(tamper):
    _,rows=fixtures()
    scale=dict(protocol={'n_calibration_rows':4},per_row=[dict(metadata=row['metadata'],
        coordinate_grad_norm=4.,weighted_geometry_grad_norm=2.) for row in rows],
        mean_coordinate_grad_norm=4.,mean_weighted_geometry_grad_norm=2.,unclipped_scale=1.,scale=1.)
    assert runner.validate_scale(scale,rows)==1.
    if tamper=='coefficient': scale['scale']=2.
    elif tamper=='mean': scale['mean_coordinate_grad_norm']=3.
    elif tamper=='raw': scale['unclipped_scale']=4.
    elif tamper=='negative_norm': scale['per_row'][0]['coordinate_grad_norm']=-1.
    elif tamper=='nonfinite_norm': scale['per_row'][0]['weighted_geometry_grad_norm']=float('nan')
    elif tamper=='row_order': scale['per_row'].reverse()
    else: scale['protocol']['n_calibration_rows']=32
    with pytest.raises(ValueError): runner.validate_scale(scale,rows)


def test_gradient_scale_clipping_is_fixed_not_validation_selected():
    assert runner.scale_from_norms(1.,100.)==(.005,.1)
    assert runner.scale_from_norms(100.,1.)==(50.,10.)
    for values in ((0.,1.),(1.,0.),(1.,float('inf'))):
        with pytest.raises(ValueError): runner.scale_from_norms(*values)


def test_cpu_calibrate_then_matched_four_step_latest_arms(tmp_path,monkeypatch):
    cache,initial,head=synthetic_inputs(tmp_path,monkeypatch)
    initial_hash=runner.sha(initial)
    scale=tmp_path/'scale.json'
    common=dict(cache=str(cache),initial=str(initial),device='cpu',allow_smoke=True,steps=4)
    runner.run(SimpleNamespace(**common,mode='calibrate',out=str(scale),scale=None,arm=None))
    calibrated=json.loads(scale.read_text())
    ratio=.5*calibrated['mean_coordinate_grad_norm']/calibrated['mean_weighted_geometry_grad_norm']
    assert calibrated['scale']==min(10.,max(.1,ratio))
    assert len(calibrated['per_row'])==4
    assert calibrated['head_tensor_sha256']==state_sha256(head)
    checkpoints={}
    for arm in ('coordinate','geometry'):
        output=tmp_path/arm
        args=SimpleNamespace(**common,mode='train',out=str(output),scale=str(scale),arm=arm)
        runner.run(args)
        ck=torch.load(output/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
        checkpoints[arm]=ck
        assert ck['step']==4 and ck['arm']==arm and ck['frozen_inputs_verified']
        assert ck['initial_head_tensor_sha256']==state_sha256(head)
        assert ck['parameter_delta_l2']>0 and ck['head_tensor_sha256']!=state_sha256(head)
        assert not (output/'checkpoint_best.pt').exists()
        metrics=[json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
        assert [r['step'] for r in metrics]==[1,2,3,4]
        assert [r['metadata']['horizon'] for r in metrics]==[1,4,16,64]
        for name,p in head.named_parameters():
            if not p.requires_grad: assert torch.equal(ck['head'][name],head.state_dict()[name])
        with pytest.raises(ValueError,match='existing training'): runner.run(args)
    assert checkpoints['coordinate']['protocol']['geometry_lambda']==0.
    assert checkpoints['geometry']['protocol']['geometry_lambda']==calibrated['scale']
    assert runner.sha(initial)==initial_hash
    assert checkpoints['coordinate']['head_tensor_sha256']!=checkpoints['geometry']['head_tensor_sha256']
