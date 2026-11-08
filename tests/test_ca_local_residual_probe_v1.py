"""Synthetic CPU contracts. No formal cache or trajectory is read here."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from run_scripts import run_ca_local_residual_probe_v1 as p


@pytest.fixture(autouse=True)
def cpu_threads():
    prior=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(prior)


def row(horizon=1,length=6):
    generator=torch.Generator().manual_seed(19)
    anchor=torch.stack((torch.arange(length)*3.8,torch.sin(torch.arange(length).float()),
                        torch.cos(torch.arange(length).float())),1)
    bb=anchor[None,:,None,:].expand(1,length,4,3).clone()
    target=bb.clone();target[0,2,1,1]+=.7
    H=torch.randn(1,length,128,generator=generator)
    return dict(metadata=dict(protein='p',trajectory='p_T320_R0',temperature_k=320.,
        anchor_frame=7,horizon=horizon,target_frame=7+horizon,horizon_ps=horizon*1000.,
        context_frames=list(range(8)),sampling_seed=0),
        observed_mask=torch.ones(length,dtype=torch.bool),observed_R=torch.eye(3)[None,None].repeat(1,length,1,1),
        observed_backbone=bb,target_backbone=target,anchor_H=H,
        oracle_H=H+torch.randn(1,length,128,generator=generator)*.2,
        forecast_H=H.expand(12,-1,-1)+torch.randn(12,length,128,generator=generator)*.2,
        embedding_aatype=torch.arange(length)%20,embedding_residue_index=torch.arange(length),
        embedding_chain_id=torch.zeros(length,dtype=torch.long),chemical_aatype=torch.arange(length)%20,
        geometry_row_ids=torch.arange(length),peptide_edges=torch.stack((torch.arange(length-1),torch.arange(1,length)),1),
        atom_mask=torch.ones(length,4,dtype=torch.bool),target_atom_mask=torch.ones(length,4,dtype=torch.bool),
        topology={'sha256':'synthetic'},fingerprints={})


def state():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(32)
        return p.GeometricDeltaCADecoder(**p.CONFIG).state_dict()


def test_three_zero_update_models_match_raw_projected_and_do_not_change_rng():
    old=torch.get_rng_state().clone()
    models,proof=p.zero_update_check(state(),[row()])
    assert torch.equal(old,torch.get_rng_state())
    assert proof['all_three_raw_and_projected_bitwise_equal']
    assert set(models)==set(p.ROUTES)
    assert models['local_residual'].local_residual.weight.numel()==384
    for stage in p.STAGES:
        assert len({proof['rows'][0]['route_output_sha256'][arm][stage] for arm in p.ROUTES})==1


@pytest.mark.parametrize('arm',p.ARMS)
def test_actual_projected_mse_has_finite_decoder_gradients_and_detaches_cache(arm):
    r=row();r['oracle_H'].requires_grad_();r['target_backbone'].requires_grad_()
    model=p.make_decoder(state(),arm)
    before=p.state_sha256(model);cached=p.row_fingerprint(r)
    loss,norm=p.loss_backward(model,r)
    assert loss.isfinite() and norm.isfinite() and norm>0
    assert r['oracle_H'].grad is None and r['target_backbone'].grad is None
    assert all(x.grad is not None and x.grad.isfinite().all() for x in model.parameters())
    if arm=='local_residual':assert model.local_residual.weight.grad.abs().sum()>0
    assert p.state_sha256(model)==before and p.row_fingerprint(r)==cached


@pytest.mark.parametrize('arm',p.ROUTES)
def test_future_labels_and_chemical_metadata_never_enter_decode(arm):
    r=row();model=p.make_decoder(state(),arm).eval()
    a=p.decode(model,r,r['oracle_H'])
    changed=deepcopy(r)
    changed['target_backbone'].fill_(float('nan'));changed['target_atom_mask'].fill_(False)
    changed['chemical_aatype'].fill_(20);changed['peptide_edges']=torch.empty(0,2,dtype=torch.long)
    b=p.decode(model,changed,r['oracle_H'])
    assert all(torch.equal(a[s],b[s]) for s in p.STAGES)


def test_masked_mse_denominator_and_member_rmsd_order():
    r=row();m=r['observed_mask'];target=r['target_backbone'][0,:,1,:].double()
    samples=torch.stack((target+1.,target+2.))
    assert float(p.masked_mse(samples[:1],target[None],m[None]))==pytest.approx(1.,abs=1e-12)
    scored=p.score_leaf(samples,r,p.topology_support(r))
    assert scored['scalars']['coordinate_mse']==pytest.approx(2.5)
    assert scored['scalars']['ca_rmsd']==pytest.approx(1.5*np.sqrt(3))
    assert scored['scalars']['ca_rmsd']!=pytest.approx(np.sqrt(3*2.5))
    for member in scored['per_member']:
        assert member['ca_rmsd']==pytest.approx(np.sqrt(3*member['coordinate_mse']))


def test_zero_persistence_denominator_is_null():
    r=row();r['target_backbone']=r['observed_backbone'].clone()
    scored=p.score_leaf(r['target_backbone'][:,:,1,:],r,p.topology_support(r))
    assert scored['scalars']['ca_rmsd']==0
    assert scored['scalars']['rmsd_to_persistence_ratio'] is None


def test_psf_graph_preserves_masked_intermediate_two_hop_and_cross_chain_bonds():
    r=row(length=6);r['observed_mask'][1]=False
    r['embedding_chain_id']=torch.arange(6)
    bonds,pairs,evidence=p.topology_support(r)
    assert [0,2] not in pairs.tolist()  # 0-1-2 exists through masked residue1.
    assert [0,3] in pairs.tolist()     # Three hops remains nonlocal.
    assert [2,3] in bonds.tolist()     # Actual edge survives legacy chain mismatch.
    assert evidence['n_legacy_edges']==0 and evidence['n_psf_edges']==3


def test_psf_gap_is_nonlocal_but_does_not_change_projection():
    r=row();model=p.make_decoder(state(),'base_tuned')
    first=p.decode(model,r,r['oracle_H'])
    r['peptide_edges']=torch.tensor([[0,1],[1,2],[3,4],[4,5]])
    _,pairs,evidence=p.topology_support(r)
    assert [2,3] in pairs.tolist() and not evidence['legacy_psf_equal']
    second=p.decode(model,r,r['oracle_H'])
    assert all(torch.equal(first[s],second[s]) for s in p.STAGES)


@pytest.mark.parametrize('edges',[[[0,0]],[[0,6]],[[0,1],[0,1]]])
def test_illegal_edges_fail(edges):
    r=row();r['peptide_edges']=torch.tensor(edges)
    with pytest.raises(ValueError):p.topology_support(r)


def test_empty_bond_support_is_null_not_zero_and_counts_retained():
    r=row();r['peptide_edges']=torch.empty(0,2,dtype=torch.long)
    scored=p.score_leaf(r['target_backbone'][:,:,1,:],r,p.topology_support(r))
    assert scored['scalars']['psf_ca_bond_mean_angstrom'] is None
    assert scored['scalars']['psf_ca_bond_invalid_fraction'] is None
    assert scored['geometry_counts']['n_psf_edges']==0


def test_wrong_horizon_cycle_changes_only_H_and_requires_exact_observed_pairing():
    rows=[row(h) for h in p.HORIZONS]
    donors=p.wrong_horizon_donors(rows)
    assert donors=={0:1,1:2,2:3,3:0}
    rows[1]['observed_R'][0,0,0,0]+=1e-6
    with pytest.raises(ValueError,match='observed/topology'):p.wrong_horizon_donors(rows)


def test_wrong_horizon_rejects_missing_or_duplicate_key():
    rows=[row(h) for h in p.HORIZONS]
    with pytest.raises(ValueError,match='complete'):p.wrong_horizon_donors(rows[:3])
    with pytest.raises(ValueError,match='unique'):p.wrong_horizon_donors(rows+[row()])


def test_coordinate_energy_score_keeps_cached_gauge_without_second_alignment(monkeypatch):
    r=row();y=r['target_backbone'][0,:,1,:].double()
    x=(y+torch.tensor([3.,0.,0.]))[None].expand(12,-1,-1).clone()
    # A refit would erase this translation. The cached gauge must retain it.
    result=p.endpoint_ensemble_scores(x,y,r)
    assert result['coordinate_energy_score']==pytest.approx(np.sqrt(3),rel=1e-6)
    assert result['physical_distance_energy_score']==pytest.approx(0,abs=1e-12)
    assert result['contact_brier']==0


def test_ensemble_U_and_distance_V_match_direct_small_pair_sums():
    r=row();g=torch.Generator().manual_seed(1)
    target=r['target_backbone'][0,:,1,:].double()
    samples=target[None]+torch.randn(12,len(target),3,generator=g,dtype=torch.float64)*.4
    actual=p.endpoint_ensemble_scores(samples,target,r)
    flat=samples.flatten(1).numpy();truth=target.flatten().numpy()
    a=np.linalg.norm(flat-truth,axis=1).mean()/np.sqrt(flat.shape[1])
    b=np.mean([np.linalg.norm(flat[i]-flat[j]) for i in range(12) for j in range(i+1,12)])/np.sqrt(flat.shape[1])
    assert actual['coordinate_energy_score']==pytest.approx(a-.5*b,abs=1e-12)
    ii,jj=torch.triu_indices(len(target),len(target),offset=3)
    dist=(samples[:,ii]-samples[:,jj]).norm(dim=-1).numpy()
    real=(target[ii]-target[jj]).norm(dim=-1).numpy()
    expected=np.linalg.norm(dist-real,axis=1).mean()/np.sqrt(len(ii))-.5*np.mean(
        [[np.linalg.norm(a-b) for b in dist] for a in dist])/np.sqrt(len(ii))
    assert actual['physical_distance_energy_score']==pytest.approx(expected,abs=1e-10)


def test_learning_rate_has500_calls_but499_nonzero():
    values=[p.learning_rate(i) for i in range(1,501)]
    assert values[49]==1e-4 and values[-1]==0
    assert sum(v>0 for v in values)==499
    with pytest.raises(ValueError):p.learning_rate(0)


def test_exclusive_writes_reject_overwrite_and_nonfinite(monkeypatch,tmp_path):
    monkeypatch.setattr(p,'local',lambda x:tmp_path/str(x))
    p.write_new('one.json',{'x':1})
    with pytest.raises(FileExistsError):p.write_new('one.json',{'x':2})
    assert json.loads((tmp_path/'one.json').read_text())=={'x':1}
    with pytest.raises(ValueError):p.write_new('bad.json',{'x':float('nan')})
    assert not (tmp_path/'bad.json').exists()


def test_no_encoder_kernel_or_forecast_sampling_constructor_in_new_entry():
    import ast,inspect
    tree=ast.parse(inspect.getsource(p))
    forbidden={'CADynamics','StageB1b','CAComponentBundle','Calibration','TrajectoryClipDataset'}
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in forbidden for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('sample','rollout','_read_clip') for n in ast.walk(tree))
    assert not torch.cuda.is_initialized()


def test_phase_failure_is_preserved_with_actual_pid_and_cannot_restart(monkeypatch,tmp_path):
    import os
    from pathlib import Path
    monkeypatch.setattr(p,'RESULTS',Path('result'))
    monkeypatch.setattr(p,'local',lambda x:tmp_path/Path(x))
    monkeypatch.setattr(p,'sha',lambda x:'f'*64)
    p.write_new(p.RESULTS/'protocol.json',{'schema_version':p.VERSION,'hyperparameters':p.HYPER})
    monkeypatch.setattr(p,'reseed',lambda:None)
    def reject(*args):raise ValueError('frozen source changed')
    monkeypatch.setattr(p,'verify',reject)
    monkeypatch.setattr(p,'smoke',lambda *a:pytest.fail('sampling/training must not start after hash failure'))
    with pytest.raises(ValueError,match='frozen source changed'):p.execute('smoke')
    evidence=json.loads((tmp_path/'result/smoke.execution.json').read_text())
    assert evidence['exit_code']==1 and evidence['status']=='failed' and evidence['pid']==os.getpid()
    assert 'frozen source changed' in evidence['error']
    old=(tmp_path/'result/smoke.execution.json').read_bytes()
    with pytest.raises(FileExistsError):p.execute('smoke')
    assert (tmp_path/'result/smoke.execution.json').read_bytes()==old
