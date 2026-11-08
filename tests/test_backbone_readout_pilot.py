"""Synthetic CPU evaluation contracts; no data split or GPU access."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from protein_jepa_frame.models.backbone_readout_v2 import BackboneReadoutConfig, BackboneReadoutV2
from protein_jepa_frame.utils.contracts import state_sha256
from scripts import eval_backbone_readout_pilot as evaluator


def fixture(horizon=1, length=5):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(222)
        head = BackboneReadoutV2(BackboneReadoutConfig(raw_h_dim=128, c_hidden=16,
            n_heads=4, n_ipa_blocks=1, torsion_bins=12, dropout=0)).eval()
    heads = {name: copy.deepcopy(head) for name in evaluator.HEADS}
    g = torch.Generator().manual_seed(44)
    bb = torch.randn(1, length, 4, 3, generator=g)
    bb += torch.arange(length)[None, :, None, None] * torch.tensor([3., .4, .1])
    H = torch.randn(1, length, 128, generator=g)
    seed = int(hashlib.sha256(f'p|p_T320_R0|7|{horizon}'.encode()).hexdigest()[:12],16)%(2**31-1)
    row = dict(metadata=dict(protein='p', trajectory='p_T320_R0', temperature_k=320.,
        anchor_frame=7, target_frame=7+horizon, horizon=horizon, sampling_seed=seed),
        observed_mask=torch.ones(length, dtype=torch.bool),
        embedding_aatype=torch.arange(length) % 21,
        embedding_residue_index=torch.arange(length), embedding_chain_id=torch.zeros(length, dtype=torch.long),
        chemical_aatype=torch.zeros(length, dtype=torch.long), geometry_row_ids=torch.arange(20, 20+length),
        peptide_edges=torch.stack([torch.arange(length-1), torch.arange(1,length)], -1),
        atom_mask=torch.ones(length,4,dtype=torch.bool), target_atom_mask=torch.ones(length,4,dtype=torch.bool),
        observed_R=torch.eye(3).expand(1,length,3,3).clone(), observed_backbone=bb,
        target_backbone=bb + .1*torch.randn(bb.shape,generator=g), anchor_H=H,
        oracle_H=H+.2*torch.randn(H.shape,generator=g), oracle_CA=bb[:,:,1].clone(),
        forecast_H=H.expand(12,-1,-1).clone()+.1*torch.randn(12,length,128,generator=g),
        forecast_CA=bb[:,:,1].expand(12,-1,-1).clone()+.2*torch.randn(12,length,3,generator=g),
        boundary=dict(forecast_before_target_read=True,future_reencoding_for_forecast=0,
                      observed_frames=list(range(8)),label_frames=[7+horizon]))
    return heads,row


def test_all_head_routes_keep_CA_complete_samples_and_no_parameter_gradients():
    heads,row = fixture()
    before = {k:state_sha256(v) for k,v in heads.items()}
    rng = torch.random.get_rng_state().clone()
    result = evaluator.score_row(heads,row)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert len(result['routes']) == 14
    for name,route in result['routes'].items():
        n = 12 if name.endswith('/forecast') else 1
        assert route['n_samples'] == n
        assert route['error']['n_samples'] == route['legacy_geometry']['n_samples'] == n
        assert len(route['explicit_geometry']['per_sample']['c_n']) == n
        if '/' in name: assert route['fixed_ca_bitwise_equal']
    assert result['routes']['target_MD']['error']['mean']['coordinate_rmsd_angstrom'] == 0
    assert result['topology']['edge_sets_equal']
    assert all(state_sha256(head)==before[name] for name,head in heads.items())
    assert all(p.grad is None for head in heads.values() for p in head.parameters())
    for variant in ('oracle','oracle_H_anchor','oracle_H_zero','oracle_H_permuted'):
        assert result['routes'][f'init/{variant}']['fixed_ca_sha256'] == result['routes']['geometry/oracle']['fixed_ca_sha256']


def test_target_coordinates_and_target_masks_never_change_decoded_atoms():
    heads,row = fixture()
    first = evaluator.score_row(heads,row)
    changed = copy.deepcopy(row)
    changed['target_backbone'] += 3.
    changed['target_atom_mask'][-1,3] = False
    second = evaluator.score_row(heads,changed)
    for name in first['routes']:
        if '/' in name:
            assert first['routes'][name]['backbone_sha256'] == second['routes'][name]['backbone_sha256']
    assert second['scoring']['shared_scored_atom_counts'] == [5,5,5,4]
    assert first['routes']['init/forecast']['error']['mean'] != second['routes']['init/forecast']['error']['mean']


def test_insertion_codes_and_cross_chain_edges_are_reported_as_legacy_mismatch():
    heads,row = fixture()
    row['embedding_residue_index'] = torch.tensor([1,1,1,2,42])
    row['embedding_chain_id'][-1] = 1
    result = evaluator.score_row(heads,row)
    topology = result['topology']
    assert topology['legacy_v2_inferred_edges'] == [(2,3)]
    assert topology['explicit_only'] == [(0,1),(1,2),(3,4)]
    assert not topology['legacy_connectivity_is_physical']
    for route in result['routes'].values():
        assert route['explicit_geometry']['counts']['n_connection_candidates'] == [4]*route['n_samples']


def test_padding_nan_is_ignored_and_H_permutation_is_only_over_observed_rows():
    heads,row = fixture()
    row['observed_mask'][-1] = False
    row['atom_mask'][-1] = row['target_atom_mask'][-1] = False
    for key in ('anchor_H','oracle_H','forecast_H','oracle_CA','forecast_CA','observed_R','observed_backbone','target_backbone'):
        row[key][:,-1] = torch.nan
    result = evaluator.score_row(heads,row)
    assert result['H_control_permutation']['valid_rows'] == [0,1,2,3]
    assert sorted(result['H_control_permutation']['source_rows']) == [0,1,2,3]
    assert result['scoring']['shared_scored_atom_counts'] == [4]*4
    assert all(r['fixed_ca_bitwise_equal'] for name,r in result['routes'].items() if '/' in name)


def test_train_mode_and_changed_fixed_CA_are_rejected():
    heads,row = fixture()
    heads['coordinate'].dropout.train()
    with pytest.raises(ValueError,match='strict eval'):
        evaluator.score_row(heads,row)
    heads['coordinate'].eval()
    original = heads['geometry'].forward
    def bad(*args):
        result = original(*args)
        result['backbone'] = result['backbone'].clone()
        result['backbone'][...,1,:] += .01
        return result
    heads['geometry'].forward = bad
    with pytest.raises(ValueError,match='fixed CA'):
        evaluator.score_row(heads,row)


def test_malformed_cache_row_fails_closed():
    _,row = fixture()
    for key,value in (('peptide_edges',torch.tensor([[0,1],[0,1]])),
                      ('geometry_row_ids',torch.zeros(5,dtype=torch.long)),
                      ('forecast_H',row['forecast_H'][:1]),
                      ('oracle_CA',row['oracle_CA'].double())):
        with pytest.raises(ValueError): evaluator.validate_row({**row,key:value})


def test_equal_sample_window_protein_summary_does_not_pool_atoms():
    def row(protein,value):
        return dict(protein=protein,horizon=1,routes={'init/oracle':dict(
            error={'mean':{'coordinate_rmsd_angstrom':value}}, legacy_geometry={'mean':{}},
            explicit_geometry={'losses':{'c_n':value},'counts':{'n_c_n':[1]}})})
    summary = evaluator.summarize([row('p',1.),row('p',3.),row('q',10.)])
    assert summary['1']['means']['init/oracle']['error']['coordinate_rmsd_angstrom'] == dict(mean=6.,n_proteins=2,n_valid_windows=3)
    assert summary['1']['means']['init/oracle']['explicit_geometry']['c_n']['mean'] == 6.


def test_empty_explicit_topology_keeps_raw_zero_but_is_not_good_aggregate_geometry():
    heads,row = fixture()
    row['peptide_edges'] = torch.empty(0,2,dtype=torch.long)
    result = evaluator.score_row(heads,row)
    assert result['routes']['init/oracle']['explicit_geometry']['losses']['c_n'] == 0.
    summary = evaluator.summarize([result])
    assert summary['1']['means']['init/oracle']['explicit_geometry']['c_n'] == dict(
        mean=None,n_proteins=0,n_valid_windows=0)
    assert summary['1']['means']['init/oracle']['explicit_geometry']['backbone_interresidue_clash']['n_proteins'] == 1


def test_panel_requires_shared_anchor_complete_horizons_and_unique_keys():
    rows = [fixture(h)[1] for h in (1,4,16,64)]
    evaluator.validate_panel(rows,expected_windows=4,expected_proteins=1)
    bad = copy.deepcopy(rows);bad[-1]['metadata']['trajectory']='other'
    with pytest.raises(ValueError,match='share observed'): evaluator.validate_panel(bad,4,1)
    with pytest.raises(ValueError): evaluator.validate_panel(rows[:-1],4,1)
    with pytest.raises(ValueError): evaluator.validate_panel(rows[:3]+rows[:1],4,1)


def cache_protocol(n=4):
    return dict(mode='val',dataset_role='development_validation',forecast_observed_only=True,
        before_target_read=True,n_forecast_samples=12,horizons=[1,4,16,64],n_windows=n,
        max_residues=96,context_frames=8,seed=0)


def completion():
    return dict(completed_utc='synthetic',frozen_inputs_verified=True,
                model_sha256_before='a'*64,model_sha256_after='a'*64)


def test_cache_protocol_fails_closed_for_training_or_future_conditioned_forecast():
    cache=dict(schema_version='backbone_readout_cache_v1',protocol=cache_protocol(),completion=completion(),
               rows=[fixture(h)[1] for h in (1,4,16,64)])
    evaluator.validate_cache(cache,allow_smoke=True)
    with pytest.raises(ValueError,match='smoke'): evaluator.validate_cache(cache)
    for key,value in (('mode','train'),('forecast_observed_only',False),('before_target_read',False),
                      ('n_forecast_samples',1),('horizons',[1,4,16]),('max_residues',192)):
        wrong=copy.deepcopy(cache);wrong['protocol'][key]=value
        with pytest.raises(ValueError): evaluator.validate_cache(wrong,allow_smoke=True)


def test_cpu_smoke_roundtrip_hashes_snapshots_and_no_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(evaluator,'REPO',tmp_path)
    monkeypatch.setattr(evaluator,'SOURCES',('fake_source.py',))
    source=tmp_path/'fake_source.py';source.write_text('# synthetic provenance source\n')
    dependency=tmp_path/'input.txt';dependency.write_text('synthetic frozen input\n')
    cache_path=tmp_path/'cache.pt'
    cache=dict(schema_version='backbone_readout_cache_v1',protocol=cache_protocol(),completion=completion(),
        rows=[fixture(h)[1] for h in (1,4,16,64)],
        source_hashes={str(source):evaluator.sha(source)},input_hashes={str(dependency):evaluator.sha(dependency)})
    torch.save(cache,cache_path)
    head=fixture()[0]['init'];paths={name:tmp_path/f'{name}.pt' for name in evaluator.HEADS}
    initial_hash=state_sha256(head)
    base_protocol=dict(schema_version='fixed_ca_backbone_readout_training_v1',seed=42)
    payload=head.checkpoint_dict();payload.update(step=0,protocol=dict(base_protocol,mode='init',optimizer_steps=0),
        source_hashes=cache['source_hashes'],input_hashes={str(cache_path):evaluator.sha(cache_path)},
        frozen_inputs_verified=True,head_tensor_sha256=initial_hash)
    torch.save(payload,paths['init'])
    weights=dict(c_n=1.,ca_c_n=4.,c_n_ca=4.,backbone_interresidue_clash=100.)
    scale_path=tmp_path/'scale.json'
    scale=dict(frozen_inputs_verified=True,head_tensor_sha256=initial_hash,
        protocol=dict(mode='calibrate',train_cache=str(cache_path),initial=str(paths['init']),geometry_weights=weights,n_calibration_rows=4),
        per_row=[dict(coordinate_grad_norm=4.,weighted_geometry_grad_norm=2.) for _ in range(4)],
        mean_coordinate_grad_norm=4.,mean_weighted_geometry_grad_norm=2.,unclipped_scale=1.,scale=1.)
    scale_path.write_text(json.dumps(scale))
    for name in ('coordinate','geometry'):
        payload=copy.deepcopy(head.checkpoint_dict())
        payload['head']['psi_head.2.bias']+=.01 if name=='coordinate' else .02
        payload.update(step=4,arm=name,protocol=dict(base_protocol,mode='train',arm=name,steps=4,optimizer_steps=4,
            formal=False,checkpoint_selection='fixed latest; no validation selection',validation_used_for_training=False,
            encoder_CA_dynamics_frozen=True,geometry_weights=weights,geometry_lambda=0. if name=='coordinate' else 1.,
            train_cache=str(cache_path),initial=str(paths['init']),scale_file=str(scale_path),scale_sha256=evaluator.sha(scale_path)),
            source_hashes=cache['source_hashes'],input_hashes={str(path):evaluator.sha(path) for path in (cache_path,paths['init'],scale_path)},
            frozen_inputs_verified=True,head_tensor_sha256=state_sha256(payload['head']),
            initial_head_tensor_sha256=initial_hash,parameter_delta_l2=.01)
        torch.save(payload,paths[name])
    output=tmp_path/'evaluation.json'
    report=evaluator.evaluate(cache_path,paths,output,allow_smoke=True)
    assert report['n_windows']==4 and report['n_proteins']==1 and not report['formal_panel']
    assert report['frozen_inputs_verified'] and report['target_backbone_or_target_mask_given_to_head'] is False
    assert report['oracle_inputs_are_future_derived'] is True
    assert report['source_hashes']['fake_source.py']==evaluator.sha(output.with_suffix('.source_snapshot')/'fake_source.py')
    before=output.read_bytes()
    with pytest.raises(ValueError,match='overwrite'): evaluator.evaluate(cache_path,paths,output,allow_smoke=True)
    assert output.read_bytes()==before
    source.write_text('# changed source\n')
    with pytest.raises(ValueError,match='hash mismatch'):
        evaluator.evaluate(cache_path,paths,tmp_path/'must_fail.json',allow_smoke=True)
    assert not (tmp_path/'must_fail.json').exists()


def test_existing_output_paths_are_never_overwritten(tmp_path,monkeypatch):
    monkeypatch.setattr(evaluator,'REPO',tmp_path)
    output = tmp_path/'eval.json'
    for path in (output,output.with_suffix('.protocol.json'),output.with_suffix('.source_snapshot')):
        path.write_text('keep')
        with pytest.raises(ValueError,match='overwrite'):
            evaluator.evaluate(tmp_path/'missing_cache.pt',{},output)
        assert path.read_text() == 'keep'
        path.unlink()
    with pytest.raises(ValueError,match='excluded'): evaluator.local(tmp_path/'test.jsonl')


def test_real_topology_subset_contract_connects_to_v3_loss():
    # The topology object, not legacy residue numbers, remaps a reordered crop.
    import runpy
    from protein_jepa_frame.models.stage_c.backbone_objectives_v3 import backbone_geometry_losses_v3
    parse = runpy.run_path(str(Path(__file__).with_name('test_backbone_topology_v2.py')))['parse']
    topology = parse().subset(torch.tensor([3,1,2]))
    assert topology.geometry_row_ids.tolist() == [3,1,2]
    assert topology.embedding_residue_index.tolist() == [2,1,1]
    assert topology.peptide_edges.tolist() == [[1,2],[2,0]]
    coordinates = torch.randn(1,3,4,3,generator=torch.Generator().manual_seed(72),requires_grad=True)
    result = backbone_geometry_losses_v3(coordinates,chemical_aatype=topology.chemical_aatype,
        atom_mask=topology.atom_mask,geometry_row_ids=topology.geometry_row_ids,peptide_edges=topology.peptide_edges)
    assert result['counts']['n_connection_candidates'].tolist() == [2]
    sum(result['losses'].values()).backward()
    assert torch.isfinite(coordinates.grad).all() and coordinates.grad.abs().sum()>0
