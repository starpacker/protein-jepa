"""Synthetic statistical/provenance tests; no real evaluation is opened."""
import copy
import hashlib
import math

import numpy as np
import pytest

from scripts import summarize_backbone_readout_pilot as summary


def route(nco=1., *, n=1, geometry=None, support=None, atom_samples=None):
    errors = []
    if atom_samples is None:
        atom_samples = [([1,1,1], [nco,nco,nco]) for _ in range(n)]
    for counts, rms in atom_samples:
        sample = {}
        for atom, count, value in zip(('N','C','O'), counts, rms):
            sample['n_atoms_'+atom] = count
            sample['coordinate_rmsd_'+atom+'_angstrom'] = value if count else None
        sample.update(coordinate_rmsd_angstrom=nco, shape_kabsch_rmsd_angstrom=0.,
                      n_atoms=sum(counts), n_atoms_CA=0, coordinate_rmsd_CA_angstrom=None)
        errors.append(sample)
    values = {metric:list((geometry or {}).get(metric,[0.]*n)) for metric in summary.GEOMETRY}
    counts = {summary.count_key(metric):list((support or {}).get(metric,[1]*n)) for metric in summary.GEOMETRY}
    losses = {}
    for metric in summary.GEOMETRY:
        valid = [v for v,c in zip(values[metric],counts[summary.count_key(metric)]) if c]
        losses[metric] = sum(valid)/len(valid) if valid else 0.
    return dict(n_samples=n, error=dict(n_samples=n,schema_version='backbone_metrics_v2',per_sample=errors,
        mean=dict(coordinate_rmsd_angstrom=nco,shape_kabsch_rmsd_angstrom=0.)),
        legacy_geometry=dict(n_samples=n,schema_version='backbone_metrics_v2',mean={'example':2.}),
        explicit_geometry=dict(schema_version='backbone_objectives_v3.0_explicit_peptide_edges',
            per_sample=values,counts=counts,losses=losses,
            conventions=dict(connection_tolerance_factor=12.,clash_overlap_angstrom=1.5,arithmetic='torch.float64')),
        fixed_ca_bitwise_equal=True,fixed_ca_sha256='same')


def row(protein, candidate, reference, *, horizon=1):
    return dict(protein=protein,horizon=horizon,routes={'a':candidate,'b':reference})


def test_nco_is_count_weighted_within_sample_then_equal_sample_mean():
    item=route(n=2,atom_samples=[([1,2,3],[1.,2.,3.]),([10,1,1],[5.,0.,4.])])
    expected=(math.sqrt(36/6)+math.sqrt(266/12))/2
    actual=summary.primary(item)['coordinate_rmsd_NCO_angstrom']
    assert actual==pytest.approx(expected,abs=1e-14)
    assert abs(actual-math.sqrt((36+266)/(6+12)))>.1
    assert abs(actual-((1+2+3+5+0+4)/6))>.1


def test_zero_support_is_null_and_empty_members_do_not_lower_valid_mean():
    empty=route(atom_samples=[([0,0,0],[None,None,None])],
        support={metric:[0] for metric in summary.GEOMETRY})
    assert all(value is None for value in summary.primary(empty).values())
    mixed=route(n=2,atom_samples=[([0,0,0],[None,None,None]),([1,1,1],[4.,4.,4.])],
        support={metric:[0,2] for metric in summary.GEOMETRY},
        geometry={metric:[0.,6.] for metric in summary.GEOMETRY})
    assert summary.primary(mixed)==dict(coordinate_rmsd_NCO_angstrom=4.,
        c_n=6.,ca_c_n=6.,c_n_ca=6.,backbone_interresidue_clash=6.)
    mixed['explicit_geometry']['per_sample']['c_n'][0]=2.
    with pytest.raises(ValueError,match='empty'):summary.primary(mixed)


def test_pairing_support_and_equal_protein_weighting_with_unequal_window_counts():
    metric='c_n'
    geom=lambda value:route(geometry={metric:[0. if value is None else value]},support={metric:[0 if value is None else 1]})
    rows=[row('A',geom(None),geom(100.)),row('A',geom(8.),geom(4.)),
          row('B',geom(100.),geom(None)),row('C',geom(3.),geom(1.)),row('C',geom(9.),geom(3.))]
    result=summary.paired(rows,'a','b',metric)
    assert result['n_proteins']==2 and result['n_paired_windows']==3
    assert result['candidate_mean']==7. and result['reference_mean']==3. and result['delta']==4.
    assert result['ci95']==[4.,4.]
    assert [entry['protein'] for entry in result['per_protein']]==['A','C']
    assert [entry['n_paired_windows'] for entry in result['per_protein']]==[1,2]
    empty=summary.paired([row('A',geom(None),geom(2.))],'a','b',metric)
    assert empty['n_proteins']==empty['n_paired_windows']==0 and empty['per_protein']==[]
    assert all(empty[key] is None for key in ('candidate_mean','reference_mean','delta','ci95'))


def test_10000_paired_bootstrap_reproduces_independent_sorted_protein_draws():
    rows=[row('z',route(30.),route(10.)),row('a',route(0.),route(10.)),
          row('m',route(11.),route(10.)),row('b',route(10.),route(10.)),
          row('z',route(30.),route(10.))]
    actual=summary.paired(rows,'a','b','coordinate_rmsd_NCO_angstrom')
    # Construct the sorted per-protein contrasts independently of the tested
    # reducer, and draw actual protein values rather than calling its bootstrap.
    contrasts=np.array([-10.,0.,1.,20.],dtype=np.float64)
    random=np.random.default_rng(20261101)
    replicates=random.choice(contrasts,size=(10000,4),replace=True).mean(axis=1)
    expected=np.percentile(replicates,[2.5,97.5])
    assert actual['delta']==pytest.approx(2.75)
    assert actual['ci95']==pytest.approx(expected,abs=1e-14)
    assert actual==summary.paired(list(reversed(rows)),'a','b','coordinate_rmsd_NCO_angstrom')
    assert [p['protein'] for p in actual['per_protein']]==['a','b','m','z']


def complete_row(index,horizon):
    routes={}
    for name in summary.ROUTES:
        n=12 if name.endswith('/forecast') else 1
        routes[name]=route(n=n)
        routes[name]['fixed_ca_sha256']='forecastCA' if n==12 else 'oracleCA'
    return dict(protein=f'p{index:02d}',trajectory=f't{index:02d}',temperature_k=(320.,348.,379.,413.,450.)[index%5],
        anchor_frame=7,horizon=horizon,target_frame=7+horizon,routes=routes,
        scoring=dict(target_mask_used_only_for_scoring=True,arithmetic='CPU float64'),
        cache_boundary=dict(forecast_before_target_read=True,future_reencoding_for_forecast=0))


def report():
    return dict(schema_version=summary.EVALUATOR_VERSION,formal_panel=True,
        dataset_role='development_validation',n_windows=200,n_proteins=50,
        frozen_inputs_verified=True,optimizer_steps=0,test_opened=False,
        target_backbone_or_target_mask_given_to_head=False,oracle_inputs_are_future_derived=True,
        cache_protocol=dict(mode='val',dataset_role='development_validation',n_windows=200,n_forecast_samples=12,
            seed=0,horizons=[1,4,16,64],context_frames=8,max_residues=96,forecast_observed_only=True,before_target_read=True),
        per_window=[complete_row(i,h) for i in range(50) for h in (1,4,16,64)])


def test_analysis_has_all_14_routes_and_exactly_300_registered_intervals():
    rows=[complete_row(0,h) for h in (1,4,16,64)]
    result=summary.analysis(rows)
    assert set(result['descriptive'])==set(summary.GROUPS)
    assert all(set(routes)==set(summary.ROUTES) and len(routes)==14 for routes in result['descriptive'].values())
    count=sum(len(metrics) for category in ('main_contrasts','H_controls')
              for groups in result[category].values() for metrics in groups.values())
    assert count==300
    assert result['main_contrasts']['overall']['geometry-coordinate/forecast']['coordinate_rmsd_NCO_angstrom']['n_paired_windows']==4


def test_formal_report_accepts_only_complete_14_route_12_member_panel():
    good=report()
    assert len(summary.validate_report(good))==200
    mutations=[lambda r:r.update(formal_panel=False),lambda r:r.update(n_windows=4),
        lambda r:r['per_window'][0]['routes'].pop('geometry/oracle_H_zero'),
        lambda r:r['per_window'][0]['routes']['init/forecast'].update(n_samples=1),
        lambda r:r['per_window'][0]['routes']['coordinate/oracle'].update(fixed_ca_sha256='changed'),
        lambda r:r['per_window'][0]['routes']['init/oracle'].update(fixed_ca_bitwise_equal=False),
        lambda r:r['per_window'].__setitem__(1,copy.deepcopy(r['per_window'][0])),
        lambda r:r['per_window'][0]['cache_boundary'].update(forecast_before_target_read=False)]
    for mutate in mutations:
        bad=copy.deepcopy(good);mutate(bad)
        with pytest.raises(ValueError):summary.validate_report(bad)


def scale_fixture():
    protocol=dict(schema_version='fixed_ca_backbone_readout_training_v1',mode='calibrate',formal=True,seed=42,
        optimizer='AdamW',learning_rate=1e-4,final_learning_rate=1e-5,warmup_steps=25,weight_decay=1e-4,
        gradient_clip=1.,geometry_weights=summary.WEIGHTS,checkpoint_selection='fixed latest; no validation selection',
        encoder_CA_dynamics_frozen=True,validation_used_for_training=False,test_opened=False,bioemu2_rerun=False,
        n_calibration_rows=32,optimizer_steps=0)
    training=[dict(metadata={'index':i}) for i in range(40)]
    rows=[dict(metadata=training[i]['metadata'],coordinate_grad_norm=float(i+1),
               weighted_geometry_grad_norm=1. if i<31 else 100.) for i in range(32)]
    coord=sum(row['coordinate_grad_norm'] for row in rows)/32
    geom=sum(row['weighted_geometry_grad_norm'] for row in rows)/32
    raw=.5*coord/geom
    return dict(protocol=protocol,frozen_inputs_verified=True,head_tensor_sha256='initial',per_row=rows,
        mean_coordinate_grad_norm=coord,mean_weighted_geometry_grad_norm=geom,
        unclipped_scale=raw,scale=min(10.,max(.1,raw))),training


def test_scale_uses_ratio_of_mean_norms_and_first_32_training_rows():
    scale,training=scale_fixture()
    actual=summary.check_scale(scale,training,'initial')
    expected=.5*sum(row['coordinate_grad_norm'] for row in scale['per_row'])/sum(row['weighted_geometry_grad_norm'] for row in scale['per_row'])
    mean_ratios=np.mean([.5*row['coordinate_grad_norm']/row['weighted_geometry_grad_norm'] for row in scale['per_row']])
    assert actual['unclipped_scale']==expected and abs(expected-mean_ratios)>1.
    wrong=copy.deepcopy(scale);wrong['per_row'][0]['metadata']={'index':32}
    with pytest.raises(ValueError,match='first 32'):summary.check_scale(wrong,training,'initial')
    wrong=copy.deepcopy(scale);wrong['per_row'][0]['coordinate_grad_norm']=-1.
    with pytest.raises(ValueError):summary.check_scale(wrong,training,'initial')
    wrong=copy.deepcopy(scale);wrong['scale']+=.5
    with pytest.raises(ValueError,match='scale formula'):summary.check_scale(wrong,training,'initial')


def test_counts_and_nonfinite_statistics_fail_closed():
    for bad in (True,-1,1.5):
        item=route();item['error']['per_sample'][0]['n_atoms_N']=bad
        with pytest.raises(ValueError):summary.primary(item)
    for bad in (True,-1,1.5):
        item=route();item['explicit_geometry']['counts']['n_c_n'][0]=bad
        with pytest.raises(ValueError):summary.primary(item)
    item=route();item['explicit_geometry']['per_sample']['c_n'][0]=float('nan')
    with pytest.raises(ValueError):summary.primary(item)


def test_registration_and_summary_refuse_overwrite_and_pin_document_hash(tmp_path,monkeypatch):
    monkeypatch.setattr(summary,'REPO',tmp_path)
    document=tmp_path/summary.DOCUMENT;document.parent.mkdir(parents=True)
    document.write_text('synthetic preregistration\n')
    script=tmp_path/summary.SCRIPT;script.parent.mkdir(parents=True);script.write_text('# synthetic source\n')
    monkeypatch.setattr(summary,'DOCUMENT_SHA',hashlib.sha256(document.read_bytes()).hexdigest())
    registration=tmp_path/summary.BASE/'summary.protocol.json'
    actual=summary.register(registration)
    assert actual['bootstrap']['n']==10000 and actual['bootstrap']['seed']==20261101
    assert len(actual['routes'])==14
    with pytest.raises(ValueError,match='overwrite'):summary.register(registration)
    output=tmp_path/summary.BASE/'summary.json';output.write_text('existing')
    with pytest.raises(ValueError,match='overwrite'):summary.run(registration,output)
    assert output.read_text()=='existing'
    document.write_text('changed')
    with pytest.raises(ValueError,match='document differs'):summary.register(registration.with_name('another.json'))


def test_registration_rejects_preexisting_evaluation_without_reading_it(tmp_path,monkeypatch):
    monkeypatch.setattr(summary,'REPO',tmp_path)
    evaluation=tmp_path/summary.BASE/'evaluation200.json'
    evaluation.parent.mkdir(parents=True);evaluation.write_text('intentionally invalid JSON; must not be read')
    with pytest.raises(ValueError,match='unread/uncreated'):summary.register(evaluation.with_name('summary.protocol.json'))


def test_audit_rechecks_hashes_and_verifies_source_snapshot(tmp_path,monkeypatch):
    monkeypatch.setattr(summary,'REPO',tmp_path)
    source=tmp_path/'source.py';source.write_text('initial')
    data=tmp_path/'cache.pt';data.write_bytes(b'input')
    snapshot=tmp_path/'snapshot';snapshot.mkdir();(snapshot/'source.py').write_bytes(source.read_bytes())
    declaration=dict(source_hashes={'source.py':summary.sha(source)},input_hashes={'cache.pt':summary.sha(data)})
    audit=summary.Audit();audit.declarations(declaration,snapshot);audit.verify()
    source.write_text('changed')
    with pytest.raises(ValueError,match='input changed'):audit.verify()
    with pytest.raises(ValueError,match='hash mismatch'):summary.Audit().declarations(declaration,snapshot)
    source.write_text('initial');(snapshot/'source.py').write_text('changed snapshot')
    with pytest.raises(ValueError,match='hash mismatch'):summary.Audit().declarations(declaration,snapshot)
