import copy
import json
from collections import defaultdict

import numpy as np
import pytest

from scripts import summarize_hs_target_pilot as summary


def panel():
    return [(f'p{i:02}',f'p{i:02}_T{summary.TEMPERATURES[i//10]}',summary.TEMPERATURES[i//10],7,h)
            for i in range(50) for h in summary.HORIZONS]


def synthetic_scores():
    ensemble,endpoint = {},{}
    baseline = {}
    for arm,offset in zip(summary.ARMS,(1.,0.)):
        ensemble[arm],endpoint[arm] = {},{}
        for k in panel():
            i = int(k[0][1:])
            value = 5.+offset*(i+1)*(1. if k[4]==16 else 3. if k[4]==64 else -9.)
            ensemble[arm][k] = dict(lasp={m:value for m in summary.METRICS},
                persist={m:1. for m in summary.METRICS},
                geometry_vs_anchor={m:2. for m in summary.shared.LEGACY_GEOMETRY},
                contact_counts={m:3. for m in summary.shared.CONTACT_COUNTS})
            endpoint[arm][k] = dict(scores=dict(model={m:value for m in summary.SCORES},
                persistence={m:2. for m in summary.SCORES}),
                mean_sample_geometry={m:.1 for m in summary.GEOMETRY},
                target_geometry={m:.2 for m in summary.GEOMETRY},persistence_geometry={m:.3 for m in summary.GEOMETRY})
            baseline[k] = dict(bioemu2={m:4. for m in summary.METRICS})
    return ensemble,endpoint,baseline


def test_primary_averages_two_long_horizons_inside_each_protein_before_bootstrap():
    ensemble,endpoint,baseline = synthetic_scores()
    result = summary.analysis(ensemble,endpoint,baseline)
    primary = result['primary_long_horizon_coordinate_UES']
    delta = 2*np.arange(1,51,dtype=float)
    rng = np.random.default_rng(20261101)
    independent = np.quantile(rng.choice(delta,size=(10000,50),replace=True).mean(1),[.025,.975])
    assert primary['candidate_minus_reference']==delta.mean()
    np.testing.assert_allclose(primary['ci95'],independent,rtol=0,atol=1e-12)
    assert primary['improvement']==-delta.mean() and primary['better']=='lower'
    assert primary['n_windows']==100 and primary['n_proteins']==50
    assert [r['candidate_minus_reference'] for r in primary['per_protein']]==delta.tolist()
    assert len(result['seven_metrics']['endpoint_vs_path_mixture'])==10
    assert len(result['endpoint_scores']['endpoint_vs_path_mixture'])==10
    assert result['seven_metrics']['endpoint_vs_path_mixture']['h16']['rmsf_corr']['better']=='higher'
    assert result['canonical_ca_geometry']['endpoint_vs_path_mixture']['overall']['ca_bond_mean_angstrom']['better'] is None
    assert not {'passed','equivalent','noninferior'} & primary.keys()


def test_pairing_is_equal_protein_not_equal_window_and_row_order_invariant():
    keys = [('a','a0',320.,7,1),('a','a0',320.,7,4),('b','b0',348.,7,1)]
    first = {k:{'x':v} for k,v in zip(keys,(0.,2.,10.))}
    second = {k:{'x':0.} for k in keys}
    get = lambda r:r['x']
    result = summary.paired(first,second,keys,get,get,'lower')
    assert result['candidate']==5.5
    assert result==summary.paired(first,second,keys[::-1],get,get,'lower')
    with pytest.raises(ValueError,match='support'):
        summary.paired(first,{keys[0]:{'x':0.}},keys,get,get)
    first[keys[0]]['x'] = float('nan')
    with pytest.raises(ValueError,match='nonfinite'):
        summary.paired(first,second,keys,get,get)


def test_primary_rejects_missing_long_window_instead_of_intersection():
    ensemble,endpoint,baseline = synthetic_scores()
    removed = next(k for k in baseline if k[4]==64)
    del endpoint['endpoint'][removed]
    with pytest.raises(ValueError,match='complete standard200'):
        summary.analysis(ensemble,endpoint,baseline)


def manifest_records():
    return [dict(protein_id=f'p{i:02}',trajectory_id=f'p{i:02}_T{int(T)}_R{r}',
                 temperature_k=T,num_frames=100) for T in summary.TEMPERATURES
            for i in range(12) for r in range(5)]


def synthetic_suite(kind):
    records = manifest_records()
    keys = sorted(summary.expected_suite_keys(records,kind))
    n_proteins,n_anchors = (12,2) if kind=='suite' else (8,1)
    modes = ['correct','wrong','none'] if kind=='suite' else ['correct']
    rows = []
    for protein,traj,T,anchor,h,mode in keys:
        value = float(int(protein[1:])+h+(mode=='wrong'))
        rows.append(dict(protein=protein,traj=traj,temperature_k=T,T_star=T,anchor_frame=anchor,
            horizon=float(h),mode=mode,temperature_used_k=T if mode=='correct' else
                (450. if T==320. else 320.) if mode=='wrong' else None,
            model={m:value for m in summary.METRICS},persist={m:1. for m in summary.METRICS},
            geometry_vs_anchor={m:0. for m in summary.shared.LEGACY_GEOMETRY},
            position=dict(kabsch_ca_rmsd=value,aligned_ca_rmsd=value,persist_kabsch_ca_rmsd=2.,persist_aligned_ca_rmsd=2.)))
    def published(selected):
        def mean(get):
            buckets=defaultdict(list)
            for row in selected: buckets[row['protein']].append(get(row))
            return {'mean':sum(sum(v)/len(v) for v in buckets.values())/len(buckets)}
        out=dict(n_windows=len(selected),n_proteins=len({r['protein'] for r in selected}),
            n_trajectories=len({r['traj'] for r in selected}))
        for m in summary.METRICS:
            out[m]=mean(lambda r,m=m:r['model'][m]); out['persist_'+m]=mean(lambda r,m=m:r['persist'][m])
        for m in selected[0]['position']:out[m]=mean(lambda r,m=m:r['position'][m])
        out['skill_rmsf']=mean(lambda r:r['persist']['rmsf_mae']-r['model']['rmsf_mae'])
        out['skill_position']=mean(lambda r:r['position']['persist_kabsch_ca_rmsd']-r['position']['kabsch_ca_rmsd'])
        return out
    t2a_rows = {str(int(T)):[dict(protein=f'p{i:02}',trajectory=f'p{i:02}_T{int(T)}_R{rep}',
        n_model_samples=12*4*n_anchors,n_states=8,n_ref_frames=100,**{m:float(i) for m in summary.T2A_METRICS})
        for i in range(n_proteins) for rep in range(5)] for T in summary.TEMPERATURES}
    return dict(args=dict(temps=list(summary.TEMPERATURES),modes=modes,wrong_temp=320.,horizons=list(summary.HORIZONS),
        proteins_per_temp=n_proteins,anchors_per_trajectory=n_anchors,n_samples=12,context_frames=8,
        max_residues=96,k_states=8,seed=0,t2a=True,manifest=summary.shared.VAL),
        noise_protocol=dict(version='shared-window-noise-v1',shared_across_modes=True,
            seed_identity=['protein_id','trajectory_id','anchor_frame','horizon']),
        n_windows=len(rows),n_proteins=n_proteins,n_trajectories=n_proteins*25,per_window=rows,
        by_temperature={str(int(T)):{mode:published([r for r in rows if r['T_star']==T and r['mode']==mode])
            for mode in modes} for T in summary.TEMPERATURES},
        by_horizon={str(h):{mode:published([r for r in rows if int(r['horizon'])==h and r['mode']==mode])
            for mode in modes} for h in summary.HORIZONS},
        aggregate={mode:published([r for r in rows if r['mode']==mode]) for mode in modes},
        t2a_per_protein=t2a_rows,t2a={T:dict(n_proteins=n_proteins,n_trajectories=n_proteins*5,
            **{m:{'mean':(n_proteins-1)/2} for m in summary.T2A_METRICS}) for T in t2a_rows})


@pytest.mark.parametrize('kind,count',[('suite',7200),('matched8',800)])
def test_complete_suite_contract_and_t2a_row_units(kind,count):
    data = synthetic_suite(kind)
    keys,report = summary.audit_suite(data,'endpoint',kind)
    assert len(keys)==count and set(keys)==summary.expected_suite_keys(manifest_records(),kind)
    assert report['t2a']['320']['row_unit']=='reference_trajectory'
    assert 'n_windows' not in report['t2a']['320']
    assert report['t2a']['320']['n_trajectory_rows']==(60 if kind=='suite' else 40)


def test_suite_rejects_missing_window_wrong_temperature_and_false_aggregate():
    data = synthetic_suite('matched8')
    bad=copy.deepcopy(data); bad['per_window'].pop()
    with pytest.raises(ValueError,match='incomplete'):summary.audit_suite(bad,'endpoint','matched8')
    bad=copy.deepcopy(data); bad['aggregate']['correct']['rmsf_mae']['mean']+=1
    with pytest.raises(ValueError,match='published means'):summary.audit_suite(bad,'endpoint','matched8')
    bad=synthetic_suite('suite')
    next(r for r in bad['per_window'] if r['mode']=='wrong')['temperature_used_k']=320.
    with pytest.raises(ValueError,match='degenerate'):summary.audit_suite(bad,'endpoint','suite')


def test_suite_rejects_fabricated_global_protein_count_and_replicas():
    data=synthetic_suite('matched8')
    for row in data['per_window']:
        if row['T_star']==450.:row['protein']='different_'+row['protein']
    with pytest.raises(ValueError,match='raw distinct'):summary.audit_suite(data,'endpoint','matched8')
    data=synthetic_suite('matched8')
    data['t2a_per_protein']['320'][0]['n_model_samples']=96
    with pytest.raises(ValueError,match='sample/reference'):summary.audit_suite(data,'endpoint','matched8')


def test_manifest_keys_use_insertion_order_and_exact_anchors():
    records=manifest_records()
    # A later alphabetic ID is selected when encountered first; selection is not sorted.
    for r in records:
        if r['protein_id']=='p00':r['protein_id']='z_first';r['trajectory_id']='z_first_'+r['trajectory_id']
    keys=summary.expected_suite_keys(records,'matched8')
    assert len(keys)==800 and 'z_first' in {k[0] for k in keys}
    assert {k[3] for k in keys}=={7}
    assert 'p08' not in {k[0] for k in keys}
    records[0]['num_frames']=71
    with pytest.raises(ValueError,match='too short'):summary.expected_suite_keys(records,'matched8')


def test_registration_is_exclusive_precedes_results_and_pins_definition(tmp_path,monkeypatch):
    monkeypatch.setattr(summary,'REPO',tmp_path)
    file=summary.BASE/'summary.protocol.json'
    contract=summary.register(file)
    assert contract['primary']['horizons']==[16,64] and contract['bootstrap']['reps']==10000
    assert contract['training']['distance_weight']==0
    with pytest.raises(ValueError,match='overwrite'):summary.register(file)
    (tmp_path/file).unlink()
    (tmp_path/summary.BASE/'endpoint.endpoint.json').write_text('{}')
    with pytest.raises(ValueError,match='formal result'):summary.register(file)


def test_incomplete_execution_fails_before_checkpoint_read_and_inputs_rehash(tmp_path,monkeypatch):
    monkeypatch.setattr(summary,'REPO',tmp_path)
    root=tmp_path/summary.BASE;root.mkdir(parents=True)
    file=root/'endpoint.execution.json';file.write_text(json.dumps(dict(arm='endpoint',status='running',jobs=[])))
    inputs=summary.Inputs()
    with pytest.raises(ValueError,match='incomplete'):summary.audit_execution('endpoint',{},inputs)
    file.write_text('{}')
    with pytest.raises(ValueError,match='changed'):inputs.verify()
    with pytest.raises(ValueError,match='sealed'):summary.local(tmp_path/'test.jsonl')
    # Read-only access is allowed; output locations remain fixed to the new experiment.
    assert summary.local(tmp_path/'results/standard_eval/stopped_best/a.json')
    with pytest.raises(ValueError,match='unregistered location'):
        summary.build_summary(tmp_path/'results/standard_eval/stopped_best/summary.json')


def test_endpoint_validator_rejects_target_swap_or_future_sampler_input():
    data=dict(task_version='ca_conditional_endpoint_scores_v1',dataset_role='development_validation',
        prediction_kind='endpoint',reference_kind='single_matched_future_endpoint',
        future_information_given_to_sampler=False,checkpoint_step=500,checkpoint_training_target='path_mixture')
    with pytest.raises(ValueError,match='task/target'):summary.audit_endpoint(data,'endpoint',{})
    data['checkpoint_training_target']='endpoint';data['future_information_given_to_sampler']=True
    with pytest.raises(ValueError,match='task/target'):summary.audit_endpoint(data,'endpoint',{})
