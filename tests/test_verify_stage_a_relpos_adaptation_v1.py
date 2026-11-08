"""Synthetic only; independent scalar references, no producer math imports."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np
import pytest

PATH=Path(__file__).resolve().parents[1]/'run_scripts/verify_stage_a_relpos_adaptation_v1.py'
spec=importlib.util.spec_from_file_location('independent_relpos_test',PATH)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


def arrays(n=3):
    d=np.arange(1,n+1,dtype=np.float32)*2
    out={}
    for arm in v.ARMS:
        out.update({arm+'__pair_indices':np.array([[0,i+3] for i in range(n)],dtype=np.int64),
            arm+'__target_distance':d.copy(),arm+'__target_log':np.log1p(d).astype('f4'),
            arm+'__observed_distance':d.copy(),arm+'__persistence_log':np.log1p(d).astype('f4')})
        for role in v.ROLES:out[arm+'__'+role+'__predicted_log']=np.full(n,-.25,dtype='f4')
    return out


def records():
    rows=[]
    for i in range(100):
        scores={};n=1+i%3
        for a,arm in enumerate(v.ARMS):
            for j,role in enumerate(v.ROLES):
                error=(i%7+1)*(a+1)+j
                scores[arm+'/'+role]=dict(log_mse=float(error),mae_angstrom=float(error),
                    mse_angstrom2=float(error**2),rmse_angstrom=float(error),n_pairs=n,negative_count=0)
        scores['CA_persistence']=dict(log_mse=0.,mae_angstrom=0.,mse_angstrom2=0.,rmse_angstrom=0.,n_pairs=n,negative_count=0)
        rows.append(dict(cache_row_index=i+400,metadata=dict(protein=str(i),horizon=v.HORIZONS[i%4]),n_pairs=n,scores=scores))
    return rows


def test_scalar_reference_negative_prediction_and_raw_target():
    p=np.array([-.5,.2],dtype='f4');target=np.array([1.,3.],dtype='f4');y=np.log1p(target).astype('f4')
    result=v.metrics(p,y,target)
    distances=[math.expm1(float(x)) for x in p];errors=[x-float(t) for x,t in zip(distances,target)]
    assert result['negative_count']==1
    assert result['mse_angstrom2']==pytest.approx(sum(e*e for e in errors)/2,abs=1e-14)
    assert result['log_mse']==sum((float(a)-float(b))**2 for a,b in zip(p,y))/2
    assert v.metrics(y,y,target)['mse_angstrom2']>0  # raw target, not reconstructed target log


def test_zero_persistence_and_fp32_log_rounding():
    a=arrays();s=v.score_arrays(a)
    assert s['CA_persistence']['mse_angstrom2']==s['CA_persistence']['log_mse']==0
    assert s['pca/future_H']['negative_count']==3
    y=a['pca__target_log'];v.log_link(np.nextafter(y,np.float32(np.inf)),a['pca__target_distance'])
    with pytest.raises(ValueError,match='log label'):v.log_link(y+.01,a['pca__target_distance'])


@pytest.mark.parametrize('change',('pairs','labels','negative_target','duplicate','order','nonfinite','dtype','overflow','squared_overflow'))
def test_bad_pair_arrays_fail(change):
    a=arrays()
    if change=='pairs':a['zero__pair_indices'][0,1]+=1
    elif change=='labels':a['local__target_distance'][0]+=1
    elif change=='negative_target':a['pca__target_distance'][0]=-1
    elif change=='duplicate':a['pca__pair_indices'][1]=a['pca__pair_indices'][0]
    elif change=='order':a['pca__pair_indices']=a['pca__pair_indices'][::-1].copy()
    elif change=='nonfinite':a['pca__future_H__predicted_log'][0]=np.nan
    elif change=='dtype':a['pca__target_log']=a['pca__target_log'].astype('f8')
    elif change=='overflow':a['pca__future_H__predicted_log'][0]=1000
    else:a['pca__future_H__predicted_log'][0]=700
    with pytest.raises(ValueError):v.score_arrays(a)


def test_lazy_npz_does_not_materialize_object_H(tmp_path):
    a=arrays();a['pca__H_predicted']=np.array([object()],dtype=object)
    file=tmp_path/'scores.npz';np.savez(file,**a)
    selected=v.selected_arrays(file)
    assert 'pca__H_predicted' not in selected
    assert v.score_arrays(selected)['CA_persistence']['rmse_angstrom']==0


def test_aggregate_equal_protein_sqrt_after_mean_and_zero():
    r=records();out=v.aggregate(r);values=[x['scores']['pca/future_H']['mse_angstrom2'] for x in r]
    expected=math.sqrt(sum(values)/100)
    assert out['overall']['pca/future_H']['rmse_angstrom']==expected
    assert expected!=np.mean([x['scores']['pca/future_H']['rmse_angstrom'] for x in r])
    assert out['overall']['CA_persistence']['rmse_angstrom']==0
    assert all(out['h'+str(h)]['pca/future_H']['n_proteins']==25 for h in v.HORIZONS)
    wrong=copy.deepcopy(out);wrong['overall']['pca/future_H']['rmse_angstrom']+=.1
    with pytest.raises(ValueError):v.Audit().compare(wrong,out)


def test_bootstrap_manual_count_weight_reference_and_root_order():
    r=records();actual=v.bootstrap(r);rng=np.random.default_rng(20261102)
    # Independently turn the same prespecified random integers into counts.
    weights=np.zeros((10000,100),dtype=np.int16)
    for h in range(4):
        draws=rng.integers(0,25,size=(10000,25))
        for column in range(25):np.add.at(weights,(np.arange(10000),4*draws[:,column]+h),1)
    x=np.array([z['scores']['zero/future_H']['mse_angstrom2'] for z in r]);y=np.array([z['scores']['pca/future_H']['mse_angstrom2'] for z in r])
    dist=np.sqrt((weights@x)/100)-np.sqrt((weights@y)/100)
    ordered=np.sort(dist);bounds=[]
    for q in (.025,.975):
        k=q*9999;lo=int(math.floor(k));bounds.append(float(ordered[lo]+(k-lo)*(ordered[lo+1]-ordered[lo])))
    got=actual['zero-pca']['future_H']['rmse_angstrom']
    np.testing.assert_allclose(got['interval95'],bounds,atol=1e-14,rtol=0)
    assert got['difference']==math.sqrt(x.mean())-math.sqrt(y.mean())
    assert len(actual)==2 and set(actual['local-pca'])==set(v.ROLES)


@pytest.mark.parametrize('change',('horizon','identity','roles','paircount'))
def test_panel_contract(change):
    r=records()
    if change=='horizon':r[0]['metadata']['horizon']=4
    elif change=='identity':r[0]['metadata']['protein']=r[1]['metadata']['protein']
    elif change=='roles':del r[0]['scores']['local/predicted_H']
    else:r[0]['n_pairs']+=1
    with pytest.raises(ValueError):v.panel(r)


class RefAudit(v.Audit):
    def ref(self,ref):return Path(ref['path'])


def adaptation_log(arm='pca'):
    log=[];manifest=[]
    for i in range(500):
        ref=dict(path=str(v.PRODUCER/('train_'+arm)/'draws'/f'{i+1:04d}.pt'),sha256='a'*64)
        manifest.append(ref);scale=(i+1)/100 if i<100 else (1+math.cos(math.pi*(i-100)/400))/2
        log.append(dict(step=i+1,feature_mode=arm,cache_row_indices=[(4*i+j)%400 for j in range(4)],
            ema_before=i,ema_after=i+1,ema_momentum=.998,optimizer_updated=True,lr=[3e-4*.3*scale,3e-4*scale],
            losses={'loss/total':1.},global_grad_norm_before_clip=2.,encoder_grad_norm_before_clip=1.,
            tensor_audit=ref,tensor_audit_sha256='b'*64,draws_sha256='c'*64,input_sha256='d'*64,
            rng_sha256=dict(before=f'{i:064x}',after_forward=f'{i+1:064x}',after_backward=f'{i+1:064x}',after_update=f'{i+1:064x}'),
            ranking_eligible_counts=[6]*7,ranking_regression_count=28))
    return log,manifest


def test_all500_adaptation_lr_cycle_ema_and_tape():
    log,manifest=adaptation_log();v.training_logs(log,'pca',RefAudit(),manifest)
    assert log[0]['lr'][1]==pytest.approx(3e-6) and log[-1]['lr'][1]>0


@pytest.mark.parametrize('change',('cycle','ema','lr','rng','drawref','ranking'))
def test_adaptation_log_corruption(change):
    log,manifest=adaptation_log();r=log[400]
    if change=='cycle':r['cache_row_indices'][0]=400
    elif change=='ema':r['ema_momentum']=.999
    elif change=='lr':r['lr'][1]=0
    elif change=='rng':r['rng_sha256']['after_backward']='f'*64
    elif change=='drawref':r['tensor_audit']=manifest[0]
    else:r['ranking_eligible_counts'][0]=7
    with pytest.raises(ValueError):v.training_logs(log,'pca',RefAudit(),manifest)


def test_readout_500_fit_order_repeat_labels():
    metadata=[dict(protein=str(i),horizon=v.HORIZONS[i%4]) for i in range(500)]
    log=[dict(step=i+1,cache_row_index=i%400,**metadata[i%400],lr=.001,n_pairs=2,log_mse=0.,gradient_norm_before_clip=0.,labels_sha256=f'{i%400:064x}') for i in range(500)]
    v.readout_logs(log,metadata)
    log[400]['cache_row_index']=400
    with pytest.raises(ValueError):v.readout_logs(log,metadata)


def phase_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(v,'ROOT',tmp_path);monkeypatch.setattr(v,'PRODUCER',tmp_path/'producer')
    v.PRODUCER.mkdir();start='2026-11-02T00:00:00+00:00'
    for i,phase in enumerate(v.PHASES):
        folder=v.PRODUCER/phase;folder.mkdir()
        st=dict(pid=123+i,command=['python','producer','--phase',phase],started_utc=f'2026-11-02T00:{i+1:02d}:00+00:00',protocol_sha256=v.MAIN_PROTOCOL_SHA,phase=phase)
        r=dict(phase=phase,protocol_sha256=v.MAIN_PROTOCOL_SHA,status='completed',test_opened=False,bioemu2_rerun=False,artifacts={})
        v.write(folder/'result.json',r);v.write(folder/'started.json',st)
        v.write(folder/'execution.json',dict(st,status='completed',returncode=0,completed_utc=f'2026-11-02T00:{i+1:02d}:30+00:00',seconds=30.,result=dict(path=str(folder/'result.json'),sha256=v.sha(folder/'result.json'))))
    return dict(phase_order=list(v.PHASES),registered_utc=start)


def test_phase_order_and_honest_prescore_registration(tmp_path,monkeypatch):
    main=phase_fixture(tmp_path,monkeypatch)
    v.phase_results(main,v.Audit(),v.timestamp('2026-11-02T00:08:45+00:00'))  # after training, before score
    with pytest.raises(ValueError,match='after score'):v.phase_results(main,v.Audit(),v.timestamp('2026-11-02T00:09:01+00:00'))
    p=v.PRODUCER/'fit_local/execution.json';r=json.loads(p.read_text());r['completed_utc']='2026-11-02T00:10:00+00:00';p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='order/time'):v.phase_results(main,v.Audit(),v.timestamp('2026-11-02T00:00:00+00:00'))


def test_hash_drift_and_exclusive_output(tmp_path,monkeypatch):
    monkeypatch.setattr(v,'ROOT',tmp_path);f=tmp_path/'x.json';v.write(f,dict(a=1));audit=v.Audit();audit.read(f)
    with pytest.raises(FileExistsError):v.write(f,{})
    f.write_text('{}')
    with pytest.raises(ValueError,match='changed'):audit.finish()


def test_source_imports_numpy_stdlib_only():
    import ast
    tree=ast.parse(PATH.read_text())
    names=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):names.extend(a.name.split('.')[0] for a in node.names)
        elif isinstance(node,ast.ImportFrom):names.append(node.module.split('.')[0])
    assert set(names)<={'__future__','argparse','datetime','hashlib','json','math','os','pathlib','sys','time','numpy'}


def test_registration_failure_is_preserved_and_cannot_retry(tmp_path,monkeypatch):
    monkeypatch.setattr(v,'ROOT',tmp_path);monkeypatch.setattr(v,'PRODUCER',tmp_path/'producer');monkeypatch.setattr(v,'OUTPUT',tmp_path/'verifier')
    (v.PRODUCER/'score').mkdir(parents=True);(v.PRODUCER/'score/started.json').write_text('{}')
    with pytest.raises(ValueError,match='precede scoring'):v.register()
    r=json.loads((v.OUTPUT/'registration.execution.json').read_text())
    assert r['status']=='failed' and r['returncode']==1 and r['pid']>0
    before=(v.OUTPUT/'registration.execution.json').read_bytes()
    with pytest.raises(FileExistsError):v.register()
    assert (v.OUTPUT/'registration.execution.json').read_bytes()==before


def test_runtime_failure_and_exclusive_started_record(tmp_path,monkeypatch):
    monkeypatch.setattr(v,'ROOT',tmp_path);monkeypatch.setattr(v,'PRODUCER',tmp_path/'producer');monkeypatch.setattr(v,'OUTPUT',tmp_path/'verifier');v.OUTPUT.mkdir()
    v.write(v.OUTPUT/'protocol.json',dict(main_protocol_sha256=v.MAIN_PROTOCOL_SHA,schema_version=v.VERSION,input_hashes={},source_hashes={}))
    v.write(v.OUTPUT/'registration.execution.json',dict(status='completed',returncode=0,protocol_sha256=v.sha(v.OUTPUT/'protocol.json'),pid=5,seconds=1.,started_utc='2026-11-02T00:00:00+00:00',completed_utc='2026-11-02T00:00:01+00:00'))
    with pytest.raises(FileNotFoundError):v.run()
    r=json.loads((v.OUTPUT/'execution.json').read_text());assert r['status']=='failed' and r['returncode']==1
    before=(v.OUTPUT/'execution.json').read_bytes()
    with pytest.raises(FileExistsError):v.run()
    assert (v.OUTPUT/'execution.json').read_bytes()==before
