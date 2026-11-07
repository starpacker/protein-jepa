"""Synthetic NumPy-only score checks; no scientific inputs or producer imports."""
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
import pytest

FILE=Path(__file__).resolve().parents[1]/'run_scripts/verify_frozen_h_predictor_calibration_v1.py'
spec=importlib.util.spec_from_file_location('frozen_predictor_independent_test',FILE)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


def arrays(n=3):
    target=np.arange(1,n+1,dtype=np.float32)
    result=dict(mask=np.ones(n+3,dtype=bool),pair_indices=np.array([[0,i+3] for i in range(n)],dtype=np.int64),
        target_distance=target,target_log=np.log1p(target).astype('f4'),observed_distance=target.copy(),persistence_log=np.log1p(target).astype('f4'))
    for seed in v.SEEDS:
        for role in v.ROLES:result[f'seed{seed}__{role}__predicted_log']=np.full(n,-.5,dtype=np.float32)
    return result


def records():
    result=[]
    for i in range(100):
        a=arrays(1+i%3);scores=v.score_arrays(a)
        for seed in v.SEEDS:
            for j,role in enumerate(v.ROLES):
                error=(i%7+1)*(seed-41)+j
                scores[f'{seed}/{role}'].update(log_mse=float(error),mae_angstrom=float(error),mse_angstrom2=float(error**2),rmse_angstrom=float(error))
        result.append(dict(cache_row_index=400+i,metadata=dict(protein=str(i),horizon=v.HORIZONS[i%4]),n_pairs=len(a['pair_indices']),scores=scores))
    return result


def test_all_head_roles_negative_raw_distance_and_zero_persistence():
    a=arrays();out=v.score_arrays(a)
    assert len(out)==11 and out['CA_persistence']['log_mse']==out['CA_persistence']['mse_angstrom2']==0
    errors=[math.expm1(-.5)-d for d in (1,2,3)]
    for seed in v.SEEDS:
        for role in v.ROLES:
            value=out[f'{seed}/{role}'];assert value['negative_count']==3
            assert value['mae_angstrom']==pytest.approx(sum(abs(e) for e in errors)/3,abs=1e-14)
            assert value['mse_angstrom2']==pytest.approx(sum(e*e for e in errors)/3,abs=1e-14)


@pytest.mark.parametrize('error',('mask','duplicate','order','missing_role','wrong_dtype','bad_label','nonfinite','overflow'))
def test_corrupt_pair_arrays_rejected(error):
    a=arrays()
    if error=='mask':a['mask'][0]=False
    elif error=='duplicate':a['pair_indices'][1]=a['pair_indices'][0]
    elif error=='order':a['pair_indices']=a['pair_indices'][::-1].copy()
    elif error=='missing_role':del a['seed43__physical__predicted_log']
    elif error=='wrong_dtype':a['seed42__raw_initial__predicted_log']=a['seed42__raw_initial__predicted_log'].astype('f8')
    elif error=='bad_label':a['target_log']+=.01
    elif error=='nonfinite':a['seed43__latent__predicted_log'][0]=np.nan
    else:a['seed43__physical__predicted_log'][0]=1000
    with pytest.raises((ValueError,KeyError)):v.score_arrays(a)


def test_selected_npz_skips_all_H(tmp_path):
    a=arrays();a.update(history_H=np.array([object()],dtype=object),target_H=np.array([object()],dtype=object))
    file=tmp_path/'pairs.npz';np.savez(file,**a)
    selected=v.selected_arrays(file)
    assert 'history_H' not in selected and 'target_H' not in selected
    assert len(v.score_arrays(selected))==11


def test_equal_protein_mse_then_sqrt_and_negative_fraction():
    rows=records();out=v.aggregate(rows)
    assert len(out)==5 and len(out['overall'])==11
    values=[r['scores']['43/physical']['mse_angstrom2'] for r in rows]
    assert out['overall']['43/physical']['rmse_angstrom']==math.sqrt(np.mean(values))
    assert out['overall']['43/physical']['rmse_angstrom']!=np.mean(np.sqrt(values))
    assert out['overall']['43/physical']['mean_protein_negative_fraction']==1.
    assert out['overall']['CA_persistence']['mean_protein_negative_fraction']==0.
    assert out['overall']['43/physical']['n_pairs']==sum(r['n_pairs'] for r in rows)
    assert all(out['h'+str(h)]['43/physical']['n_proteins']==25 for h in v.HORIZONS)


def test_paired_bootstrap_independent_count_weight_reference():
    rows=records();mapping={'43/physical-minus-latent':('43/physical','43/latent')}
    output=v.paired_contrasts(rows,mapping);rng=np.random.default_rng(20261102);counts=np.zeros((10000,100),dtype=np.int16)
    for k in range(4):
        draws=rng.integers(0,25,size=(10000,25))
        for j in range(25):np.add.at(counts,(np.arange(10000),4*draws[:,j]+k),1)
    x=np.array([r['scores']['43/physical']['mse_angstrom2'] for r in rows]);y=np.array([r['scores']['43/latent']['mse_angstrom2'] for r in rows])
    dist=np.sort(np.sqrt(counts@x/100)-np.sqrt(counts@y/100));expected=[]
    for q in (.025,.975):
        z=q*9999;j=int(z);expected.append(float(dist[j]+(z-j)*(dist[j+1]-dist[j])))
    got=output['43/physical-minus-latent']['rmse_angstrom']
    np.testing.assert_allclose(got['interval95'],expected,atol=1e-14,rtol=0)
    assert got['difference']==math.sqrt(x.mean())-math.sqrt(y.mean())
    assert output['43/physical-minus-latent']['log_mse']['difference']==1.


def test_panel_rejects_horizon_or_identity_replacement():
    rows=records();rows[0]['metadata']['horizon']=4
    with pytest.raises(ValueError):v.panel(rows)
    rows=records();rows[0]['metadata']['protein']=rows[1]['metadata']['protein']
    with pytest.raises(ValueError):v.panel(rows)


def test_shared_bound_is_json_native_and_source_byte_pinned():
    assert v.common.LOG_FP32_BOUND==2**-20 and type(v.common.LOG_FP32_BOUND)is float
    assert v.sha(v.COMMON_PATH)==v.COMMON_SHA
    assert json.loads(json.dumps(dict(bound=v.common.LOG_FP32_BOUND)))['bound']==2**-20
