import pytest
from scripts.eval_lasp_ca_probe import bootstrap
from scripts.compare_lasp_ca_probes import key


def test_bootstrap_weights_proteins_equally_not_windows():
    rows=[dict(protein='a',gain=1.)]*10+[dict(protein='b',gain=3.)]
    result=bootstrap(rows,'gain')
    assert result['mean']==2.
    assert result['n_proteins']==2
    assert result['ci95'][0]<=2<=result['ci95'][1]


def test_window_identity_includes_trajectory_stride_and_times():
    row=dict(protein='p',trajectory='t',stride=4,time_ps=[[0.,4000.,8000.]])
    assert key(row)!=key(dict(row,stride=1))
    assert key(row)!=key(dict(row,trajectory='other'))
    assert key(row)!=key(dict(row,time_ps=[[1000.,5000.,9000.]]))
