import json
import sys
from scripts import summarize_lasp_ca_dynamics as summary
from scripts.eval_lasp_final import METRICS


def test_nonempty_contact_diagnostic_preserves_full_protocol_summary(tmp_path,monkeypatch):
    metrics={m:0. for m in METRICS}
    rows=[]
    for horizon,count,score in [(1,0,0.),(4,8,.4)]:
        pred=dict(metrics,weak_contact_j=score)
        rows.append(dict(protein='p',horizon=horizon,lasp=pred,persist=metrics,
            contact_counts=dict(weak_n_true=count,transient_n_true=1)))
    src=tmp_path/'input.json';out=tmp_path/'out.json'
    src.write_text(json.dumps(dict(per_window=rows)))
    monkeypatch.setattr(sys,'argv',['summary',str(src),'--out',str(out)])
    summary.main()
    d=json.loads(out.read_text())['input']
    assert d['metrics']['weak_contact_j']['candidate']['mean']==.2
    assert d['nonempty_true_contacts']['weak_contact_j']['candidate']['mean']==.4
    assert d['nonempty_true_contacts']['weak_contact_j']['n_windows']==1
