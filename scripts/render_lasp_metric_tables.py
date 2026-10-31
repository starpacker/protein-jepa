"""Render a matched, explicitly interim snapshot of existing evaluation metrics."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import html
import json
import numpy as np
import torch
from evaluation.t2a_static import static_ensemble_terms
from scripts.eval_lasp_ca_suite import reference_ensemble

ROOT = Path('results/lasp_full_v1')
OUT = ROOT / 'metric_tables.html'
TEMPS = (320, 348, 379, 413, 450)
ENSEMBLE = ('rmsf_mae', 'rmsf_corr', 'root_mean_w2', 'variance_w2', 'md_pca_w2', 'weak_contact_j', 'transient_contacts_j')
STATIC = ('coverage', 'free_energy_mae_kcal', 'population_mae', 'population_tv', 'population_jsd')


def identity(r):
    return (r['protein'], r['traj'], int(r['anchor_frame']), float(r['horizon']))


def mean(rows, key):
    grouped = defaultdict(list)
    for r in rows:
        value = float(r[key])
        if not np.isfinite(value):
            raise ValueError(f'nonfinite {key}')
        grouped[r['protein']].append(value)
    return float(np.mean([np.mean(v) for v in grouped.values()]))


def main():
    torch.set_num_threads(2)
    runs = [json.loads((ROOT / f'mid_suite/matched8/seed{s}.json').read_text()) for s in (42, 43, 44)]
    anchors = json.loads((ROOT / 't2a_baselines/anchors_8prot.json').read_text())
    temperature_rows, static_rows, counts = [], [], {}
    detail = {}
    for T in TEMPS:
        stem = ROOT / f't2a_bioemu2/bioemu2_T{T}'
        path = Path(str(stem) + '.json')
        if not path.exists():
            path = Path(str(stem) + '.partial.json')
        baseline = json.loads(path.read_text())
        groups = defaultdict(list)
        for r in baseline['per_window']:
            groups[(r['protein'], r['traj'], r['anchor_frame'])].append(r)
        complete = {g for g, rows in groups.items() if len(rows) == 4 and {r['horizon'] for r in rows} == {1, 4, 16, 64}}
        bio = [r for g, rows in groups.items() if g in complete for r in rows]
        keys = {identity(r) for r in bio}
        assert len(keys) == len(bio) and bio
        selected = [[r for r in d['per_window'] if identity(r) in keys and r['temperature_k'] == T] for d in runs]
        assert all({identity(r) for r in rows} == keys for rows in selected)
        proteins = {r['protein'] for r in bio}
        trajectories = {r['traj'] for r in bio}
        counts[T] = dict(windows=len(bio), proteins=len(proteins), trajectories=len(trajectories))
        for method in ('LASP', 'Persistence', 'BioEmu2'):
            vals = {}
            for k in ENSEMBLE:
                if method == 'LASP':
                    vals[k] = float(np.mean([mean([dict(protein=r['protein'], **r['model']) for r in rows], k) for rows in selected]))
                else:
                    field = 'persist' if method == 'Persistence' else 'bioemu2'
                    vals[k] = mean([dict(protein=r['protein'], **r[field]) for r in bio], k)
            temperature_rows.append(dict(temperature=T, method=method, **vals))
        refs = reference_ensemble('artifacts/mdcath_manifest_full/val.jsonl', trajectories, 96)
        bio_terms = []
        for (protein, traj, anchor), rows in sorted(groups.items()):
            if (protein, traj, anchor) not in complete:
                continue
            samples, mask = [], None
            for r in rows:
                p = ROOT / f't2a_bioemu2/samples/{protein}__{traj}__{anchor}__{r["horizon"]:g}.npz'
                with np.load(p, allow_pickle=False) as d:
                    current = torch.as_tensor(d['residue_mask']).bool()
                    if mask is not None:
                        assert torch.equal(mask, current)
                    mask = current
                    samples.append(torch.as_tensor(d['model_ca']).float())
            ref, ref_mask = refs[traj]
            assert torch.equal(mask, ref_mask)
            terms = static_ensemble_terms(torch.cat(samples), ref[:, mask], None, T, k_states=8, seed=0)
            bio_terms.append(dict(protein=protein, trajectory=traj, **terms))
        for method in ('LASP', 'Persistence', 'BioEmu2'):
            vals = {}
            for k in STATIC:
                if method == 'LASP':
                    vals[k] = float(np.mean([mean([r for r in d['t2a_per_protein'][str(T)] if r['trajectory'] in trajectories], k) for d in runs]))
                elif method == 'Persistence':
                    vals[k] = mean([r for r in anchors['per_protein']['persistence'][str(T)] if r['trajectory'] in trajectories], k)
                else:
                    vals[k] = mean(bio_terms, k)
            static_rows.append(dict(temperature=T, method=method, **vals))
        detail[str(T)] = dict(counts=counts[T], window_identities=sorted(keys), bioemu2_source=str(path), bioemu2_t2a=bio_terms)
        print(f'T={T}: {len(bio)} matched windows, {len(trajectories)} complete trajectories', flush=True)
    snapshot = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='interim_matched_snapshot', temperature=temperature_rows, t2a=static_rows, detail=detail)
    OUT.with_suffix('.json').write_text(json.dumps(snapshot, indent=2))
    labels = {'rmsf_mae':'RMSF MAE ↓', 'rmsf_corr':'RMSF corr ↑', 'root_mean_w2':'Root-mean W2 ↓', 'variance_w2':'Variance W2 ↓', 'md_pca_w2':'MD-PCA W2 ↓', 'weak_contact_j':'Weak contact J ↑', 'transient_contacts_j':'Transient contact J ↑', 'coverage':'Coverage ↑', 'free_energy_mae_kcal':'F-MAE ↓', 'population_mae':'Population MAE ↓', 'population_tv':'Population TV ↓', 'population_jsd':'Population JSD ↓'}
    def table(rows, metrics):
        text = '<div class="scroll"><table><thead><tr><th>温度</th><th>Method</th>' + ''.join(f'<th>{labels[k]}</th>' for k in metrics) + '</tr></thead><tbody>'
        for T in TEMPS:
            for i, method in enumerate(('LASP', 'Persistence', 'BioEmu2')):
                r = next(r for r in rows if r['temperature']==T and r['method']==method)
                text += '<tr' + (' class="group"' if i == 0 else '') + '>'
                if i == 0:
                    c = counts[T]
                    text += f'<th rowspan="3" class="temp">{T} K<small>{c["proteins"]} 蛋白 · {c["trajectories"]} 轨迹<br>{c["windows"]}/160 窗口</small></th>'
                text += f'<th class="method">{method}</th>' + ''.join(f'<td>{r[k]:.4f}</td>' for k in metrics) + '</tr>'
        return text + '</tbody></table></div>'
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>LASP · Temperature & T2A</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f4f5f7;color:#20252c;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.65}main{max-width:1500px;margin:40px auto;padding:0 28px}h1{font-size:30px;letter-spacing:-.7px;margin-bottom:8px}h2{font-size:23px;margin:0 0 5px}p{margin:8px 0}.eyebrow{font-size:12px;letter-spacing:2px;color:#5c6470}.muted,small{color:#606975}.notice{background:#fff8e8;border:1px solid #d9bc78;border-radius:10px;padding:15px 20px;margin:22px 0}.card{background:white;border:1px solid #dce0e5;border-radius:12px;padding:24px;margin:24px 0}.scroll{overflow:auto;margin-top:20px}table{width:100%;border-collapse:collapse;white-space:nowrap;font-size:14px;font-variant-numeric:tabular-nums}th,td{padding:12px 14px;text-align:right;border-bottom:1px solid #e8ebee}thead th{background:#f1f3f5;color:#434d5a;font-size:12px}thead th:first-child,thead th:nth-child(2){text-align:left}.temp{text-align:left;vertical-align:top;background:#f8f9fa;min-width:172px;font-size:17px}.temp small{display:block;font-size:11px;font-weight:400;margin-top:6px}.method{text-align:left;font-weight:550}.group>*{border-top:2px solid #dce0e5}tbody tr:hover td,tbody tr:hover .method{background:#f6f8fa}.foot{font-size:12px;color:#606975;margin-top:14px}details{margin-top:15px}code{font-size:12px;overflow-wrap:anywhere}footer{font-size:12px;color:#606975;margin:26px 0}@media(max-width:700px){main{padding:0 12px;margin:22px auto}.card{padding:16px}h1{font-size:25px}}@media print{body{background:white}main{padding:0}.card{break-inside:avoid;padding:10px}.scroll{overflow:visible}th,td{padding:6px;font-size:10px}}
</style><main><div class="eyebrow">LASP / DEVELOPMENT EVALUATION</div><h1>Temperature & T2A · 三方法指标表</h1><p class="muted">LASP · Persistence · BioEmu2 ｜ val split ｜ 原始指标，无归一化分数</p>
<div class="notice"><strong>中期匹配快照：BioEmu2 尚未完成。</strong><br>每个温度只使用三种方法均已完成、且包含全部四个 horizon 的相同轨迹与窗口。样本量列在各温度左侧；各温度覆盖范围不同。此页不代表完整 800 窗口评估或 30K 最终结果。</div>
<section class="card"><h2>01 / Temperature</h2><p class="muted">按真实温度分组；LASP 输入正确温度。所有方法使用相同窗口，先在蛋白内取均值，再对蛋白取均值。</p>'''
    document += table(temperature_rows, ENSEMBLE)
    document += '<p class="foot">↑ 越高越好，↓ 越低越好。RMSF MAE 单位 Å；W2 为本项目经验 Wasserstein 指标。LASP 为三个冻结中期 checkpoint（4000 / 5000 / 5000 steps）的均值。该表衡量分温度表现，不能证明未见温度泛化。</p></section><section class="card"><h2>02 / T2A</h2><p class="muted">每条轨迹合并四个 horizon 的样本，以同一条轨迹的真实 MD 系综定义参考状态。与上表使用相同轨迹集合。</p>'
    document += table(static_rows, STATIC)
    document += '<p class="foot">F-MAE：自由能 MAE，单位 kcal/mol。Coverage 使用 2 Å 阈值，k=8 参考状态。LASP / BioEmu2 各窗口 12 个样本；Persistence 固定在观测 anchor。数值来自本项目 T2A 实现，不等同于已发表 BioEmu benchmark。表格为点估计，不含置信区间。</p></section>'
    document += '<details><summary>数据来源与复核</summary><p>LASP：<code>mid_suite/matched8/seed{42,43,44}.json</code><br>Temperature BioEmu2：<code>t2a_bioemu2/bioemu2_T*.partial.json</code><br>T2A BioEmu2：已有保存样本，经当前共同轨迹集合重新评分；未重跑采样。<br>Persistence T2A：<code>t2a_baselines/anchors_8prot.json</code> 中相同轨迹记录。<br>完整数值和窗口身份：<a href="metric_tables.json">metric_tables.json</a>。</p></details>'
    document += f'<footer>生成于 {html.escape(snapshot["created_utc"])} · 静态快照；重新运行 scripts/render_lasp_metric_tables.py 可更新。</footer></main></html>'
    OUT.write_text(document)
    print(OUT)


if __name__ == '__main__':
    main()
