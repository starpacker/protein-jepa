"""Render completed stopped-best evaluations; refuse partial or mismatched inputs."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import html
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/standard_eval/stopped_best'
BASE = ROOT / 'results/lasp_full_v1'
TEMPS = (320, 348, 379, 413, 450)
ENSEMBLE = ('rmsf_mae', 'rmsf_corr', 'root_mean_w2', 'variance_w2', 'md_pca_w2', 'weak_contact_j', 'transient_contacts_j')
STATIC = ('coverage', 'free_energy_mae_kcal', 'population_mae', 'population_tv', 'population_jsd')
HIGH = {'rmsf_corr', 'weak_contact_j', 'transient_contacts_j', 'coverage', 'skill_rmsf', 'skill_position'}
LABEL = {'rmsf_mae':'RMSF MAE (Å)', 'rmsf_corr':'RMSF corr', 'root_mean_w2':'Root-mean W2', 'variance_w2':'Variance W2', 'md_pca_w2':'MD-PCA W2', 'weak_contact_j':'Weak contact J', 'transient_contacts_j':'Transient contact J', 'coverage':'Coverage', 'free_energy_mae_kcal':'F-MAE (kcal/mol)', 'population_mae':'Population MAE', 'population_tv':'Population TV', 'population_jsd':'Population JSD'}


def load(p):
    return json.loads(p.read_text())


def identity(r):
    return (r['protein'], r['traj'], float(r['temperature_k']), int(r['anchor_frame']), float(r['horizon']))


def balanced(rows, metric):
    groups = defaultdict(list)
    for r in rows:
        v = float(r[metric])
        if not np.isfinite(v):
            raise ValueError(f'nonfinite {metric}')
        groups[r['protein']].append(v)
    if not groups:
        raise ValueError('empty metric population')
    return float(np.mean([np.mean(v) for v in groups.values()]))


def summarize(lasp, persist, bio, metrics, **group):
    rows = []
    for method, sources in [('LASP', lasp), ('Persistence', [persist]), ('BioEmu2', [bio])]:
        vals = {k: [balanced(s, k) for s in sources] for k in metrics}
        rows.append(dict(group, method=method, values={k:float(np.mean(v)) for k,v in vals.items()}, seed_values=vals if method=='LASP' else None))
    return rows


def flatten(rows, field):
    return [dict(protein=r['protein'], **r[field]) for r in rows]


def table(rows, metrics, group_keys):
    headers = list(group_keys)+['Method']+[LABEL.get(k,k)+(' ↑' if k in HIGH else ' ↓') for k in metrics]
    text = '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in headers)+'</tr></thead><tbody>'
    for r in rows:
        text += '<tr>'+''.join('<th>'+html.escape(str(r[k]))+'</th>' for k in group_keys)+'<th>'+html.escape(r['method'])+'</th>'
        for k in metrics:
            spread = r.get('seed_values')
            extra = f'<small>{min(spread[k]):.4f}–{max(spread[k]):.4f}</small>' if spread else ''
            text += f'<td>{r["values"][k]:.4f}{extra}</td>'
        text += '</tr>'
    return text+'</tbody></table></div>'


def main():
    manifest = load(OUT/'manifest.json')
    assert manifest['status'] == 0 and manifest['noise_protocol_version']=='shared-window-noise-v1'
    selection = load(OUT/'selection.json')
    for ck in selection['checkpoints']:
        digest=hashlib.sha256()
        with (ROOT/ck['snapshot']).open('rb') as handle:
            for block in iter(lambda:handle.read(1024*1024),b''):
                digest.update(block)
        assert digest.hexdigest()==ck['sha256']
    suites = [load(OUT/f'dynamics_seed{s}_{i}.suite.json') for i,s in enumerate((42,43,44))]
    matched = [load(OUT/f'dynamics_seed{s}_{i}.matched8.json') for i,s in enumerate((42,43,44))]
    ensembles = [load(OUT/f'dynamics_seed{s}_{i}.ensemble.json') for i,s in enumerate((42,43,44))]
    for i,s in enumerate((42,43,44)):
        expected=f'results/standard_eval/stopped_best/dynamics_seed{s}/checkpoint_best.pt'
        for run in (suites[i],matched[i],ensembles[i]):
            assert run['args']['checkpoint']==expected
    bio = []
    for T in TEMPS:
        baseline = load(BASE/f't2a_bioemu2/bioemu2_T{T}.json')
        a = baseline['evaluation_args']
        for k,v in {'n_samples':12,'history_frames':8,'max_residues':96,'max_proteins':8,'max_anchors_per_trajectory':1,'seed':0,'horizons_ns':[1.0,4.0,16.0,64.0],'temperatures':[float(T)]}.items():
            assert a[k]==v
        assert baseline['n_windows']==160 and baseline['n_skipped']==0
        bio.extend(baseline['per_window'])
    keys = {identity(r) for r in bio}
    assert len(keys)==len(bio)==800
    protein_sets=[{r['protein'] for r in bio if r['temperature_k']==T} for T in TEMPS]
    assert all(p==protein_sets[0] for p in protein_sets)
    for run in matched:
        assert run['n_windows']==800 and len(run['per_window'])==800
        assert {identity(r) for r in run['per_window']}==keys
        assert run['noise_protocol']['version']=='shared-window-noise-v1'
        for k,v in {'n_samples':12,'max_residues':96,'context_frames':8,'anchors_per_trajectory':1,'proteins_per_temp':8,'k_states':8,'seed':0}.items():
            assert run['args'][k]==v
        assert run['args']['horizons']==[1,4,16,64] and run['args']['modes']==['correct']
    full_ids = None
    for run in suites:
        rows = run['per_window']
        ids = {(identity(r),r['mode']) for r in rows}
        assert run['n_windows']==len(rows)==len(ids)==7200
        assert run['noise_protocol']['version']=='shared-window-noise-v1'
        if full_ids is None: full_ids=ids
        assert ids==full_ids
        for T in TEMPS:
            for mode in ('correct','wrong','none'):
                cell = [r for r in rows if r['temperature_k']==T and r['mode']==mode]
                assert len(cell)==480 and len({r['protein'] for r in cell})==12
                expected = T if mode=='correct' else (450 if T==320 else 320) if mode=='wrong' else None
                assert all(r['temperature_used_k']==expected for r in cell)
        same348={mode:{identity(r):r for r in rows if r['temperature_k']==348 and r['mode']==mode} for mode in ('correct','none')}
        for key,row in same348['correct'].items():
            for metric in ENSEMBLE:
                assert np.isclose(row['model'][metric],same348['none'][key]['model'][metric],rtol=1e-6,atol=1e-6)
    assert all(e['n_windows']==len(e['per_window'])==200 for e in ensembles)
    ensemble_ids={identity(r) for r in ensembles[0]['per_window']}
    assert len(ensemble_ids)==200
    assert all({identity(r) for r in e['per_window']}==ensemble_ids for e in ensembles)
    bio_static = load(BASE/'t2a_bioemu2/bioemu2_t2a.json')
    assert bio_static['completeness']=={'complete':True,'expected_windows':800,'actual_windows':800}
    anchors = load(BASE/'t2a_baselines/anchors_8prot.json')
    for args in (bio_static['args'],anchors['args']):
        for k,v in {'max_frames':512,'k_states':8,'seed':0,'max_residues':96,'context_frames':8,'anchors_per_trajectory':1,'proteins_per_temp':8}.items():
            assert args[k]==v
    temp_rows, static_rows, horizon_rows, protein_rows, conditioning = [],[],[],[],[]
    bio_validation = load(OUT/'bioemu2_validation.json')
    assert bio_validation['masks_equal_reference'] and bio_validation['actual_windows']==800
    validation = {'bioemu2_saved_sample_validation':bio_validation,'complete':True,'window_count':800,'distinct_paired_proteins':len({r['protein'] for r in bio}),'paired_trajectories':len({r['traj'] for r in bio}),'suite_windows_per_seed':7200,'ensemble_windows_per_seed':200,'identities':sorted(keys),'checkpoints':selection,'notes':['LASP suite checks its window masks equal reference masks before T2A scoring. BioEmu2 saved-sample scorer applies the identical check.', 'Reference uses max_frames=512 stride construction, k=8 and seed=0; actual frame count can exceed 512 slightly due to integer stride.', 'All paired static pools contain 48 model samples per trajectory (4 horizons × 12); persistence repeats its anchor.', 'Paired temperature/protein/horizon comparisons include all seven ensemble metrics; paired T2A includes all five static metrics. BioEmu2 baseline files do not provide endpoint position metrics, so position is reported only in the full LASP conditioning sweep.', 'Metrics compare generated endpoint ensembles against the empirical future-path reference for ensemble terms; this is not a matched-time ensemble of independent MD endpoints.']}
    for T in TEMPS:
        b = [r for r in bio if r['temperature_k']==T]
        ls = [[r for r in d['per_window'] if r['temperature_k']==T] for d in matched]
        assert len(b)==160 and len({r['protein'] for r in b})==8 and len({r['traj'] for r in b})==40
        # The baseline and LASP persistence terms must reproduce the same truth/masks.
        bykey = {identity(r):r for r in b}
        for rows in ls:
            for r in rows:
                for k in ENSEMBLE:
                    assert np.isclose(r['persist'][k],bykey[identity(r)]['persist'][k],rtol=1e-5,atol=1e-5)
        temp_rows += summarize([flatten(r,'model') for r in ls],flatten(b,'persist'),flatten(b,'bioemu2'),ENSEMBLE,temperature=T)
        st = [d['t2a_per_protein'][str(T)] for d in matched]
        bs = bio_static['per_protein'][str(T)]
        ps = anchors['per_protein']['persistence'][str(T)]
        tids = {r['traj'] for r in b}
        for rr in st+[bs,ps]:
            assert len(rr)==40 and {r['trajectory'] for r in rr}==tids
        for rr in st+[bs]:
            assert all(r['n_model_samples']==48 for r in rr)
        reference_meta={r['trajectory']:(r['n_ref_frames'],r['n_states']) for r in bs}
        for rr in st+[ps]:
            assert {r['trajectory']:(r['n_ref_frames'],r['n_states']) for r in rr}==reference_meta
        assert all(r['n_model_samples']==4 for r in ps)
        static_rows += summarize(st,ps,bs,STATIC,temperature=T)
        for p in sorted({r['protein'] for r in bs}):
            protein_rows += summarize([[r for r in rr if r['protein']==p] for rr in st],[r for r in ps if r['protein']==p],[r for r in bs if r['protein']==p],STATIC,temperature=T,protein=p)
        for h in (1,4,16,64):
            bb=[r for r in b if r['horizon']==h]
            assert len(bb)==40 and len({r['protein'] for r in bb})==8
            ll=[[r for r in rows if r['horizon']==h] for rows in ls]
            horizon_rows += summarize([flatten(r,'model') for r in ll],flatten(bb,'persist'),flatten(bb,'bioemu2'),ENSEMBLE,temperature=T,horizon=h)
        for p in sorted({r['protein'] for r in b}):
            bb=[r for r in b if r['protein']==p]
            ll=[[r for r in rows if r['protein']==p] for rows in ls]
            protein_rows += summarize([flatten(r,'model') for r in ll],flatten(bb,'persist'),flatten(bb,'bioemu2'),ENSEMBLE,temperature=T,protein=p)
        for mode in ('correct','wrong','none'):
            ll=[[r for r in d['per_window'] if r['temperature_k']==T and r['mode']==mode] for d in suites]
            metrics=ENSEMBLE+('kabsch_ca_rmsd','aligned_ca_rmsd','skill_rmsf','skill_position')
            vals=defaultdict(list)
            for rows in ll:
                expanded=[dict(protein=r['protein'],**r['model'],**r['position'],skill_rmsf=r['persist']['rmsf_mae']-r['model']['rmsf_mae'],skill_position=r['position']['persist_aligned_ca_rmsd']-r['position']['aligned_ca_rmsd']) for r in rows]
                for k in metrics: vals[k].append(balanced(expanded,k))
            conditioning.append(dict(temperature=T,method=mode,values={k:float(np.mean(v)) for k,v in vals.items()},seed_values=dict(vals)))
    full_static = []
    full_anchors = load(OUT/'t2a_anchors.json')
    for T in TEMPS:
        for rr in [d['t2a_per_protein'][str(T)] for d in suites]:
            assert len(rr)==60 and len({r['protein'] for r in rr})==12
            assert all(r['n_model_samples']==96 for r in rr)
        for method, sources in [('LASP',[d['t2a_per_protein'][str(T)] for d in suites]), ('Persistence',[full_anchors['per_protein']['persistence'][str(T)]])]:
            vals = {k:[balanced(rr,k) for rr in sources] for k in STATIC}
            full_static.append(dict(temperature=T,method=method,values={k:float(np.mean(v)) for k,v in vals.items()},seed_values=vals if method=='LASP' else None))
    ensemble_rows = []
    for method, sources in [('LASP',[flatten(d['per_window'],'lasp') for d in ensembles]), ('Persistence',[flatten(ensembles[0]['per_window'],'persist')])]:
        vals={k:[balanced(rr,k) for rr in sources] for k in ENSEMBLE}
        ensemble_rows.append(dict(method=method,values={k:float(np.mean(v)) for k,v in vals.items()},seed_values=vals if method=='LASP' else None))
    historical=load(BASE.parent/'lasp_v19/bioemu2_complete.json')
    assert {identity(r) for r in historical['per_window']}==ensemble_ids
    assert historical['n_samples']==12 and historical['n_windows']==200
    for k,v in {'history_frames':8,'max_residues':96,'horizons_ns':[1.0,4.0,16.0,64.0],'cutoff':8.0,'pca_k':3}.items():
        assert historical['evaluation_args'][k]==v
    historic_index={identity(r):r for r in historical['per_window']}
    for r in ensembles[0]['per_window']:
        for k in ENSEMBLE:
            assert np.isclose(r['persist'][k],historic_index[identity(r)]['persist'][k],rtol=1e-5,atol=1e-5)
    vals={k:balanced(flatten(historical['per_window'],'bioemu2'),k) for k in ENSEMBLE}
    ensemble_rows.append(dict(method='BioEmu2',values=vals,seed_values=None))
    geometry = []
    for s,d in zip((42,43,44),ensembles):
        rr=flatten(d['per_window'],'geometry_vs_anchor')
        geometry.append(dict(seed=s,**{k:balanced(rr,k) for k in ('ca_bond','ca_clash_fraction','ca_angle_cos_mae')}))
    # Per-temperature, horizon and protein signed gaps; positive always means LASP worse.
    def gaps(rows,group):
        out=[]
        for i in range(0,len(rows),3):
            l,p,b=rows[i:i+3]
            for k,v in l['values'].items():
                sign=-1 if k in HIGH else 1
                out.append(dict({g:l[g] for g in group},metric=k,lasp=v,bioemu2=b['values'][k],persistence=p['values'][k],gap=sign*(v-b['values'][k]),seed_gaps=[sign*(x-b['values'][k]) for x in l['seed_values'][k]]))
        return out
    snapshot=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='completed_stopped_best_validation',validation=validation,temperature=temp_rows,t2a=static_rows,full_suite_t2a=full_static,ensemble200=ensemble_rows,ensemble200_bioemu2_source=str(BASE.parent/'lasp_v19/bioemu2_complete.json'),ensemble200_geometry=geometry,by_horizon=horizon_rows,by_protein=protein_rows,t3=conditioning,gaps=dict(temperature=gaps(temp_rows,['temperature']),t2a=gaps(static_rows,['temperature']),horizon=gaps(horizon_rows,['temperature','horizon']),protein=gaps(protein_rows,['temperature','protein'])),ensemble_sources=[str(OUT/f'dynamics_seed{s}_{i}.ensemble.json') for i,s in enumerate((42,43,44))])
    # Serialize before publishing to reject any nonfinite derived values.
    serialized=json.dumps(snapshot,indent=2,allow_nan=False)
    (OUT/'metric_tables.json').write_text(serialized)
    analysis = ['# Stopped-best full evaluation and BioEmu2 gaps', '', 'Frozen best checkpoints: seed42 7000 / seed43 9000 / seed44 7000 steps. All results are validation diagnostics after validation-based checkpoint selection, not an independent test. Three training seeds are point estimates with observed ranges, not confidence intervals.', '', '## Protocol and completeness', '', 'All three full suites contain 7200 unique window/arm rows, all ensemble evaluations contain 200 windows, and all matched runs contain the same 800 unique BioEmu2 windows. Each temperature uses 8 proteins, 40 trajectories and 160 windows (4 horizons; 12 samples/window). Means weight proteins equally within each temperature. T2A pools 48 samples/trajectory and uses the same masked MD reference, k=8, seed=0. All exact identities are in metric_tables.json. Persistence ensemble terms match between runners to numerical tolerance.', '', 'Positive signed gaps below mean LASP is worse; negative means LASP is better. Differences in unlike units must not be ranked together.', '', '## Per-temperature paired differences', '', '| T | Metric | LASP | BioEmu2 | Persistence | Signed gap | Seed gap min–max |','|---|---|---|---|---|---|---|']
    for row in snapshot['gaps']['temperature']+snapshot['gaps']['t2a']:
        analysis.append(f'| {row["temperature"]} | {row["metric"]} | {row["lasp"]:.4f} | {row["bioemu2"]:.4f} | {row["persistence"]:.4f} | {row["gap"]:+.4f} | {min(row["seed_gaps"]):+.4f}–{max(row["seed_gaps"]):+.4f} |')
    analysis += ['', 'F-MAE is this repository’s relative free-energy error with absent model states assigned a finite penalty equal to the reference free-energy range. It can behave differently from coverage and population TV/JSD; a lower F-MAE alone does not establish better support or physically correct equilibrium free energies. Population MAE and TV are linearly related for fixed k=8 (TV = 4 × MAE), so they are not independent successes.', '', '## Metric-level outcome', '']
    for family in ('temperature','t2a'):
        allg=snapshot['gaps'][family]
        for metric in dict.fromkeys(r['metric'] for r in allg):
            rr=[r for r in allg if r['metric']==metric]
            worst=max(rr,key=lambda r:r['gap'])
            best=min(rr,key=lambda r:r['gap'])
            analysis.append(f'{metric}: LASP better at {sum(r["gap"]<0 for r in rr)}/5 temperatures; seed-consistent deficits at {sum(min(r["seed_gaps"])>0 for r in rr)}/5. Equal-temperature mean signed gap {np.mean([r["gap"] for r in rr]):+.4f}; largest signed gap {worst["gap"]:+.4f} at {worst["temperature"]} K, smallest {best["gap"]:+.4f} at {best["temperature"]} K.')
    analysis += ['', '## Horizon and protein concentration', '', 'Horizon breakdowns below apply to window ensemble metrics. T2A combines all four horizons into one trajectory pool, so no per-horizon equilibrium inference is available from these files. Protein-level T2A differences are retained in metric_tables.json and the HTML protein section.', '']
    for metric in ('rmsf_mae','md_pca_w2','transient_contacts_j','coverage','free_energy_mae_kcal','population_tv'):
        rr=[r for r in snapshot['gaps']['horizon'] if r['metric']==metric]
        if rr:
            analysis.append(f'**{metric}**, signed gaps by horizon (equal temperature weighting): '+ '; '.join(f'h={h}: {np.mean([r["gap"] for r in rr if r["horizon"]==h]):+.4f}' for h in (1,4,16,64))+'.')
        pp=sorted([r for r in snapshot['gaps']['protein'] if r['metric']==metric],key=lambda r:r['gap'],reverse=True)
        analysis.append('Largest protein/temperature deficits: '+ '; '.join(f'{r["protein"]} at {r["temperature"]} K ({r["gap"]:+.4f})' for r in pp[:5])+'.')
        analysis.append(f'LASP better in {sum(r["gap"]<0 for r in pp)}/{len(pp)} protein/temperature cells; worse in {sum(r["gap"]>0 for r in pp)}/{len(pp)}. These cells can repeat proteins and are not independent replicates.')
    analysis += ['', '## Supplemental standard 200-window comparison', '', 'Historical BioEmu2 reference results/lasp_v19/bioemu2_complete.json matches all 200 identities exactly, with 12 samples/window, the same context/residue/horizon settings and matching persistence metrics. This population is separate from the primary 800-window temperature/T2A analysis.', '', '| Metric | LASP mean | Seed min–max | Persistence | BioEmu2 |','|---|---|---|---|---|']
    for k in ENSEMBLE:
        l,p,b=ensemble_rows
        v=l['seed_values'][k]
        analysis.append(f'| {k} | {l["values"][k]:.4f} | {min(v):.4f}–{max(v):.4f} | {p["values"][k]:.4f} | {b["values"][k]:.4f} |')
    analysis += ['', '## Full T3 conditioning utility', '', 'These differences use 12 proteins per temperature and shared sampling noise. Positive wrong-minus-correct error indicates a useful correct-temperature input for that error metric.', '']
    for T in TEMPS:
        cells={r['method']:r for r in conditioning if r['temperature']==T}
        for metric in ('rmsf_mae','md_pca_w2','aligned_ca_rmsd'):
            c,w,n=(cells[x] for x in ('correct','wrong','none'))
            penalties=[x-y for x,y in zip(w['seed_values'][metric],c['seed_values'][metric])]
            analysis.append(f'{T} K {metric}: correct {c["values"][metric]:.4f}, wrong {w["values"][metric]:.4f}, fixed-348 {n["values"][metric]:.4f}; wrong-minus-correct {np.mean(penalties):+.4f}, seed range {min(penalties):+.4f}–{max(penalties):+.4f}.')
    priorities=[]
    for metric in ('rmsf_mae','md_pca_w2','transient_contacts_j','coverage','free_energy_mae_kcal','population_tv'):
        rr=[r for family in ('temperature','t2a') for r in snapshot['gaps'][family] if r['metric']==metric]
        n=sum(min(r['seed_gaps'])>0 for r in rr)
        if n:
            w=max(rr,key=lambda r:r['gap'])
            priorities.append(f'- {metric}: {n}/5 temperatures have a deficit in every seed; start the diagnostic at {w["temperature"]} K (signed gap {w["gap"]:+.4f}, LASP {w["lasp"]:.4f}, BioEmu2 {w["bioemu2"]:.4f}).')
    analysis += ['', '## Concrete diagnostic priorities', '']+priorities
    if not priorities:
        analysis.append('No selected headline metric shows a temperature with a deficit in every seed. Inspect mixed signs and tradeoffs rather than claiming uniform superiority.')
    analysis += ['', '## Interpretation and next steps', '', '1. Select the largest consistent deficits by metric and temperature from the paired tables. Treat a deficit as seed-consistent only when all three signed seed gaps have the same positive sign. Use the protein and horizon tables to distinguish widespread deficits from a few difficult proteins; do not infer mechanism from a metric difference alone.', '2. Prioritize a frozen-model diagnostic of state occupancy: compare predicted spread, state coverage, free-energy error and population TV/JSD jointly against persistence and BioEmu2. A coverage gain with worse free energies indicates occupancy calibration needs work; it does not establish that more noise is the solution. Use the same reference states and sample budget.', '3. For dynamics, inspect the h=1/4 versus h=16/64 deficit and geometry/position tradeoff before proposing a component change. Ensemble amplitude improvements can coexist with worse endpoint accuracy. No further training was launched for this report.', '4. Use the full shared-noise T3 table to judge whether correct temperature improves each metric over wrong and fixed-348 controls. At 348 K, correct and none are the same feature by construction. These temperatures occurred in training: this is a conditioning diagnostic, not unseen-temperature generalization.', '5. Once a component-specific hypothesis is supported, propose a small controlled ablation with a prespecified primary metric and matched window set, then reserve an independent held-out evaluation before any final performance claim.', '', 'BioEmu2 has no explicit temperature input in this installed sampling API; only LASP has conditioning arms. The comparisons use this repository’s evaluation implementation and do not reproduce a published BioEmu benchmark. Three training seeds do not establish population confidence bounds. The matched baseline uses the same eight protein IDs at all five temperatures (verified); trajectories and reference distributions differ across temperatures, so a temperature trend alone is not a causal conditioning effect. Use the shared-noise arms for that question.', '']
    analysis += ['The dedicated next-step plan is [next_step_plan.md](next_step_plan.md); the diagnostic notes above describe evaluation gaps, while the plan gives the stability-first execution order.','']
    analysis += ['The separate gradient audit is available at [gradient_failure_analysis.md](../../lasp_full_v1/gradient_failure_analysis.md) with [machine-readable statistics](../../lasp_full_v1/gradient_failure_stats.json). Its unresolved caveats include norm overflow versus actual nonfinite gradient entries and the exact failing horizon under sparse logging; the collective branch skips configured clipping. Endpoint samples are scored against a path mixture, and training h=1 plug-in target RMSF is near zero. These caveats limit causal interpretation of the evaluation gaps. See [the coordinator’s next-step plan](next_step_plan.md) for the synthesis.','']
    (OUT/'gap_analysis.md').write_text('\n'.join(analysis))
    gap_html=[]
    for family in ('temperature','t2a'):
        rows=snapshot['gaps'][family]
        for metric in dict.fromkeys(r['metric'] for r in rows):
            rr=[r for r in rows if r['metric']==metric]
            gap_html.append(f'<li>{html.escape(LABEL.get(metric,metric))}: LASP better at {sum(r["gap"]<0 for r in rr)}/5 temperatures; all seeds worse at {sum(min(r["seed_gaps"])>0 for r in rr)}/5. Equal-temperature mean signed gap {np.mean([r["gap"] for r in rr]):+.4f}.</li>')
    css='body{margin:0;background:#f4f5f7;color:#20252c;font:15px/1.6 system-ui,sans-serif}main{max-width:1600px;margin:35px auto;padding:0 24px}section{background:white;border:1px solid #dce0e5;border-radius:10px;padding:22px;margin:22px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap;font-variant-numeric:tabular-nums}th,td{padding:10px 12px;border-bottom:1px solid #e8ebee;text-align:right}th{text-align:left}thead{background:#f1f3f5}small{display:block;font-size:10px;color:#606975}p{max-width:1100px}summary{cursor:pointer}a{color:#234e80}'
    doc=f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LASP stopped-best full evaluation</title><style>{css}</style><main><h1>LASP · Stopped-best full evaluation</h1><p>Frozen seed42 / 43 / 44 best checkpoints: 7000 / 9000 / 7000 steps. Full evaluation complete. Validation diagnostics; checkpoint selection used validation loss.</p><p>LASP cells show the three-seed mean; the small line is observed seed min–max, not a confidence interval. All paired tables are protein-balanced. <a href="metric_tables.json">All numerical data and identities</a> · <a href="gap_analysis.md">Gap analysis and next steps</a> · <a href="../../lasp_full_v1/metric_tables.html">Previous interim snapshot</a></p>'
    doc+=f'<p>Paired population: {validation["distinct_paired_proteins"]} distinct proteins and {validation["paired_trajectories"]} trajectories across all temperatures. Window/trajectory counts are not independent protein sample sizes.</p>'
    doc+='<section><h2>Paired outcome and next steps</h2><p>Signed gap is LASP minus BioEmu2 for errors, BioEmu2 minus LASP for scores: positive means LASP worse. Metrics retain their own units; do not compare gap magnitudes across metrics.</p><ul>'+''.join(gap_html)+'</ul><p>Next: examine seed-consistent deficits by protein and horizon; diagnose reference-state occupancy jointly with spread and geometry; use the shared-noise conditioning table before proposing a small component ablation. See the linked gap analysis for quantitative priorities. No further training was launched.</p></section>'
    doc+='<section><h2>01 / Matched temperature comparison</h2><p>800 exact common windows: 160 per temperature, 8 proteins × 5 replicas × 1 anchor × 4 horizons, 12 samples/window. LASP receives correct temperature. Metrics use the repository endpoint-vs-future-path ensemble definitions; no BioEmu2 endpoint position metric is available.</p>'+table(temp_rows,ENSEMBLE,['temperature'])+'</section>'
    seed_detail = []
    for r in temp_rows + static_rows:
        if r['method']=='LASP':
            for i,s in enumerate((42,43,44)):
                seed_detail.append(dict(temperature=r['temperature'],method=f'LASP seed{s}',values={k:v[i] for k,v in r['seed_values'].items()}))
    doc+='<section><h2>01b / Individual checkpoint values</h2><details><summary>Expand three-seed ensemble and T2A details</summary>'+table([r for r in seed_detail if 'rmsf_mae' in r['values']],ENSEMBLE,['temperature'])+table([r for r in seed_detail if 'coverage' in r['values']],STATIC,['temperature'])+'</details></section>'
    doc+='<section><h2>02 / Matched static ensemble (T2A)</h2><p>40 trajectories per temperature. Four horizons pool to 48 samples per trajectory. Coverage threshold 2 Å; k=8 reference states. Same masked MD references and pooling; persistence repeats the observed anchor. F-MAE uses the repository finite missing-state penalty (reference free-energy range), so read it jointly with coverage and population TV/JSD.</p>'+table(static_rows,STATIC,['temperature'])+'</section>'
    doc+='<section><h2>03 / Full T3 conditioning sweep</h2><p>7200 rows per seed: 5 temperatures × 12 proteins × 5 replicas × 2 anchors × 4 horizons × 3 arms. Each cell contains 480 windows and 60 trajectories. Shared window noise. Wrong temperature is 320 K except at true 320 K, where it is 450 K. None is a fixed-348 K feature control. This is not unseen-temperature generalization.</p>'+table(conditioning,ENSEMBLE+('kabsch_ca_rmsd','aligned_ca_rmsd','skill_rmsf','skill_position'),['temperature'])+'</section>'
    doc+='<section><h2>03b / Full-suite static ensemble</h2><p>12 proteins and 60 trajectories per temperature, 2 anchors × 4 horizons × 12 samples = 96 LASP samples per trajectory. Persistence uses the same anchors. This broader pool is separate from the matched BioEmu2 table.</p>'+table(full_static,STATIC,['temperature'])+'</section>'
    doc+='<section><h2>03c / Harmonized 200-window ensemble</h2><p>Separate standard ensemble pass. These windows and sampling draws differ from the 800-window temperature comparison. BioEmu2 reuses the historical complete 200-window reference; exact identities and persistence metrics agree, with 12 samples, 96 residues, 8 context frames and the same horizons. No new BioEmu2 sampling.</p>'+table(ensemble_rows,ENSEMBLE,[])+'<details><summary>Geometry relative to anchor, by seed</summary><pre>'+html.escape(json.dumps(geometry,indent=2))+'</pre></details></section>'
    doc+='<section><h2>04 / Paired horizon details</h2><p>Each temperature/horizon cell: 8 proteins, 40 trajectories, 40 windows. Horizon labels follow the evaluation runner (1 / 4 / 16 / 64 ns at the configured 1000 ps frame spacing); T2A pools all four and has no separate per-horizon equilibrium estimate.</p>'+table(horizon_rows,ENSEMBLE,['temperature','horizon'])+'</section>'
    doc+='<section><h2>05 / Paired protein details</h2><p>Each protein/temperature cell: 5 trajectories and 20 windows. These matched cells use the same eight proteins across temperatures.</p><details><summary>Expand all protein rows</summary>'+table([r for r in protein_rows if 'rmsf_mae' in r['values']],ENSEMBLE,['temperature','protein'])+table([r for r in protein_rows if 'coverage' in r['values']],STATIC,['temperature','protein'])+'</details></section>'
    doc+='<section><h2>Failure analysis and interpretation limits</h2><p><a href="../../lasp_full_v1/gradient_failure_analysis.md">Gradient failure analysis</a> · <a href="../../lasp_full_v1/gradient_failure_stats.json">Gradient statistics</a> · <a href="next_step_plan.md">Next-step plan</a></p><p>The gradient audit leaves norm overflow versus nonfinite entries and the exact failing horizon unresolved; the collective branch skips configured clipping. Endpoint samples are scored against path mixtures, and training h=1 plug-in target RMSF is near zero. These details matter when interpreting short-horizon gains or failure causes. Three dynamics seeds share one representation and do not measure representation-training variance.</p></section>'
    doc+=f'<footer>Generated {snapshot["created_utc"]}. No new training. Original ensemble and suite outputs remain available alongside this report. See <a href="manifest.json">completion manifest</a>, <a href="selection.json">checkpoint selection</a> and <a href="bioemu2_validation.json">saved-sample validation</a> for provenance.</footer></main></html>'
    (OUT/'metric_tables.html').write_text(doc)
    print('Rendered',OUT/'metric_tables.html',flush=True)


if __name__=='__main__':
    main()
