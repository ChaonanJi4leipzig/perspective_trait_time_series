'Needs completed frozen temporal/spatial scores; writes pooled five-seed figure metrics.\nInputs:paths.py;outputs:WORK,never upstream;run IDs:cap2048_v3_nested11_20261108,cap2048_spatial_gsingle_20261112,cap2048_production_nn_temporal_20261112'
from __future__ import annotations
import paths,sys
if __name__=="__main__":
 from make_runtime import cli
 cli(__doc__)
sys.path.insert(0,str(paths.MODEL_CODE))
from make_site_support import *
SCORES = ROOT.parent / 'figures_add_samples/final_pipeline_20261105/training_2048/harness/final_pipeline_20261105/training_2048/scores'
FORMAL_RUN = 'cap2048_v3_nested11_20261108'
SPATIAL_RUN = 'cap2048_spatial_gsingle_20261112'
ALIASES = dict(zip(TRAITS, ['LMA', 'N', 'LAI', 'C', 'Chl', 'EWT', 'Car', 'Lignin', 'Cellulose', 'Fibre', 'NSC']))

def normalize(df):
    d = df.copy()
    d = d.rename(columns={'nrmse_p95_p5': 'nrmse_P95_P05', 'nrmse_p95_p05': 'nrmse_P95_P05', 'n_test': 'n', 'r_squared': 'r2', 'R2': 'r2', 'trait_name': 'trait'})
    reverse = {v: k for k, v in ALIASES.items()}
    reverse.update(CHL=TRAITS[4], N=TRAITS[1], Carotenoids=TRAITS[6], Fiber=TRAITS[9])
    d['trait'] = d.trait.map(lambda x: reverse.get(x, x))
    return d

def primary(d, scope=None):
    d = normalize(d)
    for col, val in [('population', 'known_site'), ('support', 'W7_covered'), ('dimension', 'pooled')]:
        if col in d:
            d = d[d[col].eq(val)]
    if 'fold' in d:
        pooled = d.fold.astype(str).isin(['pooled', 'all_folds', 'all'])
        if pooled.any():
            d = d[pooled]
    if scope is not None:
        if 'scope' in d:
            d = d[d.scope.eq(scope)]
        else:
            d['scope'] = scope
    d = d[d.trait.isin(TRAITS)].copy()
    d['seed'] = pd.to_numeric(d.seed, errors='raise').astype(int)
    return d

def main():
    assert all(((SCORES / f).is_file() for f in ['DONE.json', 'USER_REPORT_VALIDATION.json'])), 'Formal score gate incomplete'
    sd = ROOT / 'spatial_gsingle'
    assert (sd / 'SPATIAL_GSINGLE_RESULTS.md').is_file(), 'G_single results gate incomplete'
    spatial_text = (sd / 'SPATIAL_GSINGLE_RESULTS.md').read_text()
    assert SPATIAL_RUN in spatial_text, 'G_single report must identify the selected run'
    markers = {n: read(SCORES / n) for n in ['DONE.json', 'USER_REPORT_VALIDATION.json']}
    validation = markers['USER_REPORT_VALIDATION.json']
    assert validation['status'] == 'pass' and validation['scorer_receipt_hashes'] == 'pass' and (validation['code_pins'] == 'pass'), 'Formal report gates failed'
    assert validation['freeze_sha256'] == markers['DONE.json']['identity']['freeze_sha256'], 'Formal freeze identity differs between receipts'
    for k in ['passed', 'validated', 'success']:
        if k in validation:
            assert validation[k] is True, ('Formal report validation failed', k)
    done = markers['DONE.json']
    assert sha(SCORES / 'metrics.csv') == done['files']['metrics.csv']
    raw = primary(pd.read_csv(SCORES / 'metrics.csv'))
    f = raw[raw.model.eq('NN') & raw.scope.eq('temporal') | raw.model.isin(['site_mean', 'site_season_mean'])].copy()
    f['run_id'] = FORMAL_RUN
    f['source_file'] = str(SCORES / 'metrics.csv')
    f['source_sha256'] = sha(SCORES / 'metrics.csv')
    candidates = []
    paths = [ROOT / 'source_data/spatial_Gsingle_per_seed.csv'] if (ROOT / 'evidence/SPATIAL_METRICS_READY.json').exists() else sorted(sd.rglob('*.csv'))
    for p in paths:
        if 'metric' not in p.name.lower() and p != ROOT / 'source_data/spatial_Gsingle_per_seed.csv':
            continue
        cols = set(pd.read_csv(p, nrows=0).columns)
        if not {'seed', 'trait', 'r2'}.issubset(cols) or not any((k in cols for k in ['nrmse_P95_P05', 'nrmse_p95_p5', 'nrmse_p95_p05'])):
            continue
        d = primary(pd.read_csv(p), 'spatial')
        if 'run_id' in d:
            d = d[d.run_id.eq(SPATIAL_RUN)]
        if 'model' in d:
            d = d[d.model.isin(['NN', 'NN_G_single', 'NN_gsingle', 'G_single'])]
        if len(d) == 55 and set(d.seed) == set(SEEDS) and (set(d.trait) == set(TRAITS)) and ('n' in d) and d.n.eq(49525).all():
            candidates.append((p, d))
    assert candidates, 'No published G_single per-seed pooled primary metric table found; schema needs inspection'
    candidates.sort(key=lambda v: (v[0].name != 'metrics.csv', str(v[0])))
    p, g = candidates[0]
    g = g.copy()
    g['model'] = 'NN'
    g['run_id'] = SPATIAL_RUN
    g['source_file'] = g['source_file'] if 'source_file' in g else str(p)
    g['source_sha256'] = g['source_sha256'] if 'source_sha256' in g else sha(p)
    allrows = pd.concat([f, g], ignore_index=True)
    assert len(allrows) == 330
    rows = []
    from contracts import CLASSES
    for (scope, model, trait), group in allrows.groupby(['scope', 'model', 'trait'], sort=False):
        assert len(group) == 5 and set(group.seed) == set(SEEDS)
        assert group.n.nunique() == 1 and int(group.n.iloc[0]) == {'temporal': 48656, 'spatial': 49525}[scope]
        assert group.run_id.nunique() == 1
        for metric in ['r2', 'nrmse_P95_P05']:
            v = group[metric].to_numpy(float)
            assert np.isfinite(v).all()
            rows.append(dict(trait=ALIASES[trait], source_trait=trait, trait_class=CLASSES[TRAITS.index(trait)], scope=scope, model=model, metric=metric, mean=float(v.mean()), seed_min=float(v.min()), seed_max=float(v.max()), n=int(group.n.iloc[0]), n_seeds=5, run_id=group.run_id.iloc[0], source_file=group.source_file.iloc[0], source_sha256=group.source_sha256.iloc[0]))
    pd.DataFrame(rows).to_csv(ROOT / 'source_data/Fig3_panel_a_metrics.csv', index=False)
    allrows.to_csv(ROOT / 'source_data/Fig3_panel_a_per_seed.csv', index=False)
    macros = []
    for (scope, model), group in allrows.groupby(['scope', 'model'], sort=False):
        for label, traits in [('All11', TRAITS)] + [(f'Class{cl}', [t for t, c in zip(TRAITS, CLASSES) if c == cl]) for cl in [1, 2, 3]]:
            part = group[group.trait.isin(traits)]
            v = part.groupby('seed')[['r2', 'nrmse_P95_P05']].mean()
            assert len(v) == 5 and part.trait.nunique() == len(traits)
            for metric in v:
                macros.append(dict(scope=scope, model=model, trait=label, metric=metric, mean=v[metric].mean(), seed_min=v[metric].min(), seed_max=v[metric].max(), n_traits=len(traits), run_id=part.run_id.iloc[0]))
    pd.DataFrame(macros).to_csv(ROOT / 'source_data/Fig3_class_means.csv', index=False)
    if (SCORES / 'paired_cluster_CI.csv').is_file():
        assert sha(SCORES / 'paired_cluster_CI.csv') == done['files']['paired_cluster_CI.csv'], 'Formal CI receipt hash mismatch'
        ci = pd.read_csv(SCORES / 'paired_cluster_CI.csv')
        ci = ci[ci.scope.eq('temporal') & ci.contrast.eq('NN-C') & ci.panel.eq('W7_covered') & ci.fold.astype(str).eq('pooled')].copy()
        if len(ci):
            assert np.allclose(ci.ci_level, 0.95)
            for metric, col in [('r2', 'delta_r2'), ('nrmse_P95_P05', 'delta_nrmse')]:
                points = []
                for row in ci.itertuples():
                    traits = [t for t, c in zip(TRAITS, CLASSES) if 'Class' + str(c) == row.trait] if row.trait.startswith('Class') else [row.trait]
                    part = raw[raw.scope.eq('temporal') & raw.trait.isin(traits)]
                    points.append(part.loc[part.model.eq('NN'), metric].mean() - part.loc[part.model.eq('C'), metric].mean())
                ci[col] = points
            ci['run_id'] = FORMAL_RUN
            ci['source_file'] = str(SCORES / 'paired_cluster_CI.csv')
            ci['source_sha256'] = sha(SCORES / 'paired_cluster_CI.csv')
            ci.to_csv(ROOT / 'source_data/temporal_NN_minus_C_95CI.csv', index=False)
    save(ROOT / 'evidence/PANEL_A_DONE.json', dict(formal_run=FORMAL_RUN, spatial_run=SPATIAL_RUN, formal_markers=markers, formal_metrics_sha256=sha(SCORES / 'metrics.csv'), spatial_results_sha256=sha(sd / 'SPATIAL_GSINGLE_RESULTS.md'), spatial_metrics_file=str(p), spatial_metrics_sha256=sha(p), baseline_source=FORMAL_RUN, seed_estimand='Mean of per-seed pooled metrics, not prediction ensemble'))
if __name__ == '__main__':
    main()
