"""formal rows with baselines and scorer metrics; writes Fig3 metrics/budget; temporal and spatial runs.
Inputs: code/paths.py; outputs: explicit derivation work directory, never upstream.
Original: upstream://V6/final_figures_20261113/scripts/analyze_fig3.py.
Run IDs: cap2048_v3_nested11_20261108; cap2048_spatial_gsingle_20261112; cap2048_production_nn_temporal_20261112.
Heavy derivation was not rerun when packaging. Use --help before --run."""
from __future__ import annotations
import paths, sys
if __name__ == '__main__':
    from make_runtime import cli
    cli(__doc__)
sys.path.insert(0, str(paths.MODEL_CODE))
from make_support import *
import make_metrics_core as mm
from make_metrics_fast import bootstrap

def score(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    mse = np.mean((p - y) ** 2)
    span = np.quantile(y, 0.95) - np.quantile(y, 0.05)
    return (np.array([1 - mse / np.var(y), np.sqrt(mse) / span]), span)

def budget_one(frame, column):
    result = mm.evaluate(frame, prediction_columns={'mean42_46': column}, data_version=RUN, pixel_ids_stable=False, bootstrap_draws=0)
    b = bootstrap(result['prepared'], result['scenes'], result['pairs'], draws=1000, random_seed=20261111)
    return (result, b)

def main():
    f = pd.read_parquet(paths.derived_input('formal_temporal_with_pixel_baselines.parquet'))
    base = pd.read_csv(V6 / 'source_data/Fig3_panel_a_metrics.csv')
    base['record_only'] = base.model.ne('NN')
    base['evaluation_subset'] = 'full formal known-site W7-covered'
    all_metrics = base.to_dict('records')
    baselines = []
    budgets = []
    for t in ORDER:
        j = RAW.index(t)
        y = f[f'ref__{j}'].to_numpy(float)
        ps = np.array([f[f'NN_{s}__{j}'] for s in range(42, 47)], float)
        full = np.array([score(y, p)[0] for p in ps])
        official = base[base.scope.eq('temporal') & base.model.eq('NN') & base.trait.eq(t)].set_index('metric')
        assert np.allclose(full.mean(0), official.loc[['r2', 'nrmse_P95_P05'], 'mean'], atol=1e-12, rtol=1e-12), 'Formal metric identity failed'
        canonical = pd.DataFrame(dict(site_id=f.site_id, event_id=f.target_event_id, date_utc=pd.to_datetime(f.target_event_id.str.split(':', n=1).str[1], utc=True), pixel_id=f.sample_id, reference=y, qc=True, oos=True, NN_ensemble=ps.mean(0)))
        for model, prefix in [('pixel_mean', 'pixel_mean'), ('pixel_seasonal_mean', 'pixel_season')]:
            values = f[f'{prefix}__{j}'].to_numpy(float)
            defined = np.isfinite(values)
            n = int(defined.sum())
            coverage = float(defined.mean())
            sc, span = score(y[defined], values[defined])
            matched = np.array([score(y[defined], p[defined])[0] for p in ps])
            delta = float(matched[:, 0].mean() - full[:, 0].mean())
            baselines.append(dict(trait=t, model=model, n_full=len(f), n_defined=n, coverage=coverage, r2=sc[0], nrmse_P95_P05=sc[1], reference_P95_P05=span, full_reference_P95_P05=score(y, ps[0])[1], NN_full_r2=full[:, 0].mean(), NN_matched_r2=matched[:, 0].mean(), NN_matched_r2_min=matched[:, 0].min(), NN_matched_r2_max=matched[:, 0].max(), NN_matched_nrmse=matched[:, 1].mean(), NN_matched_nrmse_min=matched[:, 1].min(), NN_matched_nrmse_max=matched[:, 1].max(), matched_r2_difference=delta, flag_coverage_below50=coverage < 0.5, flag_NN_r2_shift_gt003=abs(delta) > 0.03, season_definition='DJF12/1/2;MAM3/4/5;JJA6/7/8;SON9/10/11', fallback='none; >=1 other-year event at same parent pixel'))
            for k, metric in enumerate(['r2', 'nrmse_P95_P05']):
                common = dict(trait=t, source_trait=TRAITS[j], scope='temporal', metric=metric, n=n, coverage=coverage, run_id=RUN, record_only=False, evaluation_subset=model + ' defined subset')
                all_metrics.append(dict(**common, model=model, mean=sc[k], seed_min=sc[k], seed_max=sc[k], n_seeds=1))
                all_metrics.append(dict(**common, model='NN_matched_' + model, mean=matched[:, k].mean(), seed_min=matched[:, k].min(), seed_max=matched[:, k].max(), n_seeds=5, record_note='matched-subset sensitivity; not plotted'))
        full_result, b = budget_one(canonical, 'NN_ensemble')
        point = full_result['per_seed'].iloc[0]
        scene = full_result['scenes']
        eligible = full_result['prepared']['data']
        total = float(sum((point[f'recovered_share_{l}'] for l in mm.LEVELS)))
        subset = canonical.copy()
        subset['pixel_seasonal_mean'] = f[f'pixel_season__{j}']
        subset = subset[np.isfinite(subset.pixel_seasonal_mean)]
        baseline_result, bb = budget_one(subset, 'pixel_seasonal_mean')
        bp = baseline_result['per_seed'].iloc[0]
        for level in mm.LEVELS:
            metric = lambda prefix: float(point[f'{prefix}_{level}'])
            q = lambda prefix: np.nanquantile(b[f'{prefix}_{level}'], [0.025, 0.975])
            lo, hi = q('recovered_share')
            sl, sh = q('variance_share')
            rl, rh = q('r2')
            budgets.append(dict(trait=t, scope='temporal', level=level, share=metric('variance_share'), share_low=sl, share_high=sh, r2=metric('r2'), r2_low=rl, r2_high=rh, recovered=metric('recovered_share'), recovered_low=lo, recovered_high=hi, variance=metric('v'), n_pixels=len(eligible), n_sites=eligible.site_id.nunique(), n_scenes=len(scene), n_formal_rows=len(f), temporal_share=float(point.variance_share_date), total_recovered=total, pixel_seasonal_recovered_date=float(bp.recovered_share_date), pixel_seasonal_raw_date_r2=float(bp.r2_date), pixel_seasonal_subset_date_share=float(bp.variance_share_date), pixel_seasonal_subset_n=len(baseline_result['prepared']['data']), bootstrap_draws=1000, bootstrap_seed=20261111, estimator='5-seed mean predictions; ensemble hierarchy', run_id=RUN))
        b.to_parquet(DATA / f'Fig3b_bootstrap_{t}.parquet', index=False)
        print('FIG3', t, 'coverage', [(r['model'], round(r['coverage'], 3), round(r['matched_r2_difference'], 4)) for r in baselines[-2:]], 'temporal', point.variance_share_date, 'recovered', total, flush=True)
    pd.DataFrame(all_metrics).to_csv(DATA / 'Fig3a_metrics.csv', index=False)
    pd.DataFrame(baselines).to_csv(DATA / 'Fig3a_pixel_baselines.csv', index=False)
    pd.DataFrame(budgets).to_csv(DATA / 'Fig3b_budget.csv', index=False)
    flags = [r for r in baselines if r['flag_coverage_below50'] or r['flag_NN_r2_shift_gt003']]
    save(DATA / 'FIG3_ANALYSIS_DONE.json', dict(formal_rows=len(f), formal_metric_identity=True, flags=flags, partition_model='five-seed mean predictions', baseline_population='baseline defined subset; NN matched metrics retained', historical_events='development, calendar year!=held year, valid_mask & finite trait, no fallback', bootstrap='1000 sites-with-replacement, seed20261111; reused fast_metrics.bootstrap'))
if __name__ == '__main__':
    main()
