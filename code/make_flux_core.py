'frozen product trajectories and QC-passed flux tables;original alignment/seasonality helpers;production run.\nInputs: code/paths.py; outputs: explicit derivation work directory, never upstream.\nOriginal: upstream://V6/flux_compare/tests/run_seasonality_tests.py.\nRun IDs: cap2048_v3_nested11_20261108; cap2048_spatial_gsingle_20261112; cap2048_production_nn_temporal_20261112.\nHeavy derivation was not rerun when packaging. Use --help before --run.'
from __future__ import annotations
import paths,sys
if __name__=="__main__":
 from make_runtime import cli
 cli(__doc__)
sys.path.insert(0,str(paths.MODEL_CODE))
import os
from pathlib import Path
T = paths.WORK / "flux_tests"
T.mkdir(parents=True,exist_ok=True)
ROOT = paths.FLUX
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLCONFIGDIR'] = str(T / 'mplconfig')
import json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
TRAITS = ['LMA', 'Chl', 'EWT', 'N']
FLUX = ['GPP', 'ET', 'SWC']
SEED = 20261113
DRAWS = 10000
BLOCK = 4
SOURCE = ROOT.parent / 'source_data/Fig4_trajectories_Hesse.csv'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def doy365(dates):
    out = []
    for d in pd.DatetimeIndex(dates):
        out.append(59.5 if (d.month, d.day) == (2, 29) else pd.Timestamp(2001, d.month, d.day).dayofyear)
    return np.array(out, dtype=float)

def loyo(s):
    dates = pd.DatetimeIndex(s.index)
    doy = doy365(dates)
    yrs = dates.year
    ref = np.full(len(s), np.nan)
    n = np.zeros(len(s), dtype=int)
    contributions = []
    for i, (year, x) in enumerate(zip(yrs, doy)):
        values = []
        for other in sorted(set(yrs) - {year}):
            ix = np.flatnonzero((yrs == other) & s.notna().to_numpy())
            xs = doy[ix]
            where = np.searchsorted(xs, x)
            left = right = None
            value = np.nan
            if where < len(xs) and np.isclose(xs[where], x):
                left = right = int(ix[where])
                value = float(s.iloc[left])
            elif 0 < where < len(xs):
                left, right = (int(ix[where - 1]), int(ix[where]))
                width = (dates[right] - dates[left]) / pd.Timedelta(days=1)
                if width <= 16:
                    value = float(np.interp(x, [doy[left], doy[right]], [s.iloc[left], s.iloc[right]]))
            if np.isfinite(value):
                values.append(value)
                contributions.append(dict(product_date=dates[i], variable=s.name, target_year=int(year), reference_year=int(other), left_date=dates[left], right_date=dates[right], bracket_days=(dates[right] - dates[left]) / pd.Timedelta(days=1), value=value))
        n[i] = len(values)
        if len(values) >= 2:
            ref[i] = np.mean(values)
    return (ref, n, contributions)

def historical_reference(old_daily, dates):
    old = old_daily.copy()
    old['md'] = old.index.strftime('%m-%d')
    old = old[old.md != '02-29']
    cal = pd.date_range('2001-01-01', '2001-12-31').strftime('%m-%d')
    out = pd.DataFrame(index=cal)
    for var, col in [('GPP', 'GPP_NT_VUT_REF'), ('ET', 'ET')]:
        daily_mean = old.groupby('md')[col].mean().reindex(cal).to_numpy()
        cycle = np.r_[daily_mean[-7:], daily_mean, daily_mean[:7]]
        smooth = pd.Series(cycle).rolling(15, center=True, min_periods=8).mean().iloc[7:-7].to_numpy()
        out[var + '_historical_daily_mean'] = daily_mean
        out[var + '_historical_W7'] = smooth
    out.index.name = 'month_day'
    out.to_csv(T / 'historical_climatology_used.csv')

    def mapped(col):
        return np.array([(out.loc['02-28', col] + out.loc['03-01', col]) / 2 if (d.month, d.day) == (2, 29) else out.loc[d.strftime('%m-%d'), col] for d in dates])
    return {v: mapped(v + '_historical_W7') for v in ['GPP', 'ET']}

def bootstrap_indices(n):
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, n - BLOCK + 1, size=(DRAWS, int(np.ceil(n / BLOCK))))
    bank = (starts[:, :, None] + np.arange(BLOCK)).reshape(DRAWS, -1)[:, :n]
    np.savez_compressed(T / 'bootstrap_product_indices.npz', indices=bank, seed=SEED, block=BLOCK)
    return bank

def bootstrap_spearman(x, y, mask, bank):
    valid = np.isfinite(x) & np.isfinite(y) & mask
    n = int(valid.sum())
    if n < 3 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return (np.nan, n, np.nan, np.nan, 0)
    rho = float(spearmanr(x[valid], y[valid]).statistic)
    bvalid = valid[bank]
    m = bvalid.sum(axis=1)
    rx = rankdata(np.where(bvalid, x[bank], np.inf), axis=1, method='average')
    ry = rankdata(np.where(bvalid, y[bank], np.inf), axis=1, method='average')
    rx = np.where(bvalid, rx - (m[:, None] + 1) / 2, 0.0)
    ry = np.where(bvalid, ry - (m[:, None] + 1) / 2, 0.0)
    denom = np.sqrt(np.sum(rx * rx, axis=1) * np.sum(ry * ry, axis=1))
    draw = np.full(DRAWS, np.nan)
    good = (m >= 3) & (denom > 0)
    draw[good] = np.sum(rx[good] * ry[good], axis=1) / denom[good]
    low, high = np.nanquantile(draw, [0.025, 0.975])
    return (rho, n, float(low), float(high), int(good.sum()))

def phenology_one(series, year, phase, q):
    s = series.copy()
    idx = pd.DatetimeIndex(s.index)
    if phase == 'SOS':
        base = s.loc[pd.Timestamp(year, 1, 1):pd.Timestamp(year, 3, 31, 23, 59, 59)].dropna()
    else:
        base = s.loc[pd.Timestamp(year, 11, 15):pd.Timestamp(year + 1, 1, 15, 23, 59, 59)].dropna()
    summer = s.loc[pd.Timestamp(year, 6, 1):pd.Timestamp(year, 8, 31, 23, 59, 59)].dropna()
    out = dict(baseline=float(base.median()) if len(base) else np.nan, n_baseline=len(base), peak=float(summer.max()) if len(summer) else np.nan, n_summer=len(summer), threshold_value=np.nan, date=pd.NaT, bracket_start=pd.NaT, bracket_end=pd.NaT, bracket_days=np.nan, flag='missing baseline or summer peak', n_crossings=0)
    if not len(base) or not len(summer):
        return out
    peakdate = summer.idxmax()
    amp = out['peak'] - out['baseline']
    threshold = out['baseline'] + q * amp
    out['threshold_value'] = threshold
    if amp <= 0:
        out['flag'] = 'non-positive amplitude'
        return out
    if phase == 'SOS':
        window = s.loc[pd.Timestamp(year, 1, 1):peakdate].dropna()
    else:
        window = s.loc[peakdate:pd.Timestamp(year + 1, 1, 15, 23, 59, 59)].dropna()
    candidates = []
    for i in range(1, len(window)):
        a, b = (float(window.iloc[i - 1]), float(window.iloc[i]))
        left, right = (window.index[i - 1], window.index[i])
        crossed = a < threshold <= b if phase == 'SOS' else a >= threshold > b
        if crossed:
            date = left + (right - left) * ((threshold - a) / (b - a))
            candidates.append((date, left, right, (right - left) / pd.Timedelta(days=1)))
    out['n_crossings'] = len(candidates)
    if not candidates:
        out['flag'] = 'no bracketing crossing'
        return out
    date, left, right, width = candidates[0] if phase == 'SOS' else candidates[-1]
    out.update(date=date, bracket_start=left, bracket_end=right, bracket_days=float(width), flag='uncertain: bracket >16 d' if width > 16 else 'resolved: bracket <=16 d')
    return out

def phenology(frame, daily):
    data = {v: frame[v] for v in ['GPP', 'Chl', 'N']}
    data['GPP_daily'] = daily['GPP_NT_VUT_REF'].rolling(15, center=True, min_periods=8).mean()
    data['GPP_daily'].index = data['GPP_daily'].index + pd.Timedelta(hours=12)
    rows = []
    pairs = [(y, 'SOS') for y in [2023, 2024, 2025, 2026]] + [(y, 'EOS') for y in [2022, 2023, 2024, 2025]]
    for q in [0.2, 0.5]:
        for year, phase in pairs:
            row = dict(year=year, phase=phase, threshold=q, focus=(year, phase) in [(2022, 'EOS'), (2026, 'SOS')])
            results = {v: phenology_one(s, year, phase, q) for v, s in data.items()}
            for v, r in results.items():
                for key, value in r.items():
                    row[v + '_' + key] = value
            for trait in ['Chl', 'N']:
                for gpp in ['GPP', 'GPP_daily']:
                    a, b = (results[trait], results[gpp])
                    label = trait + '_minus_' + gpp
                    row[label + '_days'] = (a['date'] - b['date']) / pd.Timedelta(days=1) if pd.notna(a['date']) and pd.notna(b['date']) else np.nan
                    row[label + '_interval_low_days'] = (a['bracket_start'] - b['bracket_end']) / pd.Timedelta(days=1) if pd.notna(a['date']) and pd.notna(b['date']) else np.nan
                    row[label + '_interval_high_days'] = (a['bracket_end'] - b['bracket_start']) / pd.Timedelta(days=1) if pd.notna(a['date']) and pd.notna(b['date']) else np.nan
            row['flag'] = 'missing' if pd.isna(row['Chl_minus_GPP_days']) else 'uncertain bracket' if max(results['Chl']['bracket_days'], results['GPP']['bracket_days']) > 16 else 'resolved'
            rows.append(row)
    ph = pd.DataFrame(rows)
    ph.to_csv(T / 'phenology_dates_Hesse.csv', index=False)
    summaries = []
    for q in [0.2, 0.5]:
        for phase in ['SOS', 'EOS']:
            for trait in ['Chl', 'N']:
                for gpp in ['GPP', 'GPP_daily']:
                    sub = ph[(ph.threshold == q) & (ph.phase == phase)]
                    for scope in ['all_estimates', 'resolved_brackets_only']:
                        if scope == 'resolved_brackets_only':
                            sub = sub[sub[trait + '_bracket_days'].le(16) & sub[gpp + '_bracket_days'].le(16)]
                        delta = sub[trait + '_minus_' + gpp + '_days'].dropna()
                        summaries.append(dict(phase=phase, threshold=q, trait=trait, GPP_variant=gpp, scope=scope, n=len(delta), mean_delta=delta.mean(), median_delta=delta.median(), min_delta=delta.min(), max_delta=delta.max()))
    summary = pd.DataFrame(summaries)
    summary.to_csv(T / 'phenology_summary.csv', index=False)
    return (ph, summary)

def main():
    paths = [SOURCE, ROOT / 'flux_vs_traits_aligned_Hesse.csv', ROOT / 'work_v2/Hesse_daily_QC.csv', ROOT / 'work/Hesse_daily_QC.csv', ROOT / 'work/Hesse_climatology.csv', ROOT / 'FigS_flux_vs_traits.png']
    hashes = {str(p): sha(p) for p in paths}
    a = pd.read_csv(ROOT / 'flux_vs_traits_aligned_Hesse.csv', parse_dates=['product_date']).set_index('product_date')
    src = pd.read_csv(SOURCE)
    src.trait = src.trait.replace({'CHL': 'Chl'})
    src.date = pd.to_datetime(src.date, utc=True, format='mixed').dt.tz_localize(None)
    expected = src[src.kind.eq('Production NN')].pivot(index='date', columns='trait', values='mean').sort_index()
    np.testing.assert_allclose(a[TRAITS], expected[TRAITS], rtol=0, atol=1e-12, equal_nan=True)
    assert len(a) == 113 and (a.index.to_series().diff().dropna() == pd.Timedelta(days=14)).all()
    f = a[TRAITS].copy()
    for v, c in [('GPP', 'GPP_NT_VUT_REF'), ('ET', 'ET'), ('SWC', 'SWC')]:
        f[v] = a[c + '_mean']
        f[v + '_n_qc_days'] = a[c + '_n_qc_days']
    history = pd.read_csv(ROOT / 'work/Hesse_daily_QC.csv', parse_dates=['date']).set_index('date')
    oldref = historical_reference(history, f.index)
    contributions = []
    for v in TRAITS + FLUX:
        ref, n, rows = loyo(f[v])
        contributions.extend(rows)
        f[v + '_LOYO_reference'] = ref
        f[v + '_n_reference_years'] = n
        f[v + '_anomaly_LOYO'] = f[v] - ref
    for v in ['GPP', 'ET']:
        f[v + '_historical_W7'] = oldref[v]
        f[v + '_anomaly_historical'] = f[v] - oldref[v]
    old = pd.read_csv(ROOT / 'work/Hesse_climatology.csv').set_index('month_day')
    diffs = {}
    for v, c in [('GPP', 'GPP_NT_VUT_REF'), ('ET', 'ET')]:
        vals = np.array([old.loc[d.strftime('%m-%d'), c + '_climatology_mean'] for d in f.index])
        diffs[v] = float(np.max(np.abs(oldref[v] - vals)))
    f.to_csv(T / 'anomaly_series_Hesse.csv')
    pd.DataFrame(contributions).to_csv(T / 'LOYO_reference_contributions.csv', index=False)
    bank = bootstrap_indices(len(f))
    rows = []
    specs = [('GPP', 'historical', 'primary'), ('GPP', 'LOYO', 'sensitivity'), ('ET', 'historical', 'primary'), ('ET', 'LOYO', 'sensitivity'), ('SWC', 'LOYO', 'primary')]
    for test, representation, seasons in [('test1', 'anomaly', ['all_dates', 'Apr-Oct']), ('test2', 'raw', ['Jun-Aug']), ('test2', 'anomaly', ['Jun-Aug'])]:
        options = specs if representation == 'anomaly' else [(v, 'none', 'primary') for v in FLUX]
        for flux, ref, role in options:
            for trait in TRAITS:
                for season in seasons:
                    mask = np.ones(len(f), dtype=bool) if season == 'all_dates' else f.index.month.isin(range(4, 11) if season == 'Apr-Oct' else [6, 7, 8])
                    x = f[trait + '_anomaly_LOYO'].to_numpy() if representation == 'anomaly' else f[trait].to_numpy()
                    y = f[flux + '_anomaly_' + ref].to_numpy() if representation == 'anomaly' else f[flux].to_numpy()
                    rho, n, lo, hi, nb = bootstrap_spearman(x, y, mask, bank)
                    rows.append(dict(test=test, representation=representation, trait=trait, flux=flux, flux_reference=ref, role=role, season=season, rho=rho, n=n, CI95_low=lo, CI95_high=hi, bootstrap_valid_draws=nb, block_product_dates=BLOCK, bootstrap_draws=DRAWS, seed=SEED))
            print('completed', test, representation, flux, ref, flush=True)
    cor = pd.DataFrame(rows)
    cor.to_csv(T / 'correlations_tests.csv', index=False)
    daily = pd.read_csv(ROOT / 'work_v2/Hesse_daily_QC.csv', parse_dates=['date']).set_index('date')
    ph, summary = phenology(f, daily)
    assert hashes == {str(p): sha(p) for p in paths}, 'A protected input changed'
    prov = dict(source_sha256=hashes, protected_inputs_unchanged=True, n_product_dates=len(f), LOYO_no_extrapolation=True, LOYO_max_gap_days=16, LOYO_min_other_years=2, historical_climatology='Multi-year daily month/day mean then centred +/-7 d mean (one smoothing only); circular year boundary', max_abs_difference_vs_v1_saved_W7=diffs, SWC_reference='LOYO only: 2014-2020 SWC depth unverified', bootstrap=dict(block=BLOCK, draws=DRAWS, seed=SEED, sampling='Non-circular moving blocks on all 113 original dates; mask seasons and invalid pairs after drawing; truncate to 113 slots'), phenology_threshold_equation='baseline + q*(peak - baseline), q=0.2 or 0.5', snow_or_season_rule_changed=False, cpu_threads=1)
    (T / 'provenance.json').write_text(json.dumps(prov, indent=2))
    print(cor[cor.trait.isin(['Chl', 'N']) & (cor.flux == 'GPP')].to_string(index=False), flush=True)
    print(ph[['year', 'phase', 'threshold', 'GPP_date', 'Chl_date', 'Chl_minus_GPP_days', 'GPP_bracket_days', 'Chl_bracket_days', 'flag']].to_string(index=False), flush=True)
    print(summary[(summary.trait == 'Chl') & (summary.GPP_variant == 'GPP')].to_string(index=False), flush=True)
if __name__ == '__main__':
    main()
