'downloaded ICOS archives;original HH-to-daily QC and date alignment;production run.\nInputs: code/paths.py; outputs: explicit derivation work directory, never upstream.\nOriginal: upstream://V6/flux_compare/plot_flux_compare.py.\nRun IDs: cap2048_v3_nested11_20261108; cap2048_spatial_gsingle_20261112; cap2048_production_nn_temporal_20261112.\nHeavy derivation was not rerun when packaging. Use --help before --run.'
from __future__ import annotations
import paths,sys
if __name__=="__main__":
 from make_runtime import cli
 cli(__doc__)
sys.path.insert(0,str(paths.MODEL_CODE))
import os
from pathlib import Path
HERE = paths.FLUX
WORK = paths.WORK / 'icos_hesse'
WORK.mkdir(parents=True,exist_ok=True)
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLCONFIGDIR'] = str(WORK / 'mplconfig')
import hashlib, json, zipfile, io, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr
FIGROOT = HERE.parent
DOWNLOAD = HERE / 'icos_download_20261113'
TRAITS = ['LMA', 'Chl', 'EWT', 'N']
FLUX = ['GPP_NT_VUT_REF', 'ET', 'SWC']
COLORS = ['#009E73', '#D55E00', '#0072B2', '#CC79A7']
UNITS = ['g m$^{-2}$', '$\\mu$g cm$^{-2}$', 'mg cm$^{-2}$', 'mg cm$^{-2}$']
START, END = (pd.Timestamp('2022-06-01'), pd.Timestamp('2026-09-30'))
HES = next((DOWNLOAD / 'FR-Hes_FLUXMET').glob('*.zip'))
SFN = next((DOWNLOAD / 'DE-SfN_FULLSET').glob('*.zip'))
ANC = next((DOWNLOAD / 'FR-Hes_ancillary_archive').glob('*.zip'))

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def zip_csv(path, needle, **kwargs):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if needle in n and n.endswith('.csv')]
        assert len(names) == 1, (path, needle, names)
        with z.open(names[0]) as f:
            d = pd.read_csv(f, na_values=[-9999, -9999.0], **kwargs)
    return (d, names[0])

def verify_depth():
    d, n = zip_csv(ANC, 'VARINFO_FLUXNET_HH', encoding='utf-8')
    ids = d.loc[d.VARIABLE.eq('VAR_INFO_VARNAME') & d.DATAVALUE.eq('SWC_F_MDS_1'), 'GROUP_ID']
    group = d[d.GROUP_ID.isin(ids)]
    heights = pd.to_numeric(group.loc[group.VARIABLE.eq('VAR_INFO_HEIGHT'), 'DATAVALUE'])
    assert len(heights) == 1 and float(heights.iloc[0]) == -0.05
    group.to_csv(WORK / 'SWC_depth_evidence.csv', index=False)
    return dict(column='SWC_F_MDS_1', depth_m=0.05, depth_cm=5, source=str(ANC.relative_to(HERE)) + '!' + n, group_id=int(ids.iloc[0]), height_relative_to_ground_m=-0.05, metadata_date='20220518')

def hes_daily():
    cols = ['TIMESTAMP_START', 'TIMESTAMP_END', 'GPP_NT_VUT_REF', 'NEE_VUT_REF_QC', 'LE_F_MDS', 'LE_F_MDS_QC', 'SWC_F_MDS_1', 'SWC_F_MDS_1_QC']
    raw, member = zip_csv(HES, 'FLUXMET_HH', usecols=cols)
    starts = pd.DatetimeIndex(pd.to_datetime(raw.TIMESTAMP_START.astype(str), format='%Y%m%d%H%M'))
    ends = pd.DatetimeIndex(pd.to_datetime(raw.TIMESTAMP_END.astype(str), format='%Y%m%d%H%M'))
    assert starts.is_unique and starts.is_monotonic_increasing
    assert (ends - starts == pd.Timedelta(minutes=30)).all()
    expected = pd.date_range('2022-01-01', '2026-08-31 23:30', freq='30min')
    assert starts.equals(expected), 'HH date continuity changed'
    raw.index = starts
    daily = pd.DataFrame(index=pd.date_range('2022-01-01', '2026-08-31', name='date'))
    rules = [('GPP_NT_VUT_REF', 'GPP_NT_VUT_REF', 'NEE_VUT_REF_QC', 1.03772448), ('ET', 'LE_F_MDS', 'LE_F_MDS_QC', 86400 / 2450000.0), ('SWC', 'SWC_F_MDS_1', 'SWC_F_MDS_1_QC', 1.0)]
    for var, column, qc, factor in rules:
        means = raw[column].resample('D').mean() * factor
        good = (raw[qc].isin([0, 1]) & raw[column].notna()).resample('D').sum()
        daily[var + '_raw_daily_mean'] = means
        daily[var + '_n_good_hh'] = good
        daily[var + '_n_finite_hh'] = raw[column].resample('D').count()
        daily[var + '_qc_fraction'] = good / 48
        daily[var] = means.where(good / 48 >= 0.5)
    daily.to_csv(WORK / 'Hesse_daily_QC.csv')
    qc_stats = {v: {'qc_codes': {str(k): int(n) for k, n in raw[qc].value_counts(dropna=False).items()}, 'n_pass_days': int(daily[v].notna().sum()), 'first_pass_date': str(daily[v].first_valid_index()), 'last_pass_date': str(daily[v].last_valid_index())} for v, _, qc, _ in rules}
    return (daily, member, qc_stats)

def sfn_daily():
    raw, member = zip_csv(SFN, 'FULLSET_DD_')
    dates = pd.DatetimeIndex(pd.to_datetime(raw.TIMESTAMP.astype(str), format='%Y%m%d'))
    assert dates.equals(pd.date_range('2012-01-01', '2014-12-31'))
    d = pd.DataFrame(index=dates)
    d.index.name = 'date'
    for v, col, qc, factor in [('GPP_NT_VUT_REF', 'GPP_NT_VUT_REF', 'NEE_VUT_REF_QC', 1), ('ET', 'LE_F_MDS', 'LE_F_MDS_QC', 86400 / 2450000.0), ('SWC', 'SWC_F_MDS_1', 'SWC_F_MDS_1_QC', 1)]:
        if col not in raw:
            d[v] = np.nan
            d[v + '_qc_fraction'] = np.nan
        else:
            q = raw[qc] if qc in raw else pd.Series(np.nan, index=raw.index)
            d[v + '_qc_fraction'] = q.to_numpy()
            d[v] = np.where(q.ge(0.5), raw[col] * factor, np.nan)
    d.to_csv(WORK / 'Schechenfilz_daily_QC.csv')
    return (d, member)

def climatology(daily):
    md = pd.date_range('2001-01-01', '2001-12-31').strftime('%m-%d')
    out = pd.DataFrame(index=pd.Index(md, name='month_day'))
    peryear = []
    for var in FLUX:
        means = daily[var].rolling(15, center=True, min_periods=8).mean()
        count = daily[var].rolling(15, center=True, min_periods=1).count()
        sm, ct, dd = ([], [], [])
        for year in [2012, 2013, 2014]:
            dates = pd.to_datetime([f'{year}-{s}' for s in md])
            a = means.reindex(dates).to_numpy()
            b = count.reindex(dates).to_numpy()
            sm.append(a)
            ct.append(b)
            dd.append(daily[var].reindex(dates).to_numpy())
            peryear.append(pd.DataFrame(dict(year=year, month_day=md, variable=var, w7_mean=a, n_qc_days=b)))
        sm = np.asarray(sm)
        ct = np.asarray(ct)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            out[var + '_daily_climatology'] = np.nanmean(dd, axis=0)
            out[var + '_climatology_mean'] = np.nanmean(sm, axis=0)
            out[var + '_climatology_p10'] = np.nanpercentile(sm, 10, axis=0)
            out[var + '_climatology_p90'] = np.nanpercentile(sm, 90, axis=0)
            out[var + '_historical_n_qc_days_min'] = np.nanmin(np.where(np.isfinite(sm), ct, np.nan), axis=0)
        out[var + '_climatology_n_years'] = np.isfinite(sm).sum(axis=0)
    out.to_csv(WORK / 'Schechenfilz_climatology.csv')
    pd.concat(peryear).to_csv(WORK / 'Schechenfilz_historical_W7.csv', index=False)
    return out

def map_clim(c, dates, col):
    return np.array([(c.loc['02-28', col] + c.loc['03-01', col]) / 2 if t.strftime('%m-%d') == '02-29' else c.loc[t.strftime('%m-%d'), col] for t in dates])

def traits(site):
    f = pd.read_csv(FIGROOT / 'source_data' / f'Fig4_trajectories_{site}.csv')
    f.trait = f.trait.replace({'CHL': 'Chl'})
    f.date = pd.to_datetime(f.date, utc=True, format='mixed').dt.tz_localize(None)
    pred = f[f.kind.eq('Production NN')]
    assert pred.run_id.eq('cap2048_production_nn_temporal_20261112').all()
    assert pred.loc[pred.coverage.lt(0.8), 'mean'].isna().all()
    t = pred.pivot(index='date', columns='trait', values='mean')[TRAITS].sort_index()
    assert len(t) == 113 and (t.index.to_series().diff().dropna() == pd.Timedelta(days=14)).all()
    t.index.name = 'product_date'
    return (t, f)

def correlate(t, site):
    rows = []
    for var in FLUX:
        for trait in TRAITS:
            for season in ['all_dates', 'Apr-Oct']:
                s = t.loc[:, [trait, var + '_mean']]
                if season == 'Apr-Oct':
                    s = s.loc[s.index.month.isin(range(4, 11))]
                s = s.dropna()
                n = len(s)
                rho = float(spearmanr(s.iloc[:, 0], s.iloc[:, 1]).statistic) if n >= 3 and s.iloc[:, 0].nunique() > 1 and (s.iloc[:, 1].nunique() > 1) else np.nan
                rows.append(dict(site=site, trait=trait, flux=var, season=season, spearman_rho=rho, n=n, comparison='concurrent only'))
    return rows

def phenology(table):
    """Grid midpoint transitions only; never interpolate across missing trait dates."""
    rows = []
    candidates = []
    for year in range(2022, 2027):
        for var, column in [('GPP', 'GPP_NT_VUT_REF_mean'), ('Chl', 'Chl'), ('N', 'N')]:
            s = table.loc[table.index.year == year, column]
            threshold = (s.min() + s.max()) / 2
            for phase in ['greenup', 'senescence']:
                crossings = []
                gapped = []
                for i in range(1, len(s)):
                    a, b = (s.iloc[i - 1], s.iloc[i])
                    if not (np.isfinite(a) and np.isfinite(b)):
                        continue
                    dt = s.index[i]
                    rising = a < threshold <= b and 3 <= dt.month <= 7
                    falling = a >= threshold > b and 8 <= dt.month <= 12
                    if phase == 'greenup' and rising or (phase == 'senescence' and falling):
                        crossings.append((s.index[i - 1], dt))
                for left, right in crossings:
                    candidates.append(dict(year=year, variable=var, phase=phase, bracket_start=left, crossing_product_date=right, midpoint=threshold))
                pair = (crossings[0] if phase == 'greenup' else crossings[-1]) if crossings else (pd.NaT, pd.NaT)
                rows.append(dict(year=year, variable=var, phase=phase, seasonal_min=s.min(), seasonal_max=s.max(), midpoint=threshold, bracket_start=pair[0], crossing_product_date=pair[1], n_directional_crossings=len(crossings), n_finite_dates=int(s.notna().sum()), complete_calendar_year=year in [2023, 2024, 2025], status='observed adjacent-date crossing' if crossings else 'not observed: gaps or truncated year'))
    result = pd.DataFrame(rows)
    result.to_csv(HERE / 'phenology_crossings.csv', index=False)
    pd.DataFrame(candidates).to_csv(WORK / 'phenology_crossing_candidates.csv', index=False)
    compare = []
    for year in range(2022, 2027):
        for phase in ['greenup', 'senescence']:
            sub = result[(result.year == year) & (result.phase == phase)].set_index('variable')
            for trait in ['Chl', 'N']:
                a = sub.loc[trait, 'crossing_product_date']
                b = sub.loc['GPP', 'crossing_product_date']
                lag = (a - b).days if pd.notna(a) and pd.notna(b) else np.nan
                compare.append(dict(year=year, phase=phase, trait=trait, trait_crossing_date=a, GPP_crossing_date=b, trait_minus_GPP_days=lag, complete_calendar_year=year in [2023, 2024, 2025]))
    pd.DataFrame(compare).to_csv(HERE / 'phenology_timing_comparison.csv', index=False)
    return result

def main():
    sourcepaths = [FIGROOT / 'source_data' / f'Fig4_trajectories_{s}.csv' for s in ['Hesse', 'Schechenfilz']]
    hashes = {str(p): sha(p) for p in sourcepaths}
    v1 = json.loads((HERE / 'work/provenance.json').read_text())['input_sha256']
    assert all((v1[str(p)] == hashes[str(p)] for p in sourcepaths)), 'Trait source changed since v1'
    query = json.loads((WORK / 'DE-SfN_query_results.json').read_text())['results']['bindings']
    assert len(query) == 2 and all(('2012-2014' in row['fileName']['value'] for row in query))
    manifest = pd.read_csv(DOWNLOAD / 'download_manifest.csv')
    for _, r in manifest.iterrows():
        assert sha(HERE / r.local_path) == r.SHA256
    depth = verify_depth()
    hd, hm, qstats = hes_daily()
    sd, sm = sfn_daily()
    clim = climatology(sd)
    tables = {}
    frames = {}
    corr = []
    for site in ['Hesse', 'Schechenfilz']:
        t, f = traits(site)
        for v in FLUX:
            if site == 'Hesse':
                count = hd[v].rolling(15, center=True, min_periods=1).count()
                mean = hd[v].rolling(15, center=True, min_periods=8).mean()
                grid = t.index.normalize()
                t[v + '_mean'] = mean.reindex(grid).to_numpy()
                t[v + '_n_qc_days'] = count.reindex(grid, fill_value=0).to_numpy().astype(int)
                assert t.loc[t.index >= pd.Timestamp('2026-09-01'), v + '_mean'].isna().all()
            else:
                t[v + '_mean'] = np.nan
                t[v + '_n_qc_days'] = 0
                for c in clim:
                    if c.startswith(v + '_'):
                        t[c] = map_clim(clim, t.index, c)
        t['n_qc_days'] = t['GPP_NT_VUT_REF_n_qc_days']
        t['comparison_type'] = 'concurrent' if site == 'Hesse' else 'not_concurrent_climatology'
        t['source'] = '11676/3FV_P_h69OveJojAgQKrPEZD!' + hm if site == 'Hesse' else '11676/1vAdB1FBLLkHIkzB3V8rt8tT!' + sm + ';2012-2014;NOT concurrent'
        t['SWC_depth_cm'] = 5.0 if site == 'Hesse' else np.nan
        t.to_csv(HERE / f'flux_vs_traits_aligned_{site}.csv')
        tables[site] = t
        frames[site] = f
        corr.extend(correlate(t, site))
    correlations = pd.DataFrame(corr)
    correlations.to_csv(HERE / 'correlations.csv', index=False)
    phenology(tables['Hesse'])
    summaries = []
    for year, g in hd.groupby(hd.index.year):
        summer = g.loc[g.index.month.isin([6, 7, 8])]
        record = dict(year=year, period='Jun-Aug')
        for v in FLUX:
            record[v + '_mean'] = summer[v].mean()
            record[v + '_min'] = summer[v].min()
            record[v + '_n_days'] = int(summer[v].notna().sum())
        summaries.append(record)
    pd.DataFrame(summaries).to_csv(HERE / 'summer_flux_summary.csv', index=False)
    hd.groupby(hd.index.year)[FLUX].count().to_csv(WORK / 'Hesse_annual_qc_counts.csv')
    for site, t in tables.items():
        t.groupby(t.index.year)[[v + '_mean' for v in FLUX]].count().to_csv(WORK / f'{site}_annual_product_counts.csv')
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white', 'axes.unicode_minus': False})
    fig, axes = plt.subplots(7, 2, figsize=(15, 13.2), sharex='col', gridspec_kw={'height_ratios': [1.3, 1, 1, 1, 1, 1, 1], 'hspace': 0.17, 'wspace': 0.22})
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.083, top=0.91)
    daily_dates = pd.date_range(START, END)
    for ci, site in enumerate(['Hesse', 'Schechenfilz']):
        table = tables[site]
        for ax in axes[:, ci]:
            ax.set_xlim(START, END)
            for year in range(2021, 2027):
                ax.axvspan(pd.Timestamp(year, 11, 1), pd.Timestamp(year + 1, 3, 1), color='#EAEFF4', alpha=0.65, lw=0, zorder=0)
                ax.axvline(pd.Timestamp(year, 1, 1), color='.72', lw=0.5, zorder=1)
            ax.grid(axis='y', color='.9', lw=0.4)
            ax.tick_params(length=2.5, labelsize=9)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        subtitle = 'concurrent 2022–2026-08 (ICOS interim release)' if site == 'Hesse' else '2012–2014 climatology · NOT concurrent'
        label = 'Hesse · FR-Hes' if site == 'Hesse' else 'Schechenfilz Nord · DE-SfN'
        axes[0, ci].set_title(label + '\n' + subtitle, fontsize=12, pad=10)
        for ri, v in enumerate(FLUX):
            ax = axes[ri, ci]
            ax.set_ylabel(['GPP_NT_VUT_REF\\n(gC m$^{-2}$ d$^{-1}$)', 'ET\\n(mm d$^{-1}$)', 'SWC (%)'][ri].replace('\\n', '\n'))
            if site == 'Hesse':
                ax.plot(hd.index, hd[v], color='.72', lw=0.45, zorder=2)
                ax.scatter(table.index, table[v + '_mean'], s=11, color='.1', zorder=4)
                if v == 'SWC':
                    ax.text(0.015, 0.94, '5 cm depth · SWC_F_MDS_1', transform=ax.transAxes, fontsize=8.5, bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1))
            elif clim[v + '_climatology_mean'].notna().any():
                ax.fill_between(daily_dates, map_clim(clim, daily_dates, v + '_climatology_p10'), map_clim(clim, daily_dates, v + '_climatology_p90'), color='.65', alpha=0.25, lw=0)
                ax.plot(daily_dates, map_clim(clim, daily_dates, v + '_daily_climatology'), color='.72', lw=0.45, zorder=2)
                ax.plot(daily_dates, map_clim(clim, daily_dates, v + '_climatology_mean'), color='.25', lw=1, ls='--', zorder=3)
                ax.scatter(table.index, table[v + '_climatology_mean'], s=9, color='.15', zorder=4)
            else:
                ax.set_ylim(0, 1)
                ax.set_yticks([])
                ax.text(0.5, 0.5, 'SWC unavailable in the archive', ha='center', va='center', transform=ax.transAxes, color='.4')
        for ti, (trait, color, unit) in enumerate(zip(TRAITS, COLORS, UNITS)):
            ax = axes[ti + 3, ci]
            ax.plot(table.index, table[trait], color=color, lw=1.25, marker='.', ms=3, zorder=3)
            f = frames[site]
            ref = f[f.kind.eq('EnMAP reference') & f.trait.eq(trait) & f['plot'].astype(str).str.lower().eq('true')]
            ax.scatter(ref.date, ref['mean'], facecolors='none', edgecolors=color, linewidths=1.1, s=27, zorder=5)
            ax.set_ylabel(trait + '\n(' + unit + ')', color=color)
            r = correlations[(correlations.site == site) & (correlations.trait == trait) & (correlations.flux == 'GPP_NT_VUT_REF')].set_index('season')

            def fmt(season):
                rho = r.loc[season, 'spearman_rho']
                return ('N/A' if pd.isna(rho) else f'{rho:.2f}') + f" (n={int(r.loc[season, 'n'])})"
            ax.text(0.015, 0.96, '$\\rho$ = ' + fmt('all_dates') + '; Apr–Oct $\\rho$ = ' + fmt('Apr-Oct'), transform=ax.transAxes, va='top', fontsize=8.7, bbox=dict(facecolor='white', alpha=0.88, edgecolor='none', pad=1))
            ax.margins(y=0.29)
        axes[-1, ci].xaxis.set_major_locator(mdates.YearLocator())
        axes[-1, ci].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axes[-1, ci].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
        axes[-1, ci].set_xlabel('Date · 2022-06 to 2026-09')
    handles = [Line2D([], [], color='.72', lw=0.7, label='QC-passed daily / daily climatological mean'), Line2D([], [], color='.2', marker='.', ls='', label='±7 d mean at product dates'), Line2D([], [], color='.25', ls='--', label='DE-SfN climatology'), Patch(facecolor='.65', alpha=0.25, label='Interannual P10–P90'), Line2D([], [], color='.3', marker='o', mfc='none', ls='', label='EnMAP reference')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.53, 0.98), ncol=3, frameon=False, fontsize=9, handlelength=1.8, columnspacing=1.5)
    fig.text(0.07, 0.035, 'Blue-grey shading: Nov–Feb. Traits and W7 gaps are unchanged. ' + '$\\rho$ uses concurrent GPP only; Apr–Oct is a subset.' + '\nDE-SfN climatology is repeated by year; tower–window centre distance: 2.37 km (tower outside). Hesse distance: 0.47 km (inside).', fontsize=9, va='center')
    dest = HERE / 'FigS_flux_vs_traits.png'
    arch = HERE / 'archive_v1/FigS_flux_vs_traits_v1_climatology.png'
    if not arch.exists():
        assert dest.exists()
        dest.rename(arch)
    fig.savefig(dest, dpi=350, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    assert hashes == {str(p): sha(p) for p in sourcepaths}
    provenance = dict(version=2, trait_sha256=hashes, trait_unchanged_since_v1=True, SWC_depth=depth, Hesse_member=hm, Schechenfilz_member=sm, Hesse_qc=qstats, Hesse_n_daily=len(hd), Schechenfilz_n_daily=len(sd), Schechenfilz_SWC_available=bool(sd.SWC.notna().any()), Hesse_n_aligned={v: int(tables['Hesse'][v + '_mean'].notna().sum()) for v in FLUX}, HH_mean_rule='Mean of all finite HH values on QC-passed days; good HH count / 48 >=0.5', QC_good_codes=[0, 1], GPP_conversion=1.03772448, ET_conversion=86400 / 2450000.0, time_zone='local standard time: use TIMESTAMP_START without DST', cpu_threads=1, sealed_value_reads=0, snow_rule_changed=False, v1_png_sha256=sha(arch), v2_png_sha256=sha(dest))
    (WORK / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))
    print(correlations[correlations.site == 'Hesse'].to_string(index=False))
if __name__ == '__main__':
    main()
