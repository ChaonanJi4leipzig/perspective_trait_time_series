'downloaded fixed-source ICOS archives and precedence receipts;original HH-to-daily QC;production run.\nInputs: code/paths.py; outputs: explicit derivation work directory, never upstream.\nOriginal: upstream://V6/flux_compare/work_v3/process_three.py.\nRun IDs: cap2048_v3_nested11_20261108; cap2048_spatial_gsingle_20261112; cap2048_production_nn_temporal_20261112.\nHeavy derivation was not rerun when packaging. Use --help before --run.'
from __future__ import annotations
import paths,sys
if __name__=="__main__":
 from make_runtime import cli
 cli(__doc__)
sys.path.insert(0,str(paths.MODEL_CODE))
import os
from pathlib import Path
W = paths.WORK
P = paths.FLUX
J = paths.PROJECT
D = P / 'icos_download_20261113'
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[k] = '1'
os.environ['MPLCONFIGDIR'] = str(paths.WORK / '.mplconfig')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib, json, zipfile, io, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
SITES = {'IT-Cp2': 'Castelporziano', 'NL-Loo': 'Loobos', 'FI-Hyy': 'Hyytiala'}
FLUX = ['GPP', 'ET', 'SWC']
FACTOR = {'GPP': 1.03772448, 'ET': 86400 / 2450000.0, 'SWC': 1.0}
RAW = {'GPP': 'GPP_NT_VUT_REF', 'ET': 'LE_F_MDS', 'SWC': 'SWC_F_MDS_1'}
QC = {'GPP': 'NEE_VUT_REF_QC', 'ET': 'LE_F_MDS_QC', 'SWC': 'SWC_F_MDS_1_QC'}
STOP = pd.Timestamp('2026-08-31')
DATES = pd.date_range('2022-01-01', STOP, name='date')
POLICY = json.loads(paths.ICOS_PRECEDENCE.read_text())

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def read_csv_member(path, pattern, usecols=None):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if pattern in n and n.endswith('.csv')]
        assert len(names) == 1, (path, pattern, names)
        with z.open(names[0]) as f:
            df = pd.read_csv(f, na_values=-9999, usecols=usecols, encoding='utf-8', encoding_errors='replace')
    return (df, names[0])

def depth_schedule(site, which):
    if which == 'primary':
        path = next(W.glob(site + '_ICOSETC_*VARINFO_FLUXNET_HH_INTERIM_L2.csv'))
    else:
        path = W / (site + '_BIFVARINFO_DD_SWC.csv')
    df = pd.read_csv(path)
    ids = df.loc[df.VARIABLE.eq('VAR_INFO_VARNAME') & df.DATAVALUE.eq('SWC_F_MDS_1'), 'GROUP_ID']
    rows = []
    for gid, g in df[df.GROUP_ID.isin(ids)].groupby('GROUP_ID'):
        v = g.set_index('VARIABLE').DATAVALUE.to_dict()
        stamp = str(v.get('VAR_INFO_DATE', ''))
        date = pd.to_datetime(stamp[:8], format='%Y%m%d', errors='coerce')
        height = pd.to_numeric(v.get('VAR_INFO_HEIGHT'), errors='coerce')
        if pd.notna(date) and pd.notna(height):
            rows.append(dict(date=date, depth_cm=-100 * height, group_id=int(gid), n_profiles=v.get('VAR_INFO_AGG_NUMLOC'), model=v.get('VAR_INFO_MODEL')))
    return (sorted(rows, key=lambda r: r['date']), str(path))

def map_depth(index, schedule):
    values = np.full(len(index), np.nan)
    for r in schedule:
        values[pd.DatetimeIndex(index) >= r['date']] = r['depth_cm']
    return values

def primary_daily(site):
    rec = next((r for r in POLICY['selected_products'] if r['site'] == site and r['purpose'] == 'partitioned_GPP_and_LE'))
    path = D / rec['label'] / rec['file_name']
    meta = json.loads((path.parent / 'metadata.json').read_text())
    tz = meta['specificInfo']['acquisition']['station']['specificInfo']['timeZoneOffset']
    assert tz == {'FI-Hyy': 2}.get(site, 1)
    cols = ['TIMESTAMP_START', 'TIMESTAMP_END'] + list(RAW.values()) + list(QC.values())
    raw, member = read_csv_member(path, 'FLUXMET_HH_', cols)
    starts = pd.DatetimeIndex(pd.to_datetime(raw.TIMESTAMP_START.astype(str), format='%Y%m%d%H%M'))
    ends = pd.DatetimeIndex(pd.to_datetime(raw.TIMESTAMP_END.astype(str), format='%Y%m%d%H%M'))
    assert starts.is_unique and starts.is_monotonic_increasing and (ends - starts == pd.Timedelta(minutes=30)).all()
    assert starts[0] == pd.Timestamp(meta['specificInfo']['acquisition']['interval']['start']).tz_localize(None) + pd.Timedelta(hours=tz)
    assert ends[-1] == pd.Timestamp(meta['specificInfo']['acquisition']['interval']['stop']).tz_localize(None) + pd.Timedelta(hours=tz)
    index = pd.date_range(starts[0].normalize(), starts[-1].normalize(), name='date')
    raw.index = starts
    out = pd.DataFrame(index=index)
    for v in FLUX:
        vals = raw[RAW[v]]
        good = raw[QC[v]].isin([0, 1]) & vals.notna()
        out[v + '_raw_daily_mean'] = vals.resample('D').mean() * FACTOR[v]
        out[v + '_n_good_hh'] = good.resample('D').sum()
        out[v + '_n_finite_hh'] = vals.resample('D').count()
        out[v + '_qc_fraction'] = out[v + '_n_good_hh'] / 48
        out[v] = out[v + '_raw_daily_mean'].where(out[v + '_qc_fraction'] >= 0.5)
    out['source_role'] = 'primary'
    out['source'] = '11676/' + rec['selected_PID'] + '!' + member
    out['file'] = str(path)
    out['UTC_offset_hours'] = tz
    schedule, depth_file = depth_schedule(site, 'primary')
    out['SWC_depth_cm'] = map_depth(out.index, schedule)
    return (out, dict(PID=rec['selected_PID'], file=str(path), member=member, sha256=sha(path), UTC_offset_hours=tz, n_HH=len(raw), expected_HH=len(index) * 48, missing_HH_slots=len(index) * 48 - len(raw), SWC_schedule=schedule, SWC_metadata=depth_file))

def local_daily(site):
    path = J / POLICY['local_sources'][site]
    cols = ['TIMESTAMP'] + list(RAW.values()) + list(QC.values())
    raw, member = read_csv_member(path, 'FLUXMET_DD_', cols)
    out = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(raw.TIMESTAMP.astype(str), format='%Y%m%d'), name='date'))
    for v in FLUX:
        factor = FACTOR[v] if v != 'GPP' else 1.0
        out[v + '_raw_daily_mean'] = raw[RAW[v]].to_numpy() * factor
        out[v + '_qc_fraction'] = raw[QC[v]].to_numpy()
        out[v] = out[v + '_raw_daily_mean'].where(out[v + '_qc_fraction'] >= 0.5)
        out[v + '_n_good_hh'] = np.nan
        out[v + '_n_finite_hh'] = np.nan
    schedule, df = depth_schedule(site, 'local')
    out['SWC_depth_cm'] = map_depth(out.index, schedule)
    pid = {'IT-Cp2': 'B6lVx5OEcdr4ZS-3ix2K7iPP', 'NL-Loo': 'WewhyVCaeifJzeZN8L6CdbW3', 'FI-Hyy': 'pIDsMGxYyXTg31dHecNO55Nq'}[site]
    out['source'] = '11676/' + pid + '!' + member
    out['file'] = str(path)
    out['source_role'] = 'fallback'
    out['UTC_offset_hours'] = 2 if site == 'FI-Hyy' else 1
    return (out, dict(PID=pid, file=str(path), member=member, sha256=sha(path), SWC_schedule=schedule, SWC_metadata=df))

def build_climatology(site, short, old, target_depth):
    rows = []
    used = {}
    cal = pd.date_range('2001-01-01', '2001-12-31').strftime('%m-%d')
    climat = pd.DataFrame(index=pd.Index(cal, name='month_day'))
    hist = old.loc['2015-01-01':'2021-12-31'].copy()
    for v in FLUX:
        acceptable = []
        for year in range(2015, 2022):
            days = pd.date_range(f'{year}-01-01', f'{year}-12-31')
            g = hist.reindex(days)
            count = int(g[v].notna().sum())
            fraction = count / len(days)
            match = g.SWC_depth_cm.eq(target_depth) if v == 'SWC' else pd.Series(True, index=g.index)
            good_depth = bool(match.loc[g[v].notna()].all()) if count else False
            accepted = fraction >= 0.8 and (v != 'SWC' or good_depth)
            if accepted:
                acceptable.append(year)
            rows.append(dict(site=site, short_name=short, variable=v, year=year, days=len(days), n_qc_days=count, qc_fraction=fraction, target_depth_cm=target_depth if v == 'SWC' else np.nan, depth_matches=good_depth if v == 'SWC' else True, qualifying=accepted, source=old.source.iloc[0], file=old.file.iloc[0]))
        used[v] = dict(qualifying_years=acceptable, n_qualifying_years=len(acceptable), climatology_available=len(acceptable) >= 4, primary_anomaly_reference='historical 2015-2021' if len(acceptable) >= 4 else 'LOYO 2022-2026', depth_cm=target_depth if v == 'SWC' else None)
        if len(acceptable) >= 4:
            h = hist.loc[hist.index.year.isin(acceptable)].copy()
            h = h[~((h.index.month == 2) & (h.index.day == 29))]
            daily_mean = h.groupby(h.index.strftime('%m-%d'))[v].mean().reindex(cal).to_numpy()
            wrapped = np.r_[daily_mean[-7:], daily_mean, daily_mean[:7]]
            mean = pd.Series(wrapped).rolling(15, center=True, min_periods=8).mean().iloc[7:-7].to_numpy()
            climat[v + '_daily_climatology'] = daily_mean
            climat[v + '_climatology'] = mean
        else:
            climat[v + '_daily_climatology'] = np.nan
            climat[v + '_climatology'] = np.nan
    hist.to_csv(W / f'historical_daily_{short}.csv')
    climat.to_csv(W / f'climatology_{short}.csv')
    return (climat, rows, used)

def map_clim(c, index, col):
    return np.array([(c.loc['02-28', col] + c.loc['03-01', col]) / 2 if (d.month, d.day) == (2, 29) else c.loc[d.strftime('%m-%d'), col] for d in index])

def main():
    trajectory = P.parent / 'source_data/Fig4_trajectories_Hesse.csv'
    grid = pd.read_csv(trajectory, usecols=['date', 'kind'])
    grid = grid[grid.kind.eq('Production NN')]
    dates = pd.DatetimeIndex(pd.to_datetime(grid.date, utc=True, format='mixed').dt.tz_localize(None).drop_duplicates().sort_values())
    assert len(dates) == 113 and (dates.to_series().diff().dropna() == pd.Timedelta(days=14)).all()
    coverage = []
    history_acceptance = []
    flags = []
    provenance = {}
    aligned_counts = []
    for site, short in SITES.items():
        primary, pmeta = primary_daily(site)
        old, lmeta = local_daily(site)
        daily = primary.reindex(DATES)
        for year in range(2022, 2027):
            mask = daily.index.year == year
            if year not in set(primary.index.year):
                daily.loc[mask, old.columns] = old.reindex(daily.index[mask]).to_numpy()
        assert daily.source.notna().all()
        assert daily.groupby(daily.index.year).source.nunique().eq(1).all()
        for v in FLUX:
            daily[v + '_QC_failed'] = daily[v + '_qc_fraction'].lt(0.5) | daily[v + '_qc_fraction'].isna()
        daily['GPP_below_minus1'] = daily.GPP.lt(-1)
        daily['SWC_outside_0_100'] = daily.SWC.lt(0) | daily.SWC.gt(100)
        for v in ['GPP', 'SWC']:
            raw = daily[v + '_raw_daily_mean']
            bad = raw.lt(-1) if v == 'GPP' else raw.lt(0) | raw.gt(100)
            for date, r in daily.loc[bad].iterrows():
                flags.append(dict(site=site, short_name=short, date=date, variable=v, value=r[v + '_raw_daily_mean'], QC_passed=pd.notna(r[v]), reason='GPP < -1 gC m-2 d-1' if v == 'GPP' else 'SWC outside 0-100%', source=r.source))
        target_depth = float(primary.loc[primary.index >= pd.Timestamp('2022-01-01'), 'SWC_depth_cm'].dropna().mode().iloc[0])
        clim, accept, used = build_climatology(site, short, old, target_depth)
        history_acceptance += accept
        for v in FLUX:
            daily[v + '_climatology'] = map_clim(clim, daily.index, v + '_climatology')
        daily.to_csv(P / f'flux_daily_{short}.csv')
        aligned = pd.DataFrame(index=dates)
        aligned.index.name = 'date'
        for v in FLUX:
            counts = []
            means = []
            contributors = []
            for date in dates:
                d = date.normalize()
                window = daily.loc[d - pd.Timedelta(days=7):d + pd.Timedelta(days=7)]
                valid = window[window[v].notna()]
                n = len(valid)
                counts.append(n)
                means.append(valid[v].mean() if n >= 8 and date.normalize() <= STOP else np.nan)
                contributors.append(';'.join(sorted(valid.source.unique())))
            aligned[v] = means
            aligned['n_days_' + v] = counts
            aligned[v + '_climatology'] = map_clim(clim, dates, v + '_climatology')
            aligned['source_contributing_' + v] = contributors
            if v == 'SWC':
                for date in dates:
                    d = date.normalize()
                    win = daily.loc[d - pd.Timedelta(days=7):d + pd.Timedelta(days=7)]
                    deps = win.loc[win.SWC.notna(), 'SWC_depth_cm'].dropna().unique()
                    if len(deps) != 1 or not np.isclose(deps[0], target_depth):
                        aligned.loc[date, 'SWC_climatology'] = np.nan
        sources = []
        roles = []
        depths = []
        depth_notes = []
        for date in dates:
            d = date.normalize()
            win = daily.loc[d - pd.Timedelta(days=7):d + pd.Timedelta(days=7)]
            sources.append(';'.join(sorted(win.source.unique())) if len(win) else 'No coverage after 2026-08-31')
            roles.append(';'.join(sorted(win.source_role.unique())) if len(win) else 'missing')
            deps = win.loc[win.SWC.notna(), 'SWC_depth_cm'].dropna().unique()
            depths.append(float(deps[0]) if len(deps) == 1 else np.nan)
            depth_notes.append(';'.join(map(lambda v: f'{v:g} cm', deps)) if len(deps) else 'no QC-passed SWC')
        aligned['source'] = sources
        aligned['source_role'] = roles
        aligned['SWC_depth_cm'] = depths
        aligned['SWC_depth_note'] = depth_notes
        assert aligned.loc[aligned.index >= pd.Timestamp('2026-09-01'), FLUX].isna().all().all()
        aligned.to_csv(P / f'flux_aligned_{short}.csv')
        for year, g in daily.groupby(daily.index.year):
            for v in FLUX:
                n = int(g[v].notna().sum())
                coverage.append(dict(site=site, short_name=short, year=int(year), variable=v, n_qc_days=n, expected_days=len(g), QC_fraction=n / len(g), period='Jan-Aug' if year == 2026 else 'full calendar year', full_year_fraction=n / (366 if year % 4 == 0 else 365), source_role=g.source_role.iloc[0], source=g.source.iloc[0], SWC_depths_cm=';'.join(map(str, g.SWC_depth_cm.dropna().unique())) if v == 'SWC' else ''))
        aligned_counts.append(dict(site=site, short_name=short, **{v: int(aligned[v].notna().sum()) for v in FLUX}))
        provenance[site] = dict(short_name=short, primary=pmeta, local=lmeta, climatology=used, nominal_primary_years=sorted(set(primary.index.year)), fallback_years=sorted(set(daily.loc[daily.source_role.eq('fallback')].index.year)), concurrent_SWC_depths_cm=daily.groupby(daily.index.year).SWC_depth_cm.apply(lambda s: sorted(s.dropna().unique().tolist())).to_dict())
        print('PROCESSED', site, used, flush=True)
        check_plot(site, short, daily)
    pd.DataFrame(coverage).to_csv(W / 'coverage_by_year.csv', index=False)
    pd.DataFrame(history_acceptance).to_csv(W / 'climatology_year_acceptance.csv', index=False)
    pd.DataFrame(flags, columns=['site', 'short_name', 'date', 'variable', 'value', 'QC_passed', 'reason', 'source']).to_csv(W / 'implausible_values.csv', index=False)
    pd.DataFrame(aligned_counts).to_csv(W / 'aligned_counts.csv', index=False)
    (W / 'processing_provenance.json').write_text(json.dumps(provenance, indent=2, default=str))
    protected = json.loads((W / 'protected_existing_files.json').read_text())
    assert all((sha(path) == h for path, h in protected.items())), 'An existing Hesse/Schechenfilz output changed'
    print('EXISTING OUTPUT HASHES UNCHANGED', len(protected), flush=True)

def check_plot(site, short, d):
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6})
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8), sharex=True, gridspec_kw={'hspace': 0.16})
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.095, top=0.88)
    for ax, v, label in zip(axes, FLUX, ['GPP (gC m$^{-2}$ d$^{-1}$)', 'ET (mm d$^{-1}$)', 'SWC (%)']):
        ax.plot(d.index, d[v], color='.45', lw=0.55, zorder=2)
        ax.plot(d.index, d[v + '_climatology'], color='#E69F00', lw=1.1, ls='--', zorder=3)
        failed = d[v + '_QC_failed']
        finite = d[v + '_raw_daily_mean'].notna()
        ax.scatter(d.index[failed & finite], d.loc[failed & finite, v + '_raw_daily_mean'], color='#C44E52', marker='x', s=7, lw=0.5, alpha=0.55, zorder=4)
        missing = failed & ~finite
        ax.scatter(d.index[missing], np.full(missing.sum(), 0.025), transform=ax.get_xaxis_transform(), marker='|', color='#C44E52', s=6, lw=0.5, alpha=0.5)
        fallback = d.source_role.eq('fallback')
        if fallback.any():
            ax.axvspan(d.index[fallback].min(), d.index[fallback].max() + pd.Timedelta(days=1), color='#0072B2', alpha=0.08, lw=0, zorder=0)
        if d[v + '_climatology'].isna().all():
            ax.text(0.99, 0.94, 'No eligible historical climatology (<4 years)', transform=ax.transAxes, ha='right', va='top', fontsize=8.2, bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1))
        ax.set_ylabel(label)
        ax.grid(axis='y', color='.9', lw=0.4)
        ax.set_xlim(pd.Timestamp('2022-01-01'), STOP)
        for year in range(2022, 2027):
            ax.axvline(pd.Timestamp(year, 1, 1), color='.8', lw=0.5, zorder=0)
    axes[0].set_title(f'{short} · {site} — daily QC check (phase 1)', loc='left', fontsize=12, pad=10)
    note = 'SWC_F_MDS_1: 5 cm; 4-profile mean.'
    if site == 'NL-Loo':
        note += ' 2022 mapped fallback: 3 cm; do not pool depths for anomalies.'
    else:
        note += ''
    axes[-1].text(0.01, 0.94, note, transform=axes[-1].transAxes, va='top', fontsize=8.5, bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].set_xlabel('Local standard date (through 2026-08-31)')
    handles = [Line2D([], [], color='.45', lw=0.7, label='QC-passed daily'), Line2D([], [], color='#E69F00', ls='--', label='2015–2021 climatology (when eligible)'), Line2D([], [], color='#C44E52', marker='x', ls='', label='QC-failed daily mean'), Line2D([], [], color='#C44E52', marker='|', ls='', label='Missing daily value')]
    if site == 'NL-Loo':
        handles.append(Patch(facecolor='#0072B2', alpha=0.08, label='Whole-year fallback'))
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.54, 0.985), fontsize=8.8)
    fig.text(0.08, 0.025, 'QC: good HH codes 0/1 ≥50% of 48 slots; local DD fallback uses supplied daily QC fraction. Flags do not remove otherwise QC-passed values.', fontsize=8.5)
    fig.savefig(W / f'flux_check_{short}.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
if __name__ == '__main__':
    main()
