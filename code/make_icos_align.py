'fixed Fig4 production cubes,ICOS daily/ancillary;writes SI4aligned tables;production run.\nInputs: code/paths.py; outputs: explicit derivation work directory, never upstream.\nOriginal: upstream://V6/final_figures_20261113/si4/build_si4_data.py.\nRun IDs: cap2048_v3_nested11_20261108; cap2048_spatial_gsingle_20261112; cap2048_production_nn_temporal_20261112.\nHeavy derivation was not rerun when packaging. Use --help before --run.'
from __future__ import annotations
import paths,sys
if __name__=="__main__":
 from make_runtime import cli
 cli(__doc__)
sys.path.insert(0,str(paths.MODEL_CODE))
import os
from pathlib import Path
O = paths.WORK
V = paths.V6
F = V / 'flux_compare'
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLCONFIGDIR'] = str(O / 'mplconfig')
import hashlib, json, zipfile, importlib.util
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from pyproj import Transformer, Geod
spec = importlib.util.spec_from_file_location('core', Path(__file__).with_name('make_flux_core.py'))
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
os.environ['MPLCONFIGDIR'] = str(O / 'mplconfig')
SITES = {'Hesse': dict(station='FR-Hes', rows=[480, 530], cols=[460, 510]), 'Castelporziano': dict(station='IT-Cp2', rows=[475, 525], cols=[475, 525])}
Q = ['Chl', 'CCC', 'LAI']
SEEDS = [42, 43, 44, 45, 46]

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def field_data():
    h = pd.read_csv(V / 'hesse_drought_2022_2026/ancillary_LAI_wide.csv', na_values=-9999)
    archive = F / 'icos_download_20261113/IT-Cp2_ancillary_archive/ICOSETC_IT-Cp2_ARCHIVE_INTERIM_L2.zip'
    with zipfile.ZipFile(archive) as z:
        member = next((n for n in z.namelist() if '_ANCILLARY_' in n))
        with z.open(member) as f:
            a = pd.read_csv(f, na_values=-9999, encoding='utf-8', encoding_errors='replace')
        lai = a[a.VARIABLE_GROUP.eq('GRP_LAI')]
        lai.to_csv(O / 'IT_Cp2_GRP_LAI_long.csv', index=False)
        cp = lai.pivot_table(index='GROUP_ID', columns='VARIABLE', values='DATAVALUE', aggfunc='first').reset_index()
        cp.to_csv(O / 'IT_Cp2_GRP_LAI_wide.csv', index=False)
        groups = a[a.DATAVALUE.astype(str).str.contains('DHP|hemispher|Sunscan|PAI', case=False, regex=True)].VARIABLE_GROUP.value_counts()
        groups.rename_axis('group').reset_index(name='matching_records').to_csv(O / 'IT_Cp2_field_method_search.csv', index=False)
    notes = []
    excluded = []
    result = {}
    for site, wide, method in [('Hesse', h, 'SunScan'), ('Castelporziano', cp, 'DHP / hemispherical photo')]:
        assert wide.LAI_TYPE.dropna().eq('PAI').all()
        w = wide[wide.LAI_TYPE.eq('PAI') & wide.LAI_CANOPY_TYPE.eq('Overstory')].copy()
        w['LAI'] = pd.to_numeric(w.LAI, errors='coerce')
        d = pd.to_numeric(w.LAI_DATE, errors='coerce')
        w['date'] = pd.to_datetime(d.astype('Int64').astype(str), format='%Y%m%d', errors='coerce')
        undated = w[w.date.isna()]
        undated.to_csv(O / f'field_undated_{site}.csv', index=False)
        w = w[w.date.notna()]
        assert not w.duplicated(['date', 'LAI_STATISTIC']).any()
        field = w.pivot(index='date', columns='LAI_STATISTIC', values='LAI').rename(columns={'Mean': 'PAI_mean', 'Standard Deviation': 'PAI_sd', 'Minimum': 'PAI_min', 'Maximum': 'PAI_max'}).reset_index()
        n = w[w.LAI_STATISTIC.eq('Mean')].set_index('date').LAI_STATISTIC_NUMBER
        field['n_plots'] = pd.to_numeric(field.date.map(n), errors='coerce')
        field['method'] = method
        field['canopy'] = 'Overstory'
        bad = field.PAI_mean.notna() & ~field.PAI_mean.between(0, 12)
        for _, r in field[bad].iterrows():
            excluded.append(dict(site=site, date=r.date, mean=r.PAI_mean, sd=r.PAI_sd, reason='physically invalid aggregate mean; excluded from display/availability', source_record_unchanged=True))
        if site == 'Hesse':
            assert field.loc[bad, 'date'].tolist() == [pd.Timestamp('2025-07-09')]
        field = field[~bad & field.PAI_mean.notna()].sort_values('date').reset_index(drop=True)
        field.to_csv(O / f'field_PAI_{site}.csv', index=False)
        result[site] = field
        notes.append(dict(site=site, method=method, type='PAI', canopy='Overstory', source=str(V / 'hesse_drought_2022_2026/ancillary_LAI_wide.csv') if site == 'Hesse' else str(archive) + '!' + member, PID='11676/hnh5IHKw9yDIC3RbGWvvFidz' if site == 'Hesse' else '11676/fFH_pQ3l3LLjQpw2QEGv2hWF', first_date=str(field.date.min().date()), last_date=str(field.date.max().date()), undated_mean_records=int(undated.LAI_STATISTIC.eq('Mean').sum())))
    pd.DataFrame(excluded).to_csv(O / 'field_excluded.csv', index=False)
    pd.DataFrame(notes).to_csv(O / 'field_sources.csv', index=False)
    return result

def model(site, geometry, seedrows):
    path = V / f'production_nn/{site}/trait_cube.npz'
    manifest = json.loads((path.parent / 'manifest.json').read_text())
    assert sha(path) == manifest['files']['trait_cube.npz']
    m = manifest['grid']
    a, b, c, d, e, f = m['transform'][:6]
    crs = m['crs']
    metadata = json.loads((F / f"icos_download_20261113/{SITES[site]['station']}_FLUXMET/metadata.json").read_text())
    loc = metadata['specificInfo']['acquisition']['station']['location']
    lon, lat = (loc['lon'], loc['lat'])
    tx, ty = Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform(lon, lat)
    tc, tr = np.linalg.solve(np.array([[a, b], [d, e]]), np.array([tx - c, ty - f]))
    tower_row, tower_col = (int(np.floor(tr)), int(np.floor(tc)))
    r0, r1 = SITES[site]['rows']
    c0, c1 = SITES[site]['cols']
    sr0, sc0 = (tower_row - 8, tower_col - 8)
    assert r0 <= sr0 and sr0 + 17 <= r1 and (c0 <= sc0) and (sc0 + 17 <= c1)
    cx, cy = np.array([[a, b], [d, e]]) @ np.array([(c0 + c1) / 2, (r0 + r1) / 2]) + [c, f]
    clon, clat = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform(cx, cy)
    geometry.append(dict(site=site, CRS=crs, window_rows=f'{r0}:{r1}', window_cols=f'{c0}:{c1}', tower_parent_row=tr, tower_parent_col=tc, tower_containing_pixel_row=tower_row, tower_containing_pixel_col=tower_col, tower_subset_rows=f'{sr0}:{sr0 + 17}', tower_subset_cols=f'{sc0}:{sc0 + 17}', subset_size_m=510, tower_lat=lat, tower_lon=lon, centre_lat=clat, centre_lon=clon, tower_centre_distance_m=Geod(ellps='WGS84').inv(lon, lat, clon, clat)[2]))
    out = {}
    with np.load(path, allow_pickle=False) as z:
        assert z['per_seed'].shape[0] == 5 and np.array_equal(z['seeds'], SEEDS)
        assert np.array_equal(z['parent_y'], np.arange(r0, r1)) and np.array_equal(z['parent_x'], np.arange(c0, c1))
        names = [str(t).split(' ')[0] for t in z['traits']]
        ic, il = (names.index('Chl'), names.index('LAI'))
        ps, mean, forest, support = (z['per_seed'], z['mean'], z['forest_mask'].astype(bool), z['w7_support'].astype(bool))
        dates = pd.DatetimeIndex(pd.to_datetime(z['dates'].astype(str), utc=True, format='mixed')).tz_localize(None)
        fort = z['is_fortnightly'].astype(bool)
        assert fort.sum() == 113
        subset = np.zeros((50, 50), bool)
        subset[sr0 - r0:sr0 - r0 + 17, sc0 - c0:sc0 - c0 + 17] = True
        for label, spatial in [('Fig4_window', np.ones((50, 50), bool)), ('tower_500m', subset)]:
            mask = forest & spatial
            rows = []
            for i in np.flatnonzero(fort):
                keep = mask & support[i]
                n = int(keep.sum())
                coverage = n / int(mask.sum())
                good = coverage >= 0.8
                row = dict(date=dates[i], site=site, region=label, forest_pixels=int(mask.sum()), supported_forest_pixels=n, W7_coverage=coverage, gate80=good)
                if good:
                    quantities = {'Chl': ps[:, i, ic], 'LAI': ps[:, i, il], 'CCC': ps[:, i, ic] * ps[:, i, il] * 0.01}
                    for q, v in quantities.items():
                        perseed = np.array([float(v[s][keep].mean()) for s in range(5)])
                        row[q] = float(mean[i, ic if q == 'Chl' else il][keep].mean()) if q != 'CCC' else float(v.mean(0)[keep].mean())
                        row[q + '_lo'] = float(perseed.min())
                        row[q + '_hi'] = float(perseed.max())
                        for s, value in zip(SEEDS, perseed):
                            seedrows.append(dict(site=site, region=label, date=dates[i], quantity=q, seed=s, forest_mean=value))
                    row['CCC_product_of_spatial_means_diagnostic'] = row['Chl'] * row['LAI'] * 0.01
                else:
                    for q in Q:
                        row[q] = row[q + '_lo'] = row[q + '_hi'] = np.nan
                rows.append(row)
            frame = pd.DataFrame(rows).set_index('date')
            out[label] = frame
    return out

def shape(series, year, baseline, peak):
    v = series[series.index.year == year].dropna()
    diff = np.abs(np.diff(v.to_numpy()))
    noise = float(np.median(diff)) if len(diff) else np.nan
    amp = peak - baseline
    flag = bool(np.isfinite(amp) and (amp <= 0 or (np.isfinite(noise) and amp < 2 * noise)))
    return (amp, noise, flag)

def timing(site, f):
    rows = []
    peaks = json.loads((F / 'work_v4/PRE_SPECIFIED_PEAK_WINDOWS.json').read_text())
    mismatch = peaks.get(site, {'template_peak_mismatch': False})['template_peak_mismatch']
    for year in range(2022, 2027):
        sos = {v: core.phenology_one(f[v], year, 'SOS', 0.2) for v in ['GPP'] + Q}
        for quantity in Q:
            for metric in ['SOS', 'peak']:
                if metric == 'SOS' and year == 2022:
                    continue
                info = {}
                flags = []
                for v in ['GPP', quantity]:
                    if metric == 'SOS':
                        r = sos[v].copy()
                        amp, noise, bad = shape(f[v], year, r['baseline'], r['peak'])
                        r['candidate_date'] = r['date']
                    else:
                        g = f.loc[(f.index.year == year) & f.index.month.isin(range(4, 11)), v].dropna()
                        base = f.loc[(f.index.year == year) & f.index.month.isin([1, 2, 3]), v].median()
                        candidate = g.idxmax() if len(g) else pd.NaT
                        peak = g.max() if len(g) else np.nan
                        amp, noise, bad = shape(f[v], year, sos[v]['baseline'], sos[v]['peak'])
                        r = dict(candidate_date=candidate, date=candidate, peak=peak, baseline=base, bracket_start=candidate, bracket_end=candidate, bracket_days=0.0 if len(g) else np.nan, n_baseline=int(f.loc[(f.index.year == year) & f.index.month.isin([1, 2, 3]), v].notna().sum()), n_summer=int(len(g)))
                    r.update(amplitude=amp, noise=noise, no_resolvable_seasonal_cycle=bad)
                    if bad:
                        flags.append(v + ': no resolvable seasonal cycle')
                    info[v] = r
                if mismatch:
                    flags.append('template peak mismatch')
                app = bool(flags)
                a, b = (info[quantity], info['GPP'])
                possible = pd.notna(a['candidate_date']) and pd.notna(b['candidate_date'])
                if not possible:
                    flags.append('missing baseline/peak/crossing or S2/W7 support')
                wide = bool(metric == 'SOS' and possible and (max(a['bracket_days'], b['bracket_days']) > 16))
                if wide:
                    flags.append('uncertain: bracket >16 d')
                partial = bool(metric == 'peak' and year in [2022, 2026])
                if partial:
                    flags.append('partial Apr-Oct record')
                if metric == 'peak' and any((pd.isna(x['baseline']) for x in info.values())):
                    flags.append('seasonal-cycle baseline unavailable')
                row = dict(site=site, year=year, quantity=quantity, metric=metric, threshold=0.2 if metric == 'SOS' else np.nan, growing_season='Apr-Oct' if metric == 'peak' else 'Jan-Mar baseline / Jun-Aug peak', GPP_candidate_date=b['candidate_date'], trait_candidate_date=a['candidate_date'], candidate_delta_days=(a['candidate_date'] - b['candidate_date']) / pd.Timedelta(days=1) if possible else np.nan, GPP_date=b['candidate_date'] if not app else pd.NaT, trait_date=a['candidate_date'] if not app else pd.NaT, delta_days=(a['candidate_date'] - b['candidate_date']) / pd.Timedelta(days=1) if possible and (not app) else np.nan, GPP_bracket_start=b['bracket_start'], GPP_bracket_end=b['bracket_end'], GPP_bracket_days=b['bracket_days'], trait_bracket_start=a['bracket_start'], trait_bracket_end=a['bracket_end'], trait_bracket_days=a['bracket_days'], GPP_amplitude=b['amplitude'], trait_amplitude=a['amplitude'], GPP_noise=b['noise'], trait_noise=a['noise'], template_not_applicable=app, partial_record=partial, flag='template not applicable: ' + '; '.join(flags) if app else '; '.join(flags) if flags else 'applicable')
                rows.append(row)
    return rows

def main():
    protected = {str(p): sha(p) for p in F.rglob('*') if p.is_file() and p.suffix in ['.csv', '.png', '.py', '.json', '.md'] and ('IT-Cp2_ancillary_archive' not in p.parts) and (p.name != 'download_manifest.csv')}
    (O / 'protected_input_hashes.json').write_text(json.dumps(protected, indent=2))
    inputs = [V / f'production_nn/{s}/{f}' for s in SITES for f in ['trait_cube.npz', 'manifest.json']]
    inputs += [V / 'hesse_drought_2022_2026/ancillary_LAI_wide.csv', V / 'hesse_lai_compare/plot_hesse_lai.py', V / 'hesse_lai_compare/plot_hesse_canopy_gpp.py']
    hashes = {str(p): sha(p) for p in inputs}
    field = field_data()
    geometry = []
    seedrows = []
    stats = []
    events = []
    available = []
    for site in SITES:
        data = model(site, geometry, seedrows)
        if site == 'Hesse':
            flux = pd.read_csv(F / 'flux_vs_traits_aligned_Hesse.csv', parse_dates=['product_date']).set_index('product_date').GPP_NT_VUT_REF_mean
            daily = pd.read_csv(F / 'work_v2/Hesse_daily_QC.csv', parse_dates=['date']).set_index('date').GPP_NT_VUT_REF
        else:
            flux = pd.read_csv(F / 'flux_aligned_Castelporziano.csv', parse_dates=['date']).set_index('date').GPP
            daily = pd.read_csv(F / 'flux_daily_Castelporziano.csv', parse_dates=['date']).set_index('date').GPP
        daily.rename('GPP').to_csv(O / f'daily_GPP_{site}.csv')
        for region, m in data.items():
            assert m.index.equals(flux.index)
            m['GPP'] = flux
            m.to_csv(O / (f'model_series_{site}.csv' if region == 'Fig4_window' else f'model_series_500m_{site}.csv'))
            for q in Q:
                for season in ['all_dates', 'Apr-Oct']:
                    matched = m[[q, 'GPP']]
                    if season == 'Apr-Oct':
                        matched = matched[matched.index.month.isin(range(4, 11))]
                    matched = matched.dropna()
                    n = len(matched)
                    rho = float(spearmanr(matched[q], matched.GPP).statistic) if n >= 3 else np.nan
                    stats.append(dict(site=site, region=region, quantity=q, season=season, spearman_rho=rho, n=n))
            if region == 'Fig4_window':
                events.extend(timing(site, m))
        for year in range(2022, 2027):
            f = field[site]
            yearfield = f[f.date.dt.year.eq(year)]
            m = data['Fig4_window']
            available.append(dict(site=site, year=year, GPP_QC_days=int(daily[daily.index.year == year].notna().sum()), GPP_expected_days=243 if year == 2026 else 366 if year == 2024 else 365, ground_PAI_dates=int(yearfield.date.nunique()), ground_PAI_dates_in_figure=int(yearfield[yearfield.date.ge(pd.Timestamp('2022-06-01')) & yearfield.date.le(pd.Timestamp('2026-09-30'))].date.nunique()), ground_method='SunScan overstory' if site == 'Hesse' else 'DHP hemispherical photo overstory', S2_product_dates=int((m.index.year == year).sum()), S2_product_dates_gate80=int(m.loc[m.index.year == year, 'gate80'].sum())))
    pd.DataFrame(geometry).to_csv(O / 'window_geometry.csv', index=False)
    pd.DataFrame(seedrows).to_csv(O / 'per_seed_forest_means.csv', index=False)
    pd.DataFrame(stats).to_csv(O / 'spearman_GPP.csv', index=False)
    pd.DataFrame(events).to_csv(O / 'seasonal_timing.csv', index=False)
    pd.DataFrame(available).to_csv(O / 'availability.csv', index=False)
    assert all((sha(p) == v for p, v in hashes.items()))
    assert all((sha(p) == v for p, v in protected.items()))
    (O / 'source_provenance.json').write_text(json.dumps(dict(input_SHA256=hashes, CCC='per pixel and seed: Chl * LAI * 0.01; seed mean then supported-forest mean', gate='independent 80% W7 forest support gate for each spatial region', timing_peak_window='Apr-Oct; observed gated-product maximum; flag partial 2022/2026', main_window='Fig4 fixed 50x50', sensitivity='tower-containing parent pixel +/-8 (17x17,510m), clipped check, forest-only, own80% gate', cpu_threads=1, no_sealed_reads=True), indent=2))
    print(pd.DataFrame(available).to_string(index=False), flush=True)
    print(pd.DataFrame(stats).to_string(index=False), flush=True)
    print(pd.DataFrame(events)[['site', 'year', 'quantity', 'metric', 'GPP_date', 'trait_date', 'delta_days', 'flag']].to_string(index=False), flush=True)
if __name__ == '__main__':
    main()
