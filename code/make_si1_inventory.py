'development cohort and corrected S2 stores;writes pixel-valid inventory;inventory.\nInputs: code/paths.py; outputs: explicit derivation work directory, never upstream.\nOriginal: upstream://V6/inventories.py.\nRun IDs: cap2048_v3_nested11_20261108; cap2048_spatial_gsingle_20261112; cap2048_production_nn_temporal_20261112.\nHeavy derivation was not rerun when packaging. Use --help before --run.'
from __future__ import annotations
import paths,sys
if __name__=="__main__":
 from make_runtime import cli
 cli(__doc__)
sys.path.insert(0,str(paths.MODEL_CODE))
from make_support import *
from concurrent.futures import ThreadPoolExecutor
import zarr
from numcodecs import blosc
blosc.set_nthreads(1)
zarr.config.set({'async.concurrency': 2, 'threading.max_workers': 2})

def dates(root):
    a = root['time']
    v = a[:]
    if np.issubdtype(a.dtype, np.number):
        u, e = a.attrs['units'].split(' since ')
        return pd.DatetimeIndex(pd.to_datetime(e, utc=True) + pd.to_timedelta(v, unit={'days': 'D', 'hours': 'h', 'minutes': 'm', 'seconds': 's', 'milliseconds': 'ms', 'microseconds': 'us', 'nanoseconds': 'ns'}[u]))
    return pd.DatetimeIndex(pd.to_datetime(v, utc=True))

def wc(site):
    p = PROJECT / f'experiments/enmap_v2_worldcover_source_topup_20261022/worldcover_zarr/site_{site}_worldcover2021_30m.zarr'
    a = zarr.open_array(str(p), mode='r')
    v = np.asarray(a[:])
    return (v.reshape(v.shape[-2:]), dict(a.attrs), str(p))

def site_job(item):
    site, stores = item
    dest = CACHE / f's2_site_{site}.csv'
    gridfile = CACHE / f'support_site_{site}.csv'
    if dest.exists() and gridfile.exists():
        return str(site)
    land, attrs, wpath = wc(site)
    veg = np.isin(land, CONFIG['vegetation_classes'])
    den = int(veg.sum())
    assert den > 0
    timeline = pd.DatetimeIndex(np.concatenate([pd.date_range(f'{year}-01-01 12:00', f'{year}-12-31 12:00', freq='14D', tz='UTC').values for year in range(2022, 2026)])).tz_localize('UTC')
    masks = np.zeros((len(timeline), *land.shape), bool)
    obs = []
    present = set()
    all_dates = []
    for rec in stores.to_dict('records'):
        root = zarr.open_group(rec['path'], mode='r')
        t = dates(root)
        a = root['advanced_valid_mask']
        present.add(int(rec['year']))
        all_dates.extend(t)
        transform = root.attrs.get('grid_reference_transform_30m')
        assert transform and np.allclose(np.array(transform)[:6], attrs['transform'][:6], atol=0.001, rtol=0), (site, 'grid mismatch')
        yy = np.floor((np.arange(land.shape[0]) + 0.5) * a.shape[-2] / land.shape[0]).astype(int)
        xx = np.floor((np.arange(land.shape[1]) + 0.5) * a.shape[-1] / land.shape[1]).astype(int)
        for j, date in enumerate(t):
            valid = np.asarray(a.get_orthogonal_selection((j, yy, xx))) > 0
            n = int((valid & veg).sum())
            obs.append(dict(site_id=site, date_utc=date, year=int(rec['year']), valid_vegetation_pixels=n, vegetation_pixels=den, valid_fraction=n / den, valid_observation=n / den >= CONFIG['valid_observation_threshold'], threshold=CONFIG['valid_observation_threshold'], store=rec['path'], store_attrs_sha256=rec['zattrs_sha256'], rebuild_run=rec['rebuild_run'], source_version=rec['source_version'], vegetation_classes='10|20|30|40|90|95'))
            ix = np.flatnonzero(np.abs((timeline - date).total_seconds()) <= CONFIG['s2_support_window_days'] * 86400)
            for i in ix:
                masks[i] |= valid
    coverage = []
    first = min(all_dates)
    last = max(all_dates)
    for i, date in enumerate(timeline):
        missing = date.year not in present or date < first - pd.Timedelta(days=7) or date > last + pd.Timedelta(days=7)
        fraction = float((masks[i] & veg).sum() / den) if not missing else np.nan
        coverage.append(dict(site_id=site, date_utc=date, year=date.year, season={12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM', 6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}[date.month], valid_fraction=fraction, supported=float(fraction >= CONFIG['valid_observation_threshold']) if not missing else np.nan, threshold=CONFIG['valid_observation_threshold'], vegetation_pixels=den, missing_store_year=date.year not in present, window_days=7, step_days=14, grid='30 m trait centres, nearest categorical mask', worldcover=wpath))
    pd.DataFrame(obs).to_csv(dest, index=False)
    pd.DataFrame(coverage).to_csv(gridfile, index=False)
    print('S2 inventory', site, len(obs), 'observations', flush=True)
    return str(site)

def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    inv = pd.read_csv(paths.INVENTORY_SOURCE / 's2_store_metadata_inventory.csv')
    study_sites = set(pd.read_csv(paths.INVENTORY_SOURCE / 'development_events.csv').site_id)
    inv = inv[inv.site_id.isin(study_sites)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(site_job, list(inv.groupby('site_id', sort=True))))
    csv('tables/s2_observations.csv', pd.concat([pd.read_csv(p) for p in sorted(CACHE.glob('s2_site_*.csv'))], ignore_index=True))
    csv('tables/s2_support_14day.csv', pd.concat([pd.read_csv(p) for p in sorted(CACHE.glob('support_site_*.csv'))], ignore_index=True))
if __name__ == '__main__':
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19
    main()
