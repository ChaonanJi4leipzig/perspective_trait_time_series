'corrected S2,soil vectors,frozen feature recipe;native-window feature helper;production run.\nInputs: code/paths.py; outputs: explicit derivation work directory, never upstream.\nOriginal: upstream://V6/production_nn/prepare_features.py.\nRun IDs: cap2048_v3_nested11_20261108; cap2048_spatial_gsingle_20261112; cap2048_production_nn_temporal_20261112.\nHeavy derivation was not rerun when packaging. Use --help before --run.'
from __future__ import annotations
import paths,sys
if __name__=="__main__":
 from make_runtime import cli
 cli(__doc__)
sys.path.insert(0,str(paths.MODEL_CODE))
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[key] = '1'
from make_production_common import *
import zarr
import numcodecs
from functools import lru_cache
import time
from feature_ops import build
from sequence_backend import initial, update, state_arrays
from batched_features import observations, centre_batch
from grid_mapping import store_grid, same_grid
numcodecs.blosc.set_nthreads(1)
if hasattr(zarr, 'config'):
    zarr.config.set({'async.concurrency': 1, 'threading.max_workers': 1})
PROJECT = paths.PROJECT

class Features:

    def __init__(self, site, yy, xx):
        self.site = site
        self.yy, self.xx = (np.asarray(yy, int), np.asarray(xx, int))
        self.y0 = max(0, (int(self.yy.min()) - 6) // 2 * 2)
        self.x0 = max(0, (int(self.xx.min()) - 6) // 2 * 2)
        self.y1 = min(1000, (int(self.yy.max()) + 8) // 2 * 2)
        self.x1 = min(1000, (int(self.xx.max()) + 8) // 2 * 2)
        require(self.y0 % 2 == self.x0 % 2 == 0, 'Crop phase')
        self.y, self.x = (self.yy - self.y0, self.xx - self.x0)
        lp = PROJECT / f'experiments/enmap_v2_worldcover_source_topup_20261022/worldcover_zarr/site_{site}_worldcover2021_30m.zarr'
        tp = PROJECT / f'input_data/enmap/traits_region_v2/site_{site}_enmap_traits_region_30km_30m.zarr'
        self.grid = store_grid(tp, True)
        require(same_grid(self.grid, store_grid(lp)), 'Formal landcover grid')
        lc = zarr.open_array(str(lp), mode='r')
        self.lc = np.asarray(lc[0, self.y0:self.y1, self.x0:self.x1])
        self.lc_attrs = sha(lp / '.zattrs')
        self.soil = np.load(ROOT / f'feature_reference/{site}/soil.npy')
        self.schema = pd.read_csv(ROOT / 'bundle/schema_input.csv')
        self.scenes, self.stores = ([], {})
        for year in range(2021, 2028):
            p = PROJECT / f'input_data/s2/region/site_{site}/site_{site}_s2_region_{year}.zarr'
            if not p.exists():
                continue
            require(same_grid(self.grid, store_grid(p)), 'S2 formal grid ' + str(p))
            g = zarr.open_group(str(p), mode='r')
            t = g['time']
            unit, origin = t.attrs['units'].split(' since ')
            u = {'days': 'D', 'seconds': 's', 'milliseconds': 'ms', 'microseconds': 'us', 'nanoseconds': 'ns'}[unit]
            times = (np.datetime64(origin) + t[:].astype('timedelta64[' + u + ']')).astype('datetime64[ns]')
            self.stores[str(p)] = dict(group=g, attrs_sha256=sha(p / '.zattrs'), first=str(times.min()), last=str(times.max()), n_scenes=len(times))
            for ti, date in enumerate(times):
                self.scenes.append(dict(path=str(p), time=str(date), index=ti, year=year))
        self.scenes.sort(key=lambda s: (s['time'], s['year'], s['index']))
        self.times = np.array([s['time'] for s in self.scenes], dtype='datetime64[ns]')

    @lru_cache(maxsize=12)
    def load(self, index):
        s = self.scenes[index]
        g = self.stores[s['path']]['group']
        require(sha(Path(s['path']) / '.zattrs') == self.stores[s['path']]['attrs_sha256'], 'S2 changed during read')
        arr = []
        for name, res in [('reflectance_10m', 10), ('reflectance_20m', 20)]:
            a = g[name]
            scale = 30 / res
            raw = np.asarray(a[s['index'], int(self.y0 * scale):int(self.y1 * scale), int(self.x0 * scale):int(self.x1 * scale)])
            v = raw.astype('f4')
            v[raw == a.fill_value] = np.nan
            arr.append(v * np.float32(a.attrs.get('scale_factor', 1)) + np.float32(a.attrs.get('add_offset', 0)))
        m = np.asarray(g['advanced_valid_mask'][s['index'], self.y0 * 3 // 2:self.y1 * 3 // 2, self.x0 * 3 // 2:self.x1 * 3 // 2]) > 0
        return (*arr, m)

    def features(self, date):
        date = str(pd.Timestamp(date).tz_convert(None) if pd.Timestamp(date).tzinfo else pd.Timestamp(date))
        dt = ((self.times - np.datetime64(date)) / np.timedelta64(1, 'D')).astype('f4')
        n = len(self.y)
        state = initial(n)
        near, nearest = (np.full(n, np.inf), np.full(n, -1))
        for si in np.flatnonzero(abs(dt) <= 28):
            a, b, mask = self.load(int(si))
            ch, ok, native, nm = observations(a, b, mask, self.y, self.x)
            ix = np.flatnonzero(ok)
            update(state, ix, np.full(len(ix), dt[si], 'f4'), ch[ix], native[ix], nm[ix], int(si), 2 * np.pi * pd.Timestamp(self.times[si]).dayofyear / 365)
            better = ok & (abs(dt[si]) <= 21) & (abs(dt[si]) < near)
            near[better], nearest[better] = (abs(dt[si]), si)
        centre = np.zeros((n, 137), 'f4')
        for si in np.unique(nearest[nearest >= 0]):
            ix = np.flatnonzero(nearest == si)
            a, b, mask = self.load(int(si))
            centre[ix] = centre_batch(a, b, mask, self.lc, self.y[ix], self.x[ix])
        arrays = state_arrays(state, centre, self.soil)
        b, c, bn, cn = build(arrays['base_w7'], arrays['centre_w7'], arrays['base_w21'], [date] * n, [self.site] * n, {'context': True}, self.schema)
        x = np.column_stack([b, c]).astype('f4')
        names = pd.read_csv(ROOT / 'bundle/feature_schema.csv')
        require(names[names.array.eq('NN')].sort_values('index').feature.tolist() == bn + cn, 'Feature names/order')
        require(x.shape == (n, 804) and np.isfinite(x).all(), 'Feature finite/shape')
        boundary = bool(np.datetime64(date) - np.timedelta64(28, 'D') < self.times.min() or np.datetime64(date) + np.timedelta64(28, 'D') > self.times.max())
        return (x, arrays['base_w7'][:, 176] > 0, boundary)

    def provenance(self):
        return dict(crop=[self.y0, self.y1, self.x0, self.x1], landcover_attrs_sha256=self.lc_attrs, stores={p: {k: v for k, v in r.items() if k != 'group'} for p, r in self.stores.items()}, code_sha256=verify_code(), soil_sha256=sha(ROOT / f'feature_reference/{self.site}/soil.npy'))

def validate():
    verify_code()
    checks = {}
    for site in [45, 15]:
        p = ROOT / f'feature_reference/{site}'
        r = read(p / 'DONE.json')
        for n, h in r['files'].items():
            require(sha(p / n) == h, 'Feature reference SHA')
        rr = pd.read_parquet(p / 'rows.parquet')
        require(set(rr.target_event_id) <= set(allowed().event_id), 'Feature event allowlist')
        f = Features(site, rr.y.to_numpy(), rr.x.to_numpy())
        actual, support, _ = f.features(r['trait_time'])
        expected = np.load(p / 'x.npy')
        err = np.abs(actual.astype('f8') - expected)
        passed = bool(np.allclose(actual, expected, atol=1e-06, rtol=1e-06))
        require(np.array_equal(support, expected[:, 176] > 0), 'W7 identity')
        checks[str(site)] = dict(passed=passed, exact=bool(np.array_equal(actual, expected)), max_abs_error=float(err.max()), differing_elements=int((err != 0).sum()), atol=1e-06, rtol=1e-06, n_rows=len(rr), event=r['event'], trait_time=r['trait_time'], sample_ids_sha256=r['sample_ids_sha256'], per_column_max_abs=err.max(0).tolist(), provenance=f.provenance())
        save(ROOT / f'evidence/FEATURE_IDENTITY_site{site}.json', checks[str(site)])
        print('FEATURE_GATE', site, 'pass', passed, 'max_abs', err.max(), flush=True)
        if not passed:
            save(ROOT / 'FEATURE_VALIDATION_DONE.json', dict(passed=False, checks=checks, stop_reason='Frozen panel feature mismatch'))
            raise ValueError('STOP: panel feature identity failed')
        f.load.cache_clear()
    save(ROOT / 'FEATURE_VALIDATION_DONE.json', dict(passed=True, checks=checks, sealed_value_reads=0))

def stage():
    require(read(ROOT / 'FEATURE_VALIDATION_DONE.json')['passed'], 'Feature gate')
    a = allowed()
    ts = pd.read_csv(ROOT / 'bundle/timesteps.csv')
    screen = pd.read_csv(ROOT / 'bundle/screening_size50_dates.csv')
    identity_dates = pd.DatetimeIndex(pd.to_datetime(read(ROOT / 'bundle/screening_identity.json')['dates'], utc=True))
    for site_string, window in CFG['windows'].items():
        site = int(site_string)
        folder = ROOT / window['name']
        feature_folder = folder / 'features'
        feature_folder.mkdir(exist_ok=True)
        dd = pd.DatetimeIndex(pd.to_datetime(screen.loc[screen.site_id.eq(site), 'date'].drop_duplicates(), utc=True)) + pd.Timedelta(hours=12)
        require(len(dd) == 113 and np.array_equal(dd.asi8, identity_dates.asi8), 'Screening dates exact identity')
        legal_events = set(a.loc[a.site_id.eq(site), 'event_id'])
        event_times = ts[ts.site_id.eq(site) & ts.event_id.isin(legal_events)].copy()
        require(set(event_times.event_id) == legal_events, 'All development event dates represented')
        event_times.to_csv(folder / 'development_acquisitions.csv', index=False)
        refs = pd.DatetimeIndex(pd.to_datetime(event_times.trait_time.unique(), utc=True))
        dates = dd.union(refs).sort_values()
        py = np.arange(*window['rows'])
        px = np.arange(*window['cols'])
        yy, xx = np.meshgrid(py, px, indexing='ij')
        f = Features(site, yy.ravel(), xx.ravel())
        forest = f.lc[yy - f.y0, xx - f.x0] == 10
        np.save(folder / 'forest_mask.npy', forest)
        np.save(folder / 'parent_y.npy', py)
        np.save(folder / 'parent_x.npy', px)
        date_rows = []
        for i, date in enumerate(dates):
            path = feature_folder / f'date{i:03}.npy'
            rp = feature_folder / f'date{i:03}.json'
            stamp = date.isoformat()
            if rp.exists():
                rec = read(rp)
                require(rec['date'] == stamp and sha(path) == rec['feature_sha256'], 'Feature checkpoint identity')
                require(sha(feature_folder / rec['support_file']) == rec['support_sha256'], 'Support checkpoint identity')
            else:
                start = time.monotonic()
                x, support, boundary = f.features(stamp)
                np.save(path, x)
                sp = feature_folder / f'support{i:03}.npy'
                np.save(sp, support.reshape(50, 50))
                rec = dict(index=i, date=stamp, feature_sha256=sha(path), support_file=sp.name, support_sha256=sha(sp), supported_pixels=int(support.sum()), forest_supported_pixels=int((support.reshape(50, 50) & forest).sum()), context_boundary_truncated=boundary, feature_seconds=time.monotonic() - start, is_fortnightly=bool(date in dd), is_development_acquisition=bool(date in refs))
                save(rp, rec)
            date_rows.append(rec)
            if i % 10 == 0:
                print('FEATURES', site, i, len(dates), stamp, rec['supported_pixels'], flush=True)
        pd.DataFrame(date_rows).to_csv(folder / 'dates.csv', index=False)
        grid = dict(crs=f.grid[0].to_string(), transform=list(f.grid[1]), rows=window['rows'], cols=window['cols'])
        save(folder / 'FEATURES_DONE.json', dict(site_id=site, name=window['name'], grid=grid, n_dates=len(dates), n_fortnightly_dates=113, n_development_events=len(legal_events), n_development_acquisition_times=len(refs), forest_pixels=int(forest.sum()), date_records=date_rows, provenance=f.provenance(), sealed_value_reads=0))
        f.load.cache_clear()
    save(ROOT / 'FEATURE_STAGING_DONE.json', dict(passed=True, sites=[45, 15]))
