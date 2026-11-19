"""local S2 timestep metadata and anonymous PC STAC;writes additional dates,never imagery assets;inventory.
Inputs: code/paths.py; outputs: explicit derivation work directory, never upstream.
Original: upstream://V6/final_figures_20261113/si1_pc/build_pc_fill.py.
Run IDs: cap2048_v3_nested11_20261108; cap2048_spatial_gsingle_20261112; cap2048_production_nn_temporal_20261112.
Heavy derivation was not rerun when packaging. Use --help before --run."""
from __future__ import annotations
import paths, sys
if __name__ == '__main__':
    from make_runtime import cli
    cli(__doc__)
sys.path.insert(0, str(paths.MODEL_CODE))
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[k] = '1'
from pathlib import Path
import json, hashlib, time, re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import zarr, numcodecs
from pyproj import Transformer
from shapely.geometry import Point, shape
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
from urllib3.util.retry import Retry
OUT = paths.WORK
PROJECT = paths.PROJECT
DATA = OUT / 'source_data'
WORK = OUT / 'si1_pc'
START = pd.Timestamp('2022-04-01', tz='UTC')
STOP = pd.Timestamp('2026-10-01', tz='UTC')
API = 'https://planetarycomputer.microsoft.com/api/stac/v1'
WINDOW = '2022-04-01T00:00:00Z/2026-09-30T23:59:59.999999Z'
numcodecs.blosc.set_nthreads(1)
zarr.config.set({'async.concurrency': 1, 'threading.max_workers': 1})

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write_json(p, v):
    p.write_text(json.dumps(v, indent=2, default=str) + '\n')

def log(v):
    with (WORK / 'query_log.jsonl').open('a') as f:
        f.write(json.dumps({'utc': now(), **v}, default=str) + '\n')

class PoliteIO(StacApiIO):

    def request(self, href, method=None, headers=None, parameters=None):
        assert href.startswith(API), href
        assert not any((x in href.lower() for x in ['token=', 'sig=', '/assets/']))
        t = time.monotonic()
        try:
            return super().request(href, method, headers, parameters)
        finally:
            log({'event': 'request', 'endpoint': href.split('?')[0], 'method': method or 'GET', 'seconds': round(time.monotonic() - t, 3)})
            time.sleep(0.3)

def inventory(site_ids=None):
    c = pd.read_csv(paths.derived_input('SI1_counts_site_year.csv'))
    s = pd.read_csv(paths.derived_input('SI1_S2_observations.csv'))
    if site_ids is not None:
        c=c[c.site_id.isin(site_ids)].copy()
        s=s[s.site_id.isin(site_ids)].copy()
    s['date_utc'] = pd.to_datetime(s.date_utc, utc=True, format='mixed')
    names = c.groupby('site_id').site_name.first().to_dict()
    coverage = []
    local = []
    centres = []
    gaps = []
    for sid in sorted(names):
        store_paths = sorted((PROJECT / f'input_data/s2/region/site_{sid}').glob(f'site_{sid}_s2_region_*.zarr'))
        assert store_paths, sid
        centre_candidates = []
        for p in store_paths:
            year = int(re.search('_(\\d{4})\\.zarr$', p.name).group(1))
            attrs = json.loads((p / '.zattrs').read_text())
            b = attrs.get('bounds_utm') or attrs['grid_reference_bounds_utm']
            crs = attrs.get('crs') or attrs['grid_reference_crs']
            lon, lat = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            centre_candidates.append((p, lon, lat, attrs.get('site_lon'), attrs.get('site_lat')))
            g = zarr.open_group(str(p), mode='r')
            ta = g['time']
            units = ta.attrs['units']
            scale, origin = units.split(' since ')
            unit = {'milliseconds': 'ms', 'microseconds': 'us', 'nanoseconds': 'ns', 'seconds': 's', 'days': 'D'}[scale]
            tt = pd.to_datetime(ta[:], unit=unit, origin=pd.Timestamp(origin).tz_localize(None), utc=True)
            assert tt.is_monotonic_increasing and len(tt) > 0
            aa = max(START, pd.Timestamp(f'{year}-01-01', tz='UTC'), tt.min().normalize())
            bb = min(STOP - pd.Timedelta(days=1), pd.Timestamp(f'{year}-12-31', tz='UTC'), tt.max().normalize())
            coverage.append(dict(site_id=sid, site_name=names[sid], store_year=year, store=str(p), first_timestep=tt.min(), last_timestep=tt.max(), n_timesteps=len(tt), coverage_start=aa, coverage_end=bb, zattrs_sha256=sha(p / '.zattrs'), time_header_sha256=sha(p / 'time/.zarray')))
            for t in tt[(tt >= START) & (tt < STOP)]:
                local.append(dict(site_id=sid, year=year, datetime_utc=t, date_utc=t.strftime('%Y-%m-%d'), store=str(p)))
        assert max((x[1] for x in centre_candidates)) - min((x[1] for x in centre_candidates)) < 1e-06, sid
        assert max((x[2] for x in centre_candidates)) - min((x[2] for x in centre_candidates)) < 1e-06, sid
        p, lon, lat, slon, slat = centre_candidates[0]
        centres.append(dict(site_id=sid, site_name=names[sid], longitude=lon, latitude=lat, centre_source=str(p / '.zattrs'), centre_rule='WGS84 transform of bounds_utm midpoint', attribute_site_lon=slon, attribute_site_lat=slat))
        for year in range(2022, 2027):
            a = max(START, pd.Timestamp(f'{year}-01-01', tz='UTC'))
            b = min(STOP - pd.Timedelta(days=1), pd.Timestamp(f'{year}-12-31', tz='UTC'))
            cov = sorted([(x['coverage_start'], x['coverage_end']) for x in coverage if x['site_id'] == sid and x['store_year'] == year and (x['coverage_start'] <= x['coverage_end'])])
            expected = bool(c.loc[(c.site_id == sid) & (c.year == year), 'S2_store_present'].iloc[0])
            assert expected == bool(cov), (sid, year)
            cursor = a
            for x, y in cov:
                if cursor < x:
                    gaps.append(dict(site_id=sid, site_name=names[sid], year=year, start_date=cursor.strftime('%Y-%m-%d'), end_date=(x - pd.Timedelta(days=1)).strftime('%Y-%m-%d'), reason='before_first_store_timestep', store_present=True))
                cursor = max(cursor, y + pd.Timedelta(days=1))
            if cursor <= b:
                gaps.append(dict(site_id=sid, site_name=names[sid], year=year, start_date=cursor.strftime('%Y-%m-%d'), end_date=b.strftime('%Y-%m-%d'), reason='after_last_store_timestep' if cov else 'no_local_store_for_year', store_present=bool(cov)))
    ld = pd.DataFrame(local)
    assert set(zip(ld.site_id, ld.datetime_utc)) == set(zip(s.site_id, s.date_utc)), 'Local inventory does not reproduce source observations'
    gap = pd.DataFrame(gaps)
    gap['n_days'] = (pd.to_datetime(gap.end_date) - pd.to_datetime(gap.start_date)).dt.days + 1
    gap.to_csv(DATA / 'SI1_S2_gap_periods.csv', index=False)
    pd.DataFrame(coverage).to_csv(WORK / 'local_store_coverage.csv', index=False)
    pd.DataFrame(centres).to_csv(WORK / 'site_centres.csv', index=False)
    ld.to_csv(WORK / 'local_timesteps.csv.gz', index=False)
    return (c, s, ld, gap, pd.DataFrame(centres))

def query(centres):
    retry = Retry(total=7, connect=5, read=5, status=7, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=['GET', 'POST'], respect_retry_after_header=True)
    io = PoliteIO(headers={'User-Agent': 'stac-metadata-client/1.0'}, timeout=(20, 90), max_retries=retry)
    io.session.trust_env = False
    catalog = Client.open(API, stac_io=io)
    allrows = []
    for r in centres.itertuples(index=False):
        dest = WORK / f'items_site_{r.site_id}.csv.gz'
        if dest.exists():
            rows = pd.read_csv(dest).to_dict('records')
            log({'event': 'cache_reuse', 'site_id': r.site_id, 'n_items': len(rows)})
        else:
            point = {'type': 'Point', 'coordinates': [r.longitude, r.latitude]}
            params = dict(collections=['sentinel-2-l2a'], intersects=point, datetime=WINDOW, limit=500, fields={'include': ['id', 'type', 'geometry', 'collection', 'properties'], 'exclude': ['assets', 'links']})
            log({'event': 'query_start', 'site_id': r.site_id, 'site_name': r.site_name, 'parameters': params})
            for attempt in range(4):
                rows = []
                try:
                    search = catalog.search(**params)
                    for page_number, page in enumerate(search.pages_as_dicts(), 1):
                        for item in page['features']:
                            p = item['properties']
                            assert shape(item['geometry']).covers(Point(r.longitude, r.latitude)), item['id']
                            rows.append(dict(site_id=r.site_id, site_name=r.site_name, id=item['id'], datetime_utc=p['datetime'], date_utc=p['datetime'][:10], mgrs_tile=p.get('s2:mgrs_tile'), relative_orbit=p.get('sat:relative_orbit'), orbit_state=p.get('sat:orbit_state'), cloud_cover=p.get('eo:cloud_cover'), processing_baseline=p.get('s2:processing_baseline'), product_uri=p.get('s2:product_uri'), generation_time=p.get('s2:generation_time'), geometry_json=json.dumps(item['geometry'], separators=(',', ':')), properties_json=json.dumps(p, separators=(',', ':'))))
                        log({'event': 'page', 'site_id': r.site_id, 'page': page_number, 'page_items': len(page['features']), 'total_items': len(rows)})
                    break
                except Exception as exc:
                    log({'event': 'query_retry', 'site_id': r.site_id, 'attempt': attempt + 1, 'error': str(exc)})
                    if attempt == 3:
                        raise
                    time.sleep(min(60, 5 * 2 ** attempt))
            assert rows, r.site_id
            frame = pd.DataFrame(rows).drop_duplicates(['site_id', 'id']).sort_values(['datetime_utc', 'id'])
            frame.to_csv(dest, index=False)
            rows = frame.to_dict('records')
            log({'event': 'query_complete', 'site_id': r.site_id, 'n_items': len(rows)})
        allrows.extend(rows)
        print('STAC', r.site_id, r.site_name, len(rows), flush=True)
    result = pd.DataFrame(allrows).sort_values(['site_id', 'datetime_utc', 'id'])
    result.to_csv(DATA / 'SI1_pc_stac_items.csv.gz', index=False)
    return result

def derive(c, s, local, gaps, items):
    items['cloud_cover'] = pd.to_numeric(items.cloud_cover, errors='coerce')
    dates = items.groupby(['site_id', 'date_utc']).agg(n_items=('id', 'size'), min_cloud_cover=('cloud_cover', 'min'), n_cloud_known=('cloud_cover', 'count')).reset_index()
    dates['year'] = dates.date_utc.str[:4].astype(int)
    dates['in_gap'] = False
    dates['gap_reason'] = ''
    for r in gaps.itertuples(index=False):
        mask = dates.site_id.eq(r.site_id) & dates.date_utc.ge(r.start_date) & dates.date_utc.le(r.end_date)
        dates.loc[mask, 'in_gap'] = True
        dates.loc[mask, 'gap_reason'] = r.reason
    dates['pc_fill'] = dates.in_gap & dates.min_cloud_cover.le(50)
    validkeys = set(zip(s.loc[s.valid_observation, 'site_id'], s.loc[s.valid_observation, 'date_utc'].dt.strftime('%Y-%m-%d')))
    fillkeys = set(zip(dates.loc[dates.pc_fill, 'site_id'], dates.loc[dates.pc_fill, 'date_utc']))
    assert not validkeys & fillkeys
    dates.to_csv(DATA / 'SI1_pc_dates.csv', index=False)
    dates[dates.pc_fill].to_csv(DATA / 'SI1_pc_fill_dates.csv', index=False)
    rows = []
    checks = []
    missing = []
    for r in c.itertuples(index=False):
        ds = dates[(dates.site_id == r.site_id) & (dates.year == r.year)]
        ss = s[(s.site_id == r.site_id) & (s.year == r.year)]
        rows.append(dict(site_id=r.site_id, site=r.site_name, site_name=r.site_name, year=r.year, n_local_valid=int(ss.valid_observation.sum()), n_pc_fill=int(ds.pc_fill.sum()), n_pc_all_in_gap=int(ds.in_gap.sum()), store_present=bool(r.S2_store_present), n_combined=int(ss.valid_observation.sum() + ds.pc_fill.sum())))
        ll = local[(local.site_id == r.site_id) & (local.year == r.year)]
        if r.S2_store_present:
            pc = set(ds.date_utc)
            hit = ll.date_utc.isin(pc)
            checks.append(dict(site_id=r.site_id, site_name=r.site_name, year=r.year, n_local_timesteps=len(ll), n_overlap=int(hit.sum()), overlap_fraction=float(hit.mean())))
            for x in ll.loc[~hit].itertuples(index=False):
                missing.append(dict(site_id=r.site_id, site_name=r.site_name, year=r.year, date_utc=x.date_utc, datetime_utc=x.datetime_utc, store=x.store))
    counts = pd.DataFrame(rows)
    checks = pd.DataFrame(checks)
    counts.to_csv(DATA / 'SI1_counts_site_year_pc_fill.csv', index=False)
    checks.to_csv(WORK / 'sanity_overlap_site_year.csv', index=False)
    pd.DataFrame(missing, columns=['site_id', 'site_name', 'year', 'date_utc', 'datetime_utc', 'store']).to_csv(WORK / 'sanity_missing_dates.csv', index=False)
    totals = counts.groupby(['site_id', 'site_name'])[['n_local_valid', 'n_pc_fill', 'n_pc_all_in_gap', 'n_combined']].sum().sort_values('n_pc_fill', ascending=False)
    totals.to_csv(WORK / 'counts_per_site.csv')
    summary = dict(query_utc=now(), api=API, collection='sentinel-2-l2a', datetime=WINDOW, sites=int(len(counts.site_id.unique())), n_site_years=len(counts), gap_periods=len(gaps), gap_days=int(gaps.n_days.sum()), n_raw_items=len(items), n_local_valid=int(counts.n_local_valid.sum()), n_pc_fill=int(counts.n_pc_fill.sum()), n_pc_all_in_gap=int(counts.n_pc_all_in_gap.sum()), n_combined=int(counts.n_combined.sum()), median_combined_dates_per_site_year=float(counts.n_combined.median()), n_local_timesteps=int(checks.n_local_timesteps.sum()), n_overlap=int(checks.n_overlap.sum()), overlap_fraction=float(checks.n_overlap.sum() / checks.n_local_timesteps.sum()), min_site_year_overlap=float(checks.overlap_fraction.min()), source_observations_sha256=sha(DATA / 'SI1_S2_observations.csv'))
    write_json(DATA / 'SI1_pc_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(totals.to_string(), flush=True)
if __name__ == '__main__':
    c, s, local, gaps, centres = inventory()
    print('INVENTORY', len(centres), len(local), len(gaps), flush=True)
    items = query(centres)
    derive(c, s, local, gaps, items)
