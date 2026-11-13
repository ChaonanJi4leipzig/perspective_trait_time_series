"""formal_temporal_primary.parquet and development teacher stores; writes per-pixel baseline staging; temporal run.
Inputs: code/paths.py; outputs: explicit derivation work directory, never upstream.
Original: upstream://V6/final_figures_20261113/scripts/pixel_baselines.py.
Run IDs: cap2048_v3_nested11_20261108; cap2048_spatial_gsingle_20261112; cap2048_production_nn_temporal_20261112.
Heavy derivation was not rerun when packaging. Use --help before --run."""
from __future__ import annotations
import paths, sys
if __name__ == '__main__':
    from make_runtime import cli
    cli(__doc__)
sys.path.insert(0, str(paths.MODEL_CODE))
from make_support import *
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_site_support as fs
import numcodecs, zarr
numcodecs.blosc.set_nthreads(1)
zarr.config.set({'async.concurrency': 1, 'threading.max_workers': 1})

def site_job(item):
    sid, rows = item
    dest = DATA / 'pixel_cache' / f'site_{sid}.parquet'
    receipt = dest.with_suffix('.json')
    if dest.exists() and receipt.exists():
        return dest
    access = fs.gate()
    events = access.assignment[access.assignment.site_id.eq(sid) & access.assignment.event_id.isin(access.allowed_events)].sort_values(['date', 'event_id'])
    keys = np.unique(rows.y.to_numpy(int) * 1000 + rows.x.to_numpy(int))
    yy = keys // 1000
    xx = keys % 1000
    n = len(keys)
    sums = np.zeros((5, 4, 11, n), float)
    counts = np.zeros((5, 4, 11, n), 'i4')
    g = fs.teacher(sid)
    assert g['traits_map'].chunks[0] == g['valid_mask'].chunks[0] == 1
    names = np.asarray(g['trait'][:]).astype(str).tolist()
    jj = np.asarray([names.index(t) for t in TRAITS])
    audit = []
    for e in events.itertuples():
        access.authorize_events([e.event_id])
        products = fs.products_for(g, e.event_id, access)
        values = np.full((11, n), np.nan, 'f4')
        for p in products:
            access.authorize_events([e.event_id])
            ti = p['time_index']
            qa = np.asarray(g['valid_mask'].vindex[np.full(n, ti), yy, xx]) > 0
            v = np.asarray(g['traits_map'].vindex[np.full(11 * n, ti), np.repeat(jj, n), np.tile(yy, 11), np.tile(xx, 11)]).reshape(11, n)
            take = qa[None] & np.isfinite(v) & ~np.isfinite(values)
            values[take] = v[take]
        good = np.isfinite(values)
        year = int(e.date[:4]) - 2022
        si = ['DJF', 'MAM', 'JJA', 'SON'].index(str(season([pd.Timestamp(e.date).month])[0]))
        sums[year, si] += np.where(good, values, 0)
        counts[year, si] += good
        audit.append(dict(event_id=e.event_id, date=e.date, valid_values=int(good.sum()), products=products))
    result = rows[['sample_id']].copy()
    pos = np.searchsorted(keys, rows.y.to_numpy(int) * 1000 + rows.x.to_numpy(int))
    years = rows.year.to_numpy(int) - 2022
    ss = season(pd.to_datetime(rows.target_event_id.str.split(':', n=1).str[1]).dt.month)
    ss = np.array([['DJF', 'MAM', 'JJA', 'SON'].index(s) for s in ss])
    all_s = sums.sum((0, 1))
    all_c = counts.sum((0, 1))
    year_s = sums.sum(1)
    year_c = counts.sum(1)
    season_s = sums.sum(0)
    season_c = counts.sum(0)
    for j in range(11):
        total = all_s[j, pos] - year_s[years, j, pos]
        den = all_c[j, pos] - year_c[years, j, pos]
        seasonal = season_s[ss, j, pos] - sums[years, ss, j, pos]
        sd = season_c[ss, j, pos] - counts[years, ss, j, pos]
        result[f'pixel_mean__{j}'] = np.divide(total, den, out=np.full(len(rows), np.nan), where=den > 0)
        result[f'pixel_mean_n__{j}'] = den
        result[f'pixel_season__{j}'] = np.divide(seasonal, sd, out=np.full(len(rows), np.nan), where=sd > 0)
        result[f'pixel_season_n__{j}'] = sd
    dest.parent.mkdir(exist_ok=True)
    result.to_parquet(dest, index=False)
    save(receipt, dict(site_id=int(sid), formal_rows=len(rows), unique_parent_pixels=n, events=audit, held_year_excluded=True, season='DJF=12/1/2,MAM=3/4/5,JJA=6/7/8,SON=9/10/11', product_rule='earliest UTC/product ID; first QA-valid finite value per trait, once per event', historical_QA='valid_mask & finite trait; no W7 requirement on historical teacher', sealed_value_reads=0))
    print('PIXEL_BASELINE', sid, len(rows), n, len(events), flush=True)
    return dest

def main():
    f = pd.read_parquet(paths.derived_input('formal_temporal_primary.parquet'))
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(site_job, list(f.groupby('site_id', sort=True))))
    b = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    assert b.sample_id.is_unique and set(b.sample_id) == set(f.sample_id)
    f.merge(b, on='sample_id', validate='one_to_one').to_parquet(DATA / 'formal_temporal_with_pixel_baselines.parquet', index=False)
    save(DATA / 'PIXEL_BASELINES_DONE.json', dict(rows=len(f), sites=f.site_id.nunique(), cache_hashes={str(p.relative_to(OUT)): sha(p) for p in paths}, sealed_value_reads=0))
if __name__ == '__main__':
    main()
