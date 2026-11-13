'frozen assignment,30m grid,corrected stores;development-gated read helpers;all runs.\nInputs: code/paths.py; outputs: explicit derivation work directory, never upstream.\nOriginal: upstream://V6/figure_sites.py.\nRun IDs: cap2048_v3_nested11_20261108; cap2048_spatial_gsingle_20261112; cap2048_production_nn_temporal_20261112.\nHeavy derivation was not rerun when packaging. Use --help before --run.'
from __future__ import annotations
import paths,sys
if __name__=="__main__":
 from make_runtime import cli
 cli(__doc__)
sys.path.insert(0,str(paths.MODEL_CODE))
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import sys, json, hashlib
import numpy as np, pandas as pd, zarr, numcodecs
numcodecs.blosc.set_nthreads(1)
if hasattr(zarr, 'config'):
    zarr.config.set({'async.concurrency': 1, 'threading.max_workers': 1})
ROOT = paths.V6
PROJECT = paths.PROJECT
sys.path.insert(0, str(paths.MODEL_CODE))
from contracts import TRAITS, sha, IDENTITY_COLUMNS
from access import AccessGate
from grid_mapping import store_grid, same_grid
SITES = {'Hesse': dict(site_id=45, y0=480, x0=460, size=50), 'Schechenfilz': dict(site_id=15, y0=430, x0=540, size=50)}
SHORT = ['LMA', 'CHL', 'EWT', 'N']
SELECT = [0, 4, 5, 1]
SEEDS = list(range(42, 47))
BUNDLE = PROJECT / 'analysis/0_perspective/figures_add_samples/final_pipeline_20261105/training_2048/harness/final_pipeline_20261105/training_2048/bundle'
RUN_JZ = str(paths.JZ_ROOT / 'training2048_v3_nested11_20261108/harness/final_pipeline_20261105/training_2048')

def read(p):
    return json.loads(Path(p).read_text())

def save(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(obj, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))
    q.replace(p)

def gate():
    return AccessGate(ROOT / 'inputs/fold_assignment_v3.parquet', mode='execute')

def times(g):
    a = g['time']
    unit, origin = a.attrs['units'].split(' since ')
    u = {'days': 'D', 'seconds': 's', 'milliseconds': 'ms', 'microseconds': 'us', 'nanoseconds': 'ns'}[unit]
    return (np.datetime64(origin) + np.asarray(a[:]).astype('timedelta64[' + u + ']')).astype('datetime64[ns]')

def teacher(site):
    return zarr.open_group(str(PROJECT / f'input_data/enmap/traits_region_v2/site_{site}_enmap_traits_region_30km_30m.zarr'), mode='r')

def land(site):
    return np.asarray(zarr.open_array(str(PROJECT / f'experiments/enmap_v2_worldcover_source_topup_20261022/worldcover_zarr/site_{site}_worldcover2021_30m.zarr'), mode='r')[:]).reshape(1000, 1000)

def products_for(g, event, access):
    access.authorize_events([event])
    e = access.events.loc[event]
    pp = np.asarray(g['source_product_id'][:]).astype(str)
    tt = times(g)
    out = []
    for product in str(e.product_ids).split(';'):
        ix = np.flatnonzero(pp == product)
        assert len(ix) == 1, (event, product)
        ti = int(ix[0])
        assert str(tt[ti].astype('datetime64[D]')) == str(e.date)
        out.append(dict(time_index=ti, product=product, time=str(tt[ti])))
    return sorted(out, key=lambda v: (v['time'], v['product']))

def read_teacher(g, product, event, access, window=None):
    access.authorize_events([event])
    assert product['product'] in str(access.events.loc[event].product_ids).split(';')
    assert g['traits_map'].chunks[0] == g['valid_mask'].chunks[0] == 1
    names = np.asarray(g['trait'][:]).astype(str).tolist()
    jj = [names.index(t) for t in TRAITS]
    ti = product['time_index']
    ys, xs = window or (slice(None), slice(None))
    v = np.asarray(g['traits_map'].get_orthogonal_selection((ti, jj, ys, xs)), dtype='f4')
    m = np.asarray(g['valid_mask'][ti, ys, xs]) > 0
    qa = m & np.isfinite(v).all(0)
    return (v, qa)

def s2_catalog(site):
    result = []
    tp = PROJECT / f'input_data/enmap/traits_region_v2/site_{site}_enmap_traits_region_30km_30m.zarr'
    grid = store_grid(tp, True)
    lp = PROJECT / f'experiments/enmap_v2_worldcover_source_topup_20261022/worldcover_zarr/site_{site}_worldcover2021_30m.zarr'
    assert same_grid(grid, store_grid(lp))
    for year in range(2021, 2028):
        p = PROJECT / f'input_data/s2/region/site_{site}/site_{site}_s2_region_{year}.zarr'
        if not p.exists():
            continue
        assert same_grid(grid, store_grid(p))
        g = zarr.open_group(str(p), mode='r')
        tt = times(g)
        h = sha(p / '.zattrs')
        for ti, t in enumerate(tt):
            result.append(dict(path=str(p), time_index=ti, time=str(t), attrs_sha256=h))
    return sorted(result, key=lambda r: (r['time'], r['path'], r['time_index']))

def supported30(scene, cache):
    p = Path(scene['path'])
    key = hashlib.sha256(json.dumps(scene, sort_keys=True).encode()).hexdigest()
    dest = cache / (key + '.npz')
    cache.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        with np.load(dest) as z:
            return np.unpackbits(z['bits'], count=1000000).reshape(1000, 1000).astype(bool)
    assert sha(p / '.zattrs') == scene['attrs_sha256']
    g = zarr.open_group(str(p), mode='r')
    ti = scene['time_index']
    a = g['reflectance_10m']
    b = g['reflectance_20m']
    m = g['advanced_valid_mask']
    assert a.chunks[0] == b.chunks[0] == m.chunks[0] == 1
    aa = np.asarray(a[ti])
    bb = np.asarray(b[ti])
    mm = np.asarray(m[ti]) > 0
    af = np.isfinite(aa).all(-1) & (aa != a.fill_value).all(-1)
    bf = np.isfinite(bb).all(-1) & (bb != b.fill_value).all(-1)
    any10 = af.reshape(1000, 3, 1000, 3).any((1, 3))
    c = np.clip(np.rint((np.arange(1000) + 0.5) * 1.5 - 0.5).astype(int), 0, 1499)
    ok = any10 & bf[c[:, None], c[None, :]] & mm[c[:, None], c[None, :]]
    assert sha(p / '.zattrs') == scene['attrs_sha256']
    np.savez_compressed(dest, bits=np.packbits(ok))
    return ok
