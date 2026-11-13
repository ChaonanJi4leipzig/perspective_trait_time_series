'JZ formal frozen folds, development assignment, raw frozen labels; stdout export; temporal run.\nInputs: code/paths.py; outputs: explicit derivation work directory, never upstream.\nOriginal: upstream://V6/final_figures_20261113/scripts/export_formal_jz.py.\nRun IDs: cap2048_v3_nested11_20261108; cap2048_spatial_gsingle_20261112; cap2048_production_nn_temporal_20261112.\nHeavy derivation was not rerun when packaging. Use --help before --run.'
from __future__ import annotations
import paths,sys
if __name__=="__main__":
 from make_runtime import cli
 cli(__doc__)
sys.path.insert(0,str(paths.MODEL_CODE))
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[k] = '1'
from pathlib import Path
import numpy as np, pandas as pd, io, json, sys, zipfile, hashlib, time
ROOT = Path(str(paths.JZ_ROOT / 'training2048_v3_nested11_20261108/harness/final_pipeline_20261105/training_2048'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def npy(x):
    b = io.BytesIO()
    np.save(b, x, allow_pickle=False)
    return b.getvalue()
assignment = ROOT / 'bundle/fold_assignment_v3.parquet'
assert sha(assignment) == '0bff3ca4397b1cce3c249e4c6c3bbdac4cd5c901268a45843a2e524c317d863e'
a = pd.read_parquet(assignment)
allow = a[a.new_role.eq('development') & a.new_fold.isin([1, 2, 3, 4]) & a.qa_gate.eq('existing_or_future_frozen_sampling_QA')]
allowed = set(allow.event_id)
manifest = read(ROOT / 'bundle/manifest.json')
parts = []
sources = {}
known = {}
extra = {}
started = time.monotonic()
for year in range(2022, 2027):
    folder = ROOT / 'folds' / f'temporal_{year}'
    fr = read(folder / 'DONE.json')
    assert sha(folder / 'panel.parquet') == fr['files']['panel.parquet']
    r = pd.read_parquet(folder / 'panel.parquet')
    assert set(r.target_event_id) <= allowed
    mask = r.site_id.isin(fr['known_sites']) & r.w7_support_count.gt(0)
    known[str(year)] = fr['known_sites']
    take = np.flatnonzero(mask)
    d = r.iloc[take].copy().reset_index(drop=True)
    y = np.full((len(d), 11), np.nan, 'f4')
    for event, ix in d.groupby('target_event_id', sort=False).indices.items():
        assert event in allowed and event in manifest['events']
        pkg = manifest['events'][event]
        p = Path(pkg['location'].removeprefix('jz:'))
        assert sha(p / 'rows.parquet') == pkg['files_sha256']['rows.parquet']
        meta = pd.read_parquet(p / 'rows.parquet', columns=['sample_id', 'target_event_id', 'fixed_test_anchor'])
        assert set(meta.target_event_id) == {event}
        pos = pd.Index(meta.sample_id).get_indexer(d.iloc[ix].sample_id)
        assert (pos >= 0).all() and meta.iloc[pos].fixed_test_anchor.all()
        assert sha(p / 'labels.npy') == pkg['files_sha256']['labels.npy']
        y[ix] = np.load(p / 'labels.npy', mmap_mode='r')[pos]
        sources[str(p / 'labels.npy')] = pkg['files_sha256']['labels.npy']
    for j in range(11):
        d[f'ref__{j}'] = y[:, j]
    for seed in range(42, 47):
        u = ROOT / 'units' / f'temporal_{year}_s{seed}_combine'
        ur = read(u / 'DONE.json')
        assert sha(u / 'predictions.npz') == ur['files']['predictions.npz']
        with np.load(u / 'predictions.npz') as z:
            assert np.array_equal(z['sample_id'], r.sample_id.to_numpy(str))
            for j in range(11):
                d[f'NN_{seed}__{j}'] = z['NN'][take, j]
        sources[str(u / 'predictions.npz')] = ur['files']['predictions.npz']
    loc = np.flatnonzero(r.site_id.eq(32))
    assert set(r.iloc[loc].target_event_id) <= allowed
    assert sha(folder / 'panel_x.npy') == fr['files']['panel_x.npy']
    extra[f'site32_panel_x_{year}.npy'] = npy(np.load(folder / 'panel_x.npy', mmap_mode='r')[loc])
    b = io.BytesIO()
    r.iloc[loc].to_parquet(b, index=False)
    extra[f'site32_panel_rows_{year}.parquet'] = b.getvalue()
    d['fold'] = str(year)
    parts.append(d)
    sources[str(folder / 'panel.parquet')] = fr['files']['panel.parquet']
    sources[str(folder / 'panel_x.npy')] = fr['files']['panel_x.npy']
    print('FOLD', year, 'primary', len(d), file=sys.stderr, flush=True)
full = pd.concat(parts, ignore_index=True)
assert len(full) == 48656 and full.sample_id.is_unique
buf = io.BytesIO()
full.to_parquet(buf, index=False, compression='zstd')
extra['formal_temporal_primary.parquet'] = buf.getvalue()
extra['FORMAL_EXPORT_RECEIPT.json'] = json.dumps(dict(run_id='cap2048_v3_nested11_20261108', rows=len(full), events=full.target_event_id.nunique(), sites=full.site_id.nunique(), known_sites_by_year=known, source_hashes=sources, assignment_sha256=sha(assignment), all_events_development=True, sealed_value_reads=0, JZ_writes=0, elapsed_seconds=time.monotonic() - started, export_sha256={k: hashlib.sha256(v).hexdigest() for k, v in extra.items()}), indent=2).encode()
with zipfile.ZipFile(sys.stdout.buffer, 'w', compression=zipfile.ZIP_STORED) as z:
    for name, value in extra.items():
        z.writestr(name, value)
