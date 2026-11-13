'JZ frozen development panel features and models;read-only stdout extraction;temporal run.\nInputs: code/paths.py; outputs: explicit derivation work directory, never upstream.\nOriginal: upstream://V6/final_figures_20261113/scripts/collect_si3_jz.py.\nRun IDs: cap2048_v3_nested11_20261108; cap2048_spatial_gsingle_20261112; cap2048_production_nn_temporal_20261112.\nHeavy derivation was not rerun when packaging. Use --help before --run.'
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
import sys, io, json, zipfile, argparse, hashlib
import numpy as np, pandas as pd
p = argparse.ArgumentParser()
p.add_argument('--event', required=True)
args = p.parse_args()
event = args.event
year = int(event.split(':')[1][:4])
assert event.startswith('32:')
root = Path(str(paths.JZ_ROOT / 'training2048_v3_nested11_20261108/harness/final_pipeline_20261105/training_2048'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def npy(v):
    f = io.BytesIO()
    np.save(f, v, allow_pickle=False)
    return f.getvalue()
a = pd.read_parquet(root / 'bundle/fold_assignment_v3.parquet')
allowed = set(a[a.new_role.eq('development') & a.new_fold.isin([1, 2, 3, 4])].event_id)
assert event in allowed
fold = root / 'folds' / f'temporal_{year}'
fr = read(fold / 'DONE.json')
assert 32 in fr['known_sites']
assert sha(fold / 'panel.parquet') == fr['files']['panel.parquet']
panel = pd.read_parquet(fold / 'panel.parquet')
ix = np.flatnonzero(panel.target_event_id.eq(event))
rows = panel.iloc[ix].copy()
assert len(rows) and set(rows.target_event_id) <= {event}
assert sha(fold / 'panel_x.npy') == fr['files']['panel_x.npy']
files = {'panel_x.npy': npy(np.load(fold / 'panel_x.npy', mmap_mode='r')[ix]), 'fold_DONE.json': json.dumps(fr).encode()}
b = io.BytesIO()
rows.to_parquet(b, index=False)
files['panel_rows.parquet'] = b.getvalue()
pkg = read(root / 'bundle/manifest.json')['events'][event]
p = Path(pkg['location'].removeprefix('jz:'))
assert sha(p / 'rows.parquet') == pkg['files_sha256']['rows.parquet']
er = pd.read_parquet(p / 'rows.parquet', columns=['sample_id', 'target_event_id'])
assert set(er.target_event_id) == {event}
pos = pd.Index(er.sample_id).get_indexer(rows.sample_id)
assert (pos >= 0).all()
assert sha(p / 'labels.npy') == pkg['files_sha256']['labels.npy']
files['labels.npy'] = npy(np.load(p / 'labels.npy', mmap_mode='r')[pos])
model_hashes = {}
for seed in range(42, 47):
    u = root / 'units' / f'temporal_{year}_s{seed}_combine'
    rec = read(u / 'DONE.json')
    assert sha(u / 'predictions.npz') == rec['files']['predictions.npz']
    with np.load(u / 'predictions.npz') as z:
        assert np.array_equal(z['sample_id'], panel.sample_id.to_numpy(str))
        files[f'NN_panel_s{seed}.npy'] = npy(z['NN'][ix])
    md = root / 'units' / f'temporal_{year}_s{seed}_nn_final'
    mr = read(md / 'DONE.json')
    assert sha(md / 'NN.pt') == mr['files']['NN.pt']
    model_hashes[str(seed)] = mr['files']['NN.pt']
    for n in ['NN.pt', 'DONE.json', 'selection.json']:
        files[f'models/{md.name}/{n}'] = (md / n).read_bytes()
files['COLLECTION_DONE.json'] = json.dumps(dict(event=event, rows=len(rows), models=model_hashes, source_package=pkg, sealed_value_reads=0, JZ_writes=0, files_sha256={k: hashlib.sha256(v).hexdigest() for k, v in files.items()}), indent=2).encode()
with zipfile.ZipFile(sys.stdout.buffer, 'w', compression=zipfile.ZIP_STORED) as z:
    for n, v in files.items():
        z.writestr(n, v)
