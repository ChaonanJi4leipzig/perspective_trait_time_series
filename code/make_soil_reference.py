'JZ frozen development x[:,346:436],read-only;extracts constant90-value soil vectors;production run.\nInputs: code/paths.py; outputs: explicit derivation work directory, never upstream.\nOriginal: upstream://V6/production_nn/new_sites_20261113/extract_soil_jz.py.\nRun IDs: cap2048_v3_nested11_20261108; cap2048_spatial_gsingle_20261112; cap2048_production_nn_temporal_20261112.\nHeavy derivation was not rerun when packaging. Use --help before --run.'
from __future__ import annotations
import paths,sys
if __name__=="__main__":
 from make_runtime import cli
 cli(__doc__)
sys.path.insert(0,str(paths.MODEL_CODE))
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[k] = '1'
import argparse, base64, hashlib, io, json, time
from pathlib import Path
import numpy as np
import pandas as pd

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()
p = argparse.ArgumentParser()
p.add_argument('--receipt-sha', required=True)
p.add_argument('--controls', required=True)
args = p.parse_args()
root = Path(str(paths.JZ_ROOT / 'production_nn_cap2048_temporal_20261112'))
started = time.monotonic()
receipt = root / 'training/DONE.json'
assert sha(receipt) == args.receipt_sha
r = json.loads(receipt.read_text())
assert r['n_train'] == 827091 and r['sealed_value_reads'] == 0
assignment = root / 'bundle/fold_assignment_v3.parquet'
assert sha(assignment) == r['assignment_sha256'] == '0bff3ca4397b1cce3c249e4c6c3bbdac4cd5c901268a45843a2e524c317d863e'
a = pd.read_parquet(assignment)
allowed = a[a.new_role.eq('development') & a.new_fold.isin([1, 2, 3, 4]) & a.qa_gate.eq('existing_or_future_frozen_sampling_QA')]
rows = root / 'training/rows.parquet'
assert sha(rows) == r['files']['rows.parquet']
meta = pd.read_parquet(rows, columns=['site_id', 'target_event_id'])
assert len(meta) == 827091
assert set(meta.target_event_id) <= set(allowed.event_id)
source = root / 'training/x.npy'
actual_sha = sha(source)
assert actual_sha == r['files']['x.npy']
x = np.load(source, mmap_mode='r', allow_pickle=False)
assert x.shape == (827091, 804) and x.dtype == np.dtype('float32')
controls = json.loads(args.controls)
checks = {}
payloads = {}
for sid in [32, 40, 23, 45, 15]:
    ids = np.flatnonzero(meta.site_id.eq(sid).to_numpy())
    assert len(ids)
    assert set(meta.iloc[ids].target_event_id) <= set(allowed[allowed.site_id.eq(sid)].event_id)
    vector = np.array(x[ids[0], 346:436], copy=True)
    assert vector.shape == (90,) and np.isfinite(vector).all()
    for begin in range(0, len(ids), 4096):
        values = x[ids[begin:begin + 4096], 346:436]
        assert np.array_equal(values, np.broadcast_to(vector, values.shape)), ('Site constancy failed', sid)
    b = io.BytesIO()
    np.save(b, vector, allow_pickle=False)
    raw = b.getvalue()
    h = hashlib.sha256(raw).hexdigest()
    check = dict(development_rows=len(ids), development_events=int(meta.iloc[ids].target_event_id.nunique()), site_constancy_exact=True, shape=[90], dtype=str(vector.dtype), npy_sha256=h, npy_bytes=len(raw))
    if str(sid) in controls:
        assert h == controls[str(sid)], ('Positive control failed', sid)
        check['local_positive_control_byte_identical'] = True
    else:
        payloads[str(sid)] = base64.b64encode(raw).decode()
    checks[str(sid)] = check
print(json.dumps(dict(source_path=str(source), source_sha256=actual_sha, source_bytes=source.stat().st_size, source_shape=list(x.shape), source_receipt=str(receipt), source_receipt_sha256=args.receipt_sha, row_metadata_path=str(rows), row_metadata_sha256=r['files']['rows.parquet'], assignment_sha256=sha(assignment), checks=checks, new_vector_npy_base64=payloads, elapsed_seconds=time.monotonic() - started, sealed_rows_read=0, JZ_writes=0, job_submitted=False)))
