'frozen production config/bundle/model receipts;read helpers;production run.\nInputs: code/paths.py; outputs: explicit derivation work directory, never upstream.\nOriginal: upstream://V6/production_nn/common.py.\nRun IDs: cap2048_v3_nested11_20261108; cap2048_spatial_gsingle_20261112; cap2048_production_nn_temporal_20261112.\nHeavy derivation was not rerun when packaging. Use --help before --run.'
from __future__ import annotations
import paths,sys
if __name__=="__main__":
 from make_runtime import cli
 cli(__doc__)
sys.path.insert(0,str(paths.MODEL_CODE))
import os
import sys
import json
import hashlib
from pathlib import Path
ROOT = paths.PRODUCTION
sys.path.insert(0, str(paths.PRODUCTION / 'vendor'))
import numpy as np
import pandas as pd
from contracts import require, sha, digest, TRAITS, nn_config
from make_support import save

def read(path):
    return json.loads(Path(path).read_text())
CFG = read(ROOT / 'config.json')
FORMAL = paths.FORMAL
ASSIGNMENT_SHA = '0bff3ca4397b1cce3c249e4c6c3bbdac4cd5c901268a45843a2e524c317d863e'

def verify_code():
    pins = read(ROOT / 'evidence/FROZEN_CODE.json')
    for name, record in pins.items():
        require(sha(ROOT / name) == record['sha256'], 'Frozen code changed: ' + name)
    return digest(pins)

def allowed():
    p = ROOT / 'bundle/fold_assignment_v3.parquet'
    require(sha(p) == ASSIGNMENT_SHA, 'Assignment SHA')
    a = pd.read_parquet(p)
    good = a.new_role.eq('development') & a.new_fold.isin([1, 2, 3, 4])
    good &= a.qa_gate.eq('existing_or_future_frozen_sampling_QA')
    good &= a.date.between('2022-01-01', '2026-09-18')
    return a.loc[good].copy()

def verify_source():
    source = read(ROOT / 'bundle/source.json')
    manifest = read(ROOT / 'bundle/manifest.json')
    require(sha(ROOT / 'bundle/manifest.json') == source['manifest_sha256'], 'Manifest SHA')
    require(len(source['expected']) == 437, 'Source cohort')
    require(set(source['expected']) == set(manifest['events']) == set(manifest['hs_events']), 'Incomplete S2/HS cohort')
    require(set(source['expected']) <= set(allowed().event_id), 'Sealed/nondevelopment source')
    for name, expected in source['metadata_sha256'].items():
        p = ROOT / 'bundle' / name
        if p.exists():
            require(sha(p) == expected, 'Metadata SHA ' + name)
    return (source, manifest)

def bound_arrays(folder, names, receipt):
    result = {}
    for name in names:
        p = Path(folder) / (name + '.npy')
        require(sha(p) == receipt['files'][name + '.npy'], 'Array SHA ' + str(p))
        result[name] = np.load(p, mmap_mode='r', allow_pickle=False)
    return result

def commit(folder, extra):
    folder = Path(folder)
    files = {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'DONE.json' and (not p.name.endswith('.tmp'))}
    save(folder / 'DONE.json', dict(**extra, files=files))

def load_training():
    p = ROOT / 'training'
    r = read(p / 'DONE.json')
    require(r['n_train'] == 827091 and r['sealed_value_reads'] == 0, 'Training gate')
    for n, h in r['files'].items():
        require(sha(p / n) == h, 'Production training changed: ' + n)
    rows = pd.read_parquet(p / 'rows.parquet')
    x, y, hs = [np.load(p / (n + '.npy'), allow_pickle=False) for n in ['x', 'y', 'hs']]
    require(x.shape == (827091, 804) and y.shape == (827091, 11) and (hs.shape == (827091, 224)), 'Training shapes')
    return (rows, x, y, hs, r)
