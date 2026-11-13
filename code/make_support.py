'Needs frozen model-code constants; writes staged figure outputs only. Shared temporal/spatial/production run namespace.'
from pathlib import Path
import sys,os,json,hashlib
import numpy as np,pandas as pd
import paths
sys.path.insert(0,str(paths.MODEL_CODE))
from contracts import TRAITS
V6=paths.V6;PROJECT=paths.PROJECT;OUT=paths.WORK;DATA=paths.DERIVED
DATA.mkdir(parents=True,exist_ok=True)
ORDER = ['LAI', 'LMA', 'Chl', 'Car', 'EWT', 'N', 'C', 'NSC', 'Lignin', 'Fibre', 'Cellulose']
RAW = ['LMA', 'N', 'LAI', 'C', 'Chl', 'EWT', 'Car', 'Lignin', 'Cellulose', 'Fibre', 'NSC']
TC = {t: '#9B5A9A' for t in ORDER}
RUN = 'cap2048_v3_nested11_20261108'
SPATIAL = 'cap2048_spatial_gsingle_20261112'
PRODUCTION = 'cap2048_production_nn_temporal_20261112'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def save(p, d):
    p = Path(p)
    assert p.resolve().is_relative_to(OUT)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(d, indent=2, default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)))
    tmp.replace(p)

def season(month):
    return np.asarray(['DJF', 'DJF', 'MAM', 'MAM', 'MAM', 'JJA', 'JJA', 'JJA', 'SON', 'SON', 'SON', 'DJF'])[np.asarray(month) - 1]

ROOT=paths.WORK
CACHE=paths.WORK/'inventory_cache'
CONFIG={'vegetation_classes':[10,20,30,40,90,95],'valid_observation_threshold':.5,'s2_support_window_days':7}
def csv(name,frame):
 target=paths.WORK/name;target.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(target,index=False)
