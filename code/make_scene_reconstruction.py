'selected development scene, frozen NN models and804feature code; writes SI3 maps;temporal run.\nInputs: code/paths.py; outputs: explicit derivation work directory, never upstream.\nOriginal: upstream://V6/final_figures_20261113/scripts/infer_si3_scene.py.\nRun IDs: cap2048_v3_nested11_20261108; cap2048_spatial_gsingle_20261112; cap2048_production_nn_temporal_20261112.\nHeavy derivation was not rerun when packaging. Use --help before --run.'
from __future__ import annotations
import paths,sys
if __name__=="__main__":
 from make_runtime import cli
 cli(__doc__)
sys.path.insert(0,str(paths.MODEL_CODE))
from make_support import V6 as PARENT_V6, DATA as FINAL_DATA
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_site_support import *
ROOT = FINAL_DATA / 'si3_work'
SITES = {'Castelporziano': dict(site_id=32, y0=475, x0=475, size=50)}
from functools import lru_cache
import argparse, time, torch
from feature_ops import build
from batched_features import observations, centre_batch
from sequence_backend import initial, update, state_arrays
from nn_model import predict

class SceneFeatures:

    def __init__(self, name):
        self.name = name
        self.site = SITES[name]['site_id']
        self.cat = read(ROOT / 'inputs' / f's2_catalog_{name}.json')
        self.tt = np.array([s['time'] for s in self.cat], dtype='datetime64[ns]')
        self.lc = land(self.site)
        self.schema = pd.read_csv(ROOT / 'inputs/schema_input.csv')
        self.soil = np.load(ROOT / 'evaluation_sites' / name / 'frozen/soil90.npy')

    @lru_cache(maxsize=32)
    def load(self, si):
        s = self.cat[si]
        p = Path(s['path'])
        assert sha(p / '.zattrs') == s['attrs_sha256']
        g = zarr.open_group(str(p), mode='r')
        aa = []
        for name in ['reflectance_10m', 'reflectance_20m']:
            a = g[name]
            raw = np.asarray(a[s['time_index']])
            v = raw.astype('f4')
            v[raw == a.fill_value] = np.nan
            v = v * np.float32(a.attrs.get('scale_factor', 1)) + np.float32(a.attrs.get('add_offset', 0))
            aa.append(v)
        return (*aa, np.asarray(g['advanced_valid_mask'][s['time_index']]) > 0)

    def features(self, date, y, x):
        n = len(y)
        date = str(date)
        dt = ((self.tt - np.datetime64(date)) / np.timedelta64(1, 'D')).astype('f4')
        state = initial(n)
        near = np.full(n, np.inf)
        nearest = np.full(n, -1)
        for si in np.flatnonzero(abs(dt) <= 28):
            a, b, m = self.load(int(si))
            ch, ok, native, nm = observations(a, b, m, y, x)
            ix = np.flatnonzero(ok)
            update(state, ix, np.full(len(ix), dt[si], 'f4'), ch[ix], native[ix], nm[ix], int(si), 2 * np.pi * pd.Timestamp(self.tt[si]).dayofyear / 365)
            better = ok & (abs(dt[si]) <= 21) & (abs(dt[si]) < near)
            near[better] = abs(dt[si])
            nearest[better] = si
        centre = np.zeros((n, 137), 'f4')
        for si in np.unique(nearest[nearest >= 0]):
            ix = np.flatnonzero(nearest == si)
            a, b, m = self.load(int(si))
            centre[ix] = centre_batch(a, b, m, self.lc, y[ix], x[ix])
        z = state_arrays(state, centre, self.soil)
        b, c, _, _ = build(z['base_w7'], z['centre_w7'], z['base_w21'], [date] * n, [self.site] * n, {'context': True}, self.schema)
        xx = np.column_stack([b, c]).astype('f4')
        assert xx.shape == (n, 804) and np.isfinite(xx).all()
        return (xx, z['base_w7'][:, 176] > 0)

def main(name):
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.time()
    out = ROOT / 'evaluation_sites' / name
    frozen = out / 'frozen'
    chosen = read(ROOT / 'evidence' / f'scene_selection_{name}.json')
    assert sha(ROOT / 'source_data' / f'fig3b_scene_selection_{name}.csv') == chosen['selection_csv_sha256']
    assert pd.read_csv(ROOT / 'source_data' / f'fig3b_scene_selection_{name}.csv').selected.sum() == 1
    access = gate()
    event = chosen['event_id']
    year = int(chosen['year'])
    access.authorize_events([event])
    r = access.metadata(frozen / 'panel_rows.parquet', IDENTITY_COLUMNS + ['w7_support_count', 'input_available'])
    access.register(r)
    access.grant(name, r, {'features', 'prediction', 'labels'})
    features = SceneFeatures(name)
    expected = np.load(frozen / 'panel_x.npy')
    rebuilt = np.empty_like(expected)
    for date, ix in r.groupby('trait_time', sort=False).indices.items():
        access.authorize(name, r.iloc[ix], 'features')
        date = str(pd.Timestamp(date).tz_convert(None).to_datetime64())
        xx, ok = features.features(date, r.iloc[ix].y.to_numpy(int), r.iloc[ix].x.to_numpy(int))
        rebuilt[ix] = xx
        assert np.array_equal(ok, r.iloc[ix].w7_support_count.gt(0).to_numpy())
    diff = abs(rebuilt - expected)
    feature_check = dict(rows=len(r), columns=804, max_abs=float(diff.max()), different_elements=int((diff != 0).sum()), atol=1e-05, rtol=1e-06, passed=bool(np.allclose(rebuilt, expected, atol=1e-05, rtol=1e-06)))
    save(out / 'FEATURE_VALIDATION.json', feature_check)
    assert feature_check['passed'], feature_check
    models = {}
    checks = []
    tol = 8 * np.finfo(np.float32).eps
    for seed in SEEDS:
        md = frozen / 'models' / f'temporal_{year}_s{seed}_nn_final'
        rec = read(md / 'DONE.json')
        assert sha(md / 'NN.pt') == rec['files']['NN.pt']
        u = rec['identity']['unit']
        assert u['kind'] == 'nn_final' and u['scope'] == 'temporal' and (int(u['fold']) == year) and (u['seed'] == seed)
        ck = torch.load(md / 'NN.pt', map_location='cpu', weights_only=True)
        models[seed] = ck
        access.authorize(name, r, 'prediction')
        p = predict(ck, rebuilt, 'cpu')
        gold = np.load(frozen / f'NN_panel_s{seed}.npy')
        err = abs(p - gold)
        checks.append(dict(seed=seed, traits=TRAITS, max_abs_per_trait=err.max(0).tolist(), passed=bool(np.allclose(p, gold, atol=tol, rtol=tol))))
    save(out / 'PREDICTION_VALIDATION.json', dict(checks=checks, atol=float(tol), rtol=float(tol), passed=all((c['passed'] for c in checks)), torch=torch.__version__))
    assert all((c['passed'] for c in checks)), checks
    tg = teacher(SITES[name]['site_id'])
    products = products_for(tg, event, access)
    access.authorize(name, r, 'labels')
    truth = np.load(frozen / 'labels.npy')
    teacher_rows = np.empty_like(truth)
    for prod in products:
        ii = np.flatnonzero(r.trait_time_index.eq(prod['time_index']).to_numpy())
        if not len(ii):
            continue
        v, qa = read_teacher(tg, prod, event, access)
        yy = r.iloc[ii].y.to_numpy(int)
        xx = r.iloc[ii].x.to_numpy(int)
        assert qa[yy, xx].all()
        teacher_rows[ii] = v[:, yy, xx].T
    assert np.array_equal(teacher_rows, truth)
    save(out / 'TEACHER_VALIDATION.json', dict(exact=True, rows=len(r), max_abs_per_trait=abs(teacher_rows - truth).max(0).tolist()))
    print(name, event, 'ALL MANDATORY VALIDATIONS PASSED', feature_check, flush=True)
    ref = np.full((4, 1000, 1000), np.nan, 'f4')
    pred = np.full_like(ref, np.nan)
    mask = np.zeros((1000, 1000), bool)
    source = np.full((1000, 1000), -1, 'i2')
    lc = features.lc
    for prod in products:
        v, qa = read_teacher(tg, prod, event, access)
        support = np.zeros_like(mask)
        dt = np.abs((features.tt - np.datetime64(prod['time'])) / np.timedelta64(1, 'D'))
        for si in np.flatnonzero(dt <= 7):
            support |= supported30(features.cat[int(si)], ROOT / 'inputs/site_support_cache' / name)
        take = qa & support & ~np.isin(lc, [0, 70, 80]) & ~mask
        y, x = np.where(take)
        for first in range(0, len(y), 20000):
            yy = y[first:first + 20000]
            xx = x[first:first + 20000]
            matrix, ok = features.features(prod['time'], yy, xx)
            assert ok.all(), 'Coverage selector and formal features disagree'
            pp = np.mean([predict(models[s], matrix, 'cpu')[:, SELECT] for s in SEEDS], axis=0, dtype='f8').astype('f4')
            ref[:, yy, xx] = v[SELECT][:, yy, xx]
            pred[:, yy, xx] = pp.T
            mask[yy, xx] = True
            source[yy, xx] = prod['time_index']
            print(name, 'inferred', int(mask.sum()), '/', int(chosen['common_valid_pixels']), flush=True)
    assert mask.sum() == int(chosen['common_valid_pixels'])
    np.savez_compressed(out / 'scene_maps.npz', reference=ref, prediction=pred, mask=mask, source_time_index=source, traits=np.array(SHORT), event=event, products=json.dumps(products), parent_window=np.array([SITES[name][k] for k in ['y0', 'x0', 'size']]))
    stats = []
    quantile_rows = []
    for j, t in enumerate(SHORT):
        y = ref[j, mask].astype(float)
        p = pred[j, mask].astype(float)
        q = np.linspace(0.05, 0.95, 19)
        qy = np.quantile(y, q)
        qp = np.quantile(p, q)
        a, b = np.polyfit(qy, qp, 1)
        stats.append(dict(site=name, event_id=event, date=chosen['date'], trait=t, paired_pixels=len(y), qq_slope=a, qq_intercept=b, pixel_r2=1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum(), pixel_rmse=np.sqrt(((p - y) ** 2).mean()), mean_offset=(p - y).mean(), reference_p02=np.quantile(y, 0.02), reference_p05=np.quantile(y, 0.05), reference_p95=np.quantile(y, 0.95), reference_p98=np.quantile(y, 0.98)))
        quantile_rows.extend((dict(site=name, trait=t, quantile=qq, reference=rr, prediction=pp) for qq, rr, pp in zip(q, qy, qp)))
    pd.DataFrame(stats).to_csv(ROOT / 'source_data' / f'Fig3b_statistics_{name}.csv', index=False)
    pd.DataFrame(quantile_rows).to_csv(ROOT / 'source_data' / f'Fig3b_quantiles_{name}.csv', index=False)
    save(out / 'EVALUATION_INFERENCE_DONE.json', dict(site=name, event=event, year=year, common_valid_pixels=int(mask.sum()), rows_validated=len(r), feature_validation=feature_check, prediction_validation_passed=True, teacher_validation_exact=True, selection_csv_sha256=chosen['selection_csv_sha256'], map_sha256=sha(out / 'scene_maps.npz'), model_sha256={str(s): sha(frozen / 'models' / f'temporal_{year}_s{s}_nn_final/NN.pt') for s in SEEDS}, seeds=SEEDS, device='cpu', threads=4, billed_GPU_hours=0, wall_seconds=time.time() - start, assignment_sha256=sha(access.path), audit=access.audit))
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('site', choices=SITES)
    main(ap.parse_args().site)
