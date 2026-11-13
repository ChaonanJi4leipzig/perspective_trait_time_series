'fixed windows and frozen production models;stages features and infers without training;production run.\nInputs: code/paths.py; outputs: explicit derivation work directory, never upstream.\nOriginal: upstream://V6/production_nn/new_sites_20261113/run_window.py.\nRun IDs: cap2048_v3_nested11_20261108; cap2048_spatial_gsingle_20261112; cap2048_production_nn_temporal_20261112.\nHeavy derivation was not rerun when packaging. Use --help before --run.'
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
import sys, time, argparse
CONTROL = paths.PRODUCTION / 'new_sites_20261113'
PROD = paths.PRODUCTION
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_production_common import *
from make_production_sites import SITES

def stage(name):
    import make_production_features as pf
    verify_code()
    require(read(PROD / 'FEATURE_VALIDATION_DONE.json')['passed'], 'Frozen feature gate')
    soil = read(CONTROL / 'soil_extraction_receipt.json')
    spec = SITES[name]
    sid = spec['site_id']
    folder = paths.WORK / 'production' / name
    folder.mkdir(parents=True, exist_ok=True)
    ff = folder / 'features'
    ff.mkdir(exist_ok=True)
    require(sha(PROD / f'feature_reference/{sid}/soil.npy') == soil['checks'][str(sid)]['npy_sha256'], 'Soil SHA')
    approved = allowed()
    events = set(approved.loc[approved.site_id.eq(sid), 'event_id'])
    timesteps = pd.read_csv(PROD / 'bundle/timesteps.csv')
    acquisitions = timesteps[timesteps.site_id.eq(sid) & timesteps.event_id.isin(events)].copy()
    require(set(acquisitions.event_id) == events, 'Development acquisition completeness')
    acquisitions.to_csv(folder / 'development_acquisitions.csv', index=False)
    regular = pd.DatetimeIndex(pd.to_datetime(read(PROD / 'bundle/screening_identity.json')['dates'], utc=True))
    require(len(regular) == 113, '113 frozen dates')
    refs = pd.DatetimeIndex(pd.to_datetime(acquisitions.trait_time.unique(), utc=True, format='mixed'))
    dates = regular.union(refs).sort_values()
    py = np.arange(*spec['rows'])
    px = np.arange(*spec['cols'])
    yy, xx = np.meshgrid(py, px, indexing='ij')
    f = pf.Features(sid, yy.ravel(), xx.ravel())
    forest = f.lc[yy - f.y0, xx - f.x0] == 10
    candidate = pd.read_csv(PROD.parent / 'site_alternatives/candidate_table.csv')
    candidate = candidate[candidate.site_id.eq(sid)].iloc[0]
    require(abs(float(forest.mean()) - candidate.forest_fraction) < 1e-12, 'Frozen window forest fraction')
    np.save(folder / 'forest_mask.npy', forest)
    np.save(folder / 'parent_y.npy', py)
    np.save(folder / 'parent_x.npy', px)
    records = []
    start = time.monotonic()
    cpu = time.process_time()
    for i, date in enumerate(dates):
        path = ff / f'date{i:03}.npy'
        rp = ff / f'date{i:03}.json'
        stamp = date.isoformat()
        if rp.exists():
            rec = read(rp)
            require(rec['date'] == stamp and sha(path) == rec['feature_sha256'], 'Feature checkpoint')
            require(sha(ff / rec['support_file']) == rec['support_sha256'], 'Support checkpoint')
        else:
            begin = time.monotonic()
            x, support, boundary = f.features(stamp)
            np.save(path, x)
            sp = ff / f'support{i:03}.npy'
            np.save(sp, support.reshape(50, 50))
            rec = dict(index=i, date=stamp, feature_sha256=sha(path), support_file=sp.name, support_sha256=sha(sp), supported_pixels=int(support.sum()), forest_supported_pixels=int((support.reshape(50, 50) & forest).sum()), context_boundary_truncated=boundary, feature_seconds=time.monotonic() - begin, is_fortnightly=bool(date in regular), is_development_acquisition=bool(date in refs))
            save(rp, rec)
        records.append(rec)
        if i % 10 == 0:
            print('FEATURES', name, i, len(dates), stamp, rec['supported_pixels'], flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(folder / 'dates.csv', index=False)
    screen = pd.read_csv(PROD.parent / 'site_alternatives/w7_dates.csv')
    screen = screen[screen.site_id.eq(sid) & screen.window.eq(candidate.window)].sort_values('date')
    regular_rows = frame[frame.is_fortnightly]
    require(len(screen) == len(regular_rows) == 113, 'Independent screening rows')
    require(np.array_equal(regular_rows.date.str[:10], screen.date), 'Screening date identity')
    require(np.allclose(regular_rows.supported_pixels / 2500, screen.window_fraction, atol=1e-12, rtol=0), 'Exact window W7 screening identity')
    require(np.allclose(regular_rows.forest_supported_pixels / int(forest.sum()), screen.forest_fraction, atol=1e-12, rtol=0), 'Exact forest W7 screening identity')
    grid = dict(crs=f.grid[0].to_string(), transform=list(f.grid[1]), rows=spec['rows'], cols=spec['cols'])
    save(folder / 'FEATURES_DONE.json', dict(site_id=sid, name=name, grid=grid, n_dates=len(dates), n_fortnightly_dates=113, n_development_events=len(events), n_development_acquisition_times=len(refs), forest_pixels=int(forest.sum()), date_records=records, provenance=f.provenance(), sealed_value_reads=0, independent_screening_exact=True, elapsed_wall_seconds=time.monotonic() - start, CPU_seconds=time.process_time() - cpu, original_feature_kernel=str(Path(pf.__file__)), original_feature_kernel_sha256=sha(Path(pf.__file__))))
    print('FEATURES_DONE', name, flush=True)

def inference(name):
    import torch
    from nn_model import predict
    torch.set_num_threads(2)
    verify_code()
    require(read(PROD / 'EQUIVALENCE_GATE.json')['passed'], 'Frozen equivalence gate')
    require(read(CONTROL / 'NN_POSITIVE_CONTROL.json')['passed'], 'Local NN positive control')
    spec = SITES[name]
    folder = paths.WORK / 'production' / name
    fr = read(folder / 'FEATURES_DONE.json')
    dates = pd.read_csv(folder / 'dates.csv')
    training = read(PROD / 'TRAINING_DONE.json')
    models = []
    for seed in CFG['seeds']:
        p = PROD / 'units' / f'seed{seed}_final/NN.pt'
        require(sha(p) == training['model_sha256'][str(seed)], 'Frozen production NN SHA')
        ck = torch.load(p, map_location='cpu', weights_only=False)
        require(ck['config'] == nn_config('temporal') and ck['seed'] == seed, 'Frozen recipe')
        models.append(ck)
    per_seed = np.full((5, len(dates), 11, 50, 50), np.nan, 'f4')
    support = np.zeros((len(dates), 50, 50), bool)
    start = time.monotonic()
    cpu = time.process_time()
    for i, row in enumerate(dates.itertuples()):
        path = folder / 'features' / f'date{i:03}.npy'
        sp = folder / 'features' / row.support_file
        require(sha(path) == row.feature_sha256 and sha(sp) == row.support_sha256, 'Inference input SHA')
        x = np.load(path)
        valid = np.load(sp).ravel()
        support[i] = valid.reshape(50, 50)
        if valid.any():
            for j, ck in enumerate(models):
                values = np.full((2500, 11), np.nan, 'f4')
                values[valid] = predict(ck, x[valid], 'cpu')
                per_seed[j, i] = values.T.reshape(11, 50, 50)
        if i % 20 == 0:
            print('INFER', name, i, len(dates), 'supported', int(valid.sum()), flush=True)
    mean = np.mean(per_seed, axis=0, dtype=np.float32)
    require(np.array_equal(np.isfinite(mean).all(1), support), 'Missing W7 is NaN')
    require(np.array_equal(np.isfinite(per_seed).all(2), np.broadcast_to(support, (5, *support.shape))), 'Seed support identity')
    forest = np.load(folder / 'forest_mask.npy')
    py = np.load(folder / 'parent_y.npy')
    px = np.load(folder / 'parent_x.npy')
    require(np.array_equal(py, np.arange(*spec['rows'])) and np.array_equal(px, np.arange(*spec['cols'])), 'Window identity')
    cube = folder / 'trait_cube.npz'
    np.savez(cube, mean=mean, per_seed=per_seed, w7_support=support, forest_mask=forest, dates=dates.date.to_numpy(str), parent_y=py, parent_x=px, traits=np.asarray(TRAITS), seeds=np.asarray(CFG['seeds']), is_fortnightly=dates.is_fortnightly.to_numpy(bool), is_development_acquisition=dates.is_development_acquisition.to_numpy(bool), context_boundary_truncated=dates.context_boundary_truncated.to_numpy(bool))
    coverage = dates[['date', 'is_fortnightly', 'is_development_acquisition', 'context_boundary_truncated']].copy()
    coverage['window_supported_fraction'] = support.mean((1, 2))
    coverage['forest_supported_fraction'] = support[:, forest].mean(1)
    coverage['window_coverage80'] = coverage.window_supported_fraction >= 0.8
    coverage['forest_coverage80'] = coverage.forest_supported_fraction >= 0.8
    coverage.to_csv(folder / 'coverage.csv', index=False)
    files = {n: sha(folder / n) for n in ['trait_cube.npz', 'dates.csv', 'coverage.csv', 'forest_mask.npy', 'parent_y.npy', 'parent_x.npy', 'development_acquisitions.csv', 'FEATURES_DONE.json']}
    rec = dict(run_id=CFG['run_id'], state='complete', production_artifact=True, accuracy_claim=None, model_sha256=training['model_sha256'], selected_epochs=training['selected_epochs'], files=files, grid=fr['grid'], summary=dict(site_id=spec['site_id'], name=name, n_dates=len(dates), n_fortnightly_dates=113, forest_pixels=int(forest.sum()), forest_fraction=float(forest.mean()), fortnightly_forest_dates_coverage80=int(coverage.loc[coverage.is_fortnightly, 'forest_coverage80'].sum())), elapsed_wall_seconds=time.monotonic() - start, CPU_seconds=time.process_time() - cpu, sealed_value_reads=0, GPU_hours=0, new_fits=0, soil_extraction_receipt_sha256=sha(CONTROL / 'soil_extraction_receipt.json'), original_inference_kernel_sha256=sha(PROD / 'vendor/nn_model.py'))
    save(folder / 'manifest.json', rec)
    save(folder / 'INFERENCE_DONE.json', rec)
    print('INFERENCE_DONE', name, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('site', choices=SITES)
    p.add_argument('stage', choices=['features', 'inference'])
    args = p.parse_args()
    (stage if args.stage == 'features' else inference)(args.site)
