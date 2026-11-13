'Needs frozen Hesse production cube and development teacher windows; writes trajectories and cubes.\nInputs:paths.py;outputs:WORK,never upstream;run IDs:cap2048_v3_nested11_20261108,cap2048_spatial_gsingle_20261112,cap2048_production_nn_temporal_20261112'
from __future__ import annotations
import paths,sys
if __name__=="__main__":
 from make_runtime import cli
 cli(__doc__)
sys.path.insert(0,str(paths.MODEL_CODE))
from make_site_support import *
PROD = paths.PRODUCTION

def main():
    marker_path = PROD / 'PRODUCTION_INFERENCE_DONE.json'
    assert marker_path.is_file(), 'Production inference is pending: do not open cubes'
    marker = read(marker_path)
    assert marker['state'] == 'complete' and all(marker['gates'].values()) and (marker['sealed_value_reads'] == 0)
    assert sha(PROD / 'PRODUCT_MANIFEST.json') == marker['product_manifest_sha256']
    product = read(PROD / 'PRODUCT_MANIFEST.json')
    records = {}
    for name, spec in SITES.items():
        source_name = {'Hesse': 'Hesse', 'Schechenfilz': 'Schechenfilz_Nord'}[name]
        folder = PROD / source_name
        p = folder / 'trait_cube.npz'
        cube_hash = sha(p)
        assert str(p.resolve()) == marker['cubes'][source_name]
        assert cube_hash == product['files'][source_name + '/trait_cube.npz']
        manifest = read(folder / 'manifest.json')
        assert sha(folder / 'manifest.json') == product['files'][source_name + '/manifest.json']
        assert manifest['state'] == 'complete' and manifest['production_artifact']
        with np.load(p) as z:
            mean = z['mean']
            per_seed = z['per_seed']
            support = z['w7_support'].astype(bool)
            forest = z['forest_mask'].astype(bool)
            dates = z['dates'].astype(str)
            traits = z['traits'].astype(str).tolist()
            py = z['parent_y']
            px = z['parent_x']
            fortnight = z['is_fortnightly'].astype(bool)
            boundary = z['context_boundary_truncated'].astype(bool)
            assert traits == TRAITS and np.array_equal(z['seeds'], SEEDS)
            assert mean.shape == (len(dates), 11, 50, 50) and per_seed.shape == (5, len(dates), 11, 50, 50)
            assert np.array_equal(mean, np.mean(per_seed, axis=0, dtype='f4'), equal_nan=True)
            assert np.array_equal(np.isfinite(mean).all(1), support)
            assert np.array_equal(np.isfinite(per_seed).all(2), np.broadcast_to(support, (5, *support.shape)))
        assert np.array_equal(py, np.arange(spec['y0'], spec['y0'] + 50)) and np.array_equal(px, np.arange(spec['x0'], spec['x0'] + 50))
        assert np.array_equal(forest, land(spec['site_id'])[np.ix_(py, px)] == 10)
        tt = pd.DatetimeIndex(pd.to_datetime(dates, utc=True, format='mixed'))
        assert tt.is_unique and tt.is_monotonic_increasing
        assert fortnight.sum() == 113
        assert list(pd.Series(tt[fortnight].year).value_counts().sort_index()) == [16, 26, 26, 26, 19]
        assert str(tt[fortnight].min().date()) == '2022-06-04' and str(tt[fortnight].max().date()) == '2026-09-19'
        counts = support[:, forest].sum(1)
        coverage = counts / int(forest.sum())
        good = coverage >= 0.8
        trajectory = []
        for i in np.flatnonzero(fortnight):
            for j, t in zip(SELECT, SHORT):
                trajectory.append(dict(site=name, site_id=spec['site_id'], trait=t, date=dates[i], year=int(tt[i].year), kind='Production NN', mean=float(mean[i, j][forest & support[i]].mean()) if good[i] else np.nan, forest_pixels=int(forest.sum()), supported_forest_pixels=int(counts[i]), coverage=float(coverage[i]), context_boundary_truncated=bool(boundary[i]), run_id=marker['run_id'], source_file=str(p), source_sha256=cube_hash))
        refs = pd.read_csv(ROOT / 'source_data' / f'Fig4_reference_candidates_{name}.csv')
        refs['kind'] = 'EnMAP reference'
        refs['date_utc_earliest'] = refs.products.map(lambda s: min((v['time'] for v in json.loads(s))))
        refs['run_id'] = 'frozen EnMAP teacher'
        refs['coverage'] = refs.teacher_qa_coverage
        refs['supported_forest_pixels'] = refs.qa_forest_pixels
        refs.to_csv(ROOT / 'source_data' / f'Fig4_reference_all_{name}.csv', index=False)
        refs[(refs.coverage > 0) & (refs.coverage < 0.8)].to_csv(ROOT / 'source_data' / f'Fig4_partial_references_{name}.csv', index=False)
        refs[~refs['plot']].to_csv(ROOT / 'source_data' / f'Fig4_omitted_development_references_{name}.csv', index=False)
        chosen_refs = refs[refs['plot']].copy()
        chosen_refs.to_csv(ROOT / 'source_data' / f'Fig4_reference_used_{name}.csv', index=False)
        pd.concat([pd.DataFrame(trajectory), chosen_refs], ignore_index=True).to_csv(ROOT / 'source_data' / f'Fig4_trajectories_{name}.csv', index=False)
        cv = pd.DataFrame(dict(date=dates, year=tt.year, is_fortnightly=fortnight, forest_pixels=int(forest.sum()), supported_forest_pixels=counts, coverage=coverage, plot=good, context_boundary_truncated=boundary))
        cv.to_csv(ROOT / 'source_data' / f'Fig4_coverage_{name}.csv', index=False)
        yr = cv[cv.is_fortnightly].groupby('year', sort=True).agg(n_dates=('date', 'size'), n_dates_ge80=('plot', 'sum'), mean_forest_coverage=('coverage', 'mean'), min_forest_coverage=('coverage', 'min'), n_context_boundary_flags=('context_boundary_truncated', 'sum')).reset_index()
        yr['fraction_dates_ge80'] = yr.n_dates_ge80 / yr.n_dates
        yr.to_csv(ROOT / 'source_data' / f'Fig4_coverage_by_year_{name}.csv', index=False)
        ix = np.flatnonzero(fortnight & (tt.year == 2024))
        assert len(ix) == 26
        candidates = ix[good[ix]]
        assert len(candidates)
        target = pd.Timestamp('2024-07-22T12:00:00Z')
        front = min(candidates, key=lambda i: (abs(tt[i] - target), tt[i]))
        front_index = int(np.flatnonzero(ix == front).item())
        dest = ROOT / 'figure_data' / f'Fig4_cube_{name}.npz'
        dest.parent.mkdir(exist_ok=True)
        cube = np.transpose(mean[ix][:, SELECT], (1, 0, 2, 3))
        np.savez_compressed(dest, data=cube, trait_names=np.array(SHORT), display_dates=dates[ix], parent_y=py, parent_x=px, t0=front_index, site=name, forest_mask=forest, w7_support=support[ix])
        records[name] = dict(source_name=source_name, production_run=marker['run_id'], source_cube=str(p), source_cube_sha256=cube_hash, manifest_sha256=sha(folder / 'manifest.json'), window=spec, forest_pixels=int(forest.sum()), n_product_dates=113, n_2024_cube_dates=26, front_face_date=str(tt[front].date()), front_face_UTC=str(tt[front]), front_face_coverage=float(coverage[front]), front_index=front_index, reference_dates_used=sorted(chosen_refs.date.unique().tolist()), reference_dates_omitted=sorted(refs.loc[~refs['plot'], 'date'].unique().tolist()), n_dates_below80=int((~good[fortnight]).sum()), source_model_sha256=manifest['model_sha256'], production_selected_epochs=marker['selected_epochs'], production_GPU_billed_hours=marker['gpu_billed_hours'])
    assert records['Hesse']['source_model_sha256'] == records['Schechenfilz']['source_model_sha256'], 'Both sites must use the same all-data ensemble'
    assert set(records['Hesse']['source_model_sha256']) == set(map(str, SEEDS))
    save(ROOT / 'evidence/FIG4_DATA_DONE.json', dict(production_marker_sha256=sha(marker_path), sites=records, product_receipt_sha256=sha(PROD / 'PRODUCTION_RECEIPT.json'), production_reference_role='in-sample descriptive teacher overlay; generalization evidence only from independent CV', figure_worker_GPU_hours=0))
if __name__ == '__main__':
    main()
