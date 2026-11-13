'frozen production cubes and development teacher windows;writes trajectories,refs,cubes;production run.\nInputs: code/paths.py; outputs: explicit derivation work directory, never upstream.\nOriginal: upstream://V6/production_nn/new_sites_20261113/assemble_fig4_inputs.py.\nRun IDs: cap2048_v3_nested11_20261108; cap2048_spatial_gsingle_20261112; cap2048_production_nn_temporal_20261112.\nHeavy derivation was not rerun when packaging. Use --help before --run.'
from __future__ import annotations
import paths,sys
if __name__=="__main__":
 from make_runtime import cli
 cli(__doc__)
sys.path.insert(0,str(paths.MODEL_CODE))
from pathlib import Path
import sys, argparse, json
CONTROL = paths.PRODUCTION / 'new_sites_20261113'
PROD = paths.PRODUCTION
FIG = paths.V6
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_production_common import *
from make_production_sites import SITES
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_site_support as fs

def write_csv(frame, stem, name):
    columns = pd.read_csv(FIG / 'source_data' / f'{stem}_Hesse.csv', nrows=0).columns
    frame.reindex(columns=columns).to_csv(paths.DERIVED / f'{stem}_{name}.csv', index=False)

def main(name):
    spec = SITES[name]
    sid = spec['site_id']
    folder = paths.WORK / 'production' / name
    if not (folder/'manifest.json').exists():folder=PROD/name
    output_folder=paths.WORK/'production'/name;output_folder.mkdir(parents=True,exist_ok=True)
    m = read(folder / 'manifest.json')
    require(m['state'] == 'complete' and m['run_id'] == CFG['run_id'], 'New site completion')
    for n, h in m['files'].items():
        require(sha(folder / n) == h, 'Product hash ' + n)
    cube_file = folder / 'trait_cube.npz'
    cube_hash = sha(cube_file)
    with np.load(cube_file) as z:
        mean = z['mean']
        seeds = z['per_seed']
        support = z['w7_support']
        forest = z['forest_mask']
        dates = z['dates'].astype(str)
        py = z['parent_y']
        px = z['parent_x']
        fortnight = z['is_fortnightly']
        boundary = z['context_boundary_truncated']
        require(z['traits'].tolist() == TRAITS and z['seeds'].tolist() == CFG['seeds'], 'Trait/seed identity')
        require(np.array_equal(mean, np.mean(seeds, axis=0, dtype='f4'), equal_nan=True), 'Five-seed mean identity')
        require(np.array_equal(np.isfinite(mean).all(1), support), 'No gap fill')
    tt = pd.DatetimeIndex(pd.to_datetime(dates, utc=True, format='mixed'))
    counts = support[:, forest].sum(1)
    coverage = counts / int(forest.sum())
    good = coverage >= 0.8
    require(int(fortnight.sum()) == 113, '113 regular dates')
    trajectory = []
    for i in np.flatnonzero(fortnight):
        for j, t in zip(fs.SELECT, fs.SHORT):
            trajectory.append(dict(site=name, site_id=sid, trait=t, date=dates[i], year=int(tt[i].year), kind='Production NN', mean=float(mean[i, j][forest & support[i]].mean()) if good[i] else np.nan, forest_pixels=int(forest.sum()), supported_forest_pixels=int(counts[i]), coverage=float(coverage[i]), context_boundary_truncated=bool(boundary[i]), run_id=CFG['run_id'], source_file=str(cube_file), source_sha256=cube_hash))
    access = fs.AccessGate(PROD / 'bundle/fold_assignment_v3.parquet', mode='execute')
    require(sha(PROD / 'bundle/fold_assignment_v3.parquet') == sha(FIG / 'inputs/fold_assignment_v3.parquet'), 'Development assignment identity')
    legal = allowed()
    legal = legal[legal.site_id.eq(sid)].sort_values(['date', 'event_id'])
    teacher = fs.teacher(sid)
    references = []
    for row in legal.itertuples():
        access.authorize_events([row.event_id])
        products = fs.products_for(teacher, row.event_id, access)
        mosaic = np.full((11, 50, 50), np.nan, 'f4')
        filled = np.zeros((50, 50), bool)
        for product in products:
            values, qa = fs.read_teacher(teacher, product, row.event_id, access, (slice(*spec['rows']), slice(*spec['cols'])))
            take = qa & forest & ~filled
            mosaic[:, take] = values[:, take]
            filled |= take
        n = int(filled.sum())
        fraction = n / int(forest.sum())
        plot = fraction >= 0.8
        for j, trait in zip(fs.SELECT, fs.SHORT):
            references.append(dict(site=name, site_id=sid, event_id=row.event_id, date=row.date, year=int(row.date[:4]), trait=trait, mean=float(mosaic[j, filled].mean()) if n else np.nan, qa_forest_pixels=n, forest_pixels=int(forest.sum()), teacher_qa_coverage=fraction, plot=plot, omission_reason='' if plot else 'teacher forest QA coverage below80%', products=json.dumps(products)))
    refs = pd.DataFrame(references)
    write_csv(refs, 'Fig4_reference_candidates', name)
    refs['kind'] = 'EnMAP reference'
    refs['date_utc_earliest'] = refs.products.map(lambda s: min((v['time'] for v in json.loads(s))))
    refs['run_id'] = 'frozen EnMAP teacher'
    refs['coverage'] = refs.teacher_qa_coverage
    refs['supported_forest_pixels'] = refs.qa_forest_pixels
    write_csv(refs, 'Fig4_reference_all', name)
    write_csv(refs[(refs.coverage > 0) & (refs.coverage < 0.8)], 'Fig4_partial_references', name)
    write_csv(refs[~refs['plot']], 'Fig4_omitted_development_references', name)
    chosen = refs[refs['plot']].copy()
    write_csv(chosen, 'Fig4_reference_used', name)
    write_csv(pd.concat([pd.DataFrame(trajectory), chosen], ignore_index=True), 'Fig4_trajectories', name)
    a = access.assignment
    excluded = a[a.site_id.eq(sid) & ~a.event_id.isin(access.allowed_events)][['event_id', 'date', 'new_role']]
    write_csv(excluded, 'Fig4_sealed_or_excluded_reference_dates', name)
    cv = pd.DataFrame(dict(date=dates, year=tt.year, is_fortnightly=fortnight, forest_pixels=int(forest.sum()), supported_forest_pixels=counts, coverage=coverage, plot=good, context_boundary_truncated=boundary))
    write_csv(cv, 'Fig4_coverage', name)
    yr = cv[cv.is_fortnightly].groupby('year').agg(n_dates=('date', 'size'), n_dates_ge80=('plot', 'sum'), mean_forest_coverage=('coverage', 'mean'), min_forest_coverage=('coverage', 'min'), n_context_boundary_flags=('context_boundary_truncated', 'sum')).reset_index()
    yr['fraction_dates_ge80'] = yr.n_dates_ge80 / yr.n_dates
    write_csv(yr, 'Fig4_coverage_by_year', name)
    ix = np.flatnonzero(fortnight & (tt.year == 2024))
    require(len(ix) == 26, '26 cube dates')
    candidates = ix[good[ix] & tt[ix].month.isin([6, 7, 8])]
    require(len(candidates) > 0, 'No passing2024 JJA front-face date')
    target = pd.Timestamp('2024-07-27T12:00:00Z')
    front = min(candidates, key=lambda i: (abs(tt[i] - target), tt[i]))
    front_index = int(np.flatnonzero(ix == front).item())
    np.savez_compressed(output_folder / f'Fig4_cube_{name}.npz', data=np.transpose(mean[ix][:, fs.SELECT], (1, 0, 2, 3)), trait_names=np.asarray(fs.SHORT), display_dates=dates[ix], parent_y=py, parent_x=px, t0=front_index, site=name, forest_mask=forest, w7_support=support[ix])
    candidate = pd.read_csv(FIG / 'site_alternatives/candidate_table.csv')
    candidate = candidate[candidate.site_id.eq(sid)].iloc[0]
    used = sorted(chosen.date.unique().tolist())
    expected = str(candidate.ref80_dates).split('|')
    bad = ~good[fortnight]
    regular_dates = tt[fortnight]
    runs = []
    begin = None
    for i, missing in enumerate(np.r_[bad, False]):
        if missing and begin is None:
            begin = i
        if not missing and begin is not None:
            runs.append((i - begin, begin, i - 1))
            begin = None
    gap = max(runs, default=(0, 0, 0))
    record = dict(name=name, station=spec['station'], site_id=sid, production_run=CFG['run_id'], window=spec, forest_pixels=int(forest.sum()), forest_fraction=float(forest.mean()), n_dates_passing=int(good[fortnight].sum()), n_product_dates=113, front_face_date=str(tt[front].date()), front_face_coverage=float(coverage[front]), front_index=front_index, front_face_rule='2024-07-27 if forest W7>=80%; otherwise nearest passing June-August2024 product date; ties earlier', reference_dates_used=used, reference_dates_omitted=sorted(refs[~refs['plot']].date.unique().tolist()), references_in_sample=True, reference_count_expected=int(candidate.n_forest_ref80), reference_count_actual=len(used), reference_expectation_matches=used == expected, reference_difference_explanation='' if used == expected else 'New references additionally require all11 finite; the frozen alternatives audit used valid_mask only', sealed_dates_metadata_only=sorted(a[a.site_id.eq(sid) & a.new_role.eq('fold0')].date.tolist()), longest_gap_dates=int(gap[0]), longest_gap_days=int(gap[0] * 14), longest_gap_start=str(regular_dates[gap[1]].date()) if gap[0] else None, longest_gap_end=str(regular_dates[gap[2]].date()) if gap[0] else None, source_cube_sha256=cube_hash, model_sha256=m['model_sha256'], feature_wall_seconds=read(folder / 'FEATURES_DONE.json')['elapsed_wall_seconds'], inference_wall_seconds=m['elapsed_wall_seconds'], feature_CPU_seconds=read(folder / 'FEATURES_DONE.json')['CPU_seconds'], inference_CPU_seconds=m['CPU_seconds'], GPU_hours=0, source_csvs={p.name: sha(p) for p in (FIG / 'source_data').glob(f'Fig4_*_{name}.csv')}, sealed_values_read=0)
    save(output_folder / 'FIG4_DATA_DONE.json', dict(sites={name: record}))
    print(json.dumps(record, indent=2), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('site', choices=SITES)
    a = p.parse_args()
    main(a.site)
