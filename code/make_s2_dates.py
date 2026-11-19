'corrected S2 stores and development assignment; writes inventories and Fig4 acquisition ticks;production run.\nInputs: code/paths.py; outputs: explicit derivation work directory, never upstream.\nOriginal: upstream://V6/final_figures_20261113/scripts/s2_sources.py.\nRun IDs: cap2048_v3_nested11_20261108; cap2048_spatial_gsingle_20261112; cap2048_production_nn_temporal_20261112.\nHeavy derivation was not rerun when packaging. Use --help before --run.'
from __future__ import annotations
import paths,sys
if __name__=="__main__":
 from make_runtime import cli
 cli(__doc__)
sys.path.insert(0,str(paths.MODEL_CODE))
from make_support import *
import zarr, numcodecs
numcodecs.blosc.set_nthreads(1)
zarr.config.set({'async.concurrency': 1, 'threading.max_workers': 1})
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_site_support as fs

def inventory():
    old = V6.parent / 'figures_v4/tables'
    obs = pd.read_csv(old / 's2_observations.csv')
    inv = pd.read_csv(old / 's2_store_metadata_inventory.csv')
    access = fs.gate()
    a = access.assignment
    events = a[a.event_id.isin(access.allowed_events)].copy()
    classes = pd.read_csv(old / 'site_colour_classes.csv')
    sites = set(events.site_id)
    inventory = []
    for r in inv[inv.site_id.isin(sites)].itertuples():
        p = Path(r.path)
        assert sha(p / '.zattrs') == r.zattrs_sha256
        assert sha(p / 'time/.zarray') == r.time_header_sha256
        inventory.append(dict(site_id=r.site_id, year=r.year, path=r.path, attrs_sha256=r.zattrs_sha256))
    start = pd.Timestamp('2022-04-01', tz='UTC')
    end = pd.Timestamp('2026-10-01', tz='UTC')
    obs['date_utc'] = pd.to_datetime(obs.date_utc, utc=True, format='mixed')
    obs = obs[obs.site_id.isin(sites) & obs.date_utc.ge(start) & obs.date_utc.lt(end)]
    obs.to_csv(DATA / 'SI1_S2_observations.csv', index=False)
    events.to_csv(DATA / 'SI1_development_events.csv', index=False)
    classes.to_csv(DATA / 'SI1_site_classes.csv', index=False)
    counts = []
    for sid in sorted(sites):
        for year in range(2022, 2027):
            present = any((r['site_id'] == sid and r['year'] == year for r in inventory))
            e = events[events.site_id.eq(sid) & events.year.eq(year)]
            s = obs[obs.site_id.eq(sid) & obs.year.eq(year)]
            counts.append(dict(site_id=sid, site_name=e.site_name.iloc[0] if len(e) else events[events.site_id.eq(sid)].site_name.iloc[0], year=year, n_enmap_development=len(e), n_s2_valid=int(s.valid_observation.sum()) if present else np.nan, S2_store_present=present, start='2022-04-01' if year == 2022 else f'{year}-01-01', end='2026-09-30' if year == 2026 else f'{year}-12-31'))
    pd.DataFrame(counts).to_csv(DATA / 'SI1_counts_site_year.csv', index=False)
    save(DATA / 'SI1_inventory_validation.json', dict(old_inventory_reused=True, source_inventory_sha256=sha(old / 's2_store_metadata_inventory.csv'), source_observation_sha256=sha(old / 's2_observations.csv'), all_store_metadata_hashes_match=True, stores=inventory, sealed_events_excluded_metadata_only=int(a.new_role.eq('fold0').sum()), events=len(events), start='2022-04-01', end='2026-09-30', valid_definition='advanced_valid_mask>=1 on nearest20m at30m trait centres; >=50% of AOI WorldCover vegetation10/20/30/40/90/95; unchanged inventories.py definition'))
def ticks(names=None):
    for name, sid, y0, x0 in [('Hesse', 45, 480, 460), ('Castelporziano', 32, 475, 475)]:
        if names is not None and name not in names:continue
        forest = fs.land(sid)[y0:y0 + 50, x0:x0 + 50] == 10
        yy = np.floor((np.arange(y0, y0 + 50) + 0.5) * 1.5).astype(int)
        xx = np.floor((np.arange(x0, x0 + 50) + 0.5) * 1.5).astype(int)
        rows = []
        for year in range(2022, 2027):
            p = PROJECT / f'input_data/s2/region/site_{sid}/site_{sid}_s2_region_{year}.zarr'
            if not p.exists():
                continue
            h = sha(p / '.zattrs')
            g = zarr.open_group(str(p), mode='r')
            tt = fs.times(g)
            assert g['scl_20m'].chunks[0] == g['advanced_valid_mask'].chunks[0] == 1
            for i, date in enumerate(tt):
                if date < np.datetime64('2022-06-01') or date >= np.datetime64('2026-10-01'):
                    continue
                scl = np.asarray(g['scl_20m'].get_orthogonal_selection((i, yy, xx)))
                advanced = np.asarray(g['advanced_valid_mask'].get_orthogonal_selection((i, yy, xx))) > 0
                valid = np.isin(scl, [4, 5, 6]) & advanced
                fraction = float(valid[forest].mean())
                rows.append(dict(site=name, site_id=sid, date_utc=str(date), year=int(str(date)[:4]), valid_forest_pixels=int((valid & forest).sum()), forest_pixels=int(forest.sum()), valid_fraction=fraction, plot=fraction >= 0.5, rule='SCL4/5/6 & advanced_valid_mask, >=50% fixed-window WorldCover10 forest', store=str(p), store_attrs_sha256=h))
            assert sha(p / '.zattrs') == h
        pd.DataFrame(rows).to_csv(DATA / f'Fig4_S2_valid_dates_{name}.csv', index=False)
        print('S2_TICKS', name, len(rows), sum((r['plot'] for r in rows)), flush=True)
def main(names=None,ticks_only=False):
    if not ticks_only:inventory()
    ticks(names)

if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--site',choices=['Hesse','Castelporziano'])
    p.add_argument('--ticks-only',action='store_true')
    args=p.parse_args()
    main([args.site] if args.site else None,args.ticks_only)
