'frozen development panel and valid teacher/S2 masks; writes pre-inference SI3 candidate table;temporal run.\nInputs: code/paths.py; outputs: explicit derivation work directory, never upstream.\nOriginal: upstream://V6/final_figures_20261113/scripts/select_si3_scene.py.\nRun IDs: cap2048_v3_nested11_20261108; cap2048_spatial_gsingle_20261112; cap2048_production_nn_temporal_20261112.\nHeavy derivation was not rerun when packaging. Use --help before --run.'
from __future__ import annotations
import paths,sys
if __name__=="__main__":
 from make_runtime import cli
 cli(__doc__)
sys.path.insert(0,str(paths.MODEL_CODE))
from make_support import *
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_site_support as fs
RULE = 'Development + known site in temporal_Y primary population; May-September if any; descending full-scene union(valid_mask & all11 finite & exact W7 support & WorldCover not0/70/80); tie descending forest-window teacher QA fraction; then chronological date/event_id. Earliest UTC/product_id, first common-valid source.'

def main():
    sid = 32
    access = fs.gate()
    meta = pd.read_parquet(DATA / 'formal_temporal_primary.parquet', columns=['target_event_id', 'site_id'])
    primary = set(meta[meta.site_id.eq(sid)].target_event_id)
    g = fs.teacher(sid)
    lc = fs.land(sid)
    forest = lc[475:525, 475:525] == 10
    cat = fs.s2_catalog(sid)
    ct = np.array([s['time'] for s in cat], dtype='datetime64[ns]')
    rows = []
    events = access.assignment[access.assignment.site_id.eq(sid) & access.assignment.event_id.isin(access.allowed_events)].sort_values(['date', 'event_id'])
    for e in events.itertuples():
        products = fs.products_for(g, e.event_id, access)
        qa_union = np.zeros((1000, 1000), bool)
        common = np.zeros_like(qa_union)
        for p in products:
            v, qa = fs.read_teacher(g, p, e.event_id, access)
            qa_union |= qa
            if e.event_id in primary:
                dt = np.abs((ct - np.datetime64(p['time'])) / np.timedelta64(1, 'D'))
                supported = np.zeros_like(qa)
                for i in np.flatnonzero(dt <= 7):
                    supported |= fs.supported30(cat[int(i)], DATA / 'SI3_support_cache_Castelporziano')
                common |= qa & supported & ~np.isin(lc, [0, 70, 80])
        month = pd.Timestamp(e.date).month
        rows.append(dict(site='Castelporziano', site_id=sid, event_id=e.event_id, date=e.date, year=int(e.year), month=month, known_site=e.event_id in primary, growing_season=5 <= month <= 9, full_scene_teacher_qa_pixels=int(qa_union.sum()), common_valid_pixels=int(common.sum()), forest_teacher_qa_fraction=float(qa_union[475:525, 475:525][forest].mean()), forest_pixels=int(forest.sum()), products=json.dumps(products)))
        print('CANDIDATE', e.event_id, int(common.sum()), flush=True)
    f = pd.DataFrame(rows)
    pool = f[f.known_site]
    fallback = not pool.growing_season.any()
    pool = pool if fallback else pool[pool.growing_season]
    pool = pool.sort_values(['common_valid_pixels', 'forest_teacher_qa_fraction', 'date', 'event_id'], ascending=[False, False, True, True])
    assert len(pool) and pool.iloc[0].common_valid_pixels > 0
    f['rank'] = f.event_id.map({e: i + 1 for i, e in enumerate(pool.event_id)})
    f['selected'] = f.event_id.eq(pool.iloc[0].event_id)
    f['all_months_fallback'] = fallback
    f['selection_rule'] = RULE
    path = DATA / 'SI3_scene_selection_Castelporziano.csv'
    f.to_csv(path, index=False)
    best = pool.iloc[0].to_dict()
    best.update(selection_rule=RULE, all_months_fallback=fallback, selection_csv_sha256=sha(path))
    save(DATA / 'SI3_scene_selection_Castelporziano.json', best)
    save(DATA / 'SI3_s2_catalog_Castelporziano.json', cat)
    print('SELECTED', best['event_id'], best['common_valid_pixels'], flush=True)
if __name__ == '__main__':
    main()
