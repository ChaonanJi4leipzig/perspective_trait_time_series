"""accepted tower records,public metadata and local WorldCover;writes display classes;inventory.
Inputs: code/paths.py; outputs: explicit derivation work directory, never upstream.
Original: upstream://V6/final_figures_20261113/si1_pc/journal/build_classes.py.
Run IDs: cap2048_v3_nested11_20261108; cap2048_spatial_gsingle_20261112; cap2048_production_nn_temporal_20261112.
Heavy derivation was not rerun when packaging. Use --help before --run."""
from __future__ import annotations
import paths, sys
if __name__ == '__main__':
    from make_runtime import cli
    cli(__doc__)
sys.path.insert(0, str(paths.MODEL_CODE))
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[k] = '1'
from pathlib import Path
import json, hashlib
from datetime import datetime, timezone
import numpy as np, pandas as pd, zarr, numcodecs
from affine import Affine
from pyproj import Transformer, Geod
W = paths.WORK
OUT = paths.WORK
PROJECT = paths.PROJECT
DATA = OUT / 'source_data'
numcodecs.blosc.set_nthreads(1)
zarr.config.set({'async.concurrency': 1, 'threading.max_workers': 1})
EARLIER = paths.EARLIER_CLASSES
BASE = DATA / 'SI1_site_classes.csv'
old = pd.read_csv(BASE).fillna('')
earlier = pd.read_csv(EARLIER).fillna('').set_index('site_id')
centres = pd.read_csv(paths.SI1 / 'site_centres.csv').set_index('site_id')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def class_for(code):
    if code in ['ENF', 'EBF', 'DNF', 'DBF', 'MF']:
        return 'forest'
    if code == 'GRA':
        return 'grassland'
    if code == 'CRO':
        return 'cropland'
    if code == 'WET':
        return 'wetland'
    if code in ['OSH', 'CSH', 'SAV', 'WSA']:
        return 'shrubland_savanna'
    if code in ['WAT', 'BSV', 'URB', 'SNO']:
        return 'other'
    return 'unclassified'
WC_MAP = {0: 'unclassified', 10: 'forest', 20: 'shrubland_savanna', 30: 'grassland', 40: 'cropland', 50: 'other', 60: 'other', 70: 'other', 80: 'other', 90: 'wetland', 95: 'wetland', 100: 'other'}
WC_NAMES = {0: 'NoData', 10: 'Tree cover', 20: 'Shrubland', 30: 'Grassland', 40: 'Cropland', 50: 'Built-up', 60: 'Bare/sparse vegetation', 70: 'Snow/ice', 80: 'Permanent water bodies', 90: 'Herbaceous wetland', 95: 'Mangroves', 100: 'Moss/lichen'}
PUBLIC = {6: dict(igbp='WAT', display_class='other', tower_id='DE-Dgw', latitude=53.1514, longitude=13.0543, source='https://fluxnet.org/sites/siteinfo/DE-Dgw', evidence='FLUXNET station DE-Dgw Dagowsee; identical local coordinates; station IGBP WAT (water bodies).'), 25: dict(igbp='WET', display_class='wetland', tower_id='FI-Lom', latitude=67.99724, longitude=24.209179, source='https://meta.icos-cp.eu/resources/stations/ES_FI-Lom', evidence='ICOS FI-Lom Lompolojankka; main ecosystem Permanent Wetlands; pristine minerotrophic mire; coordinates agree within 5 m.'), 47: dict(igbp='', display_class='forest', tower_id='', latitude=51.1833, longitude=-0.85, source='https://deims.org/d47ec839-5d20-4315-9f88-1e9edbab22e8', evidence='Published DEIMS-SDR Alice Holt station description, operated by Forest Research/UK ECN; exact representative coordinates; ecosystem Forest/Mixed Forest. This is a regional station identity, so no FLUXNET tower ID or tower IGBP code is inferred.')}
SHORT = {0: 'Neustift', 2: 'Chamau', 3: 'Laegern', 6: 'Dagowsee', 7: 'Gebesee', 9: 'Hainich', 10: 'Huetelmoor', 13: 'Leinefelde', 15: 'Schechenfilz N.', 16: 'Tharandt', 17: 'Zarnekow', 19: 'Amoladeras', 20: 'Llano de los Juanes', 22: 'Lanjaron', 23: 'Hyytiala', 24: 'Lettosuo', 25: 'Lompolojankka', 26: 'Sodankyla', 28: 'Grignon', 29: 'Puechabon', 31: 'Castellaro', 32: 'Castelporziano', 33: 'Ispra', 35: 'Monte Bondone', 36: 'Parco Ticino', 37: 'Renon', 39: 'Torgnon', 40: 'Loobos', 41: 'Degero', 43: 'Le Bray', 44: 'Bilos', 45: 'Hesse', 46: 'Font-Blanche', 47: 'Alice Holt', 48: 'Wetzstein', 49: 'Norunda', 52: 'Mitra', 53: 'Las Majadas', 54: 'Landes forest', 56: 'Les Vosges', 57: 'Bavarian Forest'}
rows = []
decisions = []
conflicts = []
worldcover = []
for r in old.itertuples(index=False):
    sid = r.site_id
    e = earlier.loc[sid]
    centre = centres.loc[sid]
    if r.display_class != 'unknown':
        assert r.match_distance_m <= 300 and r.fluxnet_site_id
        tier = 1
        igbp = r.igbp
        display = class_for(igbp)
        tower = r.fluxnet_site_id
        source = r.source
        sourcehash = r.source_sha256
        evidence = f'Strict name/code agreement and centre-to-shuttle coordinate distance {r.match_distance_m:.3f} m (≤300 m); tower IGBP {igbp}; accepted existing assignment.'
        decisions.append(dict(site_id=sid, source_tier=2, candidate_code=e.igbp, candidate_class=class_for(e.igbp), status='agrees_with_higher_precedence_tier_1', evidence=e.land_cover_evidence, source=str(EARLIER)))
    elif e.fluxnet_match_method == 'historical_fluxnet_site_record_exact_coordinate' and e.igbp:
        assert float(e.distance_km) <= 0.3
        assert abs(float(e.latitude) - centre.latitude) < 1e-06 and abs(float(e.longitude) - centre.longitude) < 1e-06
        tier = 2
        igbp = e.igbp
        display = class_for(igbp)
        tower = e.fluxnet_site_id
        source = str(EARLIER)
        sourcehash = sha(EARLIER)
        evidence = f'{e.land_cover_evidence}; {e.fluxnet_site_id} {e.fluxnet_site_name}; {e.land_cover_detail}; local source coordinates identical; include={e.include} is an old modelling-domain flag, not an ecosystem-validity flag.'
    else:
        if sid == 25:
            note = 'Reject earlier FI-Ken/Kenttarova ENF mapping: wrong station name and 1794.073 m separation; not an eligible higher-precedence assignment. Use exact-match FI-Lom public metadata.'
            conflicts.append(dict(site_id=sid, site=r.site_name, rejected_tier=2, rejected_assignment='FI-Ken Kenttarova / ENF / forest', retained_tier=3, retained_assignment='FI-Lom Lompolojankka / WET / wetland', resolution=note, source=str(EARLIER)))
        elif e.land_cover_class != 'Unmapped':
            note = 'Earlier assignment is based on WorldCover + ForestPaths + ForTy inference, not a tower/literature assignment; do not promote it to tier 2.'
        else:
            note = 'Earlier table provides no reliable ecosystem assignment.'
        decisions.append(dict(site_id=sid, source_tier=2, candidate_code=e.igbp or e.local_land_cover_code, candidate_class=class_for(e.land_cover_class), status='not_eligible', evidence=note, source=str(EARLIER)))
        if sid in PUBLIC:
            p = PUBLIC[sid]
            tier = 3
            igbp = p['igbp']
            display = p['display_class']
            tower = p['tower_id']
            source = p['source']
            sourcehash = ''
            evidence = p['evidence']
            distance = Geod(ellps='WGS84').inv(centre.longitude, centre.latitude, p['longitude'], p['latitude'])[2]
            assert distance < 10
        else:
            assert sid in [54, 56, 57]
            tier = 4
            igbp = ''
            tower = ''
            p = PROJECT / f'experiments/enmap_v2_worldcover_source_topup_20261022/worldcover_zarr/site_{sid}_worldcover2021_30m.zarr'
            a = zarr.open_array(str(p), mode='r')
            attrs = dict(a.attrs)
            tr = Affine(*attrs['transform'][:6])
            cx, cy = Transformer.from_crs('EPSG:4326', attrs['crs'], always_xy=True).transform(centre.longitude, centre.latitude)
            cc, rr = ~tr * (cx, cy)
            dx = abs(tr.a)
            dy = abs(tr.e)
            x0 = max(0, int(np.floor(cc - 500 / dx - 1)))
            x1 = min(a.shape[-1], int(np.ceil(cc + 500 / dx + 1)))
            y0 = max(0, int(np.floor(rr - 500 / dy - 1)))
            y1 = min(a.shape[-2], int(np.ceil(rr + 500 / dy + 1)))
            sub = np.asarray(a[0, y0:y1, x0:x1] if a.ndim == 3 else a[y0:y1, x0:x1])
            yy, xx = np.meshgrid(np.arange(y0, y1) + 0.5, np.arange(x0, x1) + 0.5, indexing='ij')
            xs = tr.a * xx + tr.b * yy + tr.c
            ys = tr.d * xx + tr.e * yy + tr.f
            mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= 500 ** 2
            v = sub[mask]
            valid = v[v != 0]
            codes, n = np.unique(valid, return_counts=True)
            code = int(codes[np.argmax(n)])
            assert len(valid) == len(v) and len(valid) > 0
            display = WC_MAP[code]
            source = str(p)
            sourcehash = sha(p / '.zattrs')
            fraction = float((valid == code).mean())
            evidence = f'Local ESA WorldCover 2021 v200, 30 m nearest-neighbour map; fixed 500 m radius in projected metre CRS; pixel-centre inclusion; dominant code {code} ({WC_NAMES[code]}) = {(valid == code).sum()}/{len(valid)} pixels ({fraction:.2%}); valid coverage {len(valid)}/{len(v)}. Mapped to display class only; no tower ID or IGBP subtype inferred.'
            worldcover.append(dict(site_id=sid, site=r.site_name, radius_m=500, n_pixels=len(v), n_valid=len(valid), dominant_worldcover_code=code, dominant_fraction=fraction, histogram_json=json.dumps(dict(zip(codes.tolist(), n.tolist()))), source=str(p), zattrs_sha256=sourcehash, centre_longitude=centre.longitude, centre_latitude=centre.latitude))
            decisions.append(dict(site_id=sid, source_tier=3, candidate_code='', candidate_class='', status='no_unique_station_identity_at_local_coordinates', evidence='Generic regional name does not establish an exact tower identity; nearby station IDs are not transferred by nearest-only matching.', source='anonymous public station metadata search'))
    label = (tower + ' ' if tower else '') + SHORT[sid] + ('†' if tier > 1 else '')
    rows.append(dict(site_id=sid, site=r.site_name, site_name=r.site_name, short_name=SHORT[sid], tower_id=tower, plot_label=label, display_class=display, igbp=igbp, source_tier=tier, evidence_text=evidence, source=source, source_sha256=sourcehash, dagger=tier > 1, original_display_class=r.display_class, earlier_land_cover_detail=e.land_cover_detail, earlier_land_cover_evidence=e.land_cover_evidence, earlier_site_domain=e.site_domain, earlier_include=e.include))
filled = pd.DataFrame(rows)
filled.to_csv(DATA / 'SI1_site_classes_filled.csv', index=False)
pd.DataFrame(decisions).to_csv(W / 'source_decisions.csv', index=False)
pd.DataFrame(conflicts).to_csv(W / 'class_conflicts.csv', index=False)
pd.DataFrame(worldcover).to_csv(W / 'worldcover_500m_audit.csv', index=False)
pd.DataFrame([dict(worldcover_code=k, worldcover_name=WC_NAMES[k], display_class=v) for k, v in WC_MAP.items()]).to_csv(W / 'worldcover_class_mapping.csv', index=False)
summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), n_sites=len(filled), n_filled=int((filled.original_display_class == 'unknown').sum()), tier_counts=filled.source_tier.value_counts().sort_index().to_dict(), filled_tier_counts=filled[filled.original_display_class == 'unknown'].source_tier.value_counts().sort_index().to_dict(), class_counts=filled.display_class.value_counts().to_dict(), unclassified=filled.loc[filled.display_class.eq('unclassified'), 'site'].tolist(), n_conflicts=len(conflicts), old_classes_sha256=sha(BASE), earlier_classes_sha256=sha(EARLIER), public_metadata=PUBLIC)
assert len(filled) == 41 and filled.site_id.is_unique and (len(summary['class_counts']) <= 6)
(W / 'classes_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
