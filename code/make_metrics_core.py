'canonical frozen rows; pure three-level metrics helper; temporal run.\nInputs: code/paths.py; outputs: explicit derivation work directory, never upstream.\nOriginal: upstream://V6/final_figures_20261113/scripts/hierarchy_metrics.py.\nRun IDs: cap2048_v3_nested11_20261108; cap2048_spatial_gsingle_20261112; cap2048_production_nn_temporal_20261112.\nHeavy derivation was not rerun when packaging. Use --help before --run.'
from __future__ import annotations
import paths,sys
if __name__=="__main__":
 from make_runtime import cli
 cli(__doc__)
sys.path.insert(0,str(paths.MODEL_CODE))
from collections.abc import Mapping, Sequence
import numpy as np
import pandas as pd
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 20261111
TRAIT_ORDER = ('LAI', 'LMA', 'Chl', 'EWT', 'Car', 'N', 'C', 'NSC', 'Lignin', 'Cellulose', 'Fibre')
LEVELS = ('site', 'date', 'scene')
STATISTICS = tuple((f'{prefix}_{level}' for prefix in ('r2', 'variance_share', 'recovered_share', 'v') for level in LEVELS)) + ('delta_r', 'delta_sign_agreement', 'delta_amplitude_ratio', 'delta_ols_slope', 'climatology_date_r2')

def _var(values):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return np.nan
    if np.ptp(x) == 0:
        return 0.0
    return float(np.mean((x - x.mean()) ** 2))

def _r2(error_sum, reference_sum):
    return float(1 - error_sum / reference_sum) if reference_sum > 0 else np.nan

def _flag(values, name):
    if not values.dropna().isin([True, False, 0, 1]).all():
        raise ValueError(f'{name} must be boolean/0/1; decode QC upstream')
    return values.fillna(False).to_numpy(dtype=bool)

def prepare_samples(frame, *, prediction_columns, baseline_columns=(), shared_columns=(), reference_column='reference', qc_column='qc', oos_columns=('oos',)):
    """Return a shared masked table and its positional mask in original rows.

    prediction_columns maps seed ID to column for ONE model. shared_columns
    can additionally constrain other candidate models without averaging them
    as if they were seeds. Baselines are never estimated here. Eligibility
    (at least two OOS scenes/site) is evaluated AFTER the common mask.
    """
    if not isinstance(prediction_columns, Mapping) or not prediction_columns:
        raise ValueError('prediction_columns must map at least one seed to a column')
    seeds = {str(k): v for k, v in prediction_columns.items()}
    if len(seeds) != len(prediction_columns):
        raise ValueError('seed IDs are ambiguous after string conversion')
    seeds = dict(sorted(seeds.items()))
    if not oos_columns:
        raise ValueError('explicit OOS flag column(s) required')
    signals = list(dict.fromkeys([*seeds.values(), *baseline_columns, *shared_columns]))
    required = ['site_id', 'event_id', 'date_utc', 'pixel_id', reference_column, qc_column, *oos_columns, *signals]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'missing canonical columns: {sorted(missing)}')
    f = frame.copy()
    f['date_utc'] = pd.to_datetime(f.date_utc, utc=True, errors='raise', format='mixed')
    mask = _flag(f[qc_column], qc_column)
    for column in oos_columns:
        mask &= _flag(f[column], column)
    for column in [reference_column, *signals]:
        mask &= np.isfinite(f[column].to_numpy(dtype=float))
    selected = f.loc[mask].copy()
    for label in ['scope', 'model', 'trait', 'data_version']:
        if label in selected and selected[label].nunique(dropna=False) > 1:
            raise ValueError(f'one {label} per evaluation call required')
    if selected[['site_id', 'event_id', 'date_utc', 'pixel_id']].isna().any().any():
        raise ValueError('valid rows need complete identity/UTC metadata')
    if selected.duplicated(['site_id', 'event_id', 'pixel_id']).any():
        raise ValueError('duplicate scene/pixel identity: resolve product overlap upstream')
    if selected.groupby(['site_id', 'event_id']).date_utc.nunique().gt(1).any():
        raise ValueError('one canonical UTC per event required; retain product UTC separately')
    scene_meta = selected[['site_id', 'event_id', 'date_utc']].drop_duplicates()
    if scene_meta.duplicated(['site_id', 'date_utc']).any():
        raise ValueError('multiple event IDs at the same site UTC; define scene grouping upstream')
    counts = selected.groupby('site_id').event_id.nunique()
    eligible = set(counts[counts.ge(2)].index)
    mask &= f.site_id.isin(eligible).to_numpy()
    data = f.loc[mask].copy().reset_index(drop=True)
    return dict(data=data, shared_mask=mask, predictions=seeds, baselines=list(baseline_columns), reference=reference_column, mask_columns=[reference_column, *signals, qc_column, *oos_columns], n_input=len(f), n_sites=data.site_id.nunique())

def scene_statistics(prepared):
    """Sufficient statistics on the SAME valid pixels for every signal."""
    data = prepared['data']
    signals = list(dict.fromkeys([*prepared['predictions'].values(), *prepared['baselines']]))
    records = []
    for (site, event), g in data.groupby(['site_id', 'event_id'], sort=False):
        ref = g[prepared['reference']].to_numpy(dtype=float)
        mean_ref = float(ref.mean())
        rec = dict(site_id=site, event_id=event, date_utc=g.date_utc.iloc[0], n=len(g), m_ref=mean_ref, v_ref=_var(ref))
        for signal in signals:
            pred = g[signal].to_numpy(dtype=float)
            mean_pred = float(pred.mean())
            rec['m::' + signal] = mean_pred
            rec['mse_within::' + signal] = float(np.mean((pred - mean_pred - (ref - mean_ref)) ** 2))
        records.append(rec)
    columns = ['site_id', 'event_id', 'date_utc', 'n', 'm_ref', 'v_ref']
    columns += [key + signal for signal in signals for key in ['m::', 'mse_within::']]
    return pd.DataFrame(records, columns=columns)

def _site_components(values, sites):
    x = np.asarray(values, dtype=float)
    codes, labels = pd.factorize(sites, sort=False)
    counts = np.bincount(codes)
    means = np.bincount(codes, weights=x) / counts
    anomaly = x - means[codes]
    for code in range(len(labels)):
        ix = codes == code
        if np.ptp(x[ix]) == 0:
            means[code] = x[ix][0]
            anomaly[ix] = 0
    return (means, means[codes], anomaly)

def three_level_metrics(scenes, prediction_column):
    """Site R² equal-sites; date and scene R² equal-scenes, as in §3."""
    if scenes.empty:
        return {f'{p}_{l}': np.nan for p in ['r2', 'v', 'variance_share', 'recovered_share'] for l in LEVELS}
    ref_site, mapped_ref_site, ref_date = _site_components(scenes.m_ref, scenes.site_id)
    pred_site, _, pred_date = _site_components(scenes['m::' + prediction_column], scenes.site_id)
    r2_site = _r2(np.sum((pred_site - ref_site) ** 2), len(ref_site) * _var(ref_site))
    r2_date = _r2(np.sum((pred_date - ref_date) ** 2), np.sum(ref_date ** 2))
    r2_scene = _r2(scenes['mse_within::' + prediction_column].mean(), scenes.v_ref.mean())
    variances = [_var(mapped_ref_site), float(np.mean(ref_date ** 2)), float(scenes.v_ref.mean())]
    total = sum(variances)
    out = {}
    for level, variance, score in zip(LEVELS, variances, [r2_site, r2_date, r2_scene]):
        share = variance / total if total > 0 else np.nan
        recovered = 0.0 if share == 0 else share * max(score, 0) if np.isfinite(score) else np.nan
        out.update({f'r2_{level}': score, f'v_{level}': variance, f'variance_share_{level}': share, f'recovered_share_{level}': recovered})
    return out

def climatology_date_r2(scenes, climatology_column):
    """Site-season scene means minus the BASELINE'S own equal-scene site mean."""
    if climatology_column is None or scenes.empty:
        return np.nan
    _, _, ref = _site_components(scenes.m_ref, scenes.site_id)
    _, _, pred = _site_components(scenes['m::' + climatology_column], scenes.site_id)
    return _r2(np.sum((pred - ref) ** 2), np.sum(ref ** 2))

def make_delta_pairs(prepared, *, pixel_ids_stable):
    """Adjacent OOS scenes only. Explicit identity semantics, no fallback.

    pixel_ids_stable is bool or site->bool mapping. True uses the intersection
    of shared-valid pixel IDs; false uses each scene's own shared-valid rows.
    Empty common support or dt<3d stays as an excluded audit record; an
    excluded pair is NEVER replaced by a non-adjacent pair.
    """
    if not isinstance(pixel_ids_stable, (bool, np.bool_, Mapping)):
        raise ValueError('pixel_ids_stable must be explicit bool or site->bool mapping')
    data = prepared['data']
    signals = list(dict.fromkeys([*prepared['predictions'].values(), *prepared['baselines']]))
    records = []
    for site, g in data.groupby('site_id', sort=False):
        if isinstance(pixel_ids_stable, Mapping):
            if site not in pixel_ids_stable or not isinstance(pixel_ids_stable[site], (bool, np.bool_)):
                raise ValueError(f'explicit pixel identity semantics missing for site {site}')
            stable = bool(pixel_ids_stable[site])
        else:
            stable = bool(pixel_ids_stable)
        scenes = list(g.groupby('event_id', sort=False))
        scenes.sort(key=lambda item: item[1].date_utc.iloc[0])
        for (event1, first), (event2, second) in zip(scenes[:-1], scenes[1:]):
            dt = (second.date_utc.iloc[0] - first.date_utc.iloc[0]).total_seconds() / 86400
            rec = dict(site_id=site, event_1=event1, event_2=event2, date_1=first.date_utc.iloc[0], date_2=second.date_utc.iloc[0], delta_t_days=dt, mode='common_pixels' if stable else 'own_pixels', n_first=len(first), n_second=len(second), n_common=np.nan, included=False, exclusion_reason='', delta_ref=np.nan)
            rec.update({'delta::' + signal: np.nan for signal in signals})
            if dt < 3:
                rec['exclusion_reason'] = 'delta_t_lt_3_days'
            else:
                common = pd.Index(first.pixel_id).intersection(pd.Index(second.pixel_id), sort=False)
                rec['n_common'] = len(common)
                if stable and (not len(common)):
                    rec['exclusion_reason'] = 'no_common_valid_pixels'
                else:
                    if stable:
                        first = first.set_index('pixel_id').loc[common]
                        second = second.set_index('pixel_id').loc[common]
                    rec['included'] = True
                    rec['delta_ref'] = float(second[prepared['reference']].mean() - first[prepared['reference']].mean())
                    for signal in signals:
                        rec['delta::' + signal] = float(second[signal].mean() - first[signal].mean())
            records.append(rec)
    columns = ['site_id', 'event_1', 'event_2', 'date_1', 'date_2', 'delta_t_days', 'mode', 'n_first', 'n_second', 'n_common', 'included', 'exclusion_reason', 'delta_ref']
    return pd.DataFrame(records, columns=columns + ['delta::' + s for s in signals])

def delta_statistics(pairs, prediction_column, reference_span):
    """r/amplitude/OLS use all included pairs; sign uses ref-only dead zone."""
    if not np.isfinite(reference_span) or reference_span < 0:
        raise ValueError('reference_span must be finite and nonnegative')
    p = pairs.loc[pairs.included.eq(True)]
    ref = p.delta_ref.to_numpy(dtype=float)
    pred = p['delta::' + prediction_column].to_numpy(dtype=float)
    threshold = 0.05 * reference_span
    active = np.abs(ref) >= threshold
    vr, vp = (_var(ref), _var(pred))
    cov = float(np.mean((ref - ref.mean()) * (pred - pred.mean()))) if len(ref) else np.nan
    return dict(delta_r=cov / np.sqrt(vr * vp) if vr > 0 and vp > 0 else np.nan, delta_sign_agreement=float(np.mean(np.sign(pred[active]) == np.sign(ref[active]))) if active.any() else np.nan, delta_amplitude_ratio=float(np.sqrt(vp / vr)) if vr > 0 else np.nan, delta_ols_slope=cov / vr if vr > 0 else np.nan, n_pairs=len(ref), n_sign_pairs=int(active.sum()), dead_zone=threshold)

def _span(values):
    a = np.asarray(values, dtype=float)
    return float(np.subtract(*np.quantile(a, [0.95, 0.05]))) if len(a) else np.nan

def _per_seed(scenes, pairs, predictions, climatology_column, reference_span):
    records = []
    for seed, column in predictions.items():
        record = dict(seed=seed, **three_level_metrics(scenes, column))
        if np.isfinite(reference_span):
            record.update(delta_statistics(pairs, column, reference_span))
        else:
            record.update({k: np.nan for k in ['delta_r', 'delta_sign_agreement', 'delta_amplitude_ratio', 'delta_ols_slope', 'dead_zone']})
            record.update(n_pairs=0, n_sign_pairs=0)
        record['climatology_date_r2'] = climatology_date_r2(scenes, climatology_column)
        records.append(record)
    return pd.DataFrame(records)

def site_cluster_bootstrap(prepared, scenes, pairs, *, climatology_column=None, draws=BOOTSTRAP_DRAWS, random_seed=BOOTSTRAP_SEED):
    """Resample sites, retaining all scenes/pairs. Compute seed stats then mean.

    Repeated draws of the same site are independent cluster occurrences for
    site-mean R². The reference P95−P05/dead zone is re-estimated per draw on
    all sampled valid pixel references, including cluster multiplicities.
    Scene/pair sufficient statistics are reused; no model is evaluated here.
    """
    if not isinstance(draws, int) or draws < 0:
        raise ValueError('draws must be a nonnegative integer')
    data = prepared['data']
    sites = sorted(data.site_id.unique(), key=str)
    columns = ['draw', 'n_unique_source_sites', *STATISTICS]
    if not sites or draws == 0:
        return pd.DataFrame(columns=columns)
    scene_blocks = {site: scenes[scenes.site_id.eq(site)] for site in sites}
    pair_blocks = {site: pairs[pairs.site_id.eq(site)] for site in sites}
    reference_blocks = {site: data.loc[data.site_id.eq(site), prepared['reference']].to_numpy(dtype=float) for site in sites}
    rng = np.random.default_rng(random_seed)
    results = []
    for draw in range(draws):
        indices = rng.integers(0, len(sites), size=len(sites))
        ss, pp, rr = ([], [], [])
        for occurrence, index in enumerate(indices):
            site = sites[int(index)]
            sb = scene_blocks[site].copy()
            sb['site_id'] = occurrence
            ss.append(sb)
            pb = pair_blocks[site].copy()
            pb['site_id'] = occurrence
            pp.append(pb)
            rr.append(reference_blocks[site])
        per_seed = _per_seed(pd.concat(ss, ignore_index=True), pd.concat(pp, ignore_index=True), prepared['predictions'], climatology_column, _span(np.concatenate(rr)))
        mean = np.mean(per_seed[list(STATISTICS)].to_numpy(dtype=float), axis=0)
        results.append(dict(draw=draw, n_unique_source_sites=len(set(indices)), **dict(zip(STATISTICS, mean))))
    return pd.DataFrame(results, columns=columns)

def evaluate(frame, *, prediction_columns, data_version, pixel_ids_stable, baseline_columns=(), climatology_column=None, shared_columns=(), reference_column='reference', qc_column='qc', oos_columns=('oos',), bootstrap_draws=BOOTSTRAP_DRAWS, bootstrap_seed=BOOTSTRAP_SEED):
    """All §3 point stats, per-seed ranges and site-cluster percentile CIs.

    Exactly one trait/scope/model per call. Caller supplies the data_version
    and pixel-ID semantics explicitly. Missing climatology returns NaN with
    an explicit missing-input record. The returned tables may be saved by a
    later authorised phase; this function never writes or reads files.
    """
    if not isinstance(data_version, str) or not data_version:
        raise ValueError('data_version must be explicit')
    if 'data_version' in frame and (not frame.data_version.eq(data_version).all()):
        raise ValueError('data_version parameter conflicts with frame metadata')
    if climatology_column is not None and climatology_column not in baseline_columns:
        raise ValueError('climatology must be a supplied baseline on the shared mask')
    prepared = prepare_samples(frame, prediction_columns=prediction_columns, baseline_columns=baseline_columns, shared_columns=shared_columns, reference_column=reference_column, qc_column=qc_column, oos_columns=oos_columns)
    scenes = scene_statistics(prepared)
    pairs = make_delta_pairs(prepared, pixel_ids_stable=pixel_ids_stable)
    reference_span = _span(prepared['data'][reference_column])
    per_seed = _per_seed(scenes, pairs, prepared['predictions'], climatology_column, reference_span)
    per_seed['data_version'] = data_version
    bootstrap = site_cluster_bootstrap(prepared, scenes, pairs, climatology_column=climatology_column, draws=bootstrap_draws, random_seed=bootstrap_seed)
    summary = []
    for metric in STATISTICS:
        values = per_seed[metric].to_numpy(dtype=float)
        defined = np.isfinite(values)
        mean = float(values.mean()) if defined.all() else np.nan
        minimum = float(values.min()) if defined.all() else np.nan
        maximum = float(values.max()) if defined.all() else np.nan
        draws = bootstrap[metric].to_numpy(dtype=float)
        draws = draws[np.isfinite(draws)]
        low, high = np.quantile(draws, [0.025, 0.975]) if len(draws) else [np.nan, np.nan]
        summary.append(dict(data_version=data_version, metric=metric, mean=mean, seed_min=minimum, seed_max=maximum, n_seeds=len(values), defined_seeds=int(defined.sum()), ci_low=low, ci_high=high, bootstrap_draws=bootstrap_draws, defined_bootstrap_draws=len(draws), bootstrap_seed=bootstrap_seed, n_sites=prepared['n_sites'], n_scenes=len(scenes), n_pixels=len(prepared['data']), n_pairs=int(per_seed.n_pairs.iloc[0]), n_sign_pairs=int(per_seed.n_sign_pairs.iloc[0])))
    for output in [prepared['data'], scenes, pairs, bootstrap]:
        output['data_version'] = data_version
    return dict(prepared=prepared, scenes=scenes, pairs=pairs, per_seed=per_seed, summary=pd.DataFrame(summary), bootstrap=bootstrap, missing_inputs=[] if climatology_column is not None else ['DATA MISSING: site-season climatology baseline'], provenance=dict(data_version=data_version, shared_mask_columns=prepared['mask_columns'], pixel_ids_stable=pixel_ids_stable, variance_ddof=0, minimum_oos_scenes_per_site=2, scene_weight='equal', sign_dead_zone_reference='pooled shared-valid reference pixels in eligible sites; re-estimated per bootstrap draw', undefined_seed_policy='strict arithmetic mean; undefined propagates', undefined_bootstrap_policy='finite draw means only; defined_bootstrap_draws reported'))
