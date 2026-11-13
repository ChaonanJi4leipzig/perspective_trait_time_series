'prepared scene sufficient statistics;1000 site-cluster draws;temporal run.\nInputs: code/paths.py; outputs: explicit derivation work directory, never upstream.\nOriginal: upstream://V6/final_figures_20261113/scripts/hierarchy_fast.py.\nRun IDs: cap2048_v3_nested11_20261108; cap2048_spatial_gsingle_20261112; cap2048_production_nn_temporal_20261112.\nHeavy derivation was not rerun when packaging. Use --help before --run.'
from __future__ import annotations
import paths,sys
if __name__=="__main__":
 from make_runtime import cli
 cli(__doc__)
sys.path.insert(0,str(paths.MODEL_CODE))
from make_support import *
import make_metrics_core as mm

def wquantile_sorted(values, weights, qs):
    cumulative = np.cumsum(weights)
    n = int(cumulative[-1])
    positions = np.asarray(qs) * (n - 1)
    lo = np.floor(positions).astype(int)
    hi = np.ceil(positions).astype(int)
    a = values[np.searchsorted(cumulative, lo, side='right')]
    b = values[np.searchsorted(cumulative, hi, side='right')]
    return a + (b - a) * (positions - lo)

def ratio(a, b):
    return np.divide(a, b, out=np.full(np.broadcast(a, b).shape, np.nan), where=np.asarray(b) > 0)

def bootstrap(prepared, scenes, pairs, *, climatology_column=None, draws=1000, random_seed=20261111):
    data = prepared['data']
    sites = sorted(data.site_id.unique(), key=str)
    ns = len(sites)
    ids = {s: i for i, s in enumerate(sites)}
    sc = scenes.site_id.map(ids).to_numpy()
    pc = pairs.site_id.map(ids).to_numpy()
    dc = data.site_id.map(ids).to_numpy()
    nscene = np.bincount(sc, minlength=ns)
    ref = scenes.m_ref.to_numpy(float)
    M = np.bincount(sc, weights=ref, minlength=ns) / nscene
    a = ref - M[sc]
    for i in range(ns):
        take = sc == i
        if np.ptp(ref[take]) == 0:
            M[i] = ref[take][0]
            a[take] = 0.0
    within = scenes.v_ref.to_numpy(float)
    valid = pairs.included.to_numpy(bool)
    dref = pairs.delta_ref.to_numpy(float)
    order = np.argsort(data[prepared['reference']].to_numpy(float), kind='stable')
    raw = data[prepared['reference']].to_numpy(float)[order]
    rawcodes = dc[order]
    seedstats = []
    for column in prepared['predictions'].values():
        p = scenes['m::' + column].to_numpy(float)
        PM = np.bincount(sc, weights=p, minlength=ns) / nscene
        pa = p - PM[sc]
        for i in range(ns):
            take = sc == i
            if np.ptp(p[take]) == 0:
                PM[i] = p[take][0]
                pa[take] = 0.0
        seedstats.append((PM, pa, scenes['mse_within::' + column].to_numpy(float), pairs['delta::' + column].to_numpy(float)))
    if climatology_column:
        b = scenes['m::' + climatology_column].to_numpy(float)
        B = np.bincount(sc, weights=b, minlength=ns) / nscene
        ba = b - B[sc]
        for i in range(ns):
            take = sc == i
            if np.ptp(b[take]) == 0:
                ba[take] = 0.0
    rng = np.random.default_rng(random_seed)
    records = []
    for draw in range(draws):
        counts = np.bincount(rng.integers(0, ns, size=ns), minlength=ns)
        sw = counts[sc]
        n = float(sw.sum())
        pw = counts[pc] * valid
        mappedmean = float(np.dot(sw, M[sc]) / n)
        variances = np.array([np.dot(sw, (M[sc] - mappedmean) ** 2) / n, np.dot(sw, a * a) / n, np.dot(sw, within) / n])
        constant_site = np.ptp(M[counts > 0]) == 0
        if constant_site:
            variances[0] = 0.0
        shares = ratio(variances, variances.sum())
        span = np.diff(wquantile_sorted(raw, counts[rawcodes], [0.05, 0.95]))[0]
        dead = 0.05 * span
        use = pw > 0
        pr = dref[use]
        weights = pw[use]
        npair = weights.sum()
        active = np.abs(pr) >= dead
        means = []
        for PM, pa, mse, dp in seedstats:
            r2s = np.array([1 - ratio(np.dot(counts, (PM - M) ** 2), 0.0 if constant_site else np.dot(counts, (M - np.dot(counts, M) / ns) ** 2)), 1 - ratio(np.dot(sw, (pa - a) ** 2), np.dot(sw, a * a)), 1 - ratio(np.dot(sw, mse), np.dot(sw, within))], float)
            out = {}
            for j, l in enumerate(mm.LEVELS):
                out.update({f'r2_{l}': r2s[j], f'v_{l}': variances[j], f'variance_share_{l}': shares[j], f'recovered_share_{l}': 0.0 if shares[j] == 0 else shares[j] * max(r2s[j], 0) if np.isfinite(r2s[j]) else np.nan})
            pp = dp[use]
            if npair:
                ym = np.dot(weights, pr) / npair
                pm = np.dot(weights, pp) / npair
                vy = np.dot(weights, (pr - ym) ** 2) / npair
                vp = np.dot(weights, (pp - pm) ** 2) / npair
                cov = np.dot(weights, (pr - ym) * (pp - pm)) / npair
                if np.ptp(pr) == 0:
                    vy = 0.0
                    cov = 0.0
                if np.ptp(pp) == 0:
                    vp = 0.0
                    cov = 0.0
                out.update(delta_r=float(ratio(cov, np.sqrt(vy * vp))), delta_amplitude_ratio=float(np.sqrt(ratio(vp, vy))), delta_ols_slope=float(ratio(cov, vy)), delta_sign_agreement=float(ratio(np.dot(weights[active], np.sign(pp[active]) == np.sign(pr[active])), weights[active].sum())))
            else:
                out.update({k: np.nan for k in ['delta_r', 'delta_amplitude_ratio', 'delta_ols_slope', 'delta_sign_agreement']})
            out['climatology_date_r2'] = float(1 - ratio(np.dot(sw, (ba - a) ** 2), np.dot(sw, a * a))) if climatology_column else np.nan
            means.append([out[k] for k in mm.STATISTICS])
        records.append(dict(draw=draw, n_unique_source_sites=int(np.count_nonzero(counts)), **dict(zip(mm.STATISTICS, np.mean(means, axis=0)))))
    return pd.DataFrame(records)

def evaluate(f, cols, signals, *, draws=1000):
    result = mm.evaluate(f, prediction_columns=cols, data_version=VERSION, pixel_ids_stable=False, baseline_columns=['site_mean', 'site_season'], climatology_column='site_season', shared_columns=signals, bootstrap_draws=0)
    b = bootstrap(result['prepared'], result['scenes'], result['pairs'], climatology_column='site_season', draws=draws, random_seed=CONFIG['bootstrap_seed'])
    b['data_version'] = VERSION
    for i, r in result['summary'].iterrows():
        values = b[r.metric].dropna()
        low, high = values.quantile([0.025, 0.975]) if len(values) else [np.nan, np.nan]
        result['summary'].loc[i, ['ci_low', 'ci_high', 'bootstrap_draws', 'defined_bootstrap_draws']] = [low, high, draws, len(values)]
    result['bootstrap'] = b
    return result
