#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER_ROOT = Path(__file__).resolve().parents[1]
ECOSPECTRA_ROOT = Path("/home/chaonan.ji/data/projects/ecospectra")

OUT_TABLE_DIR = PAPER_ROOT / "tables" / "event_aligned_trait_flux_composite"
OUT_FIG_DIR = PAPER_ROOT / "figures" / "generated" / "event_aligned_trait_flux_composite"

UNIFIED_2022_TABLE = (
    PAPER_ROOT
    / "tables"
    / "fig3_all_sites_cap2000_flux_traits"
    / "all_sites_unified_biweekly_analysis_2022.csv"
)
FLUXNET_MAPPING = PAPER_ROOT / "tables" / "fig3_all_sites_cap2000_flux_traits" / "site_fluxnet_mapping.csv"
S2_ROOT = ECOSPECTRA_ROOT / "input_data" / "s2" / "region"
ERA5_ROOT = ECOSPECTRA_ROOT / "input_data" / "era5_land_point"
TRAIT_SUMMARY_GLOB = "tables/**/*trait_predictions*summary*.csv"

DEFAULT_EVENT_YEARS = (2019, 2020, 2022)
DEFAULT_TAU_MIN = -6
DEFAULT_TAU_MAX = 18
DEFAULT_BOOTSTRAP_REPS = 10000
DEFAULT_SEED = 20260601
DEFAULT_RECOVERY_BAND = 0.5
DEFAULT_RECOVERY_PERSIST_BINS = 2

FOREST_SCOPE_TYPES = {
    "Evergreen forest",
    "Deciduous forest",
    "Mixed forest",
    "Tree cover",
    "Open/savanna",
}

PROCESS_DOC = """# Event-aligned trait-flux composite process

This document is the preregistered process for the superposed-epoch composite
that aligns drought/heat events across forest flux sites. The figure target is
not a larger-n version of the 2022 flux scatter. It is a separate, event-level
time-alignment check: greenness, reconstructed chlorophyll, and measured GPP
are deseasonalized, aligned to driver-defined event onset, and composited with
site-clustered uncertainty.

## Phase 0: Figure target

Default conclusion tested: after drought/heat onset, NDVI returns toward its
seasonal expectation earlier than reconstructed chlorophyll and measured GPP.
The evidentiary claim is temporal corroboration of a chlorophyll legacy, not
pixel-level inflation of the flux sample size.

## Phase 1: Data assembly

Scope: forest PFT sites with eddy-covariance flux, usable Sentinel-2 dense
reflectance, reconstructed traits, and driver coverage over the target event
years. Default event years are 2019, 2020, and 2022.

Inputs per site:

- Sentinel-2 footprint-matched reflectance/index series, preferably 2017/2018
  through 2023.
- Reconstructed traits: chlorophyll primary; LAI, N, LMA, and EWT optional.
- Gap-filled GPP, using GPP_NT_VUT_REF as the default and GPP_DT as a
  sensitivity where available.
- ERA5-Land or flux-met drivers: root-zone/upper-layer SWC, VPD, and Tmax.

Spatial support default: tower footprint pixels, or a fixed tower-near radius
when footprints are unavailable. The full 30 km tile is reserved for the
pixel-scale decoupling claim, not for flux corroboration.

## Phase 2: Deseasonalization

For each site and variable, build a day-of-year climatology from baseline years
while excluding target event years. Default center is a robust median; default
scale is 1.4826 * MAD with a small positive floor. Smooth the climatology with a
harmonic fit of order 2, with rolling-DOY/LOESS as a sensitivity.

Default anomaly:

```text
z_V(t) = (V(t) - mu_V(DOY[t])) / sigma_V(DOY[t])
```

All plotted variables, including drivers and GPP, must be converted to z
anomalies before event alignment.

## Phase 3: Driver-based event detection

Events are detected only from climate drivers. NDVI, reconstructed traits, and
flux outcomes are never used to choose events.

Default thresholds:

- Drought: z_SWC <= -1.0 for at least 28 consecutive days in the growing season.
- Heat: z_VPD >= +1.5 for at least 14 consecutive days, or Tmax above its 90th
  percentile as a sensitivity.
- Compound event: overlapping drought and heat spells.

Each spell records onset, peak, type, and severity. Default severity is
cumulative positive VPD anomaly during the spell; min z_SWC is a sensitivity.
DoA gate: exclude or flag site-event periods whose reconstructed traits fall
outside the hyperspectral-anchored domain of applicability.

Thresholds are preregistered and run as sensitivity settings:
z_SWC <= -1.0, -1.28, -1.5; z_VPD >= +1.5 and +2.0.

## Phase 4: Event-relative extraction

Default event grid is monthly tau = -6 ... +18 months relative to event onset.
For each site-event-variable, resample deseasonalized z values to this grid.
Mask bins with insufficient observations. Default minimum coverage is at least
one valid biweekly optical observation per monthly bin for S2 variables, and at
least 50 percent daily coverage for flux/driver variables.

## Phase 5: Composite and uncertainty

Events are nested within site. The primary uncertainty method is cluster
bootstrap by site: resample sites with replacement, carrying all events for each
selected site. Report both n_sites and n_site_events.

Mixed-model sensitivity:

```text
z_V ~ ns(tau, df=4) + (1 | site) + (1 | site:event)
```

## Phase 6: Legacy statistics

Recovery time for variable V is the first tau where the composite remains within
|z| < 0.5 for at least two consecutive bins. Primary legacy offset is the
chlorophyll and GPP residual anomaly at the NDVI recovery time.

Secondary statistics:

- Cross-correlation or concordance between chlorophyll and GPP composite curves.
- Severity scaling: per-event legacy_offset ~ severity + (1 | site).

All intervals use site-clustered bootstrap or mixed-model uncertainty.

## Phase 7: Stratification

Default main text is pooled forest events. Stratify by event type
(drought/heat/compound) and biome (Mediterranean/temperate) only when each
stratum has enough sites for a readable interval. Otherwise, stratified panels
belong in the Supplement.

Mechanistic consistency checks are optional: EWT with ET, chlorophyll with GPP.

## Phase 8: Figure layout

Top panel: driver anomalies, SWC and/or VPD, with event onset marked at tau = 0.

Main panel: NDVI, reconstructed chlorophyll, and GPP z-anomaly composites with
site-clustered confidence bands. Mark the zero line, onset, NDVI recovery
crossing, and chlorophyll/GPP legacy offsets at that crossing.

Figure annotation must include n_sites, n_site_events, event-type composition,
event thresholds, recovery band, and bootstrap type.

## Phase 9: Reporting rules

Always state that flux is site-level. Pixel data can show the decoupling
distribution, but cannot increase the effective flux n. Event detection is
driver-based and non-circular. If the composite legacy is weak or the confidence
interval crosses zero, report it as suggestive rather than consistent.

## Pseudocode skeleton

```python
for site in sites:
    for V in [NDVI, chl, LAI, N, LMA, EWT, GPP, SWC, VPD]:
        mu, sigma = robust_doy_climatology(
            V,
            baseline_years,
            smoother=\"harmonic\",
            exclude=event_years,
        )
        z[V] = (V - mu[doy]) / sigma[doy]

events = []
for site in sites:
    spells = detect_spells(
        z_swc_threshold=-1.0,
        drought_min_days=28,
        z_vpd_threshold=1.5,
        heat_min_days=14,
        season=\"growing\",
    )
    for spell in spells:
        if doa_mask(site, spell.period) == \"in_domain\":
            events.append(spell)

tau = monthly_grid(-6, +18)
for event in events:
    for V in variables:
        A[V][event] = resample_to_tau(z[V], around=event.onset, grid=tau)

for V in variables:
    composite[V], ci[V] = cluster_bootstrap_by_site(A[V], B=10000)

rt = {V: first_stable_tau(abs(composite[V]) < 0.5, persist=2)
      for V in [\"NDVI\", \"Chlorophyll\", \"GPP\"]}
legacy_offset = {
    \"Chlorophyll\": composite[\"Chlorophyll\"][rt[\"NDVI\"]],
    \"GPP\": composite[\"GPP\"][rt[\"NDVI\"]],
}
```
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit inputs and, when a prepared event-relative z-anomaly table is "
            "available, plot a driver-defined event-aligned trait/flux composite."
        )
    )
    parser.add_argument("--output-table-dir", type=Path, default=OUT_TABLE_DIR)
    parser.add_argument("--output-fig-dir", type=Path, default=OUT_FIG_DIR)
    parser.add_argument("--event-relative-table", type=Path, default=None)
    parser.add_argument("--make-figure", action="store_true")
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tau-min", type=int, default=DEFAULT_TAU_MIN)
    parser.add_argument("--tau-max", type=int, default=DEFAULT_TAU_MAX)
    parser.add_argument("--event-years", type=int, nargs="+", default=list(DEFAULT_EVENT_YEARS))
    parser.add_argument("--write-process", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def forest_type_label(row: pd.Series) -> str:
    igbp = str(row.get("igbp", "")).strip()
    pft = str(row.get("pft_group", "")).strip()
    wc = str(row.get("worldcover_class", "")).strip()
    value = igbp if igbp and igbp.lower() != "nan" else pft
    if not value or value.lower() == "nan":
        value = wc
    if value in {"ENF", "EBF"}:
        return "Evergreen forest"
    if value in {"DBF", "DNF"}:
        return "Deciduous forest"
    if value == "MF":
        return "Mixed forest"
    if value == "Tree cover":
        return "Tree cover"
    if value in {"OSH", "SAV"}:
        return "Open/savanna"
    if value in {"GRA", "Grassland"}:
        return "Grassland"
    if value in {"CRO", "Cropland"}:
        return "Cropland"
    if value in {"WET", "Wetland"}:
        return "Wetland"
    return "Unclassified"


def load_site_metadata(unified_path: Path, mapping_path: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    if unified_path.exists():
        unified = pd.read_csv(unified_path)
        cols = [
            c
            for c in [
                "site_id",
                "site_name",
                "fluxnet_site_id",
                "igbp",
                "worldcover_class",
                "pft_group",
            ]
            if c in unified.columns
        ]
        if cols:
            parts.append(unified[cols].drop_duplicates("site_id"))
    if mapping_path.exists():
        mapping = pd.read_csv(mapping_path)
        cols = [
            c
            for c in [
                "site_id",
                "site_name",
                "fluxnet_site_id",
                "has_2022",
                "first_year",
                "last_year",
                "fluxnet_product_name",
            ]
            if c in mapping.columns
        ]
        if cols:
            parts.append(mapping[cols].drop_duplicates("site_id"))
    if not parts:
        return pd.DataFrame(columns=["site_id"])

    meta = parts[0].copy()
    for part in parts[1:]:
        meta = meta.merge(part, on="site_id", how="outer", suffixes=("", "_map"))
        for col in ["site_name", "fluxnet_site_id"]:
            alt = f"{col}_map"
            if alt in meta.columns:
                meta[col] = meta[col].where(meta[col].notna() & (meta[col].astype(str) != ""), meta[alt])
                meta = meta.drop(columns=[alt])

    meta["site_id"] = pd.to_numeric(meta["site_id"], errors="coerce").astype("Int64")
    meta = meta[meta["site_id"].notna()].copy()
    meta["site_id"] = meta["site_id"].astype(int)
    for col in ["igbp", "pft_group", "worldcover_class"]:
        if col not in meta.columns:
            meta[col] = np.nan
    meta["forest_type"] = meta.apply(forest_type_label, axis=1)
    meta["forest_scope"] = meta["forest_type"].isin(FOREST_SCOPE_TYPES)
    return meta


def _years_from_zarr_names(root: Path, pattern: str) -> dict[int, set[int]]:
    years: dict[int, set[int]] = {}
    if not root.exists():
        return years
    regex = re.compile(pattern)
    for path in root.glob("site_*/*.zarr"):
        match = regex.search(path.name)
        if not match:
            continue
        site_id = int(match.group("site"))
        year = int(match.group("year"))
        years.setdefault(site_id, set()).add(year)
    return years


def collect_s2_years(root: Path = S2_ROOT) -> dict[int, set[int]]:
    return _years_from_zarr_names(root, r"site_(?P<site>\d+)_s2_region_(?P<year>\d{4})\.zarr")


def collect_era5_years(root: Path = ERA5_ROOT) -> dict[int, set[int]]:
    return _years_from_zarr_names(root, r"site_(?P<site>\d+)_yr(?P<year>\d{4})\.zarr")


def collect_trait_years() -> tuple[dict[int, set[int]], list[str]]:
    years: dict[int, set[int]] = {}
    sources: list[str] = []
    for path in sorted(PAPER_ROOT.glob(TRAIT_SUMMARY_GLOB)):
        try:
            head = pd.read_csv(path, nrows=3)
        except Exception:
            continue
        if "site_id" not in head.columns or "date" not in head.columns:
            continue
        try:
            df = pd.read_csv(path, usecols=["site_id", "date"])
        except Exception:
            continue
        dates = pd.to_datetime(df["date"], errors="coerce")
        df = df.assign(year=dates.dt.year)
        df = df.dropna(subset=["site_id", "year"])
        if df.empty:
            continue
        sources.append(str(path))
        for site_id, sub in df.groupby(pd.to_numeric(df["site_id"], errors="coerce").astype("Int64")):
            if pd.isna(site_id):
                continue
            years.setdefault(int(site_id), set()).update(int(y) for y in sub["year"].dropna().unique())
    return years, sources


def flux_year_set(row: pd.Series) -> set[int]:
    first = pd.to_numeric(row.get("first_year"), errors="coerce")
    last = pd.to_numeric(row.get("last_year"), errors="coerce")
    if np.isfinite(first) and np.isfinite(last):
        return set(range(int(first), int(last) + 1))
    return set()


def compact_years(years: Iterable[int]) -> str:
    vals = sorted({int(y) for y in years})
    if not vals:
        return ""
    runs: list[str] = []
    start = prev = vals[0]
    for year in vals[1:]:
        if year == prev + 1:
            prev = year
            continue
        runs.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = year
    runs.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(runs)


def missing_years(available: set[int], required: Iterable[int]) -> list[int]:
    return [int(y) for y in required if int(y) not in available]


def build_data_availability_audit(event_years: Iterable[int]) -> tuple[pd.DataFrame, dict[str, object]]:
    event_years = tuple(int(y) for y in event_years)
    meta = load_site_metadata(UNIFIED_2022_TABLE, FLUXNET_MAPPING)
    s2_years = collect_s2_years()
    era5_years = collect_era5_years()
    trait_years, trait_sources = collect_trait_years()

    site_ids = set(meta["site_id"].dropna().astype(int).tolist())
    site_ids |= set(s2_years)
    site_ids |= set(era5_years)
    site_ids |= set(trait_years)

    rows: list[dict[str, object]] = []
    for site_id in sorted(site_ids):
        m = meta.loc[meta["site_id"] == site_id]
        rec = m.iloc[0] if not m.empty else pd.Series(dtype=object)
        flux_years = flux_year_set(rec)
        s2 = s2_years.get(site_id, set())
        era5 = era5_years.get(site_id, set())
        trait = trait_years.get(site_id, set())
        forest_scope = bool(rec.get("forest_scope", False))
        has_flux = bool(str(rec.get("fluxnet_site_id", "")).strip()) and bool(flux_years)

        blockers: list[str] = []
        if not forest_scope:
            blockers.append("outside_forest_scope")
        if not has_flux:
            blockers.append("missing_flux_record")
        for name, values in [("s2", s2), ("era5_driver", era5), ("trait_reconstruction", trait), ("flux", flux_years)]:
            miss = missing_years(values, event_years)
            if miss:
                blockers.append(f"missing_{name}_event_years:{compact_years(miss)}")
        if len([y for y in trait if y not in event_years]) < 4:
            blockers.append("trait_climatology_baseline_lt4_years")
        if len([y for y in era5 if y not in event_years]) < 4:
            blockers.append("driver_climatology_baseline_lt4_years")
        if len([y for y in s2 if y not in event_years]) < 4:
            blockers.append("s2_climatology_baseline_lt4_years")

        rows.append(
            {
                "site_id": site_id,
                "site_name": rec.get("site_name", ""),
                "fluxnet_site_id": rec.get("fluxnet_site_id", ""),
                "forest_type": rec.get("forest_type", "Unclassified"),
                "forest_scope": forest_scope,
                "has_flux_record": has_flux,
                "s2_years": compact_years(s2),
                "era5_driver_years": compact_years(era5),
                "trait_reconstruction_years": compact_years(trait),
                "flux_years": compact_years(flux_years),
                "missing_s2_event_years": compact_years(missing_years(s2, event_years)),
                "missing_era5_event_years": compact_years(missing_years(era5, event_years)),
                "missing_trait_event_years": compact_years(missing_years(trait, event_years)),
                "missing_flux_event_years": compact_years(missing_years(flux_years, event_years)),
                "baseline_trait_years_excluding_events": len([y for y in trait if y not in event_years]),
                "baseline_s2_years_excluding_events": len([y for y in s2 if y not in event_years]),
                "baseline_era5_years_excluding_events": len([y for y in era5 if y not in event_years]),
                "ready_for_preregistered_composite": len(blockers) == 0,
                "blockers": ";".join(blockers),
            }
        )

    audit = pd.DataFrame(rows)
    forest_flux = audit[(audit["forest_scope"]) & (audit["has_flux_record"])]
    summary = {
        "event_years": list(event_years),
        "n_sites_in_metadata_or_inputs": int(len(audit)),
        "n_forest_sites": int(audit["forest_scope"].sum()),
        "n_forest_flux_sites": int(len(forest_flux)),
        "n_ready_for_preregistered_composite": int(audit["ready_for_preregistered_composite"].sum()),
        "trait_summary_sources": trait_sources,
        "status": "ready" if audit["ready_for_preregistered_composite"].any() else "blocked_missing_multiyear_inputs",
    }
    return audit, summary


def harmonic_climatology(
    data: pd.DataFrame,
    value_col: str,
    date_col: str = "date",
    exclude_years: Iterable[int] = (),
    order: int = 2,
    scale_floor: float = 1e-6,
) -> pd.DataFrame:
    """Return smoothed robust DOY center/scale for one site-variable series."""

    df = data[[date_col, value_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col])
    if exclude_years:
        df = df[~df[date_col].dt.year.isin({int(y) for y in exclude_years})]
    if df.empty:
        raise ValueError("Cannot build climatology from an empty baseline.")

    df["doy"] = df[date_col].dt.dayofyear.clip(upper=365)
    grouped = df.groupby("doy")[value_col]
    center = grouped.median()
    mad = grouped.apply(lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))))
    scale = (1.4826 * mad).clip(lower=scale_floor)

    doy = np.arange(1, 366)
    center_grid = center.reindex(doy).interpolate(limit_direction="both").to_numpy(dtype=float)
    scale_grid = scale.reindex(doy).interpolate(limit_direction="both").to_numpy(dtype=float)

    x = harmonic_design(doy, order=order)
    beta_center, *_ = np.linalg.lstsq(x, center_grid, rcond=None)
    beta_scale, *_ = np.linalg.lstsq(x, np.log(np.clip(scale_grid, scale_floor, None)), rcond=None)
    return pd.DataFrame(
        {
            "doy": doy,
            "mu": x @ beta_center,
            "sigma": np.exp(x @ beta_scale).clip(min=scale_floor),
        }
    )


def harmonic_design(doy: np.ndarray, order: int = 2) -> np.ndarray:
    cols = [np.ones_like(doy, dtype=float)]
    angle = 2.0 * np.pi * doy.astype(float) / 365.0
    for k in range(1, order + 1):
        cols.append(np.sin(k * angle))
        cols.append(np.cos(k * angle))
    return np.column_stack(cols)


def standardize_by_climatology(
    data: pd.DataFrame,
    climatology: pd.DataFrame,
    value_col: str,
    date_col: str = "date",
) -> pd.Series:
    df = data[[date_col, value_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    doy = df[date_col].dt.dayofyear.clip(upper=365)
    clim = climatology.set_index("doy")
    mu = doy.map(clim["mu"])
    sigma = doy.map(clim["sigma"])
    return (pd.to_numeric(df[value_col], errors="coerce") - mu.to_numpy()) / sigma.to_numpy()


def detect_driver_spells(
    daily: pd.DataFrame,
    date_col: str = "date",
    z_swc_col: str = "z_SWC",
    z_vpd_col: str = "z_VPD",
    drought_threshold: float = -1.0,
    drought_min_days: int = 28,
    heat_threshold: float = 1.5,
    heat_min_days: int = 14,
    growing_months: tuple[int, ...] = (4, 5, 6, 7, 8, 9),
) -> pd.DataFrame:
    """Detect non-circular climate events from driver anomalies only."""

    df = daily.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    df = df[df[date_col].dt.month.isin(growing_months)].copy()
    if df.empty:
        return pd.DataFrame(columns=["onset", "end", "type", "duration_days", "severity"])

    drought = pd.to_numeric(df.get(z_swc_col), errors="coerce") <= drought_threshold
    heat = pd.to_numeric(df.get(z_vpd_col), errors="coerce") >= heat_threshold
    drought_spells = _boolean_spells(df[date_col], drought.to_numpy(), drought_min_days, "drought")
    heat_spells = _boolean_spells(df[date_col], heat.to_numpy(), heat_min_days, "heat")

    events = drought_spells + heat_spells
    if not events:
        return pd.DataFrame(columns=["onset", "end", "type", "duration_days", "severity"])

    rows = []
    for event in events:
        sub = df[(df[date_col] >= event["onset"]) & (df[date_col] <= event["end"])]
        overlap_heat = bool((pd.to_numeric(sub.get(z_vpd_col), errors="coerce") >= heat_threshold).any())
        overlap_drought = bool((pd.to_numeric(sub.get(z_swc_col), errors="coerce") <= drought_threshold).any())
        event_type = "compound" if overlap_heat and overlap_drought else event["type"]
        z_vpd = pd.to_numeric(sub.get(z_vpd_col), errors="coerce")
        severity = float(np.nansum(np.clip(z_vpd.to_numpy(dtype=float), 0, None)))
        rows.append({**event, "type": event_type, "severity": severity})
    out = pd.DataFrame(rows).drop_duplicates(subset=["onset", "end", "type"])
    return out.sort_values(["onset", "end"]).reset_index(drop=True)


def _boolean_spells(dates: pd.Series, mask: np.ndarray, min_days: int, event_type: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    dates = pd.to_datetime(dates).reset_index(drop=True)
    mask = np.asarray(mask, dtype=bool)
    start_idx: int | None = None
    for i, value in enumerate(mask.tolist() + [False]):
        if value and start_idx is None:
            start_idx = i
        if (not value) and start_idx is not None:
            end_idx = i - 1
            duration = int((dates.iloc[end_idx] - dates.iloc[start_idx]).days) + 1
            if duration >= min_days:
                rows.append(
                    {
                        "onset": dates.iloc[start_idx],
                        "end": dates.iloc[end_idx],
                        "type": event_type,
                        "duration_days": duration,
                    }
                )
            start_idx = None
    return rows


def load_event_relative_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"site_id", "event_id", "tau_month", "variable", "z"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Event-relative table missing required columns: {missing}")
    df["site_id"] = pd.to_numeric(df["site_id"], errors="coerce").astype("Int64")
    df["tau_month"] = pd.to_numeric(df["tau_month"], errors="coerce")
    df["z"] = pd.to_numeric(df["z"], errors="coerce")
    df = df.dropna(subset=["site_id", "tau_month", "variable", "z"]).copy()
    df["site_id"] = df["site_id"].astype(int)
    if "coverage_ok" in df.columns:
        df = df[df["coverage_ok"].astype(bool)].copy()
    return df


def cluster_bootstrap_composite(
    df: pd.DataFrame,
    variables: Iterable[str],
    tau: np.ndarray,
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    n_pairs_by_var: dict[str, int] = {}
    n_sites_by_var: dict[str, int] = {}

    for variable in variables:
        sub = df[df["variable"] == variable].copy()
        if sub.empty:
            continue
        pair = (
            sub.groupby(["site_id", "event_id", "tau_month"], as_index=False)["z"]
            .mean()
            .dropna(subset=["z"])
        )
        sites = np.array(sorted(pair["site_id"].dropna().unique()), dtype=int)
        n_sites_by_var[variable] = int(len(sites))
        n_pairs_by_var[variable] = int(pair[["site_id", "event_id"]].drop_duplicates().shape[0])
        for t in tau:
            vals = pair.loc[np.isclose(pair["tau_month"], t), "z"].to_numpy(dtype=float)
            center = float(np.nanmean(vals)) if len(vals) else math.nan
            boot_vals: list[float] = []
            for _ in range(reps):
                sampled_sites = rng.choice(sites, size=len(sites), replace=True)
                sampled_parts = []
                for site in sampled_sites:
                    sampled_parts.append(pair[(pair["site_id"] == site) & (np.isclose(pair["tau_month"], t))])
                sampled = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame()
                boot_vals.append(float(np.nanmean(sampled["z"])) if not sampled.empty else math.nan)
            ci_low, ci_high = np.nanpercentile(boot_vals, [2.5, 97.5])
            rows.append(
                {
                    "variable": variable,
                    "tau_month": float(t),
                    "z_mean": center,
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "n_site_events_at_tau": int(pair.loc[np.isclose(pair["tau_month"], t), ["site_id", "event_id"]].drop_duplicates().shape[0]),
                }
            )
    summary = {
        "n_sites": int(df["site_id"].dropna().nunique()),
        "n_sites_by_variable": n_sites_by_var,
        "n_site_events_by_variable": n_pairs_by_var,
        "bootstrap_reps": int(reps),
        "bootstrap_cluster": "site",
        "seed": int(seed),
    }
    return pd.DataFrame(rows), summary


def first_stable_recovery_tau(
    composite: pd.DataFrame,
    variable: str,
    band: float = DEFAULT_RECOVERY_BAND,
    persist_bins: int = DEFAULT_RECOVERY_PERSIST_BINS,
) -> float:
    sub = composite[composite["variable"] == variable].sort_values("tau_month")
    if sub.empty:
        return math.nan
    ok = np.abs(sub["z_mean"].to_numpy(dtype=float)) < band
    tau = sub["tau_month"].to_numpy(dtype=float)
    for i in range(0, len(ok) - persist_bins + 1):
        if np.all(ok[i : i + persist_bins]):
            return float(tau[i])
    return math.nan


def legacy_metrics(composite: pd.DataFrame) -> dict[str, object]:
    rt = {
        variable: first_stable_recovery_tau(composite, variable)
        for variable in ["NDVI", "Chlorophyll", "GPP"]
    }
    offsets: dict[str, float] = {}
    ndvi_rt = rt.get("NDVI", math.nan)
    if np.isfinite(ndvi_rt):
        for variable in ["Chlorophyll", "GPP"]:
            sub = composite[(composite["variable"] == variable) & (np.isclose(composite["tau_month"], ndvi_rt))]
            offsets[variable] = float(sub["z_mean"].iloc[0]) if not sub.empty else math.nan
    return {
        "recovery_band_abs_z": DEFAULT_RECOVERY_BAND,
        "recovery_persist_bins": DEFAULT_RECOVERY_PERSIST_BINS,
        "recovery_time_months": rt,
        "legacy_offset_at_ndvi_recovery": offsets,
    }


def plot_composite(
    composite: pd.DataFrame,
    out_dir: Path,
    summary: dict[str, object],
    metrics: dict[str, object],
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.1, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.55], "hspace": 0.10},
        constrained_layout=False,
    )

    styles = {
        "SWC": {"color": "#8c6d31", "label": "Soil water", "linestyle": "-"},
        "VPD": {"color": "#c44e52", "label": "VPD", "linestyle": "-"},
        "NDVI": {"color": "#2f2f2f", "label": "NDVI", "linestyle": "--"},
        "Chlorophyll": {"color": "#2ca25f", "label": "Chlorophyll", "linestyle": "-"},
        "GPP": {"color": "#2b6cb0", "label": "GPP", "linestyle": "-"},
    }
    panels = [(axes[0], ["SWC", "VPD"]), (axes[1], ["NDVI", "Chlorophyll", "GPP"])]
    for ax, variables in panels:
        ax.axhline(0, color="#7a7a7a", lw=0.7, zorder=0)
        ax.axvline(0, color="#4a4a4a", lw=0.9, zorder=0)
        for variable in variables:
            sub = composite[composite["variable"] == variable].sort_values("tau_month")
            if sub.empty:
                continue
            style = styles[variable]
            x = sub["tau_month"].to_numpy(dtype=float)
            y = sub["z_mean"].to_numpy(dtype=float)
            lo = sub["ci_low"].to_numpy(dtype=float)
            hi = sub["ci_high"].to_numpy(dtype=float)
            ax.fill_between(x, lo, hi, color=style["color"], alpha=0.16, linewidth=0)
            ax.plot(x, y, color=style["color"], lw=1.8, linestyle=style["linestyle"], label=style["label"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)
        ax.set_ylabel("Standardized anomaly (z)", fontsize=8.5)

    ndvi_rt = metrics.get("recovery_time_months", {}).get("NDVI", math.nan)
    if np.isfinite(ndvi_rt):
        axes[1].axvline(float(ndvi_rt), color="#2f2f2f", lw=0.9, linestyle=":")
        axes[1].text(
            float(ndvi_rt),
            0.97,
            "NDVI recovery",
            transform=axes[1].get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="right",
            fontsize=7.5,
            color="#2f2f2f",
        )

    axes[0].legend(frameon=False, fontsize=8, loc="upper right", ncol=2, handlelength=2.0, columnspacing=1.0)
    axes[1].legend(frameon=False, fontsize=8, loc="upper right", ncol=3, handlelength=2.0, columnspacing=1.0)
    axes[1].set_xlabel("Months since driver-defined event onset", fontsize=9)

    n_sites = summary.get("n_sites", "NA")
    n_pairs = summary.get("n_site_events_by_variable", {})
    gpp_pairs = n_pairs.get("GPP", "NA") if isinstance(n_pairs, dict) else "NA"
    axes[0].set_title(
        f"Driver-aligned forest events, site-clustered bootstrap (n_sites={n_sites}, GPP site-events={gpp_pairs})",
        fontsize=9.5,
        loc="left",
    )
    fig.align_ylabels(axes)

    outputs: list[str] = []
    for suffix in ["png", "pdf", "svg"]:
        path = out_dir / f"fig_event_aligned_trait_flux_composite.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def write_process_doc(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "event_aligned_composite_process.md"
    path.write_text(PROCESS_DOC + "\n")
    return path


def main() -> None:
    args = parse_args()
    args.output_table_dir.mkdir(parents=True, exist_ok=True)
    args.output_fig_dir.mkdir(parents=True, exist_ok=True)

    process_path = write_process_doc(args.output_table_dir) if args.write_process else None
    audit, audit_summary = build_data_availability_audit(args.event_years)
    audit_path = args.output_table_dir / "data_availability_audit.csv"
    audit.to_csv(audit_path, index=False)

    run_summary: dict[str, object] = {
        "process_doc": str(process_path) if process_path else "",
        "data_availability_audit": str(audit_path),
        "audit_summary": audit_summary,
        "figure_outputs": [],
        "event_relative_table": str(args.event_relative_table) if args.event_relative_table else "",
        "preregistered_defaults": {
            "event_years": list(args.event_years),
            "tau_months": [int(args.tau_min), int(args.tau_max)],
            "drought_threshold": "z_SWC <= -1.0 for >=28 days",
            "heat_threshold": "z_VPD >= +1.5 for >=14 days",
            "recovery_band_abs_z": DEFAULT_RECOVERY_BAND,
            "recovery_persist_bins": DEFAULT_RECOVERY_PERSIST_BINS,
            "bootstrap_cluster": "site",
            "bootstrap_reps": int(args.bootstrap_reps),
        },
    }

    if args.make_figure and args.event_relative_table is not None:
        event_df = load_event_relative_table(args.event_relative_table)
        tau = np.arange(int(args.tau_min), int(args.tau_max) + 1, dtype=float)
        variables = ["SWC", "VPD", "NDVI", "Chlorophyll", "GPP"]
        composite, comp_summary = cluster_bootstrap_composite(
            event_df,
            variables=variables,
            tau=tau,
            reps=int(args.bootstrap_reps),
            seed=int(args.seed),
        )
        composite_path = args.output_table_dir / "event_aligned_composite_summary.csv"
        composite.to_csv(composite_path, index=False)
        metrics = legacy_metrics(composite)
        outputs = plot_composite(composite, args.output_fig_dir, comp_summary, metrics)
        run_summary.update(
            {
                "composite_summary": str(composite_path),
                "composite_resampling": comp_summary,
                "legacy_metrics": metrics,
                "figure_outputs": outputs,
            }
        )
    elif args.make_figure:
        run_summary["figure_status"] = (
            "not_generated: provide --event-relative-table with site_id,event_id,tau_month,variable,z"
        )
    else:
        run_summary["figure_status"] = "not_requested"

    run_summary_path = args.output_table_dir / "run_summary.json"
    run_summary_path.write_text(json.dumps(run_summary, indent=2, allow_nan=True) + "\n")
    print(json.dumps(run_summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
