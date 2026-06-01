#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
import xarray as xr


PAPER_ROOT = Path(__file__).resolve().parents[1]
ECOSPECTRA_ROOT = Path("/home/chaonan.ji/data/projects/ecospectra")

ALL_SITES = ECOSPECTRA_ROOT / "input_data" / "fluxnet_sites.csv"
ENMAP_SITE_SUBSET = ECOSPECTRA_ROOT / "input_data" / "fluxnet_sites_4enmap.csv"
CURRENT_FILTERED = ECOSPECTRA_ROOT / "input_data" / "filtered_sites.csv"
FLUXNET_SHUTTLE = ECOSPECTRA_ROOT / "input_data" / "fluxnet_shuttle.csv"
ENMAP_SCENE_HITS = ECOSPECTRA_ROOT / "input_data" / "enmap" / "enmap_scene_site_hits.csv"
PRISMA_SCENE_HITS = ECOSPECTRA_ROOT / "input_data" / "prisma" / "prisma_scene_site_hits.csv"
ENMAP_TRAITS_DIR = ECOSPECTRA_ROOT / "input_data" / "enmap" / "traits_region"
PRISMA_TRAITS_DIR = ECOSPECTRA_ROOT / "input_data" / "prisma" / "traits_region"
PRISMA_QUERY_AOIS = ECOSPECTRA_ROOT / "input_data" / "prisma" / "manifests" / "prisma_query_aois.csv"
PRISMA_QUERY_TRACK = ECOSPECTRA_ROOT / "input_data" / "prisma" / "manifests" / "prisma_query_aois_track.csv"

OUT_DIR = PAPER_ROOT / "tables" / "site_inclusion_audit_prisma"

TARGET_YEARS = [2022, 2023]
FLUXNET_MAX_DISTANCE_KM = 5.0
FLUXNET_SOURCE_VARS = ["GPP_NT_VUT_REF", "GPP_DT_VUT_REF", "NEE_VUT_REF", "RECO_NT_VUT_REF"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit site inclusion/exclusion with current EnMAP, PRISMA, and GPP coverage.")
    parser.add_argument("--all-sites", type=Path, default=ALL_SITES)
    parser.add_argument("--enmap-site-subset", type=Path, default=ENMAP_SITE_SUBSET)
    parser.add_argument("--current-filtered-sites", type=Path, default=CURRENT_FILTERED)
    parser.add_argument("--fluxnet-shuttle", type=Path, default=FLUXNET_SHUTTLE)
    parser.add_argument("--enmap-scene-hits", type=Path, default=ENMAP_SCENE_HITS)
    parser.add_argument("--prisma-scene-hits", type=Path, default=PRISMA_SCENE_HITS)
    parser.add_argument("--enmap-traits-dir", type=Path, default=ENMAP_TRAITS_DIR)
    parser.add_argument("--prisma-traits-dir", type=Path, default=PRISMA_TRAITS_DIR)
    parser.add_argument("--prisma-query-aois", type=Path, default=PRISMA_QUERY_AOIS)
    parser.add_argument("--prisma-query-track", type=Path, default=PRISMA_QUERY_TRACK)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--target-years", type=int, nargs="+", default=TARGET_YEARS)
    parser.add_argument("--min-prisma-per-year", type=int, default=1)
    parser.add_argument("--fluxnet-max-distance-km", type=float, default=FLUXNET_MAX_DISTANCE_KM)
    return parser.parse_args()


def normalize_reason(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def reason_group(reason: str) -> str:
    if not reason:
        return "included_current_or_no_recorded_reason"
    if "overlap" in reason:
        return "overlap_excluded"
    if "no fluxnet" in reason or "no gpp" in reason:
        return "no_gpp_or_no_fluxnet_excluded"
    if "no hrl" in reason:
        return "no_hrl_excluded"
    if "no enmap" in reason:
        return "no_enmap_excluded"
    if "small part" in reason:
        return "weak_enmap_overlap_excluded"
    if "noisy" in reason:
        return "enmap_noisy_excluded"
    return "other_recorded_exclusion"


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(a))


def build_fluxnet_mapping(sites: pd.DataFrame, shuttle_path: Path, max_distance_km: float) -> pd.DataFrame:
    shuttle = pd.read_csv(shuttle_path)
    shuttle = shuttle.copy()
    shuttle["first_year"] = pd.to_numeric(shuttle["first_year"], errors="coerce")
    shuttle["last_year"] = pd.to_numeric(shuttle["last_year"], errors="coerce")
    lat2 = shuttle["location_lat"].astype(float).to_numpy()
    lon2 = shuttle["location_long"].astype(float).to_numpy()
    rows: list[dict[str, object]] = []
    for rec in sites.itertuples(index=False):
        site_id = int(rec.site_id)
        distances = haversine_km(float(rec.location_lat), float(rec.location_lon), lat2, lon2)
        candidates = shuttle.assign(distance_km=distances)
        candidates = candidates[candidates["distance_km"] <= float(max_distance_km)].copy()
        if candidates.empty:
            rows.append({"site_id": site_id, "fluxnet_site_id": "", "fluxnet_distance_km": math.nan})
            continue
        pick = candidates.sort_values(["distance_km", "last_year"], ascending=[True, False]).iloc[0]
        rows.append(
            {
                "site_id": site_id,
                "fluxnet_site_id": str(pick["site_id"]),
                "fluxnet_site_name": str(pick["site_name"]),
                "fluxnet_data_hub": str(pick["data_hub"]),
                "fluxnet_first_year": int(pick["first_year"]) if np.isfinite(pick["first_year"]) else math.nan,
                "fluxnet_last_year": int(pick["last_year"]) if np.isfinite(pick["last_year"]) else math.nan,
                "fluxnet_distance_km": float(pick["distance_km"]),
                "fluxnet_product_name": str(pick["fluxnet_product_name"]),
                "fluxnet_download_link": str(pick["download_link"]),
            }
        )
    return pd.DataFrame(rows)


def find_fluxnet_zip(site_code: str, product_name: str = "") -> Path | None:
    search_dirs = [
        PAPER_ROOT / "tables" / "fig3_all_sites_cap2000_flux_traits" / "fluxnet",
        ECOSPECTRA_ROOT / "input_data" / "fluxnet",
        ECOSPECTRA_ROOT / "input_data" / "0_others",
        ECOSPECTRA_ROOT / "input_data",
    ]
    patterns = []
    if product_name and product_name.lower() != "nan":
        patterns.append(product_name)
    if site_code and site_code.lower() != "nan":
        patterns.append(f"*{site_code}*FLUXNET*.zip")
    for root in search_dirs:
        if not root.exists():
            continue
        for pattern in patterns:
            candidates = sorted(root.glob(pattern))
            if candidates:
                return candidates[0]
    return None


def read_fluxnet_daily(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        daily = [n for n in names if n.lower().endswith(".csv") and "fluxmet_dd" in n.lower()]
        if not daily:
            daily = [
                n
                for n in names
                if n.lower().endswith(".csv") and ("_dd_" in n.lower() or "daily" in n.lower()) and "bif" not in n.lower()
            ]
        if not daily:
            return pd.DataFrame(columns=["date"])
        with zf.open(daily[0]) as handle:
            df = pd.read_csv(handle)
    timestamp_col = "TIMESTAMP" if "TIMESTAMP" in df.columns else "TIMESTAMP_START"
    if timestamp_col not in df.columns:
        return pd.DataFrame(columns=["date"])
    df["date"] = pd.to_datetime(df[timestamp_col].astype(str).str.slice(0, 8), format="%Y%m%d", errors="coerce")
    available = [v for v in FLUXNET_SOURCE_VARS if v in df.columns]
    if not available:
        return pd.DataFrame(columns=["date"])
    df[available] = df[available].replace(-9999, np.nan)
    return df.groupby("date", as_index=False)[available].mean(numeric_only=True)


def add_local_gpp_status(mapping: pd.DataFrame, target_years: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rec in mapping.itertuples(index=False):
        site_id = int(rec.site_id)
        site_code = str(getattr(rec, "fluxnet_site_id", ""))
        product_name = str(getattr(rec, "fluxnet_product_name", ""))
        zip_path = find_fluxnet_zip(site_code, product_name)
        row: dict[str, object] = {
            "site_id": site_id,
            "fluxnet_local_zip": str(zip_path) if zip_path else "",
            "gpp_local_zip_available": bool(zip_path),
        }
        for year in target_years:
            row[f"gpp_finite_days_{year}"] = 0
        if zip_path is None:
            rows.append(row)
            continue
        try:
            daily = read_fluxnet_daily(zip_path)
            gpp_col = "GPP_NT_VUT_REF" if "GPP_NT_VUT_REF" in daily.columns else "GPP_DT_VUT_REF" if "GPP_DT_VUT_REF" in daily.columns else ""
            row["gpp_local_column"] = gpp_col
            if gpp_col:
                for year in target_years:
                    sub = daily[pd.to_datetime(daily["date"]).dt.year == int(year)]
                    row[f"gpp_finite_days_{year}"] = int(np.isfinite(pd.to_numeric(sub[gpp_col], errors="coerce")).sum())
        except Exception as exc:
            row["gpp_read_error"] = str(exc)
        rows.append(row)
    out = pd.DataFrame(rows)
    year_cols = [f"gpp_finite_days_{year}" for year in target_years]
    out["gpp_local_finite_all_target_years"] = out[year_cols].ge(1).all(axis=1) if year_cols else False
    return out


def scene_year_counts(path: Path, date_col: str, id_col: str = "site_id") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["site_id"])
    df = pd.read_csv(path)
    if id_col not in df.columns or date_col not in df.columns:
        return pd.DataFrame(columns=["site_id"])
    df["site_id"] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    df["year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year
    df = df.dropna(subset=["site_id", "year"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["site_id"])
    return df.groupby(["site_id", "year"]).size().unstack(fill_value=0).reset_index()


def parse_years_from_product_ids(values: np.ndarray) -> list[int]:
    years: list[int] = []
    for value in values:
        match = re.search(r"PRS_L2D_STD_(\d{4})", str(value))
        if match:
            years.append(int(match.group(1)))
    return years


def parse_years_from_iso(values: np.ndarray) -> list[int]:
    years: list[int] = []
    parsed = pd.to_datetime(pd.Series(values).astype(str), errors="coerce")
    for value in parsed.dropna():
        years.append(int(value.year))
    return years


def zarr_trait_year_counts(root: Path, sensor: str, fallback_scene_counts: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pattern = f"site_*_{sensor}_traits_region_30km_30m.zarr"
    for path in sorted(root.glob(pattern)):
        match = re.search(r"site_(\d+)_", path.name)
        if not match:
            continue
        site_id = int(match.group(1))
        row: dict[str, object] = {"site_id": site_id, f"{sensor}_traits_zarr": str(path)}
        try:
            ds = xr.open_zarr(path, consolidated=False)
            row[f"{sensor}_traits_time_count"] = int(ds.sizes.get("time", 0))
            row[f"{sensor}_traits_trait_count"] = int(ds.sizes.get("trait", 0))
            years: list[int] = []
            source = ""
            if sensor == "prisma":
                if "acquisition_time_iso" in ds:
                    years = parse_years_from_iso(ds["acquisition_time_iso"].values)
                    source = "zarr_acquisition_time_iso"
                if not years and "source_product_id" in ds:
                    years = parse_years_from_product_ids(ds["source_product_id"].values)
                    source = "zarr_source_product_id"
            if not years and "time" in ds and np.issubdtype(ds["time"].dtype, np.datetime64):
                years = [int(v.year) for v in pd.to_datetime(ds["time"].values)]
                source = "zarr_time"
            if not years and fallback_scene_counts is not None:
                match_row = fallback_scene_counts[fallback_scene_counts["site_id"].astype(int) == site_id]
                if not match_row.empty:
                    for col in fallback_scene_counts.columns:
                        if col == "site_id":
                            continue
                        if isinstance(col, str) and col.startswith(f"{sensor}_scene_count_"):
                            year = int(col.rsplit("_", 1)[-1])
                        elif isinstance(col, str) and col.startswith("prisma_scene_count_"):
                            year = int(col.rsplit("_", 1)[-1])
                        elif isinstance(col, str) and col.startswith("enmap_scene_count_"):
                            year = int(col.rsplit("_", 1)[-1])
                        elif isinstance(col, str) and col.isdigit():
                            year = int(col)
                        elif isinstance(col, (int, np.integer)):
                            year = int(col)
                        else:
                            continue
                        count = int(match_row.iloc[0][col])
                        years.extend([year] * count)
                    source = "scene_hits_fallback_for_legacy_zarr"
            row[f"{sensor}_traits_year_source"] = source
            for year, count in pd.Series(years, dtype="int64").value_counts().sort_index().items() if years else []:
                row[f"{sensor}_traits_count_{int(year)}"] = int(count)
        except Exception as exc:
            row[f"{sensor}_traits_error"] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows)


def flatten_year_counts(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["site_id"])
    out = df.copy()
    for col in list(out.columns):
        if col == "site_id":
            continue
        if isinstance(col, (int, np.integer)) or (isinstance(col, str) and col.isdigit()):
            out = out.rename(columns={col: f"{prefix}_count_{int(col)}"})
    return out


def add_threshold_flags(df: pd.DataFrame, prefix: str, target_years: list[int], thresholds: list[int]) -> pd.DataFrame:
    out = df.copy()
    for year in target_years:
        col = f"{prefix}_count_{year}"
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    for threshold in thresholds:
        cols = [f"{prefix}_count_{year}" for year in target_years]
        out[f"{prefix}_ge{threshold}_all_target_years"] = out[cols].ge(threshold).all(axis=1)
    return out


def extracted_track_site_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=lambda c: c in {"site", "site_id"})
    if "site_id" in df.columns:
        vals = pd.to_numeric(df["site_id"], errors="coerce")
    elif "site" in df.columns:
        vals = df["site"].astype(str).str.extract(r"(\d+)")[0]
        vals = pd.to_numeric(vals, errors="coerce")
    else:
        return set()
    return set(vals.dropna().astype(int).unique().tolist())


def classify_row(row: pd.Series, min_prisma: int, target_years: list[int]) -> str:
    if bool(row["in_current_filtered37"]):
        return "included_current37"
    if bool(row["discarded_because_overlap"]):
        return "excluded_overlap_duplicate"
    if bool(row["discarded_because_recorded_no_fluxnet"]):
        return "excluded_recorded_no_fluxnet"
    if not bool(row["gpp_catalog_covers_target_years"]):
        return "excluded_or_risky_missing_gpp_target_years"
    prisma_ok = bool(row.get(f"prisma_traits_ge{min_prisma}_all_target_years", False))
    enmap_ok = bool(row.get("enmap_traits_ge1_all_target_years", False))
    if prisma_ok and not enmap_ok:
        return "reconsider_prisma_rescues_hyperspectral_coverage"
    if prisma_ok:
        return "reconsider_prisma_available"
    if bool(row["discarded_because_no_enmap_or_weak_enmap"]):
        return "excluded_enmap_problem_prisma_not_sufficient"
    if bool(row["discarded_because_no_hrl"]):
        return "excluded_no_hrl_still_blocked"
    return "not_included_needs_manual_review"


def write_notes(out_dir: Path, summary: dict[str, object]) -> Path:
    path = out_dir / "README.md"
    target_years = ", ".join(str(v) for v in summary["target_years"])
    text = f"""# Site Inclusion Audit With Existing PRISMA

Generated by:

```bash
python docs/paper0_perspective/scripts/audit_site_inclusion_with_prisma.py
```

## Scope

- The current repository does not expose a 52-row site table.
- Audited site universes found in current state:
  - `input_data/fluxnet_sites.csv`: {summary["n_all_sites"]} sites.
  - `input_data/fluxnet_sites_4enmap.csv`: {summary["n_enmap_subset"]} sites.
  - `input_data/prisma/manifests/prisma_query_aois.csv`: {summary["n_prisma_query_aois"]} sites.
  - `input_data/filtered_sites.csv`: {summary["n_current_filtered"]} currently included sites.
- Target years checked for annual sufficiency: {target_years}.
- Default PRISMA sufficiency flag: at least {summary["min_prisma_per_year"]} processed PRISMA trait timestamp per target year.

## Headline Counts

- Excluded from current 37 because of recorded overlap: {summary["n_recorded_overlap_excluded"]}.
- Excluded from current 37 because of recorded no-FLUXNET/no-GPP: {summary["n_recorded_no_fluxnet_excluded"]}.
- Excluded from current 37 for neither overlap nor recorded no-GPP: {summary["n_excluded_neither_overlap_nor_recorded_no_fluxnet"]}.
- Current processed PRISMA traits-region sites: {summary["n_prisma_traits_sites"]}.
- Excluded sites with any processed PRISMA traits-region: {summary["n_excluded_with_any_prisma_traits"]}.
- Excluded sites with PRISMA meeting the default target-year sufficiency flag: {summary["n_excluded_prisma_sufficient_default"]}.

## Interpretation

Use `site_inclusion_audit.csv` as the authoritative row-level table. The `recommended_status` column separates original recorded reasons from current data availability. PRISMA can rescue an EnMAP coverage problem only when the processed PRISMA traits data meet the target-year threshold; it does not solve duplicated/overlapping site design decisions or missing GPP requirements.
"""
    path.write_text(text)
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_years = sorted(set(int(v) for v in args.target_years))

    all_sites = pd.read_csv(args.all_sites)
    all_sites["site_id"] = all_sites["site_id"].astype(int)
    all_sites["reason_abandon_norm"] = all_sites["reason_abandon"].map(normalize_reason)
    all_sites["recorded_reason_group"] = all_sites["reason_abandon_norm"].map(reason_group)

    enmap_subset_ids = set(pd.read_csv(args.enmap_site_subset)["site_id"].astype(int).tolist()) if args.enmap_site_subset.exists() else set()
    filtered_ids = set(pd.read_csv(args.current_filtered_sites)["site_id"].astype(int).tolist()) if args.current_filtered_sites.exists() else set()
    prisma_query_ids = set(pd.read_csv(args.prisma_query_aois)["site_id"].astype(int).tolist()) if args.prisma_query_aois.exists() else set()
    prisma_track_ids = extracted_track_site_ids(args.prisma_query_track)

    audit = all_sites.copy()
    audit["in_enmap_site_subset50"] = audit["site_id"].isin(enmap_subset_ids)
    audit["in_current_filtered37"] = audit["site_id"].isin(filtered_ids)
    audit["in_prisma_query_aois50"] = audit["site_id"].isin(prisma_query_ids)
    audit["in_prisma_query_track"] = audit["site_id"].isin(prisma_track_ids)

    audit["discarded_from_current37"] = ~audit["in_current_filtered37"]
    audit["discarded_because_overlap"] = audit["discarded_from_current37"] & audit["reason_abandon_norm"].str.contains("overlap", na=False)
    audit["discarded_because_recorded_no_fluxnet"] = audit["discarded_from_current37"] & audit["reason_abandon_norm"].str.contains("no fluxnet|no gpp", regex=True, na=False)
    audit["discarded_because_no_hrl"] = audit["discarded_from_current37"] & audit["reason_abandon_norm"].str.contains("no hrl", na=False)
    audit["discarded_because_no_enmap_or_weak_enmap"] = audit["discarded_from_current37"] & audit["reason_abandon_norm"].str.contains(
        "no enmap|small part|noisy", regex=True, na=False
    )

    flux_map = build_fluxnet_mapping(audit, args.fluxnet_shuttle, args.fluxnet_max_distance_km)
    flux_local = add_local_gpp_status(flux_map, target_years)
    audit = audit.merge(flux_map, on="site_id", how="left").merge(flux_local, on="site_id", how="left")
    audit["gpp_catalog_covers_target_years"] = (
        pd.to_numeric(audit["fluxnet_first_year"], errors="coerce").le(min(target_years))
        & pd.to_numeric(audit["fluxnet_last_year"], errors="coerce").ge(max(target_years))
    )
    audit["gpp_missing_or_insufficient_for_target_years"] = ~audit["gpp_catalog_covers_target_years"]

    enmap_scene_counts = flatten_year_counts(scene_year_counts(args.enmap_scene_hits, "scene_date"), "enmap_scene")
    prisma_scene_counts = flatten_year_counts(scene_year_counts(args.prisma_scene_hits, "acquisition_time"), "prisma_scene")
    audit = audit.merge(enmap_scene_counts, on="site_id", how="left").merge(prisma_scene_counts, on="site_id", how="left")

    enmap_traits = zarr_trait_year_counts(args.enmap_traits_dir, "enmap")
    prisma_traits = zarr_trait_year_counts(args.prisma_traits_dir, "prisma", fallback_scene_counts=prisma_scene_counts)
    audit = audit.merge(enmap_traits, on="site_id", how="left").merge(prisma_traits, on="site_id", how="left")

    for col in ["enmap_traits_time_count", "enmap_traits_trait_count", "prisma_traits_time_count", "prisma_traits_trait_count"]:
        if col not in audit.columns:
            audit[col] = 0
        audit[col] = pd.to_numeric(audit[col], errors="coerce").fillna(0).astype(int)
    audit["has_any_enmap_traits_region"] = audit["enmap_traits_time_count"] > 0
    audit["has_any_prisma_traits_region"] = audit["prisma_traits_time_count"] > 0

    audit = add_threshold_flags(audit, "enmap_traits", target_years, thresholds=[1, 2, 3])
    audit = add_threshold_flags(audit, "prisma_traits", target_years, thresholds=[1, 2, 3])
    audit = add_threshold_flags(audit, "enmap_scene", target_years, thresholds=[1, 2, 3])
    audit = add_threshold_flags(audit, "prisma_scene", target_years, thresholds=[1, 2, 3])

    audit["hyperspectral_traits_ge1_any_sensor_all_target_years"] = audit["enmap_traits_ge1_all_target_years"] | audit[
        "prisma_traits_ge1_all_target_years"
    ]
    audit["prisma_can_expand_over_enmap_ge1"] = audit["prisma_traits_ge1_all_target_years"] & ~audit["enmap_traits_ge1_all_target_years"]
    audit["recommended_status"] = audit.apply(lambda row: classify_row(row, args.min_prisma_per_year, target_years), axis=1)

    front_cols = [
        "site_id",
        "site_name",
        "in_current_filtered37",
        "in_enmap_site_subset50",
        "in_prisma_query_aois50",
        "reason_abandon",
        "recorded_reason_group",
        "recommended_status",
        "discarded_because_overlap",
        "discarded_because_recorded_no_fluxnet",
        "discarded_because_no_enmap_or_weak_enmap",
        "discarded_because_no_hrl",
        "gpp_catalog_covers_target_years",
        "gpp_local_finite_all_target_years",
        "fluxnet_site_id",
        "fluxnet_first_year",
        "fluxnet_last_year",
        "enmap_traits_time_count",
        "prisma_traits_time_count",
        "enmap_traits_ge1_all_target_years",
        "prisma_traits_ge1_all_target_years",
        "prisma_traits_ge2_all_target_years",
        "prisma_traits_ge3_all_target_years",
        "prisma_can_expand_over_enmap_ge1",
    ]
    other_cols = [c for c in audit.columns if c not in front_cols]
    audit = audit[front_cols + other_cols]
    audit = audit.sort_values("site_id")

    audit_path = args.output_dir / "site_inclusion_audit.csv"
    audit.to_csv(audit_path, index=False)

    excluded = audit[audit["discarded_from_current37"]].copy()
    excluded_path = args.output_dir / "excluded_sites_by_reason.csv"
    excluded.to_csv(excluded_path, index=False)

    candidates = excluded[
        excluded[f"prisma_traits_ge{args.min_prisma_per_year}_all_target_years"]
        | excluded["has_any_prisma_traits_region"]
        | excluded["in_prisma_query_aois50"]
    ].copy()
    candidates_path = args.output_dir / "excluded_sites_with_prisma_context.csv"
    candidates.to_csv(candidates_path, index=False)

    summary = {
        "target_years": target_years,
        "min_prisma_per_year": int(args.min_prisma_per_year),
        "n_all_sites": int(len(audit)),
        "n_enmap_subset": int(audit["in_enmap_site_subset50"].sum()),
        "n_prisma_query_aois": int(audit["in_prisma_query_aois50"].sum()),
        "n_current_filtered": int(audit["in_current_filtered37"].sum()),
        "n_recorded_overlap_excluded": int(audit["discarded_because_overlap"].sum()),
        "n_recorded_no_fluxnet_excluded": int(audit["discarded_because_recorded_no_fluxnet"].sum()),
        "n_excluded_neither_overlap_nor_recorded_no_fluxnet": int(
            (audit["discarded_from_current37"] & ~audit["discarded_because_overlap"] & ~audit["discarded_because_recorded_no_fluxnet"]).sum()
        ),
        "n_prisma_traits_sites": int(audit["has_any_prisma_traits_region"].sum()),
        "n_enmap_traits_sites": int(audit["has_any_enmap_traits_region"].sum()),
        "n_excluded_with_any_prisma_traits": int((audit["discarded_from_current37"] & audit["has_any_prisma_traits_region"]).sum()),
        "n_excluded_prisma_sufficient_default": int(
            (audit["discarded_from_current37"] & audit[f"prisma_traits_ge{args.min_prisma_per_year}_all_target_years"]).sum()
        ),
        "recommended_status_counts": audit["recommended_status"].value_counts(dropna=False).to_dict(),
        "recorded_reason_group_counts": audit["recorded_reason_group"].value_counts(dropna=False).to_dict(),
        "outputs": {
            "site_inclusion_audit": str(audit_path),
            "excluded_sites_by_reason": str(excluded_path),
            "excluded_sites_with_prisma_context": str(candidates_path),
        },
    }
    summary_path = args.output_dir / "site_inclusion_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    notes_path = write_notes(args.output_dir, summary)
    summary["outputs"]["notes"] = str(notes_path)
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")

    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
