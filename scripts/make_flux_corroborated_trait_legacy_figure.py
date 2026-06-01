#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import zipfile

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats


PAPER_ROOT = Path(__file__).resolve().parents[1]
ECOSPECTRA_ROOT = Path("/home/chaonan.ji/data/projects/ecospectra")

RECOVERY_TABLE = PAPER_ROOT / "tables" / "fig4_drought_anchor_xgb" / "cap5000_sites_recovery_timeline_2022_2023aug.csv"
UNIFIED_2022_TABLE = (
    PAPER_ROOT
    / "tables"
    / "fig3_all_sites_cap2000_flux_traits"
    / "all_sites_unified_biweekly_analysis_2022.csv"
)
PIXEL_TRAIT_PREDICTIONS = (
    PAPER_ROOT
    / "tables"
    / "fig4_drought_anchor_xgb"
    / "cap5000_sites_xgb_biweekly_trait_predictions_2022_2023aug.parquet"
)
FLUXNET_MAPPING = PAPER_ROOT / "tables" / "fig3_all_sites_cap2000_flux_traits" / "site_fluxnet_mapping.csv"
FLUXNET_ZIP_DIR = PAPER_ROOT / "tables" / "fig3_all_sites_cap2000_flux_traits" / "fluxnet"

OUT_TABLE_DIR = PAPER_ROOT / "tables" / "flux_corroborated_trait_legacy"
OUT_FIG_DIR = PAPER_ROOT / "figures" / "generated" / "flux_corroborated_trait_legacy"

EVENT_DATE = pd.Timestamp("2022-08-06")
BASELINE_START = pd.Timestamp("2022-05-14")
BASELINE_END = pd.Timestamp("2022-06-25")
TROUGH_START = pd.Timestamp("2022-07-23")
TROUGH_END = pd.Timestamp("2022-08-20")
RECOVERY_2023_START = pd.Timestamp("2023-07-08")
RECOVERY_2023_END = pd.Timestamp("2023-08-19")
DROUGHT_SWC_RELATIVE_DROP_THRESHOLD = 0.10

FLUXNET_SOURCE_VARS = [
    "GPP_NT_VUT_REF",
    "NEE_VUT_REF",
    "RECO_NT_VUT_REF",
    "LE_F_MDS",
    "VPD_F_MDS",
    "SWC_F_MDS_1",
    "SWC_F_MDS_2",
]
FLUX_VARS = ["GPP_NT_VUT_REF", "NEP_VUT_REF"]

RECOVERY_THRESHOLD = 70.0
S2_ROOT = ECOSPECTRA_ROOT / "input_data" / "s2" / "region"
S2_WINDOW_DAYS = 28.0
RESAMPLING_SEED = 20260601
N_BOOTSTRAP = 20000
N_PERMUTATIONS = 100000
FOREST_SCOPE_TYPES = {
    "Evergreen forest",
    "Deciduous forest",
    "Mixed forest",
    "Tree cover",
    "Open/savanna",
}
STRICT_FOREST_SCOPE_TYPES = {
    "Evergreen forest",
    "Deciduous forest",
    "Mixed forest",
    "Tree cover",
}
PRIMARY_GPP_METRIC = "gpp_persistence_z_2022_2023"
PRIMARY_GPP_LABEL = "2022-2023 GPP persistence anomaly (z)"
GPP_METRICS = {
    "gpp_pct_2023_vs_baseline": {
        "column": "gpp_anomaly_pct_2023_vs_baseline",
        "label": "2023 GPP anomaly vs 2022 baseline (%)",
    },
    "gpp_z_2023_vs_interannual": {
        "column": "gpp_anomaly_z_2023_vs_interannual",
        "label": "2023 GPP anomaly vs same-season climatology (z)",
    },
    "gpp_persistence_z_2022_2023": {
        "column": "gpp_persistence_z_2022_2023",
        "label": PRIMARY_GPP_LABEL,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Perspective-style flux-corroborated trait legacy figure: "
            "NDVI recovery vs chlorophyll recovery, colored by 2023 flux anomaly."
        )
    )
    parser.add_argument("--recovery-table", type=Path, default=RECOVERY_TABLE)
    parser.add_argument("--unified-2022-table", type=Path, default=UNIFIED_2022_TABLE)
    parser.add_argument("--pixel-trait-predictions", type=Path, default=PIXEL_TRAIT_PREDICTIONS)
    parser.add_argument("--fluxnet-mapping", type=Path, default=FLUXNET_MAPPING)
    parser.add_argument("--fluxnet-zip-dir", type=Path, default=FLUXNET_ZIP_DIR)
    parser.add_argument("--output-table-dir", type=Path, default=OUT_TABLE_DIR)
    parser.add_argument("--output-figure-dir", type=Path, default=OUT_FIG_DIR)
    parser.add_argument("--recovery-threshold", type=float, default=RECOVERY_THRESHOLD)
    parser.add_argument(
        "--primary-gpp-metric",
        choices=sorted(GPP_METRICS),
        default=PRIMARY_GPP_METRIC,
        help="GPP metric used for the main forest-domain color scale and primary association test.",
    )
    return parser.parse_args()


def setup_style() -> None:
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.grid": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(axis="both", width=0.5, length=3)
    ax.grid(False)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, preview: bool = True) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for ext in ("pdf", "svg", "eps"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(str(path))
    if preview:
        for ext in ("png", "jpg"):
            path = out_dir / f"{stem}.{ext}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            paths.append(str(path))
    plt.close(fig)
    return paths


def recovery_score(value: float, baseline: float, trough: float, min_relative_magnitude: float = 0.02) -> tuple[float, float, bool]:
    if not (np.isfinite(value) and np.isfinite(baseline) and np.isfinite(trough)):
        return math.nan, math.nan, False
    denom = float(baseline) - float(trough)
    scale = max(abs(float(baseline)), 1e-12)
    interpretable = abs(denom) / scale >= float(min_relative_magnitude)
    if abs(denom) <= 1e-12:
        return math.nan, math.nan, False
    pct = 100.0 * (float(value) - float(trough)) / denom
    return pct, float(np.clip(pct, 0.0, 100.0)), bool(interpretable)


def finite_median(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").astype(float)
    return float(np.nanmedian(vals)) if np.isfinite(vals).any() else math.nan


def median_in_stage(df: pd.DataFrame, variable: str, stage: str) -> float:
    sub = df[(df["variable"] == variable) & (df["stage"] == stage)]
    if sub.empty:
        return math.nan
    return finite_median(sub["value"])


def safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype=np.float32)
    den = np.asarray(den, dtype=np.float32)
    out = np.full(num.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-12)
    out[finite] = num[finite] / den[finite]
    return out


def nearest_20m_index(index_30m: int, max_size: int) -> int:
    raw = int(round(((float(index_30m) + 0.5) * 1.5) - 0.5))
    return int(np.clip(raw, 0, max_size - 1))


def pixel_uid(site_id: int, y: int, x: int) -> str:
    return f"site_{int(site_id)}_y_{int(y)}_x_{int(x)}"


def in_recovery_windows(date: pd.Timestamp) -> bool:
    date = pd.Timestamp(date)
    return bool(
        (BASELINE_START <= date <= BASELINE_END)
        or (TROUGH_START <= date <= TROUGH_END)
        or (RECOVERY_2023_START <= date <= RECOVERY_2023_END)
    )


def load_recovery_site_metrics(path: Path, threshold: float) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    required = {
        "site_id",
        "site_name",
        "date",
        "stage",
        "variable",
        "value",
        "recovery_pct",
        "recovery_pct_clipped",
        "recovery_interpretable",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing recovery columns in {path}: {sorted(missing)}")
    df["site_id"] = df["site_id"].astype(int)
    df["recovery_pct"] = pd.to_numeric(df["recovery_pct"], errors="coerce")
    df["recovery_pct_clipped"] = pd.to_numeric(df["recovery_pct_clipped"], errors="coerce")
    df["recovery_interpretable"] = df["recovery_interpretable"].astype(bool)

    rows: list[dict[str, object]] = []
    for site_id, site in df.groupby("site_id"):
        row: dict[str, object] = {"site_id": int(site_id), "site_name": str(site["site_name"].dropna().iloc[0])}
        for variable in ["NDVI", "Chlorophyll", "LMA", "GPP_NT_VUT_REF"]:
            var = site[(site["variable"] == variable) & (site["stage"] == "2023 recovery")].copy()
            if variable == "GPP_NT_VUT_REF":
                continue
            prefix = variable.lower().replace("chlorophyll", "chl")
            row[f"{prefix}_recovery_pct_raw_2023"] = finite_median(var["recovery_pct"]) if not var.empty else math.nan
            row[f"{prefix}_recovery_pct_2023"] = finite_median(var["recovery_pct_clipped"]) if not var.empty else math.nan
            row[f"{prefix}_recovery_interpretable"] = bool(var["recovery_interpretable"].mean() >= 0.5) if not var.empty else False
        ndvi = row.get("ndvi_recovery_pct_2023", math.nan)
        chl = row.get("chl_recovery_pct_2023", math.nan)
        row["legacy_gap_pct"] = float(ndvi - chl) if np.isfinite(ndvi) and np.isfinite(chl) else math.nan
        row["quadrant"] = quadrant_label(ndvi, chl, threshold)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("site_id")


def build_drought_driver_table(recovery_path: Path) -> pd.DataFrame:
    recovery = pd.read_csv(recovery_path, parse_dates=["date"])
    rows: list[dict[str, object]] = []
    for site_id, site_df in recovery.groupby("site_id"):
        swc_pre = median_in_stage(site_df, "SWC", "pre-event")
        swc_trough = median_in_stage(site_df, "SWC", "drought trough")
        tmax_pre = median_in_stage(site_df, "Tmax", "pre-event")
        tmax_trough = median_in_stage(site_df, "Tmax", "drought trough")
        precip_pre = median_in_stage(site_df, "P", "pre-event")
        precip_trough = median_in_stage(site_df, "P", "drought trough")
        swc_drop = swc_pre - swc_trough if np.isfinite(swc_pre) and np.isfinite(swc_trough) else math.nan
        swc_relative_drop = swc_drop / abs(swc_pre) if np.isfinite(swc_drop) and np.isfinite(swc_pre) and abs(swc_pre) > 1e-12 else math.nan
        rows.append(
            {
                "site_id": int(site_id),
                "driver_swc_pre_event": swc_pre,
                "driver_swc_trough": swc_trough,
                "driver_swc_drop": swc_drop,
                "driver_swc_relative_drop": swc_relative_drop,
                "driver_tmax_delta_trough_minus_pre": (
                    tmax_trough - tmax_pre if np.isfinite(tmax_trough) and np.isfinite(tmax_pre) else math.nan
                ),
                "driver_precip_delta_trough_minus_pre": (
                    precip_trough - precip_pre if np.isfinite(precip_trough) and np.isfinite(precip_pre) else math.nan
                ),
                "driver_drought_exposed": bool(
                    np.isfinite(swc_relative_drop) and swc_relative_drop >= DROUGHT_SWC_RELATIVE_DROP_THRESHOLD
                ),
                "driver_drought_exposure_rule": (
                    f"ERA5-Land SWC pre-event to trough relative drop >= {DROUGHT_SWC_RELATIVE_DROP_THRESHOLD:.0%}"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("site_id")


def quadrant_label(ndvi: float, chl: float, threshold: float) -> str:
    if not (np.isfinite(ndvi) and np.isfinite(chl)):
        return "not_interpretable"
    ndvi_ok = ndvi >= threshold
    chl_ok = chl >= threshold
    if ndvi_ok and chl_ok:
        return "greenness_and_trait_recovered"
    if ndvi_ok and not chl_ok:
        return "legacy_quadrant"
    if not ndvi_ok and chl_ok:
        return "trait_only_recovered"
    return "unrecovered"


def load_site_metadata(unified_path: Path, mapping_path: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    if unified_path.exists():
        unified = pd.read_csv(unified_path)
        cols = [c for c in ["site_id", "site_name", "fluxnet_site_id", "igbp", "worldcover_class", "pft_group"] if c in unified.columns]
        if cols:
            parts.append(unified[cols].drop_duplicates("site_id"))
    if mapping_path.exists():
        mapping = pd.read_csv(mapping_path)
        cols = [c for c in ["site_id", "site_name", "fluxnet_site_id", "has_2022", "first_year", "last_year", "fluxnet_product_name"] if c in mapping.columns]
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
    meta["site_id"] = meta["site_id"].astype(int)
    if "pft_group" not in meta.columns:
        meta["pft_group"] = np.nan
    if "worldcover_class" not in meta.columns:
        meta["worldcover_class"] = np.nan
    if "igbp" not in meta.columns:
        meta["igbp"] = np.nan
    meta["forest_type"] = meta.apply(forest_type_label, axis=1)
    return meta


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
    if value in {"GRA", "Grassland"}:
        return "Grassland"
    if value in {"CRO", "Cropland"}:
        return "Cropland"
    if value in {"WET", "Wetland"}:
        return "Wetland"
    if value in {"OSH", "SAV"}:
        return "Open/savanna"
    return "Unclassified"


def is_forest_scope(forest_type: object) -> bool:
    return str(forest_type) in FOREST_SCOPE_TYPES


def is_strict_forest_scope(forest_type: object) -> bool:
    return str(forest_type) in STRICT_FOREST_SCOPE_TYPES


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def finite_nanmean(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else math.nan


def same_doy_window_medians(
    daily: pd.DataFrame,
    column: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[int, float]:
    if column not in daily.columns or "date" not in daily.columns:
        return {}
    out: dict[int, float] = {}
    years = sorted(pd.to_datetime(daily["date"], errors="coerce").dropna().dt.year.unique().tolist())
    for year in years:
        try:
            y_start = pd.Timestamp(year=int(year), month=start.month, day=start.day)
            y_end = pd.Timestamp(year=int(year), month=end.month, day=end.day)
        except ValueError:
            continue
        value = window_median(daily, column, y_start, y_end)
        if np.isfinite(value):
            out[int(year)] = float(value)
    return out


def interannual_anomaly(
    annual_values: dict[int, float],
    target_year: int,
    exclude_years: set[int],
) -> dict[str, float | int]:
    target = annual_values.get(int(target_year), math.nan)
    reference = [float(v) for y, v in annual_values.items() if y not in exclude_years and np.isfinite(v)]
    if len(reference) < 2 or not np.isfinite(target):
        return {
            "target": target,
            "reference_mean": math.nan,
            "reference_sd": math.nan,
            "reference_n_years": int(len(reference)),
            "anomaly_abs": math.nan,
            "anomaly_pct": math.nan,
            "anomaly_z": math.nan,
        }
    mean = float(np.mean(reference))
    sd = float(np.std(reference, ddof=1))
    anomaly_abs = float(target - mean)
    return {
        "target": float(target),
        "reference_mean": mean,
        "reference_sd": sd,
        "reference_n_years": int(len(reference)),
        "anomaly_abs": anomaly_abs,
        "anomaly_pct": 100.0 * anomaly_abs / abs(mean) if abs(mean) > 1e-12 else math.nan,
        "anomaly_z": anomaly_abs / sd if sd > 1e-12 else math.nan,
    }


def find_fluxnet_zip(site_code: str, zip_dir: Path) -> Path | None:
    if not isinstance(site_code, str) or not site_code or site_code.lower() == "nan":
        return None
    search_dirs = [
        zip_dir,
        ECOSPECTRA_ROOT / "input_data" / "fluxnet",
        ECOSPECTRA_ROOT / "input_data" / "0_others",
        ECOSPECTRA_ROOT / "input_data",
    ]
    for root in search_dirs:
        if not root.exists():
            continue
        candidates = sorted(root.glob(f"*{site_code}*FLUXNET*.zip"))
        if candidates:
            return candidates[0]
    return None


def read_fluxnet_daily(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        fluxmet_daily = [n for n in names if n.lower().endswith(".csv") and "fluxmet_dd" in n.lower()]
        daily = [
            n
            for n in names
            if n.lower().endswith(".csv") and ("_dd_" in n.lower() or "daily" in n.lower()) and "bif" not in n.lower()
        ]
        chosen = (fluxmet_daily or daily)[0] if (fluxmet_daily or daily) else None
        if chosen is None:
            raise ValueError(f"No daily FLUXNET CSV found in {zip_path}")
        with zf.open(chosen) as handle:
            df = pd.read_csv(handle)
    timestamp_col = "TIMESTAMP" if "TIMESTAMP" in df.columns else "TIMESTAMP_START"
    ts = df[timestamp_col].astype(str).str.slice(0, 8)
    df["date"] = pd.to_datetime(ts, format="%Y%m%d", errors="coerce")
    available = [v for v in FLUXNET_SOURCE_VARS if v in df.columns]
    if not available:
        return pd.DataFrame(columns=["date"])
    df[available] = df[available].replace(-9999, np.nan)
    daily = df.groupby("date", as_index=False)[available].mean(numeric_only=True)
    if "NEE_VUT_REF" in daily.columns:
        daily["NEP_VUT_REF"] = -pd.to_numeric(daily["NEE_VUT_REF"], errors="coerce")
    return daily


def window_median(df: pd.DataFrame, column: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if column not in df.columns:
        return math.nan
    sub = df[(df["date"] >= start) & (df["date"] <= end)]
    if sub.empty:
        return math.nan
    vals = pd.to_numeric(sub[column], errors="coerce").astype(float)
    return float(np.nanmedian(vals)) if np.isfinite(vals).any() else math.nan


def compute_flux_metrics(meta: pd.DataFrame, zip_dir: Path) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    daily_by_site: dict[int, pd.DataFrame] = {}
    for rec in meta.sort_values("site_id").itertuples(index=False):
        site_id = int(rec.site_id)
        site_code = str(getattr(rec, "fluxnet_site_id", ""))
        row: dict[str, object] = {"site_id": site_id, "fluxnet_site_id": site_code}
        zip_path = find_fluxnet_zip(site_code, zip_dir)
        row["fluxnet_zip"] = str(zip_path) if zip_path else ""
        if zip_path is None:
            for var in FLUX_VARS:
                add_empty_flux_fields(row, var)
            rows.append(row)
            continue
        try:
            daily = read_fluxnet_daily(zip_path)
        except Exception as exc:
            row["flux_read_error"] = str(exc)
            for var in FLUX_VARS:
                add_empty_flux_fields(row, var)
            rows.append(row)
            continue
        daily_by_site[site_id] = daily
        for var in FLUX_VARS:
            baseline = window_median(daily, var, BASELINE_START, BASELINE_END)
            trough = window_median(daily, var, TROUGH_START, TROUGH_END)
            late = window_median(daily, var, RECOVERY_2023_START, RECOVERY_2023_END)
            rec_pct, rec_clip, interpretable = recovery_score(late, baseline, trough)
            short = "gpp" if var == "GPP_NT_VUT_REF" else "nep"
            row[f"{short}_baseline_2022"] = baseline
            row[f"{short}_trough_2022"] = trough
            row[f"{short}_median_2023"] = late
            row[f"{short}_anomaly_abs_2023_vs_baseline"] = late - baseline if np.isfinite(late) and np.isfinite(baseline) else math.nan
            row[f"{short}_anomaly_pct_2023_vs_baseline"] = (
                100.0 * (late - baseline) / abs(baseline)
                if np.isfinite(late) and np.isfinite(baseline) and abs(baseline) > 1e-12
                else math.nan
            )
            row[f"{short}_recovery_pct_raw_2023"] = rec_pct
            row[f"{short}_recovery_pct_2023"] = rec_clip
            row[f"{short}_recovery_interpretable"] = interpretable

            trough_annual = same_doy_window_medians(daily, var, TROUGH_START, TROUGH_END)
            recovery_annual = same_doy_window_medians(daily, var, RECOVERY_2023_START, RECOVERY_2023_END)
            trough_z = interannual_anomaly(trough_annual, target_year=2022, exclude_years={2022, 2023})
            recovery_z = interannual_anomaly(recovery_annual, target_year=2023, exclude_years={2022, 2023})
            row[f"{short}_trough_anomaly_z_2022_vs_interannual"] = trough_z["anomaly_z"]
            row[f"{short}_trough_anomaly_pct_2022_vs_interannual"] = trough_z["anomaly_pct"]
            row[f"{short}_trough_interannual_mean"] = trough_z["reference_mean"]
            row[f"{short}_trough_interannual_sd"] = trough_z["reference_sd"]
            row[f"{short}_trough_interannual_n_years"] = trough_z["reference_n_years"]
            row[f"{short}_anomaly_z_2023_vs_interannual"] = recovery_z["anomaly_z"]
            row[f"{short}_anomaly_pct_2023_vs_interannual"] = recovery_z["anomaly_pct"]
            row[f"{short}_recovery_interannual_mean"] = recovery_z["reference_mean"]
            row[f"{short}_recovery_interannual_sd"] = recovery_z["reference_sd"]
            row[f"{short}_recovery_interannual_n_years"] = recovery_z["reference_n_years"]
            row[f"{short}_persistence_z_2022_2023"] = finite_nanmean(
                [float(trough_z["anomaly_z"]), float(recovery_z["anomaly_z"])]
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("site_id"), daily_by_site


def add_empty_flux_fields(row: dict[str, object], var: str) -> None:
    short = "gpp" if var == "GPP_NT_VUT_REF" else "nep"
    for suffix in [
        "baseline_2022",
        "trough_2022",
        "median_2023",
        "anomaly_abs_2023_vs_baseline",
        "anomaly_pct_2023_vs_baseline",
        "recovery_pct_raw_2023",
        "recovery_pct_2023",
        "trough_anomaly_z_2022_vs_interannual",
        "trough_anomaly_pct_2022_vs_interannual",
        "trough_interannual_mean",
        "trough_interannual_sd",
        "anomaly_z_2023_vs_interannual",
        "anomaly_pct_2023_vs_interannual",
        "recovery_interannual_mean",
        "recovery_interannual_sd",
        "persistence_z_2022_2023",
    ]:
        row[f"{short}_{suffix}"] = math.nan
    row[f"{short}_trough_interannual_n_years"] = 0
    row[f"{short}_recovery_interannual_n_years"] = 0
    row[f"{short}_recovery_interpretable"] = False


def corr_value(x: pd.Series, y: pd.Series, method: str = "pearson") -> tuple[float, int]:
    pair = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(pair) < 3:
        return math.nan, int(len(pair))
    xv = pair["x"].astype(float)
    yv = pair["y"].astype(float)
    if method == "spearman":
        xv = xv.rank(method="average")
        yv = yv.rank(method="average")
    if float(xv.std(ddof=0)) <= 1e-12 or float(yv.std(ddof=0)) <= 1e-12:
        return math.nan, int(len(pair))
    return float(np.corrcoef(xv, yv)[0, 1]), int(len(pair))


def corr_test_value(x: pd.Series, y: pd.Series, method: str = "spearman") -> tuple[float, float, int]:
    pair = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(pair) < 3:
        return math.nan, math.nan, int(len(pair))
    xv = pair["x"].astype(float)
    yv = pair["y"].astype(float)
    if float(xv.std(ddof=0)) <= 1e-12 or float(yv.std(ddof=0)) <= 1e-12:
        return math.nan, math.nan, int(len(pair))
    if method == "spearman":
        result = stats.spearmanr(xv, yv)
    elif method == "pearson":
        result = stats.pearsonr(xv, yv)
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    return float(result.statistic), float(result.pvalue), int(len(pair))


def mann_whitney_value(a: pd.Series, b: pd.Series) -> tuple[float, float, int, int]:
    av = pd.to_numeric(a, errors="coerce").dropna().astype(float)
    bv = pd.to_numeric(b, errors="coerce").dropna().astype(float)
    if len(av) < 1 or len(bv) < 1:
        return math.nan, math.nan, int(len(av)), int(len(bv))
    result = stats.mannwhitneyu(av, bv, alternative="two-sided")
    return float(result.statistic), float(result.pvalue), int(len(av)), int(len(bv))


def residualize(y: pd.Series, x: pd.Series) -> pd.Series:
    pair = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")})
    out = pd.Series(np.nan, index=pair.index, dtype=float)
    finite = pair.dropna()
    if len(finite) < 3 or float(finite["x"].std(ddof=0)) <= 1e-12:
        return out
    coef = np.polyfit(finite["x"].astype(float), finite["y"].astype(float), deg=1)
    out.loc[finite.index] = finite["y"].astype(float) - (coef[0] * finite["x"].astype(float) + coef[1])
    return out


def add_metric_association(
    summary: dict[str, object],
    site: pd.DataFrame,
    metric_key: str,
    metric_col: str,
) -> None:
    if metric_col not in site.columns:
        return
    pair = site[np.isfinite(site["legacy_gap_pct"]) & np.isfinite(site[metric_col])].copy()
    legacy = pair[pair["quadrant"] == "legacy_quadrant"]
    nonlegacy = pair[pair["quadrant"] != "legacy_quadrant"]
    rho, rho_p, rho_n = corr_test_value(pair["legacy_gap_pct"], pair[metric_col], method="spearman")
    pearson, pearson_p, _ = corr_test_value(pair["legacy_gap_pct"], pair[metric_col], method="pearson")
    chl_rho, chl_p, chl_n = corr_test_value(pair["chl_recovery_pct_2023"], pair[metric_col], method="spearman")
    ndvi_rho, ndvi_p, ndvi_n = corr_test_value(pair["ndvi_recovery_pct_2023"], pair[metric_col], method="spearman")
    u_stat, u_p, legacy_n, nonlegacy_n = mann_whitney_value(legacy[metric_col], nonlegacy[metric_col])
    legacy_median = float(np.nanmedian(legacy[metric_col])) if len(legacy) else math.nan
    nonlegacy_median = float(np.nanmedian(nonlegacy[metric_col])) if len(nonlegacy) else math.nan
    summary.update(
        {
            f"{metric_key}_pairwise_n": int(len(pair)),
            f"{metric_key}_legacy_n": int(len(legacy)),
            f"{metric_key}_nonlegacy_n": int(len(nonlegacy)),
            f"{metric_key}_legacy_median": legacy_median,
            f"{metric_key}_nonlegacy_median": nonlegacy_median,
            f"{metric_key}_legacy_minus_nonlegacy_median": (
                legacy_median - nonlegacy_median if np.isfinite(legacy_median) and np.isfinite(nonlegacy_median) else math.nan
            ),
            f"{metric_key}_mann_whitney_u": u_stat,
            f"{metric_key}_mann_whitney_p": u_p,
            f"{metric_key}_mann_whitney_legacy_n": legacy_n,
            f"{metric_key}_mann_whitney_nonlegacy_n": nonlegacy_n,
            f"{metric_key}_spearman_legacy_gap": rho,
            f"{metric_key}_spearman_legacy_gap_p": rho_p,
            f"{metric_key}_spearman_legacy_gap_n": rho_n,
            f"{metric_key}_pearson_legacy_gap": pearson,
            f"{metric_key}_pearson_legacy_gap_p": pearson_p,
            f"{metric_key}_spearman_chl_recovery": chl_rho,
            f"{metric_key}_spearman_chl_recovery_p": chl_p,
            f"{metric_key}_spearman_chl_recovery_n": chl_n,
            f"{metric_key}_spearman_ndvi_recovery": ndvi_rho,
            f"{metric_key}_spearman_ndvi_recovery_p": ndvi_p,
            f"{metric_key}_spearman_ndvi_recovery_n": ndvi_n,
        }
    )


def paired_site_data(site: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    metric_col = GPP_METRICS[metric_key]["column"]
    required = ["site_id", "site_name", "forest_type", "legacy_gap_pct", "quadrant", metric_col]
    pair = site[required].copy()
    pair = pair[np.isfinite(pair["legacy_gap_pct"]) & np.isfinite(pair[metric_col])].copy()
    return pair.sort_values("site_id").reset_index(drop=True)


def spearman_array(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or len(y) < 3:
        return math.nan
    if float(np.nanstd(x)) <= 1e-12 or float(np.nanstd(y)) <= 1e-12:
        return math.nan
    xr = stats.rankdata(np.asarray(x, dtype=float), method="average")
    yr = stats.rankdata(np.asarray(y, dtype=float), method="average")
    if float(np.nanstd(xr)) <= 1e-12 or float(np.nanstd(yr)) <= 1e-12:
        return math.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def site_level_resampling_summary(
    pair: pd.DataFrame,
    metric_key: str,
    n_bootstrap: int = N_BOOTSTRAP,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = RESAMPLING_SEED,
) -> dict[str, object]:
    metric_col = GPP_METRICS[metric_key]["column"]
    x = pd.to_numeric(pair["legacy_gap_pct"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pair[metric_col], errors="coerce").to_numpy(dtype=float)
    n = int(len(pair))
    observed = spearman_array(x, y)
    out: dict[str, object] = {
        f"{metric_key}_site_resampling_n": n,
        f"{metric_key}_bootstrap_replicates": int(n_bootstrap),
        f"{metric_key}_permutation_replicates": int(n_permutations),
        f"{metric_key}_resampling_seed": int(seed),
        f"{metric_key}_bootstrap_spearman_median": math.nan,
        f"{metric_key}_bootstrap_spearman_ci_low": math.nan,
        f"{metric_key}_bootstrap_spearman_ci_high": math.nan,
        f"{metric_key}_bootstrap_spearman_valid": 0,
        f"{metric_key}_permutation_spearman_p_two_sided": math.nan,
        f"{metric_key}_permutation_spearman_p_directional_negative": math.nan,
    }
    if n < 3 or not np.isfinite(observed):
        return out
    rng = np.random.default_rng(seed)
    boot = np.full(int(n_bootstrap), np.nan, dtype=float)
    for idx in range(int(n_bootstrap)):
        take = rng.integers(0, n, size=n)
        boot[idx] = spearman_array(x[take], y[take])
    valid_boot = boot[np.isfinite(boot)]
    if valid_boot.size:
        out.update(
            {
                f"{metric_key}_bootstrap_spearman_median": float(np.nanmedian(valid_boot)),
                f"{metric_key}_bootstrap_spearman_ci_low": float(np.nanquantile(valid_boot, 0.025)),
                f"{metric_key}_bootstrap_spearman_ci_high": float(np.nanquantile(valid_boot, 0.975)),
                f"{metric_key}_bootstrap_spearman_valid": int(valid_boot.size),
            }
        )

    perm = np.full(int(n_permutations), np.nan, dtype=float)
    xr = stats.rankdata(x, method="average")
    yr = stats.rankdata(y, method="average")
    for idx in range(int(n_permutations)):
        perm[idx] = float(np.corrcoef(xr, rng.permutation(yr))[0, 1])
    valid_perm = perm[np.isfinite(perm)]
    if valid_perm.size:
        out[f"{metric_key}_permutation_spearman_p_two_sided"] = float(
            (np.count_nonzero(np.abs(valid_perm) >= abs(observed)) + 1) / (valid_perm.size + 1)
        )
        out[f"{metric_key}_permutation_spearman_p_directional_negative"] = float(
            (np.count_nonzero(valid_perm <= observed) + 1) / (valid_perm.size + 1)
        )
    return out


def leave_one_out_table(pair: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    metric_col = GPP_METRICS[metric_key]["column"]
    rows: list[dict[str, object]] = []
    for rec in pair.itertuples(index=False):
        keep = pair[pair["site_id"].astype(int) != int(rec.site_id)].copy()
        rho, pval, n = corr_test_value(keep["legacy_gap_pct"], keep[metric_col], method="spearman")
        rows.append(
            {
                "omitted_site_id": int(rec.site_id),
                "omitted_site_name": str(rec.site_name),
                "omitted_forest_type": str(rec.forest_type),
                "omitted_quadrant": str(rec.quadrant),
                "n_remaining": int(n),
                "spearman_rho": rho,
                "spearman_p_scipy": pval,
            }
        )
    return pd.DataFrame(rows).sort_values("omitted_site_id")


def add_leave_one_out_summary(summary: dict[str, object], loo: pd.DataFrame, metric_key: str) -> None:
    vals = pd.to_numeric(loo["spearman_rho"], errors="coerce").dropna().astype(float)
    if vals.empty:
        summary.update(
            {
                f"{metric_key}_loo_min_rho": math.nan,
                f"{metric_key}_loo_max_rho": math.nan,
                f"{metric_key}_loo_negative_fraction": math.nan,
                f"{metric_key}_loo_all_negative": False,
            }
        )
        return
    summary.update(
        {
            f"{metric_key}_loo_min_rho": float(vals.min()),
            f"{metric_key}_loo_max_rho": float(vals.max()),
            f"{metric_key}_loo_negative_fraction": float(np.mean(vals < 0)),
            f"{metric_key}_loo_all_negative": bool((vals < 0).all()),
        }
    )


def build_summary(site: pd.DataFrame, threshold: float, scope_name: str, primary_metric: str) -> dict[str, object]:
    gpp_flux = site[np.isfinite(site["gpp_anomaly_pct_2023_vs_baseline"])].copy()
    gpp_pair = site[np.isfinite(site["legacy_gap_pct"]) & np.isfinite(site["gpp_anomaly_pct_2023_vs_baseline"])].copy()
    legacy_pair = gpp_pair[gpp_pair["quadrant"] == "legacy_quadrant"]
    nonlegacy_pair = gpp_pair[gpp_pair["quadrant"] != "legacy_quadrant"]
    gpp_resid = residualize(gpp_pair["gpp_anomaly_pct_2023_vs_baseline"], gpp_pair["ndvi_recovery_pct_2023"])
    partial_r, partial_n = corr_value(gpp_pair["chl_recovery_pct_2023"], gpp_resid, method="pearson")
    nep_gap, nep_p, nep_n = corr_test_value(site["legacy_gap_pct"], site["nep_anomaly_pct_2023_vs_baseline"], method="spearman")
    summary: dict[str, object] = {
        "analysis_scope": scope_name,
        "primary_gpp_metric": primary_metric,
        "primary_gpp_metric_column": GPP_METRICS[primary_metric]["column"],
        "n_sites_total": int(site["site_id"].nunique()),
        "n_sites_with_gpp_flux": int(len(gpp_flux)),
        "n_sites_with_pairwise_gpp_pct": int(len(gpp_pair)),
        "n_sites_with_nep_flux": int(np.isfinite(site["nep_anomaly_pct_2023_vs_baseline"]).sum()),
        "recovery_threshold_pct": float(threshold),
        "legacy_quadrant_n_all": int((site["quadrant"] == "legacy_quadrant").sum()),
        "legacy_quadrant_n_pairwise_gpp_pct": int(len(legacy_pair)),
        "nonlegacy_n_pairwise_gpp_pct": int(len(nonlegacy_pair)),
        "legacy_gpp_anomaly_median_pct": (
            float(np.nanmedian(legacy_pair["gpp_anomaly_pct_2023_vs_baseline"])) if len(legacy_pair) else math.nan
        ),
        "nonlegacy_gpp_anomaly_median_pct": (
            float(np.nanmedian(nonlegacy_pair["gpp_anomaly_pct_2023_vs_baseline"])) if len(nonlegacy_pair) else math.nan
        ),
        "legacy_minus_nonlegacy_gpp_anomaly_pct": (
            float(
                np.nanmedian(legacy_pair["gpp_anomaly_pct_2023_vs_baseline"])
                - np.nanmedian(nonlegacy_pair["gpp_anomaly_pct_2023_vs_baseline"])
            )
            if len(legacy_pair) and len(nonlegacy_pair)
            else math.nan
        ),
        "partial_pearson_chl_vs_gpp_anomaly_given_ndvi": partial_r,
        "partial_n": partial_n,
        "spearman_legacy_gap_vs_nep_anomaly": nep_gap,
        "spearman_legacy_gap_vs_nep_anomaly_p": nep_p,
        "corr_n_legacy_gap_vs_nep": nep_n,
    }
    for metric_key, spec in GPP_METRICS.items():
        add_metric_association(summary, site, metric_key, spec["column"])

    # Backwards-compatible aliases for older prose/notes.
    raw_key = "gpp_pct_2023_vs_baseline"
    summary["spearman_legacy_gap_vs_gpp_anomaly"] = summary.get(f"{raw_key}_spearman_legacy_gap", math.nan)
    summary["spearman_legacy_gap_vs_gpp_anomaly_p"] = summary.get(f"{raw_key}_spearman_legacy_gap_p", math.nan)
    summary["pearson_legacy_gap_vs_gpp_anomaly"] = summary.get(f"{raw_key}_pearson_legacy_gap", math.nan)
    summary["corr_n_legacy_gap_vs_gpp"] = summary.get(f"{raw_key}_spearman_legacy_gap_n", 0)
    summary["spearman_chl_recovery_vs_gpp_anomaly"] = summary.get(f"{raw_key}_spearman_chl_recovery", math.nan)
    summary["spearman_chl_recovery_vs_gpp_anomaly_p"] = summary.get(f"{raw_key}_spearman_chl_recovery_p", math.nan)
    summary["corr_n_chl_vs_gpp"] = summary.get(f"{raw_key}_spearman_chl_recovery_n", 0)
    summary["spearman_ndvi_recovery_vs_gpp_anomaly"] = summary.get(f"{raw_key}_spearman_ndvi_recovery", math.nan)
    summary["spearman_ndvi_recovery_vs_gpp_anomaly_p"] = summary.get(f"{raw_key}_spearman_ndvi_recovery_p", math.nan)
    summary["corr_n_ndvi_vs_gpp"] = summary.get(f"{raw_key}_spearman_ndvi_recovery_n", 0)
    return summary


def marker_map(categories: list[str]) -> dict[str, str]:
    markers = ["o", "^", "s", "D", "P", "X", "v", "<", ">"]
    preferred = {
        "Evergreen forest": "^",
        "Deciduous forest": "s",
        "Mixed forest": "D",
        "Tree cover": "o",
        "Grassland": "P",
        "Cropland": "X",
        "Wetland": "v",
        "Open/savanna": "<",
        "Unclassified": "o",
    }
    out: dict[str, str] = {}
    for idx, cat in enumerate(categories):
        out[cat] = preferred.get(cat, markers[idx % len(markers)])
    return out


def metric_value_fmt(value: float, metric_key: str) -> str:
    if not np.isfinite(value):
        return "NA"
    if metric_key.endswith("_baseline"):
        return f"{value:+.1f}%"
    return f"{value:+.2f}"


def load_pixel_chlorophyll_recovery(
    pred_path: Path,
    site_ids: list[int],
    threshold: float,
) -> pd.DataFrame:
    if not pred_path.exists():
        return pd.DataFrame()
    cols = ["site_id", "date", "y", "x", "pixel_uid", "hrl_tcd", "trait", "prediction", "valid_obs_count"]
    pred = pd.read_parquet(pred_path, columns=cols)
    pred = pred[(pred["site_id"].astype(int).isin([int(v) for v in site_ids])) & (pred["trait"] == "Chlorophyll")].copy()
    if pred.empty:
        return pd.DataFrame()
    pred["date"] = pd.to_datetime(pred["date"])
    rows: list[dict[str, object]] = []
    for keys, sub in pred.groupby(["site_id", "pixel_uid", "y", "x"], sort=True):
        site_id, uid, y, x = keys
        baseline = pd.to_numeric(
            sub[(sub["date"] >= BASELINE_START) & (sub["date"] <= BASELINE_END)]["prediction"],
            errors="coerce",
        )
        trough = pd.to_numeric(
            sub[(sub["date"] >= TROUGH_START) & (sub["date"] <= TROUGH_END)]["prediction"],
            errors="coerce",
        )
        recovery = pd.to_numeric(
            sub[(sub["date"] >= RECOVERY_2023_START) & (sub["date"] <= RECOVERY_2023_END)]["prediction"],
            errors="coerce",
        )
        baseline_med = float(np.nanmedian(baseline)) if np.isfinite(baseline).any() else math.nan
        trough_med = float(np.nanmedian(trough)) if np.isfinite(trough).any() else math.nan
        recovery_med = float(np.nanmedian(recovery)) if np.isfinite(recovery).any() else math.nan
        rec_raw, rec_clip, interpretable = recovery_score(recovery_med, baseline_med, trough_med, min_relative_magnitude=0.02)
        rows.append(
            {
                "site_id": int(site_id),
                "pixel_uid": str(uid),
                "y": int(y),
                "x": int(x),
                "chl_baseline_median": baseline_med,
                "chl_trough_median": trough_med,
                "chl_recovery_value_2023": recovery_med,
                "chl_recovery_pct_raw_2023": rec_raw,
                "chl_recovery_pct_2023": rec_clip,
                "chl_recovery_interpretable": bool(interpretable),
                "chl_legacy_pixel": bool(np.isfinite(rec_clip) and rec_clip < threshold),
            }
        )
    return pd.DataFrame(rows)


def build_pixel_ndvi_timeseries(
    pred_path: Path,
    site_ids: list[int],
    out_path: Path,
    force: bool = False,
) -> pd.DataFrame:
    if out_path.exists() and not force:
        cached = pd.read_csv(out_path, parse_dates=["date"])
        cached = cached[cached["site_id"].astype(int).isin([int(v) for v in site_ids])].copy()
        if not cached.empty:
            return cached
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("xarray is required to build pixel-level Sentinel-2 NDVI support.") from exc

    pred_cols = ["site_id", "date", "y", "x", "pixel_uid", "trait"]
    pred = pd.read_parquet(pred_path, columns=pred_cols)
    pred = pred[(pred["site_id"].astype(int).isin([int(v) for v in site_ids])) & (pred["trait"] == "Chlorophyll")].copy()
    if pred.empty:
        return pd.DataFrame()
    pred["date"] = pd.to_datetime(pred["date"])
    rows: list[dict[str, object]] = []
    for site_id, site_pred in pred.groupby("site_id", sort=True):
        site_id = int(site_id)
        pixels = site_pred[["site_id", "y", "x", "pixel_uid"]].drop_duplicates().sort_values(["y", "x"]).copy()
        dates = pd.DatetimeIndex([d for d in sorted(site_pred["date"].dropna().unique()) if in_recovery_windows(pd.Timestamp(d))])
        if pixels.empty or dates.empty:
            continue
        print(f"[pixel-ndvi] site={site_id} pixels={len(pixels)} recovery-window dates={len(dates)}", flush=True)
        y_values = pixels["y"].to_numpy(dtype=np.int32)
        x_values = pixels["x"].to_numpy(dtype=np.int32)
        pixel_records = pixels.to_dict("records")
        for year, year_dates in pd.Series(dates).groupby(pd.Series(dates).dt.year):
            s2_path = S2_ROOT / f"site_{site_id}" / f"site_{site_id}_s2_region_{int(year)}.zarr"
            if not s2_path.exists():
                print(f"[warn] missing Sentinel-2 zarr for pixel NDVI site={site_id} year={int(year)}: {s2_path}", flush=True)
                continue
            ds = xr.open_zarr(s2_path, consolidated=False)
            source_times = ds.time.values.astype("datetime64[ns]")
            y10_min, y10_max = int(y_values.min() * 3), int(y_values.max() * 3 + 3)
            x10_min, x10_max = int(x_values.min() * 3), int(x_values.max() * 3 + 3)
            y20_values = np.array([nearest_20m_index(int(v), int(ds.sizes["y_20m"])) for v in y_values], dtype=np.int32)
            x20_values = np.array([nearest_20m_index(int(v), int(ds.sizes["x_20m"])) for v in x_values], dtype=np.int32)
            y20_min, y20_max = int(y20_values.min()), int(y20_values.max() + 1)
            x20_min, x20_max = int(x20_values.min()), int(x20_values.max() + 1)

            for center_date in pd.DatetimeIndex(year_dates):
                trait_time = np.datetime64(center_date, "ns")
                delta_all = ((source_times - trait_time) / np.timedelta64(1, "D")).astype(np.float32)
                time_indices = np.where(np.abs(delta_all) <= S2_WINDOW_DAYS)[0]
                if time_indices.size == 0:
                    for pix in pixel_records:
                        rows.append(
                            {
                                "site_id": site_id,
                                "date": pd.Timestamp(center_date),
                                "y": int(pix["y"]),
                                "x": int(pix["x"]),
                                "pixel_uid": str(pix["pixel_uid"]),
                                "ndvi": math.nan,
                                "n_valid_obs": 0,
                            }
                        )
                    continue

                refl10 = (
                    ds["reflectance_10m"]
                    .isel(time=time_indices.tolist(), y_10m=slice(y10_min, y10_max), x_10m=slice(x10_min, x10_max))
                    .transpose("time", "y_10m", "x_10m", "band_10m")
                    .values
                    .astype(np.float32, copy=False)
                )
                refl20 = (
                    ds["reflectance_20m"]
                    .isel(time=time_indices.tolist(), y_20m=slice(y20_min, y20_max), x_20m=slice(x20_min, x20_max))
                    .transpose("time", "y_20m", "x_20m", "band_20m")
                    .values
                    .astype(np.float32, copy=False)
                )
                adv = (
                    ds["advanced_valid_mask"]
                    .isel(time=time_indices.tolist(), y_20m=slice(y20_min, y20_max), x_20m=slice(x20_min, x20_max))
                    .transpose("time", "y_20m", "x_20m")
                    .values
                )

                for pix, y20, x20 in zip(pixel_records, y20_values, x20_values, strict=False):
                    y = int(pix["y"])
                    x = int(pix["x"])
                    y10_rel = int(y * 3 - y10_min)
                    x10_rel = int(x * 3 - x10_min)
                    sub10 = refl10[:, y10_rel : y10_rel + 3, x10_rel : x10_rel + 3, :]
                    valid_sub = np.isfinite(sub10).all(axis=-1)
                    masked10 = np.where(valid_sub[..., None], sub10, 0.0).astype(np.float32, copy=False)
                    valid_counts_10m = valid_sub.sum(axis=(1, 2)).astype(np.float32, copy=False)
                    mean10 = np.divide(
                        masked10.sum(axis=(1, 2), dtype=np.float32),
                        valid_counts_10m[:, None],
                        out=np.full((len(time_indices), masked10.shape[-1]), np.nan, dtype=np.float32),
                        where=valid_counts_10m[:, None] > 0,
                    )
                    red = mean10[:, 2]
                    nir = mean10[:, 3]
                    ndvi_values = safe_divide(nir - red, nir + red)
                    pix20 = refl20[:, int(y20 - y20_min), int(x20 - x20_min), :]
                    pix_adv = adv[:, int(y20 - y20_min), int(x20 - x20_min)]
                    valid_obs = (
                        valid_sub.any(axis=(1, 2))
                        & np.isfinite(pix20).all(axis=1)
                        & (pix_adv > 0)
                        & np.isfinite(ndvi_values)
                    )
                    vals = ndvi_values[valid_obs].astype(float)
                    rows.append(
                        {
                            "site_id": site_id,
                            "date": pd.Timestamp(center_date),
                            "y": y,
                            "x": x,
                            "pixel_uid": str(pix["pixel_uid"]),
                            "ndvi": float(np.nanmedian(vals)) if vals.size else math.nan,
                            "n_valid_obs": int(vals.size),
                        }
                    )
            ds.close()

    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def build_pixel_ndvi_recovery(ndvi_ts: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if ndvi_ts.empty:
        return pd.DataFrame()
    ndvi = ndvi_ts.copy()
    ndvi["date"] = pd.to_datetime(ndvi["date"])
    rows: list[dict[str, object]] = []
    for keys, sub in ndvi.groupby(["site_id", "pixel_uid", "y", "x"], sort=True):
        site_id, uid, y, x = keys
        baseline = pd.to_numeric(sub[(sub["date"] >= BASELINE_START) & (sub["date"] <= BASELINE_END)]["ndvi"], errors="coerce")
        trough = pd.to_numeric(sub[(sub["date"] >= TROUGH_START) & (sub["date"] <= TROUGH_END)]["ndvi"], errors="coerce")
        recovery = pd.to_numeric(
            sub[(sub["date"] >= RECOVERY_2023_START) & (sub["date"] <= RECOVERY_2023_END)]["ndvi"],
            errors="coerce",
        )
        baseline_med = float(np.nanmedian(baseline)) if np.isfinite(baseline).any() else math.nan
        trough_med = float(np.nanmedian(trough)) if np.isfinite(trough).any() else math.nan
        recovery_med = float(np.nanmedian(recovery)) if np.isfinite(recovery).any() else math.nan
        rec_raw, rec_clip, interpretable = recovery_score(recovery_med, baseline_med, trough_med, min_relative_magnitude=0.01)
        rows.append(
            {
                "site_id": int(site_id),
                "pixel_uid": str(uid),
                "y": int(y),
                "x": int(x),
                "ndvi_baseline_median": baseline_med,
                "ndvi_trough_median": trough_med,
                "ndvi_recovery_value_2023": recovery_med,
                "ndvi_recovery_pct_raw_2023": rec_raw,
                "ndvi_recovery_pct_2023": rec_clip,
                "ndvi_recovery_interpretable": bool(interpretable),
                "ndvi_recovered_pixel": bool(np.isfinite(rec_clip) and rec_clip >= threshold),
            }
        )
    return pd.DataFrame(rows)


def build_pixel_recovery_table(
    pred_path: Path,
    site: pd.DataFrame,
    site_ids: list[int],
    metric_key: str,
    threshold: float,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not site_ids:
        return pd.DataFrame(), {}
    chl = load_pixel_chlorophyll_recovery(pred_path, site_ids, threshold)
    if chl.empty:
        return pd.DataFrame(), {}
    ndvi_cache = out_dir / "pixel_ndvi_timeseries_forest_pairwise.csv"
    ndvi_ts = build_pixel_ndvi_timeseries(pred_path, site_ids, ndvi_cache, force=False)
    ndvi = build_pixel_ndvi_recovery(ndvi_ts, threshold)
    if chl.empty or ndvi.empty:
        return pd.DataFrame(), {}
    pixel = ndvi.merge(chl, on=["site_id", "pixel_uid", "y", "x"], how="inner")
    pixel["pixel_gap_pct_2023"] = (
        pd.to_numeric(pixel["ndvi_recovery_pct_2023"], errors="coerce")
        - pd.to_numeric(pixel["chl_recovery_pct_2023"], errors="coerce")
    )
    metric_col = GPP_METRICS[metric_key]["column"]
    enrich_cols = [
        "site_id",
        "site_name",
        "forest_type",
        "quadrant",
        "legacy_gap_pct",
        "driver_drought_exposed",
        "driver_swc_relative_drop",
        "ndvi_recovery_pct_2023",
        "chl_recovery_pct_2023",
        metric_col,
    ]
    site_enrich = site[[c for c in enrich_cols if c in site.columns]].drop_duplicates("site_id")
    pixel = pixel.merge(site_enrich, on="site_id", how="left", suffixes=("", "_site"))
    pixel["pixel_recovery_interpretable"] = (
        pixel["ndvi_recovery_interpretable"].astype(bool) & pixel["chl_recovery_interpretable"].astype(bool)
    )
    pixel["pixel_legacy_quadrant"] = (
        pixel["pixel_recovery_interpretable"]
        & (pd.to_numeric(pixel["ndvi_recovery_pct_2023"], errors="coerce") >= threshold)
        & (pd.to_numeric(pixel["chl_recovery_pct_2023"], errors="coerce") < threshold)
    )
    desc = pixel[pixel["pixel_recovery_interpretable"]].copy()
    legacy_n = int(pixel.loc[desc.index, "pixel_legacy_quadrant"].sum()) if not desc.empty else 0
    summary = {
        "pixel_scope_site_n": int(pd.Series(site_ids).nunique()),
        "pixel_total_n": int(len(pixel)),
        "pixel_interpretable_n": int(len(desc)),
        "pixel_legacy_quadrant_n": legacy_n,
        "pixel_legacy_quadrant_fraction": float(legacy_n / len(desc)) if len(desc) else math.nan,
        "pixel_gap_median": float(np.nanmedian(desc["pixel_gap_pct_2023"])) if len(desc) else math.nan,
        "pixel_gap_positive_fraction": (
            float(np.mean(pd.to_numeric(desc["pixel_gap_pct_2023"], errors="coerce") > 0)) if len(desc) else math.nan
        ),
        "pixel_ndvi_source": str(ndvi_cache),
        "pixel_flux_inference_guardrail": (
            "Pixel counts describe greenness-chlorophyll decoupling and within-site trait heterogeneity only; "
            "flux corroboration remains site-level because GPP is one footprint-integrated observation per site."
        ),
    }
    return pixel, summary


def plot_pixel_decoupling_hexbin(pixel: pd.DataFrame, summary: dict[str, object], out_dir: Path, threshold: float) -> list[str]:
    if pixel.empty:
        return []
    plot_df = pixel[pixel["pixel_recovery_interpretable"]].copy()
    plot_df = plot_df[
        np.isfinite(plot_df["ndvi_recovery_pct_2023"]) & np.isfinite(plot_df["chl_recovery_pct_2023"])
    ].copy()
    if plot_df.empty:
        return []
    fig, ax = plt.subplots(figsize=(4.2, 3.55))
    cmap = LinearSegmentedColormap.from_list("pixel_density", ["#F7F7F7", "#91BFDB", "#2C7FB8", "#08306B"])
    hb = ax.hexbin(
        plot_df["ndvi_recovery_pct_2023"],
        plot_df["chl_recovery_pct_2023"],
        gridsize=36,
        extent=(0, 100, 0, 100),
        mincnt=1,
        cmap=cmap,
        linewidths=0,
    )
    ax.axvspan(threshold, 100, ymin=0.0, ymax=threshold / 100.0, facecolor="#F4E7CB", alpha=0.40, lw=0)
    ax.axvline(threshold, color="#777777", lw=0.65, ls=":")
    ax.axhline(threshold, color="#777777", lw=0.65, ls=":")
    ax.axline((0, 0), slope=1, color="#BBBBBB", lw=0.5, ls="--", zorder=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Pixel NDVI recovery in 2023 (%)")
    ax.set_ylabel("Pixel chlorophyll recovery in 2023 (%)")
    ax.text(threshold + 2, 7, "legacy pixels", fontsize=7, ha="left", va="bottom", color="#5C4033")
    legacy_pct = 100.0 * float(summary.get("pixel_legacy_quadrant_fraction", math.nan))
    note = (
        f"{summary.get('pixel_scope_site_n', 0)} forest flux sites\n"
        f"interpretable pixels n={summary.get('pixel_interpretable_n', 0):,}\n"
        f"legacy pixels={legacy_pct:.1f}%\n"
        "descriptive; no pixel p value"
    )
    ax.text(
        0.03,
        0.97,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=2.0),
    )
    cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_label("Pixels per hexagon")
    cbar.ax.tick_params(labelsize=7, width=0.5, length=3)
    style_axis(ax)
    fig.tight_layout()
    return save_figure(fig, out_dir, "fig_pixel_greenness_chlorophyll_decoupling_hexbin")


def plot_pixel_gap_violin(
    pixel: pd.DataFrame,
    out_dir: Path,
    threshold: float,
    gpp_metric: str = PRIMARY_GPP_METRIC,
) -> list[str]:
    if pixel.empty:
        return []
    metric_col = GPP_METRICS[gpp_metric]["column"]
    plot_df = pixel[pixel["pixel_recovery_interpretable"] & np.isfinite(pixel["pixel_gap_pct_2023"])].copy()
    plot_df = plot_df[np.isfinite(plot_df[metric_col])].copy()
    if plot_df.empty:
        return []
    site_order = (
        plot_df[["site_id", "site_name", metric_col]]
        .drop_duplicates("site_id")
        .sort_values(metric_col, ascending=True)["site_id"]
        .astype(int)
        .tolist()
    )
    data = [
        pd.to_numeric(plot_df.loc[plot_df["site_id"].astype(int) == site_id, "pixel_gap_pct_2023"], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
        for site_id in site_order
    ]
    site_metric = plot_df[["site_id", metric_col]].drop_duplicates("site_id").set_index("site_id")[metric_col].astype(float)
    site_names = plot_df[["site_id", "site_name"]].drop_duplicates("site_id").set_index("site_id")["site_name"].astype(str)
    site_gap = plot_df[["site_id", "legacy_gap_pct"]].drop_duplicates("site_id").set_index("site_id")["legacy_gap_pct"].astype(float)
    site_exposed = (
        plot_df[["site_id", "driver_drought_exposed"]].drop_duplicates("site_id").set_index("site_id")["driver_drought_exposed"].astype(bool)
        if "driver_drought_exposed" in plot_df.columns
        else pd.Series(dtype=bool)
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    cmap = LinearSegmentedColormap.from_list("suppressed_recovered", ["#D55E00", "#F7F7F7", "#0072B2"])
    finite_vals = site_metric[np.isfinite(site_metric)]
    vmax = max(1.0, min(3.0, float(np.nanpercentile(np.abs(finite_vals), 85)))) if len(finite_vals) else 2.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    ax.axhspan(0, 105, color="#F4E7CB", lw=0, alpha=0.28, zorder=0)
    parts = ax.violinplot(data, positions=np.arange(1, len(site_order) + 1), widths=0.72, showextrema=False)
    for body, site_id in zip(parts["bodies"], site_order):
        color = cmap(norm(float(site_metric.loc[site_id]))) if site_id in site_metric.index else "#BBBBBB"
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_linewidth(0.35)
        body.set_alpha(0.82)
    medians = [float(np.nanmedian(v)) if len(v) else math.nan for v in data]
    ax.scatter(
        np.arange(1, len(site_order) + 1),
        medians,
        s=18,
        color="white",
        edgecolor="black",
        linewidth=0.45,
        zorder=3,
        label="pixel median",
    )
    site_gap_values = [float(site_gap.loc[site_id]) if site_id in site_gap.index else math.nan for site_id in site_order]
    ax.scatter(
        np.arange(1, len(site_order) + 1),
        site_gap_values,
        s=23,
        marker="D",
        color="#222222",
        edgecolor="white",
        linewidth=0.35,
        zorder=4,
        label="site-level gap",
    )
    if not site_exposed.empty:
        exposed_positions = [idx + 1 for idx, site_id in enumerate(site_order) if bool(site_exposed.get(site_id, False))]
        if exposed_positions:
            ax.scatter(
                exposed_positions,
                [-99.0] * len(exposed_positions),
                marker="^",
                s=18,
                color="#222222",
                linewidth=0.0,
                zorder=4,
                label="SWC-drop exposed",
            )
    ax.axhline(0, color="#333333", lw=0.65)
    ax.set_ylim(-105, 105)
    ax.set_ylabel("Pixel legacy gap: NDVI - chlorophyll recovery (%)")
    ax.set_xlabel("Forest flux sites sorted by GPP persistence anomaly")
    labels = [f"{sid}\n{site_names.loc[sid].split(',')[0][:10]}" if sid in site_names.index else str(sid) for sid in site_order]
    ax.set_xticks(np.arange(1, len(site_order) + 1), labels)
    ax.tick_params(axis="x", labelsize=6.4)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.018)
    cbar.set_label(GPP_METRICS[gpp_metric]["label"])
    cbar.ax.tick_params(labelsize=7, width=0.5, length=3)
    ax.legend(loc="lower left", frameon=False, fontsize=6.6, handletextpad=0.4, borderaxespad=0.2)
    style_axis(ax)
    fig.tight_layout()
    return save_figure(fig, out_dir, "fig_flux_corroborated_gap_pixel_violin")


def plot_quadrant(
    site: pd.DataFrame,
    summary: dict[str, object],
    out_dir: Path,
    threshold: float,
    gpp_metric: str = PRIMARY_GPP_METRIC,
    stem: str = "fig_flux_corroborated_trait_legacy_quadrant",
    scope_label: str = "All PFTs",
) -> list[str]:
    metric_col = GPP_METRICS[gpp_metric]["column"]
    metric_label = GPP_METRICS[gpp_metric]["label"]
    fig = plt.figure(figsize=(7.2, 5.7))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[4.7, 0.95, 0.28],
        height_ratios=[0.85, 4.2],
        left=0.10,
        right=0.96,
        bottom=0.25,
        top=0.92,
        hspace=0.04,
        wspace=0.10,
    )
    ax_histx = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_histx)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    cax = fig.add_subplot(gs[1, 2])

    cmap = LinearSegmentedColormap.from_list("suppressed_recovered", ["#D55E00", "#F7F7F7", "#0072B2"])
    vals = pd.to_numeric(site[metric_col], errors="coerce")
    finite_vals = vals[np.isfinite(vals)]
    if finite_vals.empty:
        vmax = 50.0 if gpp_metric == "gpp_pct_2023_vs_baseline" else 2.0
    else:
        vmax = float(np.nanpercentile(np.abs(finite_vals), 85))
        if gpp_metric == "gpp_pct_2023_vs_baseline":
            vmax = max(20.0, min(80.0, vmax))
        else:
            vmax = max(1.0, min(3.0, vmax))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    ax.axvspan(threshold, 100, ymin=0.0, ymax=threshold / 110.0, facecolor="#F4E7CB", alpha=0.55, lw=0)
    ax.axvline(threshold, color="#777777", lw=0.65, ls=":")
    ax.axhline(threshold, color="#777777", lw=0.65, ls=":")
    ax.axline((0, 0), slope=1, color="#BBBBBB", lw=0.5, ls="--", zorder=0)

    categories = [c for c in site["forest_type"].dropna().unique().tolist()]
    markers = marker_map(categories)
    for category in categories:
        sub = site[site["forest_type"] == category].copy()
        has = np.isfinite(sub[metric_col])
        if has.any():
            ax.scatter(
                sub.loc[has, "ndvi_recovery_pct_2023"],
                sub.loc[has, "chl_recovery_pct_2023"],
                c=sub.loc[has, metric_col],
                cmap=cmap,
                norm=norm,
                marker=markers[category],
                s=44,
                edgecolor="black",
                linewidth=0.35,
                alpha=0.95,
                zorder=3,
            )
        if (~has).any():
            ax.scatter(
                sub.loc[~has, "ndvi_recovery_pct_2023"],
                sub.loc[~has, "chl_recovery_pct_2023"],
                marker=markers[category],
                s=34,
                facecolor="#DDDDDD",
                edgecolor="#777777",
                linewidth=0.35,
                alpha=0.75,
                zorder=2,
            )
        uncertain = has & (~(sub["ndvi_recovery_interpretable"].astype(bool) & sub["chl_recovery_interpretable"].astype(bool)))
        if uncertain.any():
            ax.scatter(
                sub.loc[uncertain, "ndvi_recovery_pct_2023"],
                sub.loc[uncertain, "chl_recovery_pct_2023"],
                facecolors="none",
                edgecolors="#222222",
                marker=markers[category],
                s=72,
                linewidth=0.7,
                alpha=0.95,
                zorder=4,
            )

    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("NDVI recovery in 2023 (%)")
    ax.set_ylabel("Chlorophyll recovery in 2023 (%)")
    ax.text(
        threshold + 2,
        8,
        "legacy quadrant",
        fontsize=7,
        ha="left",
        va="bottom",
        color="#5C4033",
    )
    note = (
        f"{scope_label}\n"
        f"site-level paired n={summary[f'{gpp_metric}_pairwise_n']} "
        f"({summary['n_sites_with_gpp_flux']} finite GPP)\n"
        f"rho={summary[f'{gpp_metric}_spearman_legacy_gap']:+.2f}, "
        f"p={format_p(summary[f'{gpp_metric}_spearman_legacy_gap_p'])}\n"
        f"legacy median {metric_value_fmt(summary[f'{gpp_metric}_legacy_median'], gpp_metric)}\n"
        f"other median {metric_value_fmt(summary[f'{gpp_metric}_nonlegacy_median'], gpp_metric)}; "
        f"U p={format_p(summary[f'{gpp_metric}_mann_whitney_p'])}"
    )
    ax.text(
        0.03,
        0.97,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=2.2),
    )
    style_axis(ax)

    bins = np.linspace(0, 100, 11)
    ax_histx.hist(site["ndvi_recovery_pct_2023"].dropna(), bins=bins, color="#BDBDBD", edgecolor="white", linewidth=0.4)
    ax_histx.axvline(threshold, color="#777777", lw=0.65, ls=":")
    ax_histx.set_ylabel("Sites")
    ax_histx.tick_params(axis="x", labelbottom=False)
    style_axis(ax_histx)

    ax_histy.hist(
        site["chl_recovery_pct_2023"].dropna(),
        bins=bins,
        orientation="horizontal",
        color="#BDBDBD",
        edgecolor="white",
        linewidth=0.4,
    )
    ax_histy.axhline(threshold, color="#777777", lw=0.65, ls=":")
    ax_histy.set_xlabel("Sites")
    ax_histy.tick_params(axis="y", labelleft=False)
    style_axis(ax_histy)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(metric_label)
    cbar.ax.tick_params(labelsize=7, width=0.5, length=3)

    handles = [
        Line2D(
            [0],
            [0],
            marker=markers[cat],
            color="black",
            linestyle="None",
            markerfacecolor="#DDDDDD",
            markeredgewidth=0.35,
            markersize=5.4,
            label=cat,
        )
        for cat in categories
    ]
    if (~(site["ndvi_recovery_interpretable"].astype(bool) & site["chl_recovery_interpretable"].astype(bool))).any():
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="#222222",
                linestyle="None",
                markerfacecolor="none",
                markeredgewidth=0.7,
                markersize=6.2,
                label="weak recovery denominator",
            )
        )
    fig.legend(
        handles=handles,
        title="PFT / land cover",
        loc="lower center",
        bbox_to_anchor=(0.49, 0.035),
        ncol=4,
        frameon=False,
        handletextpad=0.4,
        columnspacing=0.9,
    )
    fig.text(0.02, 0.965, "a", fontsize=8, fontweight="bold", ha="left", va="top")
    return save_figure(fig, out_dir, stem)


def plot_forest_composite(
    site: pd.DataFrame,
    summary: dict[str, object],
    clock: pd.DataFrame,
    out_dir: Path,
    threshold: float,
    gpp_metric: str = PRIMARY_GPP_METRIC,
) -> list[str]:
    metric_col = GPP_METRICS[gpp_metric]["column"]
    metric_label = GPP_METRICS[gpp_metric]["label"]
    fig = plt.figure(figsize=(7.2, 5.75))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.25, 0.95, 0.06],
        left=0.08,
        right=0.94,
        bottom=0.16,
        top=0.91,
        wspace=0.22,
    )
    ax = fig.add_subplot(gs[0, 0])
    clock_gs = gs[0, 1].subgridspec(3, 1, hspace=0.24)
    clock_axes = [fig.add_subplot(clock_gs[i, 0]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 2])

    cmap = LinearSegmentedColormap.from_list("suppressed_recovered", ["#D55E00", "#F7F7F7", "#0072B2"])
    vals = pd.to_numeric(site[metric_col], errors="coerce")
    finite_vals = vals[np.isfinite(vals)]
    if finite_vals.empty:
        vmax = 2.0
    else:
        vmax = float(np.nanpercentile(np.abs(finite_vals), 85))
        vmax = max(1.0, min(3.0, vmax))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    ax.axvspan(threshold, 100, ymin=0.0, ymax=threshold / 110.0, facecolor="#F4E7CB", alpha=0.55, lw=0)
    ax.axvline(threshold, color="#777777", lw=0.65, ls=":")
    ax.axhline(threshold, color="#777777", lw=0.65, ls=":")
    ax.axline((0, 0), slope=1, color="#BBBBBB", lw=0.5, ls="--", zorder=0)

    categories = [c for c in site["forest_type"].dropna().unique().tolist()]
    markers = marker_map(categories)
    for category in categories:
        sub = site[site["forest_type"] == category].copy()
        has = np.isfinite(sub[metric_col])
        if has.any():
            ax.scatter(
                sub.loc[has, "ndvi_recovery_pct_2023"],
                sub.loc[has, "chl_recovery_pct_2023"],
                c=sub.loc[has, metric_col],
                cmap=cmap,
                norm=norm,
                marker=markers[category],
                s=39,
                edgecolor="black",
                linewidth=0.35,
                alpha=0.95,
                zorder=3,
            )
        if (~has).any():
            ax.scatter(
                sub.loc[~has, "ndvi_recovery_pct_2023"],
                sub.loc[~has, "chl_recovery_pct_2023"],
                marker=markers[category],
                s=31,
                facecolor="#DDDDDD",
                edgecolor="#777777",
                linewidth=0.35,
                alpha=0.72,
                zorder=2,
            )
        uncertain = has & (~(sub["ndvi_recovery_interpretable"].astype(bool) & sub["chl_recovery_interpretable"].astype(bool)))
        if uncertain.any():
            ax.scatter(
                sub.loc[uncertain, "ndvi_recovery_pct_2023"],
                sub.loc[uncertain, "chl_recovery_pct_2023"],
                facecolors="none",
                edgecolors="#222222",
                marker=markers[category],
                s=66,
                linewidth=0.7,
                alpha=0.95,
                zorder=4,
            )

    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("NDVI recovery in 2023 (%)")
    ax.set_ylabel("Chlorophyll recovery in 2023 (%)")
    ax.text(threshold + 2, 7, "legacy quadrant", fontsize=7, ha="left", va="bottom", color="#5C4033")
    perm_p = summary.get(f"{gpp_metric}_permutation_spearman_p_two_sided", math.nan)
    ci_low = summary.get(f"{gpp_metric}_bootstrap_spearman_ci_low", math.nan)
    ci_high = summary.get(f"{gpp_metric}_bootstrap_spearman_ci_high", math.nan)
    ci_text = f"[{ci_low:+.2f}, {ci_high:+.2f}]" if np.isfinite(ci_low) and np.isfinite(ci_high) else "NA"
    note = (
        f"site-level paired n={summary[f'{gpp_metric}_pairwise_n']}\n"
        f"rho={summary[f'{gpp_metric}_spearman_legacy_gap']:+.2f}, "
        f"perm p={format_p(perm_p)}\n"
        f"bootstrap 95% CI {ci_text}\n"
        f"MW p={format_p(summary[f'{gpp_metric}_mann_whitney_p'])}"
    )
    ax.text(
        0.03,
        0.97,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=2.0),
    )
    style_axis(ax)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(metric_label)
    cbar.ax.tick_params(labelsize=7, width=0.5, length=3)

    colors = {"NDVI": "#333333", "Chlorophyll": "#009E73", "GPP": "#0072B2"}
    linestyles = {"NDVI": "--", "Chlorophyll": "-", "GPP": "-"}
    selected = [int(v) for v in clock["site_id"].dropna().unique()[:3]] if not clock.empty else []
    for idx, ax_clock in enumerate(clock_axes):
        if idx >= len(selected):
            ax_clock.set_visible(False)
            continue
        site_id = selected[idx]
        ax_clock.axvspan(TROUGH_START, TROUGH_END, color="#F4E7CB", lw=0, alpha=0.80)
        ax_clock.axhline(100, color="#333333", lw=0.50)
        ax_clock.axhline(0, color="#777777", lw=0.45, ls=":")
        for variable in ["NDVI", "Chlorophyll", "GPP"]:
            sub = clock[(clock["site_id"].astype(int) == site_id) & (clock["variable"] == variable)].sort_values("date")
            if sub.empty:
                continue
            ax_clock.plot(
                sub["date"],
                sub["recovery_pct_clipped"],
                color=colors[variable],
                ls=linestyles[variable],
                lw=0.95,
                marker="o",
                ms=1.8,
                label=variable,
            )
        meta = site[site["site_id"].astype(int) == site_id].iloc[0]
        ax_clock.set_title(f"site {site_id}, {meta['site_name']}", loc="left", fontsize=6.7, pad=1.0)
        ax_clock.set_ylim(-5, 105)
        ax_clock.set_ylabel("Recovery (%)", fontsize=6.7)
        ax_clock.tick_params(axis="both", labelsize=6.4)
        style_axis(ax_clock)
        if idx < len(selected) - 1:
            ax_clock.tick_params(axis="x", labelbottom=False)
    if selected:
        last_visible = clock_axes[len(selected) - 1]
        last_visible.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[4, 7, 10, 1, 4, 7]))
        last_visible.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        last_visible.set_xlim(pd.Timestamp("2022-04-01"), pd.Timestamp("2023-08-31"))
        clock_axes[0].legend(
            loc="upper center",
            bbox_to_anchor=(0.58, 1.28),
            frameon=False,
            ncol=3,
            handlelength=1.2,
            columnspacing=0.6,
            fontsize=6.5,
            borderaxespad=0.0,
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker=markers[cat],
            color="black",
            linestyle="None",
            markerfacecolor="#DDDDDD",
            markeredgewidth=0.35,
            markersize=5.0,
            label=cat,
        )
        for cat in categories
    ]
    if (~(site["ndvi_recovery_interpretable"].astype(bool) & site["chl_recovery_interpretable"].astype(bool))).any():
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="#222222",
                linestyle="None",
                markerfacecolor="none",
                markeredgewidth=0.7,
                markersize=5.8,
                label="weak recovery denominator",
            )
        )
    fig.legend(
        handles=handles,
        title="Forest-domain PFT",
        loc="lower center",
        bbox_to_anchor=(0.47, 0.02),
        ncol=3,
        frameon=False,
        handletextpad=0.4,
        columnspacing=0.8,
        fontsize=6.8,
        title_fontsize=7.0,
    )
    fig.text(0.02, 0.965, "a", fontsize=8, fontweight="bold", ha="left", va="top")
    fig.text(0.56, 0.965, "b", fontsize=8, fontweight="bold", ha="left", va="top")
    return save_figure(fig, out_dir, "fig_flux_corroborated_trait_legacy_quadrant")


def biweekly_flux_series(site_id: int, daily: pd.DataFrame, dates: pd.DatetimeIndex, var: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date in dates:
        start = pd.Timestamp(date) - pd.Timedelta(days=7)
        end = pd.Timestamp(date) + pd.Timedelta(days=6)
        value = window_median(daily, var, start, end)
        rows.append({"site_id": int(site_id), "date": pd.Timestamp(date), "variable": var, "value": value})
    out = pd.DataFrame(rows)
    baseline = window_median(out, "value", BASELINE_START, BASELINE_END)
    trough = window_median(out, "value", TROUGH_START, TROUGH_END)
    scores = [recovery_score(v, baseline, trough) for v in out["value"].astype(float)]
    out["baseline_median"] = baseline
    out["trough_median"] = trough
    out["recovery_pct"] = [s[0] for s in scores]
    out["recovery_pct_clipped"] = [s[1] for s in scores]
    out["recovery_interpretable"] = [s[2] for s in scores]
    return out


def select_clock_sites(site: pd.DataFrame, max_sites: int = 3, gpp_metric: str = PRIMARY_GPP_METRIC) -> list[int]:
    metric_col = GPP_METRICS[gpp_metric]["column"]
    sub = site[
        np.isfinite(site[metric_col])
        & np.isfinite(site["legacy_gap_pct"])
        & (site["quadrant"] == "legacy_quadrant")
    ].copy()
    forest_like = sub["forest_type"].astype(str).str.contains("forest|Tree cover", case=False, regex=True)
    if forest_like.any():
        sub = sub[forest_like].copy()
    corroborated_2023 = sub[
        (pd.to_numeric(sub[metric_col], errors="coerce") < 0)
        & (pd.to_numeric(sub.get("gpp_anomaly_z_2023_vs_interannual", np.nan), errors="coerce") < 0)
    ].copy()
    if len(corroborated_2023) >= max_sites:
        sub = corroborated_2023
    else:
        corroborated = sub[pd.to_numeric(sub[metric_col], errors="coerce") < 0].copy()
        if not corroborated.empty:
            sub = corroborated
    if sub.empty:
        sub = site[np.isfinite(site[metric_col]) & np.isfinite(site["legacy_gap_pct"])].copy()
    if sub.empty:
        return []
    sub["_quality"] = sub["ndvi_recovery_interpretable"].astype(bool).astype(int) + sub["chl_recovery_interpretable"].astype(bool).astype(int)
    sub["_score"] = 0.5 * sub["legacy_gap_pct"].astype(float) - 18.0 * sub[metric_col].astype(float) + 20.0 * sub["_quality"]
    picked: list[int] = []
    used_types: set[str] = set()
    for rec in sub.sort_values("_score", ascending=False).itertuples(index=False):
        forest_type = str(rec.forest_type)
        if forest_type in used_types and len(sub["forest_type"].unique()) >= max_sites:
            continue
        picked.append(int(rec.site_id))
        used_types.add(forest_type)
        if len(picked) >= max_sites:
            return picked
    for rec in sub.sort_values("_score", ascending=False).itertuples(index=False):
        sid = int(rec.site_id)
        if sid not in picked:
            picked.append(sid)
        if len(picked) >= max_sites:
            break
    return picked


def build_recovery_clock_table(
    recovery_path: Path,
    site: pd.DataFrame,
    daily_by_site: dict[int, pd.DataFrame],
    selected_sites: list[int],
) -> pd.DataFrame:
    recovery = pd.read_csv(recovery_path, parse_dates=["date"])
    dates = pd.DatetimeIndex(sorted(recovery["date"].dropna().unique()))
    rows: list[pd.DataFrame] = []
    for site_id in selected_sites:
        base = recovery[
            (recovery["site_id"].astype(int) == int(site_id))
            & (recovery["variable"].isin(["NDVI", "Chlorophyll"]))
        ].copy()
        base = base[
            [
                "site_id",
                "site_name",
                "date",
                "variable",
                "value",
                "recovery_pct",
                "recovery_pct_clipped",
                "recovery_interpretable",
            ]
        ]
        rows.append(base)
        if site_id in daily_by_site:
            gpp = biweekly_flux_series(site_id, daily_by_site[site_id], dates, "GPP_NT_VUT_REF")
            gpp["site_name"] = str(site.loc[site["site_id"] == site_id, "site_name"].iloc[0])
            gpp["variable"] = "GPP"
            rows.append(gpp[base.columns])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_recovery_clocks(clock: pd.DataFrame, site: pd.DataFrame, out_dir: Path) -> list[str]:
    if clock.empty:
        return []
    selected = [int(v) for v in clock["site_id"].dropna().unique()]
    fig, axes = plt.subplots(len(selected), 1, figsize=(7.2, max(2.0, 1.35 * len(selected))), sharex=True, sharey=True)
    if len(selected) == 1:
        axes = [axes]
    colors = {"NDVI": "#333333", "Chlorophyll": "#009E73", "GPP": "#0072B2"}
    linestyles = {"NDVI": "--", "Chlorophyll": "-", "GPP": "-"}
    labels = {"NDVI": "NDVI", "Chlorophyll": "Chlorophyll", "GPP": "GPP"}
    for ax, site_id in zip(axes, selected):
        ax.axvspan(TROUGH_START, TROUGH_END, color="#F4E7CB", lw=0, alpha=0.85)
        ax.axhline(100, color="#333333", lw=0.55)
        ax.axhline(0, color="#777777", lw=0.45, ls=":")
        for variable in ["NDVI", "Chlorophyll", "GPP"]:
            sub = clock[(clock["site_id"].astype(int) == site_id) & (clock["variable"] == variable)].sort_values("date")
            if sub.empty:
                continue
            ax.plot(
                sub["date"],
                sub["recovery_pct_clipped"],
                color=colors[variable],
                ls=linestyles[variable],
                lw=1.0,
                marker="o",
                ms=2.1,
                label=labels[variable],
            )
        meta = site[site["site_id"].astype(int) == site_id].iloc[0]
        title = f"site {site_id}, {meta['site_name']} ({meta['forest_type']})"
        ax.set_title(title, loc="left", fontsize=7)
        ax.set_ylim(-5, 105)
        ax.set_ylabel("Recovery (%)")
        style_axis(ax)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    axes[-1].set_xlim(pd.Timestamp("2022-04-01"), pd.Timestamp("2023-08-31"))
    axes[0].legend(loc="upper right", frameon=False, ncol=3, handlelength=1.4, columnspacing=0.9)
    fig.text(0.02, 0.975, "b", fontsize=8, fontweight="bold", ha="left", va="top")
    fig.tight_layout(rect=(0.03, 0.02, 0.99, 0.96))
    return save_figure(fig, out_dir, "fig_flux_corroborated_recovery_clocks")


def build_trait_space_table(recovery_path: Path, site: pd.DataFrame, forest_only: bool = False) -> pd.DataFrame:
    recovery = pd.read_csv(recovery_path, parse_dates=["date"])
    keep_stages = ["pre-event", "drought trough", "2023 recovery"]
    keep = recovery[
        recovery["stage"].isin(keep_stages)
        & recovery["variable"].isin(["Chlorophyll", "LMA"])
    ].copy()
    stage = (
        keep.groupby(["site_id", "site_name", "stage", "variable"], as_index=False)
        .agg(value=("value", "median"), recovery_pct=("recovery_pct_clipped", "median"))
    )
    wide = stage.pivot_table(index=["site_id", "site_name", "stage"], columns="variable", values="value").reset_index()
    wide = wide.merge(site[["site_id", "forest_type", "quadrant"]], on="site_id", how="left")
    if forest_only:
        wide = wide[wide["forest_type"].apply(is_forest_scope)].copy()
    order = {"pre-event": 0, "drought trough": 1, "2023 recovery": 2}
    wide["stage_order"] = wide["stage"].map(order).astype(int)
    return wide.sort_values(["site_id", "stage_order"])


def plot_trait_space_trajectories(traj: pd.DataFrame, out_dir: Path) -> list[str]:
    if traj.empty:
        return []
    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    categories = [c for c in traj["forest_type"].dropna().unique().tolist()]
    palette = {
        "Evergreen forest": "#0072B2",
        "Deciduous forest": "#D55E00",
        "Mixed forest": "#CC79A7",
        "Tree cover": "#666666",
        "Grassland": "#E69F00",
        "Cropland": "#56B4E9",
        "Wetland": "#009E73",
        "Open/savanna": "#999999",
        "Unclassified": "#333333",
    }
    markers = {"pre-event": "o", "drought trough": "^", "2023 recovery": "s"}
    for site_id, sub in traj.groupby("site_id"):
        sub = sub.sort_values("stage_order")
        if len(sub) < 2 or not {"Chlorophyll", "LMA"}.issubset(sub.columns):
            continue
        forest_type = str(sub["forest_type"].iloc[0])
        color = palette.get(forest_type, "#555555")
        x = pd.to_numeric(sub["Chlorophyll"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub["LMA"], errors="coerce").to_numpy(dtype=float)
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            continue
        ax.plot(x, y, color=color, lw=0.75, alpha=0.56, zorder=1)
        for i in range(len(x) - 1):
            ax.annotate(
                "",
                xy=(x[i + 1], y[i + 1]),
                xytext=(x[i], y[i]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.65, alpha=0.50, mutation_scale=6),
                zorder=1,
            )
        for stage, marker in markers.items():
            s = sub[sub["stage"] == stage]
            if s.empty:
                continue
            ax.scatter(
                s["Chlorophyll"],
                s["LMA"],
                marker=marker,
                s=22,
                color=color,
                edgecolor="black",
                linewidth=0.25,
                alpha=0.88,
                zorder=2,
            )
    ax.set_xlabel("Chlorophyll content")
    ax.set_ylabel("Leaf mass per area")
    style_axis(ax)
    stage_handles = [
        Line2D([0], [0], marker=m, linestyle="None", markerfacecolor="#BBBBBB", markeredgecolor="black", markersize=5, label=s)
        for s, m in markers.items()
    ]
    type_handles = [
        Line2D([0], [0], color=palette.get(cat, "#555555"), lw=1.2, label=cat)
        for cat in categories
    ]
    leg1 = ax.legend(handles=stage_handles, loc="upper left", frameon=False, title="Stage")
    ax.add_artist(leg1)
    ax.legend(handles=type_handles, loc="center left", bbox_to_anchor=(1.01, 0.45), frameon=False, title="PFT / land cover")
    fig.text(0.02, 0.975, "c", fontsize=8, fontweight="bold", ha="left", va="top")
    fig.tight_layout(rect=(0.03, 0.02, 0.78, 0.96))
    return save_figure(fig, out_dir, "fig_trait_space_trajectories_chl_lma")


# Figure A: multi-trait recovery-clock composite.
# Ordered to read as the recovery-phase "different clocks" story (structure and
# nutrient track greenness; pigment lags). Colours follow the Okabe-Ito palette
# used elsewhere in this script. The onset-leading EWT/LMA signal is a separate,
# softer claim shown in the trait-greenness bar panel, not here.
TRAIT_CLOCK_VARIABLES = ["NDVI", "LAI", "Nitrogen", "Chlorophyll", "EWT", "LMA"]
TRAIT_CLOCK_MIN_SITES = 5
TRAIT_CLOCK_STYLE: dict[str, dict[str, object]] = {
    "NDVI": {"color": "#333333", "ls": "--", "label": "NDVI (greenness)", "band": False, "z": 5},
    "LAI": {"color": "#009E73", "ls": "-", "label": "LAI (structure)", "band": False, "z": 3},
    "Nitrogen": {"color": "#56B4E9", "ls": "-", "label": "Nitrogen", "band": False, "z": 3},
    "Chlorophyll": {"color": "#D55E00", "ls": "-", "label": "Chlorophyll (pigment)", "band": True, "z": 4},
    "EWT": {"color": "#0072B2", "ls": "-", "label": "EWT (water)", "band": False, "z": 2},
    "LMA": {"color": "#CC79A7", "ls": "-", "label": "LMA (leaf economics)", "band": False, "z": 2},
}


def build_trait_recovery_clock_composite(
    recovery_path: Path,
    forest_site: pd.DataFrame,
    variables: list[str] | None = None,
    min_sites: int = TRAIT_CLOCK_MIN_SITES,
) -> pd.DataFrame:
    """Cross-site median recovery clock per trait for forest-domain sites.

    For each variable and biweekly date the composite is the cross-site median of
    the within-site recovery percentage (``recovery_pct_clipped``), taken over
    sites whose recovery denominator is interpretable. This is a single-event
    (2022) composite that reuses the within-2022 baseline-to-trough recovery
    scale, so it does not depend on a multi-year trait climatology and is robust
    to the current 2022--2023 trait-reconstruction window. Dates with fewer than
    ``min_sites`` interpretable sites are dropped.
    """
    variables = variables or TRAIT_CLOCK_VARIABLES
    recovery = pd.read_csv(recovery_path, parse_dates=["date"])
    forest_ids = {int(s) for s in forest_site["site_id"].astype(int).tolist()}
    sub = recovery[
        recovery["site_id"].astype(int).isin(forest_ids) & recovery["variable"].isin(variables)
    ].copy()
    sub["recovery_pct_clipped"] = pd.to_numeric(sub["recovery_pct_clipped"], errors="coerce")
    if "recovery_interpretable" in sub.columns:
        sub = sub[sub["recovery_interpretable"].astype(bool)]
    rows: list[dict[str, object]] = []
    for (variable, date), grp in sub.groupby(["variable", "date"]):
        vals = pd.to_numeric(grp["recovery_pct_clipped"], errors="coerce").dropna().astype(float)
        if len(vals) < int(min_sites):
            continue
        rows.append(
            {
                "variable": str(variable),
                "date": pd.Timestamp(date),
                "median_recovery": float(np.nanmedian(vals)),
                "q25_recovery": float(np.nanquantile(vals, 0.25)),
                "q75_recovery": float(np.nanquantile(vals, 0.75)),
                "n_sites": int(len(vals)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["variable", "date"]).reset_index(drop=True)


def plot_trait_recovery_clock_composite(composite: pd.DataFrame, out_dir: Path) -> list[str]:
    if composite.empty:
        return []
    present = set(composite["variable"].astype(str))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.axvspan(TROUGH_START, TROUGH_END, color="#F4E7CB", lw=0, alpha=0.85, zorder=0)
    ax.axhline(100, color="#333333", lw=0.55, zorder=1)
    ax.axhline(0, color="#777777", lw=0.45, ls=":", zorder=1)
    for variable in TRAIT_CLOCK_VARIABLES:
        if variable not in present:
            continue
        spec = TRAIT_CLOCK_STYLE[variable]
        sub = composite[composite["variable"].astype(str) == variable].sort_values("date")
        if sub.empty:
            continue
        if bool(spec["band"]):
            ax.fill_between(
                sub["date"],
                sub["q25_recovery"],
                sub["q75_recovery"],
                color=str(spec["color"]),
                alpha=0.16,
                lw=0,
                zorder=int(spec["z"]) - 1,
            )
        ax.plot(
            sub["date"],
            sub["median_recovery"],
            color=str(spec["color"]),
            ls=str(spec["ls"]),
            lw=1.3,
            marker="o",
            ms=2.3,
            label=str(spec["label"]),
            zorder=int(spec["z"]),
        )
    ax.set_ylim(-5, 108)
    ax.set_ylabel("Recovery toward 2022 pre-event baseline (%)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_xlim(pd.Timestamp("2022-04-01"), pd.Timestamp("2023-08-31"))
    style_axis(ax)
    ax.legend(loc="lower right", frameon=False, ncol=2, handlelength=1.6, columnspacing=1.1)
    fig.tight_layout()
    return save_figure(fig, out_dir, "fig_trait_recovery_clock_composite")


def write_figure_notes(
    out_dir: Path,
    summaries: dict[str, dict[str, object]],
    site: pd.DataFrame,
    primary_metric: str,
    pixel_summary: dict[str, object] | None = None,
) -> Path:
    path = out_dir / "figure_notes.md"
    forest = summaries["forest_main"]
    all_pft = summaries["all_pft_sensitivity"]
    strict = summaries["strict_forest_sensitivity"]
    drought = summaries.get("drought_exposed_forest_sensitivity", {})
    primary_label = GPP_METRICS[primary_metric]["label"]
    pixel_summary = pixel_summary or {}
    primary_perm_p = forest.get(f"{primary_metric}_permutation_spearman_p_two_sided", math.nan)
    primary_ci_low = forest.get(f"{primary_metric}_bootstrap_spearman_ci_low", math.nan)
    primary_ci_high = forest.get(f"{primary_metric}_bootstrap_spearman_ci_high", math.nan)
    primary_loo_min = forest.get(f"{primary_metric}_loo_min_rho", math.nan)
    primary_loo_max = forest.get(f"{primary_metric}_loo_max_rho", math.nan)
    pixel_legacy_pct = 100.0 * float(pixel_summary.get("pixel_legacy_quadrant_fraction", math.nan))
    counter = site[
        (site["site_id"].astype(int) == 44)
        & np.isfinite(site["gpp_anomaly_pct_2023_vs_baseline"])
    ]
    if counter.empty:
        counter_text = "- No predefined evergreen counterexample row was available in the regenerated table."
    else:
        rec = counter.iloc[0]
        counter_text = (
            f"- Evergreen counterexample check: site {int(rec['site_id'])} ({rec['site_name']}) has "
            f"2023 GPP anomaly {rec['gpp_anomaly_pct_2023_vs_baseline']:+.1f}% "
            f"({rec['gpp_anomaly_z_2023_vs_interannual']:+.2f} z) but primary persistence "
            f"{rec[primary_metric]:+.2f} z while falling in the legacy quadrant. "
            f"The site-level chlorophyll recovery used in the quadrant panel is clipped to {rec['chl_recovery_pct_2023']:.1f}% from raw "
            f"{rec['chl_recovery_pct_raw_2023']:+.1f}%, with "
            f"`chl_recovery_interpretable={bool(rec['chl_recovery_interpretable'])}`. The pixel violin uses only "
            "pixels with interpretable NDVI and chlorophyll denominators, so its pixel median can differ from the "
            "site-aggregate denominator flag. Treat this as a recovery-denominator/domain warning, not as a deterministic-law failure."
        )
    text = f"""# Flux-Corroborated Trait Legacy Figure Notes

Generated by:

```bash
python docs/paper0_perspective/scripts/make_flux_corroborated_trait_legacy_figure.py
```

## Data Contract

- Breadth set: 37 cap5000 sites from `cap5000_sites_recovery_timeline_2022_2023aug.csv`.
- Main corroboration scope: forest-domain PFTs `{", ".join(sorted(FOREST_SCOPE_TYPES))}`. Wetland sites are retained in the all-PFT sensitivity, not in the main forest-domain analysis, because their available IGBP/WorldCover labels are wetland/grassland rather than forest.
- Recovery definition: `(value_2023 - value_trough) / (value_preevent - value_trough) * 100`, clipped to 0-100% for plotting.
- Windows:
  - Pre-event baseline: {BASELINE_START.date()} to {BASELINE_END.date()}.
  - Drought trough: {TROUGH_START.date()} to {TROUGH_END.date()}.
  - 2023 recovery readout: {RECOVERY_2023_START.date()} to {RECOVERY_2023_END.date()}.
- Main flux color/statistic: `{primary_metric}` = {primary_label}. The z-score reference is the same day-of-year window across available non-2022/non-2023 years; the persistence metric averages the 2022 trough-window z-score and 2023 recovery-window z-score.
- Forest-domain flux coverage: {forest["n_sites_with_gpp_flux"]} sites have finite GPP; {forest[f"{primary_metric}_pairwise_n"]} have both a finite greenness--chlorophyll gap and the primary GPP metric.
- Drought-driver sensitivity: ERA5-Land soil-water exposure is flagged when the pre-event to trough SWC drop is at least {DROUGHT_SWC_RELATIVE_DROP_THRESHOLD:.0%} of pre-event SWC. This driver-only screen is independent of GPP and trait recovery outcomes.
- All-PFT sensitivity flux coverage: {all_pft["n_sites_with_gpp_flux"]} sites have finite GPP; {all_pft[f"{primary_metric}_pairwise_n"]} have pairwise primary-metric data.
- Marker shape: FLUXNET IGBP where available, otherwise WorldCover/PFT fallback. Open marker rings denote weak recovery denominators (`NDVI` or chlorophyll recovery not interpretable under the trait/index-specific baseline--trough contrast rule).
- Pixel data contract: pixel-level NDVI and chlorophyll recovery are used only for the greenness--chlorophyll decoupling distribution and within-site chlorophyll heterogeneity. They are not used to increase the effective flux sample size because each eddy-covariance GPP value is a site/footprint-integrated observation.

## Forest-Domain Main Readout

- Legacy quadrant threshold: NDVI recovery >= {forest["recovery_threshold_pct"]:.0f}% and chlorophyll recovery < {forest["recovery_threshold_pct"]:.0f}%.
- Legacy quadrant: {forest["legacy_quadrant_n_all"]}/{forest["n_sites_total"]} forest-domain sites.
- Primary Spearman: NDVI-minus-chlorophyll recovery gap vs `{primary_metric}` rho={forest[f"{primary_metric}_spearman_legacy_gap"]:+.2f}, p={format_p(forest[f"{primary_metric}_spearman_legacy_gap_p"])}, n={forest[f"{primary_metric}_spearman_legacy_gap_n"]}.
- Site bootstrap: rho median={forest.get(f"{primary_metric}_bootstrap_spearman_median", math.nan):+.2f}, 95% CI [{primary_ci_low:+.2f}, {primary_ci_high:+.2f}], valid replicates={forest.get(f"{primary_metric}_bootstrap_spearman_valid", 0)}.
- Site-label permutation: two-sided p={format_p(primary_perm_p)}; negative-direction p={format_p(forest.get(f"{primary_metric}_permutation_spearman_p_directional_negative", math.nan))}.
- Leave-one-out: rho range [{primary_loo_min:+.2f}, {primary_loo_max:+.2f}], all-negative={forest.get(f"{primary_metric}_loo_all_negative", False)}.
- Primary median separation is reported as underpowered: legacy median {metric_value_fmt(forest[f"{primary_metric}_legacy_median"], primary_metric)} vs other {metric_value_fmt(forest[f"{primary_metric}_nonlegacy_median"], primary_metric)}, Mann--Whitney U={forest[f"{primary_metric}_mann_whitney_u"]:.1f}, p={format_p(forest[f"{primary_metric}_mann_whitney_p"])}.
- Same-season 2023 GPP z-score sensitivity: legacy median {metric_value_fmt(forest["gpp_z_2023_vs_interannual_legacy_median"], "gpp_z_2023_vs_interannual")} vs other {metric_value_fmt(forest["gpp_z_2023_vs_interannual_nonlegacy_median"], "gpp_z_2023_vs_interannual")}, Mann--Whitney p={format_p(forest["gpp_z_2023_vs_interannual_mann_whitney_p"])}. This remains a sensitivity, not the headline.
- Raw 2023-vs-baseline percent sensitivity: rho={forest["gpp_pct_2023_vs_baseline_spearman_legacy_gap"]:+.2f}, p={format_p(forest["gpp_pct_2023_vs_baseline_spearman_legacy_gap_p"])}; legacy median {forest["legacy_gpp_anomaly_median_pct"]:+.1f}% vs other {forest["nonlegacy_gpp_anomaly_median_pct"]:+.1f}%, Mann--Whitney p={format_p(forest["gpp_pct_2023_vs_baseline_mann_whitney_p"])}.

## Pixel-Level Decoupling Readout

- Pixel display scope: {pixel_summary.get("pixel_scope_site_n", 0)} forest-domain pairwise flux sites.
- Pixel recovery rows: {pixel_summary.get("pixel_total_n", 0)} total, {pixel_summary.get("pixel_interpretable_n", 0)} interpretable for both axes.
- Pixel legacy-quadrant share: {pixel_summary.get("pixel_legacy_quadrant_n", 0)} pixels = {pixel_legacy_pct:.1f}% of interpretable pixels.
- Pixel gap median: {pixel_summary.get("pixel_gap_median", math.nan):+.1f} percentage points; positive-gap pixels: {100.0 * float(pixel_summary.get("pixel_gap_positive_fraction", math.nan)):.1f}% of interpretable pixels.
- This is a descriptive Claim-A display of greenness--chlorophyll decoupling. No flux association, correlation p value, or independent-observation claim is made at the pixel level.

This supports the narrower Perspective claim that, within the forest-domain flux subset, a greenness--chlorophyll recovery gap is associated with reduced independent GPP persistence. It is not framed as a deterministic quadrant law.

## Sensitivities

- Strict forest sensitivity (excluding open/savanna): primary rho={strict[f"{primary_metric}_spearman_legacy_gap"]:+.2f}, p={format_p(strict[f"{primary_metric}_spearman_legacy_gap_p"])}, n={strict[f"{primary_metric}_spearman_legacy_gap_n"]}; Mann--Whitney p={format_p(strict[f"{primary_metric}_mann_whitney_p"])}.
- Drought-exposed forest sensitivity (SWC drop >= {DROUGHT_SWC_RELATIVE_DROP_THRESHOLD:.0%}): primary rho={drought.get(f"{primary_metric}_spearman_legacy_gap", math.nan):+.2f}, p={format_p(drought.get(f"{primary_metric}_spearman_legacy_gap_p", math.nan))}, n={drought.get(f"{primary_metric}_spearman_legacy_gap_n", 0)}; permutation p={format_p(drought.get(f"{primary_metric}_permutation_spearman_p_two_sided", math.nan))}. This is reported as a scope diagnostic because the threshold is a driver screen, not a flux outcome.
- All-PFT sensitivity: primary rho={all_pft[f"{primary_metric}_spearman_legacy_gap"]:+.2f}, p={format_p(all_pft[f"{primary_metric}_spearman_legacy_gap_p"])}, n={all_pft[f"{primary_metric}_spearman_legacy_gap_n"]}; Mann--Whitney p={format_p(all_pft[f"{primary_metric}_mann_whitney_p"])}.
{counter_text}

## Boundary

Keep this as an illustrative, descriptive figure. Do not frame it as formal variance partitioning. The residual check remains a sensitivity only: Pearson correlation between chlorophyll recovery and raw GPP anomaly residual after linear NDVI adjustment is {forest["partial_pearson_chl_vs_gpp_anomaly_given_ndvi"]:+.2f} in the forest-domain subset. The safe manuscript wording is therefore forest-domain corroboration, with managed/non-forest land-cover types shown as sensitivity rather than folded into the main claim.

## Outputs

- Main forest-domain figure: `figures/generated/flux_corroborated_trait_legacy/fig_flux_corroborated_trait_legacy_quadrant.*`
- Pixel decoupling figure: `figures/generated/flux_corroborated_trait_legacy/fig_pixel_greenness_chlorophyll_decoupling_hexbin.*`
- Within-site legacy-gap figure: `figures/generated/flux_corroborated_trait_legacy/fig_flux_corroborated_gap_pixel_violin.*`
- Drought-exposed forest sensitivity figure: `figures/generated/flux_corroborated_trait_legacy/fig_flux_corroborated_trait_legacy_quadrant_drought_exposed.*`
- All-PFT supplementary figure: `figures/generated/flux_corroborated_trait_legacy/fig_flux_corroborated_trait_legacy_quadrant_all_pft.*`
- Recovery clocks candidate: `figures/generated/flux_corroborated_trait_legacy/fig_flux_corroborated_recovery_clocks.*`
- Chlorophyll-LMA trajectory candidate: `figures/generated/flux_corroborated_trait_legacy/fig_trait_space_trajectories_chl_lma.*`
- Source data: `tables/flux_corroborated_trait_legacy/flux_trait_legacy_site_metrics.csv`
- Pixel source data: `tables/flux_corroborated_trait_legacy/pixel_recovery_forest_pairwise.csv`
- Drought-driver source data: `tables/flux_corroborated_trait_legacy/flux_trait_legacy_drought_driver_exposure.csv`
- Leave-one-out source data: `tables/flux_corroborated_trait_legacy/flux_trait_legacy_site_level_leave_one_out.csv`
- Diagnostic summary: `tables/flux_corroborated_trait_legacy/flux_trait_legacy_diagnostic_summary.json`
"""
    path.write_text(text)
    return path


def main() -> None:
    args = parse_args()
    setup_style()
    args.output_table_dir.mkdir(parents=True, exist_ok=True)
    args.output_figure_dir.mkdir(parents=True, exist_ok=True)

    recovery_site = load_recovery_site_metrics(args.recovery_table, threshold=args.recovery_threshold)
    meta = load_site_metadata(args.unified_2022_table, args.fluxnet_mapping)
    site = recovery_site.merge(meta, on=["site_id"], how="left", suffixes=("", "_meta"))
    if "site_name_meta" in site.columns:
        site["site_name"] = site["site_name"].where(site["site_name"].notna(), site["site_name_meta"])
        site = site.drop(columns=["site_name_meta"])
    flux, daily_by_site = compute_flux_metrics(site[["site_id", "fluxnet_site_id"]].drop_duplicates(), args.fluxnet_zip_dir)
    site = site.merge(flux.drop(columns=["fluxnet_site_id"], errors="ignore"), on="site_id", how="left")
    drought_driver = build_drought_driver_table(args.recovery_table)
    site = site.merge(drought_driver, on="site_id", how="left")
    site["quadrant"] = site.apply(
        lambda r: quadrant_label(r["ndvi_recovery_pct_2023"], r["chl_recovery_pct_2023"], args.recovery_threshold),
        axis=1,
    )
    site["forest_scope"] = site["forest_type"].apply(is_forest_scope)
    site["strict_forest_scope"] = site["forest_type"].apply(is_strict_forest_scope)
    site["recovery_denominator_weak"] = ~(
        site["ndvi_recovery_interpretable"].astype(bool) & site["chl_recovery_interpretable"].astype(bool)
    )
    forest_site = site[site["forest_scope"]].copy()
    strict_forest_site = site[site["strict_forest_scope"]].copy()
    drought_exposed_forest_site = forest_site[forest_site["driver_drought_exposed"].astype(bool)].copy()
    summaries = {
        "forest_main": build_summary(
            forest_site,
            threshold=args.recovery_threshold,
            scope_name="forest_main",
            primary_metric=args.primary_gpp_metric,
        ),
        "strict_forest_sensitivity": build_summary(
            strict_forest_site,
            threshold=args.recovery_threshold,
            scope_name="strict_forest_sensitivity",
            primary_metric=args.primary_gpp_metric,
        ),
        "drought_exposed_forest_sensitivity": build_summary(
            drought_exposed_forest_site,
            threshold=args.recovery_threshold,
            scope_name="drought_exposed_forest_sensitivity",
            primary_metric=args.primary_gpp_metric,
        ),
        "all_pft_sensitivity": build_summary(
            site,
            threshold=args.recovery_threshold,
            scope_name="all_pft_sensitivity",
            primary_metric=args.primary_gpp_metric,
        ),
    }
    forest_pair = paired_site_data(forest_site, args.primary_gpp_metric)
    summaries["forest_main"].update(site_level_resampling_summary(forest_pair, args.primary_gpp_metric))
    loo = leave_one_out_table(forest_pair, args.primary_gpp_metric)
    add_leave_one_out_summary(summaries["forest_main"], loo, args.primary_gpp_metric)
    drought_pair = paired_site_data(drought_exposed_forest_site, args.primary_gpp_metric)
    summaries["drought_exposed_forest_sensitivity"].update(
        site_level_resampling_summary(drought_pair, args.primary_gpp_metric)
    )

    site_path = args.output_table_dir / "flux_trait_legacy_site_metrics.csv"
    summary_path = args.output_table_dir / "flux_trait_legacy_diagnostic_summary.json"
    summary_csv_path = args.output_table_dir / "flux_trait_legacy_diagnostic_summary.csv"
    loo_path = args.output_table_dir / "flux_trait_legacy_site_level_leave_one_out.csv"
    drought_driver_path = args.output_table_dir / "flux_trait_legacy_drought_driver_exposure.csv"
    site.to_csv(site_path, index=False)
    loo.to_csv(loo_path, index=False)
    site[
        [
            "site_id",
            "site_name",
            "forest_type",
            "forest_scope",
            "driver_swc_pre_event",
            "driver_swc_trough",
            "driver_swc_drop",
            "driver_swc_relative_drop",
            "driver_tmax_delta_trough_minus_pre",
            "driver_precip_delta_trough_minus_pre",
            "driver_drought_exposed",
            "quadrant",
            "legacy_gap_pct",
            args.primary_gpp_metric,
        ]
    ].to_csv(drought_driver_path, index=False)
    summary_path.write_text(json.dumps(summaries, indent=2, allow_nan=True) + "\n")
    pd.DataFrame(list(summaries.values())).to_csv(summary_csv_path, index=False)

    figure_outputs: list[str] = []
    pixel, pixel_summary = build_pixel_recovery_table(
        args.pixel_trait_predictions,
        forest_site,
        forest_pair["site_id"].astype(int).tolist(),
        args.primary_gpp_metric,
        args.recovery_threshold,
        args.output_table_dir,
    )
    pixel_path = args.output_table_dir / "pixel_recovery_forest_pairwise.csv"
    if not pixel.empty:
        pixel.to_csv(pixel_path, index=False)
        summaries["forest_main"].update(pixel_summary)
        summary_path.write_text(json.dumps(summaries, indent=2, allow_nan=True) + "\n")
        pd.DataFrame(list(summaries.values())).to_csv(summary_csv_path, index=False)
        figure_outputs.extend(plot_pixel_decoupling_hexbin(pixel, pixel_summary, args.output_figure_dir, args.recovery_threshold))
        figure_outputs.extend(plot_pixel_gap_violin(pixel, args.output_figure_dir, args.recovery_threshold, args.primary_gpp_metric))
    selected = select_clock_sites(forest_site, gpp_metric=args.primary_gpp_metric)
    clock = build_recovery_clock_table(args.recovery_table, forest_site, daily_by_site, selected)
    clock_path = args.output_table_dir / "flux_recovery_clock_source_data.csv"
    clock.to_csv(clock_path, index=False)
    figure_outputs.extend(
        plot_forest_composite(
            forest_site,
            summaries["forest_main"],
            clock,
            args.output_figure_dir,
            threshold=args.recovery_threshold,
            gpp_metric=args.primary_gpp_metric,
        )
    )
    figure_outputs.extend(
        plot_quadrant(
            drought_exposed_forest_site,
            summaries["drought_exposed_forest_sensitivity"],
            args.output_figure_dir,
            threshold=args.recovery_threshold,
            gpp_metric=args.primary_gpp_metric,
            stem="fig_flux_corroborated_trait_legacy_quadrant_drought_exposed",
            scope_label="Drought-exposed forest sensitivity",
        )
    )
    figure_outputs.extend(
        plot_quadrant(
            site,
            summaries["all_pft_sensitivity"],
            args.output_figure_dir,
            threshold=args.recovery_threshold,
            gpp_metric=args.primary_gpp_metric,
            stem="fig_flux_corroborated_trait_legacy_quadrant_all_pft",
            scope_label="All PFT sensitivity",
        )
    )
    figure_outputs.extend(plot_recovery_clocks(clock, site, args.output_figure_dir))

    traj = build_trait_space_table(args.recovery_table, site, forest_only=True)
    traj_path = args.output_table_dir / "trait_space_chl_lma_trajectory_source_data.csv"
    traj.to_csv(traj_path, index=False)
    figure_outputs.extend(plot_trait_space_trajectories(traj, args.output_figure_dir))

    clock_composite = build_trait_recovery_clock_composite(args.recovery_table, forest_site)
    clock_composite_path = args.output_table_dir / "trait_recovery_clock_composite_source_data.csv"
    clock_composite.to_csv(clock_composite_path, index=False)
    figure_outputs.extend(plot_trait_recovery_clock_composite(clock_composite, args.output_figure_dir))

    run_summary = {
        "site_metrics": str(site_path),
        "diagnostic_summary_json": str(summary_path),
        "diagnostic_summary_csv": str(summary_csv_path),
        "leave_one_out_source_data": str(loo_path),
        "drought_driver_source_data": str(drought_driver_path),
        "pixel_source_data": str(pixel_path) if not pixel.empty else "",
        "clock_source_data": str(clock_path),
        "clock_composite_source_data": str(clock_composite_path),
        "trajectory_source_data": str(traj_path),
        "figures": figure_outputs,
        "selected_clock_sites": selected,
        "primary_gpp_metric": args.primary_gpp_metric,
        "site_level_resampling": {
            "bootstrap_replicates": N_BOOTSTRAP,
            "permutation_replicates": N_PERMUTATIONS,
            "seed": RESAMPLING_SEED,
        },
        "pixel_summary": pixel_summary,
        "forest_scope_types": sorted(FOREST_SCOPE_TYPES),
        "windows": {
            "baseline": [str(BASELINE_START.date()), str(BASELINE_END.date())],
            "trough": [str(TROUGH_START.date()), str(TROUGH_END.date())],
            "recovery_2023": [str(RECOVERY_2023_START.date()), str(RECOVERY_2023_END.date())],
        },
    }
    notes_path = write_figure_notes(args.output_table_dir, summaries, site, args.primary_gpp_metric, pixel_summary=pixel_summary)
    run_summary["figure_notes"] = str(notes_path)
    run_summary_path = args.output_table_dir / "run_summary.json"
    run_summary_path.write_text(json.dumps(run_summary, indent=2, allow_nan=True) + "\n")
    print(json.dumps(run_summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
