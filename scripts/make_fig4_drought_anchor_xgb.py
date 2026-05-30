#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import joblib
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio
from rasterio.transform import rowcol
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

PAPER_ROOT = Path(__file__).resolve().parents[1]
ECOSPECTRA_ROOT = Path("/home/chaonan.ji/data/projects/ecospectra")
TRAITS_ROOT = ECOSPECTRA_ROOT / "models" / "traits_model"
if str(TRAITS_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(TRAITS_ROOT / "scripts"))

from stage2_models.dev_region_dataset import load_split
from stage2_models.trait_centered_dataset import load_trait_centered_partitions

BASE_SCRIPT = PAPER_ROOT / "scripts" / "make_fig3_site43_44.py"
TABLE_DIR = PAPER_ROOT / "tables" / "fig4_drought_anchor_xgb"
FIG_DIR = PAPER_ROOT / "figures" / "generated"
MODEL_CACHE = TABLE_DIR / "model_cache"
_FEATURE_PIXEL_TABLE: pd.DataFrame | None = None

TASK_MANIFEST = (
    ECOSPECTRA_ROOT
    / "output_data/traits_model/outputs/ablations/technical_report_all20_main_cap5000_20260522"
    / "trait_relation_task_manifest_xgb_multioutput.csv"
)

ANCHOR_SITE_IDS = [43, 44]
ANCHOR_SITE_NAMES = {43: "Le Bray", 44: "Bilos"}
ANALYSIS_START = pd.Timestamp("2022-01-08")
ANALYSIS_END = pd.Timestamp("2023-08-19")
ANALYSIS_TAG = "2022_2023aug"
EVENT_DATE = pd.Timestamp("2022-08-06")
BASELINE_START = pd.Timestamp("2022-05-14")
BASELINE_END = pd.Timestamp("2022-06-25")
TROUGH_START = pd.Timestamp("2022-07-23")
TROUGH_END = pd.Timestamp("2022-08-20")
RECOVERY_START = pd.Timestamp("2022-09-17")
RECOVERY_END = pd.Timestamp("2022-10-29")
TRUE_RECOVERY_START = pd.Timestamp("2023-07-08")
TRUE_RECOVERY_END = pd.Timestamp("2023-08-19")

TRAITS = [
    ("EWT", "EWT (mg/cm²)", "EWT", "#0072B2"),
    ("Chlorophyll", "Chl content (μg/cm²)", "Chlorophyll", "#009E73"),
    ("LAI", "LAI (m²/m²)", "LAI", "#CC79A7"),
    ("Nitrogen", "N content (mg/cm²)", "Nitrogen", "#D55E00"),
    ("LMA", "LMA (g/m²)", "LMA", "#E69F00"),
]

RECOVERY_VARIABLES = [
    ("EWT", "EWT", "#0072B2", "-"),
    ("Chlorophyll", "Chlorophyll", "#009E73", "-"),
    ("LAI", "LAI", "#CC79A7", "-"),
    ("Nitrogen", "Nitrogen", "#D55E00", "-"),
    ("LMA", "LMA", "#E69F00", "-"),
    ("NDVI", "NDVI", "#333333", "--"),
    ("EVI", "EVI", "#999999", "--"),
    ("NDWI", "NDWI", "#56B4E9", "--"),
    ("NBR", "NBR", "#882255", "--"),
    ("CIRE", "CIRE", "#44AA99", "--"),
    ("NDRE", "NDRE", "#117733", "--"),
    ("MSI", "MSI", "#AA4499", "--"),
    ("SWC", "SWC", "#777777", ":"),
    ("Tair", "Tair", "#CC6677", ":"),
    ("Tmax", "Tmax", "#AA3377", ":"),
    ("P", "Precip.", "#6699CC", ":"),
    ("SWdown", "SWdown", "#DDCC77", ":"),
]

PANEL_A_CONTEXT_VARIABLES = [
    ("NDVI", "NDVI_median", "NDVI"),
    ("EVI", "EVI_median", "EVI"),
    ("NDWI", "NDWI_median", "NDWI"),
    ("NBR", "NBR_median", "NBR"),
    ("CIRE", "CIRE_median", "CIRE"),
    ("NDRE", "NDRE_median", "NDRE"),
    ("MSI", "MSI_median", "MSI"),
    ("SWC", "volumetric_soil_water_layer_1", "SWC"),
    ("Tair", "2m_temperature_C", "Tair"),
    ("Tmax", "2m_temperature_max_C", "Tmax"),
    ("P", "total_precipitation_mm", "Precip."),
    ("SWdown", "surface_solar_radiation_downwards_MJ_m2_day", "SWdown"),
]

XGB_PARAMS = {
    "EWT": dict(n_estimators=500, learning_rate=0.03, max_depth=10, min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=0.0001),
    "Chlorophyll": dict(n_estimators=500, learning_rate=0.03, max_depth=10, min_child_weight=10, subsample=0.8, colsample_bytree=0.8, reg_lambda=0.0001),
    "LAI": dict(n_estimators=500, learning_rate=0.03, max_depth=10, min_child_weight=20, subsample=0.6, colsample_bytree=1.0, reg_lambda=10.0),
    "Nitrogen": dict(n_estimators=500, learning_rate=0.03, max_depth=10, min_child_weight=20, subsample=1.0, colsample_bytree=0.6, reg_lambda=0.01),
    "LMA": dict(n_estimators=500, learning_rate=0.03, max_depth=10, min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=0.0001),
}


def _load_base_module():
    spec = importlib.util.spec_from_file_location("paper0_fig3_site43_44", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base_module()


def _site_name_map() -> dict[int, str]:
    table = ECOSPECTRA_ROOT / "input_data" / "fluxnet_sites.csv"
    if not table.exists():
        return dict(ANCHOR_SITE_NAMES)
    df = pd.read_csv(table)
    out = {int(row.site_id): str(row.site_name) for row in df.itertuples(index=False)}
    out.update(ANCHOR_SITE_NAMES)
    return out


SITE_NAMES = _site_name_map()


def site_label(site_id: int) -> str:
    return SITE_NAMES.get(int(site_id), f"site {int(site_id)}")


def cap5000_site_ids() -> list[int]:
    row = _manifest_row()
    df = pd.read_parquet(Path(row["feature_table"]), columns=["site_id"])
    return sorted(int(v) for v in pd.Series(df["site_id"]).dropna().unique())


def _feature_pixel_table() -> pd.DataFrame:
    global _FEATURE_PIXEL_TABLE
    if _FEATURE_PIXEL_TABLE is None:
        row = _manifest_row()
        _FEATURE_PIXEL_TABLE = pd.read_parquet(Path(row["feature_table"]), columns=["site_id", "y", "x"])
    return _FEATURE_PIXEL_TABLE


def _sample_bank_pixels(site_id: int, max_pixels: int) -> pd.DataFrame:
    df = _feature_pixel_table()
    pix = df[df["site_id"].astype(int) == int(site_id)][["site_id", "y", "x"]].drop_duplicates().copy()
    if pix.empty:
        return pd.DataFrame(columns=["site_id", "y", "x", "hrl_tcd"])
    try:
        center_y, center_x = _site_center_index(site_id)
        dist2 = np.square(pix["y"].astype(float) - float(center_y)) + np.square(pix["x"].astype(float) - float(center_x))
        pix = pix.assign(_dist2=dist2).sort_values(["_dist2", "y", "x"]).drop(columns=["_dist2"])
    except Exception:
        pix = pix.sort_values(["y", "x"])
    if len(pix) > int(max_pixels):
        pix = pix.head(int(max_pixels)).sort_values(["y", "x"])
    pix["site_id"] = int(site_id)
    pix["y"] = pix["y"].astype(int)
    pix["x"] = pix["x"].astype(int)
    pix["hrl_tcd"] = math.nan
    return pix.reset_index(drop=True)


def _limit_pixels_to_nearest(site_id: int, pixels: pd.DataFrame, max_pixels: int) -> pd.DataFrame:
    if int(max_pixels) <= 0 or len(pixels) <= int(max_pixels):
        return pixels.reset_index(drop=True)
    try:
        center_y, center_x = _site_center_index(site_id)
        dist2 = np.square(pixels["y"].astype(float) - float(center_y)) + np.square(pixels["x"].astype(float) - float(center_x))
        return (
            pixels.assign(_dist2=dist2)
            .sort_values(["_dist2", "y", "x"])
            .head(int(max_pixels))
            .drop(columns=["_dist2"])
            .sort_values(["y", "x"])
            .reset_index(drop=True)
        )
    except Exception:
        return pixels.sort_values(["y", "x"]).head(int(max_pixels)).reset_index(drop=True)


def _output_prefix(site_ids: list[int], scope: str) -> str:
    if scope == "anchor" or site_ids == ANCHOR_SITE_IDS:
        return "site43_44"
    if scope == "cap5000":
        return "cap5000_sites"
    joined = "_".join(str(v) for v in site_ids[:6])
    suffix = "" if len(site_ids) <= 6 else f"_plus{len(site_ids) - 6}"
    return f"sites_{joined}{suffix}"


def setup_style() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.grid": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="both", labelsize=7, width=0.6, length=3)
    ax.grid(False)


def panel_label(ax, letter: str, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold", color="black")


def save_figure(fig, stem: str) -> None:
    for ext in ("pdf", "svg", "eps"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _manifest_row() -> pd.Series:
    df = pd.read_csv(TASK_MANIFEST)
    if df.empty:
        raise ValueError(f"Empty task manifest: {TASK_MANIFEST}")
    return df.iloc[0]


def _fit_x_scaler(x: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(np.asarray(x, dtype=np.float32)).astype(np.float32, copy=False)
    return scaler, x_scaled


def _transform_x(scaler: StandardScaler, x: np.ndarray) -> np.ndarray:
    return scaler.transform(np.asarray(x, dtype=np.float32)).astype(np.float32, copy=False)


def _biweekly_dates_between(start: pd.Timestamp = ANALYSIS_START, end: pd.Timestamp = ANALYSIS_END) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="14D")


def _meta_row(site_id: int, center_date: pd.Timestamp, pix: dict[str, object]) -> dict[str, object]:
    return {
        "site_id": int(site_id),
        "site_label": f"site {int(site_id)}, {site_label(int(site_id))}",
        "date": pd.Timestamp(center_date),
        "year": int(pd.Timestamp(center_date).year),
        "y": int(pix["y"]),
        "x": int(pix["x"]),
        "pixel_uid": f"site_{int(site_id)}_y_{int(pix['y'])}_x_{int(pix['x'])}",
        "hrl_tcd": float(pix["hrl_tcd"]),
    }


def _site_center_index(site_id: int) -> tuple[int, int]:
    sites = pd.read_csv(ECOSPECTRA_ROOT / "input_data" / "fluxnet_sites.csv").set_index("site_id")
    lat = float(sites.loc[int(site_id), "location_lat"])
    lon = float(sites.loc[int(site_id), "location_lon"])
    tcd_path = base.HRL_TCD_ROOT / f"site_{int(site_id)}_hrl2021_tcd_30m.tif"
    with rasterio.open(tcd_path) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x_utm, y_utm = transformer.transform(lon, lat)
        row, col = rowcol(src.transform, x_utm, y_utm)
        row = int(np.clip(row, 0, src.height - 1))
        col = int(np.clip(col, 0, src.width - 1))
    return row, col


def _forest_window(site_id: int, window_pixels: int) -> pd.DataFrame:
    center_y, center_x = _site_center_index(site_id)
    tcd_path = base.HRL_TCD_ROOT / f"site_{int(site_id)}_hrl2021_tcd_30m.tif"
    with rasterio.open(tcd_path) as src:
        y_idx = base._window_indices(center_y, window_pixels, src.height)
        x_idx = base._window_indices(center_x, window_pixels, src.width)
        tcd = src.read(1, window=((int(y_idx[0]), int(y_idx[-1]) + 1), (int(x_idx[0]), int(x_idx[-1]) + 1)))
    yy, xx = np.meshgrid(y_idx, x_idx, indexing="ij")
    keep = (tcd >= 30) & (tcd != 255)
    return pd.DataFrame(
        {
            "site_id": int(site_id),
            "y": yy[keep].astype(int),
            "x": xx[keep].astype(int),
            "hrl_tcd": tcd[keep].astype(float),
        }
    )


def _low_obs_weights(x: np.ndarray, feature_names: list[str], threshold: int = 3, low_weight: float = 0.5) -> np.ndarray:
    mask_idx = [i for i, name in enumerate(feature_names) if str(name).startswith("mask__")]
    if not mask_idx:
        return np.ones(x.shape[0], dtype=np.float32)
    obs = np.asarray(x[:, mask_idx], dtype=np.float32).sum(axis=1)
    return np.where(obs < float(threshold), float(low_weight), 1.0).astype(np.float32)


def load_training_payload(train_fraction: float = 1.0) -> dict[str, object]:
    row = _manifest_row()
    split = load_split(Path(row["split_path"]), str(row["split_name"]))
    return load_trait_centered_partitions(
        Path(row["feature_table"]),
        split,
        random_seed=int(row.get("seed", 42)),
        train_fraction=float(train_fraction),
        pooled_training=True,
        drop_absolute_doy=True,
        tabular_mode="hybrid",
        input_feature_profile=str(row["input_feature_profile"]),
        sample_metadata_profile="none",
    )


def train_or_load_models(force: bool = False, train_fraction: float = 1.0, n_jobs: int = 12) -> dict[str, object]:
    models_path = MODEL_CACHE / "fig4_xgb_5trait_models.joblib"
    if models_path.exists() and not force:
        return joblib.load(models_path)

    payload = load_training_payload(train_fraction=train_fraction)
    trait_names = [str(v) for v in payload["trait_names"]]
    feature_names = [str(v) for v in payload["feature_names"]]
    x_train = np.asarray(payload["x_train"], dtype=np.float32)
    y_train = np.asarray(payload["y_train"], dtype=np.float32)
    mask_train = np.asarray(payload["label_mask_train"], dtype=np.float32)
    x_scaler, x_scaled = _fit_x_scaler(x_train)
    weights = _low_obs_weights(x_train, feature_names)

    models = {}
    y_scalers = {}
    for short, full_name, _label, _color in TRAITS:
        if full_name not in trait_names:
            raise ValueError(f"Missing trait in training payload: {full_name}")
        idx = trait_names.index(full_name)
        valid = (mask_train[:, idx] > 0) & np.isfinite(y_train[:, idx])
        if int(valid.sum()) < 10:
            raise ValueError(f"Too few valid rows for {short}: {int(valid.sum())}")
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y_train[valid, idx].reshape(-1, 1)).reshape(-1)
        params = XGB_PARAMS[short]
        model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42,
            n_jobs=int(n_jobs),
            tree_method="hist",
            verbosity=0,
            **params,
        )
        print(f"[fit] {short} rows={int(valid.sum())} params={params}", flush=True)
        model.fit(x_scaled[valid], y_scaled, sample_weight=weights[valid])
        models[short] = model
        y_scalers[short] = y_scaler

    out = {
        "models": models,
        "y_scalers": y_scalers,
        "x_scaler": x_scaler,
        "feature_names": feature_names,
        "trait_names": trait_names,
        "train_fraction": float(train_fraction),
        "xgb_params": XGB_PARAMS,
        "task_manifest": str(TASK_MANIFEST),
    }
    joblib.dump(out, models_path)
    return out


def _build_site_sequences_recovery(
    site_id: int,
    window_pixels: int,
    smoke: bool = False,
    max_pixels_per_site: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        pixels = _forest_window(site_id, window_pixels=window_pixels)
    except Exception as exc:
        print(f"[warn] tower forest window failed for site={site_id}: {exc}; using cap5000 sample-bank pixels", flush=True)
        pixels = pd.DataFrame()
    if (not smoke) and len(pixels) < 25:
        fallback_max = int(max_pixels_per_site) if int(max_pixels_per_site) > 0 else max(25, int(window_pixels) * int(window_pixels))
        fallback = _sample_bank_pixels(site_id, max_pixels=max(25, fallback_max))
        if not fallback.empty:
            print(
                f"[warn] site={site_id} tower forest pixels n={len(pixels)}; using cap5000 sample-bank pixels n={len(fallback)}",
                flush=True,
            )
            pixels = fallback
    if (not smoke) and int(max_pixels_per_site) > 0 and len(pixels) > int(max_pixels_per_site):
        before = len(pixels)
        pixels = _limit_pixels_to_nearest(site_id, pixels, int(max_pixels_per_site))
        print(f"[site] site={site_id} pixel cap {before}->{len(pixels)}", flush=True)
    if smoke:
        pixels = pixels.head(min(25, len(pixels))).copy()
        dates = pd.DatetimeIndex([pd.Timestamp("2022-07-09"), pd.Timestamp("2023-08-05")])
    else:
        dates = _biweekly_dates_between()
    if pixels.empty:
        raise ValueError(f"No HRL forest pixels found in tower window for site {site_id}")

    y_values = pixels["y"].to_numpy(dtype=np.int32)
    x_values = pixels["x"].to_numpy(dtype=np.int32)
    pixel_records = pixels.reset_index(drop=True).to_dict("records")
    meta_rows: list[dict[str, object]] = []
    flat_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []

    for year, year_dates in pd.Series(dates).groupby(pd.Series(dates).dt.year):
        s2_path = base.S2_ROOT / f"site_{site_id}" / f"site_{site_id}_s2_region_{int(year)}.zarr"
        if not s2_path.exists():
            print(f"[warn] missing Sentinel-2 zarr for site={site_id} year={int(year)}: {s2_path}", flush=True)
            continue
        ds = base.xr.open_zarr(s2_path, consolidated=False)
        source_times = ds.time.values.astype("datetime64[ns]")
        y10_min, y10_max = int(y_values.min() * 3), int(y_values.max() * 3 + 3)
        x10_min, x10_max = int(x_values.min() * 3), int(x_values.max() * 3 + 3)
        y20_values = np.array([base._nearest_20m_index(int(v), int(ds.sizes["y_20m"])) for v in y_values], dtype=np.int32)
        x20_values = np.array([base._nearest_20m_index(int(v), int(ds.sizes["x_20m"])) for v in x_values], dtype=np.int32)
        y20_min, y20_max = int(y20_values.min()), int(y20_values.max() + 1)
        x20_min, x20_max = int(x20_values.min()), int(x20_values.max() + 1)

        for center_date in pd.DatetimeIndex(year_dates):
            trait_time = np.datetime64(center_date, "ns")
            delta_all = ((source_times - trait_time) / np.timedelta64(1, "D")).astype(np.float32)
            time_indices = np.where(np.abs(delta_all) <= 28.0)[0]
            context_payload: dict[str, object] = {"site_id": int(site_id), "date": pd.Timestamp(center_date)}
            if time_indices.size == 0:
                for name in base.SPECTRAL_INDICES:
                    context_payload[f"{name}_median"] = math.nan
                context_rows.append(context_payload)
                for pix in pixel_records:
                    seq, mask = base._make_empty_sequence()
                    meta = _meta_row(site_id, center_date, pix)
                    meta_rows.append(meta)
                    flat_rows.append({**meta, **base._flatten_sequence(seq, mask)})
                continue

            refl10 = (
                ds["reflectance_10m"]
                .isel(time=time_indices.tolist(), y_10m=slice(y10_min, y10_max), x_10m=slice(x10_min, x10_max))
                .transpose("time", "y_10m", "x_10m", "band_10m")
                .values
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
            selected_times = source_times[time_indices]
            selected_delta = delta_all[time_indices]
            context_values: dict[str, list[float]] = {name: [] for name in base.SPECTRAL_INDICES}

            for pix, y20, x20 in zip(pixel_records, y20_values, x20_values, strict=False):
                y = int(pix["y"])
                x = int(pix["x"])
                y10_rel = int(y * 3 - y10_min)
                x10_rel = int(x * 3 - x10_min)
                sub10 = refl10[:, y10_rel : y10_rel + 3, x10_rel : x10_rel + 3, :]
                valid_sub = np.isfinite(sub10).all(axis=-1)
                masked10 = np.where(valid_sub[..., None], sub10, 0.0).astype(np.float32, copy=False)
                valid_counts = valid_sub.sum(axis=(1, 2)).astype(np.float32, copy=False)
                mean10 = np.divide(
                    masked10.sum(axis=(1, 2), dtype=np.float32),
                    valid_counts[:, None],
                    out=np.zeros((len(time_indices), masked10.shape[-1]), dtype=np.float32),
                    where=valid_counts[:, None] > 0,
                )
                pix20 = refl20[:, int(y20 - y20_min), int(x20 - x20_min), :]
                pix_adv = adv[:, int(y20 - y20_min), int(x20 - x20_min)]
                valid_obs = valid_sub.any(axis=(1, 2)) & np.isfinite(pix20).all(axis=1) & (pix_adv > 0)
                bands = np.concatenate([mean10, pix20], axis=1).astype(np.float32, copy=False)
                indices = base._spectral_indices(bands)
                records: list[dict[str, float]] = []
                for obs_idx in np.where(valid_obs)[0].tolist():
                    ts = pd.Timestamp(selected_times[obs_idx])
                    doy = float(ts.dayofyear)
                    angle = 2.0 * np.pi * doy / 365.0
                    rec = {
                        "delta_days": float(selected_delta[obs_idx]),
                        "doy_sin": float(np.sin(angle)),
                        "doy_cos": float(np.cos(angle)),
                        "weight": base._window_weight(float(selected_delta[obs_idx]), core_days=14, window_days=28),
                    }
                    for band_idx, band_name in enumerate(base.SEQ_CHANNEL_NAMES[:10]):
                        rec[band_name] = float(bands[obs_idx, band_idx])
                    for index_name in base.SPECTRAL_INDICES:
                        value = float(indices[index_name][obs_idx])
                        rec[index_name] = value
                        if np.isfinite(value):
                            context_values[index_name].append(value)
                    records.append(rec)
                selected = base._select_sequence_records(records, max_len=8, core_days=14)
                seq, mask = base._records_to_sequence(selected, max_len=8)
                meta = _meta_row(site_id, center_date, pix)
                meta_rows.append(meta)
                flat_rows.append({**meta, **base._flatten_sequence(seq, mask)})

            for name in base.SPECTRAL_INDICES:
                vals = np.asarray(context_values[name], dtype=float)
                context_payload[f"{name}_median"] = float(np.nanmedian(vals)) if vals.size else math.nan
            context_rows.append(context_payload)
        ds.close()

    return pd.DataFrame(meta_rows), pd.DataFrame(flat_rows), pd.DataFrame(context_rows)


def _climate_context_recovery(site_id: int, dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year, year_dates in pd.Series(dates).groupby(pd.Series(dates).dt.year):
        path = base.ERA5_ROOT / f"site_{site_id}" / f"site_{site_id}_yr{int(year)}.zarr"
        if not path.exists():
            print(f"[warn] missing ERA5-Land zarr for site={site_id} year={int(year)}: {path}", flush=True)
            continue
        ds = base.xr.open_zarr(path, consolidated=False)
        for date in pd.DatetimeIndex(year_dates):
            start = np.datetime64(date - pd.Timedelta(days=7), "ns")
            end = np.datetime64(date + pd.Timedelta(days=6), "ns")
            sub = ds.sel(time=slice(start, end))
            row: dict[str, object] = {"site_id": int(site_id), "date": pd.Timestamp(date)}
            if sub.sizes.get("time", 0) == 0:
                row["2m_temperature_C"] = math.nan
                row["2m_temperature_max_C"] = math.nan
                row["total_precipitation_mm"] = math.nan
                row["surface_solar_radiation_downwards_MJ_m2_day"] = math.nan
                row["volumetric_soil_water_layer_1"] = math.nan
                rows.append(row)
                continue
            temp = sub["2m_temperature"].values.astype(float)
            row["2m_temperature_C"] = float(np.nanmean(temp - 273.15 if np.nanmean(temp) > 100 else temp))
            if "2m_temperature_max" in sub:
                tmax = sub["2m_temperature_max"].values.astype(float)
                row["2m_temperature_max_C"] = float(np.nanmean(tmax - 273.15 if np.nanmean(tmax) > 100 else tmax))
            else:
                row["2m_temperature_max_C"] = row["2m_temperature_C"]
            precip = sub["total_precipitation"].values.astype(float)
            row["total_precipitation_mm"] = float(np.nansum(precip * 1000.0))
            rad = sub["surface_solar_radiation_downwards"].values.astype(float)
            row["surface_solar_radiation_downwards_MJ_m2_day"] = float(np.nanmean(rad / 1_000_000.0))
            soil = sub["volumetric_soil_water_layer_1"].values.astype(float)
            row["volumetric_soil_water_layer_1"] = float(np.nanmean(soil))
            rows.append(row)
        ds.close()
    return pd.DataFrame(rows)


def predict_sites(
    model_payload: dict[str, object],
    site_ids: list[int],
    output_prefix: str,
    force: bool = False,
    window_pixels: int = 30,
    smoke: bool = False,
    max_pixels_per_site: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    suffix = "_smoke" if smoke else ""
    pred_path = TABLE_DIR / f"{output_prefix}_xgb_biweekly_trait_predictions_{ANALYSIS_TAG}{suffix}.parquet"
    context_path = TABLE_DIR / f"{output_prefix}_xgb_biweekly_context_{ANALYSIS_TAG}{suffix}.csv"
    if pred_path.exists() and context_path.exists() and not force:
        return pd.read_parquet(pred_path), pd.read_csv(context_path, parse_dates=["date"])

    feature_names = list(model_payload["feature_names"])
    x_scaler = model_payload["x_scaler"]
    models = model_payload["models"]
    y_scalers = model_payload["y_scalers"]
    pred_frames: list[pd.DataFrame] = []
    context_frames: list[pd.DataFrame] = []
    site_cache = TABLE_DIR / "site_cache" / output_prefix
    site_cache.mkdir(parents=True, exist_ok=True)

    for site_id in site_ids:
        site_pred_path = site_cache / f"site_{int(site_id)}_predictions_{ANALYSIS_TAG}{suffix}.parquet"
        site_context_path = site_cache / f"site_{int(site_id)}_context_{ANALYSIS_TAG}{suffix}.csv"
        if site_pred_path.exists() and site_context_path.exists() and not force:
            pred_frames.append(pd.read_parquet(site_pred_path))
            context_frames.append(pd.read_csv(site_context_path, parse_dates=["date"]))
            print(f"[site] reuse cache site={site_id}", flush=True)
            continue
        print(f"[site] build sequences site={site_id}", flush=True)
        meta, flat, spectral_context = _build_site_sequences_recovery(
            site_id,
            window_pixels=window_pixels,
            smoke=smoke,
            max_pixels_per_site=max_pixels_per_site,
        )
        climate = _climate_context_recovery(site_id, pd.DatetimeIndex(sorted(pd.to_datetime(meta["date"]).unique())))
        context = spectral_context.merge(climate, on=["site_id", "date"], how="outer").sort_values("date")
        site_pred_frames: list[pd.DataFrame] = []
        x = flat.reindex(columns=feature_names).to_numpy(dtype=np.float32, copy=True)
        x_scaled = _transform_x(x_scaler, x)
        for short, full_name, label, _color in TRAITS:
            pred_scaled = np.asarray(models[short].predict(x_scaled), dtype=np.float64).reshape(-1, 1)
            pred = np.asarray(y_scalers[short].inverse_transform(pred_scaled), dtype=np.float64).reshape(-1)
            frame = meta.copy()
            frame["model"] = "xgb"
            frame["model_label"] = "XGBoost"
            frame["trait"] = short
            frame["trait_name"] = full_name
            frame["trait_label"] = label
            frame["prediction"] = pred
            frame["valid_obs_count"] = flat[[c for c in flat.columns if str(c).startswith("mask__")]].sum(axis=1).to_numpy(dtype=int)
            site_pred_frames.append(frame)
        site_pred = pd.concat(site_pred_frames, ignore_index=True)
        site_pred.to_parquet(site_pred_path, index=False)
        context.to_csv(site_context_path, index=False)
        pred_frames.append(site_pred)
        context_frames.append(context)

    pred = pd.concat(pred_frames, ignore_index=True)
    context = pd.concat(context_frames, ignore_index=True)
    pred.to_parquet(pred_path, index=False)
    context.to_csv(context_path, index=False)
    return pred, context


def summarize_predictions(pred: pd.DataFrame, output_prefix: str) -> pd.DataFrame:
    out = (
        pred.groupby(["site_id", "site_label", "date", "model", "model_label", "trait", "trait_name", "trait_label"], as_index=False)
        .agg(
            prediction_median=("prediction", "median"),
            prediction_p10=("prediction", lambda s: float(np.nanquantile(s, 0.10))),
            prediction_p90=("prediction", lambda s: float(np.nanquantile(s, 0.90))),
            n_pixels=("prediction", "size"),
            valid_obs_median=("valid_obs_count", "median"),
        )
        .sort_values(["site_id", "date", "trait"])
    )
    out.to_csv(TABLE_DIR / f"{output_prefix}_xgb_biweekly_trait_predictions_{ANALYSIS_TAG}_summary.csv", index=False)
    return out


def _stage(date: pd.Timestamp) -> str:
    if BASELINE_START <= date <= BASELINE_END:
        return "pre-event"
    if TROUGH_START <= date <= TROUGH_END:
        return "drought trough"
    if RECOVERY_START <= date <= RECOVERY_END:
        return "apparent recovery"
    if TRUE_RECOVERY_START <= date <= TRUE_RECOVERY_END:
        return "2023 recovery"
    return "other"


def _recovery_score(value: float, baseline: float, trough: float, min_relative_magnitude: float = 0.02) -> tuple[float, float, bool]:
    denom = float(baseline) - float(trough)
    if not (np.isfinite(value) and np.isfinite(baseline) and np.isfinite(trough)):
        return math.nan, math.nan, False
    scale = max(abs(float(baseline)), 1e-12)
    interpretable = abs(denom) / scale >= float(min_relative_magnitude)
    if abs(denom) <= 1e-12:
        return math.nan, math.nan, False
    pct = 100.0 * (float(value) - float(trough)) / denom
    return pct, float(np.clip(pct, 0.0, 100.0)), bool(interpretable)


def build_event_table(summary: pd.DataFrame, context: pd.DataFrame, output_prefix: str) -> pd.DataFrame:
    rows = []
    for site_id, site_df in summary.groupby("site_id"):
        for short, full_name, label, _color in TRAITS:
            s = site_df[site_df["trait"] == short].copy()
            baseline = s[(s["date"] >= BASELINE_START) & (s["date"] <= BASELINE_END)]["prediction_median"].astype(float)
            trough = s[(s["date"] >= TROUGH_START) & (s["date"] <= TROUGH_END)]["prediction_median"].astype(float)
            base_med = float(np.nanmedian(baseline))
            base_iqr = float(np.nanquantile(baseline, 0.75) - np.nanquantile(baseline, 0.25)) if len(baseline) else math.nan
            trough_med = float(np.nanmedian(trough)) if len(trough) else math.nan
            for rec in s.itertuples(index=False):
                value = float(rec.prediction_median)
                if np.isfinite(base_med) and abs(base_med) > 1e-12:
                    pct = 100.0 * (value - base_med) / abs(base_med)
                else:
                    pct = math.nan
                robust = (value - base_med) / base_iqr if np.isfinite(base_iqr) and abs(base_iqr) > 1e-12 else math.nan
                recovery_pct, recovery_pct_clipped, recovery_interpretable = _recovery_score(value, base_med, trough_med)
                rows.append(
                    {
                        "site_id": int(site_id),
                        "site_name": site_label(int(site_id)),
                        "date": pd.Timestamp(rec.date),
                        "relative_days": int((pd.Timestamp(rec.date) - EVENT_DATE).days),
                        "relative_biweek": int(round((pd.Timestamp(rec.date) - EVENT_DATE).days / 14.0)),
                        "stage": _stage(pd.Timestamp(rec.date)),
                        "trait": short,
                        "trait_name": full_name,
                        "trait_label": label,
                        "prediction_median": value,
                        "prediction_p10": float(rec.prediction_p10),
                        "prediction_p90": float(rec.prediction_p90),
                        "baseline_median": base_med,
                        "baseline_iqr": base_iqr,
                        "trough_median": trough_med,
                        "percent_departure": pct,
                        "robust_departure": robust,
                        "recovery_pct": recovery_pct,
                        "recovery_pct_clipped": recovery_pct_clipped,
                        "recovery_interpretable": recovery_interpretable,
                        "n_pixels": int(rec.n_pixels),
                        "valid_obs_median": float(rec.valid_obs_median),
                    }
                )
    event = pd.DataFrame(rows)
    event.to_csv(TABLE_DIR / f"{output_prefix}_xgb_event_aligned_trait_departures_{ANALYSIS_TAG}.csv", index=False)

    ctx = context.copy()
    ctx["site_name"] = ctx["site_id"].map(lambda v: site_label(int(v)))
    ctx["relative_days"] = (pd.to_datetime(ctx["date"]) - EVENT_DATE).dt.days
    ctx["relative_biweek"] = (ctx["relative_days"] / 14.0).round().astype(int)
    ctx["stage"] = pd.to_datetime(ctx["date"]).map(_stage)
    ctx.to_csv(TABLE_DIR / f"{output_prefix}_xgb_event_context_{ANALYSIS_TAG}.csv", index=False)
    return event


def _stage_points(event: pd.DataFrame) -> pd.DataFrame:
    keep = event[event["stage"].isin(["pre-event", "drought trough", "apparent recovery", "2023 recovery"])].copy()
    return (
        keep.groupby(["site_id", "site_name", "stage", "trait"], as_index=False)
        .agg(value=("prediction_median", "median"), departure=("percent_departure", "median"))
    )


def _site_panel(ax, event: pd.DataFrame, site_id: int, title: str) -> None:
    sub = event[event["site_id"] == site_id]
    ax.axvspan(TROUGH_START, TROUGH_END, color="#F4E7CB", linewidth=0, zorder=0)
    ax.axhline(0, color="#777777", lw=0.5, ls=":")
    ax.axhline(100, color="black", lw=0.6)
    ax.axhline(50, color="#BBBBBB", lw=0.45, ls=":")
    for short, _full, label, color in TRAITS:
        s = sub[sub["trait"] == short].sort_values("date")
        ax.plot(s["date"], s["recovery_pct_clipped"], marker="o", ms=2.6, lw=1.0, color=color, label=label)
    ax.set_title(title, loc="left")
    ax.set_ylabel("Recovery to baseline (%)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_xlim(pd.Timestamp("2022-04-15"), pd.Timestamp("2023-08-31"))
    ax.set_ylim(-5, 105)
    style_axis(ax)


def _index_recovery_text(context: pd.DataFrame, site_ids: list[int]) -> str:
    lines = []
    for site_id in site_ids:
        site_name = site_label(site_id)
        s = context[context["site_id"] == site_id].copy().sort_values("date")
        if s.empty:
            continue
        recovered = []
        for var in ("NDVI_median", "NDWI_median"):
            base_val = s[(s["date"] >= BASELINE_START) & (s["date"] <= BASELINE_END)][var].median()
            sub = s[(s["date"] > TROUGH_END) & np.isfinite(s[var])]
            hit = sub[sub[var] >= base_val]["date"]
            recovered.append(pd.Timestamp(hit.iloc[0]).strftime("%b %Y") if not hit.empty else "not by Aug 2023")
        lines.append(f"{site_name}: NDVI {recovered[0]}, NDWI {recovered[1]}")
    return "\n".join(lines)


def _nitrogen_recovery_text(event: pd.DataFrame, site_ids: list[int]) -> str:
    vals = []
    for site_id in site_ids:
        site_name = site_label(site_id)
        late = event[
            (event["site_id"] == site_id)
            & (event["trait"] == "Nitrogen")
            & (event["date"] >= RECOVERY_START)
            & (event["date"] <= RECOVERY_END)
        ]
        true = event[
            (event["site_id"] == site_id)
            & (event["trait"] == "Nitrogen")
            & (event["date"] >= TRUE_RECOVERY_START)
            & (event["date"] <= TRUE_RECOVERY_END)
        ]
        if late.empty or true.empty:
            continue
        vals.append(
            f"{site_name}: {float(late['percent_departure'].median()):+.1f}% -> "
            f"{float(true['percent_departure'].median()):+.1f}%"
        )
    return "\n".join(vals)


def build_recovery_table(event: pd.DataFrame, context: pd.DataFrame, output_prefix: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    trait_rows = event[
        [
            "site_id",
            "site_name",
            "date",
            "relative_days",
            "stage",
            "trait",
            "trait_label",
            "prediction_median",
            "baseline_median",
            "trough_median",
            "recovery_pct",
            "recovery_pct_clipped",
            "recovery_interpretable",
        ]
    ].copy()
    trait_rows = trait_rows.rename(columns={"trait": "variable", "trait_label": "variable_label", "prediction_median": "value"})
    trait_rows["variable_family"] = "trait"
    rows.append(trait_rows)

    ctx = context.copy()
    ctx["date"] = pd.to_datetime(ctx["date"])
    ctx["site_name"] = ctx["site_id"].map(lambda v: site_label(int(v)))
    ctx["relative_days"] = (ctx["date"] - EVENT_DATE).dt.days
    ctx["stage"] = ctx["date"].map(_stage)
    context_specs = [
        ("NDVI", "NDVI_median", "NDVI", "index", 0.01),
        ("EVI", "EVI_median", "EVI", "index", 0.01),
        ("NDWI", "NDWI_median", "NDWI", "index", 0.01),
        ("NBR", "NBR_median", "NBR", "index", 0.01),
        ("CIRE", "CIRE_median", "CIRE", "index", 0.02),
        ("NDRE", "NDRE_median", "NDRE", "index", 0.01),
        ("MSI", "MSI_median", "MSI", "index", 0.02),
        ("SWC", "volumetric_soil_water_layer_1", "SWC", "climate", 0.02),
        ("Tair", "2m_temperature_C", "Tair", "climate", 0.02),
        ("Tmax", "2m_temperature_max_C", "Tmax", "climate", 0.02),
        ("P", "total_precipitation_mm", "Precip.", "climate", 0.05),
        ("SWdown", "surface_solar_radiation_downwards_MJ_m2_day", "SWdown", "climate", 0.02),
    ]
    context_rows: list[dict[str, object]] = []
    for site_id, site_df in ctx.groupby("site_id"):
        for variable, column, label, family, min_mag in context_specs:
            if column not in site_df.columns:
                continue
            s = site_df[["site_id", "site_name", "date", "relative_days", "stage", column]].copy()
            baseline = s[(s["date"] >= BASELINE_START) & (s["date"] <= BASELINE_END)][column].astype(float)
            trough = s[(s["date"] >= TROUGH_START) & (s["date"] <= TROUGH_END)][column].astype(float)
            base_med = float(np.nanmedian(baseline)) if len(baseline) else math.nan
            trough_med = float(np.nanmedian(trough)) if len(trough) else math.nan
            for _idx, rec in s.iterrows():
                value = float(rec[column])
                recovery_pct, recovery_pct_clipped, interpretable = _recovery_score(
                    value,
                    base_med,
                    trough_med,
                    min_relative_magnitude=min_mag,
                )
                context_rows.append(
                    {
                        "site_id": int(site_id),
                        "site_name": site_label(int(site_id)),
                        "date": pd.Timestamp(rec["date"]),
                        "relative_days": int(rec["relative_days"]),
                        "stage": str(rec["stage"]),
                        "variable": variable,
                        "variable_label": label,
                        "value": value,
                        "baseline_median": base_med,
                        "trough_median": trough_med,
                        "recovery_pct": recovery_pct,
                        "recovery_pct_clipped": recovery_pct_clipped,
                        "recovery_interpretable": bool(interpretable),
                        "variable_family": family,
                    }
                )
    if context_rows:
        rows.append(pd.DataFrame(context_rows))
    recovery = pd.concat(rows, ignore_index=True)
    recovery = recovery.sort_values(["site_id", "variable_family", "variable", "date"])
    recovery.to_csv(TABLE_DIR / f"{output_prefix}_recovery_timeline_{ANALYSIS_TAG}.csv", index=False)
    return recovery


def _style_for_variable(variable: str) -> tuple[str, str]:
    for name, _label, color, ls in RECOVERY_VARIABLES:
        if name == variable:
            return color, ls
    return "#444444", "-"


def _plot_recovery_timeline_axis(ax, recovery: pd.DataFrame, site_id: int, title: str, variables: list[str] | None = None) -> None:
    variables = variables or [name for name, _label, _color, _ls in RECOVERY_VARIABLES]
    sub = recovery[(recovery["site_id"] == int(site_id)) & (recovery["variable"].isin(variables))].copy()
    ax.axvspan(TROUGH_START, TROUGH_END, color="#F4E7CB", linewidth=0, zorder=0)
    ax.axhline(0, color="#777777", lw=0.5, ls=":")
    ax.axhline(50, color="#BBBBBB", lw=0.45, ls=":")
    ax.axhline(100, color="black", lw=0.6)
    for variable in variables:
        s = sub[sub["variable"] == variable].sort_values("date")
        if s.empty:
            continue
        if not bool(np.isfinite(s["recovery_pct_clipped"]).any()):
            continue
        color, ls = _style_for_variable(variable)
        label = str(s["variable_label"].iloc[0])
        ax.plot(s["date"], s["recovery_pct_clipped"], color=color, ls=ls, lw=1.0, marker="o", ms=2.1, label=label)
    ax.set_title(title, loc="left")
    ax.set_ylabel("Recovery to baseline (%)")
    ax.set_ylim(-5, 105)
    ax.set_xlim(pd.Timestamp("2022-04-15"), pd.Timestamp("2023-08-31"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    style_axis(ax)


def _robust_baseline_departure(values: pd.Series, dates: pd.Series) -> pd.Series:
    dates = pd.to_datetime(dates)
    vals = pd.to_numeric(values, errors="coerce").astype(float)
    baseline = vals[(dates >= BASELINE_START) & (dates <= BASELINE_END)]
    base_med = float(np.nanmedian(baseline)) if len(baseline) else math.nan
    base_iqr = float(np.nanquantile(baseline, 0.75) - np.nanquantile(baseline, 0.25)) if len(baseline) else math.nan
    if not np.isfinite(base_iqr) or abs(base_iqr) <= 1e-12:
        fallback = float(np.nanstd(baseline)) if len(baseline) else math.nan
        base_iqr = fallback if np.isfinite(fallback) and abs(fallback) > 1e-12 else math.nan
    if not np.isfinite(base_iqr) or abs(base_iqr) <= 1e-12:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (vals - base_med) / base_iqr


def _plot_all_variable_anomaly_axis(ax, event: pd.DataFrame, context: pd.DataFrame, site_id: int, title: str) -> None:
    ax.axvspan(TROUGH_START, TROUGH_END, color="#F4E7CB", linewidth=0, zorder=0)
    ax.axhline(0, color="black", lw=0.6)
    plotted = set()
    sub_event = event[event["site_id"] == int(site_id)].copy()
    for short, _full, label, color in TRAITS:
        s = sub_event[sub_event["trait"] == short].sort_values("date")
        if s.empty:
            continue
        ax.plot(
            s["date"],
            np.clip(pd.to_numeric(s["robust_departure"], errors="coerce"), -4.0, 4.0),
            marker="o",
            ms=2.0,
            lw=0.85,
            color=color,
            ls="-",
            label=label,
        )
        plotted.add(short)

    ctx = context[context["site_id"] == int(site_id)].copy().sort_values("date")
    for variable, column, label in PANEL_A_CONTEXT_VARIABLES:
        if column not in ctx.columns:
            continue
        dep = _robust_baseline_departure(ctx[column], ctx["date"])
        if not np.isfinite(dep).any():
            continue
        color, ls = _style_for_variable(variable)
        ax.plot(
            ctx["date"],
            np.clip(dep, -4.0, 4.0),
            marker="o",
            ms=1.7,
            lw=0.75,
            color=color,
            ls=ls,
            label=label,
        )
        plotted.add(variable)

    ax.set_title(title, loc="left")
    ax.set_ylabel("Baseline departure\n(IQR units, clipped)")
    ax.set_ylim(-4.2, 4.2)
    ax.set_xlim(pd.Timestamp("2022-04-15"), pd.Timestamp("2023-08-31"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    style_axis(ax)


def plot_all_site_recovery_figures(recovery: pd.DataFrame, site_ids: list[int], output_prefix: str) -> None:
    out_dir = FIG_DIR / "fig4_recovery_timeline_cap5000_sites"
    out_dir.mkdir(parents=True, exist_ok=True)
    variables = [name for name, _label, _color, _ls in RECOVERY_VARIABLES]
    multipage_path = FIG_DIR / f"fig4_recovery_timeline_{output_prefix}.pdf"
    with PdfPages(multipage_path) as pdf:
        for site_id in site_ids:
            fig, ax = plt.subplots(figsize=(7.2, 3.0))
            _plot_recovery_timeline_axis(ax, recovery, site_id, f"site {site_id}, {site_label(site_id)}", variables=variables)
            ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0.0, -0.24), handlelength=1.6)
            fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.30)
            safe_name = str(site_label(site_id)).replace("/", "_").replace(" ", "_")
            stem = f"fig4_recovery_timeline_site{int(site_id):02d}_{safe_name}"
            fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
            fig.savefig(out_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    ncols = 3
    nrows = int(math.ceil(len(site_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, min(12.0, 1.55 * nrows)), sharex=True, sharey=True)
    axes_arr = np.asarray(axes).reshape(-1)
    compact_vars = ["SWC", "NDVI", "NDWI", "LAI", "LMA"]
    for ax, site_id in zip(axes_arr, site_ids, strict=False):
        _plot_recovery_timeline_axis(ax, recovery, site_id, f"{site_id} {site_label(site_id)}", variables=compact_vars)
        ax.set_title(f"{site_id} {site_label(site_id)}", loc="left", fontsize=6.5)
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=5.8)
        ax.tick_params(axis="y", labelsize=5.8)
    for ax in axes_arr[len(site_ids) :]:
        ax.axis("off")
    handles, labels = axes_arr[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.52, 0.995), handlelength=1.3)
    fig.supylabel("Recovery to baseline (%)", x=0.01, fontsize=8)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.965, bottom=0.035, wspace=0.16, hspace=0.42)
    fig.savefig(FIG_DIR / f"fig4_recovery_timeline_{output_prefix}_contact_sheet.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"fig4_recovery_timeline_{output_prefix}_contact_sheet.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _format_time_axis(ax) -> None:
    ax.set_xlim(pd.Timestamp("2022-04-15"), pd.Timestamp("2023-08-31"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))


def _plot_era5_axis(ax, context: pd.DataFrame) -> None:
    ctx = context.copy()
    ctx["date"] = pd.to_datetime(ctx["date"])
    ax2 = ax.twinx()
    for site_id, color, label in [(43, "#0072B2", "Le Bray"), (44, "#D55E00", "Bilos")]:
        s = ctx[ctx["site_id"] == site_id].sort_values("date")
        base = s[(s["date"] >= BASELINE_START) & (s["date"] <= BASELINE_END)]["volumetric_soil_water_layer_1"].median()
        ax.plot(
            s["date"],
            100.0 * (s["volumetric_soil_water_layer_1"] - base) / abs(base),
            color=color,
            lw=1.0,
            marker="o",
            ms=2.3,
            label=f"{label} SWC",
        )
        if "2m_temperature_max_C" in s.columns:
            tbase = s[(s["date"] >= BASELINE_START) & (s["date"] <= BASELINE_END)]["2m_temperature_max_C"].median()
            ax2.plot(
                s["date"],
                s["2m_temperature_max_C"] - tbase,
                color=color,
                lw=0.85,
                ls="--",
                marker="o",
                ms=2.0,
                label=f"{label} Tmax",
            )
    ax.axvspan(TROUGH_START, TROUGH_END, color="#F4E7CB", linewidth=0, zorder=0)
    ax.axhline(0, color="black", lw=0.6)
    ax2.axhline(0, color="#777777", lw=0.5, ls=":")
    ax.set_title("ERA5-Land context", loc="left")
    ax.set_ylabel("SWC departure (%)")
    ax2.set_ylabel("Tmax departure (°C)")
    _format_time_axis(ax)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, ncol=4, loc="upper left", handlelength=1.5)
    style_axis(ax)
    style_axis(ax2)
    ax2.spines["right"].set_visible(True)


def _plot_trait_percent_axis(ax, event: pd.DataFrame, site_id: int, title: str) -> None:
    sub = event[event["site_id"] == int(site_id)].copy()
    ax.axvspan(TROUGH_START, TROUGH_END, color="#F4E7CB", linewidth=0, zorder=0)
    ax.axhline(0, color="black", lw=0.6)
    for short, _full, label, color in TRAITS:
        s = sub[sub["trait"] == short].sort_values("date")
        if s.empty:
            continue
        ax.plot(s["date"], s["percent_departure"], marker="o", ms=2.3, lw=0.95, color=color, label=label)
    ax.set_title(title, loc="left")
    ax.set_ylabel("Trait departure\nfrom baseline (%)")
    ax.set_ylim(-45, 45)
    _format_time_axis(ax)
    ax.legend(frameon=False, ncol=5, loc="upper left", handlelength=1.4)
    style_axis(ax)


def _plot_spectral_axis(ax, context: pd.DataFrame) -> None:
    ctx = context.copy()
    ctx["date"] = pd.to_datetime(ctx["date"])
    index_specs = [
        ("NDVI", "NDVI_median"),
        ("NDWI", "NDWI_median"),
        ("MSI", "MSI_median"),
    ]
    for site_id, site_ls in [(43, "-"), (44, "--")]:
        s = ctx[ctx["site_id"] == site_id].sort_values("date")
        for variable, column in index_specs:
            if column not in s.columns:
                continue
            base = s[(s["date"] >= BASELINE_START) & (s["date"] <= BASELINE_END)][column].median()
            color, _ls = _style_for_variable(variable)
            ax.plot(
                s["date"],
                100.0 * (s[column] - base) / abs(base),
                color=color,
                ls=site_ls,
                lw=0.9,
                marker="o",
                ms=1.9,
                label=variable if site_id == 43 else None,
            )
    ax.axvspan(TROUGH_START, TROUGH_END, color="#F4E7CB", linewidth=0, zorder=0)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Sentinel-2 spectral-index context", loc="left")
    ax.set_ylabel("Index departure\nfrom baseline (%)")
    ax.text(0.01, 0.95, "NDVI: greenness\nNDWI: water-related\nMSI: stress\nsolid: Le Bray; dashed: Bilos", transform=ax.transAxes, fontsize=6.4, va="top")
    _format_time_axis(ax)
    ax.set_ylim(-80, 80)
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.34, 1.03), handlelength=1.4)
    style_axis(ax)


def plot_figure(event: pd.DataFrame, context: pd.DataFrame, recovery: pd.DataFrame) -> None:
    setup_style()
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.25), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 1.05], "hspace": 0.22})
    ax_a, ax_b, ax_c, ax_d = axes

    recovery_vars = ["SWC", "NDVI", "NDWI", "LAI", "LMA"]
    _plot_recovery_timeline_axis(ax_a, recovery, 43, "Le Bray recovery timeline", variables=recovery_vars)
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=7, loc="upper center", bbox_to_anchor=(0.52, 0.995), handlelength=1.3, columnspacing=0.9)
    panel_label(ax_a, "a", x=-0.075, y=1.02)
    _plot_recovery_timeline_axis(ax_b, recovery, 44, "Bilos recovery timeline", variables=recovery_vars)
    panel_label(ax_b, "b", x=-0.075, y=1.02)
    _plot_era5_axis(ax_c, context)
    panel_label(ax_c, "c", x=-0.075, y=1.02)
    _plot_spectral_axis(ax_d, context)
    panel_label(ax_d, "d", x=-0.075, y=1.02)

    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    ax_d.set_xlabel("Date")
    fig.subplots_adjust(top=0.94, left=0.105, right=0.90, bottom=0.09)
    save_figure(fig, "fig4_trait_anomaly_syndromes")


def write_provenance(model_payload: dict[str, object], train_fraction: float, window_pixels: int, smoke: bool) -> None:
    payload = {
        "figure": "fig4_trait_anomaly_syndromes",
        "model": "local XGBoost single-trait models",
        "train_fraction": float(train_fraction),
        "window_pixels": int(window_pixels),
        "smoke": bool(smoke),
        "task_manifest": str(TASK_MANIFEST),
        "feature_names_count": len(model_payload["feature_names"]),
        "traits": [{"short": s, "name": n, "label": l} for s, n, l, _c in TRAITS],
        "anchor_sites": {str(k): v for k, v in ANCHOR_SITE_NAMES.items()},
        "event_date": str(EVENT_DATE.date()),
        "analysis_window": [str(ANALYSIS_START.date()), str(ANALYSIS_END.date())],
        "baseline_window": [str(BASELINE_START.date()), str(BASELINE_END.date())],
        "drought_trough_window": [str(TROUGH_START.date()), str(TROUGH_END.date())],
        "recovery_window": [str(RECOVERY_START.date()), str(RECOVERY_END.date())],
        "true_recovery_window": [str(TRUE_RECOVERY_START.date()), str(TRUE_RECOVERY_END.date())],
        "atmospheric_demand_note": "ERA5-Land local zarrs do not include humidity or dew-point fields; Fig. 4 panel c uses Tmax departure as a heat-demand proxy rather than VPD.",
    }
    (TABLE_DIR / "fig4_drought_anchor_xgb_provenance.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-models", action="store_true")
    parser.add_argument("--force-predictions", action="store_true")
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--window-pixels", type=int, default=30)
    parser.add_argument("--max-pixels-per-site", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--site-scope", choices=["anchor", "cap5000"], default="anchor")
    parser.add_argument("--site-ids", default="", help="Optional comma-separated site IDs overriding --site-scope.")
    args = parser.parse_args()
    setup_style()
    if args.site_ids.strip():
        site_ids = sorted({int(v) for v in args.site_ids.replace(" ", "").split(",") if v})
        scope = "custom"
    elif args.site_scope == "cap5000":
        site_ids = cap5000_site_ids()
        scope = "cap5000"
    else:
        site_ids = list(ANCHOR_SITE_IDS)
        scope = "anchor"
    output_prefix = _output_prefix(site_ids, scope)
    model_payload = train_or_load_models(force=bool(args.force_models), train_fraction=float(args.train_fraction), n_jobs=int(args.n_jobs))
    pred, context = predict_sites(
        model_payload,
        site_ids=site_ids,
        output_prefix=output_prefix,
        force=bool(args.force_predictions),
        window_pixels=int(args.window_pixels),
        smoke=bool(args.smoke),
        max_pixels_per_site=int(args.max_pixels_per_site),
    )
    summary = summarize_predictions(pred, output_prefix=output_prefix)
    event = build_event_table(summary, context, output_prefix=output_prefix)
    recovery = build_recovery_table(event, context, output_prefix=output_prefix)
    if set(ANCHOR_SITE_IDS).issubset(set(site_ids)):
        plot_figure(
            event[event["site_id"].isin(ANCHOR_SITE_IDS)].copy(),
            context[context["site_id"].isin(ANCHOR_SITE_IDS)].copy(),
            recovery[recovery["site_id"].isin(ANCHOR_SITE_IDS)].copy(),
        )
    if scope == "cap5000" or len(site_ids) > 2:
        plot_all_site_recovery_figures(recovery, site_ids=site_ids, output_prefix=output_prefix)
    write_provenance(model_payload, train_fraction=float(args.train_fraction), window_pixels=int(args.window_pixels), smoke=bool(args.smoke))
    print(
        json.dumps(
            {
                "figure": str(FIG_DIR / "fig4_trait_anomaly_syndromes.pdf"),
                "site_scope": scope,
                "site_count": len(site_ids),
                "output_prefix": output_prefix,
                "table_dir": str(TABLE_DIR),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
