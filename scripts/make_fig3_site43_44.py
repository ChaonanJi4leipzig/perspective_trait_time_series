#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import zipfile

import joblib
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio
from rasterio.transform import rowcol
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import torch
import xarray as xr

PAPER_ROOT = Path(__file__).resolve().parents[1]
ECOSPECTRA_ROOT = Path("/home/chaonan.ji/data/projects/ecospectra")
TRAITS_ROOT = ECOSPECTRA_ROOT / "models" / "traits_model"
if str(TRAITS_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(TRAITS_ROOT / "scripts"))

from stage2_models.featurize_trait_centered_tabular import _nearest_20m_index, _spectral_indices, _window_weight
from stage2_models.trait_centered_dataset import (
    SEQ_CHANNEL_NAMES,
    filter_feature_columns_by_profile,
    load_trait_centered_partitions,
    trait_centered_column_groups,
)
from stage2_models.trait_centered_mlp import TemporalMLPRegressor
from stage2_models.trait_centered_temporal_transformer import TemporalTokenTransformer
from stage2_models.trait_centered_dataset import load_table

RUN_ROOT = (
    ECOSPECTRA_ROOT
    / "output_data/traits_model/outputs/ablations/reintegrated_bdc_gate_20260506"
    / "cap2000_train20_besthp_lowobs_weighted_5models"
)
BASE_RUN = (
    ECOSPECTRA_ROOT
    / "output_data/traits_model/outputs/ablations/reintegrated_bdc_gate_20260506"
    / "cap2000_sharded_sequence_5traits"
)
FEATURE_TABLE = BASE_RUN / "contracts/lc_default_main/trait_centered_filtered.parquet"
SPLIT_PATH = (
    ECOSPECTRA_ROOT
    / "output_data/traits_model/outputs/datasets/trait_centered/splits/split_global_grouped_70_15_15.json"
)
SITE_TABLE = ECOSPECTRA_ROOT / "input_data/fluxnet_sites.csv"
FLUXNET_SHUTTLE = ECOSPECTRA_ROOT / "input_data/fluxnet_shuttle.csv"
S2_ROOT = ECOSPECTRA_ROOT / "input_data/s2/region"
ERA5_ROOT = ECOSPECTRA_ROOT / "input_data/era5_land_point"
HRL_TCD_ROOT = ECOSPECTRA_ROOT / "input_data/enmap/landcover/hrl2021_30m_site_aligned/tcd"
TRAITS_REGION_ROOT = ECOSPECTRA_ROOT / "input_data/enmap/traits_region"

FIG_DIR = PAPER_ROOT / "figures" / "generated"
TABLE_DIR = PAPER_ROOT / "tables" / "fig3_site43_44"
MODEL_CACHE = TABLE_DIR / "model_cache"
ICOS_TOKEN_PATH = PAPER_ROOT / "scripts" / "icos_token.txt"

TRAIT_ORDER = ["LAI", "LMA", "Nitrogen", "Chlorophyll", "Lignin"]
MODEL_ORDER = ["rf", "mlp", "transformer"]
MODEL_LABELS = {"rf": "RF", "mlp": "MLP", "transformer": "Transformer"}
SITE_IDS = [43, 44]
SITE_LABELS = {43: "site 43, Le Bray", 44: "site 44, Bilos"}
SITE_FLUXNET_CODES = {44: "FR-Bil"}
TRAIT_LABELS = {
    "LAI": "LAI (m2 m-2)",
    "LMA": "LMA (g m-2)",
    "Nitrogen": "N (mg cm-2)",
    "Chlorophyll": "Chl (ug cm-2)",
    "Lignin": "Lignin (mg cm-2)",
}
TRAIT_CANONICAL = {
    "LAI": "LAI (m²/m²)",
    "LMA": "LMA (g/m²)",
    "Nitrogen": "N content (mg/cm²)",
    "Chlorophyll": "Chl content (μg/cm²)",
    "Lignin": "Lignin (mg/cm²)",
}
SPECTRAL_INDICES = ["NDVI", "EVI", "NDWI", "NBR", "CIRE", "NDRE", "MSI"]
SPECTRAL_PLOT_INDICES = ["NDVI", "EVI", "NDWI", "NBR", "NDRE"]
CLIMATE_VARS = [
    "2m_temperature",
    "total_precipitation",
    "surface_solar_radiation_downwards",
    "volumetric_soil_water_layer_1",
]
FLUXNET_SOURCE_VARS = [
    "GPP_NT_VUT_REF",
    "NEE_VUT_REF",
    "RECO_NT_VUT_REF",
    "LE_F_MDS",
    "VPD_F_MDS",
    "SWC_F_MDS_1",
    "SWC_F_MDS_2",
]
FLUXNET_PLOT_VARS = [
    ("GPP_NT_VUT_REF", "GPP"),
    ("NEP_VUT_REF", "NEP"),
    ("ET_F_MDS_mm_day", "ET"),
    ("VPD_F_MDS", "VPD"),
    ("SWC_F_MDS_1", "SWC1"),
    ("SWC_F_MDS_2", "SWC2"),
    ("WUE_GPP_ET", "WUE"),
    ("RECO_NT_VUT_REF", "Reco"),
]
FLUXNET_CONTEXT_VARS = [name for name, _label in FLUXNET_PLOT_VARS] + ["NEE_VUT_REF", "LE_F_MDS"]

FIG_W_DOUBLE = 183 / 25.4
FIG_H_MAX = 170 / 25.4
FS_PANEL = 8.0
FS_AXIS = 8.0
FS_TICK = 7.0
FS_TITLE = 7.0
FS_LEG = 7.0
FS_NOTE = 7.0
AXIS_LW = 0.5
FIT_LW = 1.0
BLACK = "#000000"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
BLUE = "#0072B2"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#666666"


def setup_style() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": FS_AXIS,
            "axes.titlesize": FS_TITLE,
            "axes.labelsize": FS_AXIS,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "legend.fontsize": FS_LEG,
            "axes.grid": False,
            "axes.linewidth": AXIS_LW,
            "xtick.major.width": AXIS_LW,
            "ytick.major.width": AXIS_LW,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def style_axis(ax, keep_all_spines: bool = False) -> None:
    if keep_all_spines:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(AXIS_LW)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(AXIS_LW)
        ax.spines["bottom"].set_linewidth(AXIS_LW)
    ax.tick_params(axis="both", labelsize=FS_TICK, width=AXIS_LW, length=3)
    ax.grid(False)


def panel_label(ax, letter: str, x: float = -0.08, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FS_PANEL,
        fontweight="bold",
        color=BLACK,
    )


def panel_label_inside(ax, letter: str) -> None:
    ax.text(
        0.006,
        0.92,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FS_PANEL,
        fontweight="bold",
        color=BLACK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
    )


def save_publication_figure(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "eps"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _load_split() -> dict[str, object]:
    payload = json.loads(SPLIT_PATH.read_text())
    splits = payload["splits"] if isinstance(payload, dict) and "splits" in payload else payload
    for split in splits:
        if split.get("split_name") == "global_grouped_70_15_15_seed42":
            return split
    raise ValueError("split_global_grouped_70_15_15_seed42 not found")


def _metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    truth = y_true[finite].astype(float, copy=False)
    pred = y_pred[finite].astype(float, copy=False)
    n = int(truth.size)
    if n == 0:
        return {"n_valid": 0, "r2": math.nan, "rmse": math.nan, "nrmse_pct": math.nan}
    rmse = float(np.sqrt(np.mean(np.square(pred - truth))))
    trait_range = float(np.nanmax(truth) - np.nanmin(truth)) if n else math.nan
    nrmse = float((rmse / trait_range) * 100.0) if np.isfinite(trait_range) and trait_range > 0 else math.nan
    r2 = float(r2_score(truth, pred)) if len(np.unique(truth)) > 1 else math.nan
    return {"n_valid": n, "r2": r2, "rmse": rmse, "nrmse_pct": nrmse}


def build_skill_table(force: bool = False) -> pd.DataFrame:
    out_path = TABLE_DIR / "site43_44_model_skill.csv"
    if out_path.exists() and not force:
        return pd.read_csv(out_path)
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        for trait in TRAIT_ORDER:
            pred_path = RUN_ROOT / model / trait / "test_predictions.csv"
            pred_df = pd.read_csv(pred_path)
            true_cols = [c for c in pred_df.columns if c.startswith("true__")]
            pred_cols = [c for c in pred_df.columns if c.startswith("pred__")]
            if len(true_cols) != 1 or len(pred_cols) != 1:
                raise ValueError(f"Unexpected prediction schema in {pred_path}")
            for site_id in SITE_IDS:
                site_df = pred_df[pred_df["site_id"].astype(int) == int(site_id)]
                metrics = _metric_row(site_df[true_cols[0]].to_numpy(), site_df[pred_cols[0]].to_numpy())
                rows.append(
                    {
                        "site_id": int(site_id),
                        "site_label": SITE_LABELS[int(site_id)],
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "trait": trait,
                        "trait_label": TRAIT_LABELS[trait],
                        **metrics,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def plot_skill_heatmaps(skill_df: pd.DataFrame) -> None:
    row_order = [(site, trait) for site in SITE_IDS for trait in TRAIT_ORDER]
    row_labels = [f"{site}\n{trait}" for site, trait in row_order]
    col_labels = [MODEL_LABELS[m] for m in MODEL_ORDER]

    def matrix(metric: str) -> np.ndarray:
        values = np.full((len(row_order), len(MODEL_ORDER)), np.nan, dtype=float)
        for i, (site, trait) in enumerate(row_order):
            for j, model in enumerate(MODEL_ORDER):
                match = skill_df[
                    (skill_df["site_id"].astype(int) == site)
                    & (skill_df["trait"] == trait)
                    & (skill_df["model"] == model)
                ]
                if not match.empty:
                    values[i, j] = float(match.iloc[0][metric])
        return values

    r2 = matrix("r2")
    nrmse = matrix("nrmse_pct")
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, 4.9), gridspec_kw={"wspace": 0.40})
    cmap_r2 = LinearSegmentedColormap.from_list("r2", ["#F6F6F6", "#CFE5D8", "#7CC4B0", BLUE])
    cmap_err = LinearSegmentedColormap.from_list("nrmse", ["#F6F6F6", "#F2DEC3", ORANGE, VERMILION])
    for ax, values, title, cbar_label, cmap, vmin, vmax, letter in [
        (axes[0], r2, "Site-specific model skill", "R2", cmap_r2, 0.0, 0.8, "a"),
        (axes[1], nrmse, "Normalized error", "nRMSE (%)", cmap_err, 0.0, 22.0, "b"),
    ]:
        im = ax.imshow(values, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        ax.set_xticks(np.arange(len(col_labels)), col_labels)
        ax.set_yticks(np.arange(len(row_labels)), row_labels)
        ax.tick_params(length=0)
        ax.set_title(title, loc="left", fontsize=FS_TITLE, pad=5)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values[i, j]
                text = "NA" if not np.isfinite(val) else f"{val:.2f}" if cbar_label == "R2" else f"{val:.1f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=FS_TICK, color=BLACK)
        style_axis(ax, keep_all_spines=True)
        panel_label(ax, letter, x=-0.20, y=1.04)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
        cb.set_label(cbar_label, fontsize=FS_AXIS)
        cb.ax.tick_params(labelsize=FS_TICK, width=AXIS_LW, length=3)
        cb.outline.set_linewidth(AXIS_LW)
    save_publication_figure(fig, "fig3_site43_44_skill_heatmaps")


def _site_metadata() -> pd.DataFrame:
    sites = pd.read_csv(SITE_TABLE)
    return sites[sites["site_id"].astype(int).isin(SITE_IDS)].copy()


def _site_center_index(site_id: int) -> tuple[int, int]:
    sites = _site_metadata().set_index("site_id")
    lat = float(sites.loc[int(site_id), "location_lat"])
    lon = float(sites.loc[int(site_id), "location_lon"])
    tcd_path = HRL_TCD_ROOT / f"site_{site_id}_hrl2021_tcd_30m.tif"
    with rasterio.open(tcd_path) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x_utm, y_utm = transformer.transform(lon, lat)
        row, col = rowcol(src.transform, x_utm, y_utm)
        row = int(np.clip(row, 0, src.height - 1))
        col = int(np.clip(col, 0, src.width - 1))
    return row, col


def _window_indices(center: int, size: int, limit: int) -> np.ndarray:
    start = int(center) - size // 2
    start = max(0, min(start, limit - size))
    return np.arange(start, start + size, dtype=np.int32)


def _forest_window(site_id: int, window_pixels: int) -> pd.DataFrame:
    center_y, center_x = _site_center_index(site_id)
    tcd_path = HRL_TCD_ROOT / f"site_{site_id}_hrl2021_tcd_30m.tif"
    with rasterio.open(tcd_path) as src:
        y_idx = _window_indices(center_y, window_pixels, src.height)
        x_idx = _window_indices(center_x, window_pixels, src.width)
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


def _biweekly_dates(year: int = 2022) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-08", periods=26, freq="14D")


def _select_sequence_records(records: list[dict[str, float]], max_len: int = 8, core_days: int = 14) -> list[dict[str, float]]:
    if not records:
        return []
    core = [r for r in records if abs(float(r["delta_days"])) <= float(core_days)]
    outer = [r for r in records if abs(float(r["delta_days"])) > float(core_days)]
    core = sorted(core, key=lambda r: (abs(float(r["delta_days"])), float(r["delta_days"])))
    outer = sorted(outer, key=lambda r: (abs(float(r["delta_days"])), float(r["delta_days"])))
    selected = core[:max_len]
    if len(selected) < max_len:
        selected.extend(outer[: max_len - len(selected)])
    return sorted(selected, key=lambda r: float(r["delta_days"]))[:max_len]


def _make_empty_sequence(max_len: int = 8) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((max_len, len(SEQ_CHANNEL_NAMES)), dtype=np.float32), np.zeros((max_len,), dtype=bool)


def _records_to_sequence(records: list[dict[str, float]], max_len: int = 8) -> tuple[np.ndarray, np.ndarray]:
    seq, mask = _make_empty_sequence(max_len=max_len)
    for slot_idx, rec in enumerate(records[:max_len]):
        for ch_idx, name in enumerate(SEQ_CHANNEL_NAMES):
            seq[slot_idx, ch_idx] = float(rec.get(name, 0.0))
        mask[slot_idx] = True
    return seq, mask


def _flatten_sequence(seq: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {
        "feat__obs_count": float(mask.sum()),
        "feat__valid_fraction": float(mask.mean()) if mask.size else 0.0,
    }
    for slot in range(seq.shape[0]):
        for ch_idx, ch_name in enumerate(SEQ_CHANNEL_NAMES):
            out[f"seq__t{slot}_{ch_name}"] = float(seq[slot, ch_idx])
        out[f"mask__t{slot}"] = bool(mask[slot])
    return out


def build_site_sequences(site_id: int, window_pixels: int, smoke: bool = False) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    pixels = _forest_window(site_id, window_pixels=window_pixels)
    if smoke:
        pixels = pixels.head(min(25, len(pixels))).copy()
        dates = pd.DatetimeIndex([pd.Timestamp("2022-07-09")])
    else:
        dates = _biweekly_dates(2022)
    if pixels.empty:
        raise ValueError(f"No HRL forest pixels found in tower window for site {site_id}")

    s2_path = S2_ROOT / f"site_{site_id}" / f"site_{site_id}_s2_region_2022.zarr"
    ds = xr.open_zarr(s2_path, consolidated=False)
    source_times = ds.time.values.astype("datetime64[ns]")
    y_values = pixels["y"].to_numpy(dtype=np.int32)
    x_values = pixels["x"].to_numpy(dtype=np.int32)
    y10_min, y10_max = int(y_values.min() * 3), int(y_values.max() * 3 + 3)
    x10_min, x10_max = int(x_values.min() * 3), int(x_values.max() * 3 + 3)
    y20_values = np.array([_nearest_20m_index(int(v), int(ds.sizes["y_20m"])) for v in y_values], dtype=np.int32)
    x20_values = np.array([_nearest_20m_index(int(v), int(ds.sizes["x_20m"])) for v in x_values], dtype=np.int32)
    y20_min, y20_max = int(y20_values.min()), int(y20_values.max() + 1)
    x20_min, x20_max = int(x20_values.min()), int(x20_values.max() + 1)

    meta_rows: list[dict[str, object]] = []
    seq_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    flat_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []
    pixel_records = pixels.reset_index(drop=True).to_dict("records")

    for center_date in dates:
        trait_time = np.datetime64(center_date, "ns")
        delta_all = ((source_times - trait_time) / np.timedelta64(1, "D")).astype(np.float32)
        time_indices = np.where(np.abs(delta_all) <= 28.0)[0]
        context_payload: dict[str, object] = {"site_id": int(site_id), "date": pd.Timestamp(center_date)}
        if time_indices.size == 0:
            for name in SPECTRAL_INDICES:
                context_payload[f"{name}_median"] = math.nan
            context_rows.append(context_payload)
            for pix in pixel_records:
                seq, mask = _make_empty_sequence()
                meta = _meta_row(site_id, center_date, pix)
                meta_rows.append(meta)
                seq_rows.append(seq)
                mask_rows.append(mask)
                flat_rows.append({**meta, **_flatten_sequence(seq, mask)})
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
        context_values: dict[str, list[float]] = {name: [] for name in SPECTRAL_INDICES}

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
            indices = _spectral_indices(bands)
            records: list[dict[str, float]] = []
            for obs_idx in np.where(valid_obs)[0].tolist():
                ts = pd.Timestamp(selected_times[obs_idx])
                doy = float(ts.dayofyear)
                angle = 2.0 * np.pi * doy / 365.0
                rec = {
                    "delta_days": float(selected_delta[obs_idx]),
                    "doy_sin": float(np.sin(angle)),
                    "doy_cos": float(np.cos(angle)),
                    "weight": _window_weight(float(selected_delta[obs_idx]), core_days=14, window_days=28),
                }
                for band_idx, band_name in enumerate(SEQ_CHANNEL_NAMES[:10]):
                    rec[band_name] = float(bands[obs_idx, band_idx])
                for index_name in SPECTRAL_INDICES:
                    value = float(indices[index_name][obs_idx])
                    rec[index_name] = value
                    if np.isfinite(value):
                        context_values[index_name].append(value)
                records.append(rec)
            selected = _select_sequence_records(records, max_len=8, core_days=14)
            seq, mask = _records_to_sequence(selected, max_len=8)
            meta = _meta_row(site_id, center_date, pix)
            meta_rows.append(meta)
            seq_rows.append(seq)
            mask_rows.append(mask)
            flat_rows.append({**meta, **_flatten_sequence(seq, mask)})

        for name in SPECTRAL_INDICES:
            vals = np.asarray(context_values[name], dtype=float)
            context_payload[f"{name}_median"] = float(np.nanmedian(vals)) if vals.size else math.nan
        context_rows.append(context_payload)

    return (
        pd.DataFrame(meta_rows),
        np.stack(seq_rows, axis=0).astype(np.float32),
        np.stack(mask_rows, axis=0).astype(bool),
        pd.DataFrame(flat_rows),
        pd.DataFrame(context_rows),
    )


def build_direct_s2_spectral_context(force: bool = False, window_pixels: int = 30, smoke: bool = False) -> pd.DataFrame:
    suffix = "_smoke" if smoke else ""
    out_path = TABLE_DIR / f"site43_44_s2_direct_spectral_indices_2022{suffix}.csv"
    if out_path.exists() and not force:
        return pd.read_csv(out_path, parse_dates=["date"])
    frames = [
        _build_site_direct_s2_spectral_context(site_id, window_pixels=window_pixels, smoke=smoke)
        for site_id in SITE_IDS
    ]
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(out_path, index=False)
    return out


def _build_site_direct_s2_spectral_context(site_id: int, window_pixels: int, smoke: bool = False) -> pd.DataFrame:
    pixels = _forest_window(site_id, window_pixels=window_pixels)
    if smoke:
        pixels = pixels.head(min(25, len(pixels))).copy()
    if pixels.empty:
        raise ValueError(f"No HRL forest pixels found in tower window for site {site_id}")

    s2_path = S2_ROOT / f"site_{site_id}" / f"site_{site_id}_s2_region_2022.zarr"
    ds = xr.open_zarr(s2_path, consolidated=False)
    source_times = ds.time.values.astype("datetime64[ns]")
    time_indices = np.arange(len(source_times), dtype=np.int32)
    if smoke:
        time_indices = time_indices[: min(3, len(time_indices))]
    source_times = source_times[time_indices]

    y_values = pixels["y"].to_numpy(dtype=np.int32)
    x_values = pixels["x"].to_numpy(dtype=np.int32)
    y10_min, y10_max = int(y_values.min() * 3), int(y_values.max() * 3 + 3)
    x10_min, x10_max = int(x_values.min() * 3), int(x_values.max() * 3 + 3)
    y20_values = np.array([_nearest_20m_index(int(v), int(ds.sizes["y_20m"])) for v in y_values], dtype=np.int32)
    x20_values = np.array([_nearest_20m_index(int(v), int(ds.sizes["x_20m"])) for v in x_values], dtype=np.int32)
    y20_min, y20_max = int(y20_values.min()), int(y20_values.max() + 1)
    x20_min, x20_max = int(x20_values.min()), int(x20_values.max() + 1)

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

    values: dict[str, list[list[float]]] = {name: [[] for _ in range(len(time_indices))] for name in SPECTRAL_INDICES}
    valid_counts = np.zeros(len(time_indices), dtype=np.int32)
    for pix, y20, x20 in zip(pixels.reset_index(drop=True).to_dict("records"), y20_values, x20_values, strict=False):
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
            out=np.zeros((len(time_indices), masked10.shape[-1]), dtype=np.float32),
            where=valid_counts_10m[:, None] > 0,
        )
        pix20 = refl20[:, int(y20 - y20_min), int(x20 - x20_min), :]
        pix_adv = adv[:, int(y20 - y20_min), int(x20 - x20_min)]
        valid_obs = valid_sub.any(axis=(1, 2)) & np.isfinite(pix20).all(axis=1) & (pix_adv > 0)
        bands = np.concatenate([mean10, pix20], axis=1).astype(np.float32, copy=False)
        indices = _spectral_indices(bands)
        for obs_idx in np.where(valid_obs)[0].tolist():
            valid_counts[obs_idx] += 1
            for name in SPECTRAL_INDICES:
                value = float(indices[name][obs_idx])
                if np.isfinite(value):
                    values[name][obs_idx].append(value)

    rows: list[dict[str, object]] = []
    for obs_idx, ts in enumerate(source_times):
        row: dict[str, object] = {
            "site_id": int(site_id),
            "date": pd.Timestamp(ts),
            "n_valid_pixels": int(valid_counts[obs_idx]),
        }
        for name in SPECTRAL_INDICES:
            arr = np.asarray(values[name][obs_idx], dtype=float)
            row[f"{name}_median"] = float(np.nanmedian(arr)) if arr.size else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _meta_row(site_id: int, center_date: pd.Timestamp, pix: dict[str, object]) -> dict[str, object]:
    return {
        "site_id": int(site_id),
        "site_label": SITE_LABELS[int(site_id)],
        "date": pd.Timestamp(center_date),
        "year": int(pd.Timestamp(center_date).year),
        "y": int(pix["y"]),
        "x": int(pix["x"]),
        "pixel_uid": f"site_{site_id}_y_{int(pix['y'])}_x_{int(pix['x'])}",
        "hrl_tcd": float(pix["hrl_tcd"]),
    }


def _nn_predict(model: str, trait: str, sequence: np.ndarray, mask: np.ndarray, device: str) -> np.ndarray:
    run_dir = RUN_ROOT / model / trait
    payload = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    norm = payload.get("normalization", {})
    mean = np.asarray(norm.get("mean"), dtype=np.float32)
    std = np.asarray(norm.get("std"), dtype=np.float32)
    channel_mask = np.asarray(norm.get("channel_mask"), dtype=bool)
    x = sequence.astype(np.float32, copy=True)
    if mean.size and std.size and channel_mask.size:
        for idx, should_norm in enumerate(channel_mask.tolist()):
            if should_norm:
                x[:, :, idx] = (x[:, :, idx] - mean[idx]) / std[idx]
    target = payload.get("target_scaler", {})
    target_mean = float(np.asarray(target.get("mean", [0.0]), dtype=float)[0])
    target_std = float(np.asarray(target.get("std", [1.0]), dtype=float)[0])
    input_dim = int(x.shape[-1])
    if model == "mlp":
        net = TemporalMLPRegressor(
            input_dim=input_dim,
            output_dim=1,
            aux_dim=0,
            model_dim=int(cfg.get("model_dim", 64)),
            num_layers=int(cfg.get("num_layers", 4)),
            dropout=float(cfg.get("dropout", 0.3)),
        )
    elif model == "transformer":
        net = TemporalTokenTransformer(
            input_dim=input_dim,
            output_dim=1,
            aux_dim=0,
            model_dim=int(cfg.get("model_dim", 256)),
            nhead=int(cfg.get("nhead", 4)),
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.3)),
            temporal_readout=str(cfg.get("temporal_readout", "masked_mean")),
        )
    else:
        raise ValueError(model)
    net.load_state_dict(payload["model_state"])
    dev = torch.device(device if device != "auto" and torch.cuda.is_available() else "cpu")
    net.to(dev)
    net.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 512):
            end = min(start + 512, x.shape[0])
            xb = torch.from_numpy(x[start:end]).to(dev)
            mb = torch.from_numpy(mask[start:end].astype(np.bool_)).to(dev)
            pred_scaled = net(xb, mb).detach().cpu().numpy().reshape(-1)
            preds.append(pred_scaled * target_std + target_mean)
    return np.concatenate(preds).astype(np.float32)


def _load_rf_training(max_samples_per_partition: int = 0) -> dict[str, object]:
    split = _load_split()
    return load_trait_centered_partitions(
        FEATURE_TABLE,
        split,
        random_seed=42,
        max_samples_per_partition=int(max_samples_per_partition),
        train_fraction=0.2,
        train_fraction_seed=42,
        pooled_training=True,
        input_feature_profile="with_full_time_context",
        sample_metadata_profile="none",
    )


def _rf_model(
    trait: str,
    training: dict[str, object],
    force: bool = False,
    n_estimators_override: int | None = None,
) -> RandomForestRegressor:
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    suffix = "" if n_estimators_override is None else f"_trees{int(n_estimators_override)}"
    model_path = MODEL_CACHE / f"rf_{trait}{suffix}.joblib"
    if model_path.exists() and not force:
        return joblib.load(model_path)
    manifest = pd.read_csv(RUN_ROOT / "run_manifest.csv")
    row = manifest[(manifest["model"] == "rf") & (manifest["trait_short"] == trait)].iloc[0]
    params = json.loads(row["params_json"])
    params = dict(params)
    params["n_estimators"] = int(params["n_estimators"])
    if n_estimators_override is not None:
        params["n_estimators"] = int(n_estimators_override)
    x_train = np.asarray(training["x_train"], dtype=np.float32)
    y_train = np.asarray(training["y_train"], dtype=np.float32)
    label_mask = np.asarray(training["label_mask_train"], dtype=np.float32)
    trait_names = list(training["trait_names"])
    trait_idx = trait_names.index(TRAIT_CANONICAL[trait])
    valid = (label_mask[:, trait_idx] > 0) & np.isfinite(y_train[:, trait_idx])
    feature_names = list(training["feature_names"])
    mask_cols = [i for i, name in enumerate(feature_names) if str(name).startswith("mask__")]
    obs_count = x_train[:, mask_cols].sum(axis=1) if mask_cols else np.full(x_train.shape[0], 8.0)
    weights = np.where(obs_count[valid] < 3.0, 0.5, 1.0).astype(np.float32)
    rf = RandomForestRegressor(random_state=42, n_jobs=8, **params)
    rf.fit(x_train[valid], y_train[valid, trait_idx], sample_weight=weights)
    joblib.dump(rf, model_path)
    return rf


def _rf_predict(
    trait: str,
    flat: pd.DataFrame,
    training: dict[str, object],
    force_rf: bool = False,
    n_estimators_override: int | None = None,
) -> np.ndarray:
    feature_names = list(training["feature_names"])
    medians = np.asarray(training["train_feature_medians"], dtype=np.float32)
    x = flat.reindex(columns=feature_names).to_numpy(dtype=np.float32, copy=True)
    x = np.where(np.isfinite(x), x, medians[None, :]).astype(np.float32, copy=False)
    model = _rf_model(
        trait,
        training=training,
        force=force_rf,
        n_estimators_override=n_estimators_override,
    )
    return model.predict(x).astype(np.float32)


def build_predictions(force: bool = False, force_rf: bool = False, window_pixels: int = 30, smoke: bool = False, device: str = "auto") -> tuple[pd.DataFrame, pd.DataFrame]:
    suffix = "_smoke" if smoke else ""
    pred_path = TABLE_DIR / f"site43_44_biweekly_trait_predictions_2022{suffix}.parquet"
    context_path = TABLE_DIR / f"site43_44_spectral_context_2022{suffix}.csv"
    if pred_path.exists() and context_path.exists() and not force:
        return pd.read_parquet(pred_path), pd.read_csv(context_path, parse_dates=["date"])
    training: dict[str, object] | None = None
    pred_frames: list[pd.DataFrame] = []
    context_frames: list[pd.DataFrame] = []
    for site_id in SITE_IDS:
        print(f"[features] site {site_id} window_pixels={window_pixels} smoke={smoke}", flush=True)
        meta, sequence, mask, flat, spectral_context = build_site_sequences(site_id, window_pixels=window_pixels, smoke=smoke)
        context_frames.append(spectral_context)
        for model in MODEL_ORDER:
            for trait in TRAIT_ORDER:
                if model == "rf":
                    if training is None:
                        print("[rf] loading training partitions", flush=True)
                        training = _load_rf_training(max_samples_per_partition=2000 if smoke else 0)
                    print(f"[predict] {model} {trait} site {site_id}", flush=True)
                    y_pred = _rf_predict(
                        trait,
                        flat=flat,
                        training=training,
                        force_rf=force_rf,
                        n_estimators_override=20 if smoke else None,
                    )
                else:
                    print(f"[predict] {model} {trait} site {site_id}", flush=True)
                    y_pred = _nn_predict(model, trait, sequence=sequence, mask=mask, device=device)
                frame = meta.copy()
                frame["model"] = model
                frame["model_label"] = MODEL_LABELS[model]
                frame["trait"] = trait
                frame["trait_label"] = TRAIT_LABELS[trait]
                frame["prediction"] = y_pred
                frame["valid_obs_count"] = mask.sum(axis=1).astype(int)
                pred_frames.append(frame)
    pred_df = pd.concat(pred_frames, ignore_index=True)
    context_df = pd.concat(context_frames, ignore_index=True)
    pred_df.to_parquet(pred_path, index=False)
    context_df.to_csv(context_path, index=False)
    return pred_df, context_df


def _aggregate_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        pred_df.groupby(["site_id", "site_label", "date", "model", "model_label", "trait", "trait_label"], as_index=False)
        .agg(
            prediction_median=("prediction", "median"),
            prediction_p10=("prediction", lambda s: float(np.nanquantile(s, 0.10))),
            prediction_p90=("prediction", lambda s: float(np.nanquantile(s, 0.90))),
            n_pixels=("prediction", "size"),
            valid_obs_median=("valid_obs_count", "median"),
        )
        .sort_values(["site_id", "date", "model", "trait"])
    )
    agg.to_csv(TABLE_DIR / "site43_44_biweekly_trait_predictions_2022_summary.csv", index=False)
    return agg


def _climate_context(site_id: int, dates: pd.DatetimeIndex) -> pd.DataFrame:
    ds = xr.open_zarr(ERA5_ROOT / f"site_{site_id}" / f"site_{site_id}_yr2022.zarr", consolidated=False)
    rows: list[dict[str, object]] = []
    for date in dates:
        start = np.datetime64(date - pd.Timedelta(days=7), "ns")
        end = np.datetime64(date + pd.Timedelta(days=6), "ns")
        sub = ds.sel(time=slice(start, end))
        row: dict[str, object] = {"site_id": int(site_id), "date": pd.Timestamp(date)}
        if sub.sizes.get("time", 0) == 0:
            for var in CLIMATE_VARS:
                row[var] = math.nan
            rows.append(row)
            continue
        temp = sub["2m_temperature"].values.astype(float)
        row["2m_temperature_C"] = float(np.nanmean(temp - 273.15 if np.nanmean(temp) > 100 else temp))
        precip = sub["total_precipitation"].values.astype(float)
        row["total_precipitation_mm"] = float(np.nansum(precip * 1000.0))
        rad = sub["surface_solar_radiation_downwards"].values.astype(float)
        row["surface_solar_radiation_downwards_MJ_m2_day"] = float(np.nanmean(rad / 1_000_000.0))
        soil = sub["volumetric_soil_water_layer_1"].values.astype(float)
        row["volumetric_soil_water_layer_1"] = float(np.nanmean(soil))
        rows.append(row)
    return pd.DataFrame(rows)


def _find_fluxnet_zip(site_code: str) -> Path | None:
    local_dir = TABLE_DIR / "fluxnet"
    search_dirs = [
        local_dir,
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


def _download_fluxnet_zip(site_code: str) -> Path | None:
    shuttle = pd.read_csv(FLUXNET_SHUTTLE)
    row = shuttle[shuttle["site_id"] == site_code]
    if row.empty:
        return None
    url = str(row.iloc[0]["download_link"])
    name = str(row.iloc[0]["fluxnet_product_name"])
    out = TABLE_DIR / "fluxnet" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return out
    tmp = out.with_suffix(out.suffix + ".part")
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        if str(row.iloc[0].get("data_hub", "")).upper() == "ICOS" and ICOS_TOKEN_PATH.exists():
            token = ICOS_TOKEN_PATH.read_text().strip()
            if token:
                session.cookies.set("cpauthToken", token, domain=".icos-cp.eu")
        first = session.get(url, allow_redirects=False, timeout=60)
        location = first.headers.get("location")
        if location and location.startswith("/"):
            download_url = "https://data.icos-cp.eu" + location
        else:
            download_url = location or url
        with session.get(download_url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        tmp.replace(out)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        print(f"[fluxnet] download failed for {site_code}: {exc}", flush=True)
        return None
    return out if out.exists() and out.stat().st_size > 0 else None


def _read_fluxnet_daily(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        fluxmet_daily = [n for n in names if n.lower().endswith(".csv") and "fluxmet_dd" in n.lower()]
        daily = [n for n in names if n.lower().endswith(".csv") and ("_dd_" in n.lower() or "daily" in n.lower()) and "bif" not in n.lower()]
        halfhour = [n for n in names if n.lower().endswith(".csv") and ("_hh_" in n.lower() or "_hr_" in n.lower()) and "bif" not in n.lower()]
        daily = fluxmet_daily or daily
        chosen = daily[0] if daily else halfhour[0] if halfhour else None
        if chosen is None:
            raise ValueError(f"No FLUXNET CSV found in {zip_path}")
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
    if "LE_F_MDS" in daily.columns:
        le = pd.to_numeric(daily["LE_F_MDS"], errors="coerce")
        daily["ET_F_MDS_mm_day"] = le * 86400.0 / 2_450_000.0
    if "GPP_NT_VUT_REF" in daily.columns and "ET_F_MDS_mm_day" in daily.columns:
        et = pd.to_numeric(daily["ET_F_MDS_mm_day"], errors="coerce")
        gpp = pd.to_numeric(daily["GPP_NT_VUT_REF"], errors="coerce")
        daily["WUE_GPP_ET"] = np.where(et > 0, gpp / et, np.nan)
    return daily


def _fluxnet_context(site_id: int, dates: pd.DatetimeIndex, download: bool = False) -> pd.DataFrame:
    site_code = SITE_FLUXNET_CODES.get(int(site_id))
    rows = pd.DataFrame({"site_id": int(site_id), "date": dates})
    if site_code is None:
        print(f"[fluxnet] no 2022 FLUXNET product configured for site {site_id}; FLUXNET columns will be NA", flush=True)
        for var in FLUXNET_CONTEXT_VARS:
            rows[var] = math.nan
        return rows
    zip_path = _find_fluxnet_zip(site_code)
    if zip_path is None and download:
        zip_path = _download_fluxnet_zip(site_code)
    if zip_path is None:
        print(f"[fluxnet] no local {site_code} zip found; FLUXNET columns will be NA", flush=True)
        for var in FLUXNET_CONTEXT_VARS:
            rows[var] = math.nan
        return rows
    try:
        daily = _read_fluxnet_daily(zip_path)
    except Exception as exc:
        print(f"[fluxnet] failed to read {zip_path}: {exc}", flush=True)
        for var in FLUXNET_CONTEXT_VARS:
            rows[var] = math.nan
        return rows
    out_rows: list[dict[str, object]] = []
    for date in dates:
        start = date - pd.Timedelta(days=7)
        end = date + pd.Timedelta(days=6)
        sub = daily[(daily["date"] >= start) & (daily["date"] <= end)]
        row: dict[str, object] = {"site_id": int(site_id), "date": pd.Timestamp(date)}
        for var in FLUXNET_CONTEXT_VARS:
            row[var] = float(np.nanmean(sub[var])) if var in sub.columns and not sub.empty else math.nan
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def build_site_context(site_id: int, pred_df: pd.DataFrame, spectral_context: pd.DataFrame, download_fluxnet: bool = False) -> pd.DataFrame:
    agg = _aggregate_predictions(pred_df)
    site_dates = pd.DatetimeIndex(sorted(pd.to_datetime(agg[agg["site_id"].astype(int) == int(site_id)]["date"]).unique()))
    spec = spectral_context[spectral_context["site_id"].astype(int) == int(site_id)].copy()
    climate = _climate_context(int(site_id), site_dates)
    flux = _fluxnet_context(int(site_id), site_dates, download=download_fluxnet)
    context = spec.merge(climate, on=["site_id", "date"], how="outer").merge(flux, on=["site_id", "date"], how="outer")
    context = context.sort_values("date")
    out_path = TABLE_DIR / f"site{int(site_id)}_biweekly_context_2022.csv"
    context.to_csv(out_path, index=False)
    return context


def build_site44_context(pred_df: pd.DataFrame, spectral_context: pd.DataFrame, download_fluxnet: bool = False) -> pd.DataFrame:
    return build_site_context(44, pred_df, spectral_context, download_fluxnet=download_fluxnet)


def _scale01(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").astype(float)
    lo = float(np.nanmin(arr)) if np.isfinite(arr).any() else math.nan
    hi = float(np.nanmax(arr)) if np.isfinite(arr).any() else math.nan
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.nan, index=values.index)
    return (arr - lo) / (hi - lo)


def plot_site_context(site_id: int, pred_df: pd.DataFrame, context_df: pd.DataFrame, s2_spectral_df: pd.DataFrame) -> None:
    agg = _aggregate_predictions(pred_df)
    site_id = int(site_id)
    site_agg = agg[agg["site_id"].astype(int) == site_id].copy()
    site_agg["date"] = pd.to_datetime(site_agg["date"])
    context_df = context_df.copy()
    context_df["date"] = pd.to_datetime(context_df["date"])
    s2_spectral_df = s2_spectral_df.copy()
    s2_spectral_df["date"] = pd.to_datetime(s2_spectral_df["date"])
    s2_site = s2_spectral_df[s2_spectral_df["site_id"].astype(int) == site_id].sort_values("date")
    fig = plt.figure(figsize=(FIG_W_DOUBLE, FIG_H_MAX))
    gs = fig.add_gridspec(8, 1, height_ratios=[1, 1, 1, 1, 1, 1.05, 1.05, 1.05], hspace=0.18)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(8)]
    colors = {"rf": BLUE, "mlp": ORANGE, "transformer": GREEN}
    date_min = pd.Timestamp("2022-01-01")
    date_max = pd.Timestamp("2022-12-31")

    for idx, trait in enumerate(TRAIT_ORDER):
        ax = axes[idx]
        sub = site_agg[site_agg["trait"] == trait]
        for model in MODEL_ORDER:
            m = sub[sub["model"] == model].sort_values("date")
            if m.empty:
                continue
            ax.plot(m["date"], m["prediction_median"], lw=FIT_LW, color=colors[model], label=MODEL_LABELS[model])
            ax.fill_between(
                m["date"],
                m["prediction_p10"].to_numpy(dtype=float),
                m["prediction_p90"].to_numpy(dtype=float),
                color=colors[model],
                alpha=0.08,
                linewidth=0,
            )
        ax.set_ylabel(TRAIT_LABELS[trait], fontsize=FS_TICK)
        ax.set_xlim(date_min, date_max)
        style_axis(ax)
        if idx < 7:
            ax.tick_params(labelbottom=False)
        if idx == 0:
            panel_label_inside(ax, "a")
            ax.legend(frameon=False, ncol=3, loc="upper right", handlelength=1.4, fontsize=FS_LEG)

    ax = axes[5]
    for name, color in zip(SPECTRAL_PLOT_INDICES, [BLUE, ORANGE, GREEN, PURPLE, SKY], strict=False):
        col = f"{name}_median"
        if col in s2_site.columns:
            ax.plot(s2_site["date"], s2_site[col], lw=FIT_LW, color=color, label=name)
    ax.set_ylabel("S2 index\n(raw)", fontsize=FS_TICK)
    ax.set_ylim(-0.10, 1.05)
    ax.set_xlim(date_min, date_max)
    style_axis(ax)
    panel_label_inside(ax, "b")
    ax.legend(frameon=False, ncol=5, loc="upper right", handlelength=1.2, fontsize=FS_LEG)
    ax.tick_params(labelbottom=False)

    ax = axes[6]
    climate_cols = [
        ("2m_temperature_C", "Tair"),
        ("total_precipitation_mm", "P"),
        ("surface_solar_radiation_downwards_MJ_m2_day", "SWdown"),
        ("volumetric_soil_water_layer_1", "SWC"),
    ]
    for (col, label), color in zip(climate_cols, [BLUE, ORANGE, GREEN, PURPLE], strict=False):
        if col in context_df.columns:
            ax.plot(context_df["date"], _scale01(context_df[col]), lw=FIT_LW, color=color, label=label)
    ax.set_ylabel("Climate\n(0-1)", fontsize=FS_TICK)
    ax.set_xlim(date_min, date_max)
    style_axis(ax)
    panel_label_inside(ax, "c")
    ax.legend(frameon=False, ncol=4, loc="upper right", handlelength=1.2, fontsize=FS_LEG)
    ax.tick_params(labelbottom=False)

    ax = axes[7]
    any_flux = False
    flux_colors = [BLUE, ORANGE, GREEN, PURPLE, SKY, VERMILION, GREY, BLACK]
    for (var, label), color in zip(FLUXNET_PLOT_VARS, flux_colors, strict=False):
        if var in context_df.columns and pd.to_numeric(context_df[var], errors="coerce").notna().any():
            any_flux = True
            ax.plot(context_df["date"], _scale01(context_df[var]), lw=FIT_LW, color=color, label=label)
    if not any_flux:
        ax.text(0.5, 0.50, f"site {site_id} FLUXNET 2022 data not available locally", ha="center", va="center", transform=ax.transAxes, fontsize=FS_NOTE)
    ax.set_ylabel("Flux/stress\n(0-1)", fontsize=FS_TICK)
    ax.set_xlabel("2022")
    ax.set_xlim(date_min, date_max)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    style_axis(ax)
    panel_label_inside(ax, "d")
    if any_flux:
        ax.set_ylim(-0.05, 1.45)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.legend(
            frameon=False,
            ncol=4,
            loc="upper center",
            bbox_to_anchor=(0.53, 1.03),
            handlelength=1.3,
            columnspacing=1.2,
            labelspacing=0.2,
            fontsize=FS_LEG,
        )
    save_publication_figure(fig, f"fig3_site{site_id}_2022_trait_context")


def plot_site44_context(pred_df: pd.DataFrame, context_df: pd.DataFrame) -> None:
    s2_spectral = build_direct_s2_spectral_context(force=False, window_pixels=30, smoke=False)
    plot_site_context(44, pred_df, context_df, s2_spectral)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Perspective Fig. 3 for sites 43/44 and 2022 context panels.")
    parser.add_argument("--force", action="store_true", help="Rebuild cached tables.")
    parser.add_argument("--force-rf", action="store_true", help="Retrain cached RF models.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny site-44 smoke prediction.")
    parser.add_argument("--window-pixels", type=int, default=30, help="Tower window size in 30 m pixels.")
    parser.add_argument("--device", type=str, default="auto", help="Torch device, or auto.")
    parser.add_argument("--download-fluxnet", action="store_true", help="Try downloading configured FLUXNET zips if absent locally.")
    args = parser.parse_args()

    setup_style()
    skill_df = build_skill_table(force=args.force)
    plot_skill_heatmaps(skill_df)
    pred_df, spectral_context = build_predictions(
        force=args.force,
        force_rf=args.force_rf,
        window_pixels=int(args.window_pixels),
        smoke=bool(args.smoke),
        device=str(args.device),
    )
    s2_spectral = build_direct_s2_spectral_context(
        force=args.force,
        window_pixels=int(args.window_pixels),
        smoke=bool(args.smoke),
    )
    context_outputs: dict[int, Path] = {}
    figure_outputs: list[Path] = [
        FIG_DIR / "fig3_site43_44_skill_heatmaps.pdf",
    ]
    for site_id in SITE_IDS:
        context_df = build_site_context(site_id, pred_df, spectral_context, download_fluxnet=bool(args.download_fluxnet))
        plot_site_context(site_id, pred_df, context_df, s2_spectral)
        context_outputs[int(site_id)] = TABLE_DIR / f"site{int(site_id)}_biweekly_context_2022.csv"
        figure_outputs.append(FIG_DIR / f"fig3_site{int(site_id)}_2022_trait_context.pdf")
    print(
        json.dumps(
            {
                "skill_table": str(TABLE_DIR / "site43_44_model_skill.csv"),
                "predictions": str(TABLE_DIR / ("site43_44_biweekly_trait_predictions_2022_smoke.parquet" if args.smoke else "site43_44_biweekly_trait_predictions_2022.parquet")),
                "contexts": {str(site_id): str(path) for site_id, path in context_outputs.items()},
                "figures": [str(path) for path in figure_outputs],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
