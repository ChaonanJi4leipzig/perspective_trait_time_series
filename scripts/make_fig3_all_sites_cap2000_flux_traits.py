#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from functools import lru_cache
from pathlib import Path
import sys

import joblib
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch


PAPER_ROOT = Path("/home/chaonan.ji/data/projects/ecospectra/docs/paper1_perspective")
ECOSPECTRA_ROOT = Path("/home/chaonan.ji/data/projects/ecospectra")
TRAITS_ROOT = ECOSPECTRA_ROOT / "models" / "traits_model"
BASE_SCRIPT = PAPER_ROOT / "scripts" / "make_fig3_site43_44.py"
TABLE_DIR = PAPER_ROOT / "tables" / "fig3_all_sites_cap2000_flux_traits"
FIG_DIR = PAPER_ROOT / "figures" / "generated" / "fig3_all_sites_cap2000_flux_traits"
RF_CACHE_DIR = PAPER_ROOT / "tables" / "fig3_site43_44" / "model_cache"
B5B6_CAP5000_TRANSFORMER_ARTIFACT_ROOT = (
    TABLE_DIR / "model_artifacts" / "worldcover37_b5b6filtered_cap5000_transformer"
)

SITE_TABLE = ECOSPECTRA_ROOT / "input_data" / "filtered_sites.csv"
FLUXNET_SHUTTLE = ECOSPECTRA_ROOT / "input_data" / "fluxnet_shuttle.csv"


def _load_base_module():
    spec = importlib.util.spec_from_file_location("fig3_site43_44_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base_module()

TRAIT_ORDER = base.TRAIT_ORDER
MODEL_ORDER = base.MODEL_ORDER
MODEL_LABELS = base.MODEL_LABELS
TRAIT_LABELS = base.TRAIT_LABELS
FLUXNET_PLOT_VARS = base.FLUXNET_PLOT_VARS
FLUXNET_CONTEXT_VARS = base.FLUXNET_CONTEXT_VARS
FLUXNET_SOURCE_VARS = base.FLUXNET_SOURCE_VARS
SPECTRAL_PLOT_INDICES = ["NDVI", "EVI", "NDWI", "NBR", "CIRE", "NDRE", "MSI"]
ERA5_PLOT_VARS = [
    ("2m_temperature_C", "Tair"),
    ("total_precipitation_mm", "P"),
    ("surface_solar_radiation_downwards_MJ_m2_day", "SWdown"),
    ("volumetric_soil_water_layer_1", "SWC"),
]


def setup() -> pd.DataFrame:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sites = pd.read_csv(SITE_TABLE).sort_values("site_id").reset_index(drop=True)
    site_ids = [int(v) for v in sites["site_id"].tolist()]
    labels = {int(r.site_id): f"site {int(r.site_id)}, {r.site_name}" for r in sites.itertuples(index=False)}
    base.TABLE_DIR = TABLE_DIR
    base.FIG_DIR = FIG_DIR
    base.MODEL_CACHE = RF_CACHE_DIR
    base.SITE_IDS = site_ids
    base.SITE_LABELS = labels
    base.setup_style()
    mpl.rcParams["figure.dpi"] = 150
    return sites


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(a))


def build_fluxnet_mapping(sites: pd.DataFrame, max_distance_km: float = 5.0, force: bool = False) -> pd.DataFrame:
    out_path = TABLE_DIR / "site_fluxnet_mapping.csv"
    if out_path.exists() and not force:
        return pd.read_csv(out_path)
    shuttle = pd.read_csv(FLUXNET_SHUTTLE)
    shuttle = shuttle.copy()
    shuttle["first_year"] = pd.to_numeric(shuttle["first_year"], errors="coerce")
    shuttle["last_year"] = pd.to_numeric(shuttle["last_year"], errors="coerce")
    rows: list[dict[str, object]] = []
    lat2 = shuttle["location_lat"].astype(float).to_numpy()
    lon2 = shuttle["location_long"].astype(float).to_numpy()
    for site in sites.itertuples(index=False):
        distances = haversine_km(float(site.location_lat), float(site.location_lon), lat2, lon2)
        candidates = shuttle.assign(distance_km=distances)
        candidates = candidates[candidates["distance_km"] <= float(max_distance_km)].copy()
        candidates_2022 = candidates[(candidates["first_year"] <= 2022) & (candidates["last_year"] >= 2022)].copy()
        if not candidates_2022.empty:
            pick = candidates_2022.sort_values(["distance_km", "last_year"], ascending=[True, False]).iloc[0]
        elif not candidates.empty:
            pick = candidates.sort_values(["distance_km", "last_year"], ascending=[True, False]).iloc[0]
        else:
            pick = None
        if pick is None:
            rows.append(
                {
                    "site_id": int(site.site_id),
                    "site_name": site.site_name,
                    "fluxnet_site_id": "",
                    "fluxnet_site_name": "",
                    "data_hub": "",
                    "first_year": np.nan,
                    "last_year": np.nan,
                    "distance_km": np.nan,
                    "has_2022": False,
                    "fluxnet_product_name": "",
                    "download_link": "",
                }
            )
        else:
            rows.append(
                {
                    "site_id": int(site.site_id),
                    "site_name": site.site_name,
                    "fluxnet_site_id": str(pick["site_id"]),
                    "fluxnet_site_name": str(pick["site_name"]),
                    "data_hub": str(pick["data_hub"]),
                    "first_year": int(pick["first_year"]),
                    "last_year": int(pick["last_year"]),
                    "distance_km": float(pick["distance_km"]),
                    "has_2022": bool(pick["first_year"] <= 2022 <= pick["last_year"]),
                    "fluxnet_product_name": str(pick["fluxnet_product_name"]),
                    "download_link": str(pick["download_link"]),
                }
            )
    mapping = pd.DataFrame(rows)
    mapping.to_csv(out_path, index=False)
    return mapping


def configure_fluxnet_codes(mapping: pd.DataFrame) -> None:
    code_map = {
        int(row.site_id): str(row.fluxnet_site_id)
        for row in mapping.itertuples(index=False)
        if bool(row.has_2022) and isinstance(row.fluxnet_site_id, str) and row.fluxnet_site_id
    }
    base.SITE_FLUXNET_CODES = code_map


@lru_cache(maxsize=1)
def rf_training() -> dict[str, object]:
    return base._load_rf_training(max_samples_per_partition=0)


@lru_cache(maxsize=len(TRAIT_ORDER))
def rf_model(trait: str):
    path = RF_CACHE_DIR / f"rf_{trait}.joblib"
    if path.exists():
        return joblib.load(path)
    return base._rf_model(trait, training=rf_training(), force=False)


def rf_predict_cached(trait: str, flat: pd.DataFrame) -> np.ndarray:
    training = rf_training()
    feature_names = list(training["feature_names"])
    medians = np.asarray(training["train_feature_medians"], dtype=np.float32)
    x = flat.reindex(columns=feature_names).to_numpy(dtype=np.float32, copy=True)
    x = np.where(np.isfinite(x), x, medians[None, :]).astype(np.float32, copy=False)
    return rf_model(trait).predict(x).astype(np.float32)


def aggregate_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["site_id", "site_label", "date", "model", "model_label", "trait", "trait_label"], as_index=False)
        .agg(
            prediction_median=("prediction", "median"),
            prediction_p10=("prediction", lambda s: float(np.nanquantile(s, 0.10))),
            prediction_p90=("prediction", lambda s: float(np.nanquantile(s, 0.90))),
            n_pixels=("prediction", "size"),
            valid_obs_median=("valid_obs_count", "median"),
        )
        .sort_values(["site_id", "date", "model", "trait"])
    )


PREDICTION_COLUMNS = [
    "site_id",
    "site_label",
    "date",
    "model",
    "model_label",
    "trait",
    "trait_label",
    "prediction_median",
    "prediction_p10",
    "prediction_p90",
    "n_pixels",
    "valid_obs_median",
]


def predict_b5b6_cap5000_transformer(trait: str, sequence: np.ndarray, mask: np.ndarray, device: str) -> np.ndarray:
    run_dir = B5B6_CAP5000_TRANSFORMER_ARTIFACT_ROOT / trait
    if not (run_dir / "best_model.pt").exists():
        raise FileNotFoundError(f"Missing b5/b6 cap5000 Transformer artifact: {run_dir / 'best_model.pt'}")
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
    net = base.TemporalTokenTransformer(
        input_dim=int(x.shape[-1]),
        output_dim=1,
        aux_dim=0,
        model_dim=int(cfg.get("model_dim", 256)),
        nhead=int(cfg.get("nhead", 2)),
        num_layers=int(cfg.get("num_layers", 4)),
        dropout=float(cfg.get("dropout", 0.3)),
        temporal_readout=str(cfg.get("temporal_readout", "masked_mean")),
        head_type=str(cfg.get("head_type", "linear_vector")),
    )
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


def build_b5b6_cap5000_transformer_predictions(
    sites: pd.DataFrame,
    force: bool = False,
    window_pixels: int = 30,
    device: str = "auto",
    max_sites: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_path = TABLE_DIR / "all_sites_b5b6_cap5000_transformer_biweekly_trait_predictions_2022_summary.csv"
    spectral_path = TABLE_DIR / "all_sites_cap2000_biweekly_spectral_context_2022.csv"
    if pred_path.exists() and spectral_path.exists() and not force:
        return pd.read_csv(pred_path, parse_dates=["date"]), pd.read_csv(spectral_path, parse_dates=["date"])

    pred_frames: list[pd.DataFrame] = []
    spectral_frames: list[pd.DataFrame] = []
    site_rows = sites.head(int(max_sites)).copy() if max_sites else sites
    for idx, site in enumerate(site_rows.itertuples(index=False), start=1):
        site_id = int(site.site_id)
        print(f"[b5b6 site {idx}/{len(site_rows)}] features site_id={site_id}", flush=True)
        try:
            meta, sequence, mask, _flat, spectral_context = base.build_site_sequences(
                site_id, window_pixels=int(window_pixels), smoke=False
            )
        except Exception as exc:
            print(f"[b5b6 site {site_id}] skipped feature build: {exc}", flush=True)
            continue
        spectral_frames.append(spectral_context)
        for trait in ["LAI", "LMA"]:
            print(f"[b5b6 predict] site={site_id} model=transformer trait={trait}", flush=True)
            y_pred = predict_b5b6_cap5000_transformer(trait, sequence=sequence, mask=mask, device=device)
            frame = meta[["site_id", "site_label", "date"]].copy()
            frame["model"] = "transformer_b5b6_cap5000"
            frame["model_label"] = "B5/B6 cap5000 Transformer"
            frame["trait"] = trait
            frame["trait_label"] = TRAIT_LABELS[trait]
            frame["prediction"] = y_pred
            frame["valid_obs_count"] = mask.sum(axis=1).astype(int)
            pred_frames.append(aggregate_prediction_frame(frame))
    pred = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame(columns=PREDICTION_COLUMNS)
    spectral = pd.concat(spectral_frames, ignore_index=True) if spectral_frames else pd.DataFrame()
    pred.to_csv(pred_path, index=False)
    if not spectral.empty:
        spectral.to_csv(spectral_path, index=False)
    return pred, spectral


def build_trait_predictions(
    sites: pd.DataFrame,
    force: bool = False,
    window_pixels: int = 30,
    device: str = "auto",
    max_sites: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_path = TABLE_DIR / "all_sites_cap2000_biweekly_trait_predictions_2022_summary.csv"
    spectral_path = TABLE_DIR / "all_sites_cap2000_biweekly_spectral_context_2022.csv"
    if pred_path.exists() and spectral_path.exists() and not force:
        return pd.read_csv(pred_path, parse_dates=["date"]), pd.read_csv(spectral_path, parse_dates=["date"])

    pred_frames: list[pd.DataFrame] = []
    spectral_frames: list[pd.DataFrame] = []
    site_rows = sites.head(int(max_sites)).copy() if max_sites else sites
    for idx, site in enumerate(site_rows.itertuples(index=False), start=1):
        site_id = int(site.site_id)
        print(f"[site {idx}/{len(site_rows)}] features site_id={site_id}", flush=True)
        try:
            meta, sequence, mask, flat, spectral_context = base.build_site_sequences(
                site_id, window_pixels=int(window_pixels), smoke=False
            )
        except Exception as exc:
            print(f"[site {site_id}] skipped feature build: {exc}", flush=True)
            continue
        spectral_frames.append(spectral_context)
        for model in MODEL_ORDER:
            for trait in TRAIT_ORDER:
                print(f"[predict] site={site_id} model={model} trait={trait}", flush=True)
                if model == "rf":
                    y_pred = rf_predict_cached(trait, flat=flat)
                else:
                    y_pred = base._nn_predict(model, trait, sequence=sequence, mask=mask, device=device)
                frame = meta[["site_id", "site_label", "date"]].copy()
                frame["model"] = model
                frame["model_label"] = MODEL_LABELS[model]
                frame["trait"] = trait
                frame["trait_label"] = TRAIT_LABELS[trait]
                frame["prediction"] = y_pred
                frame["valid_obs_count"] = mask.sum(axis=1).astype(int)
                pred_frames.append(aggregate_prediction_frame(frame))
    pred = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame(columns=PREDICTION_COLUMNS)
    spectral = pd.concat(spectral_frames, ignore_index=True) if spectral_frames else pd.DataFrame()
    pred.to_csv(pred_path, index=False)
    spectral.to_csv(spectral_path, index=False)
    return pred, spectral


def build_fluxnet_contexts(
    sites: pd.DataFrame,
    mapping: pd.DataFrame,
    force: bool = False,
    download: bool = False,
    max_sites: int | None = None,
) -> pd.DataFrame:
    out_path = TABLE_DIR / "all_sites_fluxnet_context_2022_biweekly.csv"
    if out_path.exists() and not force:
        return pd.read_csv(out_path, parse_dates=["date"])
    configure_fluxnet_codes(mapping)
    frames: list[pd.DataFrame] = []
    dates = base._biweekly_dates(2022)
    site_rows = sites.head(int(max_sites)).copy() if max_sites else sites
    for idx, site in enumerate(site_rows.itertuples(index=False), start=1):
        site_id = int(site.site_id)
        print(f"[fluxnet {idx}/{len(site_rows)}] site_id={site_id}", flush=True)
        df = base._fluxnet_context(site_id, dates, download=download)
        df.insert(1, "site_name", str(site.site_name))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(out_path, index=False)
    return out


def build_era5_contexts(sites: pd.DataFrame, force: bool = False, max_sites: int | None = None) -> pd.DataFrame:
    out_path = TABLE_DIR / "all_sites_era5_context_2022_biweekly.csv"
    if out_path.exists() and not force:
        return pd.read_csv(out_path, parse_dates=["date"])
    frames: list[pd.DataFrame] = []
    dates = base._biweekly_dates(2022)
    site_rows = sites.head(int(max_sites)).copy() if max_sites else sites
    for idx, site in enumerate(site_rows.itertuples(index=False), start=1):
        site_id = int(site.site_id)
        print(f"[era5 {idx}/{len(site_rows)}] site_id={site_id}", flush=True)
        try:
            df = base._climate_context(site_id, dates)
        except Exception as exc:
            print(f"[era5 site {site_id}] missing or unreadable: {exc}", flush=True)
            df = pd.DataFrame({"site_id": site_id, "date": dates})
            for col, _label in ERA5_PLOT_VARS:
                df[col] = np.nan
        df.insert(1, "site_name", str(site.site_name))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(out_path, index=False)
    return out


def scale01(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").astype(float)
    if not np.isfinite(arr).any():
        return pd.Series(np.nan, index=values.index)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.nan, index=values.index)
    return (arr - lo) / (hi - lo)


def plot_site(
    site_id: int,
    site_name: str,
    pred: pd.DataFrame,
    spectral: pd.DataFrame,
    era5: pd.DataFrame,
    flux: pd.DataFrame,
    pdf: PdfPages | None = None,
) -> Path:
    pred_site = pred[pred["site_id"].astype(int) == int(site_id)].copy()
    spectral_site = spectral[spectral["site_id"].astype(int) == int(site_id)].copy() if not spectral.empty else pd.DataFrame()
    era5_site = era5[era5["site_id"].astype(int) == int(site_id)].copy() if not era5.empty else pd.DataFrame()
    flux_site = flux[flux["site_id"].astype(int) == int(site_id)].copy()
    pred_site["date"] = pd.to_datetime(pred_site["date"])
    if not spectral_site.empty:
        spectral_site["date"] = pd.to_datetime(spectral_site["date"])
    if not era5_site.empty:
        era5_site["date"] = pd.to_datetime(era5_site["date"])
    flux_site["date"] = pd.to_datetime(flux_site["date"])
    fig, axes = plt.subplots(8, 1, figsize=(8.0, 11.0), sharex=True, constrained_layout=True)
    colors = {"rf": base.BLUE, "mlp": base.ORANGE, "transformer": base.GREEN}
    fig.suptitle(f"site {site_id}, {site_name} | cap2000 traits, S2 indices, ERA5, and FLUXNET context, 2022", fontsize=10)
    date_min = pd.Timestamp("2022-01-01")
    date_max = pd.Timestamp("2022-12-31")
    for ax, trait in zip(axes[:5], TRAIT_ORDER, strict=False):
        sub = pred_site[pred_site["trait"] == trait].sort_values("date")
        if sub.empty:
            ax.text(0.5, 0.5, "No prediction for this trait/model set", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        else:
            for model in list(pred_site["model"].dropna().unique()):
                m = sub[sub["model"] == model].sort_values("date")
                if m.empty:
                    continue
                color = colors.get(model, base.PURPLE)
                label = str(m["model_label"].iloc[0]) if "model_label" in m.columns else model
                ax.plot(m["date"], m["prediction_median"], lw=1.0, color=color, label=label)
                ax.fill_between(
                    m["date"],
                    m["prediction_p10"].to_numpy(dtype=float),
                    m["prediction_p90"].to_numpy(dtype=float),
                    color=color,
                    alpha=0.08,
                    linewidth=0,
                )
        ax.set_ylabel(TRAIT_LABELS[trait], fontsize=7)
        ax.set_xlim(date_min, date_max)
        base.style_axis(ax)
    if not pred_site.empty:
        axes[0].legend(frameon=False, ncol=3, loc="upper right", fontsize=7)

    ax = axes[5]
    spectral_colors = [base.BLUE, base.ORANGE, base.GREEN, base.PURPLE, base.SKY, base.VERMILION, base.GREY]
    any_spectral = False
    for name, color in zip(SPECTRAL_PLOT_INDICES, spectral_colors, strict=False):
        col = f"{name}_median"
        if col in spectral_site.columns and pd.to_numeric(spectral_site[col], errors="coerce").notna().any():
            any_spectral = True
            ax.plot(spectral_site["date"], scale01(spectral_site[col]), lw=1.0, color=color, label=name)
    if any_spectral:
        ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=7)
        ax.set_ylim(-0.05, 1.30)
        ax.set_yticks([0.0, 0.5, 1.0])
    else:
        ax.text(0.5, 0.5, "No Sentinel-2 spectral-index context", ha="center", va="center", transform=ax.transAxes, fontsize=8)
    ax.set_ylabel("S2 index\n(0-1)", fontsize=7)
    ax.set_xlim(date_min, date_max)
    base.style_axis(ax)

    ax = axes[6]
    era5_colors = [base.BLUE, base.ORANGE, base.GREEN, base.PURPLE]
    any_era5 = False
    for (col, label), color in zip(ERA5_PLOT_VARS, era5_colors, strict=False):
        if col in era5_site.columns and pd.to_numeric(era5_site[col], errors="coerce").notna().any():
            any_era5 = True
            ax.plot(era5_site["date"], scale01(era5_site[col]), lw=1.0, color=color, label=label)
    if any_era5:
        ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=7)
        ax.set_ylim(-0.05, 1.30)
        ax.set_yticks([0.0, 0.5, 1.0])
    else:
        ax.text(0.5, 0.5, "No ERA5-Land context", ha="center", va="center", transform=ax.transAxes, fontsize=8)
    ax.set_ylabel("ERA5\n(0-1)", fontsize=7)
    ax.set_xlim(date_min, date_max)
    base.style_axis(ax)

    ax = axes[7]
    any_flux = False
    for (var, label), color in zip(FLUXNET_PLOT_VARS, [base.BLUE, base.ORANGE, base.GREEN, base.PURPLE, base.SKY, base.VERMILION, base.GREY, base.BLACK], strict=False):
        if var in flux_site.columns and pd.to_numeric(flux_site[var], errors="coerce").notna().any():
            any_flux = True
            ax.plot(flux_site["date"], scale01(flux_site[var]), lw=1.0, color=color, label=label)
    if any_flux:
        ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=7)
        ax.set_ylim(-0.05, 1.30)
        ax.set_yticks([0.0, 0.5, 1.0])
    else:
        ax.text(0.5, 0.5, "No local/downloaded 2022 FLUXNET product", ha="center", va="center", transform=ax.transAxes, fontsize=8)
    ax.set_ylabel("Flux/stress\n(0-1)", fontsize=7)
    ax.set_xlabel("2022")
    ax.set_xlim(date_min, date_max)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    base.style_axis(ax)

    out = FIG_DIR / f"site_{int(site_id):02d}_cap2000_traits_flux_2022.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_all_sites(
    sites: pd.DataFrame,
    pred: pd.DataFrame,
    spectral: pd.DataFrame,
    era5: pd.DataFrame,
    flux: pd.DataFrame,
    max_sites: int | None = None,
) -> list[Path]:
    site_rows = sites.head(int(max_sites)).copy() if max_sites else sites
    pdf_path = FIG_DIR / "all_sites_cap2000_traits_flux_2022.pdf"
    outputs: list[Path] = []
    with PdfPages(pdf_path) as pdf:
        for site in site_rows.itertuples(index=False):
            outputs.append(plot_site(int(site.site_id), str(site.site_name), pred, spectral, era5, flux, pdf=pdf))
    return [pdf_path, *outputs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot all-site cap2000 biweekly trait predictions with FLUXNET context.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-predictions", action="store_true")
    parser.add_argument("--force-fluxnet", action="store_true")
    parser.add_argument("--force-era5", action="store_true")
    parser.add_argument("--download-fluxnet", action="store_true")
    parser.add_argument("--window-pixels", type=int, default=30)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-sites", type=int, default=0, help="Debug limit; 0 means all sites.")
    parser.add_argument("--use-b5b6-cap5000-transformer", action="store_true", help="Use pulled B5/B6-filtered cap5000 Transformer LAI/LMA artifacts.")
    args = parser.parse_args()

    sites = setup()
    max_sites = int(args.max_sites) if int(args.max_sites) > 0 else None
    mapping = build_fluxnet_mapping(sites, force=bool(args.force))
    flux = build_fluxnet_contexts(
        sites,
        mapping,
        force=bool(args.force or args.force_fluxnet),
        download=bool(args.download_fluxnet),
        max_sites=max_sites,
    )
    era5 = build_era5_contexts(
        sites,
        force=bool(args.force or args.force_era5),
        max_sites=max_sites,
    )
    if bool(args.use_b5b6_cap5000_transformer):
        pred, spectral = build_b5b6_cap5000_transformer_predictions(
            sites,
            force=bool(args.force or args.force_predictions),
            window_pixels=int(args.window_pixels),
            device=str(args.device),
            max_sites=max_sites,
        )
        pred_path = TABLE_DIR / "all_sites_b5b6_cap5000_transformer_biweekly_trait_predictions_2022_summary.csv"
    else:
        pred, spectral = build_trait_predictions(
            sites,
            force=bool(args.force or args.force_predictions),
            window_pixels=int(args.window_pixels),
            device=str(args.device),
            max_sites=max_sites,
        )
        pred_path = TABLE_DIR / "all_sites_cap2000_biweekly_trait_predictions_2022_summary.csv"
    figure_outputs = plot_all_sites(sites, pred, spectral, era5, flux, max_sites=max_sites)
    summary = {
        "mapping": str(TABLE_DIR / "site_fluxnet_mapping.csv"),
        "fluxnet_context": str(TABLE_DIR / "all_sites_fluxnet_context_2022_biweekly.csv"),
        "era5_context": str(TABLE_DIR / "all_sites_era5_context_2022_biweekly.csv"),
        "trait_predictions_summary": str(pred_path),
        "prediction_mode": "b5b6_cap5000_transformer_lai_lma" if bool(args.use_b5b6_cap5000_transformer) else "cap2000_rf_mlp_transformer_5traits",
        "spectral_context": str(TABLE_DIR / "all_sites_cap2000_biweekly_spectral_context_2022.csv"),
        "figures": [str(p) for p in figure_outputs],
        "sites_requested": int(len(sites) if max_sites is None else max_sites),
        "sites_predicted": int(pred["site_id"].nunique()) if not pred.empty else 0,
        "sites_with_era5_values": int(
            era5.groupby("site_id")[[col for col, _label in ERA5_PLOT_VARS if col in era5.columns]]
            .apply(lambda d: pd.to_numeric(d.stack(), errors="coerce").notna().any())
            .sum()
        )
        if not era5.empty
        else 0,
        "sites_with_fluxnet_values": int(
            flux.groupby("site_id")[list({v for v in FLUXNET_CONTEXT_VARS if v in flux.columns})]
            .apply(lambda d: pd.to_numeric(d.stack(), errors="coerce").notna().any())
            .sum()
        )
        if not flux.empty
        else 0,
    }
    (TABLE_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
