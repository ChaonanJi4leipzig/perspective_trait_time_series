#!/usr/bin/env python
"""Build an executed notebook for Fig. 3 WorldCover37 XGB cap20000 checks."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


PAPER_ROOT = Path("/home/chaonan.ji/data/projects/ecospectra/docs/paper1_perspective")
NOTEBOOK_PATH = PAPER_ROOT / "scripts" / "fig3_worldcover37_xgb_cap20000_apr_oct_all_sites.ipynb"


def code_cell(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def markdown_cell(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "traits",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nb.cells = [
        markdown_cell(
            """
# Fig. 3 check: WorldCover37 XGB cap20000, Apr-Oct

This notebook uses the completed `worldcover37_besthp_20260510` XGB `cap_20000` test predictions.
It filters predictions to April-October and keeps the full 37-site site list in all summary views.
Rows without Apr-Oct test samples are retained as missing values instead of being dropped.
"""
        ),
        code_cell(
            r"""
from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, Markdown, display

warnings.filterwarnings("ignore", category=FutureWarning)

RUN_ROOT = Path("/home/chaonan.ji/data/projects/ecospectra/output_data/traits_model/outputs/ablations/worldcover37_besthp_20260510/runs/exp1_data_size/baseline/cap_20000/worldcover37_global_grouped_seed42/xgb")
SITE_META_PATH = Path("/home/chaonan.ji/data/projects/ecospectra/input_data/filtered_sites.csv")
PAPER_ROOT = Path("/home/chaonan.ji/data/projects/ecospectra/docs/paper1_perspective")
TABLE_DIR = PAPER_ROOT / "tables" / "fig3_worldcover37_xgb_cap20000_apr_oct"
FIG_DIR = PAPER_ROOT / "figures" / "generated" / "fig3_worldcover37_xgb_cap20000_apr_oct"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["LAI", "LMA", "Nitrogen", "Chlorophyll", "Lignin"]
MONTH_START, MONTH_END = 4, 10

sns.set_theme(style="whitegrid", context="paper", font_scale=0.95)
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "axes.edgecolor": "#4c4c4c",
    "axes.linewidth": 0.8,
    "font.family": "DejaVu Sans",
})
"""
        ),
        markdown_cell("## Load Apr-Oct XGB test predictions"),
        code_cell(
            r"""
def load_trait_predictions(trait):
    path = RUN_ROOT / trait / "test_predictions.csv"
    header = pd.read_csv(path, nrows=0).columns.tolist()
    true_col = next(col for col in header if col.startswith("true__"))
    pred_col = next(col for col in header if col.startswith("pred__"))
    usecols = [
        "event_id", "sample_id", "site_id", "year", "y", "x",
        "trait_time", "trait_time_index", true_col, pred_col,
    ]
    df = pd.read_csv(path, usecols=usecols)
    df["date"] = pd.to_datetime(df["trait_time"])
    df = df[df["date"].dt.month.between(MONTH_START, MONTH_END)].copy()
    df["trait"] = trait
    df = df.rename(columns={true_col: "true", pred_col: "pred"})
    return df[[
        "event_id", "sample_id", "site_id", "year", "date",
        "trait_time_index", "y", "x", "trait", "true", "pred",
    ]]

pred = pd.concat([load_trait_predictions(trait) for trait in TRAITS], ignore_index=True)

site_meta = (
    pd.read_csv(SITE_META_PATH)
    [["site_id", "site_name", "location_lat", "location_lon"]]
    .drop_duplicates("site_id")
    .sort_values("site_id")
)
site_order = site_meta["site_id"].tolist()
site_meta["site_label"] = site_meta["site_id"].map(lambda x: f"{int(x):02d}") + " " + site_meta["site_name"]

coverage = (
    pred.groupby(["site_id", "year"], as_index=False)
    .size()
    .rename(columns={"size": "n_predictions"})
)
coverage_wide = (
    coverage.pivot(index="site_id", columns="year", values="n_predictions")
    .reindex(site_order)
    .fillna(0)
    .astype(int)
    .reset_index()
    .merge(site_meta[["site_id", "site_name"]], on="site_id", how="left")
)
coverage_wide = coverage_wide[["site_id", "site_name"] + [c for c in coverage_wide.columns if isinstance(c, int)]]
coverage_wide.to_csv(TABLE_DIR / "coverage_by_site_year_apr_oct.csv", index=False)

print(f"Loaded {len(pred):,} Apr-Oct prediction rows across {pred['site_id'].nunique()} sites.")
print(f"Full site list: {len(site_order)} sites.")
missing_sites = sorted(set(site_order) - set(pred["site_id"].unique()))
print("Sites with no Apr-Oct test predictions:", missing_sites if missing_sites else "none")
display(coverage_wide)
"""
        ),
        markdown_cell("## Site-trait metrics"),
        code_cell(
            r"""
def metric_row(group):
    y_true = group["true"].to_numpy(dtype=float)
    y_pred = group["pred"].to_numpy(dtype=float)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    n = int(y_true.size)
    if n == 0:
        return pd.Series({"n_valid": 0, "r2": np.nan, "rmse": np.nan, "nrmse_pct": np.nan, "bias": np.nan})
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    trait_range = float(np.nanmax(y_true) - np.nanmin(y_true))
    nrmse_pct = float(rmse / trait_range * 100.0) if trait_range > 0 else np.nan
    if np.unique(y_true).size > 1:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        r2 = np.nan
    return pd.Series({
        "n_valid": n,
        "r2": r2,
        "rmse": rmse,
        "nrmse_pct": nrmse_pct,
        "bias": float(np.mean(y_pred - y_true)),
    })

metrics = pred.groupby(["site_id", "trait"], sort=False).apply(metric_row).reset_index()
full_index = pd.MultiIndex.from_product([site_order, TRAITS], names=["site_id", "trait"]).to_frame(index=False)
metrics = full_index.merge(metrics, on=["site_id", "trait"], how="left")
metrics["n_valid"] = metrics["n_valid"].fillna(0).astype(int)
metrics = metrics.merge(site_meta, on="site_id", how="left")
metrics.to_csv(TABLE_DIR / "site_trait_metrics_apr_oct.csv", index=False)

summary = (
    metrics.groupby("trait", as_index=False)
    .agg(
        sites_with_data=("n_valid", lambda s: int((s > 0).sum())),
        median_r2=("r2", "median"),
        median_nrmse_pct=("nrmse_pct", "median"),
        total_samples=("n_valid", "sum"),
    )
)
summary.to_csv(TABLE_DIR / "trait_metric_summary_apr_oct.csv", index=False)
display(summary)
display(metrics[["site_id", "site_name", "trait", "n_valid", "r2", "nrmse_pct", "rmse", "bias"]])
"""
        ),
        markdown_cell("## R2 and nRMSE heatmaps"),
        code_cell(
            r"""
def heatmap_matrix(value_col):
    mat = metrics.pivot(index="site_label", columns="trait", values=value_col)
    site_labels = site_meta.set_index("site_id").loc[site_order, "site_label"].tolist()
    return mat.reindex(index=site_labels, columns=TRAITS)

r2_mat = heatmap_matrix("r2")
nrmse_mat = heatmap_matrix("nrmse_pct")

fig, axes = plt.subplots(1, 2, figsize=(10.8, 11.5), constrained_layout=True)
sns.heatmap(
    r2_mat,
    ax=axes[0],
    cmap="viridis",
    vmin=0,
    vmax=1,
    linewidths=0.25,
    linecolor="white",
    cbar_kws={"label": "R2"},
)
axes[0].set_title("XGB cap20000 Apr-Oct R2")
axes[0].set_xlabel("")
axes[0].set_ylabel("")

upper = np.nanpercentile(nrmse_mat.to_numpy(dtype=float), 95)
vmax = max(20.0, float(upper)) if np.isfinite(upper) else 50.0
sns.heatmap(
    nrmse_mat,
    ax=axes[1],
    cmap="mako_r",
    vmin=0,
    vmax=vmax,
    linewidths=0.25,
    linecolor="white",
    cbar_kws={"label": "nRMSE (%)"},
)
axes[1].set_title("XGB cap20000 Apr-Oct nRMSE")
axes[1].set_xlabel("")
axes[1].set_ylabel("")
for ax in axes:
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", labelsize=7)

heatmap_png = FIG_DIR / "worldcover37_xgb_cap20000_apr_oct_skill_heatmaps.png"
heatmap_pdf = FIG_DIR / "worldcover37_xgb_cap20000_apr_oct_skill_heatmaps.pdf"
fig.savefig(heatmap_png, bbox_inches="tight")
fig.savefig(heatmap_pdf, bbox_inches="tight")
plt.show()
print(heatmap_png)
print(heatmap_pdf)
"""
        ),
        markdown_cell("## Daily median observed vs predicted values"),
        code_cell(
            r"""
daily = (
    pred.groupby(["site_id", "date", "trait"], as_index=False)
    .agg(
        true_median=("true", "median"),
        pred_median=("pred", "median"),
        true_q25=("true", lambda s: float(np.nanquantile(s, 0.25))),
        true_q75=("true", lambda s: float(np.nanquantile(s, 0.75))),
        pred_q25=("pred", lambda s: float(np.nanquantile(s, 0.25))),
        pred_q75=("pred", lambda s: float(np.nanquantile(s, 0.75))),
        n=("pred", "size"),
    )
    .merge(site_meta, on="site_id", how="left")
)
daily.to_csv(TABLE_DIR / "daily_site_trait_medians_apr_oct.csv", index=False)
display(daily.head())
print(f"Daily median table rows: {len(daily):,}")
"""
        ),
        markdown_cell("## Coverage heatmap"),
        code_cell(
            r"""
count_mat = (
    pred.groupby(["site_id", "year"], as_index=False)
    .size()
    .pivot(index="site_id", columns="year", values="size")
    .reindex(site_order)
    .fillna(0)
)
count_mat.index = site_meta.set_index("site_id").loc[site_order, "site_label"]

fig, ax = plt.subplots(figsize=(5.6, 10.5), constrained_layout=True)
sns.heatmap(
    count_mat,
    ax=ax,
    cmap="Blues",
    linewidths=0.25,
    linecolor="white",
    cbar_kws={"label": "Apr-Oct test predictions"},
)
ax.set_title("Prediction coverage by site and year")
ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(axis="y", labelsize=7)
coverage_png = FIG_DIR / "worldcover37_xgb_cap20000_apr_oct_coverage.png"
fig.savefig(coverage_png, bbox_inches="tight")
plt.show()
print(coverage_png)
"""
        ),
        markdown_cell("## Per-site time series pages"),
        code_cell(
            r"""
trait_colors = {
    "LAI": "#1b7837",
    "LMA": "#5aae61",
    "Nitrogen": "#2166ac",
    "Chlorophyll": "#d6604d",
    "Lignin": "#762a83",
}

def plot_site(site_id, show=False):
    info = site_meta.loc[site_meta["site_id"] == site_id].iloc[0]
    site_label = info["site_label"]
    site_daily = daily[daily["site_id"] == site_id].copy()
    fig, axes = plt.subplots(len(TRAITS), 1, figsize=(9.0, 7.0), sharex=True, constrained_layout=True)
    fig.suptitle(f"{site_label} | XGB cap20000 Apr-Oct test predictions", y=1.02, fontsize=12)
    for ax, trait in zip(axes, TRAITS):
        sub = site_daily[site_daily["trait"] == trait].sort_values("date")
        color = trait_colors[trait]
        if sub.empty:
            ax.text(0.5, 0.5, "No Apr-Oct test samples", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        ax.plot(sub["date"], sub["true_median"], color=color, lw=1.5, marker="o", ms=2.5, label="Observed median")
        ax.plot(sub["date"], sub["pred_median"], color="#222222", lw=1.2, marker="s", ms=2.2, label="Predicted median")
        ax.fill_between(sub["date"], sub["true_q25"], sub["true_q75"], color=color, alpha=0.12, linewidth=0)
        ax.fill_between(sub["date"], sub["pred_q25"], sub["pred_q75"], color="#222222", alpha=0.08, linewidth=0)
        ax.set_ylabel(trait, rotation=0, ha="right", va="center")
        ax.grid(True, alpha=0.25)
        m = metrics[(metrics["site_id"] == site_id) & (metrics["trait"] == trait)].iloc[0]
        label = f"n={int(m['n_valid'])}"
        if np.isfinite(m["r2"]):
            label += f", R2={m['r2']:.2f}"
        if np.isfinite(m["nrmse_pct"]):
            label += f", nRMSE={m['nrmse_pct']:.1f}%"
        ax.text(0.99, 0.82, label, transform=ax.transAxes, ha="right", va="top", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 1.02), frameon=False, ncol=2)
    out = FIG_DIR / f"site_{int(site_id):02d}_xgb_cap20000_apr_oct_timeseries.png"
    fig.savefig(out, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out

site_figures = [plot_site(site_id, show=False) for site_id in site_order]
print(f"Wrote {len(site_figures)} per-site figures to {FIG_DIR}")
"""
        ),
        markdown_cell("## All 37 site figures"),
        code_cell(
            r"""
for site_id, fig_path in zip(site_order, site_figures):
    label = site_meta.loc[site_meta["site_id"] == site_id, "site_label"].iloc[0]
    display(Markdown(f"### {label}"))
    display(Image(filename=str(fig_path)))
"""
        ),
    ]
    return nb


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    nbf.write(nb, NOTEBOOK_PATH)

    client = NotebookClient(
        nb,
        timeout=1200,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Wrote executed notebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
