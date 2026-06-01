#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PAPER_ROOT = Path(__file__).resolve().parents[1]
TABLE_ROOT = PAPER_ROOT / "tables" / "fig4_drought_anchor_xgb"
PREDICTION_TABLE = TABLE_ROOT / "cap5000_sites_xgb_biweekly_trait_predictions_2022_2023aug_summary.csv"
RECOVERY_TABLE = TABLE_ROOT / "cap5000_sites_recovery_timeline_2022_2023aug.csv"
FIG_DIR = PAPER_ROOT / "figures" / "generated"
OUT_STEM = "fig4_real_trait_space_trajectory"

BASELINE_START = pd.Timestamp("2022-05-14")
BASELINE_END = pd.Timestamp("2022-06-25")
TROUGH_START = pd.Timestamp("2022-07-23")
TROUGH_END = pd.Timestamp("2022-08-20")
SAME_SEASON_START = pd.Timestamp("2023-05-13")
SAME_SEASON_END = pd.Timestamp("2023-06-24")
LATE_SUMMER_START = pd.Timestamp("2023-07-08")
LATE_SUMMER_END = pd.Timestamp("2023-08-19")

SITE_IDS = [43, 44]
SITE_NAMES = {43: "Le Bray", 44: "Bilos"}
SITE_COLORS = {43: "#0072B2", 44: "#D55E00"}
STAGE_COLORS = {
    "May-Jun 2022": "#009E73",
    "Jul-Aug 2022": "#D55E00",
    "May-Jun 2023": "#0072B2",
    "Jul-Aug 2023": "#CC79A7",
}


def setup_style() -> None:
    mpl.use("Agg")
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
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
            "savefig.dpi": 300,
        }
    )


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(width=0.5, length=3)
    ax.grid(False)


def panel_label(ax, label: str) -> None:
    ax.text(0.01, 0.99, label, transform=ax.transAxes, ha="left", va="top", fontsize=8, fontweight="bold")


def window_label(date: pd.Timestamp) -> str | None:
    if BASELINE_START <= date <= BASELINE_END:
        return "May-Jun 2022"
    if TROUGH_START <= date <= TROUGH_END:
        return "Jul-Aug 2022"
    if SAME_SEASON_START <= date <= SAME_SEASON_END:
        return "May-Jun 2023"
    if LATE_SUMMER_START <= date <= LATE_SUMMER_END:
        return "Jul-Aug 2023"
    return None


def load_prediction_space() -> pd.DataFrame:
    df = pd.read_csv(PREDICTION_TABLE, parse_dates=["date"])
    sub = df[df["site_id"].isin(SITE_IDS) & df["trait"].isin(["LAI", "Chlorophyll"])].copy()
    if sub.empty:
        raise ValueError(f"No LAI/chlorophyll rows in {PREDICTION_TABLE}")
    wide = (
        sub.pivot_table(index=["site_id", "date"], columns="trait", values="prediction_median", aggfunc="median")
        .reset_index()
        .dropna(subset=["LAI", "Chlorophyll"])
        .sort_values(["site_id", "date"])
    )
    wide["window_label"] = wide["date"].map(window_label)
    return wide


def load_recovery_space() -> pd.DataFrame:
    df = pd.read_csv(RECOVERY_TABLE, parse_dates=["date"])
    sub = df[
        df["variable"].isin(["Chlorophyll", "LMA"])
        & (df["date"] >= SAME_SEASON_START)
        & (df["date"] <= SAME_SEASON_END)
    ].copy()
    if sub.empty:
        raise ValueError(f"No same-season recovery rows in {RECOVERY_TABLE}")
    med = (
        sub.groupby(["site_id", "site_name", "variable"], as_index=False)["recovery_pct_clipped"]
        .median()
        .pivot(index=["site_id", "site_name"], columns="variable", values="recovery_pct_clipped")
        .reset_index()
        .dropna(subset=["Chlorophyll", "LMA"])
    )
    return med


def draw_site_trajectories(ax, wide: pd.DataFrame) -> None:
    for site_id, site_df in wide.groupby("site_id"):
        color = SITE_COLORS[int(site_id)]
        ax.plot(
            site_df["LAI"],
            site_df["Chlorophyll"],
            color=color,
            lw=1.0,
            alpha=0.8,
            label=SITE_NAMES[int(site_id)],
        )
        ax.scatter(
            site_df["LAI"],
            site_df["Chlorophyll"],
            s=10,
            color=color,
            alpha=0.35,
            edgecolor="none",
        )
        for label, stage_color in STAGE_COLORS.items():
            stage = site_df[site_df["window_label"] == label]
            if stage.empty:
                continue
            ax.scatter(
                stage["LAI"].median(),
                stage["Chlorophyll"].median(),
                s=42,
                facecolor=stage_color,
                edgecolor="white",
                linewidth=0.4,
                zorder=5,
            )
    ax.set_xlabel("LAI (m2 m-2)")
    ax.set_ylabel("Chlorophyll (ug cm-2)")
    ax.set_title("Le Bray and Bilos, biweekly states", fontsize=7)
    style_axis(ax)
    panel_label(ax, "a")


def draw_recovery_space(ax, recovery: pd.DataFrame) -> None:
    ax.scatter(
        recovery["Chlorophyll"],
        recovery["LMA"],
        s=18,
        color="#777777",
        alpha=0.55,
        edgecolor="white",
        linewidth=0.25,
    )
    for site_id in SITE_IDS:
        hit = recovery[recovery["site_id"] == site_id]
        if hit.empty:
            continue
        ax.scatter(
            hit["Chlorophyll"],
            hit["LMA"],
            s=54,
            color=SITE_COLORS[site_id],
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
            label=SITE_NAMES[site_id],
        )
        y = float(hit["LMA"].iloc[0])
        y_offset = -4.0 if y > 94.0 else 1.5
        ax.text(
            float(hit["Chlorophyll"].iloc[0]) + 1.5,
            y + y_offset,
            SITE_NAMES[site_id],
            fontsize=7,
            color=SITE_COLORS[site_id],
        )
    ax.axvline(70, color="#444444", lw=0.6, ls=":")
    ax.axhline(70, color="#444444", lw=0.6, ls=":")
    ax.plot([0, 105], [0, 105], color="black", lw=0.6, alpha=0.7)
    ax.set_xlim(-2, 105)
    ax.set_ylim(-2, 105)
    ax.set_xlabel("Chlorophyll recovery, May-Jun 2023 (%)")
    ax.set_ylabel("LMA recovery, May-Jun 2023 (%)")
    ax.set_title("Trait-space recovery across 37 sites", fontsize=7)
    ax.text(0.97, 0.04, f"n={recovery['site_id'].nunique()}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    style_axis(ax)
    panel_label(ax, "b")


def main() -> None:
    setup_style()
    wide = load_prediction_space()
    recovery = load_recovery_space()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    draw_site_trajectories(axes[0], wide)
    draw_recovery_space(axes[1], recovery)

    site_handles = [Line2D([0], [0], color=SITE_COLORS[s], lw=1.2, label=SITE_NAMES[s]) for s in SITE_IDS]
    stage_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markeredgecolor="white", markersize=5, label=l)
        for l, c in STAGE_COLORS.items()
    ]
    axes[0].legend(handles=site_handles + stage_handles, frameon=False, loc="upper left", handlelength=1.2, labelspacing=0.25)

    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.18, wspace=0.28)
    for ext in ("pdf", "svg", "eps", "png", "jpg"):
        fig.savefig(FIG_DIR / f"{OUT_STEM}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(FIG_DIR / f"{OUT_STEM}.pdf")


if __name__ == "__main__":
    main()
