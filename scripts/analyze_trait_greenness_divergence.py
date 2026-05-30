#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER_ROOT = Path(__file__).resolve().parents[1]
TABLE_ROOT = PAPER_ROOT / "tables" / "fig4_drought_anchor_xgb"
DEFAULT_RECOVERY_TABLE = TABLE_ROOT / "cap5000_sites_recovery_timeline_2022_2023aug.csv"
DEFAULT_OUTPUT_DIR = PAPER_ROOT / "tables" / "trait_greenness_divergence_2022_2023aug"
DEFAULT_FIGURE_DIR = PAPER_ROOT / "figures" / "generated" / "trait_greenness_divergence_2022_2023aug"

BASELINE_START = pd.Timestamp("2022-05-14")
BASELINE_END = pd.Timestamp("2022-06-25")
TROUGH_START = pd.Timestamp("2022-07-23")
TROUGH_END = pd.Timestamp("2022-08-20")
APPARENT_START = pd.Timestamp("2022-09-17")
APPARENT_END = pd.Timestamp("2022-10-29")
LATE_SUMMER_START = pd.Timestamp("2023-07-08")
LATE_SUMMER_END = pd.Timestamp("2023-08-19")

TRAITS = ["EWT", "Chlorophyll", "LAI", "Nitrogen", "LMA"]
GREENNESS_INDICES = ["NDVI", "EVI"]
SPECTRAL_INDICES = ["NDVI", "EVI", "NDWI", "NBR", "CIRE", "NDRE", "MSI"]

TRAIT_COLORS = {
    "EWT": "#0072B2",
    "Chlorophyll": "#009E73",
    "LAI": "#CC79A7",
    "Nitrogen": "#D55E00",
    "LMA": "#E69F00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify dynamic trait time series by agreement, early-warning, "
            "and legacy-effect evidence relative to greenness/spectral indices."
        )
    )
    parser.add_argument("--recovery-table", type=Path, default=DEFAULT_RECOVERY_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--late-window-start", default="2023-05-13")
    parser.add_argument("--late-window-end", default="2023-06-24")
    parser.add_argument("--recovery-threshold", type=float, default=70.0)
    parser.add_argument("--stress-threshold", type=float, default=50.0)
    parser.add_argument("--lead-days", type=int, default=14)
    parser.add_argument("--delay-days", type=int, default=28)
    parser.add_argument("--legacy-gap", type=float, default=20.0)
    parser.add_argument("--consistent-corr", type=float, default=0.55)
    parser.add_argument("--consistent-gap", type=float, default=15.0)
    return parser.parse_args()


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
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(width=0.6, length=3)


def panel_label(ax, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def load_recovery_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["date"])
    required = {
        "site_id",
        "site_name",
        "date",
        "stage",
        "variable",
        "variable_label",
        "recovery_pct_clipped",
        "recovery_interpretable",
        "variable_family",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    df["site_id"] = df["site_id"].astype(int)
    df["recovery_pct_clipped"] = pd.to_numeric(df["recovery_pct_clipped"], errors="coerce")
    df["recovery_interpretable"] = df["recovery_interpretable"].astype(bool)
    return df


def build_reference_rows(df: pd.DataFrame, variables: list[str], name: str, label: str) -> pd.DataFrame:
    sub = df[df["variable"].isin(variables)].copy()
    if sub.empty:
        return pd.DataFrame(columns=df.columns)
    rows = (
        sub.groupby(["site_id", "site_name", "date"], as_index=False)
        .agg(
            recovery_pct_clipped=("recovery_pct_clipped", "median"),
            recovery_interpretable=("recovery_interpretable", lambda s: bool(np.nanmean(s.astype(float)) >= 0.5)),
        )
        .sort_values(["site_id", "date"])
    )
    rows["relative_days"] = rows.get("relative_days", np.nan)
    rows["stage"] = rows["date"].map(stage_for_date)
    rows["variable"] = name
    rows["variable_label"] = label
    rows["value"] = np.nan
    rows["baseline_median"] = np.nan
    rows["trough_median"] = np.nan
    rows["recovery_pct"] = rows["recovery_pct_clipped"]
    rows["variable_family"] = "reference"
    return rows[df.columns.intersection(rows.columns).tolist()]


def stage_for_date(date: pd.Timestamp) -> str:
    date = pd.Timestamp(date)
    if BASELINE_START <= date <= BASELINE_END:
        return "pre-event"
    if TROUGH_START <= date <= TROUGH_END:
        return "drought trough"
    if APPARENT_START <= date <= APPARENT_END:
        return "apparent recovery"
    if LATE_SUMMER_START <= date <= LATE_SUMMER_END:
        return "2023 late summer"
    return "other"


def first_date_at_or_below(df: pd.DataFrame, column: str, threshold: float) -> pd.Timestamp | pd.NaT:
    hit = df[np.isfinite(df[column]) & (df[column] <= threshold)]["date"]
    if hit.empty:
        return pd.NaT
    return pd.Timestamp(hit.min())


def first_date_at_or_above(df: pd.DataFrame, column: str, threshold: float) -> pd.Timestamp | pd.NaT:
    hit = df[np.isfinite(df[column]) & (df[column] >= threshold)]["date"]
    if hit.empty:
        return pd.NaT
    return pd.Timestamp(hit.min())


def days_between(a: pd.Timestamp | pd.NaT, b: pd.Timestamp | pd.NaT) -> float:
    if pd.isna(a) or pd.isna(b):
        return math.nan
    return float((pd.Timestamp(a) - pd.Timestamp(b)).days)


def window_median(df: pd.DataFrame, column: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
    sub = df[(df["date"] >= start) & (df["date"] <= end)]
    if sub.empty:
        return math.nan
    vals = pd.to_numeric(sub[column], errors="coerce")
    if not np.isfinite(vals).any():
        return math.nan
    return float(np.nanmedian(vals))


def classify_site(row: pd.Series, args: argparse.Namespace) -> str:
    labels: list[str] = []
    if not bool(row["trait_interpretable"]) or not bool(row["greenness_interpretable"]):
        return "not_interpretable"
    if (
        np.isfinite(row["spearman_greenness"])
        and row["spearman_greenness"] >= args.consistent_corr
        and np.isfinite(row["late_gap_trait_minus_green"])
        and abs(row["late_gap_trait_minus_green"]) <= args.consistent_gap
    ):
        labels.append("consistent_with_greenness")
    if np.isfinite(row["onset_lead_days"]) and row["onset_lead_days"] <= -args.lead_days:
        labels.append("early_warning_candidate")
    if bool(row["legacy_effect"]):
        labels.append("legacy_effect_candidate")
    if (
        np.isfinite(row["recovery_lag_days"])
        and row["recovery_lag_days"] >= args.delay_days
        and not bool(row["legacy_effect"])
    ):
        labels.append("delayed_recovery_candidate")
    if not labels:
        labels.append("mixed_or_uncertain")
    return ";".join(labels)


def evidence_labels(row: pd.Series, args: argparse.Namespace) -> str:
    labels: list[str] = []
    if (
        np.isfinite(row["median_spearman_greenness"])
        and row["median_spearman_greenness"] >= args.consistent_corr
        and np.isfinite(row["median_late_gap_trait_minus_green"])
        and abs(row["median_late_gap_trait_minus_green"]) <= args.consistent_gap
        and row["frac_legacy_effect_sites"] < 0.20
    ):
        labels.append("consistent_with_greenness")
    if row["frac_early_warning_sites"] >= 0.30:
        labels.append("early_warning_candidate")
    if row["frac_legacy_effect_sites"] >= 0.25 or row["median_late_gap_trait_minus_green"] <= -args.legacy_gap:
        labels.append("legacy_effect_candidate")
    if np.isfinite(row["median_recovery_lag_days"]) and row["median_recovery_lag_days"] >= args.delay_days:
        labels.append("delayed_recovery_candidate")
    if not labels:
        labels.append("mixed_or_uncertain")
    return ";".join(labels)


def compute_site_trait_metrics(
    recovery: pd.DataFrame,
    greenness: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    late_start = pd.Timestamp(args.late_window_start)
    late_end = pd.Timestamp(args.late_window_end)
    rows: list[dict[str, object]] = []

    for trait in TRAITS:
        trait_df = recovery[recovery["variable"] == trait].copy()
        for site_id, site_trait in trait_df.groupby("site_id"):
            site_green = greenness[greenness["site_id"] == int(site_id)].copy()
            merged = (
                site_trait[
                    [
                        "site_id",
                        "site_name",
                        "date",
                        "recovery_pct_clipped",
                        "recovery_interpretable",
                    ]
                ]
                .merge(
                    site_green[
                        [
                            "date",
                            "recovery_pct_clipped",
                            "recovery_interpretable",
                        ]
                    ],
                    on="date",
                    how="inner",
                    suffixes=("_trait", "_green"),
                )
                .sort_values("date")
            )
            if merged.empty:
                continue
            focus = merged[(merged["date"] >= BASELINE_START) & (merged["date"] <= late_end)]
            valid = focus.dropna(subset=["recovery_pct_clipped_trait", "recovery_pct_clipped_green"])
            corr = (
                float(valid["recovery_pct_clipped_trait"].corr(valid["recovery_pct_clipped_green"], method="spearman"))
                if len(valid) >= 4
                else math.nan
            )
            onset_window = merged[(merged["date"] > BASELINE_END) & (merged["date"] <= TROUGH_END)]
            trait_onset = first_date_at_or_below(onset_window, "recovery_pct_clipped_trait", args.stress_threshold)
            green_onset = first_date_at_or_below(onset_window, "recovery_pct_clipped_green", args.stress_threshold)

            recovery_window = merged[(merged["date"] > TROUGH_END) & (merged["date"] <= late_end)]
            trait_recovered = first_date_at_or_above(
                recovery_window,
                "recovery_pct_clipped_trait",
                args.recovery_threshold,
            )
            green_recovered = first_date_at_or_above(
                recovery_window,
                "recovery_pct_clipped_green",
                args.recovery_threshold,
            )

            trait_late = window_median(merged, "recovery_pct_clipped_trait", late_start, late_end)
            green_late = window_median(merged, "recovery_pct_clipped_green", late_start, late_end)
            trait_apparent = window_median(merged, "recovery_pct_clipped_trait", APPARENT_START, APPARENT_END)
            green_apparent = window_median(merged, "recovery_pct_clipped_green", APPARENT_START, APPARENT_END)
            trait_late_summer = window_median(
                merged,
                "recovery_pct_clipped_trait",
                LATE_SUMMER_START,
                LATE_SUMMER_END,
            )
            green_late_summer = window_median(
                merged,
                "recovery_pct_clipped_green",
                LATE_SUMMER_START,
                LATE_SUMMER_END,
            )
            legacy = (
                np.isfinite(green_late)
                and np.isfinite(trait_late)
                and green_late >= args.recovery_threshold
                and trait_late < args.recovery_threshold
                and (green_late - trait_late) >= args.legacy_gap
            )

            rows.append(
                {
                    "site_id": int(site_id),
                    "site_name": str(merged["site_name"].iloc[0]),
                    "trait": trait,
                    "trait_interpretable": bool(merged["recovery_interpretable_trait"].max()),
                    "greenness_interpretable": bool(merged["recovery_interpretable_green"].max()),
                    "spearman_greenness": corr,
                    "trait_stress_onset_date": trait_onset.date().isoformat() if pd.notna(trait_onset) else "",
                    "greenness_stress_onset_date": green_onset.date().isoformat() if pd.notna(green_onset) else "",
                    "onset_lead_days": days_between(trait_onset, green_onset),
                    "trait_recovery_date": trait_recovered.date().isoformat() if pd.notna(trait_recovered) else "",
                    "greenness_recovery_date": green_recovered.date().isoformat() if pd.notna(green_recovered) else "",
                    "recovery_lag_days": days_between(trait_recovered, green_recovered),
                    "apparent_trait_recovery": trait_apparent,
                    "apparent_greenness_recovery": green_apparent,
                    "late_trait_recovery": trait_late,
                    "late_greenness_recovery": green_late,
                    "late_gap_trait_minus_green": trait_late - green_late,
                    "late_summer_trait_recovery": trait_late_summer,
                    "late_summer_greenness_recovery": green_late_summer,
                    "late_summer_gap_trait_minus_green": trait_late_summer - green_late_summer,
                    "early_warning": bool(np.isfinite(days_between(trait_onset, green_onset)) and days_between(trait_onset, green_onset) <= -args.lead_days),
                    "legacy_effect": bool(legacy),
                }
            )

    site_metrics = pd.DataFrame(rows)
    site_metrics["classification"] = site_metrics.apply(lambda row: classify_site(row, args), axis=1)
    return site_metrics.sort_values(["trait", "site_id"])


def compute_trait_summary(site_metrics: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    def finite_median(series: pd.Series) -> float:
        vals = pd.to_numeric(series, errors="coerce")
        if not np.isfinite(vals).any():
            return math.nan
        return float(np.nanmedian(vals))

    summary = (
        site_metrics.groupby("trait", as_index=False)
        .agg(
            n_sites=("site_id", "nunique"),
            n_interpretable_sites=("trait_interpretable", "sum"),
            median_spearman_greenness=("spearman_greenness", finite_median),
            median_onset_lead_days=("onset_lead_days", finite_median),
            median_recovery_lag_days=("recovery_lag_days", finite_median),
            median_abs_recovery_lag_days=("recovery_lag_days", lambda s: finite_median(pd.Series(np.abs(pd.to_numeric(s, errors="coerce"))))),
            median_apparent_trait_recovery=("apparent_trait_recovery", finite_median),
            median_apparent_greenness_recovery=("apparent_greenness_recovery", finite_median),
            median_late_trait_recovery=("late_trait_recovery", finite_median),
            median_late_greenness_recovery=("late_greenness_recovery", finite_median),
            median_late_gap_trait_minus_green=("late_gap_trait_minus_green", finite_median),
            median_late_summer_trait_recovery=("late_summer_trait_recovery", finite_median),
            median_late_summer_greenness_recovery=("late_summer_greenness_recovery", finite_median),
            frac_early_warning_sites=("early_warning", "mean"),
            frac_legacy_effect_sites=("legacy_effect", "mean"),
        )
        .sort_values("trait")
    )
    summary["frac_interpretable_sites"] = summary["n_interpretable_sites"] / summary["n_sites"]
    consistent = site_metrics["classification"].str.contains("consistent_with_greenness", regex=False)
    consistent_by_trait = consistent.groupby(site_metrics["trait"]).mean().rename("frac_consistent_sites")
    summary = summary.merge(consistent_by_trait, on="trait", how="left")
    summary["evidence_labels"] = summary.apply(lambda row: evidence_labels(row, args), axis=1)
    ordered = [
        "trait",
        "evidence_labels",
        "n_sites",
        "n_interpretable_sites",
        "frac_interpretable_sites",
        "median_spearman_greenness",
        "frac_consistent_sites",
        "frac_early_warning_sites",
        "frac_legacy_effect_sites",
        "median_onset_lead_days",
        "median_recovery_lag_days",
        "median_abs_recovery_lag_days",
        "median_apparent_trait_recovery",
        "median_apparent_greenness_recovery",
        "median_late_trait_recovery",
        "median_late_greenness_recovery",
        "median_late_gap_trait_minus_green",
        "median_late_summer_trait_recovery",
        "median_late_summer_greenness_recovery",
    ]
    return summary[ordered]


def compute_trait_index_similarity(recovery: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    late_start = pd.Timestamp(args.late_window_start)
    late_end = pd.Timestamp(args.late_window_end)
    rows: list[dict[str, object]] = []
    for trait in TRAITS:
        trait_df = recovery[recovery["variable"] == trait]
        for index in SPECTRAL_INDICES:
            index_df = recovery[recovery["variable"] == index]
            site_corrs: list[float] = []
            site_gaps: list[float] = []
            for site_id in sorted(set(trait_df["site_id"]).intersection(index_df["site_id"])):
                t = trait_df[trait_df["site_id"] == site_id][["date", "recovery_pct_clipped"]]
                i = index_df[index_df["site_id"] == site_id][["date", "recovery_pct_clipped"]]
                merged = (
                    t.merge(i, on="date", suffixes=("_trait", "_index"))
                    .sort_values("date")
                    .dropna(subset=["recovery_pct_clipped_trait", "recovery_pct_clipped_index"])
                )
                focus = merged[(merged["date"] >= BASELINE_START) & (merged["date"] <= late_end)]
                if len(focus) >= 4:
                    site_corrs.append(
                        float(
                            focus["recovery_pct_clipped_trait"].corr(
                                focus["recovery_pct_clipped_index"],
                                method="spearman",
                            )
                        )
                    )
                late = merged[(merged["date"] >= late_start) & (merged["date"] <= late_end)]
                if not late.empty:
                    site_gaps.append(
                        float(
                            np.nanmedian(late["recovery_pct_clipped_trait"])
                            - np.nanmedian(late["recovery_pct_clipped_index"])
                        )
                    )
            rows.append(
                {
                    "trait": trait,
                    "index": index,
                    "n_sites": len(site_corrs),
                    "median_spearman": float(np.nanmedian(site_corrs)) if site_corrs else math.nan,
                    "median_late_gap_trait_minus_index": float(np.nanmedian(site_gaps)) if site_gaps else math.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["trait", "index"])


def compute_window_summary(recovery: pd.DataFrame, augmented: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    late_start = pd.Timestamp(args.late_window_start)
    late_end = pd.Timestamp(args.late_window_end)
    combined = pd.concat([recovery, augmented], ignore_index=True, sort=False)
    specs = [
        ("pre_event_2022", BASELINE_START, BASELINE_END),
        ("drought_trough_2022", TROUGH_START, TROUGH_END),
        ("apparent_recovery_2022", APPARENT_START, APPARENT_END),
        ("same_season_2023", late_start, late_end),
        ("late_summer_2023", LATE_SUMMER_START, LATE_SUMMER_END),
    ]
    rows = []
    keep = TRAITS + SPECTRAL_INDICES + ["GREENNESS", "SPECTRAL_INDEX"]
    for variable in keep:
        sub = combined[combined["variable"] == variable].copy()
        if sub.empty:
            continue
        family = str(sub["variable_family"].iloc[0])
        label = str(sub["variable_label"].iloc[0])
        for window, start, end in specs:
            win = sub[(sub["date"] >= start) & (sub["date"] <= end)]
            rows.append(
                {
                    "variable": variable,
                    "variable_label": label,
                    "variable_family": family,
                    "window": window,
                    "start": start.date().isoformat(),
                    "end": end.date().isoformat(),
                    "median_recovery_pct": float(np.nanmedian(win["recovery_pct_clipped"])) if not win.empty else math.nan,
                    "p25_recovery_pct": float(np.nanquantile(win["recovery_pct_clipped"], 0.25)) if not win.empty else math.nan,
                    "p75_recovery_pct": float(np.nanquantile(win["recovery_pct_clipped"], 0.75)) if not win.empty else math.nan,
                    "n_rows": int(len(win)),
                    "n_sites": int(win["site_id"].nunique()) if not win.empty else 0,
                    "interpretable_site_fraction": float(
                        win.groupby("site_id")["recovery_interpretable"].max().mean()
                    )
                    if not win.empty
                    else math.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["variable_family", "variable", "window"])


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot_summary(
    trait_summary: pd.DataFrame,
    trait_index: pd.DataFrame,
    window_summary: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    traits = TRAITS
    indices = SPECTRAL_INDICES
    corr = trait_index.pivot(index="trait", columns="index", values="median_spearman").reindex(index=traits, columns=indices)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    ax = axes[0, 0]
    im = ax.imshow(corr.values, vmin=-0.4, vmax=0.8, cmap="RdYlBu")
    ax.set_xticks(np.arange(len(indices)), indices, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(traits)), traits)
    for i, trait in enumerate(traits):
        for j, index in enumerate(indices):
            val = corr.loc[trait, index]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5)
    ax.set_xlabel("Spectral index")
    ax.set_ylabel("Trait")
    panel_label(ax, "a")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Median Spearman")

    ax = axes[0, 1]
    summary = trait_summary.set_index("trait").reindex(traits)
    x = np.arange(len(traits))
    width = 0.25
    ax.bar(x - width, summary["frac_consistent_sites"], width=width, color="#4C78A8", label="consistent")
    ax.bar(x, summary["frac_early_warning_sites"], width=width, color="#F58518", label="early warning")
    ax.bar(x + width, summary["frac_legacy_effect_sites"], width=width, color="#B279A2", label="legacy")
    ax.set_xticks(x, traits, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of sites")
    ax.legend(frameon=False, ncol=1, loc="upper right")
    panel_label(ax, "b")
    style_axis(ax)

    ax = axes[1, 0]
    win = window_summary[
        window_summary["variable"].isin(["GREENNESS"] + traits)
        & window_summary["window"].isin(["apparent_recovery_2022", "same_season_2023", "late_summer_2023"])
    ].copy()
    pivot = win.pivot(index="variable", columns="window", values="median_recovery_pct").reindex(["GREENNESS"] + traits)
    labels = ["Sep-Oct 2022", "May-Jun 2023", "Jul-Aug 2023"]
    cols = ["apparent_recovery_2022", "same_season_2023", "late_summer_2023"]
    x = np.arange(len(pivot.index))
    for k, col in enumerate(cols):
        ax.plot(x, pivot[col], marker="o", lw=1.0, label=labels[k])
    ax.axhline(70, color="#777777", lw=0.6, ls=":")
    ax.set_xticks(x, pivot.index, rotation=35, ha="right")
    ax.set_ylim(-5, 105)
    ax.set_ylabel("Median recovery to baseline (%)")
    ax.legend(frameon=False, loc="lower right")
    panel_label(ax, "c")
    style_axis(ax)

    ax = axes[1, 1]
    y = summary["median_late_gap_trait_minus_green"]
    colors = [TRAIT_COLORS.get(t, "#555555") for t in traits]
    ax.bar(x=np.arange(len(traits)), height=y, color=colors)
    ax.axhline(0, color="black", lw=0.6)
    ax.axhline(-20, color="#777777", lw=0.6, ls=":")
    ax.set_xticks(np.arange(len(traits)), traits, rotation=35, ha="right")
    ax.set_ylabel("May-Jun 2023 gap vs greenness (pp)")
    panel_label(ax, "d")
    style_axis(ax)

    fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.12, wspace=0.34, hspace=0.46)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(figure_dir / f"trait_greenness_divergence_summary.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_site_scatter(site_metrics: pd.DataFrame, args: argparse.Namespace, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    for trait in TRAITS:
        sub = site_metrics[site_metrics["trait"] == trait]
        ax.scatter(
            sub["late_greenness_recovery"],
            sub["late_trait_recovery"],
            s=18,
            alpha=0.72,
            color=TRAIT_COLORS.get(trait, "#555555"),
            label=trait,
            edgecolor="white",
            linewidth=0.25,
        )
    ax.plot([0, 105], [0, 105], color="black", lw=0.7)
    ax.axvline(args.recovery_threshold, color="#777777", lw=0.6, ls=":")
    ax.axhline(args.recovery_threshold, color="#777777", lw=0.6, ls=":")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Greenness recovery, May-Jun 2023 (%)")
    ax.set_ylabel("Trait recovery, May-Jun 2023 (%)")
    ax.legend(frameon=False, ncol=2, loc="lower right", handletextpad=0.3, columnspacing=0.8)
    style_axis(ax)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(figure_dir / f"trait_greenness_site_scatter.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_summary_json(
    path: Path,
    args: argparse.Namespace,
    trait_summary: pd.DataFrame,
    recovery: pd.DataFrame,
) -> None:
    payload = {
        "input_recovery_table": str(args.recovery_table),
        "n_rows": int(len(recovery)),
        "n_sites": int(recovery["site_id"].nunique()),
        "traits": TRAITS,
        "greenness_reference": GREENNESS_INDICES,
        "spectral_indices": SPECTRAL_INDICES,
        "windows": {
            "baseline": [BASELINE_START.date().isoformat(), BASELINE_END.date().isoformat()],
            "drought_trough": [TROUGH_START.date().isoformat(), TROUGH_END.date().isoformat()],
            "apparent_recovery": [APPARENT_START.date().isoformat(), APPARENT_END.date().isoformat()],
            "same_season_recovery": [args.late_window_start, args.late_window_end],
            "late_summer_recovery": [LATE_SUMMER_START.date().isoformat(), LATE_SUMMER_END.date().isoformat()],
        },
        "thresholds": {
            "recovery_threshold": args.recovery_threshold,
            "stress_threshold": args.stress_threshold,
            "lead_days": args.lead_days,
            "delay_days": args.delay_days,
            "legacy_gap": args.legacy_gap,
            "consistent_corr": args.consistent_corr,
            "consistent_gap": args.consistent_gap,
        },
        "trait_evidence": trait_summary.to_dict(orient="records"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    recovery = load_recovery_table(args.recovery_table)
    greenness = build_reference_rows(recovery, GREENNESS_INDICES, "GREENNESS", "Greenness ref.")
    spectral = build_reference_rows(recovery, SPECTRAL_INDICES, "SPECTRAL_INDEX", "Spectral-index ref.")
    references = pd.concat([greenness, spectral], ignore_index=True, sort=False)

    site_metrics = compute_site_trait_metrics(recovery, greenness, args)
    trait_summary = compute_trait_summary(site_metrics, args)
    trait_index = compute_trait_index_similarity(recovery, args)
    window_summary = compute_window_summary(recovery, references, args)

    save_table(site_metrics, args.output_dir / "trait_greenness_site_classification.csv")
    save_table(trait_summary, args.output_dir / "trait_greenness_trait_summary.csv")
    save_table(trait_index, args.output_dir / "trait_index_similarity.csv")
    save_table(window_summary, args.output_dir / "trait_index_window_summary.csv")
    write_summary_json(args.output_dir / "trait_greenness_divergence_summary.json", args, trait_summary, recovery)

    plot_summary(trait_summary, trait_index, window_summary, args.figure_dir)
    plot_site_scatter(site_metrics, args, args.figure_dir)

    print("Trait-level evidence")
    cols = [
        "trait",
        "evidence_labels",
        "median_spearman_greenness",
        "frac_early_warning_sites",
        "frac_legacy_effect_sites",
        "median_late_trait_recovery",
        "median_late_greenness_recovery",
        "median_late_gap_trait_minus_green",
    ]
    print(trait_summary[cols].round(3).to_string(index=False))
    print(f"Wrote tables to {args.output_dir}")
    print(f"Wrote figures to {args.figure_dir}")


if __name__ == "__main__":
    main()
