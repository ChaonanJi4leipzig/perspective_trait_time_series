from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
FIG_DIR = ROOT / "figures" / "generated"
TABLE_DIR = ROOT / "tables"
RESULT_TABLE_DIR = PROJECT_ROOT / "models" / "traits_model" / "scripts" / "analysis" / "reporting" / "tables"
RESULT_FIG_TABLE_DIR = PROJECT_ROOT / "models" / "traits_model" / "scripts" / "analysis" / "tables" / "technical_report_all20_20260522"
OUTPUT_ROOT = PROJECT_ROOT / "output_data" / "traits_model" / "outputs" / "ablations"

FIG_W_DOUBLE = 183 / 25.4
FIG_H_MAX = 170 / 25.4

FS_PANEL = 8.0
FS_AXIS = 8.0
FS_TICK = 7.0
FS_TITLE = 7.0
FS_LEG = 7.0
FS_NOTE = 7.0

AXIS_LW = 0.5
REF_LW = 0.5
FIT_LW = 1.0

BLACK = "#000000"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
YELLOW = "#F0E442"
BLUE = "#0072B2"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#666666"
LIGHT_GREY = "#D9D9D9"
PALE_GREEN = "#DCECDF"
PALE_BLUE = "#DCEAF6"
PALE_GOLD = "#F2DEC3"
PALE_VIOLET = "#E8DCE8"
PALE_GREY = "#F2F2F2"

ALIAS_FULL = {
    "Anth": "Anthocyanin",
    "B": "Boron",
    "C": "Carbon",
    "Ca": "Calcium",
    "Cab": "Chlorophyll",
    "Car": "Carotenoid",
    "Cellulose": "Cellulose",
    "Cm": "LMA",
    "Cp": "Nitrogen",
    "Cu": "Copper",
    "Cw": "EWT",
    "Fiber": "Fiber",
    "K": "Potassium",
    "LAI": "LAI",
    "Lignin": "Lignin",
    "Mg": "Magnesium",
    "Mn": "Manganese",
    "NSC": "NSC",
    "P": "Phosphorus",
    "S": "Sulfur",
}

TRAIT_GROUP = {
    "LAI": "Structure",
    "EWT": "Water",
    "Chlorophyll": "Pigments",
    "Carotenoid": "Pigments",
    "Anthocyanin": "Pigments",
    "Nitrogen": "Leaf economics",
    "LMA": "Leaf economics",
    "Carbon": "Carbon / structure",
    "Lignin": "Carbon / structure",
    "Cellulose": "Carbon / structure",
    "Fiber": "Carbon / structure",
    "NSC": "Carbon / structure",
    "Boron": "Minerals",
    "Calcium": "Minerals",
    "Copper": "Minerals",
    "Magnesium": "Minerals",
    "Manganese": "Minerals",
    "Phosphorus": "Minerals",
    "Potassium": "Minerals",
    "Sulfur": "Minerals",
}

GROUP_COLORS = {
    "Structure": BLUE,
    "Water": SKY,
    "Pigments": GREEN,
    "Leaf economics": ORANGE,
    "Carbon / structure": PURPLE,
    "Minerals": GREY,
}


def setup():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
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


def style_axis(ax, keep_all_spines=False):
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


def panel_label(ax, letter, x=-0.12, y=1.04):
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


def figure_title(ax, text):
    ax.set_title(text, loc="left", fontsize=FS_TITLE, fontweight="normal", pad=5)


def save_publication_figure(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "eps"):
        path = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    fig.savefig(FIG_DIR / f"{stem}.jpg", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_box(ax, xy, wh, text, fc, ec=BLACK, lw=AXIS_LW, fontsize=FS_NOTE, radius=0.014):
    patch = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle=f"round,pad=0.010,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + wh[0] / 2,
        xy[1] + wh[1] / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=BLACK,
        linespacing=1.12,
    )
    return patch


def arrow(ax, start, end, color=BLACK, rad=0.0, lw=AXIS_LW, mutation_scale=8):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def trait_short_name(trait_name):
    for alias, label in ALIAS_FULL.items():
        if trait_name == label or trait_name.startswith(label):
            return label
    rules = [
        ("Chl content", "Chlorophyll"),
        ("Carotenoid", "Carotenoid"),
        ("Anthocyanin", "Anthocyanin"),
        ("N content", "Nitrogen"),
        ("LMA", "LMA"),
        ("EWT", "EWT"),
        ("C content", "Carbon"),
        ("Ca content", "Calcium"),
        ("Boron", "Boron"),
        ("Cu content", "Copper"),
        ("K content", "Potassium"),
        ("Mg content", "Magnesium"),
        ("Mn content", "Manganese"),
        ("P content", "Phosphorus"),
        ("S content", "Sulfur"),
    ]
    for prefix, label in rules:
        if trait_name.startswith(prefix):
            return label
    return re.sub(r"\s*\(.*?\)", "", trait_name)


def load_per_trait_metrics(path, model_label):
    df = pd.read_csv(path).copy()
    df["trait"] = df["trait_name"].map(trait_short_name)
    df["model_label"] = model_label
    df["group"] = df["trait"].map(TRAIT_GROUP).fillna("Other")
    return df


def load_fig3_data():
    paths = {
        "XGBoost": OUTPUT_ROOT
        / "technical_report_all20_main_cap5000_20260522"
        / "runs"
        / "20trait"
        / "xgb_multioutput"
        / "xgb"
        / "all20"
        / "test_per_trait_metrics.csv",
        "MLP": OUTPUT_ROOT
        / "trait_relation_cap5000_grouped_20260522"
        / "runs"
        / "20trait"
        / "all20_multitask"
        / "mlp"
        / "all20"
        / "test_per_trait_metrics.csv",
        "Transformer": OUTPUT_ROOT
        / "trait_relation_cap5000_grouped_20260522"
        / "runs"
        / "20trait"
        / "all20_multitask"
        / "transformer"
        / "all20"
        / "test_per_trait_metrics.csv",
    }
    frames = []
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(load_per_trait_metrics(path, label))
    per_trait = pd.concat(frames, ignore_index=True)

    synth = pd.read_csv(RESULT_TABLE_DIR / "current_experiment_synthesis_20260529.csv")
    formal = synth[synth["trust_level"].str.contains("formal", case=False, na=False)].copy()
    return per_trait, formal


def load_trait_dynamics_summary():
    path = TABLE_DIR / "trait_greenness_divergence_2022_2023aug" / "trait_greenness_trait_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    # Fallback for the Perspective figure: these are the same summary values
    # reported in the manuscript text and the archived trait-dynamics panel.
    rows = [
        {"trait": "LAI", "frac_consistent_sites": 0.297, "frac_early_warning_sites": 0.0, "frac_legacy_effect_sites": 0.0},
        {"trait": "Nitrogen", "frac_consistent_sites": 0.216, "frac_early_warning_sites": 0.0, "frac_legacy_effect_sites": 0.0},
        {"trait": "EWT", "frac_consistent_sites": 0.0, "frac_early_warning_sites": 0.405, "frac_legacy_effect_sites": 0.0},
        {"trait": "LMA", "frac_consistent_sites": 0.0, "frac_early_warning_sites": 0.324, "frac_legacy_effect_sites": 0.0},
        {"trait": "Chlorophyll", "frac_consistent_sites": 0.0, "frac_early_warning_sites": 0.0, "frac_legacy_effect_sites": 0.297},
    ]
    return pd.DataFrame(rows)


def draw_trait_space(ax, title=None, show_green_arrow=True):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nitrogen status")
    ax.set_ylabel("Leaf mass per area")
    ax.set_xticks([])
    ax.set_yticks([])
    for x, y, s, c in [
        (0.28, 0.70, 160, PALE_BLUE),
        (0.48, 0.48, 210, PALE_GREEN),
        (0.70, 0.32, 150, PALE_GOLD),
    ]:
        ax.scatter(x, y, s=s, color=c, edgecolor=BLACK, linewidth=AXIS_LW, zorder=1)
    path = np.array([[0.30, 0.56], [0.42, 0.42], [0.56, 0.38], [0.66, 0.48]])
    ax.plot(path[:, 0], path[:, 1], color=ORANGE, lw=FIT_LW, marker="o", ms=3.2, zorder=3)
    for i in range(len(path) - 1):
        arrow(ax, path[i], path[i + 1], color=ORANGE, lw=0.6, mutation_scale=7)
    if show_green_arrow:
        ax.annotate(
            "greenness\nreturns",
            xy=(0.66, 0.48),
            xytext=(0.72, 0.70),
            arrowprops=dict(arrowstyle="->", lw=REF_LW),
            fontsize=FS_NOTE,
            ha="center",
            va="center",
        )
    ax.text(0.29, 0.58, "pre-event", fontsize=FS_NOTE, ha="right")
    ax.text(0.66, 0.45, "post-event\ntrait state", fontsize=FS_NOTE, ha="left", va="top")
    if title:
        figure_title(ax, title)
    style_axis(ax, keep_all_spines=True)


def fig1_trait_snapshots_to_time_series():
    fig = plt.figure(figsize=(FIG_W_DOUBLE, 4.65))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.04, 1.0], width_ratios=[1.05, 0.72, 1.10], hspace=0.38, wspace=0.24)
    ax_g = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[0, 1])
    ax_trait = fig.add_subplot(gs[0, 2])
    ax_time = fig.add_subplot(gs[1, :])

    x = np.arange(0, 11)
    y = 0.75 - 0.42 * np.exp(-0.5 * ((x - 4.2) / 1.05) ** 2) + 0.03 * np.sin(x)
    ax_g.axvspan(3.2, 5.1, color=PALE_GOLD, lw=0)
    ax_g.plot(x, y, color=GREEN, lw=FIT_LW, marker="o", ms=3.2)
    ax_g.axhline(0.73, color=GREY, lw=REF_LW, ls="--")
    ax_g.annotate("classified as\nrecovered", xy=(8.6, y[9]), xytext=(6.4, 0.43), fontsize=FS_NOTE, arrowprops=dict(arrowstyle="->", lw=REF_LW))
    ax_g.set_xlabel("Time")
    ax_g.set_ylabel("Greenness index")
    ax_g.set_ylim(0.25, 0.86)
    ax_g.set_xticks([])
    figure_title(ax_g, "Greenness-space recovery")
    style_axis(ax_g)
    panel_label(ax_g, "a")

    ax_mid.set_xlim(0, 1)
    ax_mid.set_ylim(0, 1)
    ax_mid.axis("off")
    add_box(ax_mid, (0.10, 0.62), (0.80, 0.20), "same canopy\ncolour", PALE_GREEN, fontsize=FS_NOTE)
    add_box(ax_mid, (0.10, 0.22), (0.80, 0.20), "different\nfunctional state", PALE_VIOLET, fontsize=FS_NOTE)
    arrow(ax_mid, (0.50, 0.60), (0.50, 0.44), color=BLACK, mutation_scale=10)
    ax_mid.text(0.50, 0.51, "not equivalent to", ha="center", va="center", fontsize=FS_NOTE)

    draw_trait_space(ax_trait, "Trait-space recovery")
    panel_label(ax_trait, "b")

    ax_time.set_xlim(0, 1)
    ax_time.set_ylim(0, 1)
    ax_time.axis("off")
    ax_time.text(0.03, 0.88, "Trait observables require anchors, carriers and guardrails", fontsize=FS_AXIS, ha="left")
    stages = [
        (0.08, "Landsat-era\nspectral memory", PALE_GREEN),
        (0.30, "Sentinel-2\ntemporal carrier", PALE_GREEN),
        (0.53, "PRISMA / EnMAP /\nEMIT trait anchors", PALE_BLUE),
        (0.75, "LiDAR / SAR /\nclimate context", "#E8E4D6"),
        (0.91, "uncertainty +\nprovenance", PALE_VIOLET),
    ]
    for x0, label, color in stages:
        add_box(ax_time, (x0 - 0.075, 0.42), (0.15, 0.22), label, color, fontsize=6.2)
    for (x0, *_), (x1, *__) in zip(stages[:-1], stages[1:]):
        arrow(ax_time, (x0 + 0.075, 0.53), (x1 - 0.075, 0.53), lw=0.55)
    ax_time.plot([0.06, 0.93], [0.27, 0.27], color=BLACK, lw=FIT_LW)
    for x0, lab in [(0.08, "1980s+"), (0.30, "2015+"), (0.53, "2022+"), (0.83, "2028-2030")]:
        ax_time.scatter(x0, 0.27, s=28, color="white", edgecolor=BLACK, linewidth=AXIS_LW, zorder=3)
        ax_time.text(x0, 0.17, lab, ha="center", va="top", fontsize=FS_NOTE)
    panel_label(ax_time, "c", x=0.00, y=0.98)
    save_publication_figure(fig, "fig1_from_trait_snapshots_to_functional_time_series")


def fig2_framework():
    fig = plt.figure(figsize=(FIG_W_DOUBLE, 4.20))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 0.88], width_ratios=[1.0, 1.0, 1.0], hspace=0.35, wspace=0.26)
    ax_flow = fig.add_subplot(gs[0, :2])
    ax_class = fig.add_subplot(gs[0, 2])
    ax_contract = fig.add_subplot(gs[1, :])

    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    ax_flow.axis("off")
    pillars = [
        ((0.05, 0.60), (0.22, 0.18), "Hyperspectral\ntrait anchors\nEnMAP, PRISMA, EMIT", PALE_BLUE),
        ((0.05, 0.34), (0.22, 0.18), "Dense optical\ntime carrier\nSentinel-2, Landsat", PALE_GREEN),
        ((0.05, 0.08), (0.22, 0.18), "Structure and\nforcing context\nLiDAR, SAR, ERA5", "#E8E4D6"),
        ((0.43, 0.32), (0.21, 0.22), "Trait-state\nreconstruction", PALE_GOLD),
        ((0.75, 0.40), (0.20, 0.16), "Ecological\nclaim", PALE_VIOLET),
    ]
    for xy, wh, label, color in pillars:
        add_box(ax_flow, xy, wh, label, color, fontsize=6.4)
    for y in (0.69, 0.43, 0.17):
        arrow(ax_flow, (0.27, y), (0.43, 0.43), lw=0.55)
    arrow(ax_flow, (0.64, 0.43), (0.75, 0.48), lw=0.55)
    ax_flow.text(0.43, 0.80, "contract defines\nclaim strength", fontsize=FS_NOTE, ha="left", linespacing=1.1)
    figure_title(ax_flow, "Three-pillar observation system")
    panel_label(ax_flow, "a", x=0.00, y=0.95)

    ax_class.set_xlim(0, 1)
    ax_class.set_ylim(0, 1)
    ax_class.axis("off")
    classes = [
        ("Direct", "pigments,\nEWT", PALE_BLUE),
        ("Indirect", "N, LMA,\ncarbon traits", PALE_GREEN),
        ("Exploratory", "ITV,\nfunctional diversity", PALE_VIOLET),
    ]
    y0 = 0.72
    for i, (title, text, color) in enumerate(classes):
        add_box(ax_class, (0.12, y0 - i * 0.27), (0.76, 0.17), f"{title}\n{text}", color, fontsize=6.2)
    figure_title(ax_class, "Claim classes")
    panel_label(ax_class, "b", x=0.00, y=0.95)

    ax_contract.set_xlim(0, 1)
    ax_contract.set_ylim(0, 1)
    ax_contract.axis("off")
    items = [
        ("trait definition", 0.09, PALE_BLUE),
        ("training domain", 0.25, PALE_GREEN),
        ("validation design", 0.41, PALE_GOLD),
        ("uncertainty", 0.57, PALE_VIOLET),
        ("provenance", 0.73, "#E8E4D6"),
        ("domain flags", 0.89, PALE_GREY),
    ]
    for label, x0, color in items:
        add_box(ax_contract, (x0 - 0.065, 0.47), (0.13, 0.22), label, color, fontsize=6.1)
    for (_, x0, _), (_, x1, _) in zip(items[:-1], items[1:]):
        arrow(ax_contract, (x0 + 0.065, 0.58), (x1 - 0.065, 0.58), lw=0.5)
    ax_contract.text(0.03, 0.25, "Output: trait time series that can be questioned like field observations: by trait, site, year, disturbance context and uncertainty.", fontsize=FS_NOTE, ha="left")
    panel_label(ax_contract, "c", x=0.00, y=0.91)
    save_publication_figure(fig, "fig2_functional_observables_framework")


def fig3_performance():
    per_trait, synth = load_fig3_data()
    dynamics = load_trait_dynamics_summary()
    tdf = per_trait[per_trait["model_label"] == "Transformer"].copy()
    order = tdf.sort_values("r2", ascending=True)["trait"].tolist()

    fig = plt.figure(figsize=(FIG_W_DOUBLE, 6.25))
    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.03, 1.05],
        width_ratios=[0.70, 1.50, 0.98, 0.92],
        hspace=0.52,
        wspace=0.78,
    )
    ax_sample = fig.add_subplot(gs[0, 0])
    ax_trait = fig.add_subplot(gs[:, 1])
    ax_regime = fig.add_subplot(gs[0, 2])
    ax_delta = fig.add_subplot(gs[1, 0])
    ax_group = fig.add_subplot(gs[1, 2])
    ax_dyn = fig.add_subplot(gs[:, 3])

    sample_labels = ["sites", "site-years", "rows (x10k)", "traits"]
    sample_vals = [37, 99, 49.0144, 20]
    yy = np.arange(len(sample_labels))
    ax_sample.barh(yy, sample_vals, color=[PALE_BLUE, PALE_GREEN, PALE_GOLD, PALE_VIOLET], edgecolor=BLACK, linewidth=AXIS_LW)
    for i, v in enumerate(sample_vals):
        ax_sample.text(v + 3, i, f"{v:.0f}" if i != 2 else "49.0", va="center", fontsize=FS_NOTE)
    ax_sample.set_yticks(yy, sample_labels)
    ax_sample.set_xlabel("Count")
    ax_sample.set_xlim(0, 114)
    ax_sample.invert_yaxis()
    figure_title(ax_sample, "European forest testbed")
    style_axis(ax_sample)
    panel_label(ax_sample, "a", x=-0.28)

    model_order = ["XGBoost", "MLP", "Transformer"]
    offsets = [-0.22, 0.0, 0.22]
    colors = {"XGBoost": GREY, "MLP": SKY, "Transformer": BLUE}
    y = np.arange(len(order))
    for model, off in zip(model_order, offsets):
        sub = per_trait[per_trait["model_label"] == model].set_index("trait").loc[order]
        ax_trait.scatter(sub["r2"], y + off, s=18, color=colors[model], edgecolor=BLACK, linewidth=0.25, label=model, zorder=3)
    for yi, trait in enumerate(order):
        ax_trait.axhline(yi, color="#E8E8E8", lw=0.35, zorder=0)
    ax_trait.set_yticks(y, order)
    ax_trait.set_xlim(0.0, 0.82)
    ax_trait.set_xlabel("$R^2$ on held-out samples")
    figure_title(ax_trait, "Trait-specific skill is uneven")
    ax_trait.legend(frameon=False, loc="lower right", handlelength=1.0, labelspacing=0.25)
    style_axis(ax_trait)
    panel_label(ax_trait, "b", x=-0.22)

    regime_rows = [
        ("In-domain", "In-domain all20", "Transformer all20"),
        ("Temporal\n12-way", "Temporal 12-way transfer", "Transformer all20"),
        ("Balanced16\n1y temporal", "Balanced16 temporal 1y train", "Transformer all20"),
        ("Balanced16\n2y temporal", "Balanced16 temporal 2y train", "Transformer all20"),
        ("Spatial k8", "Spatial k8 transfer", "Transformer all20"),
        ("Balanced16\nspatial", "Balanced16 south-to-north spatial", "Transformer all20"),
        ("Trait-PCA\nspatial", "Balanced16 trait-PCA spatial", "transformer"),
    ]
    vals = []
    errs = []
    for label, fam, model in regime_rows:
        sub = synth[(synth["experiment_family"] == fam) & (synth["model"] == model)]
        vals.append(float(sub["mean_r2"].iloc[0]))
        errs.append(float(sub["mean_nrmse_pct"].iloc[0]))
    bar_colors = [BLUE, ORANGE, VERMILION, VERMILION, SKY, GREEN, GREEN]
    yy = np.arange(len(regime_rows))
    ax_regime.barh(yy, vals, color=bar_colors, edgecolor=BLACK, linewidth=AXIS_LW)
    ax_regime.axvline(vals[0], color=GREY, lw=REF_LW, ls="--")
    ax_regime.set_yticks(yy, [r[0] for r in regime_rows])
    ax_regime.set_xlabel("Mean $R^2$")
    ax_regime.set_xlim(0, 0.64)
    ax_regime.invert_yaxis()
    figure_title(ax_regime, "Transfer bottleneck")
    style_axis(ax_regime)
    panel_label(ax_regime, "c", x=-0.22)

    base = synth[(synth["experiment_family"] == "In-domain all20") & (synth["model"] == "Transformer all20")]["mean_r2"].iloc[0]
    era5 = synth[(synth["experiment_family"] == "ERA5 in-domain") & (synth["model"] == "Transformer all20 + ERA5")]["mean_r2"].iloc[0]
    b16 = synth[(synth["experiment_family"] == "Balanced16 temporal 2y train") & (synth["model"] == "Transformer all20")]["mean_r2"].iloc[0]
    b16e = synth[(synth["experiment_family"] == "ERA5 temporal") & (synth["model"] == "Transformer all20 + ERA5")].iloc[-1]["mean_r2"]
    labels = ["in-domain", "temporal\n2y train"]
    deltas = [era5 - base, b16e - b16]
    yy = np.arange(len(labels))
    ax_delta.axvline(0, color=BLACK, lw=REF_LW)
    ax_delta.barh(yy, deltas, color=[GREEN if d >= 0 else VERMILION for d in deltas], edgecolor=BLACK, linewidth=AXIS_LW)
    for i, d in enumerate(deltas):
        ax_delta.text(d + (0.002 if d >= 0 else -0.002), i, f"{d:+.3f}", va="center", ha="left" if d >= 0 else "right", fontsize=FS_NOTE)
    ax_delta.set_yticks(yy, labels)
    ax_delta.set_xlabel(r"$\Delta R^2$ with ERA5")
    ax_delta.set_xlim(-0.04, 0.04)
    ax_delta.set_xticks([-0.02, 0.00, 0.02])
    ax_delta.invert_yaxis()
    figure_title(ax_delta, "ERA5 delta")
    style_axis(ax_delta)
    panel_label(ax_delta, "d", x=-0.28)

    group = tdf.groupby("group", as_index=False).agg(mean_r2=("r2", "mean"), n=("trait", "count"))
    group = group.sort_values("mean_r2")
    ax_group.barh(group["group"], group["mean_r2"], color=[GROUP_COLORS.get(g, GREY) for g in group["group"]], edgecolor=BLACK, linewidth=AXIS_LW)
    for yi, row in group.reset_index(drop=True).iterrows():
        ax_group.text(row["mean_r2"] + 0.012, yi, f"{row['mean_r2']:.2f}  n={int(row['n'])}", va="center", fontsize=FS_NOTE)
    ax_group.set_xlim(0, 0.72)
    ax_group.set_xlabel("Mean Transformer $R^2$")
    figure_title(ax_group, "Readiness by class")
    style_axis(ax_group)
    panel_label(ax_group, "e", x=-0.22)

    dyn_by_trait = dynamics.set_index("trait")
    dyn_specs = [
        ("LAI", "recovery check", "frac_consistent_sites", GREEN),
        ("Nitrogen", "recovery check", "frac_consistent_sites", GREEN),
        ("EWT", "leading indicator", "frac_early_warning_sites", ORANGE),
        ("LMA", "leading indicator", "frac_early_warning_sites", ORANGE),
        ("Chlorophyll", "functional legacy", "frac_legacy_effect_sites", PURPLE),
    ]
    y = np.arange(len(dyn_specs))
    vals = np.array([float(dyn_by_trait.loc[trait, col]) * 100.0 for trait, _kind, col, _color in dyn_specs])
    labels = ["LAI\ncheck", "N\ncheck", "EWT\nlead", "LMA\nlead", "Chl\nlegacy"]
    colors = [color for _trait, _kind, _col, color in dyn_specs]
    ax_dyn.barh(y, vals, color=colors, edgecolor=BLACK, linewidth=AXIS_LW)
    for yi, val in enumerate(vals):
        ax_dyn.text(val + 1.1, yi, f"{val:.1f}%", va="center", fontsize=FS_NOTE)
    ax_dyn.set_yticks(y, labels)
    ax_dyn.set_xlim(0, 47)
    ax_dyn.invert_yaxis()
    ax_dyn.set_xlabel("Sites showing\npattern (%)")
    figure_title(ax_dyn, "Trait dynamics")
    style_axis(ax_dyn)
    panel_label(ax_dyn, "f", x=-0.28)

    save_publication_figure(fig, "fig3_proof_of_concept_performance")


def draw_trait_dynamics_panel(ax, letter="f"):
    dynamics = load_trait_dynamics_summary().set_index("trait")
    dyn_specs = [
        ("LAI", "LAI\ncheck", "frac_consistent_sites", GREEN),
        ("Nitrogen", "N\ncheck", "frac_consistent_sites", GREEN),
        ("EWT", "EWT\nlead", "frac_early_warning_sites", ORANGE),
        ("LMA", "LMA\nlead", "frac_early_warning_sites", ORANGE),
        ("Chlorophyll", "Chl\nlegacy", "frac_legacy_effect_sites", PURPLE),
    ]
    y = np.arange(len(dyn_specs))
    vals = np.array([float(dynamics.loc[trait, col]) * 100.0 for trait, _label, col, _color in dyn_specs])
    labels = [label for _trait, label, _col, _color in dyn_specs]
    colors = [color for _trait, _label, _col, color in dyn_specs]
    ax.barh(y, vals, color=colors, edgecolor=BLACK, linewidth=AXIS_LW)
    for yi, val in enumerate(vals):
        ax.text(val + 1.0, yi, f"{val:.1f}%", va="center", fontsize=FS_NOTE)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 47)
    ax.invert_yaxis()
    ax.set_xlabel("Sites with pattern (%)", labelpad=1)
    figure_title(ax, "Trait--greenness dynamics")
    style_axis(ax)
    panel_label(ax, letter, x=-0.20)


def load_fig4_tables():
    base = TABLE_DIR / "fig4_drought_anchor_xgb"
    context = pd.read_csv(base / "site43_44_xgb_biweekly_context_2022_2023aug.csv", parse_dates=["date"])
    traits = pd.read_csv(base / "site43_44_xgb_biweekly_trait_predictions_2022_2023aug_summary.csv", parse_dates=["date"])
    recovery = pd.read_csv(base / "site43_44_recovery_timeline_2022_2023aug.csv", parse_dates=["date"])
    return context, traits, recovery


def shade_drought(ax):
    ax.axvspan(pd.Timestamp("2022-07-23"), pd.Timestamp("2022-08-20"), color=PALE_GOLD, lw=0, zorder=0)


def format_date_axis(ax, label_bottom=True):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(labelbottom=label_bottom)
    style_axis(ax)


def fig4_anomalies():
    context, traits, recovery = load_fig4_tables()
    site_names = {43: "Le Bray", 44: "Bilos"}
    fig = plt.figure(figsize=(FIG_W_DOUBLE, 6.65))
    gs = fig.add_gridspec(
        4,
        2,
        height_ratios=[0.82, 1.10, 1.10, 0.88],
        width_ratios=[1.22, 0.94],
        hspace=0.56,
        wspace=0.36,
    )
    ax_era = fig.add_subplot(gs[0, :])
    ax_s1 = fig.add_subplot(gs[1, 0], sharex=ax_era)
    ax_s2 = fig.add_subplot(gs[2, 0], sharex=ax_era)
    ax_idx = fig.add_subplot(gs[3, :], sharex=ax_era)
    ax_space = fig.add_subplot(gs[1:3, 1])

    for sid, color in [(43, BLUE), (44, ORANGE)]:
        sub = context[context["site_id"] == sid].sort_values("date")
        shade_drought(ax_era)
        ax_era.plot(sub["date"], sub["volumetric_soil_water_layer_1"], color=color, lw=FIT_LW, label=site_names[sid])
    ax_era.set_ylabel("Soil water\n(m3 m-3)")
    ax_era.legend(frameon=False, loc="upper right", ncol=2, handlelength=1.4)
    figure_title(ax_era, "Drought driver")
    format_date_axis(ax_era, label_bottom=False)
    panel_label(ax_era, "a", x=-0.04)

    trait_cols = [("LAI", BLUE), ("LMA", ORANGE), ("Nitrogen", VERMILION), ("EWT", SKY), ("Chlorophyll", GREEN)]
    for ax, sid, letter, title in [(ax_s1, 43, "b", "Le Bray: inferred traits"), (ax_s2, 44, "c", "Bilos: inferred traits")]:
        shade_drought(ax)
        sub = traits[traits["site_id"] == sid].sort_values("date")
        ax2 = ax.twinx()
        for tr, color in trait_cols:
            tsub = sub[sub["trait"] == tr]
            target = ax if tr in ("LAI", "EWT", "Chlorophyll") else ax2
            target.plot(tsub["date"], tsub["prediction_median"], color=color, lw=FIT_LW, label=tr)
        ax.set_ylabel("LAI / EWT /\nChlorophyll")
        ax2.set_ylabel("")
        ax2.tick_params(axis="y", labelsize=FS_TICK, width=AXIS_LW, length=3)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_linewidth(AXIS_LW)
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], frameon=False, loc="upper right", ncol=3, handlelength=1.2, columnspacing=0.8, fontsize=6.0)
        figure_title(ax, title)
        format_date_axis(ax, label_bottom=False)
        panel_label(ax, letter, x=-0.08)

    for sid, color in [(43, BLUE), (44, ORANGE)]:
        sub = context[context["site_id"] == sid].sort_values("date")
        shade_drought(ax_idx)
        for idx, ls in [("NDVI_median", "-"), ("NDWI_median", "--"), ("MSI_median", ":")]:
            ax_idx.plot(sub["date"], sub[idx], color=color, ls=ls, lw=FIT_LW)
    ax_idx.set_ylabel("Spectral index")
    ax_idx.set_xlabel("Date")
    custom = [
        Line2D([0], [0], color=GREY, lw=FIT_LW, ls="-", label="NDVI"),
        Line2D([0], [0], color=GREY, lw=FIT_LW, ls="--", label="NDWI"),
        Line2D([0], [0], color=GREY, lw=FIT_LW, ls=":", label="MSI"),
        Line2D([0], [0], color=BLUE, lw=FIT_LW, label="Le Bray"),
        Line2D([0], [0], color=ORANGE, lw=FIT_LW, label="Bilos"),
    ]
    ax_idx.legend(handles=custom, frameon=False, loc="upper right", ncol=5, handlelength=1.4, columnspacing=0.9)
    figure_title(ax_idx, "Greenness, water and stress indices")
    format_date_axis(ax_idx, label_bottom=True)
    panel_label(ax_idx, "d", x=-0.04)

    stage_order = ["pre-event", "drought trough", "apparent recovery", "2023 recovery"]
    for sid, color, marker in [(43, BLUE, "o"), (44, ORANGE, "s")]:
        sub = recovery[
            (recovery["site_id"] == sid)
            & (recovery["variable"].isin(["NDVI", "Chlorophyll"]))
            & (recovery["stage"].isin(stage_order))
        ].copy()
        piv = (
            sub.pivot_table(index="stage", columns="variable", values="recovery_pct_clipped", aggfunc="median")
            .reindex(stage_order)
            .dropna()
        )
        ax_space.plot(piv["NDVI"], piv["Chlorophyll"], color=color, lw=FIT_LW, marker=marker, ms=3.6, label=site_names[sid])
        for i in range(len(piv) - 1):
            arrow(
                ax_space,
                (piv["NDVI"].iloc[i], piv["Chlorophyll"].iloc[i]),
                (piv["NDVI"].iloc[i + 1], piv["Chlorophyll"].iloc[i + 1]),
                color=color,
                lw=0.45,
                mutation_scale=6,
            )
    ax_space.text(2.0, 4.0, "trough", fontsize=5.8, color=BLACK, ha="left", va="bottom")
    ax_space.text(86.0, 62.0, "pre-event", fontsize=5.8, color=BLACK, ha="left", va="bottom")
    ax_space.text(76.0, 6.0, "2023 recovery", fontsize=5.8, color=BLACK, ha="left", va="bottom")
    ax_space.axhline(100, color=GREY, lw=REF_LW, ls="--")
    ax_space.axvline(100, color=GREY, lw=REF_LW, ls="--")
    ax_space.axhline(70, color=LIGHT_GREY, lw=REF_LW, ls=":")
    ax_space.axvline(70, color=LIGHT_GREY, lw=REF_LW, ls=":")
    ax_space.set_xlim(-5, 105)
    ax_space.set_ylim(-5, 105)
    ax_space.set_xlabel("NDVI recovery (%)", labelpad=0)
    ax_space.set_ylabel("Chlorophyll recovery (%)")
    figure_title(ax_space, "Greenness--pigment recovery")
    style_axis(ax_space)
    panel_label(ax_space, "e", x=-0.12)

    save_publication_figure(fig, "fig4_trait_anomaly_syndromes")


def fig5_roadmap():
    fig = plt.figure(figsize=(FIG_W_DOUBLE, 4.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.30)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    scenarios = [
        ("Functional recovery", GREEN, np.array([[0.18, 0.38], [0.52, 0.72], [0.24, 0.42], [0.18, 0.38]])),
        ("Functional legacy", ORANGE, np.array([[0.18, 0.38], [0.54, 0.72], [0.62, 0.58], [0.67, 0.55]])),
        ("Reorganization", VERMILION, np.array([[0.18, 0.38], [0.56, 0.72], [0.78, 0.74], [0.82, 0.78]])),
    ]
    for ax, (title, color, pts), letter in zip(axes, scenarios, ["a", "b", "c"]):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Nitrogen status")
        ax.set_ylabel("Leaf mass per area")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.scatter([0.18], [0.38], s=55, color="white", edgecolor=BLACK, linewidth=AXIS_LW, zorder=4)
        ax.text(0.18, 0.30, "pre-event", ha="center", fontsize=FS_NOTE)
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=FIT_LW, marker="o", ms=3.2, zorder=3)
        for i in range(len(pts) - 1):
            arrow(ax, pts[i], pts[i + 1], color=color, lw=0.55, mutation_scale=7)
        ax.add_patch(Rectangle((0.12, 0.32), 0.14, 0.12, facecolor=PALE_GREY, edgecolor=GREY, linewidth=0.4, zorder=0))
        if "legacy" in title.lower():
            ax.annotate("trait offset\nafter greenness\nrecovery", xy=pts[-1], xytext=(0.42, 0.22), arrowprops=dict(arrowstyle="->", lw=REF_LW), fontsize=FS_NOTE, ha="center")
        if "Reorganization" in title:
            ax.text(0.65, 0.25, "new region\nof trait space", fontsize=FS_NOTE, ha="center")
        if "recovery" in title.lower():
            ax.text(0.53, 0.22, "return within\nuncertainty", fontsize=FS_NOTE, ha="center")
        figure_title(ax, title)
        style_axis(ax, keep_all_spines=True)
        panel_label(ax, letter, x=-0.16)
    fig.text(
        0.50,
        0.02,
        "The 2022-2030 hyperspectral window makes these scenarios empirically distinguishable when trait anchors, temporal carriers and uncertainty are reported together.",
        ha="center",
        va="bottom",
        fontsize=FS_NOTE,
    )
    save_publication_figure(fig, "fig5_monitoring_roadmap")


def main():
    setup()
    fig1_trait_snapshots_to_time_series()
    fig2_framework()
    fig4_anomalies()
    fig5_roadmap()


if __name__ == "__main__":
    main()
