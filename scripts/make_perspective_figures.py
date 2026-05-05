from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "generated"
TABLE_DIR = ROOT / "tables"

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


def panel_label(ax, letter, x=0.0, y=1.02):
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


def save_publication_figure(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "eps"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.jpg", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_box(ax, xy, wh, text, fc, ec=BLACK, lw=AXIS_LW, fontsize=FS_NOTE):
    patch = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.012,rounding_size=0.018",
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
        linespacing=1.15,
    )
    return patch


def arrow(ax, start, end, color=BLACK, rad=0.0, lw=AXIS_LW):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def figure_title(ax, text):
    ax.set_title(text, loc="left", fontsize=FS_TITLE, fontweight="normal", pad=5)


def fig1_trait_snapshots_to_time_series():
    fig = plt.figure(figsize=(FIG_W_DOUBLE, 4.45))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.85, 1.12], hspace=0.32, wspace=0.28)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_bottom = fig.add_subplot(gs[1, :])

    panels = [
        (
            "Spectral-index archives",
            "Long temporal memory\nLandsat + Sentinel-2\ncontinuity since past decades",
            "Detect change, phenology\nand disturbance",
            "#E6EFD5",
        ),
        (
            "Imaging-spectroscopy traits",
            "Functional specificity\nPRISMA, EnMAP, EMIT\nand future CHIME / SBG",
            "Retrieve biochemical\nand optical trait snapshots",
            "#DCEAF6",
        ),
        (
            "Functional time series",
            "Trait anchors + time carriers\n+ structural complements\n+ uncertainty contract",
            "Monitor stress, recovery\nlegacy and model constraints",
            "#DCECDF",
        ),
    ]
    for i, (title, main, sub, color) in enumerate(panels):
        ax = axes[i]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        panel_label(ax, chr(ord("a") + i), x=0.00, y=1.02)
        ax.text(0.08, 1.02, title, ha="left", va="center", fontsize=FS_TITLE, color=BLACK)
        add_box(ax, (0.08, 0.44), (0.84, 0.34), main, color, fontsize=6.2)
        ax.text(0.50, 0.25, sub, ha="center", va="center", fontsize=FS_NOTE, color=BLACK, linespacing=1.12)
        if i < 2:
            arrow(ax, (0.94, 0.61), (1.12, 0.61), color=BLACK)

    ax_bottom.set_xlim(0, 1)
    ax_bottom.set_ylim(0, 1)
    ax_bottom.axis("off")
    ax_bottom.text(
        0.04,
        0.93,
        "From trait snapshots to functional time series",
        ha="left",
        va="center",
        fontsize=FS_TITLE,
        color=BLACK,
    )

    y = 0.46
    ax_bottom.plot([0.08, 0.92], [y, y], color=BLACK, lw=FIT_LW)
    milestones = [
        (0.10, "1980s+\nLandsat\nretrospective\ncontext", "#E6EFD5"),
        (0.34, "2015+\nSentinel-2\ndense temporal\ncarrier", "#E6EFD5"),
        (0.58, "Now\nspaceborne\ntrait anchors", "#DCEAF6"),
        (0.82, "Next\nCHIME / SBG\nmore systematic\ntrait snapshots", "#DCEAF6"),
    ]
    for x, label, color in milestones:
        ax_bottom.scatter(x, y, s=42, color=color, edgecolor=BLACK, linewidth=AXIS_LW, zorder=3)
        ax_bottom.text(x, y + 0.11, label, ha="center", va="bottom", fontsize=5.8, linespacing=1.05)
    add_box(
        ax_bottom,
        (0.19, 0.08),
        (0.62, 0.18),
        "Densify sparse hyperspectral trait retrievals\nand reconstruct trait-relevant functional states through existing archives",
        "#F2DEC3",
        fontsize=6.4,
    )
    arrow(ax_bottom, (0.50, 0.36), (0.50, 0.26), color=BLACK)
    save_publication_figure(fig, "fig1_from_trait_snapshots_to_functional_time_series")


def fig2_framework():
    fig = plt.figure(figsize=(FIG_W_DOUBLE, 3.65))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.00, 1.35], wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    systems = [
        "Field traits",
        "Imaging\nspectroscopy",
        "LiDAR / SAR /\nInSAR",
        "Sentinel-2 /\nLandsat",
        "Functional\nobservables",
    ]
    dims = ["Meaning", "Spectral", "3D", "Temporal", "Spatial", "Uncert."]
    values = np.array(
        [
            [5, 2, 2, 1, 1, 4],
            [4, 5, 2, 2, 3, 3],
            [3, 1, 5, 3, 3, 4],
            [2, 2, 2, 5, 5, 3],
            [5, 4, 5, 4, 4, 5],
        ]
    )
    cmap = LinearSegmentedColormap.from_list("functional", ["#F6F6F6", "#CFE5D8", "#7CC4B0", BLUE])
    im = ax1.imshow(values, vmin=1, vmax=5, cmap=cmap, aspect="auto")
    ax1.set_xticks(np.arange(len(dims)), dims, rotation=25, ha="right")
    ax1.set_yticks(np.arange(len(systems)), systems)
    ax1.tick_params(length=0)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax1.text(j, i, str(values[i, j]), ha="center", va="center", fontsize=FS_TICK, color=BLACK)
    figure_title(ax1, "Complementary observations")
    style_axis(ax1, keep_all_spines=True)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.040, pad=0.03)
    cbar.set_ticks([1, 3, 5])
    cbar.set_ticklabels(["Low", "Medium", "High"])
    cbar.ax.tick_params(labelsize=FS_TICK, width=AXIS_LW, length=3)
    cbar.outline.set_linewidth(AXIS_LW)
    panel_label(ax1, "a", x=-0.38, y=1.10)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    figure_title(ax2, "Sparse-to-dense monitoring")
    boxes = {
        "spectra": ((0.03, 0.68), (0.23, 0.14), "Imaging\nspectroscopy\ntrait anchors", "#DCEAF6"),
        "structure": ((0.03, 0.44), (0.23, 0.13), "LiDAR/SAR/InSAR\n3D covariates", "#E8E4D6"),
        "time": ((0.03, 0.22), (0.23, 0.14), "Multispectral\ntime series", "#E6EFD5"),
        "model": ((0.38, 0.47), (0.20, 0.18), "Temporal\nreconstruction", "#F2DEC3"),
        "unc": ((0.70, 0.50), (0.24, 0.15), "Trait estimates\nuncertainty\nprovenance", "#DCECDF"),
        "eco": ((0.70, 0.20), (0.24, 0.13), "Stress, recovery,\nlegacy metrics", "#E8DCE8"),
    }
    for xy, wh, label, color in boxes.values():
        add_box(ax2, xy, wh, label, color, fontsize=5.8)
    arrow(ax2, (0.26, 0.75), (0.38, 0.57))
    arrow(ax2, (0.26, 0.50), (0.38, 0.56))
    arrow(ax2, (0.26, 0.29), (0.38, 0.55))
    arrow(ax2, (0.58, 0.56), (0.70, 0.57))
    arrow(ax2, (0.82, 0.50), (0.82, 0.33))
    ax2.text(0.67, 0.87, "Measurement contract", ha="center", fontsize=FS_NOTE, fontweight="bold", color=BLACK)
    ax2.plot([0.42, 0.94], [0.83, 0.83], color=BLACK, lw=AXIS_LW)
    ax2.text(0.68, 0.79, "definition | domain | validation | flags", ha="center", color=BLACK, fontsize=6.0)
    panel_label(ax2, "b", x=-0.30, y=1.10)
    save_publication_figure(fig, "fig2_functional_observables_framework")


def fig3_performance():
    df = pd.read_csv(TABLE_DIR / "proof_of_concept_performance.csv")
    trait_order = ["LAI", "LMA", "Nitrogen", "Lignin", "Chlorophyll"]
    model_order = ["LinearRegression", "RF", "MLP", "Transformer"]
    model_labels = ["Linear", "RF", "MLP", "Transformer"]

    r2 = df.pivot(index="trait", columns="model", values="r2").loc[trait_order, model_order]
    nrmse = df.pivot(index="trait", columns="model", values="nrmse_pct").loc[trait_order, model_order]
    rf = df[df["model"] == "RF"].set_index("trait").loc[trait_order]
    lin = df[df["model"] == "LinearRegression"].set_index("trait").loc[trait_order]
    delta = rf["r2"] - lin["r2"]

    fig = plt.figure(figsize=(FIG_W_DOUBLE, 4.15))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.42, wspace=0.34)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    cmap = LinearSegmentedColormap.from_list("r2", ["#F6F6F6", "#CFE5D8", "#7CC4B0", BLUE])
    im = ax1.imshow(r2.values, vmin=0, vmax=0.7, cmap=cmap, aspect="auto")
    ax1.set_xticks(np.arange(len(model_labels)), model_labels)
    ax1.set_yticks(np.arange(len(trait_order)), trait_order)
    figure_title(ax1, "Trait- and model-specific skill")
    ax1.tick_params(length=0)
    for i in range(r2.shape[0]):
        for j in range(r2.shape[1]):
            ax1.text(j, i, f"{r2.iloc[i, j]:.2f}", ha="center", va="center", fontsize=FS_TICK, color=BLACK)
    style_axis(ax1, keep_all_spines=True)
    cb = fig.colorbar(im, ax=ax1, fraction=0.025, pad=0.02)
    cb.set_label("$R^2$", fontsize=FS_AXIS)
    cb.ax.tick_params(labelsize=FS_TICK, width=AXIS_LW, length=3)
    cb.outline.set_linewidth(AXIS_LW)
    panel_label(ax1, "a", x=-0.15, y=1.06)

    colors = [BLUE, SKY, VERMILION, PURPLE, ORANGE]
    ax2.barh(trait_order, delta.loc[trait_order], color=colors, edgecolor=BLACK, linewidth=AXIS_LW)
    ax2.axvline(0, color=BLACK, lw=REF_LW)
    ax2.set_xlabel(r"$\Delta R^2$ (RF - linear)")
    figure_title(ax2, "Gain over linear baseline")
    ax2.invert_yaxis()
    for y, val in enumerate(delta.loc[trait_order]):
        ax2.text(val + 0.006, y, f"{val:.2f}", va="center", fontsize=FS_NOTE, color=BLACK)
    style_axis(ax2)
    panel_label(ax2, "b", x=-0.15, y=1.06)

    x = np.arange(len(trait_order))
    ax3.plot(x, rf["nrmse_pct"], marker="o", ms=3.6, color=BLUE, lw=FIT_LW, label="RF")
    ax3.plot(x, nrmse["LinearRegression"], marker="s", ms=3.6, color=ORANGE, lw=FIT_LW, label="Linear")
    ax3.set_xticks(x, trait_order, rotation=35, ha="right")
    ax3.set_ylabel("nRMSE (%)")
    figure_title(ax3, "Trait-specific normalized error")
    ax3.legend(frameon=False, handlelength=1.4, labelspacing=0.3, fontsize=FS_LEG)
    style_axis(ax3)
    panel_label(ax3, "c", x=-0.14, y=1.06)
    save_publication_figure(fig, "fig3_proof_of_concept_performance")


def fig4_anomalies():
    t = np.arange(-4, 11)

    def logistic_drop(center, scale, depth):
        return -depth / (1 + np.exp(-(t - center) / scale))

    water = -0.70 * np.exp(-0.5 * ((t - 1) / 1.1) ** 2)
    chl = -0.55 * np.exp(-0.5 * ((t - 1.6) / 1.6) ** 2) - 0.10 * (t > 5)
    lai = logistic_drop(1.5, 1.2, 0.55) + 0.045 * np.maximum(t - 4, 0)
    lai = np.minimum(lai, 0.0)
    nitrogen = -0.18 * np.exp(-0.5 * ((t - 2) / 2.0) ** 2) - 0.18 * (t > 4)
    lma = 0.12 * np.exp(-0.5 * ((t - 3) / 2.0) ** 2) + 0.08 * (t > 4)

    series = {
        "Water trait": (water, BLUE),
        "Chlorophyll": (chl, GREEN),
        "LAI": (lai, PURPLE),
        "Nitrogen": (nitrogen, VERMILION),
        "LMA": (lma, ORANGE),
    }

    fig = plt.figure(figsize=(FIG_W_DOUBLE, 4.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.axvspan(0, 3, color="#F8EEDB", lw=0)
    ax1.axhline(0, color=BLACK, lw=REF_LW)
    for label, (y, color) in series.items():
        ax1.plot(t, y, marker="o", ms=3.2, lw=FIT_LW, label=label, color=color)
    ax1.set_xlabel("Time relative to event")
    ax1.set_ylabel("Trait anomaly")
    figure_title(ax1, "Multi-trait anomalies separate stress pathways")
    ax1.text(1.5, 0.12, "disturbance", ha="center", color=BLACK, fontsize=FS_NOTE)
    ax1.annotate("resistance", xy=(1, -0.67), xytext=(-3.5, -0.62), arrowprops=dict(arrowstyle="->", lw=REF_LW), fontsize=FS_NOTE)
    ax1.annotate("recovery", xy=(7, -0.26), xytext=(4.0, -0.80), arrowprops=dict(arrowstyle="->", lw=REF_LW), fontsize=FS_NOTE)
    ax1.annotate("legacy", xy=(10, -0.18), xytext=(7.5, 0.18), arrowprops=dict(arrowstyle="->", lw=REF_LW), fontsize=FS_NOTE)
    ax1.set_ylim(-0.9, 0.28)
    ax1.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.50, -0.16), handlelength=1.4, labelspacing=0.3)
    style_axis(ax1)
    panel_label(ax1, "a", x=-0.12, y=1.06)

    states = {
        "pre-event": (-0.05, -0.03, GREY),
        "stress": (-0.55, -0.42, VERMILION),
        "apparent\ngreenness\nrecovery": (-0.18, -0.48, ORANGE),
        "functional\nrecovery": (-0.05, -0.10, GREEN),
    }
    ax2.axhline(0, color=BLACK, lw=REF_LW)
    ax2.axvline(0, color=BLACK, lw=REF_LW)
    for label, (xv, yv, color) in states.items():
        ax2.scatter(xv, yv, s=28, color=color, edgecolor=BLACK, linewidth=AXIS_LW, zorder=3)
        ax2.text(xv + 0.03, yv + 0.03, label, fontsize=FS_NOTE, va="bottom", color=BLACK)
    arrow(ax2, states["pre-event"][:2], states["stress"][:2], GREY)
    arrow(ax2, states["stress"][:2], states["apparent\ngreenness\nrecovery"][:2], GREY)
    arrow(ax2, states["apparent\ngreenness\nrecovery"][:2], states["functional\nrecovery"][:2], GREY)
    ax2.set_xlim(-0.75, 0.25)
    ax2.set_ylim(-0.72, 0.18)
    ax2.set_xlabel("Greenness anomaly")
    ax2.set_ylabel("Biochemical anomaly")
    figure_title(ax2, "Same greenness, different function")
    style_axis(ax2)
    panel_label(ax2, "b", x=-0.14, y=1.06)
    fig.subplots_adjust(bottom=0.22)
    save_publication_figure(fig, "fig4_trait_anomaly_syndromes")


def fig5_roadmap():
    fig, ax = plt.subplots(figsize=(FIG_W_DOUBLE, 3.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    columns = [
        ("Observation\nsystems", 0.11, "#DCEAF6"),
        ("Trait anchors", 0.27, "#E6EFD5"),
        ("Temporal\nreconstruction", 0.43, "#F2DEC3"),
        ("Product\ncontract", 0.59, "#DCECDF"),
        ("Ecological\nindicators", 0.75, "#E8DCE8"),
        ("Users and\nmodels", 0.89, "#E8E4D6"),
    ]
    y0, h, w = 0.57, 0.18, 0.12
    for label, x, color in columns:
        add_box(ax, (x - w / 2, y0), (w, h), label, color, fontsize=FS_NOTE)
    for i in range(len(columns) - 1):
        arrow(ax, (columns[i][1] + w / 2, y0 + h / 2), (columns[i + 1][1] - w / 2, y0 + h / 2))

    rows = [
        ("Open spectra-trait-time benchmarks", 0.40, BLUE),
        ("Cross-mission harmonization and sensor provenance", 0.31, SKY),
        ("Uncertainty calibration and domain-of-applicability masks", 0.22, VERMILION),
        ("Links to climate drivers, flux towers and field networks", 0.13, PURPLE),
    ]
    ax.text(0.06, 0.88, "Roadmap from spectral data products to ecosystem insight", fontsize=FS_AXIS, fontweight="bold", ha="left", color=BLACK)
    ax.text(
        0.06,
        0.81,
        "The critical advance is a validated product chain that turns sparse trait anchors into usable monitoring variables.",
        fontsize=FS_NOTE,
        ha="left",
        color=BLACK,
    )
    ax.text(0.08, 0.48, "Minimum requirements for credibility", fontsize=FS_AXIS, fontweight="bold", ha="left", color=BLACK)
    for text, y, color in rows:
        ax.add_patch(Rectangle((0.08, y - 0.015), 0.020, 0.030, facecolor=color, edgecolor=BLACK, linewidth=AXIS_LW))
        ax.text(0.12, y, text, va="center", ha="left", fontsize=FS_NOTE, color=BLACK)
    save_publication_figure(fig, "fig5_monitoring_roadmap")


def main():
    setup()
    fig1_trait_snapshots_to_time_series()
    fig2_framework()
    fig3_performance()
    fig4_anomalies()
    fig5_roadmap()


if __name__ == "__main__":
    main()
