from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "generated"
TABLE_DIR = ROOT / "tables"


def setup():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_box(ax, xy, wh, text, fc, ec="#2f3a45", lw=0.9, fontsize=8):
    patch = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.018,rounding_size=0.025",
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
        color="#14212b",
        linespacing=1.2,
    )
    return patch


def arrow(ax, start, end, color="#405466", rad=0.0, lw=1.1):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def fig2_framework():
    fig = plt.figure(figsize=(8.0, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.35], wspace=0.34)
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
    cmap = LinearSegmentedColormap.from_list(
        "functional", ["#f7f3e8", "#d2e5cf", "#7bb6a4", "#2f6f73"]
    )
    im = ax1.imshow(values, vmin=1, vmax=5, cmap=cmap, aspect="auto")
    ax1.set_xticks(np.arange(len(dims)), dims, rotation=25, ha="right")
    ax1.set_yticks(np.arange(len(systems)), systems)
    ax1.tick_params(length=0)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax1.text(j, i, str(values[i, j]), ha="center", va="center", color="#1f2a33")
    ax1.set_title("Complementary observations", loc="left", fontweight="bold")
    for spine in ax1.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_ticks([1, 3, 5])
    cbar.set_ticklabels(["Low", "Medium", "High"])
    cbar.outline.set_visible(False)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title("Sparse-to-dense monitoring", loc="left", fontweight="bold")

    boxes = {
        "spectra": ((0.05, 0.68), (0.24, 0.16), "Sparse imaging\nspectroscopy\ntrait anchors", "#d9e8f5"),
        "structure": ((0.05, 0.43), (0.24, 0.14), "LiDAR / SAR / InSAR\nstructure covariates", "#e7e0d0"),
        "time": ((0.05, 0.20), (0.24, 0.16), "Dense multispectral\ntime series", "#e8efd3"),
        "model": ((0.38, 0.48), (0.24, 0.18), "Temporal\nreconstruction\nmodel", "#f4dfc5"),
        "unc": ((0.68, 0.50), (0.25, 0.18), "Trait estimates\n+ uncertainty\n+ provenance", "#d9eadf"),
        "eco": ((0.68, 0.18), (0.25, 0.16), "Stress, recovery\nand legacy\nmetrics", "#eadce8"),
    }
    for xy, wh, label, color in boxes.values():
        add_box(ax2, xy, wh, label, color)
    arrow(ax2, (0.29, 0.76), (0.38, 0.60))
    arrow(ax2, (0.29, 0.50), (0.38, 0.56))
    arrow(ax2, (0.29, 0.28), (0.38, 0.55))
    arrow(ax2, (0.62, 0.59), (0.68, 0.59))
    arrow(ax2, (0.80, 0.50), (0.80, 0.34))
    ax2.text(0.50, 0.87, "measurement contract", ha="center", fontweight="bold", color="#374b5c")
    ax2.plot([0.36, 0.95], [0.83, 0.83], color="#738899", lw=0.9)
    for x, label in [(0.45, "definition"), (0.61, "domain"), (0.77, "validation"), (0.91, "flags")]:
        ax2.text(x, 0.80, label, ha="center", color="#405466", fontsize=7)

    fig.text(0.01, 0.98, "a", fontweight="bold", fontsize=11, va="top")
    fig.text(0.43, 0.98, "b", fontweight="bold", fontsize=11, va="top")
    save(fig, "fig2_functional_observables_framework")


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

    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95], hspace=0.45, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    cmap = LinearSegmentedColormap.from_list(
        "r2", ["#f6eee6", "#d7e6cf", "#8fc0a9", "#347c7b"]
    )
    im = ax1.imshow(r2.values, vmin=0, vmax=0.7, cmap=cmap, aspect="auto")
    ax1.set_xticks(np.arange(len(model_labels)), model_labels)
    ax1.set_yticks(np.arange(len(trait_order)), trait_order)
    ax1.set_title("Trait- and model-specific skill", loc="left", fontweight="bold")
    ax1.tick_params(length=0)
    for i in range(r2.shape[0]):
        for j in range(r2.shape[1]):
            ax1.text(j, i, f"{r2.iloc[i, j]:.2f}", ha="center", va="center", color="#1f2a33")
    for spine in ax1.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax1, fraction=0.025, pad=0.02)
    cb.set_label("$R^2$")
    cb.outline.set_visible(False)

    colors = ["#2f6f73", "#5b8fb9", "#a76571", "#8c6bb1", "#c6893f"]
    ax2.barh(trait_order, delta.loc[trait_order], color=colors)
    ax2.axvline(0, color="#333333", lw=0.8)
    ax2.set_xlabel(r"$\Delta R^2$ RF - Linear")
    ax2.set_title("Gain over linear baseline", loc="left", fontweight="bold")
    ax2.invert_yaxis()
    for y, val in enumerate(delta.loc[trait_order]):
        ax2.text(val + 0.008, y, f"{val:.2f}", va="center", fontsize=7)

    x = np.arange(len(trait_order))
    ax3.plot(x, rf["nrmse_pct"], marker="o", color="#2f6f73", lw=1.5, label="RF")
    ax3.plot(x, nrmse["LinearRegression"], marker="s", color="#9a6b47", lw=1.1, label="Linear")
    ax3.set_xticks(x, trait_order, rotation=35, ha="right")
    ax3.set_ylabel("nRMSE (%)")
    ax3.set_title("Normalized error remains trait-specific", loc="left", fontweight="bold")
    ax3.legend(frameon=False)
    ax3.grid(axis="y", color="#d8d8d8", lw=0.6)

    fig.text(0.01, 0.98, "a", fontweight="bold", fontsize=11, va="top")
    fig.text(0.01, 0.45, "b", fontweight="bold", fontsize=11, va="top")
    fig.text(0.52, 0.45, "c", fontweight="bold", fontsize=11, va="top")
    save(fig, "fig3_proof_of_concept_performance")


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
        "Water trait": (water, "#3978a8"),
        "Chlorophyll": (chl, "#3f8f5b"),
        "LAI": (lai, "#7b5ba7"),
        "Nitrogen": (nitrogen, "#b45b5b"),
        "LMA": (lma, "#c18a35"),
    }

    fig = plt.figure(figsize=(7.2, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.axvspan(0, 3, color="#f1d2bf", alpha=0.55, lw=0)
    ax1.axhline(0, color="#343434", lw=0.8)
    for label, (y, color) in series.items():
        ax1.plot(t, y, marker="o", ms=3, lw=1.4, label=label, color=color)
    ax1.set_xlabel("Time relative to event")
    ax1.set_ylabel("Trait anomaly")
    ax1.set_title("Multi-trait anomalies separate stress pathways", loc="left", fontweight="bold")
    ax1.text(1.5, 0.12, "disturbance", ha="center", color="#8a4f31", fontweight="bold")
    ax1.annotate("resistance", xy=(1, -0.67), xytext=(-3.5, -0.62), arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=7)
    ax1.annotate("recovery", xy=(7, -0.26), xytext=(4.0, -0.80), arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=7)
    ax1.annotate("legacy", xy=(10, -0.18), xytext=(7.5, 0.18), arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=7)
    ax1.set_ylim(-0.9, 0.28)
    ax1.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.50, -0.13))
    ax1.grid(axis="y", color="#dddddd", lw=0.6)

    states = {
        "pre-event": (-0.05, -0.03, "#526d82"),
        "stress": (-0.55, -0.42, "#b45b5b"),
        "apparent\ngreenness\nrecovery": (-0.18, -0.48, "#c18a35"),
        "functional\nrecovery": (-0.05, -0.10, "#3f8f5b"),
    }
    ax2.axhline(0, color="#777777", lw=0.7)
    ax2.axvline(0, color="#777777", lw=0.7)
    for label, (xv, yv, color) in states.items():
        ax2.scatter(xv, yv, s=90, color=color, edgecolor="white", zorder=3)
        ax2.text(xv + 0.03, yv + 0.03, label, fontsize=7, va="bottom")
    arrow(ax2, states["pre-event"][:2], states["stress"][:2], "#606060")
    arrow(ax2, states["stress"][:2], states["apparent\ngreenness\nrecovery"][:2], "#606060")
    arrow(ax2, states["apparent\ngreenness\nrecovery"][:2], states["functional\nrecovery"][:2], "#606060")
    ax2.set_xlim(-0.75, 0.25)
    ax2.set_ylim(-0.72, 0.18)
    ax2.set_xlabel("Canopy structure anomaly")
    ax2.set_ylabel("Biochemical anomaly")
    ax2.set_title("Same greenness, different function", loc="left", fontweight="bold")

    fig.text(0.01, 0.98, "a", fontweight="bold", fontsize=11, va="top")
    fig.text(0.62, 0.98, "b", fontweight="bold", fontsize=11, va="top")
    fig.subplots_adjust(bottom=0.20)
    save(fig, "fig4_trait_anomaly_syndromes")


def fig5_roadmap():
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    columns = [
        ("Observation\nsystems", 0.11, "#d9e8f5"),
        ("Trait anchors", 0.27, "#e8efd3"),
        ("Temporal\nreconstruction", 0.43, "#f4dfc5"),
        ("Product\ncontract", 0.59, "#d9eadf"),
        ("Ecological\nindicators", 0.75, "#eadce8"),
        ("Users and\nmodels", 0.89, "#e8e2cf"),
    ]
    y0, h, w = 0.58, 0.22, 0.12
    for label, x, color in columns:
        add_box(ax, (x - w / 2, y0), (w, h), label, color, fontsize=7.5)
    for i in range(len(columns) - 1):
        arrow(ax, (columns[i][1] + w / 2, y0 + h / 2), (columns[i + 1][1] - w / 2, y0 + h / 2))

    rows = [
        ("Open spectra-trait-time benchmarks", 0.40, "#2f6f73"),
        ("Cross-mission harmonization and sensor provenance", 0.31, "#5b8fb9"),
        ("Uncertainty calibration and domain-of-applicability masks", 0.22, "#b45b5b"),
        ("Links to climate drivers, flux towers and field networks", 0.13, "#8c6bb1"),
    ]
    for text, y, color in rows:
        ax.add_patch(Rectangle((0.08, y - 0.02), 0.025, 0.04, facecolor=color, edgecolor="none"))
        ax.text(0.12, y, text, va="center", ha="left", fontsize=8)

    ax.text(0.06, 0.91, "Roadmap from spectral data products to ecosystem insight", fontsize=11, fontweight="bold", ha="left")
    ax.text(
        0.06,
        0.85,
        "The critical advance is not only more hyperspectral imagery, but a validated product chain that turns sparse trait anchors into usable monitoring variables.",
        fontsize=8,
        ha="left",
        color="#394b59",
    )
    ax.text(0.08, 0.49, "Minimum requirements for credibility", fontsize=9, fontweight="bold", ha="left")
    save(fig, "fig5_monitoring_roadmap")


def main():
    setup()
    fig2_framework()
    fig3_performance()
    fig4_anomalies()
    fig5_roadmap()


if __name__ == "__main__":
    main()
