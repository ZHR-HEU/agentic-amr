"""Generate publication-style panels for the main tool-planning figure."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TEXT = "#262626"
AXIS = "#555555"
GRID = "#E8E8E8"

PALETTE = {
    "router_soft": "#AEB8D3",
    "router_mid": "#7F8CAE",
    "router_dark": "#44537F",
    "agent_1": "#E6B39E",
    "agent_2": "#D69A84",
    "agent_3": "#C27769",
    "agent_4": "#8A5D88",
    "agent_5": "#C84D3B",
    "accent_down": "#C93A31",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.55,
        "axes.labelsize": 7,
        "axes.labelpad": 2.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 5.8,
        "legend.frameon": False,
        "legend.handlelength": 1.0,
        "legend.handletextpad": 0.35,
        "legend.labelspacing": 0.25,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.025,
        "mathtext.default": "regular",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.color": TEXT,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
    }
)


router_data = [
    ("fixed", 0.263),
    ("router-nn", 0.514),
    ("router-ml", 0.507),
    ("router-char", 0.567),
    ("router-ml-char", 0.567),
    ("router-desc", 0.477),
    ("router-emb", 0.501),
    ("best router", 0.613),
]

agent_data = [
    ("Mistral-7B", 0.856, PALETTE["agent_1"]),
    ("Gemma-4-12B", 0.876, PALETTE["agent_2"]),
    ("Qwen3-8B", 0.895, PALETTE["agent_3"]),
    ("Qwen3-30B-A3B", 0.902, PALETTE["agent_4"]),
    ("GPT-5.5", 0.921, PALETTE["agent_5"]),
]

cons_tools = ["adapt", "openset\nreject"]
agent_recall = [0.89, 1.00]
router_recall = [0.53, 0.39]
agent_spurious = [0.05, 0.18]

compound_labels = ["Best\nrouter", "Compound\nrouter", "Qwen3\n8B", "GPT\n5.5"]
compound_vals = [0.567, 0.649, 0.895, 0.921]
compound_cols = [
    PALETTE["router_mid"],
    PALETTE["router_dark"],
    PALETTE["agent_3"],
    PALETTE["agent_5"],
]


def default_output_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    paper_figs = root / "paper" / "figs"
    if paper_figs.exists():
        return paper_figs
    return root / "figs"


def finish_axis(ax, *, ylabel=None, xlabel=None, xlim=None, ylim=None, rate_y=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="both", colors=TEXT, pad=2)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(*xlim)
    if rate_y:
        ax.set_ylim(*(ylim if ylim else (0, 1.02)))
        ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
        ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    elif ylim:
        ax.set_ylim(*ylim)
    ax.set_axisbelow(True)


def save(fig, out_dir: Path, name: str):
    for ext in ("svg", "pdf", "png"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, facecolor="white")
        print(f"  {path}")
    plt.close(fig)


def draw_panel_a(ax):
    names = [item[0] for item in router_data] + [item[0] for item in agent_data]
    values = [item[1] for item in router_data] + [item[1] for item in agent_data]
    n_router = len(router_data)
    y = np.arange(len(names))

    ax.axhline(n_router - 0.5, color="#D6D6D6", lw=0.55, zorder=0)

    for i, (name, val) in enumerate(zip(names, values)):
        if i < n_router:
            color = PALETTE["router_dark"] if name == "best router" else PALETTE["router_soft"]
            marker = "o"
            size = 18 if name == "best router" else 13
            alpha = 0.62 if name != "best router" else 0.95
        else:
            color = agent_data[i - n_router][2]
            marker = "D"
            size = 20
            alpha = 0.95

        ax.hlines(i, 0, val, color=color, lw=0.8, alpha=alpha, zorder=2)
        ax.scatter(
            val,
            i,
            s=size,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )

        if name == "Qwen3-8B":
            ax.errorbar(
                val,
                i,
                xerr=[[val - 0.836], [0.947 - val]],
                fmt="none",
                ecolor=color,
                elinewidth=0.7,
                capsize=2,
                capthick=0.7,
                zorder=3,
            )

        if name == "best router" or i >= n_router:
            label_x = 0.952 if name == "Qwen3-8B" else min(val + 0.018, 0.965)
            ax.text(
                label_x,
                i,
                f"{val:.3f}",
                va="center",
                ha="left",
                fontsize=5.4,
                color=color if i >= n_router else PALETTE["router_dark"],
            )

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
    ax.grid(axis="x", color=GRID, linewidth=0.45, zorder=0)
    finish_axis(ax, xlabel="Tool-set F1 on novel scenarios", xlim=(0, 1.0))


def draw_panel_b(ax):
    x = np.arange(len(cons_tools))
    width = 0.34
    agent_bars = ax.bar(
        x - width / 2,
        agent_recall,
        width,
        label="Agent recall",
        color=PALETTE["agent_3"],
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    router_bars = ax.bar(
        x + width / 2,
        router_recall,
        width,
        label="Router recall",
        color=PALETTE["router_mid"],
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    ax.scatter(
        x - width / 2,
        agent_spurious,
        marker="v",
        s=22,
        facecolor="white",
        edgecolor=PALETTE["accent_down"],
        linewidth=0.9,
        zorder=5,
    )

    for bars, vals, color in (
        (agent_bars, agent_recall, PALETTE["agent_3"]),
        (router_bars, router_recall, PALETTE["router_mid"]),
    ):
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.026,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=5.3,
                color=color,
            )

    for xi, val in zip(x - width / 2, agent_spurious):
        ax.text(
            xi,
            val + 0.035,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=5.1,
            color=PALETTE["accent_down"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(cons_tools)
    ax.grid(axis="y", color=GRID, linewidth=0.45, zorder=0)
    finish_axis(ax, ylabel="Recall and false-call rate", ylim=(0, 1.08), rate_y=True)


def draw_panel_c(ax):
    x = np.arange(len(compound_labels))
    bars = ax.bar(
        x,
        compound_vals,
        color=compound_cols,
        edgecolor="white",
        linewidth=0.6,
        width=0.64,
        zorder=3,
    )
    for bar, val, color in zip(bars, compound_vals, compound_cols):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.022,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=5.4,
            color=color,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(compound_labels)
    ax.grid(axis="y", color=GRID, linewidth=0.45, zorder=0)
    finish_axis(ax, ylabel="Tool-set F1", rate_y=True)


def make_individual_panels(out_dir: Path):
    specs = [
        ("fig_main_a", (3.35, 2.65), draw_panel_a),
        ("fig_main_b", (2.15, 2.65), draw_panel_b),
        ("fig_main_c", (2.45, 2.65), draw_panel_c),
    ]
    for name, figsize, drawer in specs:
        fig, ax = plt.subplots(figsize=figsize)
        drawer(ax)
        fig.tight_layout(pad=0.5)
        save(fig, out_dir, name)


def make_composite(out_dir: Path):
    fig = plt.figure(figsize=(7.16, 2.45))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.0, 1.05], wspace=0.42)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    draw_panel_a(axes[0])
    draw_panel_b(axes[1])
    draw_panel_c(axes[2])
    for label, ax in zip(("a", "b", "c"), axes):
        ax.text(
            -0.15,
            1.03,
            label,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    fig.tight_layout(pad=0.45)
    save(fig, out_dir, "fig_main")


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    make_individual_panels(out_dir)
    make_composite(out_dir)
    print("Done: publication-style main figure panels saved.")


if __name__ == "__main__":
    main()
