import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "figure.dpi": 300, "savefig.dpi": 300, "axes.titlesize": 8, "axes.titleweight": "bold",
})
# npg-style palette
GREY = "#B9B9B9"; GREY2 = "#8C8C8C"; AGENT = "#E64B35"; AGENT2 = "#3C5488"; ACC = "#00A087"
R = "../results"

fig = plt.figure(figsize=(7.16, 2.15))
gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.0, 1.0], wspace=0.42)

# ---- (a) intent->tool-plan F1 (Novel vs Ext) ----
ax = fig.add_subplot(gs[0, 0])
labels = ["fixed", "router\n(char)", "router\n(embed)", "Agent\n(8B)", "Agent\n(GPT-5.5)"]
novel = [0.263, 0.567, 0.501, 0.895, 0.921]
ext = [0.250, 0.579, 0.469, 0.966, np.nan]
x = np.arange(len(labels)); w = 0.4
cols_n = [GREY, GREY, GREY, AGENT, AGENT]
cols_e = [GREY2, GREY2, GREY2, AGENT2, AGENT2]
ax.bar(x - w/2, novel, w, color=cols_n, label="Novel")
ax.bar(x + w/2, [v if v == v else 0 for v in ext], w, color=cols_e, alpha=0.85, label="Ext")
ax.errorbar(x[3] - w/2, novel[3], yerr=[[novel[3]-0.836], [0.947-novel[3]]], fmt="none", ecolor="black", elinewidth=0.7, capsize=2)
ax.axhline(0.613, ls=(0, (4, 3)), color="black", lw=0.6)
ax.text(4.5, 0.625, "best-of-6\nrouters", fontsize=5.6, ha="right", va="bottom", color="#444")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6)
ax.set_ylabel("tool-set F1"); ax.set_ylim(0, 1.02)
ax.set_title("a", loc="left", x=-0.18)
# manual legend (Novel/Ext shades)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=GREY, label="Novel"), Patch(facecolor=GREY2, label="Ext")],
          fontsize=5.8, loc="upper left", handlelength=1.0, bbox_to_anchor=(0.0, 1.02))

# ---- (b) recall on consequential tools ----
ax = fig.add_subplot(gs[0, 1])
tools = ["adapt", "openset\nreject"]; xa = np.arange(2); w = 0.38
ax.bar(xa - w/2, [0.89, 1.00], w, color=AGENT, label="Agent")
ax.bar(xa + w/2, [0.53, 0.39], w, color=GREY, label="best router")
ax.set_xticks(xa); ax.set_xticklabels(tools, fontsize=6)
ax.set_ylabel("recall (1$-$miss rate)"); ax.set_ylim(0, 1.05)
ax.set_title("b", loc="left", x=-0.28)
ax.legend(fontsize=5.8, loc="lower left", handlelength=1.0)

# ---- (c) compound-trained router still trails ----
ax = fig.add_subplot(gs[0, 2])
names = ["router\nbase-only", "router\n+compound", "Agent\n(zero-shot)"]
vals = [0.567, 0.649, 0.895]; cols = [GREY, GREY2, AGENT]
ax.bar(np.arange(3), vals, color=cols, width=0.62)
for i, v in enumerate(vals): ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=6)
ax.set_xticks(np.arange(3)); ax.set_xticklabels(names, fontsize=6)
ax.set_ylabel("novel F1"); ax.set_ylim(0, 1.02)
ax.set_title("c", loc="left", x=-0.28)

fig.savefig(R + "/fig_main.png", bbox_inches="tight", pad_inches=0.02)
print("fig_main.png written")
