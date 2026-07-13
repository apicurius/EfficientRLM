#!/usr/bin/env python3
"""The one chart: accuracy vs delegation volume, per suite, offline pass.

One point per policy (base, released, treatment@120, treatment@200), accuracy
with question-clustered SE on y, mean sub-LM calls per rollout on log x.
Arrow: base -> treatment@200. Control point joins each panel at the sweep.
Data: thesis_stats A_accuracy.csv (acc, clustered SE) + D_ops_tail.csv (subs).
Output: thesis_msc/Figures/pareto_plane.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/scratch/omeerdogan23/erlm/thesis_msc/Figures/pareto_plane.pdf"

# suite -> policy -> (subcalls, acc, se_clustered)
D = {
 "CodeQA": {"base": (4.2, .413, .047), "released": (7.0, .440, .054),
                      "treat.@120": (4.5, .473, .055), "treat.@200": (0.7, .300, .047)},
 "BrowseComp+": {"base": (85, .240, .027), "released": (162, .290, .028),
                     "treat.@120": (97, .363, .032), "treat.@200": (60, .370, .031)},
 "OOLONG": {"base": (566, .390, .058), "released": (873, .370, .057),
                 "treat.@120": (357, .310, .055), "treat.@200": (190, .370, .055)},
 "OOLONG-Pairs": {"base": (322, .355, .053), "released": (493, .493, .058),
                  "treat.@120": (249, .400, .044), "treat.@200": (263, .418, .042)},
}
STYLE = {"base": dict(c="#7f7f7f", m="o"), "released": dict(c="#7f7f7f", m="D"),
         "treat.@120": dict(c="#d9a5a3", m="s"), "treat.@200": dict(c="#c0504d", m="s")}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
})
fig, axes = plt.subplots(2, 2, figsize=(6.1, 4.4))
for ax, (suite, pols) in zip(axes.flat, D.items()):
    for pol, (x, y, se) in pols.items():
        st = STYLE[pol]
        ax.errorbar(x, y, yerr=se, fmt=st["m"], color=st["c"], ms=6,
                    capsize=2, lw=1, label=pol,
                    markerfacecolor=st["c"], markeredgecolor="white", zorder=3)
    bx, by, _ = pols["base"]; tx, ty, _ = pols["treat.@200"]
    ax.annotate("", xy=(tx, ty), xytext=(bx, by),
                arrowprops=dict(arrowstyle="->", color="#c0504d", lw=1.1, alpha=0.7))
    ax.set_xscale("log")
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_title(suite, fontsize=9)
    ax.set_ylim(0.15, 0.60)
axes[0][0].set_ylabel("accuracy"); axes[1][0].set_ylabel("accuracy")
axes[1][0].set_xlabel("mean sub-LM calls per rollout (log)")
axes[1][1].set_xlabel("mean sub-LM calls per rollout (log)")
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.4)
fig.savefig(OUT, bbox_inches="tight")
fig.savefig("/scratch/tmp/omeerdogan23/claude-1533459/-scratch-omeerdogan23-erlm/103d5f3d-d42b-452f-a292-f88580d218ee/scratchpad/pareto.png",
            dpi=110, bbox_inches="tight")
print("wrote", OUT)
