#!/usr/bin/env python3
"""Per-suite mean runtime, base vs treatment, offline pass (paper-style bars).

Values computed from timing.total over all rollouts per (suite, policy) in
offline_eval_full_20260712. Control bar joins at the final sweep. The CodeQA
pair is annotated honestly: the treatment is slower there (delegation-collapse
cap-out loops), consistent with the transfer finding.
Output: thesis_msc/Figures/runtime_offline.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/scratch/omeerdogan23/erlm/thesis_msc/Figures/runtime_offline.pdf"

suites = ["CodeQA", "BC-Plus", "OOLONG", "OOLONG-Pairs"]
base   = [261, 845, 1244, 650]
treat  = [534, 582, 880, 312]

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9.5,
    "xtick.labelsize": 9, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
})
fig, ax = plt.subplots(figsize=(6.0, 2.9))
x = range(4)
w = 0.38
b1 = ax.bar([i - w/2 for i in x], base, w, color="#4878a8", label="base model in the scaffold")
b2 = ax.bar([i + w/2 for i in x], treat, w, color="#c0504d", hatch="//",
            edgecolor="white", lw=0.5, label="treatment @200")

for i, (b, t) in enumerate(zip(base, treat)):
    ax.annotate(f"{b}", (i - w/2, b), xytext=(0, 2), textcoords="offset points",
                ha="center", fontsize=8)
    ax.annotate(f"{t}", (i + w/2, t), xytext=(0, 2), textcoords="offset points",
                ha="center", fontsize=8)
    if t < b:
        note, col = f"$\\downarrow${round((1-t/b)*100)}%  ·  {b/t:.1f}$\\times$ faster", "#2e6b34"
    else:
        note, col = f"$\\uparrow${round((t/b-1)*100)}% runtime", "#8b3a3a"
    ax.annotate(note, (i, max(b, t)), xytext=(0, 14), textcoords="offset points",
                ha="center", fontsize=8, color=col, fontweight="bold")

ax.set_xticks(list(x)); ax.set_xticklabels(suites)
ax.set_ylabel("mean runtime (seconds)")
ax.set_ylim(0, 1500)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT, bbox_inches="tight")
fig.savefig("/scratch/tmp/omeerdogan23/claude-1533459/-scratch-omeerdogan23-erlm/103d5f3d-d42b-452f-a292-f88580d218ee/scratchpad/runtime.png",
            dpi=110, bbox_inches="tight")
print("wrote", OUT)
