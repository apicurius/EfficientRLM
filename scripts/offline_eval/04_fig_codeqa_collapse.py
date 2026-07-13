#!/usr/bin/env python3
"""Figure: codeqa delegation collapse, dose/duration view.

Two panels over {base, released, treatment@120, treatment@200}:
(a) pooled accuracy with question-clustered SE, (b) mean sub-LM calls per
rollout, annotated with the zero-delegation share. Values from
thesis_stats A_accuracy.csv / D_ops_tail.csv / E_codeqa_collapse.csv
(offline_eval_full_20260712). Output: thesis_msc/Figures/codeqa_collapse.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPDF = "/scratch/omeerdogan23/erlm/thesis_msc/Figures/codeqa_collapse.pdf"

labels = ["base", "released", "treat.\n@120", "treat.\n@200"]
acc    = [0.413, 0.440, 0.473, 0.300]
acc_se = [0.047, 0.054, 0.055, 0.047]
subs   = [4.2, 7.0, 4.5, 0.7]
zshare = [0.59, 0.21, 0.55, 0.86]

colors = ["#9e9e9e", "#9e9e9e", "#4878a8", "#c0504d"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.0, 2.3))

x = range(4)
ax1.bar(x, acc, yerr=acc_se, capsize=3, color=colors, width=0.62,
        error_kw=dict(lw=0.9))
ax1.set_xticks(list(x)); ax1.set_xticklabels(labels)
ax1.set_ylabel("accuracy")
ax1.set_ylim(0, 0.60)
ax1.set_title("(a)", loc="left", fontsize=9)

ax2.bar(x, subs, color=colors, width=0.62)
for xi, (s, z) in enumerate(zip(subs, zshare)):
    ax2.annotate(f"{int(round(z*100))}% zero", (xi, s), textcoords="offset points",
                 xytext=(0, 3), ha="center", fontsize=7.5)
ax2.set_xticks(list(x)); ax2.set_xticklabels(labels)
ax2.set_ylabel("mean sub-LM calls / rollout")
ax2.set_ylim(0, 8.4)
ax2.set_title("(b)", loc="left", fontsize=9)

fig.tight_layout(w_pad=2.0)
fig.savefig(OUTPDF, bbox_inches="tight")
print("wrote", OUTPDF)
