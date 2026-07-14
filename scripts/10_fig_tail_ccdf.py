#!/usr/bin/env python3
"""Tail figure (fig:tail-ccdf): per-rollout sub-call survival curves.

CCDF P[sub-calls >= x] per policy on the offline pass, one panel per training
family, repetitions pooled. The runaway sub-call tail is the released
scaffold's own flagged open issue; the survival curve shows the whole tail
rather than the p95 summary cell of tbl:res-tail. Descriptive exhibit:
base/released are external references and identify nothing about the lever;
control curves join at the final sweep.

Output: thesis_msc/Figures/tail_ccdf.pdf
"""
import glob, json, math, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/scratch/omeerdogan23/erlm/thesis_msc/Figures/tail_ccdf.pdf"
ROOT = os.path.join(os.path.dirname(__file__), "..")
POL = {"Qwen_Qwen3-30B-A3B-Instruct-2507": "base", "authors": "released",
       "t2T_120": "treatment @120", "t2T_final": "treatment @200"}
FAMS = {
    "OOLONG": ["outputs/offline_eval_full_20260712/trec_coarse_131k",
               "outputs/offline_eval_full_20260712/trec_rep2"],
    "BrowseComp+": ["outputs/offline_eval_full_20260712/bcplus_heldout",
                    "outputs/offline_eval_full_20260712/bcplus_rep2"],
}
STYLE = {  # matches fig:codeqa-collapse / fig:pareto-plane conventions
    "base":            dict(c="#9e9e9e", ls=":",  lw=1.3),
    "released":        dict(c="#7f7f7f", ls="--", lw=1.3),
    "treatment @120":  dict(c="#4878a8", ls="-",  lw=1.5),
    "treatment @200":  dict(c="#c0504d", ls="-",  lw=1.7),
}


def load():
    data = {}
    for fam, dirs in FAMS.items():
        for d in dirs:
            for f in glob.glob(os.path.join(ROOT, d, "*/evals/*/*/results.jsonl")):
                pol = POL.get(os.path.relpath(f, os.path.join(ROOT, d)).split(os.sep)[0])
                if not pol:
                    continue
                for line in open(f):
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    m = r.get("metrics") or {}
                    data.setdefault(fam, {}).setdefault(pol, []).append(
                        float(m.get("rlm_sub_llm_calls") or 0))
    return data


def ccdf(vals):
    xs = sorted(vals)
    n = len(xs)
    # survival at each distinct value: share of rollouts with subs >= x
    pts_x, pts_y = [], []
    for i, x in enumerate(xs):
        if i > 0 and x == xs[i - 1]:
            continue
        pts_x.append(max(x, 1.0))  # log axis; x=0 plotted at 1 (P[>=1] step)
        pts_y.append((n - i) / n)
    return pts_x, pts_y


def p95(vals):
    xs = sorted(vals)
    return xs[min(len(xs) - 1, int(0.95 * len(xs)))]


data = load()
plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
})
fig, axes = plt.subplots(1, 2, figsize=(6.1, 2.9), sharey=True)

for ax, fam in zip(axes, FAMS):
    for pol in ("base", "released", "treatment @120", "treatment @200"):
        vals = data[fam][pol]
        x, y = ccdf(vals)
        st = STYLE[pol]
        ax.plot(x, y, drawstyle="steps-post", label=pol if fam == "OOLONG" else None, **st)
        q = p95(vals)
        ax.plot([max(q, 1)], [0.05], marker="|", ms=7, mew=1.4, color=st["c"])
    ax.set_xscale("log")
    ax.set_xlim(1, 2e4)
    ax.set_ylim(0, 1.02)
    ax.set_title(fam, fontsize=8.5)
    ax.set_xlabel("sub-LM calls per rollout, $x$")
    ax.axhline(0.05, color="#cccccc", lw=0.6, ls=(0, (1, 2)), zorder=0)
axes[0].set_ylabel("share of rollouts with $\\geq x$ sub-calls")
axes[0].annotate("p95", (1.3, 0.07), fontsize=6.5, color="#999999")

# selective annotation: the headline tail contrast per panel, in text ink
for ax, fam in zip(axes, FAMS):
    q_rel = p95(data[fam]["released"])
    q_trt = p95(data[fam]["treatment @200"])
    ax.annotate(f"p95 sub-calls\nreleased {q_rel:,.0f}\ntreatment @200 {q_trt:,.0f}",
                (0.97, 0.80), xycoords="axes fraction", ha="right", va="top",
                fontsize=7, color="#444444", linespacing=1.4)

fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4,
           frameon=False, columnspacing=1.4, handlelength=2.2,
           bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
for fam in FAMS:
    for pol in ("base", "released", "treatment @120", "treatment @200"):
        v = data[fam][pol]
        print(f"{fam:12s} {pol:16s} n={len(v):4d} p95={p95(v):7,.0f}")
