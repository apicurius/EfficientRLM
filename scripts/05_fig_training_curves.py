#!/usr/bin/env python3
"""Training-curves figure (fig:res-curves) + run-health cell values.

Reads the deduped wandb histories (advisor/{t2T,t2C}_full_history.json;
training rows are keyed 1..N by trainer step) and renders four panels on the
shared step axis: (a) train reward per environment, (b) mean scaffold cost on
the priced basis, (c) sequence-truncation share, (d) realized lever
coefficient. Control is drawn through its latest completed step.
Output: thesis_msc/Figures/training_curves.pdf + printed health-cell stats.
"""
import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ADV = "/scratch/omeerdogan23/erlm/.research/EfficientRLM/outputs/advisor"
OUT = "/scratch/omeerdogan23/erlm/thesis_msc/Figures/training_curves.pdf"
MAXSTEP = 200

def load(tag):
    d = json.load(open(f"{ADV}/{tag}_full_history.json"))["steps"]
    return {int(k): v for k, v in d.items() if 1 <= int(k) <= MAXSTEP}

def series(h, key):
    xs = sorted(s for s in h if key in h[s])
    return xs, [float(h[s][key]) for s in xs]

W = 10

def roll(ys, w=W):
    out = []
    for i in range(len(ys)):
        lo = max(0, i - w + 1)
        out.append(sum(ys[lo:i+1]) / (i - lo + 1))
    return out

def rollstd(ys, w=W):
    out = []
    for i in range(len(ys)):
        lo = max(0, i - w + 1)
        seg = ys[lo:i+1]
        m = sum(seg) / len(seg)
        out.append((sum((v - m) ** 2 for v in seg) / len(seg)) ** 0.5)
    return out

def full(xs, *series_list):
    """Drop the partial-window transient: keep points from the first full window."""
    k = min(W - 1, max(len(xs) - 1, 0))
    return [s[k:] for s in ([xs] + list(series_list))]

T, C = load("t2T"), load("t2C")
cmax = max(C) if C else 0
print("treatment steps:", len(T), "| control through step", cmax)

ENVS = [("oolong-spam-train", "OOLONG"), ("browsecomp-plus-train", "BC-Plus")]
TCOL = {"oolong-spam-train": "#4878a8", "browsecomp-plus-train": "#c0504d"}
CCOL = {"oolong-spam-train": "#9bb3c9", "browsecomp-plus-train": "#d9a5a3"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
})
fig, axes = plt.subplots(2, 2, figsize=(6.1, 4.25))
(ax1, ax2), (ax3, ax4) = axes

for env, lab in ENVS:
    x, y = series(T, f"reward/{env}/mean")
    xs, m, s = full(x, roll(y), rollstd(y))
    ax1.fill_between(xs, [a-b for a,b in zip(m,s)], [a+b for a,b in zip(m,s)],
                     color=TCOL[env], alpha=0.13, lw=0)
    ax1.plot(xs, m, color=TCOL[env], lw=1.2, label=f"treatment {lab}")
    x, y = series(C, f"reward/{env}/mean")
    xs, m, s = full(x, roll(y), rollstd(y))
    ax1.fill_between(xs, [a-b for a,b in zip(m,s)], [a+b for a,b in zip(m,s)],
                     color=CCOL[env], alpha=0.13, lw=0)
    ax1.plot(xs, m, color=CCOL[env], lw=1.2, ls="--", label=f"control {lab}")
ax1.set_ylabel("train reward"); ax1.set_title("(a)", loc="left")
fig.legend(*ax1.get_legend_handles_labels(), loc="upper center", ncol=4,
           frameon=False, fontsize=7.5, columnspacing=1.2, handlelength=1.8,
           bbox_to_anchor=(0.5, 1.0))

for env, lab in ENVS:
    x, y = series(T, f"metrics/{env}/adaptive_cost")
    xs, m, s = full(x, roll(y), rollstd(y))
    ax2.fill_between(xs, [a-b for a,b in zip(m,s)], [a+b for a,b in zip(m,s)],
                     color=TCOL[env], alpha=0.13, lw=0)
    ax2.plot(xs, m, color=TCOL[env], lw=1.2)
    x, y = series(C, f"metrics/{env}/adaptive_cost")
    xs, m, s = full(x, roll(y), rollstd(y))
    ax2.fill_between(xs, [a-b for a,b in zip(m,s)], [a+b for a,b in zip(m,s)],
                     color=CCOL[env], alpha=0.13, lw=0)
    ax2.plot(xs, m, color=CCOL[env], lw=1.2, ls="--")
ax2.set_ylabel("mean scaffold cost"); ax2.set_title("(b)", loc="left")

x, y = series(T, "is_truncated/all/mean")
ax3.plot(x, roll(y), color="#555555", lw=1.2, label="treatment")
x, y = series(C, "is_truncated/all/mean")
ax3.plot(x, roll(y), color="#aaaaaa", lw=1.2, ls="--", label="control")
ax3.set_ylabel("seq.-truncation share"); ax3.set_title("(c)", loc="left")
ax3.set_xlabel("training step")
ax3.legend(frameon=False, loc="upper right", fontsize=7.5)
ax3.set_ylim(0, 1)

for env, lab in ENVS:
    x, y = series(T, f"metrics/{env}/adaptive_beta")
    ax4.plot(x, roll(y), color=TCOL[env], lw=1.2, label=f"treat. {lab}")
ax4.axhline(0, color="#aaaaaa", lw=1.2, ls="--")
ax4.annotate("control $\\equiv 0$", (0.98, 0.06), xycoords="axes fraction",
             ha="right", fontsize=7, color="#777777")
ax4.set_ylabel("realized lever coeff. $\\beta$"); ax4.set_title("(d)", loc="left")
ax4.set_xlabel("training step")
ax4.set_ylim(-0.005, 0.16)

fig.tight_layout(w_pad=1.6, h_pad=1.2, rect=(0, 0, 1, 0.955))
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)

# ---- run-health cell values (treatment; control partial for reference) ----
def mean(v): return sum(v)/len(v) if v else float("nan")
for tag, H in [("T", T), ("C(partial)", C)]:
    print(f"-- {tag}")
    for env, lab in ENVS:
        _, y = series(H, f"reward/{env}/mean")
        h = len(y)//2
        print(f"  reward {lab}: first-half {mean(y[:h]):.3f} -> second-half {mean(y[h:]):.3f}")
    _, y = series(H, "is_truncated/all/mean"); print(f"  seq-trunc share (run mean): {mean(y):.3f}")
    _, y = series(H, "pre_filters/all/zero_advantage/rate"); print(f"  pre-batch zero-adv rate: {mean(y):.3f}")
    _, y = series(H, "filters/all/zero_advantage"); print(f"  post-batch zero-adv share: {mean(y):.4f}")
    _, y = series(H, "optim/grad_norm"); print(f"  grad norm: mean {mean(y):.3f}, max {max(y):.3f}")
