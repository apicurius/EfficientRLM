#!/usr/bin/env python3
"""Pre-registered BC+ comparison: bcplus-only smoke vs the 200-step multienv treatment.

Answers "are we better from the BC+ perspective?" on FOUR pre-registered axes,
each read as change-from-own-baseline (levels are not comparable: the smoke's
deterministic SHA-256 doc sampling realizes different document mixes than the
old salted-hash env, shifting the step-0 eval level).

Axes (gates fixed 2026-07-05, before smoke completion):
  A. Eval trend: eval avg@1 delta from own step-0, at raw steps AND at
     exposure-aligned steps (smoke step k ~ treatment step 2k in BC+ samples,
     since the smoke is 100% BC+ vs ~50% mix).
  B. Train solve: mean gated_reward over the last 10 steps vs first 10 steps.
  C. Operation health: sub_llm_calls trend (collapse = late mean < 0.33x early
     mean, the treatment's 17.9 -> 3.1 signature); has_final; rejections.
  D. Dead-group share: filters zero_advantage mean late vs early.

Usage:  .venv/bin/python compare_bcplus_smoke.py   (needs WANDB_API_KEY)
"""

import json
import os
import sys
from statistics import fmean

import wandb

ENTITY = "omeerdogan-koc-university"
PROJECT = "rlm-qwen3-30b"
SMOKE = "c678d3df80f44d70a7d995c520559d61"
TREATMENT_SEGMENTS = [
    "c5462c93c7a544a1b82a7473f96069e1",  # steps 0-121
    "312bf33724ea41478175180b4cc06169",  # steps ~120-200 (stitch on `step`, later wins)
]

TRAIN_KEYS = [
    "step",
    "metrics/browsecomp-plus-train/gated_reward",
    "metrics/browsecomp-plus-train/rlm_sub_llm_calls",
    "metrics/browsecomp-plus-train/rlm_has_final_answer",
    "metrics/browsecomp-plus-train/rlm_sub_llm_prompt_rejections",
    "filters/browsecomp-plus-train/zero_advantage",
]
EVAL_KEYS = ["eval/browsecomp-plus-eval/avg@1", "eval/browsecomp-plus-eval/policy_version"]


def fetch(run_id, keys):
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    rows = []
    for row in run.scan_history(keys=keys):
        if any(row.get(k) is not None for k in keys if k != "step"):
            rows.append(row)
    return rows


def stitch_train(segments):
    by_step = {}
    for seg in segments:
        for r in fetch(seg, TRAIN_KEYS):
            s = r.get("step")
            if s is not None:
                by_step[int(s)] = r  # later segment overwrites on overlap
    return [by_step[s] for s in sorted(by_step)]


def series(rows, key):
    return [(int(r["step"]), r[key]) for r in rows if r.get(key) is not None and r.get("step") is not None]


def window_mean(sr, lo, hi):
    vals = [v for s, v in sr if lo <= s <= hi]
    return fmean(vals) if vals else None


def evals(run_id):
    out = {}
    for r in fetch(run_id, EVAL_KEYS):
        pv = r.get("eval/browsecomp-plus-eval/policy_version")
        av = r.get("eval/browsecomp-plus-eval/avg@1")
        if pv is not None and av is not None:
            out[int(pv)] = av
    return dict(sorted(out.items()))


def main():
    smoke_rows = fetch(SMOKE, TRAIN_KEYS)
    trt_rows = stitch_train(TREATMENT_SEGMENTS)
    smoke_eval = evals(SMOKE)
    trt_eval = {}
    for seg in TREATMENT_SEGMENTS:
        trt_eval.update(evals(seg))
    trt_eval = dict(sorted(trt_eval.items()))

    smoke_max = max((int(r["step"]) for r in smoke_rows if r.get("step") is not None), default=0)
    early_hi = min(9, smoke_max)
    late_lo = max(smoke_max - 9, 0)

    report = {"smoke_last_step": smoke_max, "axes": {}}

    # A. eval deltas
    s0 = smoke_eval.get(0)
    t0 = trt_eval.get(0)
    a = {"smoke_evals": smoke_eval, "treatment_evals_0to80": {k: v for k, v in trt_eval.items() if k <= 80}}
    if s0 is not None:
        a["smoke_delta_final"] = (list(smoke_eval.values())[-1] - s0) if len(smoke_eval) > 1 else None
    if t0 is not None:
        for k in (39, 40, 79, 80):
            if k in trt_eval:
                a[f"treatment_delta_at_{k}"] = trt_eval[k] - t0
    report["axes"]["A_eval"] = a

    # B/C/D windows
    for name, key, run_rows in [
        ("B_train_solve", "metrics/browsecomp-plus-train/gated_reward", None),
        ("C_subcalls", "metrics/browsecomp-plus-train/rlm_sub_llm_calls", None),
        ("C_has_final", "metrics/browsecomp-plus-train/rlm_has_final_answer", None),
        ("C_rejections", "metrics/browsecomp-plus-train/rlm_sub_llm_prompt_rejections", None),
        ("D_zero_adv", "filters/browsecomp-plus-train/zero_advantage", None),
    ]:
        s = series(smoke_rows, key)
        t = series(trt_rows, key)
        report["axes"][name] = {
            "smoke_early(0..%d)" % early_hi: window_mean(s, 0, early_hi),
            "smoke_late(%d..%d)" % (late_lo, smoke_max): window_mean(s, late_lo, smoke_max),
            "treatment_0..9": window_mean(t, 0, 9),
            "treatment_raw_%d..%d" % (late_lo, smoke_max): window_mean(t, late_lo, smoke_max),
            "treatment_exposure_%d..%d" % (2 * late_lo, 2 * smoke_max): window_mean(t, 2 * late_lo, 2 * smoke_max),
            "treatment_final_190..200": window_mean(t, 190, 200),
        }

    # gate verdicts
    g = {}
    b = report["axes"]["B_train_solve"]
    sl, se = b.get("smoke_late(%d..%d)" % (late_lo, smoke_max)), b.get("smoke_early(0..%d)" % early_hi)
    g["B_solve_not_collapsing"] = (sl is not None and se is not None and sl >= min(0.3, se * 0.8))
    c = report["axes"]["C_subcalls"]
    cl, ce = c.get("smoke_late(%d..%d)" % (late_lo, smoke_max)), c.get("smoke_early(0..%d)" % early_hi)
    g["C_no_delegation_collapse"] = (cl is not None and ce is not None and ce > 0 and cl >= 0.33 * ce)
    d = report["axes"]["D_zero_adv"]
    dl = d.get("smoke_late(%d..%d)" % (late_lo, smoke_max))
    g["D_zero_adv_below_0.5_late"] = (dl is not None and dl < 0.5)
    sd = report["axes"]["A_eval"].get("smoke_delta_final")
    g["A_eval_delta_nonnegative"] = (sd is not None and sd >= 0.0)
    report["gates"] = g
    report["verdict_better_for_bcplus"] = all(v for v in g.values() if v is not None)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
