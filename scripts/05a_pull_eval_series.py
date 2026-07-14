#!/usr/bin/env python3
"""Pull both arms' in-run eval series from wandb into advisor JSONs.

Writes outputs/advisor/{t2T,t2C}_eval_series.json with, per eval step and
environment: avg@1 accuracy, mean num_turns, truncation share. Dedupe is
keep-LAST per (step, env) in _step order: crashed-and-resumed runs re-measure
steps, and the post-resume measurement is the one the run trained from.

Needs WANDB_API_KEY (rlm/.env). Feeds panels (e,f) of
scripts/05_fig_training_curves.py; rerun both at the final sweep.
"""
import json, os

import wandb

ENT_PROJ = "omeerdogan-koc-university/rlm-qwen3-30b"
RUNS = {"t2T": "07a2004dc2334010a0fd8eebf0809dc9", "t2C": "4f864caba69d400eaa4d45bee418ba1f"}
ENVS = ["oolong-trec-coarse-eval", "browsecomp-plus-eval"]
ADV = os.path.join(os.path.dirname(__file__), "..", "outputs", "advisor")

TRAIN_PREFIXES = ("reward/", "metrics/", "is_truncated/", "optim/", "filters/",
                  "pre_filters/", "loss", "entropy", "kl")

api = wandb.Api()
for tag, rid in RUNS.items():
    run = api.run(f"{ENT_PROJ}/{rid}")
    series = {}
    train = {}
    for row in run.scan_history():
        step = row.get("step")
        for env in ENVS:
            acc = row.get(f"eval/{env}/avg@1")
            if acc is None:
                continue
            series.setdefault(env, {})[step] = dict(
                acc=acc,
                num_turns=row.get(f"eval/{env}/num_turns/mean"),
                truncated=row.get(f"eval/{env}/is_truncated/mean"),
            )
        if step is not None:
            keep = {k: v for k, v in row.items()
                    if v is not None and k.startswith(TRAIN_PREFIXES)}
            if keep:
                train.setdefault(str(int(step)), {}).update(keep)  # keep-last
    out = os.path.join(ADV, f"{tag}_eval_series.json")
    json.dump({"run": rid, "series": series}, open(out, "w"), indent=1)
    outh = os.path.join(ADV, f"{tag}_full_history.json")
    json.dump({"steps": train}, open(outh, "w"))
    n = {e: len(v) for e, v in series.items()}
    print(f"{tag}: wrote {out} {n}; {outh} with {len(train)} training steps")
