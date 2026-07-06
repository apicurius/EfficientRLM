#!/usr/bin/env python3
"""Lever-direction guard (pre-registered tripwire, 2026-07-06).

Per-step, from saved rollouts: (a) share of cost-fired groups whose
cheapest-valid sibling is also the min-sub-call sibling (delegation-lean;
baseline 73% on the multienv-20 smoke), and (b) per-env mean sub-LLM calls.

Tripwire (evaluated across arms, not per run): treatment BC+ sub_llm_calls
falling below ~50% of its early-run (steps 0-20) mean while control's stays
flat = lever-owned delegation suppression -> pre-registered response is the
`iterations` cost basis for the NEXT run (never a mid-run change). If control
decays too, the drift is generic RL and the lever is exonerated.

Usage: monitor_lever_guard.py <run_dir>/run_default/rollouts [--last N]
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict


def env_of(prompt: str) -> str:
    return "bcplus" if "BrowseComp" in prompt[:4000] else "oolong"


def main() -> int:
    root = sys.argv[1]
    last = int(sys.argv[sys.argv.index("--last") + 1]) if "--last" in sys.argv else 10**9
    steps = sorted(
        int(os.path.basename(os.path.dirname(p)).split("_")[1])
        for p in glob.glob(os.path.join(root, "step_*/train_rollouts.jsonl"))
    )[-last:]
    print(f"{'step':>4} {'fired':>6} {'cheap=minsub':>12} {'sub/roll bcp':>12} {'sub/roll ool':>12}")
    for s in steps:
        groups = defaultdict(list)
        for line in open(os.path.join(root, f"step_{s}", "train_rollouts.jsonl")):
            d = json.loads(line)
            groups[(env_of(d["task"]["prompt"]), d["task"]["idx"])].append(d)
        fired = lean = 0
        subs = defaultdict(list)
        for (e, _), rs in groups.items():
            for r in rs:
                subs[e].append(r["metrics"].get("rlm_sub_llm_calls") or 0)
            vi = [r for r in rs if r["metrics"].get("adaptive_valid")]
            if len(vi) < 2 or not any((r["metrics"].get("adaptive_normalized_cost") or 0) > 0 for r in rs):
                continue
            fired += 1
            costs = [
                (r["metrics"].get("rlm_iterations") or 0)
                + math.log1p(r["metrics"].get("rlm_sub_llm_calls") or 0)
                for r in vi
            ]
            sc = [r["metrics"].get("rlm_sub_llm_calls") or 0 for r in vi]
            if sc[costs.index(min(costs))] == min(sc):
                lean += 1
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
        share = f"{lean}/{fired}" if fired else "-"
        print(
            f"{s:>4} {fired:>6} {share:>12} "
            f"{mean(subs.get('bcplus', [])):>12.1f} {mean(subs.get('oolong', [])):>12.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
