#!/usr/bin/env python3
"""Offline discovery-credit scorer (pre-registered post-A/B amendment, 2026-07-06).

Measures, from saved rollouts only (no env/stack change), the discovery-
without-selection rate in all-wrong groups, and replays the proposed
advantage-layer discovery credit on them: in a group with zero correct
rollouts, shaped_i = DELTA * discovered_i (discovered = the gold answer
string appears anywhere in the rollout's transcript), then group-mean
centering as in adaptive_group. Correctness dominance is preserved
structurally because DELTA < 1 - beta_max: every correct rollout in any
group outranks every discovered-but-wrong rollout in any group.

Baseline measurement (multienv-20 smoke, 2026-07-06): 44% of BC+ groups
all-wrong; 39% of those contained a discovering sibling; 0 died of
coordinated overflow. Expected rescue: ~17% of BC+ groups gain gradient
pointing at finalize-what-you-found.

Usage: discovery_credit_offline.py <run_dir>/run_default/rollouts [--delta 0.2]
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

DELTA = 0.2


def gold_strings(task: dict) -> list[str]:
    a = task.get("answer")
    outs = [str(x) for x in a] if isinstance(a, list) else ([str(a)] if a is not None else [])
    return [s.strip().lower() for s in outs if s and len(str(s).strip()) >= 3]


def discovered(rollout: dict, golds: list[str]) -> bool:
    if not golds:
        return False
    txt = " ".join((nd["message"].get("content") or "") for nd in rollout["nodes"]).lower()
    return any(g in txt for g in golds)


def main() -> int:
    root = sys.argv[1]
    delta = float(sys.argv[sys.argv.index("--delta") + 1]) if "--delta" in sys.argv else DELTA
    dead = disc_groups = rescued_rollouts = 0
    envs = defaultdict(lambda: [0, 0])
    for p in sorted(glob.glob(os.path.join(root, "step_*/train_rollouts.jsonl"))):
        groups = defaultdict(list)
        for line in open(p):
            d = json.loads(line)
            e = "bcplus" if "BrowseComp" in d["task"]["prompt"][:4000] else "oolong"
            groups[(e, d["task"]["idx"])].append(d)
        for (e, _), rs in groups.items():
            envs[e][1] += 1
            if any((r["metrics"].get("gated_reward") or 0) > 0 for r in rs):
                continue
            dead += 1
            envs[e][0] += 1
            golds = gold_strings(rs[0]["task"])
            flags = [discovered(r, golds) for r in rs]
            if any(flags):
                disc_groups += 1
                # replayed credit: shaped = delta*flag, centered -> nonzero advantages
                shaped = [delta if f else 0.0 for f in flags]
                base = sum(shaped) / len(shaped)
                rescued_rollouts += sum(1 for x in shaped if abs(x - base) > 1e-12)
    print(f"all-wrong groups: {dead} | with a discovering sibling: {disc_groups} "
          f"({disc_groups / max(dead, 1):.0%}) | rollouts gaining gradient at delta={delta}: {rescued_rollouts}")
    for e, (d, n) in envs.items():
        print(f"  {e}: dead {d}/{n} groups ({d / max(n, 1):.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
