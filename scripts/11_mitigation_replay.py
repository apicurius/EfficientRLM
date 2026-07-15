"""Offline counterfactual replay of the advantage operator on control's real
rollout groups: today's symmetric lever (V0) vs one-sided tail tau=0.5 (V1)
vs one-sided + delegation-floor basis (V2). No training code touched."""
import glob, json, math, os
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..", "outputs", "qwen3-30b-t2-control", "run_default", "rollouts")
BETA_MAX, SOLVE_FLOOR, GAMMA, MIN_SPAN, TAU, LAM = 0.15, 0.25, 1.0, 1.0, 0.5, 1.0

def variants(group):
    """group: list of dicts(correct, fatal, it, subs). Returns {name: advantages}"""
    n = len(group)
    valid = [g["correct"] > 0 and not g["fatal"] for g in group]
    sr = sum(valid) / n
    beta = 0.0 if sr <= SOLVE_FLOOR else BETA_MAX * ((sr - SOLVE_FLOOR) / (1 - SOLVE_FLOOR)) ** GAMMA
    out = {}
    for name, floor, tau, iters_only in (("V0_symmetric", False, None, False), ("V1_tail", False, TAU, False), ("V2_tail_floor", True, TAU, False), ("V3_iters_sym", False, None, True), ("V4_iters_tail", False, TAU, True)):
        costs = [ (g["it"] if iters_only else g["it"] + math.log1p(g["subs"])) + (LAM if (floor and g["subs"] == 0) else 0.0) for g in group]
        shaped = [1.0 if v else 0.0 for v in valid]
        vi = [i for i, v in enumerate(valid) if v]
        fired = False
        if beta > 0 and len(vi) >= 2:
            vc = [costs[i] for i in vi]
            lo, hi = min(vc), max(vc)
            span = hi - lo
            if span > 0 and span >= MIN_SPAN:
                fired = True
                for i in vi:
                    rel = (costs[i] - lo) / span
                    if tau is not None:
                        rel = max(0.0, rel - tau) / (1.0 - tau)
                    shaped[i] = 1.0 - beta * rel
        base = sum(shaped) / n
        out[name] = dict(A=[s - base for s in shaped], fired=fired, beta=beta)
    return out

groups = defaultdict(list)
for step_dir in sorted(glob.glob(f"{ROOT}/step_*")):
    for line in open(f"{step_dir}/train_rollouts.jsonl"):
        try: r = json.loads(line)
        except Exception: continue
        m = r.get("metrics") or {}
        info = r.get("info") or {}
        if isinstance(info, str):
            try: info = json.loads(info)
            except Exception: info = {}
        env = "bcp" if "browsecomp_plus_judge_score" in m else "oolong"
        task = r.get("task") or {}
        idx = task.get("idx") if isinstance(task, dict) else None
        groups[(step_dir, env, idx)].append(dict(
            correct=float(m.get("gated_reward") or 0),
            fatal=(str(r.get("stop_condition")) == "max_turns_reached" or not m.get("rlm_has_final_answer")),
            it=float(m.get("rlm_iterations") or 0),
            subs=float(m.get("rlm_sub_llm_calls") or 0)))

stats = {v: defaultdict(lambda: defaultdict(float)) for v in ("V0_symmetric","V1_tail","V2_tail_floor","V3_iters_sym","V4_iters_tail")}
sizes = defaultdict(int)
for (sd, env, gid), grp in groups.items():
    sizes[len(grp)] += 1
    if len(grp) != 4: continue
    res = variants(grp)
    valid_idx = [i for i, g in enumerate(grp) if g["correct"] > 0 and not g["fatal"]]
    for name, r in res.items():
        S = stats[name][env]
        S["groups"] += 1
        if not r["fired"]: continue
        S["fired"] += 1
        A = r["A"]
        vs = [(grp[i]["subs"], A[i]) for i in valid_idx]
        # anti-delegation gradient: within-group cov(A, subs) among valid
        if len(vs) >= 2:
            ms = sum(s for s, _ in vs) / len(vs); ma = sum(a for _, a in vs) / len(vs)
            cov = sum((s - ms) * (a - ma) for s, a in vs) / len(vs)
            var = sum((s - ms) ** 2 for s, _ in vs) / len(vs)
            if var > 0:
                S["slope_sum"] += cov / var
                S["slope_n"] += 1
        # abstinence-wins: top-advantage valid member has the min subs / subs<=1
        maxA = max(A[i] for i in valid_idx)
        tops = [i for i in valid_idx if A[i] >= maxA - 1e-12]
        min_subs = min(grp[i]["subs"] for i in valid_idx)
        if len(tops) == 1 and grp[tops[0]]["subs"] == min_subs: S["minsubs_wins"] += 1
        if len(tops) == 1 and grp[tops[0]]["subs"] <= 1: S["lowsub_wins"] += 1
        # signal mass among valid
        S["absA"] += sum(abs(A[i]) for i in valid_idx) / len(valid_idx)
print("group sizes:", dict(sizes))
for env in ("oolong", "bcp"):
    print(f"\n== {env} ==")
    print(f"{'variant':14s} {'fired':>6s} {'dA/dsubs(x1e4)':>14s} {'minsubs-wins':>12s} {'lowsub-wins':>11s} {'mean|A|':>8s}")
    for name in ("V0_symmetric","V1_tail","V2_tail_floor","V3_iters_sym","V4_iters_tail"):
        S = stats[name][env]
        f = S["fired"] or 1
        print(f"{name:14s} {int(S['fired']):6d} {1e4*S['slope_sum']/max(S['slope_n'],1):14.2f} "
          f"{S['minsubs_wins']/f:12.2f} {S['lowsub_wins']/f:11.2f} {S['absA']/f:8.4f}")
