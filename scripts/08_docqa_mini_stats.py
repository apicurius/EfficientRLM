#!/usr/bin/env python3
"""Verify the extension eval (Doc-QA mini + CodeQA rerun) against the prereg.

Stdlib-only; run on the tree (local or on the studio):
  python scripts/08_docqa_mini_stats.py outputs/offline_eval_ext_20260714 [--json OUT]

Per (family, policy): n, accuracy (reward>=0.5), question-clustered SE, mean
sub-calls, zero-sub share, cap-out rate, finalized rate. Paired deltas vs base
and @200-vs-@120 by question id. Strata by stored context size (chars/4 ~
tokens) split at 16384: fits-in-window vs exceeds-window, for P-M3/P-M4.
"""
import argparse, glob, json, math, os, sys
from collections import defaultdict

POL = {"Qwen_Qwen3-30B-A3B-Instruct-2507": "base", "authors": "authors",
       "t2T_120": "t2T_120", "t2T_final": "t2T_200", "t2T_200": "t2T_200"}
WINDOW_TOK = 16384

def mean(v): return sum(v)/len(v) if v else float("nan")
def se(v):
    if len(v) < 2: return float("nan")
    m = mean(v); return math.sqrt(sum((x-m)**2 for x in v)/(len(v)-1)/len(v))

def load(root):
    fam = {}
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p) or d == "logs": continue
        for f in glob.glob(f"{p}/*/evals/*/*/results.jsonl"):
            poldir = os.path.relpath(f, p).split(os.sep)[0]
            pol = POL.get(poldir, poldir)
            for line in open(f):
                try: r = json.loads(line)
                except Exception: continue
                m = r.get("metrics") or {}
                info = r.get("info") or {}
                if isinstance(info, str):
                    try: info = json.loads(info)
                    except Exception: info = {}
                ctx = info.get("context") or ""
                fam.setdefault(d, {}).setdefault(pol, []).append(dict(
                    qid=str(info.get("id") or r.get("example_id")),
                    acc=1.0 if float(r.get("reward") or 0) >= 0.5 else 0.0,
                    subs=float(m.get("rlm_sub_llm_calls") or 0),
                    cap=1.0 if str(r.get("stop_condition")) == "max_turns_reached" else 0.0,
                    fin=float(m.get("rlm_has_final_answer") or 0),
                    ctx_tok=len(ctx)//4,
                ))
    return fam

def bypol_stats(rows):
    byq = defaultdict(list)
    for r in rows: byq[r["qid"]].append(r["acc"])
    qmeans = [mean(v) for v in byq.values()]
    return dict(n=len(rows), nq=len(byq), acc=round(mean([r["acc"] for r in rows]), 4),
                se_cl=round(se(qmeans), 4), subs=round(mean([r["subs"] for r in rows]), 2),
                zero=round(mean([1.0 if r["subs"] == 0 else 0.0 for r in rows]), 3),
                cap=round(mean([r["cap"] for r in rows]), 3),
                fin=round(mean([r["fin"] for r in rows]), 3),
                acc_fin=round(mean([r["acc"] for r in rows if r["fin"] > 0]), 4) if any(r["fin"] > 0 for r in rows) else None)

def paired(a_rows, b_rows):
    A, B = defaultdict(list), defaultdict(list)
    for r in a_rows: A[r["qid"]].append(r["acc"])
    for r in b_rows: B[r["qid"]].append(r["acc"])
    common = set(A) & set(B)
    d = [mean(A[q]) - mean(B[q]) for q in common]
    return dict(nq=len(d), delta=round(mean(d), 4), se=round(se(d), 4),
                z=round(mean(d)/se(d), 2) if len(d) > 2 and se(d) > 0 else None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    fam = load(a.root)
    out = {}
    for d, pols in sorted(fam.items()):
        out[d] = {"per_policy": {p: bypol_stats(rows) for p, rows in pols.items()}}
        if "base" in pols:
            out[d]["paired_vs_base"] = {p: paired(pols[p], pols["base"])
                                        for p in pols if p != "base"}
        if "t2T_200" in pols and "t2T_120" in pols:
            out[d]["p200_vs_p120"] = paired(pols["t2T_200"], pols["t2T_120"])
            # strata: split questions by base-row context size
            strat = {}
            for name, lo, hi in [("short", 0, WINDOW_TOK), ("long", WINDOW_TOK, 10**12)]:
                qs = {r["qid"] for r in pols["base"] if lo <= r["ctx_tok"] < hi}
                sub200 = [r for r in pols["t2T_200"] if r["qid"] in qs]
                sub120 = [r for r in pols["t2T_120"] if r["qid"] in qs]
                strat[name] = dict(nq=len(qs), gap_200_vs_120=paired(sub200, sub120))
                for p in ("base", "authors"):
                    if p in pols:
                        strat[name][f"subs_{p}"] = round(
                            mean([r["subs"] for r in pols[p] if r["qid"] in qs]), 1)
            out[d]["strata"] = strat
    print(json.dumps(out, indent=1))
    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
