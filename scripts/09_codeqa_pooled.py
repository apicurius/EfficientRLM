#!/usr/bin/env python3
"""Pooled CodeQA statistics over the four independent runs.

The four runs (codeqa, codeqa_rep2, codeqa_rep3 in the 2026-07-12 offline
tree; the replication in the 2026-07-14 extension tree) are pooled per policy
with question-clustered SEs. codeqa_fenced is the mandate-delegation
instruction probe, NOT a repetition, and is excluded. Single-run accuracy
swings are +/-0.1, so per-run cells are never quoted as family values.

Adds accuracy-given-finalized and a paired @200-vs-@120 contrast on the
both-finalized subset: the test of whether the collapse deficit is a
mechanical artifact of cap-out scoring (a cap-out scores 0, below the 0.25
guessing floor).

Stdlib-only:
  python scripts/09_codeqa_pooled.py [--json OUT]
"""
import argparse, glob, json, math, os
from collections import defaultdict

POL = {"Qwen_Qwen3-30B-A3B-Instruct-2507": "base", "authors": "authors",
       "t2T_120": "t2T_120", "t2T_final": "t2T_200"}
RUNS = [
    ("full/codeqa", "outputs/offline_eval_full_20260712/codeqa"),
    ("full/rep2",   "outputs/offline_eval_full_20260712/codeqa_rep2"),
    ("full/rep3",   "outputs/offline_eval_full_20260712/codeqa_rep3"),
    ("ext/repl",    "outputs/offline_eval_ext_20260714/codeqa"),
]


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def cl(rows, key="acc"):
    byq = defaultdict(list)
    for r in rows:
        byq[r["qid"]].append(r[key])
    qm = [mean(v) for v in byq.values()]
    mu = mean(qm)
    se = (math.sqrt(sum((x - mu) ** 2 for x in qm) / (len(qm) - 1) / len(qm))
          if len(qm) > 1 else float("nan"))
    return mu, se, len(byq)


def load():
    pooled = defaultdict(list)
    for tag, p in RUNS:
        for f in glob.glob(f"{p}/*/evals/*/*/results.jsonl"):
            pol = POL.get(os.path.relpath(f, p).split(os.sep)[0])
            if not pol:
                continue
            for line in open(f):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                m = r.get("metrics") or {}
                info = r.get("info") or {}
                if isinstance(info, str):
                    try:
                        info = json.loads(info)
                    except Exception:
                        info = {}
                pooled[pol].append(dict(
                    run=tag, qid=str(info.get("id") or r.get("example_id")),
                    acc=1.0 if float(r.get("reward") or 0) >= 0.5 else 0.0,
                    subs=float(m.get("rlm_sub_llm_calls") or 0),
                    cap=1.0 if str(r.get("stop_condition")) == "max_turns_reached" else 0.0,
                    fin=float(m.get("rlm_has_final_answer") or 0)))
    return pooled


def paired(pooled, cond):
    a = {(r["run"], r["qid"]): r for r in pooled["t2T_200"]}
    b = {(r["run"], r["qid"]): r for r in pooled["t2T_120"]}
    byq = defaultdict(list)
    for k in set(a) & set(b):
        if cond(a[k], b[k]):
            byq[k[1]].append(a[k]["acc"] - b[k]["acc"])
    qd = [mean(v) for v in byq.values()]
    mu = mean(qd)
    se = math.sqrt(sum((x - mu) ** 2 for x in qd) / (len(qd) - 1) / len(qd))
    return dict(delta=round(mu, 4), se_cl=round(se, 4), z=round(mu / se, 2),
                nq=len(byq), pairs=sum(len(v) for v in byq.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    pooled = load()

    out = {"runs": [t for t, _ in RUNS], "per_policy": {}, "per_run_acc": {}}
    for tag, _ in RUNS:
        out["per_run_acc"][tag] = {
            pol: round(mean([r["acc"] for r in pooled[pol] if r["run"] == tag]), 3)
            for pol in POL.values()}
    print(f"{'policy':10s} {'n':>4} {'nq':>3} {'acc':>6} {'clSE':>6} {'subs':>6} "
          f"{'zero':>5} {'cap':>5} {'fin':>5} {'acc|fin':>8} {'clSE':>6}")
    for pol in ("base", "authors", "t2T_120", "t2T_200"):
        rows = pooled[pol]
        acc, se, nq = cl(rows)
        finr = [r for r in rows if r["fin"] > 0]
        af, afse, _ = cl(finr)
        cell = dict(n=len(rows), nq=nq, acc=round(acc, 4), se_cl=round(se, 4),
                    subs=round(mean([r["subs"] for r in rows]), 2),
                    zero=round(mean([1.0 if r["subs"] == 0 else 0 for r in rows]), 3),
                    cap=round(mean([r["cap"] for r in rows]), 3),
                    fin=round(mean([r["fin"] for r in rows]), 3),
                    acc_fin=round(af, 4), acc_fin_se_cl=round(afse, 4),
                    n_fin=len(finr))
        out["per_policy"][pol] = cell
        print(f"{pol:10s} {cell['n']:4d} {cell['nq']:3d} {cell['acc']:6.3f} {cell['se_cl']:6.3f} "
              f"{cell['subs']:6.2f} {cell['zero']:5.2f} {cell['cap']:5.2f} {cell['fin']:5.2f} "
              f"{cell['acc_fin']:8.3f} {cell['acc_fin_se_cl']:6.3f}")

    out["paired_200_vs_120"] = paired(pooled, lambda x, y: True)
    out["paired_200_vs_120_both_finalized"] = paired(
        pooled, lambda x, y: x["fin"] > 0 and y["fin"] > 0)
    for k in ("paired_200_vs_120", "paired_200_vs_120_both_finalized"):
        print(k, out[k])

    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
