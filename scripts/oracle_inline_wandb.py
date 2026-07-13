#!/usr/bin/env python3
"""Inline (in-run) strict-oracle fingerprint from the recovered wandb eval
samples tables.

The treatment's disk rollout dumps were lost to a machine switch; the wandb
incremental eval tables hold 25 rollouts per (eval step, env) with the full
prompt, gold answer, transcript, and reward. This adapter reuses the frozen
strict scorers of oracle_rescore.py and re-derives what the lost metrics held:

- valid-correct: reward > 0 AND the transcript commits a final answer
  (answer["ready"] = True) — cap-outs and no-finals fail this.
- cost for terciles: repl-block count, i.e. the ITERATIONS-ONLY registered
  cost basis. Sub-call counts are not reconstructible from transcripts
  (llm_query_batched hides batch sizes), and tercile assignment needs only
  within-(env, step) ranks.

Usage: oracle_inline_wandb.py TABLE_DIR [--limit N] [--offline] [--json OUT]
"""
import argparse, json, glob, sys, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from oracle_rescore import (
    oolong_strict_pass, BROWSECOMP_STRICT_JUDGE_PROMPT, _extract_exact_answer,
    _parse_bcp_judge_correct, _load_openai_key, _judge_call, fingerprint,
    stratified_sample, TERCILES, _ANSWER_ASSIGN, _resolve_literal,
)
import ast

def extract_answer(comp):
    rhs = [m.group(1).splitlines()[0] for m in _ANSWER_ASSIGN.finditer(comp)]
    if not rhs:
        return None, "no_assign"
    try:
        node = ast.parse("__x__ = " + rhs[-1].strip(), mode="exec").body[0].value
    except SyntaxError:
        return None, "syntaxerr"
    val = _resolve_literal(node)
    return (val, "ok") if val is not None else (None, "dynamic")

READY = re.compile(r"""answer\s*\[\s*['"]ready['"]\s*\]\s*=\s*True""")

def load(table_dir):
    rows = []
    for f in glob.glob(f"{table_dir}/**/samples.table.json", recursive=True):
        t = json.load(open(f)); ci = {c: i for i, c in enumerate(t["columns"])}
        for r in t["data"]:
            task = r[ci["task"]]
            if isinstance(task, str):
                task = json.loads(task)
            rows.append(dict(step=r[ci["step"]], env=r[ci["env"]],
                             idx=task.get("idx"), gold=str(task.get("answer") or ""),
                             prompt=str(task.get("prompt") or ""),
                             comp=r[ci["completion"]], reward=float(r[ci["reward"]] or 0)))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table_dir")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rows = load(args.table_dir)
    print(f"loaded {len(rows)} eval rollouts from {args.table_dir}")
    report = {"surface": "in-run eval (wandb tables)", "envs": {}}

    for envkey, short in [("oolong-trec-coarse-eval", "oolong"), ("browsecomp-plus-eval", "browsecomp")]:
        sel, ext = [], {"n": 0, "ok": 0, "dynamic": 0, "no_assign": 0, "syntaxerr": 0}
        for r in rows:
            if r["env"] != envkey or r["reward"] <= 0 or not READY.search(r["comp"]):
                continue
            text, status = extract_answer(r["comp"])
            ext["n"] += 1; ext[status] += 1
            if status == "ok":
                sel.append((r, text))
        # terciles within (env, step) by the iterations-only basis (repl blocks)
        labels, bystep = {}, {}
        for i, (r, _) in enumerate(sel):
            bystep.setdefault(r["step"], []).append(i)
        for _, idxs in bystep.items():
            srt = sorted(idxs, key=lambda i: (sel[i][0]["comp"].count("```repl"), i))
            n = len(srt)
            for rank, i in enumerate(srt):
                labels[i] = TERCILES[min(2, (3 * rank) // n)]
        if short == "oolong":
            scored = [{"tercile": labels[i], "passed": oolong_strict_pass(text, r["gold"]),
                       "cost": r["comp"].count("```repl"), "metrics": {}}
                      for i, (r, text) in enumerate(sel)]
        else:
            if args.offline:
                report["envs"][short] = {"skipped": "offline", "extraction": ext}
                continue
            key, base_url = _load_openai_key()
            steps_of = {i: sel[i][0]["step"] for i in range(len(sel))}
            ids = {i: str(sel[i][0]["idx"]) for i in range(len(sel))}
            pick = stratified_sample(labels, steps_of, ids, args.limit)
            scored = []
            for i in pick:
                r, text = sel[i]
                q = r["prompt"].split("Question:")[-1].strip()[:3000]
                prompt = BROWSECOMP_STRICT_JUDGE_PROMPT.format(
                    query=q, expected=r["gold"], predicted=_extract_exact_answer(text))
                try:
                    passed = _parse_bcp_judge_correct(_judge_call(prompt, "openai/gpt-5-nano", key, base_url))
                except Exception as ex:
                    print("judge err:", ex, file=sys.stderr); continue
                scored.append({"tercile": labels[i], "passed": passed,
                               "cost": r["comp"].count("```repl"), "metrics": {}})
        fp = fingerprint(short, scored)
        fp["extraction"] = ext
        fp["strict_rate"] = round(sum(s["passed"] for s in scored) / len(scored), 4) if scored else None
        fp["cost_basis"] = "iterations-only (repl blocks; registered alternative basis)"
        report["envs"][short] = fp
        cells = fp["terciles"]
        print(f"[{short}] valid-correct={ext['n']} extracted={ext['ok']} scored={len(scored)} "
              f"strict={fp['strict_rate']} gap={fp.get('gap_pp')}pp warn={fp['warn']} | " +
              " ".join(f"{t}:{cells[t]['pass']}/{cells[t]['n']}" for t in TERCILES if cells[t]["n"]))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print("wrote", args.json)

if __name__ == "__main__":
    main()
