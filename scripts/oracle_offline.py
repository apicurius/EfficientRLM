#!/usr/bin/env python3
"""Strict-oracle fingerprint on the OFFLINE pass rollouts (per policy).

Reuses oracle_rescore.py's frozen strict scorers and fingerprint machinery
(imported, not copied) and adapts only the record loading: vf-eval results
(completion list, top-level answer/metrics) instead of prime-rl train dumps
(nodes, task.answer). Terciles are computed within (suite, policy, rep) by the
adaptive-cost basis recomputed from iterations and sub-calls, then pooled.

Surfaces: trec (oolong strict scorer, offline/free) and bcplus (strict judge,
paid, stratified sample). This is the treatment-side EARLY reading on the
final policy; the contract cell (in-run rollouts, both arms) runs at the sweep.

Usage: oracle_offline.py OUT_DIR [--policies t2T_final,base,authors]
       [--limit N] [--offline] [--json OUT.json]
"""
import argparse, json, glob, math, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from oracle_rescore import (  # frozen strict scorers + stats + fingerprint
    oolong_strict_pass, BROWSECOMP_STRICT_JUDGE_PROMPT, _extract_exact_answer,
    _parse_bcp_judge_correct, _load_openai_key, _judge_call, fingerprint,
    stratified_sample, TERCILES, _ANSWER_ASSIGN, _resolve_literal,
)
import ast

POLMAP = {"Qwen_Qwen3-30B-A3B-Instruct-2507": "base", "authors": "authors",
          "t2T_120": "t2T_120", "t2T_final": "t2T_final", "t2C": "t2C"}
FAM = {"trec_coarse_131k": ("trec", 1), "trec_rep2": ("trec", 2),
       "bcplus_heldout": ("bcplus", 1), "bcplus_rep2": ("bcplus", 2)}


def extract_answer_offline(rec):
    """Last answer["content"] = <literal> across assistant completion messages."""
    rhs = []
    for m in rec.get("completion") or []:
        if (m or {}).get("role") != "assistant":
            continue
        c = m.get("content") or ""
        if isinstance(c, list):
            c = json.dumps(c)
        for mm in _ANSWER_ASSIGN.finditer(c):
            rhs.append(mm.group(1))
    if not rhs:
        return None, "no_assign"
    try:
        node = ast.parse("__x__ = " + rhs[-1].strip(), mode="exec").body[0].value
    except SyntaxError:
        return None, "syntaxerr"
    val = _resolve_literal(node)
    return (val, "ok") if val is not None else (None, "dynamic")


def cost_of(m):
    return float(m.get("rlm_iterations") or 0) + math.log1p(float(m.get("rlm_sub_llm_calls") or 0))


def is_valid_correct(rec):
    m = rec.get("metrics") or {}
    if float(rec.get("reward") or 0) < 0.5:
        return False
    if not float(m.get("rlm_has_final_answer") or 0):
        return False
    return str(rec.get("stop_condition")) == "has_final_answer"


def load(out_dir, policies):
    rows = {}  # (suite, pol) -> list of (rep, rec, answer_text, ext_status)
    ext = {}
    for f in glob.glob(f"{out_dir}/*/*/evals/*/*/results.jsonl"):
        parts = os.path.relpath(f, out_dir).split(os.sep)
        fam, poldir = parts[0], parts[1]
        if fam not in FAM:
            continue
        suite, rep = FAM[fam]
        pol = POLMAP.get(poldir, poldir)
        if pol not in policies:
            continue
        for line in open(f):
            rec = json.loads(line)
            if not is_valid_correct(rec):
                continue
            text, status = extract_answer_offline(rec)
            e = ext.setdefault((suite, pol), {"n": 0, "ok": 0, "dynamic": 0, "no_assign": 0, "syntaxerr": 0})
            e["n"] += 1
            e[status] += 1
            if status == "ok":
                rows.setdefault((suite, pol), []).append((rep, rec, text))
    return rows, ext


def terciles_for(rows_sp):
    """Within-rep tercile labels by cost, pooled."""
    labels = {}
    byrep = {}
    for i, (rep, rec, _) in enumerate(rows_sp):
        byrep.setdefault(rep, []).append(i)
    for _, idxs in byrep.items():
        srt = sorted(idxs, key=lambda i: (cost_of(rows_sp[i][1]["metrics"]), i))
        n = len(srt)
        for rank, i in enumerate(srt):
            labels[i] = TERCILES[min(2, (3 * rank) // n)]
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--policies", default="t2T_final,base,authors")
    ap.add_argument("--limit", type=int, default=45, help="paid judge calls PER policy")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    policies = args.policies.split(",")
    rows, ext = load(args.out_dir, policies)
    report = {"surface": "offline pass", "out_dir": args.out_dir, "policies": {}}

    for pol in policies:
        report["policies"][pol] = {}
        # ---- trec (offline, free) ----
        sp = rows.get(("trec", pol), [])
        es = ext.get(("trec", pol), {"n": 0, "ok": 0})
        labels = terciles_for(sp)
        scored = []
        for i, (rep, rec, text) in enumerate(sp):
            gold = str(rec.get("answer") or (rec.get("info") or {}).get("answer") or "")
            scored.append({"tercile": labels[i], "passed": oolong_strict_pass(text, gold),
                           "cost": cost_of(rec["metrics"]), "metrics": rec["metrics"]})
        fp = fingerprint(f"trec/{pol}", scored)
        fp["extraction"] = es
        fp["strict_rate"] = round(sum(s["passed"] for s in scored) / len(scored), 4) if scored else None
        report["policies"][pol]["trec"] = fp
        print(f"[trec/{pol}] valid-correct={es['n']} extracted={es.get('ok',0)} "
              f"strict_rate={fp['strict_rate']} gap={fp.get('gap_pp')}pp warn={fp['warn']}")

        # ---- bcplus (paid strict judge) ----
        if args.offline:
            continue
        sp = rows.get(("bcplus", pol), [])
        es = ext.get(("bcplus", pol), {"n": 0, "ok": 0})
        if not sp:
            continue
        labels = terciles_for(sp)
        steps_of = {i: sp[i][0] for i in range(len(sp))}
        ids = {i: str((sp[i][1].get("info") or {}).get("id") or i) for i in range(len(sp))}
        sel = stratified_sample(labels, steps_of, ids, args.limit)
        key, base_url = _load_openai_key()
        if not key:
            print("no API key; skipping bcplus")
            continue
        scored, errs = [], 0
        for i in sel:
            rep, rec, text = sp[i]
            info = rec.get("info") or {}
            query = str(info.get("root_prompt") or "")[:4000] or str(info.get("id"))
            gold = str(rec.get("answer") or info.get("answer") or "")
            prompt = BROWSECOMP_STRICT_JUDGE_PROMPT.format(
                query=query, expected=gold, predicted=_extract_exact_answer(text))
            try:
                passed = _parse_bcp_judge_correct(_judge_call(prompt, "openai/gpt-5-nano", key, base_url))
            except Exception as ex:
                errs += 1
                print(f"  judge error: {ex}", file=sys.stderr)
                continue
            scored.append({"tercile": labels[i], "passed": passed,
                           "cost": cost_of(rec["metrics"]), "metrics": rec["metrics"]})
        fp = fingerprint(f"bcplus/{pol}", scored)
        fp["extraction"] = es
        fp["judge_errors"] = errs
        fp["strict_rate"] = round(sum(s["passed"] for s in scored) / len(scored), 4) if scored else None
        report["policies"][pol]["bcplus"] = fp
        print(f"[bcplus/{pol}] valid-correct={es['n']} judged={len(scored)} "
              f"strict_rate={fp['strict_rate']} gap={fp.get('gap_pp')}pp warn={fp['warn']}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print("wrote", args.json)


if __name__ == "__main__":
    main()
