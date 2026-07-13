#!/usr/bin/env python3
"""Thesis-grade statistics over the offline eval results tree (stdlib only).

Reads $OUT (offline_eval_full_*) laid out as
  <family_dir>/<policy>/evals/<env>--<policy>/<runhash>/results.jsonl
and emits, under $OUT/thesis_stats/:

  A_accuracy.csv        pooled accuracy per (suite,policy): naive SE, question-
                        clustered SE, per-rep means, metric sensitivity
  B_contrasts.csv       paired-by-question accuracy contrasts vs base/authors/120
  C_paired_cost.csv     both-correct paired cost (5 channels) + levels + Wilcoxon
                        + cluster-bootstrap CI + BH q-values across the table
  D_ops_tail.csv        ops/tail cells per (suite,policy): means, p95, cap-out,
                        no-final, fatal, truncated shares
  E_codeqa_collapse.csv per-rep delegation detail for the collapse figure
  F_sensitivity.csv     cost contrasts conditioned on BASE-correct questions only
  G_per_rep.csv         per-(family_dir,policy) accuracy — anchor reconciliation
  H_provenance.txt      metadata.json digest per run (env args, judge, dates)
  SUMMARY.md            everything readable, with the report-reproduction check

Correctness convention: correct := reward >= 0.5 (binary rewards in practice).
Metric sensitivity uses _score >= 0.5 and gated_reward >= 0.5 alongside.
Pairing/cluster key: info.id when present, else example_id (per suite).
"""
import json, glob, os, sys, math, random, csv
from collections import defaultdict

random.seed(20260713)
B_BOOT = 4000

HERE = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) > 1:
    OUT = sys.argv[1]
else:
    cands = sorted(glob.glob(os.path.join(HERE, "..", "..", "outputs", "offline_eval_full_*")))
    cands = [c for c in cands if os.path.isdir(c)]
    assert cands, "no offline_eval_full_* dir found"
    OUT = cands[-1]
OUT = os.path.abspath(OUT)
ST = os.path.join(OUT, "thesis_stats")
os.makedirs(ST, exist_ok=True)

# family_dir -> (suite, rep_index)
FAMILY = {
    "trec_coarse_131k": ("trec", 1), "trec_rep2": ("trec", 2),
    "oolong_pairs_32k": ("pairs", 1), "oolong_pairs_32k_rep2": ("pairs", 2),
    "pairs_rep3": ("pairs", 3), "pairs_rep4": ("pairs", 4),
    "codeqa": ("codeqa", 1), "codeqa_rep2": ("codeqa", 2), "codeqa_rep3": ("codeqa", 3),
    "bcplus_heldout": ("bcplus", 1), "bcplus_rep2": ("bcplus", 2),
    "codeqa_fenced": ("codeqa_fenced", 1), "trec_counting": ("trec_counting", 1),
}
MAIN_SUITES = ["trec", "pairs", "codeqa", "bcplus"]
POL_ALIAS = {
    "Qwen_Qwen3-30B-A3B-Instruct-2507": "base",
    "authors": "authors", "t2T_120": "t2T_120", "t2T_final": "t2T_final",
    "t2C": "t2C", "t2C_120": "t2C_120",
}

def mean(v): return sum(v) / len(v) if v else float("nan")
def sd(v):
    if len(v) < 2: return float("nan")
    m = mean(v); return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
def se(v): return sd(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")
def pctl(v, p):
    if not v: return float("nan")
    w = sorted(v); k = (len(w) - 1) * p
    f, c = int(math.floor(k)), int(math.ceil(k))
    return w[f] if f == c else w[f] + (w[c] - w[f]) * (k - f)

def cluster_se(pairs):
    """pairs: list of (cluster_key, value). SE of the grand mean with
    clustering: SE of the mean of per-cluster means."""
    by = defaultdict(list)
    for k, v in pairs: by[k].append(v)
    cm = [mean(v) for v in by.values()]
    return mean(cm), se(cm), len(cm)

def boot_ci(vals, keys=None, b=B_BOOT):
    """Cluster bootstrap (resample clusters) 95% CI of the mean."""
    if keys is None: keys = list(range(len(vals)))
    by = defaultdict(list)
    for k, v in zip(keys, vals): by[k].append(v)
    clusters = list(by.values())
    if len(clusters) < 2: return (float("nan"), float("nan"))
    ms = []
    for _ in range(b):
        samp = [clusters[random.randrange(len(clusters))] for _ in clusters]
        flat = [x for c in samp for x in c]
        ms.append(mean(flat))
    ms.sort()
    return (ms[int(0.025 * b)], ms[int(0.975 * b) - 1])

def wilcoxon(diffs):
    """Signed-rank, normal approximation with tie/zero handling. Two-sided p."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n < 6: return float("nan")
    ranked = sorted((abs(x), i) for i, x in enumerate(d))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ranked[j + 1][0] == ranked[i][0]: j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1): ranks[ranked[k][1]] = r
        i = j + 1
    wp = sum(r for r, x in zip(ranks, d) if x > 0)
    mu = n * (n + 1) / 4
    sig = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sig == 0: return float("nan")
    z = (wp - mu) / sig
    return math.erfc(abs(z) / math.sqrt(2))

def bh(pvals):
    """Benjamini-Hochberg q-values (nan-safe)."""
    idx = [i for i, p in enumerate(pvals) if not math.isnan(p)]
    m = len(idx)
    q = [float("nan")] * len(pvals)
    for rank, i in enumerate(sorted(idx, key=lambda i: pvals[i]), 1):
        q[i] = pvals[i] * m / rank
    # enforce monotonicity from the largest p downward
    order = sorted(idx, key=lambda i: pvals[i], reverse=True)
    best = 1.0
    for i in order:
        best = min(best, q[i]); q[i] = min(best, 1.0)
    return q

# ---------------- load ----------------
rows = []          # each rollout: dict
prov = []
for f in glob.glob(f"{OUT}/*/*/evals/*/*/results.jsonl"):
    rel = os.path.relpath(f, OUT).split(os.sep)
    fam_dir, pol_dir = rel[0], rel[1]
    if fam_dir not in FAMILY: continue
    suite, rep = FAMILY[fam_dir]
    pol = POL_ALIAS.get(pol_dir, pol_dir)
    md = os.path.join(os.path.dirname(f), "metadata.json")
    if os.path.exists(md):
        try:
            m = json.load(open(md))
            prov.append((fam_dir, pol, os.path.dirname(f).split(os.sep)[-1], m))
        except Exception: pass
    for line in open(f):
        try: r = json.loads(line)
        except Exception: continue
        if r.get("reward") is None: continue
        met = r.get("metrics") or {}
        tu = r.get("token_usage") or {}
        tm = r.get("timing") or {}
        info = r.get("info") or {}
        qid = info.get("id", r.get("example_id"))
        it = float(met.get("rlm_iterations", float("nan")))
        sc = float(met.get("rlm_sub_llm_calls", float("nan")))
        rows.append(dict(
            suite=suite, rep=rep, pol=pol, qid=qid,
            rew=float(r["reward"]),
            score=float(met.get("_score", float("nan"))),
            gated=float(met.get("gated_reward", float("nan"))),
            iters=it, subs=sc,
            cost=(it + math.log1p(sc)) if not (math.isnan(it) or math.isnan(sc)) else float("nan"),
            subtok=float(met.get("rlm_sub_llm_tokens", float("nan"))),
            outtok=float(tu.get("output_tokens", float("nan"))),
            time=float(tm.get("total", float("nan"))),
            trunc=bool(r.get("is_truncated", False)),
            stop=str(r.get("stop_condition", "")),
            hasfin=float(met.get("rlm_has_final_answer", float("nan"))),
        ))
print(f"loaded {len(rows)} rollouts from {OUT}")
stops = sorted({r["stop"] for r in rows})
print("stop_conditions seen:", stops)

def correct(r): return r["rew"] >= 0.5
# pairs rewards are continuous (pair-match fraction); its canonical accuracy is
# the mean reward (matches the offline report). Other suites are binary.
CONT_SUITES = {"pairs"}
def acc_val(r): return r["rew"] if r["suite"] in CONT_SUITES else (1.0 if correct(r) else 0.0)
CAP_STOPS = {s for s in stops if "max" in s or "iteration" in s or "turn" in s}

def cell(suite, pol):
    return [r for r in rows if r["suite"] == suite and r["pol"] == pol]

POLICIES = sorted({r["pol"] for r in rows},
                  key=lambda p: ["base", "t2T_120", "t2T_final", "authors", "t2C", "t2C_120"].index(p)
                  if p in ["base", "t2T_120", "t2T_final", "authors", "t2C", "t2C_120"] else 99)
SUITES = [s for s in MAIN_SUITES if any(r["suite"] == s for r in rows)]
ABL = [s for s in ("codeqa_fenced", "trec_counting") if any(r["suite"] == s for r in rows)]
S = open(os.path.join(ST, "SUMMARY.md"), "w")
def W(*a): print(*a, file=S);

W(f"# Thesis stats over {os.path.basename(OUT)}\n")
W(f"Rollouts: {len(rows)}; policies: {POLICIES}; suites: {SUITES + ABL}")
W(f"Correct := reward >= 0.5. Cluster/pairing key: info.id. Bootstrap B={B_BOOT}.\n")

# ---------------- A: accuracy ----------------
W("## A. Pooled accuracy (naive SE vs question-clustered SE)\n")
W("| suite | policy | n | nq | acc | SE_naive | SE_clustered | per-rep | acc(_score) | acc(gated) |")
W("|---|---|---|---|---|---|---|---|---|---|")
with open(os.path.join(ST, "A_accuracy.csv"), "w", newline="") as fh:
    cw = csv.writer(fh)
    cw.writerow(["suite","policy","n","n_questions","acc","se_naive","se_clustered",
                 "per_rep_means","acc_score_metric","acc_gated_metric"])
    for suite in SUITES + ABL:
        for pol in POLICIES:
            cc = cell(suite, pol)
            if not cc: continue
            accs = [acc_val(r) for r in cc]
            n = len(accs); a = mean(accs)
            senaive = se(accs) if suite in CONT_SUITES else (math.sqrt(a * (1 - a) / n) if 0 < a < 1 else 0.0)
            _, secl, nq = cluster_se([(r["qid"], acc_val(r)) for r in cc])
            reps = sorted({r["rep"] for r in cc})
            prm = "/".join(f"{mean([acc_val(r) for r in cc if r['rep']==k]):.3f}" for k in reps)
            a_sc = mean([1.0 if r["score"] >= 0.5 else 0.0 for r in cc if not math.isnan(r["score"])])
            a_gt = mean([1.0 if r["gated"] >= 0.5 else 0.0 for r in cc if not math.isnan(r["gated"])])
            cw.writerow([suite,pol,n,nq,f"{a:.4f}",f"{senaive:.4f}",f"{secl:.4f}",prm,f"{a_sc:.4f}",f"{a_gt:.4f}"])
            W(f"| {suite} | {pol} | {n} | {nq} | {a:.3f} | {senaive:.3f} | {secl:.3f} | {prm} | {a_sc:.3f} | {a_gt:.3f} |")
W("")

# ---------------- B: paired-by-question accuracy contrasts ----------------
W("## B. Accuracy contrasts, paired by question (per-question rep-mean diff)\n")
W("| suite | contrast | nq | delta | SE_cl | 95% CI (boot) | z |")
W("|---|---|---|---|---|---|---|")
CONTRASTS = [("t2T_final","base"),("t2T_final","authors"),("t2T_final","t2T_120"),
             ("t2T_120","base"),("t2T_120","authors")]
CONTRASTS += [(a,b) for (a,b) in [("t2T_final","t2C"),("t2C","base")] if any(r["pol"]==("t2C") for r in rows)]
with open(os.path.join(ST, "B_contrasts.csv"), "w", newline="") as fh:
    cw = csv.writer(fh); cw.writerow(["suite","polA","polB","n_questions","delta","se_cl","ci_lo","ci_hi","z"])
    for suite in SUITES + ABL:
        for pa, pb in CONTRASTS:
            A, Bv = defaultdict(list), defaultdict(list)
            for r in cell(suite, pa): A[r["qid"]].append(acc_val(r))
            for r in cell(suite, pb): Bv[r["qid"]].append(acc_val(r))
            common = sorted(set(A) & set(Bv), key=str)
            if len(common) < 5: continue
            diffs = [mean(A[q]) - mean(Bv[q]) for q in common]
            d = mean(diffs); s_ = se(diffs)
            lo, hi = boot_ci(diffs)
            z = d / s_ if s_ and not math.isnan(s_) and s_ > 0 else float("nan")
            cw.writerow([suite,pa,pb,len(common),f"{d:.4f}",f"{s_:.4f}",f"{lo:.4f}",f"{hi:.4f}",f"{z:.2f}"])
            W(f"| {suite} | {pa}−{pb} | {len(common)} | {d:+.3f} | {s_:.3f} | [{lo:+.3f},{hi:+.3f}] | {z:+.2f} |")
W("")

# ---------------- C: both-correct paired cost ----------------
W("## C. Both-correct paired cost (t2T_final − comparator), per-question rep-means\n")
W("Channels: cost=it+log1p(sc), iters, subcalls, subtokens, time. Levels shown for pct check.\n")
CH = [("cost","cost"),("iters","iters"),("subs","subcalls"),("subtok","subtokens_k",1e-3),("time","time_s")]
prows = []
for comp in ["base","authors"] + (["t2C"] if any(r["pol"]=="t2C" for r in rows) else []):
    for suite in SUITES:
        A, Bv = defaultdict(list), defaultdict(list)
        for r in cell(suite,"t2T_final"):
            if correct(r): A[r["qid"]].append(r)
        for r in cell(suite,comp):
            if correct(r): Bv[r["qid"]].append(r)
        both = sorted(set(A)&set(Bv), key=str)
        n_t = len({r['qid'] for r in cell(suite,'t2T_final')})
        cov = len(both)/n_t if n_t else float('nan')
        if len(both) < 4: continue
        rec = dict(comp=comp, suite=suite, n_both=len(both), coverage=f"{cov:.2f}")
        for ch in CH:
            key, name = ch[0], ch[1]; scale = ch[2] if len(ch)>2 else 1.0
            dif = []
            for q in both:
                ta = mean([x[key] for x in A[q] if not math.isnan(x[key])])
                tb = mean([x[key] for x in Bv[q] if not math.isnan(x[key])])
                if math.isnan(ta) or math.isnan(tb): continue
                dif.append((ta-tb)*scale)
            if len(dif) < 4: continue
            lvl_t = mean([mean([x[key] for x in A[q]]) for q in both])*scale
            lvl_c = mean([mean([x[key] for x in Bv[q]]) for q in both])*scale
            lo,hi = boot_ci(dif)
            p = wilcoxon(dif)
            rec[name] = dict(d=mean(dif), se=se(dif), lo=lo, hi=hi, p=p,
                             lvl_t=lvl_t, lvl_c=lvl_c,
                             pct=100*mean(dif)/lvl_c if lvl_c else float("nan"))
        prows.append(rec)
pv = [r[nm]["p"] for r in prows for _,nm,*_ in [c if len(c)>2 else (c[0],c[1]) for c in CH] if nm in r]
# collect p-values in stable order for BH
plist, ptr = [], []
for r in prows:
    for c in CH:
        nm = c[1]
        if nm in r: plist.append(r[nm]["p"]); ptr.append((r,nm))
qs = bh(plist)
for (r,nm),q in zip(ptr,qs): r[nm]["q"] = q
with open(os.path.join(ST,"C_paired_cost.csv"),"w",newline="") as fh:
    cw = csv.writer(fh)
    cw.writerow(["comparator","suite","n_both","coverage","channel","delta","se","ci_lo","ci_hi",
                 "wilcoxon_p","bh_q","level_t2T","level_comp","pct_change"])
    for r in prows:
        W(f"**vs {r['comp']} / {r['suite']}** (n_both={r['n_both']}, coverage={r['coverage']}):")
        for c in CH:
            nm = c[1]
            if nm not in r: continue
            x = r[nm]
            cw.writerow([r["comp"],r["suite"],r["n_both"],r["coverage"],nm,
                         f"{x['d']:.3f}",f"{x['se']:.3f}",f"{x['lo']:.3f}",f"{x['hi']:.3f}",
                         f"{x['p']:.4g}" if not math.isnan(x['p']) else "",
                         f"{x['q']:.4g}" if not math.isnan(x.get('q',float('nan'))) else "",
                         f"{x['lvl_t']:.3f}",f"{x['lvl_c']:.3f}",f"{x['pct']:.1f}"])
            W(f"  - {nm}: Δ={x['d']:+.2f}±{x['se']:.2f} CI[{x['lo']:+.2f},{x['hi']:+.2f}] "
              f"p={x['p']:.3g} q={x.get('q',float('nan')):.3g} | levels {x['lvl_t']:.1f} vs {x['lvl_c']:.1f} ({x['pct']:+.0f}%)")
W("")

# ---------------- D: ops / tail ----------------
W("## D. Ops & tail cells per (suite, policy)\n")
W("| suite | policy | iters | subcalls | cost | p95 cost | cap-out | no-final | fatal | truncated |")
W("|---|---|---|---|---|---|---|---|---|---|")
with open(os.path.join(ST,"D_ops_tail.csv"),"w",newline="") as fh:
    cw = csv.writer(fh)
    cw.writerow(["suite","policy","mean_iters","mean_subcalls","mean_cost","p95_cost",
                 "capout_rate","nofinal_rate","fatal_rate","truncated_share","n"])
    for suite in SUITES + ABL:
        for pol in POLICIES:
            cc = cell(suite,pol)
            if not cc: continue
            iters=[r["iters"] for r in cc if not math.isnan(r["iters"])]
            subs=[r["subs"] for r in cc if not math.isnan(r["subs"])]
            cost=[r["cost"] for r in cc if not math.isnan(r["cost"])]
            cap=mean([1.0 if (r["stop"] in CAP_STOPS) else 0.0 for r in cc])
            nof=mean([1.0 if r["hasfin"]==0.0 else 0.0 for r in cc if not math.isnan(r["hasfin"])])
            fat=mean([1.0 if (r["stop"] in CAP_STOPS or r["hasfin"]==0.0) else 0.0 for r in cc])
            tr=mean([1.0 if r["trunc"] else 0.0 for r in cc])
            cw.writerow([suite,pol,f"{mean(iters):.2f}",f"{mean(subs):.2f}",f"{mean(cost):.2f}",
                         f"{pctl(cost,0.95):.2f}",f"{cap:.3f}",f"{nof:.3f}",f"{fat:.3f}",f"{tr:.3f}",len(cc)])
            W(f"| {suite} | {pol} | {mean(iters):.1f} | {mean(subs):.1f} | {mean(cost):.1f} | "
              f"{pctl(cost,0.95):.1f} | {cap:.2f} | {nof:.2f} | {fat:.2f} | {tr:.2f} |")
W("")

# ---------------- E: codeqa collapse per rep ----------------
W("## E. codeqa delegation detail per rep (collapse figure data)\n")
W("| policy | rep | acc | mean subs | median subs | zero-sub share | cap-out | n |")
W("|---|---|---|---|---|---|---|---|")
with open(os.path.join(ST,"E_codeqa_collapse.csv"),"w",newline="") as fh:
    cw = csv.writer(fh)
    cw.writerow(["suite","policy","rep","acc","mean_subcalls","median_subcalls","zero_subcall_share","capout_rate","n"])
    for suite in ["codeqa","codeqa_fenced"]:
        for pol in POLICIES:
            for rep in sorted({r["rep"] for r in cell(suite,pol)}):
                cc=[r for r in cell(suite,pol) if r["rep"]==rep]
                if not cc: continue
                subs=[r["subs"] for r in cc if not math.isnan(r["subs"])]
                acc=mean([1.0 if correct(r) else 0.0 for r in cc])
                z0=mean([1.0 if s==0 else 0.0 for s in subs])
                cap=mean([1.0 if r["stop"] in CAP_STOPS else 0.0 for r in cc])
                cw.writerow([suite,pol,rep,f"{acc:.3f}",f"{mean(subs):.2f}",f"{pctl(subs,0.5):.1f}",
                             f"{z0:.3f}",f"{cap:.3f}",len(cc)])
                W(f"| {suite}/{pol} | {rep} | {acc:.3f} | {mean(subs):.2f} | {pctl(subs,0.5):.1f} | {z0:.2f} | {cap:.2f} | {len(cc)} |")
W("")

# ---------------- F: base-correct-conditioned sensitivity ----------------
W("## F. Sensitivity: cost contrasts on BASE-correct questions only (treatment-independent conditioning)\n")
W("| suite | channel | n_q | Δ(t2T_final − base) | SE | 95% CI |")
W("|---|---|---|---|---|---|")
with open(os.path.join(ST,"F_sensitivity.csv"),"w",newline="") as fh:
    cw = csv.writer(fh); cw.writerow(["suite","channel","n_q","delta","se","ci_lo","ci_hi"])
    for suite in SUITES:
        basecorrect = {q for q,v in
                       [(q, any(correct(r) for r in g)) for q,g in
                        [(q,[r for r in cell(suite,"base") if r["qid"]==q])
                         for q in {r["qid"] for r in cell(suite,"base")}]] if v}
        A = defaultdict(list); Bv = defaultdict(list)
        for r in cell(suite,"t2T_final"):
            if r["qid"] in basecorrect: A[r["qid"]].append(r)
        for r in cell(suite,"base"):
            if r["qid"] in basecorrect and correct(r): Bv[r["qid"]].append(r)
        common = sorted(set(A)&set(Bv), key=str)
        if len(common) < 4: continue
        for c in CH:
            key, nm = c[0], c[1]; scale = c[2] if len(c)>2 else 1.0
            dif=[]
            for q in common:
                ta=mean([x[key] for x in A[q] if not math.isnan(x[key])])
                tb=mean([x[key] for x in Bv[q] if not math.isnan(x[key])])
                if math.isnan(ta) or math.isnan(tb): continue
                dif.append((ta-tb)*scale)
            if len(dif)<4: continue
            lo,hi=boot_ci(dif)
            cw.writerow([suite,nm,len(dif),f"{mean(dif):.3f}",f"{se(dif):.3f}",f"{lo:.3f}",f"{hi:.3f}"])
            W(f"| {suite} | {nm} | {len(dif)} | {mean(dif):+.2f} | {se(dif):.2f} | [{lo:+.2f},{hi:+.2f}] |")
W("\nNote: unlike C, treatment rollouts here are NOT conditioned on treatment correctness —")
W("the conditioning set depends only on the base policy, killing the pro-treatment-selection story if signs agree with C.\n")

# ---------------- G: per-(family_dir,policy) accuracy ----------------
with open(os.path.join(ST,"G_per_rep.csv"),"w",newline="") as fh:
    cw = csv.writer(fh); cw.writerow(["family_dir","suite","rep","policy","n","acc"])
    W("## G. Raw per-run accuracy (anchor reconciliation)\n")
    W("| family_dir | policy | n | acc |")
    W("|---|---|---|---|")
    for fam_dir,(suite,rep) in FAMILY.items():
        for pol in POLICIES:
            cc=[r for r in rows if r["suite"]==suite and r["rep"]==rep and r["pol"]==pol]
            if not cc: continue
            a=mean([1.0 if correct(r) else 0.0 for r in cc])
            cw.writerow([fam_dir,suite,rep,pol,len(cc),f"{a:.4f}"])
            W(f"| {fam_dir} | {pol} | {len(cc)} | {a:.3f} |")
W("")

# ---------------- H: provenance ----------------
with open(os.path.join(ST,"H_provenance.txt"),"w") as fh:
    for fam_dir,pol,run,m in sorted(prov, key=lambda x:(x[0],x[1])):
        keep = {k:v for k,v in m.items() if k not in ("dataset","results")}
        fh.write(f"{fam_dir}/{pol}/{run}: {json.dumps(keep)[:800]}\n")
W(f"Provenance digest: thesis_stats/H_provenance.txt ({len(prov)} runs)\n")
S.close()
print(f"wrote {ST}/SUMMARY.md and CSVs")
