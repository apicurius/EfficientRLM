"""CORRECTED offline re-derivation of the adaptive advantage operator on real
rollout groups — the v2 instrument that fixes every defect the adversarial
review (docs/PHASE0_REVIEW_MEMO_20260717.md ADDENDUM, objections 1,3,4,5,7,8,9,11)
found in scripts/11_mitigation_replay.py.

This file is self-contained and read-only w.r.t. the training stack: it ports
the live operator semantics from rlm/training/src/rlm_train/adaptive_group.py
BIT-EXACTLY (verified: recompute of the control's logged beta=0 advantage
matches to 1e-9 on all rollouts) and never imports or mutates it.

What is fixed vs v1 (11_mitigation_replay.py):
  * obj 11: validity gate is now bit-exact — fatal includes BOTH
    {"max_turns","max_turns_reached"} (v1 matched only the latter) AND the
    dead-worker error classes AND the has_final = (metric>0 OR
    stop=="has_final_answer") alternative.
  * obj 3 : a FROZEN dataset manifest (file list + per-file sha256 + group
    keys) is written on first run and VERIFIED on re-runs; the measurement
    iterates the manifest, never a time-of-execution glob.
  * group keying: (step, env, task-idx, prompt-hash) — not bare idx.
  * obj 5 : reports dA/d(iterations) alongside dA/d(subs).
  * obj 7 : tie-INCLUSIVE zero-sub top-advantage incidence (v1's unique-top
    lowsub_wins could not see tied corners).
  * obj 8 : firing-weighted TOTAL dose on the shaping INCREMENT (A_candidate
    minus the beta=0 advantage), reported per-fired-group AND per-total-group.
  * obj 9 : k-valid degeneracy census + how often tail_only collapses to
    symmetric at k-valid=2.
  * obj 4 : 1000-group-resample bootstrap CI on every headline number.
  * obj 1 : the hinge/mean-centering redistribution to below-budget and
    zero-sub siblings is QUANTIFIED, not claimed away.

Candidates: V0 (today), V3 (iterations-only sym), V4 (iterations-only +
tail_only 0.5), HINGE-DEADBAND at budgets B in {3,5,8}, and UNIQUE-SUBS as an
erroring stub (no per-sub-call prompt telemetry in the dumps).

Usage:
  python scripts/12_mitigation_rederive.py \
      --root outputs/qwen3-30b-t2-control/run_default/rollouts \
      --step-lo 1 --step-hi 99 --tag t2control_1_99 \
      --label "pre-collapse-regime validation - NOT the decision dataset"
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict

# --------------------------------------------------------------------------
# Operator semantics ported BIT-EXACTLY from
# rlm/training/src/rlm_train/adaptive_group.py (do not drift).
# --------------------------------------------------------------------------
_FATAL_STOPS = frozenset({"max_turns", "max_turns_reached"})
_DEAD_WORKER = ("Connection lost", "worker closed stdout", "SIGKILLed")


def _metric(m: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(m.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _stop(r: dict) -> str:
    return str(r.get("stop_condition", "") or "")


def _has_final(r: dict, m: dict) -> bool:
    return _metric(m, "rlm_has_final_answer") > 0.0 or _stop(r) == "has_final_answer"


def _correct(r: dict, m: dict) -> float:
    if "gated_reward" in m:
        return 1.0 if _metric(m, "gated_reward") > 0.0 else 0.0
    # operator reads trace.reward; the dump stores it under rewards.reward / reward
    rew = r.get("reward")
    if rew is None:
        rew = (r.get("rewards") or {}).get("reward")
    try:
        return 1.0 if float(rew or 0.0) > 0.0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def _fatal(r: dict, m: dict) -> bool:
    err = " ".join(str(e) for e in (r.get("errors") or ()))
    return any(x in err for x in _DEAD_WORKER) or (not _has_final(r, m)) or (_stop(r) in _FATAL_STOPS)


def _beta(solve_rate: float, *, beta_max: float, solve_floor: float, gamma: float) -> float:
    if solve_rate <= solve_floor:
        return 0.0
    ramp = (solve_rate - solve_floor) / max(1e-12, 1.0 - solve_floor)
    return beta_max * (ramp ** gamma)


# --------------------------------------------------------------------------
# Cost bases (scaffold-action costs, never tokens). The first two mirror the
# operator's registry; hinge is the reviewer's deadband candidate.
# --------------------------------------------------------------------------
def cost_iterations(m: dict) -> float:
    return _metric(m, "rlm_iterations")


def cost_iterations_log_subcalls(m: dict) -> float:
    return _metric(m, "rlm_iterations") + math.log1p(_metric(m, "rlm_sub_llm_calls"))


def cost_hinge(m: dict, *, budget: float, lam_h: float) -> float:
    """Deadband: iterations + lam_h * max(0, subs - budget). No term rewards
    low subs; the productive band (subs <= budget) is unpriced beyond turns.
    (Original HINGE_B* cells; kept as the raw-linear-hinge reference — Codex ranks
    it last for corner reward + tail-scale collapse.)"""
    subs = _metric(m, "rlm_sub_llm_calls")
    return _metric(m, "rlm_iterations") + lam_h * max(0.0, subs - budget)


def _hinge_log(m: dict, B: float) -> float:
    """H_B = log(1 + (S - B)_+) — the calibration penalty magnitude (no lambda)."""
    return math.log1p(max(0.0, _metric(m, "rlm_sub_llm_calls") - B))


def cost_log_hinge(m: dict, *, budget: float, lam: float) -> float:
    """C_LH = I + lambda_B * log(1 + (S - B)_+).  Span-calibrated log penalty on
    excess volume; compresses the 4237-scale tail the raw hinge could not."""
    return _metric(m, "rlm_iterations") + lam * _hinge_log(m, budget)


def cost_norm_hinge(m: dict, *, budget: float) -> float:
    """C_NH = I + (S - B)_+ / B  (diagnostic; Codex notes /B under-compresses the
    tail so it stays close to volume pricing after min-max normalization)."""
    return _metric(m, "rlm_iterations") + max(0.0, _metric(m, "rlm_sub_llm_calls") - budget) / budget


# --------------------------------------------------------------------------
# Candidate registry. Each candidate maps a metrics dict -> cost, and carries
# a tail_only threshold (None = symmetric). UNIQUE-SUBS is a hard stub.
# --------------------------------------------------------------------------
LAM_H = 1.0     # raw-hinge slope (reference cell only)
BUDGETS = (3, 5, 8)
LAM_MULTS = (0.5, 1.0, 2.0)


def calibrate_lambdas(parsed, budgets=BUDGETS):
    """lambda_B* = median_g span_g(I) / median_g span_g(H_B), each median taken over
    groups with >=2 valid members and a POSITIVE span for that component (penalty-active
    / iteration-varying groups). Computed once on the frozen 1-99 set and FROZEN for the
    120-200 decision pass (loaded via --lambda-from). Returns dict B -> dict(lambda_star,
    med_span_I, med_span_H, n_I, n_H)."""
    def med(xs):
        xs = sorted(xs)
        n = len(xs)
        if n == 0:
            return float("nan")
        return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    # iteration spans (component-independent)
    spanI = []
    for p in parsed:
        vi = [i for i, v in enumerate(p["valid"]) if v]
        if len(vi) < 2:
            continue
        Iv = [p["iters"][i] for i in vi]
        s = max(Iv) - min(Iv)
        if s > 0:
            spanI.append(s)
    med_I = med(spanI)
    out = {}
    for B in budgets:
        spanH = []
        for p in parsed:
            vi = [i for i, v in enumerate(p["valid"]) if v]
            if len(vi) < 2:
                continue
            Hv = [math.log1p(max(0.0, p["subs"][i] - B)) for i in vi]
            s = max(Hv) - min(Hv)
            if s > 0:
                spanH.append(s)
        med_H = med(spanH)
        lam = (med_I / med_H) if (med_H and med_H == med_H and med_H > 0) else float("nan")
        out[B] = dict(lambda_star=lam, med_span_I=med_I, med_span_H=med_H,
                      n_I=len(spanI), n_H=len(spanH))
    return out


def build_candidates(lambdas):
    """References (untransformed operator semantics) + new zero-neutralized cells.
    Per Codex: apply symmetric shaping + zero-neutralization to every NEW cell only;
    keep V0/V3/V4 as untransformed references (V4 = reconciliation cell)."""
    cands = {}
    # --- references: original operator semantics, NO zero-neutralization ---
    cands["V0_today"] = dict(cost=cost_iterations_log_subcalls, tail_only=None, zn=False,
                             role="ref", desc="I+log1p(S), symmetric (today's live basis)")
    cands["V3_iters_sym"] = dict(cost=cost_iterations, tail_only=None, zn=False,
                                 role="ref", desc="iterations-only, symmetric (best measured existing)")
    cands["V4_iters_tail"] = dict(cost=cost_iterations, tail_only=0.5, zn=False,
                                  role="ref-recon", desc="iterations-only + tail_only 0.5 (reconciliation cell)")
    # raw-linear hinges retained as references (Codex ranks last; kept for continuity)
    for B in BUDGETS:
        cands[f"HINGE_B{B}"] = dict(cost=(lambda m, B=B: cost_hinge(m, budget=B, lam_h=LAM_H)),
                                    tail_only=None, zn=False, role="ref", budget=B,
                                    desc=f"raw linear hinge I+max(0,S-{B}), symmetric (reference)")
    # --- new computable cells: symmetric + zero-neutralized ---
    for B in BUDGETS:
        lam_star = lambdas[B]["lambda_star"]
        for mult in LAM_MULTS:
            lam = lam_star * mult
            cands[f"C_LH_B{B}_x{mult:g}"] = dict(
                cost=(lambda m, B=B, lam=lam: cost_log_hinge(m, budget=B, lam=lam)),
                tail_only=None, zn=True, role="new", budget=B, lam=lam, mult=mult,
                desc=f"I+{lam:.4g}*log(1+(S-{B})+), symmetric, zero-neutralized [x{mult:g} lambda*]")
    for B in BUDGETS:
        cands[f"C_NH_B{B}"] = dict(cost=(lambda m, B=B: cost_norm_hinge(m, budget=B)),
                                   tail_only=None, zn=True, role="new-diag", budget=B,
                                   desc=f"I+(S-{B})+/{B}, symmetric, zero-neutralized (diagnostic)")
    return cands


# ---- documented stubs: telemetry the dumps do not carry ----
class UniqueSubsUnavailable(RuntimeError):
    pass


class PerTurnUniqueUnavailable(RuntimeError):
    pass


def unique_subs_stub(*_a, **_k):
    raise UniqueSubsUnavailable(
        "UNIQUE-SUBS requires transcript dedup telemetry - offline surface only "
        "(dumps carry only root nodes; per-sub-call prompts are generated inside "
        "REPL execution and are not logged per call)")


_STUBS = {
    "C_PT_perturn_waste": "C_PT = I + lambda*log(1+sum_t[R_t+(U_t-b)+]) requires PER-TURN "
                          "unique/redundant call counts (U_t, R_t=S_t-U_t) - not in dumps",
    "C_RW_rollout_waste": "C_RW = I + lambda*log(1+(S-U)+(U-B)+) requires ROLLOUT unique-call "
                          "count U (dedup telemetry) - not in dumps",
    "C_U_pure_unique":    "C_U = I + lambda*log(1+(U-B)+) requires unique-call count U "
                          "(dedup telemetry) - not in dumps",
}


def stub_raise(name):
    raise PerTurnUniqueUnavailable(f"{name}: requires per-turn/unique telemetry - {_STUBS[name]}")


# --------------------------------------------------------------------------
# Shaping (bit-exact operator advantage for a given cost vector + beta).
# --------------------------------------------------------------------------
def shape_group(correct, valid, costs, *, beta, tail_only, min_span,
                zero_neutralize=False, subs=None):
    """Returns (advantages, normalized_cost, fired). Mirrors adaptive_group_advantage
    base='correctness': shaped = correct if valid else 0; if beta>0 and >=2 valid
    and span>=min_span, re-rank valid siblings by min-max-normalized cost p_i, with
    an optional transform, then mean-center over the FULL group (len n).

    Transforms (mutually exclusive):
      * tail_only=tau   : q_i = max(0, p_i - tau)/(1 - tau)   (one-sided, references)
      * zero_neutralize : q_i = p_i for delegating (S>0) valid siblings; for zero-call
        (S==0) valid siblings q_i = P/(n - m) where P = sum of delegating p_i, n = FULL
        group size, m = #valid zero-call. This makes Delta_A_i = 0 EXACTLY for every
        valid zero-call rollout under the operator's group-SIZE centering.

    NOTE ON DENOMINATOR (fidelity fix vs the Codex shorthand): Codex wrote q_zero =
    mean(p over delegating siblings) = P/(n_valid - m), which zeroes Delta_A only if
    centering were over the VALID count. The live operator centers over the full group
    size (sum(shaped)/len(shaped), verified bit-exact), so exact neutralization requires
    the denominator (n_groupsize - m), NOT (n_valid - m). We use the operator-exact form
    so the |Delta_A_zero| < 1e-9 assertion holds; the two coincide only when all members
    are valid (never occurs here: max k_valid = 3 of 4)."""
    n = len(correct)
    shaped = [c if v else 0.0 for c, v in zip(correct, valid)]
    normalized = [0.0] * n
    fired = False
    if beta > 0.0:
        vi = [i for i, v in enumerate(valid) if v]
        if len(vi) >= 2:
            vc = [costs[i] for i in vi]
            lo, hi = min(vc), max(vc)
            span = hi - lo
            if span > 0.0 and span >= min_span:
                fired = True
                p = {i: (costs[i] - lo) / span for i in vi}
                if zero_neutralize:
                    if subs is None:
                        raise ValueError("zero_neutralize requires subs vector")
                    zero_idx = [i for i in vi if subs[i] == 0.0]
                    deleg_idx = [i for i in vi if subs[i] > 0.0]
                    m = len(zero_idx)
                    if deleg_idx:                       # some delegating siblings exist
                        P = sum(p[i] for i in deleg_idx)
                        qz = P / (n - m)                # operator-exact neutralization
                        for i in vi:
                            q = qz if i in zero_idx else p[i]
                            normalized[i] = q
                            shaped[i] = 1.0 - beta * q
                    else:                                # all-valid-zero: no shaping
                        for i in vi:
                            normalized[i] = 0.0
                            shaped[i] = 1.0
                else:
                    for i in vi:
                        nc = p[i]
                        if tail_only is not None:
                            nc = max(0.0, nc - tail_only) / (1.0 - tail_only)
                        normalized[i] = nc
                        shaped[i] = 1.0 - beta * nc
    baseline = sum(shaped) / n
    adv = [s - baseline for s in shaped]
    return adv, normalized, fired


# --------------------------------------------------------------------------
# Manifest: freeze the file list + sha256 + group keys.
# --------------------------------------------------------------------------
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _step_of(path: str) -> int:
    d = os.path.basename(os.path.dirname(path))
    return int(d.replace("step_", ""))


def enumerate_files(root: str, step_lo: int, step_hi: int):
    files = []
    for sd in glob.glob(os.path.join(root, "step_*")):
        try:
            s = int(os.path.basename(sd).replace("step_", ""))
        except ValueError:
            continue
        if step_lo <= s <= step_hi:
            f = os.path.join(sd, "train_rollouts.jsonl")
            if os.path.exists(f):
                files.append(f)
    files.sort(key=_step_of)
    return files


def group_keys_for_file(path: str):
    """Deterministic ordered list of (env, idx, prompt_hash, member_ids) for a file."""
    groups = defaultdict(list)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        m = r.get("metrics") or {}
        env = "bcp" if "browsecomp_plus_judge_score" in m else "oolong"
        t = r.get("task") or {}
        idx = t.get("idx") if isinstance(t, dict) else None
        ph = hashlib.sha256(((t.get("prompt") if isinstance(t, dict) else "") or "").encode()).hexdigest()[:16]
        groups[(env, idx, ph)].append(r.get("id"))
    out = []
    for (env, idx, ph), ids in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2])):
        out.append(dict(env=env, idx=idx, prompt_hash=ph, size=len(ids), member_ids=ids))
    return out


def build_manifest(root, step_lo, step_hi):
    files = enumerate_files(root, step_lo, step_hi)
    entries = []
    for f in files:
        entries.append(dict(
            step=_step_of(f), path=os.path.relpath(f, root), sha256=_sha256(f),
            groups=group_keys_for_file(f)))
    return dict(root=os.path.abspath(root), step_lo=step_lo, step_hi=step_hi,
                n_files=len(files), files=entries)


def verify_manifest(root, man):
    problems = []
    for e in man["files"]:
        f = os.path.join(root, e["path"])
        if not os.path.exists(f):
            problems.append(f"MISSING {e['path']}"); continue
        sh = _sha256(f)
        if sh != e["sha256"]:
            problems.append(f"SHA DRIFT {e['path']}: {e['sha256'][:12]} -> {sh[:12]}")
    return problems


# --------------------------------------------------------------------------
# Metric primitives (all operate on lists of per-group records).
# --------------------------------------------------------------------------
def fe_slope(groups, xkey):
    """Fixed-effects (within-group) pooled slope of A on x over valid members
    of fired groups: sum_g sum_i (x-xbar)(A-Abar) / sum_g sum_i (x-xbar)^2."""
    num = den = 0.0
    for g in groups:
        vm = g["valid_members"]
        if len(vm) < 2:
            continue
        xs = [mm[xkey] for mm in vm]
        as_ = [mm["A"] for mm in vm]
        xb = sum(xs) / len(xs); ab = sum(as_) / len(as_)
        for x, a in zip(xs, as_):
            num += (x - xb) * (a - ab)
            den += (x - xb) ** 2
    return (num / den) if den > 0 else float("nan")


def pg_mean_slope(groups, xkey):
    """v1's estimator: unweighted MEAN of per-group OLS slopes (cov/var per group,
    then averaged). Reproduces the memo's -8.7 ballpark. Reported alongside the
    fixed-effects pooled slope for reconciliation: this estimator equal-weights
    small-span/small-k groups whose per-group slope is high-variance and can flip
    sign, so it is the fragile one. fe_slope (variance-weighted) is the headline."""
    tot = 0.0; n = 0
    for g in groups:
        vm = g["valid_members"]
        if len(vm) < 2:
            continue
        xs = [mm[xkey] for mm in vm]
        as_ = [mm["A"] for mm in vm]
        xb = sum(xs) / len(xs); ab = sum(as_) / len(as_)
        cov = sum((x - xb) * (a - ab) for x, a in zip(xs, as_))
        var = sum((x - xb) ** 2 for x in xs)
        if var > 0:
            tot += cov / var; n += 1
    return (tot / n) if n else float("nan")


def corner_incidence(groups):
    """Tie-INCLUSIVE: fraction of fired groups where a valid zero-sub rollout
    attains (within tie tol) the max advantage. Returns (over_all_fired,
    over_fired_with_zerosub, n_fired, n_fired_with_zerosub)."""
    n = nz = top_all = top_cond = 0
    for g in groups:
        vm = g["valid_members"]
        if not vm:
            continue
        n += 1
        maxA = max(mm["A"] for mm in vm)
        has_zero = any(mm["subs"] == 0 for mm in vm)
        zero_top = any(mm["subs"] == 0 and mm["A"] >= maxA - 1e-12 for mm in vm)
        if has_zero:
            nz += 1
            if zero_top:
                top_cond += 1
        if zero_top:
            top_all += 1
    return (top_all / n if n else float("nan"),
            top_cond / nz if nz else float("nan"), n, nz)


def total_dose(all_groups):
    """Firing-weighted total shaping dose = sum over ALL groups of
    sum|A_candidate - A_beta0|. Returns (total, per_total_group, per_fired_group,
    n_total, n_fired)."""
    tot = 0.0; n_tot = 0; n_fired = 0
    for g in all_groups:
        n_tot += 1
        d = sum(abs(mm["A"] - mm["A0"]) for mm in g["all_members"])
        tot += d
        if g["fired"]:
            n_fired += 1
    return (tot, tot / n_tot if n_tot else float("nan"),
            tot / n_fired if n_fired else float("nan"), n_tot, n_fired)


def redistribution(groups):
    """Objection-1 quantification: mean positive increment (A_candidate - A_beta0)
    handed to valid siblings that are UNPENALIZED (normalized cost == 0, i.e.
    below-budget/cheapest) and specifically to valid ZERO-SUB siblings, over
    fired groups. A hinge/tail penalty on one sibling raises everyone else's
    centered advantage; this measures that leak."""
    inc_cheap = []; inc_zero = []
    frac_zero_pos = 0; n_with_zero = 0
    for g in groups:
        vm = g["valid_members"]
        cheaps = [mm["A"] - mm["A0"] for mm in vm if mm["nc"] == 0.0]
        zeros = [mm["A"] - mm["A0"] for mm in vm if mm["subs"] == 0]
        if cheaps:
            inc_cheap.append(sum(cheaps) / len(cheaps))
        if zeros:
            n_with_zero += 1
            mz = sum(zeros) / len(zeros)
            inc_zero.append(mz)
            if mz > 1e-12:
                frac_zero_pos += 1
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    return dict(mean_incr_cheapest=mean(inc_cheap),
                mean_incr_zerosub=mean(inc_zero),
                frac_fired_zerosub_gets_positive=(frac_zero_pos / n_with_zero if n_with_zero else float("nan")),
                n_fired_with_zerosub=n_with_zero)


def _solve(A, b):
    """Gaussian elimination for a small symmetric PSD system Ax=b. Returns None if
    singular. A tiny ridge (1e-10) on the diagonal guards near-collinearity."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        M[i][i] += 1e-10
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-15:
            return None
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= pv
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                f = M[r][col]
                for j in range(col, n + 1):
                    M[r][j] -= f * M[col][j]
    return [M[i][n] for i in range(n)]


def fe_multivariate(groups, feat_keys):
    """Within-group fixed-effects OLS: demean A and each feature within each fired
    group (valid members), pool, solve normal equations. Returns {feat: coef} or
    all-nan if singular/insufficient. Multicollinearity (I,S,T are correlated) is
    real; CIs from cluster-bootstrap expose it."""
    k = len(feat_keys)
    XtX = [[0.0] * k for _ in range(k)]
    Xty = [0.0] * k
    used = 0
    for g in groups:
        vm = g["valid_members"]
        if len(vm) < 2:
            continue
        means = {f: sum(mm[f] for mm in vm) / len(vm) for f in feat_keys}
        am = sum(mm["A"] for mm in vm) / len(vm)
        for mm in vm:
            xc = [mm[f] - means[f] for f in feat_keys]
            yc = mm["A"] - am
            for a in range(k):
                Xty[a] += xc[a] * yc
                for b in range(k):
                    XtX[a][b] += xc[a] * xc[b]
        used += 1
    if used < k + 1:
        return {f: float("nan") for f in feat_keys}
    sol = _solve(XtX, Xty)
    if sol is None:
        return {f: float("nan") for f in feat_keys}
    return {f: sol[i] for i, f in enumerate(feat_keys)}


SUB_BUCKETS = (("0", lambda s: s == 0), ("1-5", lambda s: 1 <= s <= 5),
               ("6-30", lambda s: 6 <= s <= 30), (">30", lambda s: s > 30))


def bucket_table(groups):
    """Per sub-call bucket over fired-group valid members: mean Delta_A (=A-A0),
    frac(Delta_A>0), and tie-inclusive top-advantage incidence."""
    # precompute group max advantage for top incidence
    out = {name: dict(n=0, dsum=0.0, pos=0, top=0) for name, _ in SUB_BUCKETS}
    for g in groups:
        vm = g["valid_members"]
        if not vm:
            continue
        maxA = max(mm["A"] for mm in vm)
        for mm in vm:
            for name, pred in SUB_BUCKETS:
                if pred(mm["subs"]):
                    o = out[name]
                    o["n"] += 1
                    d = mm["A"] - mm["A0"]
                    o["dsum"] += d
                    if d > 1e-12:
                        o["pos"] += 1
                    if mm["A"] >= maxA - 1e-12:
                        o["top"] += 1
                    break
    res = {}
    for name, o in out.items():
        n = o["n"]
        res[name] = dict(n=n, mean_dA=(o["dsum"] / n if n else float("nan")),
                         frac_pos=(o["pos"] / n if n else float("nan")),
                         top_inc=(o["top"] / n if n else float("nan")))
    return res


def productive_vs_zero(groups):
    """Delta(A_{1-5} - A_0): pooled mean advantage of the productive band minus the
    abstinence bucket over fired-group valid members. corner|zero cannot show whether
    productive delegation OUTRANKS abstinence; this can."""
    a0 = []; a15 = []
    for g in groups:
        for mm in g["valid_members"]:
            if mm["subs"] == 0:
                a0.append(mm["A"])
            elif 1 <= mm["subs"] <= 5:
                a15.append(mm["A"])
    m = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    return m(a15) - m(a0), len(a15), len(a0)


def component_span(groups):
    """Median iteration-span vs added-cost(penalty)-span among fired groups, split by
    k_valid in {2,3}. Shows whether the re-ranking rides iterations or excess volume."""
    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else float("nan")
    out = {}
    for k in (2, 3):
        sub = [g for g in groups if g["k_valid"] == k]
        out[k] = dict(n=len(sub),
                      med_iter_span=med([g["iter_span"] for g in sub]),
                      med_pen_span=med([g["pen_span"] for g in sub]))
    return out


def cost_preservation(cand_groups):
    """Advantage contrast between waste-tail (>30) and productive (1-5) band, pooled
    over fired-group valid members: mean A_{>30} - mean A_{1-5}. More negative = the
    waste tail is pushed below the productive band (better preservation of task-tied
    delegation). Reported absolute; relative-to-V0 handled by the caller."""
    tail = []; prod = []
    for g in cand_groups:
        for mm in g["valid_members"]:
            if mm["subs"] > 30:
                tail.append(mm["A"])
            elif 1 <= mm["subs"] <= 5:
                prod.append(mm["A"])
    m = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    return m(tail) - m(prod), len(tail), len(prod)


# --- within-group PAIRED band contrasts (re-review fix: replace pooled means) ---
def _paired_contrast(groups, band_hi, band_lo):
    """Per group: mean(A | band_hi) - mean(A | band_lo) over valid members, computed
    ONLY in groups that contain BOTH bands; averaged over qualifying groups. Removes
    the group-level baseline (paired), unlike the pooled difference-of-means."""
    contrasts = []
    for g in groups:
        hi = [mm["A"] for mm in g["valid_members"] if band_hi(mm["subs"])]
        lo = [mm["A"] for mm in g["valid_members"] if band_lo(mm["subs"])]
        if hi and lo:
            contrasts.append(sum(hi) / len(hi) - sum(lo) / len(lo))
    return (sum(contrasts) / len(contrasts) if contrasts else float("nan")), len(contrasts)


def prod_vs_zero_paired(groups):
    """Within-group mean A_{1-5} - mean A_0 (does productive delegation OUTRANK abstinence
    inside the same group?)."""
    return _paired_contrast(groups, lambda s: 1 <= s <= 5, lambda s: s == 0)


def cost_pres_paired(groups):
    """Within-group mean A_{>30} - mean A_{1-5} (is the waste tail pushed below the
    productive band inside the same group?). More negative = better preservation."""
    return _paired_contrast(groups, lambda s: s > 30, lambda s: 1 <= s <= 5)


def step_accounting(root, man):
    """Explicit included/omitted accounting for the nominal step range. Identifies
    absent steps and, for a terminal absent step, whether it is eval-only (no
    train_rollouts.jsonl dumped for the final checkpoint)."""
    lo, hi = man["step_lo"], man["step_hi"]
    nominal = set(range(lo, hi + 1))
    present = set(e["step"] for e in man["files"])
    absent = sorted(nominal - present)
    reasons = {}
    for s in absent:
        d = os.path.join(root, f"step_{s}")
        if os.path.isdir(d):
            files = os.listdir(d)
            has_train = "train_rollouts.jsonl" in files
            has_eval = any(f.startswith("eval_rollouts") for f in files)
            if not has_train and has_eval:
                reasons[s] = "eval-only checkpoint (train_rollouts.jsonl not dumped; terminal step)"
            elif not has_train:
                reasons[s] = f"dir present but no train_rollouts.jsonl (files: {sorted(files)[:3]})"
        else:
            reasons[s] = "step dir absent entirely"
    return dict(nominal=len(nominal), included=len(present), absent=absent, reasons=reasons)


def bootstrap(items, fn, *, n=1000, seed=0):
    """Resample items (groups) with replacement n times; return (2.5,50,97.5) pctl."""
    if not items:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    N = len(items)
    vals = []
    for _ in range(n):
        samp = [items[rng.randrange(N)] for _ in range(N)]
        v = fn(samp)
        if v == v:  # not nan
            vals.append(v)
    if not vals:
        return (float("nan"),) * 3
    vals.sort()
    q = lambda p: vals[min(len(vals) - 1, max(0, int(round(p * (len(vals) - 1)))))]
    return (q(0.025), q(0.5), q(0.975))


def bootstrap_mv(groups, feat_keys, *, n=1000, seed=0):
    """Cluster (group-resample) bootstrap for the multivariate FE regression.
    Returns {feat: (lo, med, hi)}."""
    if not groups:
        return {f: (float("nan"),) * 3 for f in feat_keys}
    rng = random.Random(seed)
    N = len(groups)
    cols = {f: [] for f in feat_keys}
    for _ in range(n):
        samp = [groups[rng.randrange(N)] for _ in range(N)]
        coef = fe_multivariate(samp, feat_keys)
        for f in feat_keys:
            if coef[f] == coef[f]:
                cols[f].append(coef[f])
    out = {}
    for f in feat_keys:
        v = sorted(cols[f])
        if not v:
            out[f] = (float("nan"),) * 3
        else:
            qq = lambda p: v[min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))]
            out[f] = (qq(0.025), qq(0.5), qq(0.975))
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def load_groups_from_manifest(root, man):
    """Yield reconstructed groups keyed by frozen manifest keys. Returns
    dict[(step,env,idx,phash)] -> list[record]."""
    by_id_needed = {}
    for e in man["files"]:
        step = e["step"]
        for gk in e["groups"]:
            for rid in gk["member_ids"]:
                by_id_needed[(step, rid)] = (step, gk["env"], gk["idx"], gk["prompt_hash"])
    groups = defaultdict(list)
    for e in man["files"]:
        step = e["step"]
        f = os.path.join(root, e["path"])
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            key = by_id_needed.get((step, r.get("id")))
            if key is None:
                continue
            groups[key].append(r)
    return groups


def evaluate(groups, candidates, *, beta_max, solve_floor, gamma, min_span):
    """Returns per-candidate structures + the bit-exactness tally."""
    # bit-exact validation against logged beta=0 fields (control ran beta_max=0)
    bx_checked = bx_mismatch_valid = bx_mismatch_adv = 0

    # precompute per-group parsed members
    parsed = []
    for key, grp in groups.items():
        step, env, idx, ph = key
        ms = [(r, r.get("metrics") or {}) for r in grp]
        correct = [_correct(r, m) for r, m in ms]
        valid = [c > 0.0 and not _fatal(r, m) for (r, m), c in zip(ms, correct)]
        sr = sum(1.0 for v in valid if v) / len(valid)
        beta = _beta(sr, beta_max=beta_max, solve_floor=solve_floor, gamma=gamma)
        subs = [_metric(m, "rlm_sub_llm_calls") for _, m in ms]
        iters = [_metric(m, "rlm_iterations") for _, m in ms]
        stoks = [_metric(m, "rlm_sub_llm_tokens") for _, m in ms]
        # beta=0 advantage (validity-gated centered correctness) — candidate-independent
        adv0, _, _ = shape_group(correct, valid, [0.0] * len(correct), beta=0.0,
                                 tail_only=None, min_span=min_span)
        # bit-exactness: the live control logged beta=0; compare recompute to logs
        for (r, m), v, a0 in zip(ms, valid, adv0):
            if "adaptive_advantage" in m and _metric(m, "adaptive_beta") == 0.0:
                bx_checked += 1
                if abs((1.0 if v else 0.0) - _metric(m, "adaptive_valid")) > 1e-9:
                    bx_mismatch_valid += 1
                if abs(a0 - _metric(m, "adaptive_advantage")) > 1e-9:
                    bx_mismatch_adv += 1
        parsed.append(dict(key=key, env=env, correct=correct, valid=valid, sr=sr,
                           beta=beta, subs=subs, iters=iters, stoks=stoks, adv0=adv0, ms=ms))

    # per-candidate group records
    results = {}
    zn_checked = zn_fail = 0     # zero-neutralization assertion tally (over ALL zn cells)
    zn_per_cand = {}             # B5-specific etc.: distinct zero-call records per cell
    for cname, spec in candidates.items():
        costfn = spec["cost"]; tail = spec["tail_only"]; zn = spec.get("zn", False)
        budget = spec.get("budget")
        all_groups = []          # every group (for dose)
        fired_groups = []        # only fired (for slope/corner/redist)
        for p in parsed:
            costs = [costfn(m) for _, m in p["ms"]]
            adv, nc, fired = shape_group(p["correct"], p["valid"], costs,
                                         beta=p["beta"], tail_only=tail, min_span=min_span,
                                         zero_neutralize=zn, subs=(p["subs"] if zn else None))
            all_members = [dict(A=adv[i], A0=p["adv0"][i]) for i in range(len(adv))]
            k_valid = sum(1 for v in p["valid"] if v)
            vc = [costs[i] for i, v in enumerate(p["valid"]) if v]
            span = (max(vc) - min(vc)) if len(vc) >= 2 else 0.0
            # added-cost span = span of the pure penalty component (excludes iterations)
            if budget is not None:
                pen = [max(0.0, p["subs"][i] - budget) for i, v in enumerate(p["valid"]) if v]
                pen_span = (max(pen) - min(pen)) if len(pen) >= 2 else 0.0
            else:
                pen_span = 0.0
            iv = [p["iters"][i] for i, v in enumerate(p["valid"]) if v]
            iter_span = (max(iv) - min(iv)) if len(iv) >= 2 else 0.0
            vm = [dict(A=adv[i], A0=p["adv0"][i], subs=p["subs"][i], iters=p["iters"][i],
                       stoks=p["stoks"][i], nc=nc[i]) for i, v in enumerate(p["valid"]) if v]
            # zero-neutralization assertion: |Delta_A| < 1e-9 for valid zero-call members.
            # FAILURE comparator is >= 1e-9 (re-review fix; was >). Counts distinct rollout
            # records (one per valid zero-call member) in fired zn groups where the transform ran.
            if zn:
                c = zn_per_cand.setdefault(cname, dict(checked=0, fail=0, zero_records_all=0))
                for mm in vm:
                    if mm["subs"] == 0.0:
                        c["zero_records_all"] += 1
                        if fired:                       # transform actually executed
                            c["checked"] += 1
                            zn_checked += 1
                            if abs(mm["A"] - mm["A0"]) >= 1e-9:
                                c["fail"] += 1
                                zn_fail += 1
            grec = dict(env=p["env"], fired=fired, k_valid=k_valid, span=span,
                        pen_span=pen_span, iter_span=iter_span,
                        all_members=all_members, valid_members=vm)
            all_groups.append(grec)
            if fired:
                fired_groups.append(grec)
        results[cname] = dict(all=all_groups, fired=fired_groups, spec=spec)

    bx = dict(checked=bx_checked, mismatch_valid=bx_mismatch_valid, mismatch_adv=bx_mismatch_adv)
    zn_stat = dict(checked=zn_checked, fail=zn_fail, per_cand=zn_per_cand)
    return results, bx, parsed, zn_stat


def census(parsed, results):
    """k-valid distribution over all groups + tail-degeneracy over fired groups."""
    kv = defaultdict(lambda: defaultdict(int))
    for p in parsed:
        k = sum(1 for v in p["valid"] if v)
        kv["pooled"][k] += 1
        kv[p["env"]][k] += 1
    # tail degeneracy: among V4 fired groups, fraction with k_valid==2 (tail==symmetric)
    deg = {}
    for cname in ("V4_iters_tail",):
        fg = results[cname]["fired"]
        n = len(fg)
        k2 = sum(1 for g in fg if g["k_valid"] == 2)
        deg[cname] = dict(n_fired=n, k2=k2, frac_k2=(k2 / n if n else float("nan")))
    return kv, deg


def span_dist(groups):
    xs = sorted(g["span"] for g in groups if g["span"] > 0)
    if not xs:
        return {}
    q = lambda p: xs[min(len(xs) - 1, int(p * (len(xs) - 1)))]
    return dict(n=len(xs), min=xs[0], p10=q(0.1), p50=q(0.5), p90=q(0.9), max=xs[-1])


def env_subset(groups, env):
    return [g for g in groups if g["env"] == env] if env != "pooled" else groups


def fmt(x, d=3):
    return "nan" if (x != x) else f"{x:.{d}f}"


REGISTERED_CELL = "C_LH_B5_x1"   # C_LH B=5, lambda = lambda_star(=2.0) * mult 1.0 = 2.0, zero-neutralized


def launch_gate(results, *, boot_n, registered=REGISTERED_CELL, v0="V0_today"):
    """Offline launch gate (prereg rule 2). ALL metrics on COMMON SUPPORT: every group
    in the manifest, non-fired groups contributing A = A_beta0 (res['all']). Six
    conditions, each verbatim; 1000 group bootstraps where a CI is used.

      1  upper95(b_subs)  < 0        (multivariate FE, common support)
      2  upper95(b_iters) < 0
      3  upper95(cost_pres_paired) < 0
      4  prod_vs_zero_paired > 0     (point estimate)
      5  dose/total  <= V0
      6  corner|zero <= V0
    """
    cand = results[registered]["all"]
    ref = results[v0]["all"]
    feats = ["iters", "subs", "stoks"]

    # 1,2: multivariate FE coefficients + cluster-bootstrap upper 97.5
    ci = bootstrap_mv(cand, feats, n=boot_n)
    coef = fe_multivariate(cand, feats)
    b_subs_u = ci["subs"][2]; b_iters_u = ci["iters"][2]

    # 3: paired cost-preservation upper 97.5 (< 0 required)
    cp_pt = cost_pres_paired(cand)[0]
    cp_ci = bootstrap(cand, lambda gs: cost_pres_paired(gs)[0], n=boot_n)
    cp_u = cp_ci[2]

    # 4: paired productive-vs-zero point estimate (> 0 required)
    pvz_pt, pvz_n = prod_vs_zero_paired(cand)
    pvz_ci = bootstrap(cand, lambda gs: prod_vs_zero_paired(gs)[0], n=boot_n)

    # 5: dose/total-group vs V0 (<= required); candidate own bootstrap CI
    dose_c = total_dose(cand)[1]
    dose_v0 = total_dose(ref)[1]
    dose_ci = bootstrap(cand, lambda gs: total_dose(gs)[1], n=boot_n)

    # 6: corner|zero vs V0 (<= required); candidate own bootstrap CI
    cz_c = corner_incidence(cand)[1]
    cz_v0 = corner_incidence(ref)[1]
    cz_ci = bootstrap(cand, lambda gs: corner_incidence(gs)[1], n=boot_n)

    rows = [
        ("1 upper95(b_subs) < 0", coef["subs"], (ci["subs"][0], ci["subs"][2]), "u95<0", b_subs_u < 0),
        ("2 upper95(b_iters) < 0", coef["iters"], (ci["iters"][0], ci["iters"][2]), "u95<0", b_iters_u < 0),
        ("3 upper95(cost_pres_paired) < 0", cp_pt, (cp_ci[0], cp_ci[2]), "u95<0", cp_u < 0),
        ("4 prod_vs_zero_paired > 0 (point)", pvz_pt, (pvz_ci[0], pvz_ci[2]), "point>0", pvz_pt > 0),
        ("5 dose/total <= V0", dose_c, (dose_ci[0], dose_ci[2]), "<=V0", dose_c <= dose_v0),
        ("6 corner|zero <= V0", cz_c, (cz_ci[0], cz_ci[2]), "<=V0", cz_c <= cz_v0),
    ]
    overall = all(r[4] for r in rows)
    extra = dict(pvz_n=pvz_n, dose_v0=dose_v0, cz_v0=cz_v0,
                 n_groups=len(cand), n_fired=sum(1 for g in cand if g["fired"]))
    return rows, overall, extra


def _bit_exact_block(P, bx):
    P("\n[BIT-EXACTNESS] recompute of control's logged beta=0 advantage vs operator log")
    P(f"  rollouts checked (beta==0 & adaptive_advantage present): {bx['checked']}")
    P(f"  valid-flag mismatches (>1e-9): {bx['mismatch_valid']}")
    P(f"  advantage   mismatches (>1e-9): {bx['mismatch_adv']}")
    ok = bx["mismatch_valid"] == 0 and bx["mismatch_adv"] == 0
    P(f"  VERDICT: {'PASS (bit-exact)' if ok else 'FAIL'}"
      + ("  [beta_max=0 arm: 0 checks means the dump is not the beta=0 control]" if bx["checked"] == 0 else ""))


def _step_acct_block(P, step_acc):
    P("\n[STEP ACCOUNTING] included vs nominal range")
    P(f"  nominal steps in range: {step_acc['nominal']}   included files: {step_acc['included']}   "
      f"absent: {step_acc['absent'] or 'none'}")
    for s, why in step_acc["reasons"].items():
        P(f"    step_{s}: {why}")


def _zn_block(P, zn_stat, registered=REGISTERED_CELL):
    P("\n[ZERO-NEUTRALIZATION ASSERTION] |Delta_A| >= 1e-9 == FAIL, over valid zero-call members")
    P("  transform on NEW cells (C_LH_*, C_NH_*); denominator = (groupsize - m), operator-exact")
    P("  (NOT Codex's delegating-count mean) so abstinence Delta_A is exactly 0.")
    P(f"  aggregate zero-call members checked (all fired zn cells): {zn_stat['checked']}, failures: {zn_stat['fail']}")
    b5 = (zn_stat.get("per_cand") or {}).get(registered)
    if b5:
        P(f"  B5-SPECIFIC [{registered}] distinct zero-call records: checked(fired)={b5['checked']} "
          f"fail={b5['fail']} total-in-manifest(incl non-fired)={b5['zero_records_all']}")
    P(f"  VERDICT: {'PASS (abstinence Delta_A exactly 0)' if zn_stat['fail']==0 else 'FAIL'}")


def report_gate(results, bx, step_acc, zn_stat, *, label, tag, manifest_mode, n_files, out_dir,
                boot_n, lambda_source, registered=REGISTERED_CELL):
    lines = []
    P = lines.append
    P("=" * 100)
    P("MITIGATION RE-DERIVATION v2 -- LAUNCH-GATE MODE (scripts/12_mitigation_rederive.py)")
    P(f"LABEL: {label}")
    P(f"registered candidate: {registered}  ({results[registered]['spec']['desc']})")
    P(f"lambda source: {lambda_source}")
    P(f"tag={tag}  manifest={manifest_mode}  n_files={n_files}  boot={boot_n}")
    P("=" * 100)
    _bit_exact_block(P, bx)
    _step_acct_block(P, step_acc)
    _zn_block(P, zn_stat, registered)

    rows, overall, extra = launch_gate(results, boot_n=boot_n, registered=registered)
    P(f"\n{'='*100}\n[LAUNCH GATE] common support: {extra['n_groups']} groups "
      f"({extra['n_fired']} fired for the registered cell); ALL conditions on common support, "
      f"non-fired groups contribute A=A_beta0")
    P(f"  paired-contrast qualifying groups (prod_vs_zero): n={extra['pvz_n']}")
    P(f"{'condition':36s} {'value':>12s} {'[boot 2.5,97.5]':>22s} {'rule':>8s} {'result':>7s}")
    for name, val, ci, rule, ok in rows:
        P(f"{name:36s} {fmt(val,5):>12s} [{fmt(ci[0],5)},{fmt(ci[1],5)}] {rule:>8s} {'PASS' if ok else 'FAIL':>7s}")
    P(f"\n  (condition 5 ref: V0 dose/total={fmt(extra['dose_v0'],5)};  "
      f"condition 6 ref: V0 corner|zero={fmt(extra['cz_v0'],5)})")
    P(f"\n  OVERALL GATE: {'PASSED' if overall else 'BLOCKED'}")

    text = "\n".join(lines)
    print(text)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"launch_gate_{tag}.txt"), "w") as fh:
        fh.write(text + "\n")
    return text


def report(results, bx, parsed, census_kv, deg, step_acc, *, label, tag, manifest_mode, n_files, out_dir,
           beta_max, solve_floor, gamma, min_span, boot_n, zn_stat, lambdas, lambda_source):
    lines = []
    P = lines.append
    P("=" * 100)
    P("MITIGATION RE-DERIVATION v2 (scripts/12_mitigation_rederive.py)")
    P(f"LABEL: {label}")
    P(f"tag={tag}  manifest={manifest_mode}  n_files={n_files}  "
      f"beta_max={beta_max} solve_floor={solve_floor} gamma={gamma} min_span={min_span} boot={boot_n}")
    P("=" * 100)

    _bit_exact_block(P, bx)
    _step_acct_block(P, step_acc)
    _zn_block(P, zn_stat)

    # lambda calibration (frozen)
    P(f"\n[LAMBDA CALIBRATION]  source={lambda_source}  (FROZEN; held fixed for the 120-200 pass)")
    P("  lambda_B* = median_g span_g(I) / median_g span_g(log(1+(S-B)+)), penalty-active groups")
    P(f"  {'B':>3s} {'lambda_B*':>12s} {'med_span_I':>12s} {'med_span_H':>12s} {'n_I':>6s} {'n_H':>6s}   sweep {{0.5x,1x,2x}}")
    for B in sorted(lambdas):
        d = lambdas[B]
        sweep = "  ".join(f"{d['lambda_star']*mu:.4g}" for mu in LAM_MULTS)
        P(f"  {B:>3d} {fmt(d['lambda_star'],4):>12s} {fmt(d['med_span_I'],3):>12s} "
          f"{fmt(d['med_span_H'],3):>12s} {d['n_I']:>6d} {d['n_H']:>6d}   [{sweep}]")

    # k-valid census
    P("\n[K-VALID CENSUS] over all reconstructed groups (all groups are size-4)")
    for env in ("pooled", "oolong", "bcp"):
        d = census_kv.get(env, {})
        tot = sum(d.values())
        P(f"  {env:7s}: total={tot}  " + "  ".join(f"k={k}:{d.get(k,0)}" for k in (0,1,2,3,4)))
    for cn, dd in deg.items():
        P(f"  tail-degeneracy [{cn}]: fired={dd['n_fired']} k2={dd['k2']} "
          f"frac(tail==symmetric @k_valid=2)={fmt(dd['frac_k2'])}")

    # main table per env
    for env in ("pooled", "oolong", "bcp"):
        P(f"\n{'='*100}\n[{env.upper()}] candidate x metric  (CI = 2.5/97.5 pctl, {boot_n} group-resamples)")
        P(f"{'candidate':16s} {'fired':>6s} {'dA/dsubs*1e4':>22s} {'dA/diters*1e4':>22s} "
          f"{'corner(all)':>16s} {'corner|zero':>12s} {'dose/total':>20s} {'dose/fired':>14s}")
        for cname, res in results.items():
            fired = env_subset(res["fired"], env)
            allg = env_subset(res["all"], env)
            nf = len(fired)
            ds = fe_slope(fired, "subs"); di = fe_slope(fired, "iters")
            ds_ci = bootstrap(fired, lambda gs: fe_slope(gs, "subs"), n=boot_n)
            di_ci = bootstrap(fired, lambda gs: fe_slope(gs, "iters"), n=boot_n)
            ca_all, ca_zero, ncf, nz = corner_incidence(fired)
            ca_ci = bootstrap(fired, lambda gs: corner_incidence(gs)[0], n=boot_n)
            tot, per_tot, per_fired, ntot, nfg = total_dose(allg)
            dose_ci = bootstrap(allg, lambda gs: total_dose(gs)[1], n=boot_n)
            P(f"{cname:16s} {nf:6d} "
              f"{fmt(1e4*ds,2):>10s}[{fmt(1e4*ds_ci[0],1)},{fmt(1e4*ds_ci[2],1)}] "
              f"{fmt(1e4*di,2):>10s}[{fmt(1e4*di_ci[0],1)},{fmt(1e4*di_ci[2],1)}] "
              f"{fmt(ca_all):>7s}[{fmt(ca_ci[0],2)},{fmt(ca_ci[2],2)}] "
              f"{fmt(ca_zero):>12s} "
              f"{fmt(per_tot,4):>10s}[{fmt(dose_ci[0],4)},{fmt(dose_ci[2],4)}] "
              f"{fmt(per_fired,4):>14s}")

    # estimator reconciliation: FE-pooled (headline) vs v1's mean-of-per-group slope
    P(f"\n{'='*100}\n[ESTIMATOR RECONCILIATION] dA/dsubs*1e4 : FE-pooled (headline, robust) vs v1 mean-of-per-group (fragile)")
    P(f"{'candidate':16s} " + "  ".join(f"{e+':FE/PG':>18s}" for e in ("pooled", "oolong", "bcp")))
    for cname, res in results.items():
        cells = []
        for env in ("pooled", "oolong", "bcp"):
            fg = env_subset(res["fired"], env)
            cells.append(f"{fmt(1e4*fe_slope(fg,'subs'),2)}/{fmt(1e4*pg_mean_slope(fg,'subs'),2)}")
        P(f"{cname:16s} " + "  ".join(f"{c:>18s}" for c in cells))

    # redistribution (objection 1) — pooled
    P(f"\n{'='*100}\n[REDISTRIBUTION / obj-1] pooled fired groups: increment (A_cand - A_beta0) leaked to siblings")
    P(f"{'candidate':16s} {'mean_incr_cheapest':>20s} {'mean_incr_zerosub':>20s} "
      f"{'frac_zerosub>0':>16s} {'n_fired_w/zero':>14s}")
    for cname, res in results.items():
        rd = redistribution(res["fired"])
        P(f"{cname:16s} {fmt(rd['mean_incr_cheapest'],5):>20s} {fmt(rd['mean_incr_zerosub'],5):>20s} "
          f"{fmt(rd['frac_fired_zerosub_gets_positive'],3):>16s} {rd['n_fired_with_zerosub']:>14d}")

    # span distribution per candidate (pooled)
    P(f"\n{'='*100}\n[SPAN DISTRIBUTION] valid-cost span among fired groups (pooled)")
    for cname, res in results.items():
        sd = span_dist(res["fired"])
        if sd:
            P(f"  {cname:16s} n={sd['n']} min={fmt(sd['min'],2)} p10={fmt(sd['p10'],2)} "
              f"p50={fmt(sd['p50'],2)} p90={fmt(sd['p90'],2)} max={fmt(sd['max'],2)}")
        else:
            P(f"  {cname:16s} (no fired groups)")

    # multivariate FE regression (obj: joint slopes, not marginal)
    feats = ["iters", "subs", "stoks"]
    P(f"\n{'='*100}\n[MULTIVARIATE FE]  A ~ within-group(iters, subs, sub_tokens), pooled fired groups")
    P("  coefficients with cluster-bootstrap 95% CI; I,S,T collinear -> read signs/CIs, not point magnitudes")
    P(f"{'candidate':16s} {'b_iters [CI]':>26s} {'b_subs*1e4 [CI]':>26s} {'b_subtok*1e7 [CI]':>28s}")
    for cname, res in results.items():
        fg = res["fired"]
        coef = fe_multivariate(fg, feats)
        ci = bootstrap_mv(fg, feats, n=min(boot_n, 400))
        P(f"{cname:16s} "
          f"{fmt(coef['iters'],3):>8s}[{fmt(ci['iters'][0],2)},{fmt(ci['iters'][2],2)}] "
          f"{fmt(1e4*coef['subs'],2):>8s}[{fmt(1e4*ci['subs'][0],1)},{fmt(1e4*ci['subs'][2],1)}] "
          f"{fmt(1e7*coef['stoks'],2):>8s}[{fmt(1e7*ci['stoks'][0],1)},{fmt(1e7*ci['stoks'][2],1)}]")

    # bucket redistribution table (S in {0,1-5,6-30,>30})
    P(f"\n{'='*100}\n[BUCKET REDISTRIBUTION] pooled fired-group valid members, by sub-call bucket")
    P("  per cell: mean_dA (A-A0) / frac(dA>0) / tie-inclusive top-incidence")
    P(f"{'candidate':16s} " + "".join(f"{('S='+name):>26s}" for name, _ in SUB_BUCKETS))
    for cname, res in results.items():
        bt = bucket_table(res["fired"])
        cells = []
        for name, _ in SUB_BUCKETS:
            b = bt[name]
            cells.append(f"{fmt(b['mean_dA'],4)}/{fmt(b['frac_pos'],2)}/{fmt(b['top_inc'],2)}(n{b['n']})")
        P(f"{cname:16s} " + "".join(f"{c:>26s}" for c in cells))

    # productive-vs-zero contrast + cost-preservation proxy (relative to V0)
    v0_cp = cost_preservation(results["V0_today"]["fired"])[0]
    P(f"\n{'='*100}\n[PRODUCTIVE-vs-ZERO  &  COST-PRESERVATION]  pooled fired-group valid members")
    P("  prod_vs_zero = mean A_{1-5} - mean A_0  (does productive delegation OUTRANK abstinence?)")
    P("  cost_pres = mean A_{>30} - mean A_{1-5}  (more negative = waste tail below productive band)")
    P(f"  V0 cost_pres baseline = {fmt(v0_cp,4)}")
    P(f"{'candidate':16s} {'prod_vs_zero [CI]':>28s} {'cost_pres':>12s} {'cost_pres - V0':>16s}")
    for cname, res in results.items():
        pvz, n15, n0 = productive_vs_zero(res["fired"])
        pvz_ci = bootstrap(res["fired"], lambda gs: productive_vs_zero(gs)[0], n=boot_n)
        cp = cost_preservation(res["fired"])[0]
        rel = (cp - v0_cp) if (cp == cp and v0_cp == v0_cp) else float("nan")
        P(f"{cname:16s} {fmt(pvz,4):>10s}[{fmt(pvz_ci[0],3)},{fmt(pvz_ci[2],3)}] "
          f"{fmt(cp,4):>12s} {fmt(rel,4):>16s}")

    # component-span attribution (k_valid = 2 vs 3)
    P(f"\n{'='*100}\n[COMPONENT-SPAN ATTRIBUTION] median iteration-span vs added-cost(penalty)-span, fired groups")
    P("  (budgeted cells only; shows whether re-ranking rides iterations or excess volume)")
    P(f"{'candidate':16s} {'k=2: iter/pen (n)':>28s} {'k=3: iter/pen (n)':>28s}")
    for cname, res in results.items():
        if res["spec"].get("budget") is None:
            continue
        cs = component_span(res["fired"])
        c2, c3 = cs[2], cs[3]
        s2 = "{}/{} (n{})".format(fmt(c2["med_iter_span"], 2), fmt(c2["med_pen_span"], 2), c2["n"])
        s3 = "{}/{} (n{})".format(fmt(c3["med_iter_span"], 2), fmt(c3["med_pen_span"], 2), c3["n"])
        P(f"{cname:16s} {s2:>28s} {s3:>28s}")

    # feasibility stubs
    P(f"\n{'='*100}\n[TELEMETRY-GATED STUBS]  (candidates NOT computable from the dumps)")
    try:
        unique_subs_stub()
    except UniqueSubsUnavailable as e:
        P(f"  UNIQUE-SUBS: {e}")
    for name in _STUBS:
        try:
            stub_raise(name)
        except PerTurnUniqueUnavailable as e:
            P(f"  {e}")

    text = "\n".join(lines)
    print(text)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"rederive_report_{tag}.txt"), "w") as fh:
        fh.write(text + "\n")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/qwen3-30b-t2-control/run_default/rollouts")
    ap.add_argument("--step-lo", type=int, default=1)
    ap.add_argument("--step-hi", type=int, default=99)
    ap.add_argument("--tag", default="t2control_1_99")
    ap.add_argument("--label", default=None,
                    help="provenance label; if omitted it is auto-generated from the step range + mode "
                         "(fixes the label bug where a 120-200 pass was mislabelled 'pre-collapse')")
    ap.add_argument("--launch-gate", action="store_true",
                    help="evaluate ONLY the registered candidate (C_LH_B5, lambda from --lambda-from) "
                         "against the six prereg conditions on common support; print PASS/FAIL + verdict")
    ap.add_argument("--manifest-mode", choices=("write", "verify", "auto"), default="auto")
    ap.add_argument("--out-dir", default="outputs/advisor/mitigation_rederive")
    ap.add_argument("--beta-max", type=float, default=0.15)
    ap.add_argument("--solve-floor", type=float, default=0.25)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--min-span", type=float, default=1.0,
                    help="registered treatment floor (plan Phase-1); operator source default is 0.0")
    ap.add_argument("--boot-n", type=int, default=1000)
    ap.add_argument("--lambda-from", default=None,
                    help="load FROZEN lambda calibration from lambda_calib_<tag>.json (use for the "
                         "120-200 decision pass so lambdas stay fixed to the 1-99 calibration)")
    args = ap.parse_args()

    # resolve root relative to repo root (this file lives in scripts/)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = args.root if os.path.isabs(args.root) else os.path.join(repo, args.root)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(repo, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    man_path = os.path.join(out_dir, f"manifest_{args.tag}.json")

    # manifest handling (obj 3: freeze, never glob into the measurement)
    if args.manifest_mode == "write" or (args.manifest_mode == "auto" and not os.path.exists(man_path)):
        man = build_manifest(root, args.step_lo, args.step_hi)
        with open(man_path, "w") as fh:
            json.dump(man, fh, indent=1)
        mode = "WRITTEN"
    else:
        with open(man_path) as fh:
            man = json.load(fh)
        problems = verify_manifest(root, man)
        if problems:
            print("MANIFEST VERIFY FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
            sys.exit(2)
        mode = "VERIFIED"

    groups = load_groups_from_manifest(root, man)

    # parse once (for calibration) via a throwaway evaluate on an empty candidate set is wasteful;
    # instead calibrate from a light parse pass then build candidates and run the full evaluate.
    # lambda calibration: freeze on first computation; load frozen for the decision pass.
    lam_path = os.path.join(out_dir, f"lambda_calib_{args.tag}.json")
    if args.lambda_from:
        src = args.lambda_from if os.path.isabs(args.lambda_from) else os.path.join(out_dir, args.lambda_from)
        with open(src) as fh:
            raw = json.load(fh)
        lambdas = {int(k): v for k, v in raw["lambdas"].items()}
        lambda_source = f"FROZEN loaded from {os.path.relpath(src, repo)}"
    else:
        # light parse for calibration
        parsed_lite = []
        for grp in groups.values():
            ms = [(r, r.get("metrics") or {}) for r in grp]
            correct = [_correct(r, m) for r, m in ms]
            valid = [c > 0.0 and not _fatal(r, m) for (r, m), c in zip(ms, correct)]
            parsed_lite.append(dict(valid=valid,
                                    subs=[_metric(m, "rlm_sub_llm_calls") for _, m in ms],
                                    iters=[_metric(m, "rlm_iterations") for _, m in ms]))
        lambdas = calibrate_lambdas(parsed_lite)
        with open(lam_path, "w") as fh:
            json.dump(dict(tag=args.tag, step_lo=args.step_lo, step_hi=args.step_hi,
                           lam_mults=list(LAM_MULTS), lambdas={str(k): v for k, v in lambdas.items()}), fh, indent=1)
        lambda_source = f"computed from this manifest ({args.tag}) -> frozen at {os.path.relpath(lam_path, repo)}"

    # step accounting (explicit included/omitted vs nominal range)
    step_acc = step_accounting(root, man)

    # auto-label (fix the provenance bug): reflect actual step range + mode
    lo, hi = man["step_lo"], man["step_hi"]
    if args.label is not None:
        label = args.label
    elif args.launch_gate:
        label = f"LAUNCH-GATE decision pass -- frozen control steps {lo}-{hi} ({args.manifest_mode}:{mode})"
    elif lo >= 120 or hi >= 120:
        label = f"late/collapse-regime pass -- frozen steps {lo}-{hi}; DECISION dataset"
    else:
        label = f"pre-collapse-regime validation -- steps {lo}-{hi}; NOT the decision dataset"

    candidates = build_candidates(lambdas)
    if args.launch_gate:
        # restrict to V0 (reference for conditions 5,6) + the registered cell
        candidates = {k: v for k, v in candidates.items() if k in ("V0_today", REGISTERED_CELL)}
        if REGISTERED_CELL not in candidates:
            print(f"registered cell {REGISTERED_CELL} not built (check lambda json)", file=sys.stderr)
            sys.exit(3)

    results, bx, parsed, zn_stat = evaluate(groups, candidates,
                                            beta_max=args.beta_max, solve_floor=args.solve_floor,
                                            gamma=args.gamma, min_span=args.min_span)

    if args.launch_gate:
        report_gate(results, bx, step_acc, zn_stat, label=label, tag=args.tag,
                    manifest_mode=f"{args.manifest_mode}:{mode}", n_files=man["n_files"],
                    out_dir=out_dir, boot_n=args.boot_n, lambda_source=lambda_source)
    else:
        census_kv, deg = census(parsed, results)
        report(results, bx, parsed, census_kv, deg, step_acc, label=label, tag=args.tag,
               manifest_mode=f"{args.manifest_mode}:{mode}", n_files=man["n_files"], out_dir=out_dir,
               beta_max=args.beta_max, solve_floor=args.solve_floor, gamma=args.gamma,
               min_span=args.min_span, boot_n=args.boot_n, zn_stat=zn_stat,
               lambdas=lambdas, lambda_source=lambda_source)


if __name__ == "__main__":
    main()
