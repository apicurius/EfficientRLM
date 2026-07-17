# Phase-0 review memo — mitigation design freeze

Date: 2026-07-17 (drafted 2026-07-16 while control trains 180→200).
Scope: walks each Phase-0 gate of
`MITIGATION_PLAN_ONESIDED_TAIL_DELEGATION_FLOOR.md` against the evidence
collected 2026-07-15/16, states what each gate's own decision rule now
implies, and isolates the items that require the sign-off decision.
Evidence sources: the closed-harness canary campaign
(`outputs/advisor/CANARY_T2C.md`, all cells 3-rep pooled n=150 unless
noted), the counterfactual replay amendment already in the plan, and the
studio run telemetry (audits 101–180 intact, 80/80 steps).

## Gate status

**Gate 1 — control@200 + sweep complete: PENDING (hours).** Control is at
step ~180 with 20 steps remaining; the pre-registered sweep executes on
the step-200 handoff. Nothing below is a measurement change; this memo
freezes *design*, and the plan's own rule (nothing edits code while a
pre-registered measurement is pending) continues to hold.

**Gate 2 — counterfactual dip read: RESOLVED, retraction branch.** The
control arm (β ≡ 0) reproduced the treatment's step-107–139 train-reward
dip at matching magnitude against its own same-machine baseline: oolong
−0.066, browsecomp −0.091, sustained across 32/33 window steps. The
dose-feedback diagnosis is therefore retracted, exactly the branch the
gate pre-registered. Mechanistic corroboration: the dip's floor coincided
with a truncation/length-drift excursion (truncation 30%→81%→37.5%,
turns 8.4→12.9→9.5) that self-reversed in a β = 0 run, and the eval
surface round-tripped to best-of-run levels at step 160 (browsecomp
0.600, oolong 0.448, both above the step-100 baseline). Consequences per
the gate's rule: the table is re-rated — mitigation 1 (β anneal /
exposure cap) loses its motivating evidence and is demoted to an optional
safety cap; prediction P-1 is re-scoped to within-run oscillation
structure (already required by the hardware decision for independent
reasons). Mitigation 2 remains justified by the collapse alone.

**Gate 3 — collapse attribution: RESOLVED, lever-attributed.** The
attribution no longer rests on the offline 2×2 alone; the closed-harness
trajectory decides it:

| CodeQA (acc / delegating share) | @120 | @140 | @160 | @200 |
|---|---|---|---|---|
| control | 0.360 / 27% | 0.327 / 37% | 0.373 / 37% | (pending) |
| treatment | 0.453 / 47% | — | — | 0.347 / 21% |
| base anchor | 0.393 / 38% | | | |

Control's delegating share oscillates in a band around the base anchor at
three checkpoints; the treatment goes from the best transfer cell on the
board to delegation extinction across steps 120–200. Cross-domain probes
(LongBench Single-Doc QA, n=50/cell) show the same shape at higher
baseline propensity: base 90%, control@140 84%, treatment@120 82% at one
third the spend, treatment@200 54%. Mechanism statement: through step 120
the lever shaped spend-per-delegation (the intended product — propensity
intact, spend ~3× down, best transfer accuracy); from 120 to 200 it
suppressed delegation propensity itself. The fence probe seals the
capability reading on identical hardware: ordered to delegate, control@160
multiplies sub-call spend 5.5× (complies, at −0.11 accuracy), while
treatment@200 shows zero behavioral response (0.7 sub-calls fenced vs 0.8
unfenced) — the collapse is unreachable by prompting, not merely
disfavored. Per the gate's rule, lever-attributed ⇒ **mitigation 2 is
mandatory for the next launch**. Two caveats stay attached: the trajectory
is arm-correlated evidence, not a randomized dose test (the arms differ
only in β, but per-checkpoint dose randomization does not exist); and the
c180/c200 control points are pending hardware (below).

**Gate 4 — user sign-off: THE OPEN ITEM.** Everything below frames that
decision.

## Design under freeze (restated, with one composition question)

Registered default per the 2026-07-15 amendment: `cost_basis =
"iterations"` + `tail_only = 0.5`. The replay showed this combination is
what actually removes the anti-delegation gradient (dA/dsubs −8.7/−4.4 →
+1.4/−2.6); tail-only alone on the subs-priced basis fails, because the
tail *is* the high-delegation side. The floor (λ·1[subcalls = 0]) is moot
under an unpriced subs term (measured corner attraction 0.05) and is
demoted to fallback.

Composition question to resolve at review: the Phase-1 floor was drafted
against the subs-priced basis (`iterations_log_subcalls_floor`). Under the
amended default the floor, if ever re-activated as fallback, composes as
`iterations + λ·1[subcalls = 0]`; the prereg should register that form
explicitly so the fallback is not improvised later.

Predictions, updated by tonight's evidence: P-2 (transfer zero-share stays
at base level at every checkpoint) is now the decisive probe and has a
sharpened operational form — the canary protocol exists, is calibrated
(rep variance known: delegating share ±9–11 points single-rep, hence 3-rep
pooling), and the in-run transfer canary is promoted to a mandatory
instrument for the next launch. P-1 is re-scoped as above. P-3 keeps the
turns/wall-time channel; the sub-call-volume component is explicitly
surrendered by the basis change and the prereg should say so. P-4
unchanged.

## What tonight's evidence does NOT license

- No claim that treatment@200 is damaged below base on transfer: pooled
  accuracy 0.347 vs anchor 0.393 is ~1.2 SE; the loss is the surrender of
  the @120 edge (0.453), not below-base capability.
- No cross-harness level comparisons: canary cells are ai16-closed;
  studio eval cells are studio-closed; the released policy's model-card
  42.0 sits under undisclosed conditioning (its own base reads 22.0 vs
  ~39 here) and stays anchor-table-only.
- No mid-run or post-hoc patch to t2: both arms complete untouched.

## Open items and contingencies

1. **c180 + c200 canary points**: adapters banked on HF; ai16 down
   (NODE_FAIL 2026-07-16 10:24, holder auto-requeued and pinned, admins
   emailed). If the node revives, the points complete on the closed
   harness. If it stays down through the sweep, the fallback is a fresh
   studio-side trajectory leg (own base anchor, all five control adapters
   re-measured under one serving config) — comparable within itself,
   never mixed with ai16 rows.
2. **Sweep** executes on the step-200 note regardless of ai16.
3. **Pilot hardware** (Phase 4) assumed ai16 in the plan; if the outage
   persists, the pilot inherits the same studio-fallback decision rule as
   the full run (calendar rule unchanged).

## Recommendation

Approve the freeze with: amended registered default (iterations +
tail_only 0.5); floor registered as fallback in its composed form; β
exposure cap retained as an optional safety knob without evidentiary
claims; P-1..P-4 as re-scoped above; in-run transfer canary mandatory.
Freeze becomes a PREREG document upon sign-off; implementation remains
gated on Gate 1 completing.

## ADDENDUM 2026-07-16 — adversarial review verdict: REJECT (design freeze withdrawn)

An independent adversarial review (Codex, full evidence bundle: plan,
memo, canary rows, replay instrument, operator source) returned REJECT on
registering V4 (iterations + tail_only 0.5) as the next-launch default.
Twelve objections, most-severe first; #1 verified against the operator
source by direct read:

1. CRITICAL, VERIFIED: tail-only is not one-sided after mean-centering.
   The operator computes shaped = 1 − β·q then subtracts the group mean;
   a non-tail valid rollout gains A = A(β=0) + β·Σq/4 whenever any
   sibling is penalized. Cheapness IS paid post-centering; zero-delegation
   rollouts share the gain; invalid rollouts' penalty is diluted by the
   same amount. Explains V1's replay failure. The plan's "cheapness earns
   nothing" claim (lines 45–47) is false for the implemented operator.
2. CRITICAL: attribution declared resolved before the pre-registered
   comparison exists (control@200 + offline 2×2 pending; canary stability
   only through 160; fence contrast is c160-vs-t200, not matched-step).
   Gate 3's RESOLVED status in this memo is withdrawn to PROVISIONAL.
3. CRITICAL: replay dataset covers control steps 1–92 only — outside the
   120–200 failure regime — and is a time-of-execution glob, not a frozen
   registered dataset.
4. HIGH: residual BCP gradient (−2.64) unquantified as benign; V3
   dominates V4 on both reported environments; no uncertainty reported.
5. HIGH: iterations-only relocates pressure onto multi-turn operation;
   dA/d(iterations) never reported; suppression-of-turns is the same
   failure class one level up.
6. HIGH: turns/wall-time preservation is asserted, not evidenced — the
   9.0-vs-12.8 exhibit was produced under the combined objective and
   cannot identify what iterations-only preserves.
7. HIGH: replay corner metrics cannot see the corner — lowsub_wins means
   subs ≤ 1 with a UNIQUE top; tail-only creates tied tops that include
   zero-sub rollouts and are excluded from the statistic.
8. HIGH: "unchanged signal mass" is conditional on fired groups; V3/V4
   fire MORE often, so cumulative shaping dose can rise while the
   conditional mean stays flat.
9. MED-HIGH: group-size-4 degeneracies — with exactly 2 valid rollouts
   tail-only ≡ symmetric; min-max endpoints trivialize τ; single outliers
   set the normalization.
10. MED-HIGH: the 20-step pilot cannot detect a failure that develops
    after step 120, and a 1-rep canary is underpowered (±9–11 pt) with no
    stop rule registered.
11. MED: replay validity gate is not bit-exact with the live operator
    (missing max_turns + dead-worker fatal classes).
12. MED: the plan document still registers/instructs the superseded
    subs-priced+floor design in Phases 1–2; not implementation-grade.

Reviewer's constructive hypothesis: a two-sided constrained/deadband
objective — no reward for abstinence, a moderate delegation region left
unpriced, a hinge cost only on sub-call volume above a pre-registered
budget, plus an aggregate delegation-rate lower constraint — validated
over the late-collapse horizon with pooled canaries and an explicit stop
rule.

CONSEQUENCE FOR THIS MEMO: the freeze recommendation is WITHDRAWN. Before
any freeze: (a) Gate 1 completes (control@200 + sweep + offline 2×2);
(b) replay re-run on a FROZEN dataset covering steps 120–200 of both
arms, bit-exact validity, reporting dA/d(iterations) and dA/d(subs),
exact zero-sub top-advantage incidence including ties, firing-weighted
total dose, and dispersion; (c) candidate set widened to {V3, V4,
deadband/hinge}; (d) pilot redesigned to cover the late-collapse horizon
with pooled canaries and a registered stop rule; (e) plan document
reconciled. The mitigation DIRECTION (stop pricing delegation volume;
protect the transferable operation layer) is unchallenged by the review;
the specific mechanism and its evidence base are what failed.

## ADDENDUM 2 (2026-07-16) — rollout forensics: the target is unnecessary sub-calls, not volume

Bucketed analysis over ~750 pooled canary rollouts (per-policy × sub-call
volume buckets: 0 / 1–5 / 6–30 / >30):
- Base and all control checkpoints: zero-delegation rollouts are as
  correct or more correct than 6–30-call rollouts (e.g. base 0.43 vs
  0.33; c120 0.38 vs 0.25). Marginal calls above ~5 do not convert.
- treatment@120's 1–5 bucket is the best cell measured (0.57 correct,
  vs 0.42 for its zero bucket): few calls, converted — the productive
  delegation band.
- >30-call rollouts: 25–29% (base, c120) spend most calls on
  near-identical invocation lines (duplicate-call share > 50%); they
  finish without cap-out and without accuracy — redundancy, the RLM
  paper's weak-model failure mode (calls issued, returns unused).

Design consequences: (a) supports the deadband/hinge candidate — the
moderate band must remain unpriced (it is where delegation value lives),
excess volume above a registered budget is where marginal value is ~zero;
(b) adds a candidate cost basis: UNIQUE (dedup-counted) sub-calls, pricing
redundancy directly rather than volume — requires one new env telemetry
channel (rlm/training scope), validatable offline from existing
transcripts before any run; (c) reporting language: "beyond a small
budget, sub-calls stop converting into correctness, and the tail is
dominated by measurable redundancy" — task-tied delegation is explicitly
honored.

## ADDENDUM 3 (2026-07-17) — design consult: zero-neutralized waste pricing

Second adversarial consult (Codex, post-control evidence bundle: corrected
validation pass, offline final table, compute-matched result, pairs
sweep-style finding, redundancy forensics). Substance:

1. NO scalar cost family survives mean-centering alone (ΔA_i = β(q̄−q_i)).
   Registered fix: a ZERO-NEUTRALIZATION transform shared by all candidates
   — zero-subcall valid rollouts receive the mean penalty of their
   delegating siblings, making abstinence exactly advantage-neutral while
   productive delegators still compete. (Raw hinges paid zero-callers in
   82–84% of fired groups; corner|zero up to 0.96.)
2. Pure unique-subs demoted: pricing distinct calls makes REPEATS free —
   the wrong incentive for a near-duplicate-dominated tail. The telemetry
   still matters: redundancy R = S−U becomes the priced quantity.
3. Recommended family: per-turn log waste
   C_PT = I + λ*·log(1+Σ_t[R_t+(U_t−b)+]), b=5 (edge of the measured
   productive band), λ* span-calibrated on the frozen 1–99 set and held
   fixed. Ranking: C_PT > C_RW (rollout-level waste) > C_LH (volume
   log-hinge) > C_U > V3 > V4 > raw hinges. Turns stay priced; additive
   joint costs only (ratios gameable via dummy turns).
4. Self-declared failure mode: the productive sweep size is
   domain-dependent — C_PT could mislabel legitimate breadth as waste on a
   new family → per-environment ranking mandatory in the decision pass;
   never-trained canaries mandatory in the pilot.
5. Six additional decision-pass metrics required (multivariate FE slopes,
   bucket redistribution with a zero-ΔA assertion, productive-vs-zero
   contrast, span attribution by k_valid, dedup audit, cost-preservation
   proxy vs V0).

Practical sequencing: the 120–200 decision pass ranks the
computable-from-dumps cells (V0/V3/V4 references + zero-neutralized C_LH
with frozen λ) — the waste family (C_PT/C_RW) requires a new env telemetry
channel (rlm/training scope) and enters via offline validation before the
pilot. Instrument extension in progress; decision data still pending (tar).
