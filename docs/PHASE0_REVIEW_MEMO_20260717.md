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
