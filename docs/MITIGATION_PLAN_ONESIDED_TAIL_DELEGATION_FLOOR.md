# Mitigation plan: one-sided tail penalty + delegation floor

**Status: PLAN ONLY (2026-07-15). No code changes now.** Nothing here touches
the live control run, the frozen contract, or the current tree. Execution is
gated on Phase 0. Conditional trigger: the dose-feedback diagnosis of the
treatment's step-100–139 reward dip (ADVISOR.md #118) plus the CodeQA
delegation collapse (@200).

Both changes live entirely in `rlm/training/src/rlm_train/adaptive_group.py`
(the legal seam) plus its config surface and test suite. prime-rl and rlm
core stay untouched, per the standing rule.

---

## Phase 0 — Preconditions (all must hold before any edit)

- [ ] Control arm reaches step 200; primary endpoint + sweep complete
      (SWEEP.md). Nothing changes while any pre-registered measurement is
      pending.
- [ ] Counterfactual read: control's reward through steps 100–139. Flat ⇒
      dose-feedback diagnosis confirmed, mitigation 1 justified as designed;
      dips like treatment ⇒ retract the feedback diagnosis, re-rate the table
      (mitigation 2 remains justified by the collapse alone).
- [ ] Collapse attribution from the control offline leg (codeqa 2×2,
      control@120/@200). Lever-attributed ⇒ mitigation 2 is mandatory for the
      next launch; duration-attributed ⇒ re-rate.
- [ ] User sign-off on the design-freeze parameters of Phase 1.

## Phase 1 — Design freeze (pre-register BEFORE implementation)

Fix and write into a new PREREG document (same discipline as
PREREG_LONGBENCH_DOCQA_MINI.md — dated, append-only):

1. **Tail threshold τ** for the one-sided penalty. Registered default:
   rel_cost > 0.5 (fixed cutoff on the min–max-normalized cost, i.e. the
   expensive half of the valid-sibling span). Alternative considered and
   rejected in the prereg: group median (ill-defined at group size 4 with
   ties; fixed cutoff composes with the existing min_span=1 floor).
2. **Floor form** for the delegation floor. Registered default: a new cost
   basis `iterations_log_subcalls_floor` = iterations + log1p(subcalls) +
   λ·1[subcalls = 0], with λ = 1.0 (one iteration-equivalent surcharge —
   same unit as the Δ_min span floor). Zero-delegation therefore never ties
   with, and never undercuts, a one-sub-call sibling. λ scope, explicit:
   λ = 1 does NOT rank zero-sub above heavily-delegating siblings
   (1.0 < log1p(5) ≈ 1.79) and is not meant to — under the τ = 0.5
   tail-only shape, cheapness earns nothing, so the corner's PULL is
   removed by τ; λ adds the PUSH away from zero in near-tie groups. A hard
   "zero-sub = most expensive" rule is rejected: zero-sub-correct rollouts
   exist legitimately on the trained families (3–14% zero-sub share on
   OOLONG offline).
3. **Predictions** (falsifiable, before any run):
   - P-1 give-back amplitude: peak-to-trough train-reward oscillation at
     matched dose shrinks vs the t2 treatment's 4–7 points.
   - P-2 corner closed: zero-delegation share on never-trained probes
     (codeqa protocol) stays within noise of the base model at every
     checkpoint, including final.
   - P-3 tail preserved: p95 cost and fatal-rate reductions at or better
     than t2 treatment levels (the one-sided lever keeps full tail pressure).
   - P-4 no dominance leak: eval accuracy non-inferior per the ε=0.05 rule.
4. **Failure readings**: P-2 fails ⇒ the corner is not incentive-driven;
   escalate to process-level operation rewards (filter/truncation-solutions
   registry). P-3 fails ⇒ the one-sided shape lost too much signal; fall
   back to symmetric shaping + floor only.

## Phase 2 — Code changes (described, NOT applied)

All in `adaptive_group.py`; both stateless and group-local; the advantage
remains a pure function of the group (the ch03 property is preserved).

1. **New cost basis** (mitigation 2): add `_cost_iterations_log_subcalls_floor`
   to `_COST_BASES` — the existing default plus the λ surcharge at
   subcalls = 0. Registry semantics unchanged (unknown names still raise);
   `DEFAULT_COST_BASIS` untouched — arms opt in via config. This mirrors how
   the four existing bases are registered; the reported cost-bases table
   (tbl:cost-bases) gains one row at population time.
2. **One-sided shaping** (mitigation 1): a new keyword on the advantage
   entry point (pattern-matching `min_span`), e.g. `tail_only: float | None`
   — when set, `rel_cost` is replaced by `max(0, rel_cost − τ)/(1 − τ)`
   before entering `shaped = 1 − β·rel_cost`. τ = the registered 0.5.
   Consequences to verify in tests: all non-tail valid-correct rollouts get
   shaped exactly 1.0 (advantage-identical); the [0.85, 1] band and the
   unconditional validity gate are unchanged; the min_span dead-zone branch
   is evaluated on the ORIGINAL span (the floor still gates re-ranking, the
   transform applies after gating).
3. **Config surface**: two new per-env args in the training TOMLs
   (`cost_basis = "iterations_log_subcalls_floor"`, `tail_only = 0.5`),
   identical across both arms of any future pair — the twin invariant stays
   a two-value diff (β_max only).
4. **Telemetry**: the per-rollout metric record already logs cost, rel_cost,
   shaped, β; add the post-transform rel_cost so the verifier can recompute
   bit-exact (invariant 4's 1e-9 check must target the new formula).

## Phase 3 — Tests and invariants (before any GPU time)

- [ ] Extend `tests/test_adaptive_group_advantage.py`: correctness dominance
      under the new shape (all correct ≥ 0.85 > 0 invalid); non-tail
      advantage equality; zero-delegation never cheapest under the floored
      basis; β=0 exact reduction to validity-gated correctness (the twin
      guarantee) with BOTH new knobs active; min_span interaction.
- [ ] Rerun the offline replay (the 933-group precedent) with the new knobs:
      bit-exact recompute via the rollout verifier's invariant 4 path.
- [ ] verify_rollouts lever gauge updated for the new rel_cost definition.

## Phase 4 — Pilot (cheap, before any 200-step run)

- [ ] 20-step smoke on the standard multienv config (the run_smoke_multienv20
      pattern), treatment knobs on: confirm lever fires, authority in band,
      no invariant alerts, zero-delegation share on train stream not rising.
- [ ] Offline probe at the smoke checkpoint on codeqa (n=50, one run,
      descriptive): zero-share vs base.

## Phase 5 — Launch (next pair, not a patch to t2)

- New matched pair from base, both arms carrying identical new knobs,
  β_max=0.15 vs 0 — the mitigation is part of the recipe, the lever remains
  the only difference. The t2 pair is never re-run or amended; its results
  stand as the dose-feedback and collapse evidence that motivated this plan.
- Reporting touchpoint (only after data): the future-work registry already includes the
  delegation floor; the one-sided variant extends the same paragraph. No
  reporting edit before results exist, per the errata discipline.

## AMENDMENT 2026-07-15 — adversarial replay verdict (evidence, no training)

`scripts/11_mitigation_replay.py` replayed five objective variants over 808
complete real groups from the control dumps (steps 1-92; counterfactual
beta from each group's own solve rate). Key metric: within-group cov(A,
sub-calls) among valid-correct — the anti-delegation gradient that is the
transferable prior behind the codeqa collapse (the treatment's own
trajectory corroborates the proxy: subs fell 4.6x while turns fell 1.2x,
exactly what a subs-priced gradient predicts).

Findings (dA/dsubs x1e4, oolong / bcp):
- V0 today: -8.74 / -4.42.
- V1 tail-only tau=0.5: -8.35 / -4.66 — **FAILS generalization**: the
  gradient is carried by the cheap-vs-tail contrast and the tail IS the
  high-delegation side; removing cheap-half ordering barely touches it.
- V2 +floor lambda=1: -6.51 / -4.41 — adjunct at best.
- V3 iterations-only basis: +1.50 / -2.35 — anti-delegation prior GONE on
  oolong, halved on bcp (residual = turns-subs correlation).
- V4 iterations-only + tail tau=0.5: +1.39 / -2.64, strict cheapest-wins
  0.44, low-sub winners 0.05 (vs 0.77/0.16 today). Signal mass (mean |A|)
  unchanged in all variants; V3/V4 fire in MORE groups (turns-unit spans
  clear Delta_min more often).

REVISED registered default: **cost_basis = "iterations" (already in the
registry) + tail_only = 0.5**. The floor is moot under an unpriced subs
term (zero-delegation earns nothing and costs nothing; corner attraction
measured at 0.05). The subs-priced+floor design of Phase 1 is demoted to
fallback.

Disclosed trade-off: iterations-only stops pricing delegation volume, so
the next run's sub-call savings may shrink to what turn compression
implies; the turns channel (the C2 evidence) and wall-time follow-through
are preserved. P-3 is re-scoped accordingly at prereg time; P-2 (transfer
zero-share at base level) stays the decisive probe. Replay limits, stated:
control-policy group composition, first-order proxy, no RL dynamics — the
pilot still gates.

## Hardware decision (user-agreed 2026-07-15)

- Phase-4 pilot (20-step smoke + codeqa zero-share probe): **ai16**,
  unconditionally — free held GPUs, all instruments local, speed immaterial
  at 20 steps.
- Full 200-step mitigated pair: **defaults to ai16** (both arms on the same
  hardware — a self-contained pair, cleaner than t2's mixed history), at
  ~7-11 days wall time; the studio remains the option if the calendar
  demands ~1-2 days. Prerequisites for an ai16 run: (a) port the dispatcher
  deadline sweep into the kuvalar runtime (the stuck-rollout defense
  provably absent there; documented gap), (b) re-scope P-1 from
  amplitude-vs-t2 (hardware-confounded: machine changes shift train-reward
  levels +0.12-0.17, measured) to within-run oscillation structure.
- Offline evaluation of the resulting adapters stays on the studio-class
  rig regardless, for canon comparability.

## Rating context

Scores and criteria behind this plan: session table of 2026-07-15
(one-sided 8.2, floor 8.0, γ-sweep 7.0, EMA held in reserve). The two chosen
items fix the incentive (no reward for minimal operation) and the landscape
(minimal is not cheapest) respectively; γ sweep stays in the future-work
registry as an independent ablation; EMA smoothing only if oscillation
persists after the shape fixes.
