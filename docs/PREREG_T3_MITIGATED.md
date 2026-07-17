# PREREG — t3-mitigated run (frozen at sign-off, 2026-07-17)

Dated, append-only. Amendments only as dated addenda; nothing below is
edited after launch.

## Objective (the only change vs t2-treatment)
cost = I + 2 * ln(1 + max(0, S - B)),  implemented as log1p; B = 5.
- Zero-neutralization transform: every valid zero-subcall rollout receives
  normalized cost = sum(p of valid delegating siblings)/(groupsize - m);
  its advantage increment vs beta=0 is exactly 0. All-zero-valid groups:
  q = 0 (no shaping). Symmetric shaping; no tail_only.
- lambda = 2: reasoned constant (two iterations per e-fold of excess),
  bracketed by the equal-voice evaluations of both measured regimes
  (1.86 / 2.73), verified at exactly 2 on the frozen decision set. FROZEN
  for the run. Shadow rule: span-matching ratio recomputed per 640-group
  window on the run's own data; shadow outside [0.93, 3.71] => STOP,
  re-register lambda, restart. Never adapted in-run.
- B = 5: productive-band rule (largest sub-call bucket with positive
  marginal correctness gain on the frozen calibration set); {3,5,8} shown
  behaviorally equivalent.
- Everything else identical to t2: beta_max=0.15, solve_floor=0.25,
  min_span=1, validity gate, correctness dominance, group size 4, same
  training mixture, same 200-step schedule, same eval cadence.

## Launch gates (must pass before GPU)
G0. Operator implementation passes the extended invariant suite (zero-
    neutrality 1e-9, budget-freeness S in 0..5, dominance, beta=0
    reduction, legacy suite green).
G1. Offline launch gate on the frozen 120-200 manifest, common support,
    1000 group bootstraps: upper95(b_subs)<0; upper95(b_iters)<0;
    upper95(cost_pres)<0; prod_vs_zero>0; dose/total<=V0;
    corner|zero<=V0. Any failure blocks launch.
G2. Dispatcher deadline sweep present in the kuvalar prime-rl runtime
    (user-applied patch; verified by grep before launch).
G3. User go on the staged diff + this document.

## In-run monitoring and kill schedule (ai16, ~1-1.5 h/step)
- Per-batch: zero-invariance audit; any valid zero-call |dA|>=1e-9 =>
  ABORT (mechanism bug).
- Per-step advisor loop vs control's complete logged history (wandb).
- Every 20 steps: in-run evals oolong + bcplus + codeqa-transfer-canary
  (n=25); canary records delegating share + finalization + cap rates.
- Per 640-group window (~80 steps): the six G1 conditions recomputed on
  the live window + shadow-lambda range check. First window ~step 80 =
  first formal in-run kill point. Two consecutive passing windows required
  before any expansion decision.
- STOP RULES (any):
  S1 zero-invariance violation (immediate);
  S2 shadow-lambda outside [0.93, 3.71];
  S3 window gate failure (any of the six);
  S4 transfer canary: delegating share declining beyond noise across 3
     consecutive eval points, or below 20% at any point (t2's terminal
     share; base band is ~38);
  S5 premature-finalization fingerprint: no-final or cap rates exceeding
     2x control's matched-step values across 2 consecutive eval points, or
     accuracy-conditional-on-low-turns collapsing;
  S6 non-inferiority early warning: guardrail eval below control's
     matched-step value by >eps=0.05 across 4 consecutive eval points
     (both trained envs simultaneously).
- CHECKPOINT SELECTION: final artifact = argmax over saved checkpoints of
  guardrail eval accuracy subject to canary-clean (delegating share within
  noise of the base band). Fixed-length completion is NOT the success
  criterion.

## Predictions (falsifiable)
P-1 Within-run reward oscillation structure comparable to control's
    (schedule-driven), no monotone late decline.
P-2 DECISIVE: transfer canary delegating share stays within noise of the
    base band (~38 +- 10) at EVERY eval point including 200.
P-3 Cost: turns at-or-below t2's trajectory; sub-call volume above t2's
    (the budget un-prices the productive band) but below control's;
    waste tail (>30-call share) below control's.
P-4 Non-inferiority: guardrail accuracy within eps=0.05 of control's
    matched-step values at 200 on both trained envs.

## Failure readings (pre-committed)
- P-2 fails => the corner is not purely incentive-driven at this pressure;
  escalate to process-level operation rewards; do not re-tune lambda/B
  post-hoc on this run's data.
- P-4 fails with P-2 passing => "any operation pricing at beta=0.15 trades
  in-distribution accuracy"; next lever is beta, not cost shape.
- S5 fires => register the iterations deadband (productive-band rule) for
  the next design; do not patch mid-run.

## Upgrade path (registered, gated)
C_PT (per-turn waste) is a separately calibrated challenger: requires the
env dedup/per-turn telemetry channel, 100% coverage, R_t = S_t - U_t
reconciliation, its own frozen lambda, and independent passage of G1-style
gates. No automatic substitution.

## ADDENDUM (2026-07-17, pre-launch) — G2 amended by user decision
Dispatcher deadline sweep NOT ported (user: launch without). Compensating
controls: (a) launcher's built-in 8-attempt auto-resume loop from last
checkpoint (interval already 5 steps — max ~5 steps redone per wedge);
(b) advisor-loop stall detection (no step progress outside eval barriers
=> restart via tmux); (c) PRIME_RL_ROLLOUT_TIMEOUT_S exported but known
unconsumed on this runtime (documented no-op). Wedge risk accepted as
recovery-cost, not run-loss.
