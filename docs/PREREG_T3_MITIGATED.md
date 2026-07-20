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

## Amendment A1 (2026-07-17, pre-step-5, user-directed)

The free-band threshold is REMOVED. Registered cost basis becomes:

    cost = rlm_iterations + 2 * ln(1 + rlm_sub_llm_calls)      (B = 0)

All other parameters unchanged: lam=2.0, zero_neutralize=true, beta_max=0.15,
solve_floor=0.25, gamma=1.0, min_span=1.0. Config-only change (B is a kwarg);
zero code modifications; the adaptive_group.py implementation is byte-identical
to commit 43c9346.

Rationale and evidence (frozen t2-control 120-200 window, ADVISOR.md #154-155):
1. Advantage-space audit under zero-neutralization shows small delegators
   (1<=S<=5) remain NET-FAVORED at B=0 (mean dA +0.0076, 27% pushed down, vs
   +0.0119 / 20% at B=5): the abstention guard is zero-neutralization (exact),
   not the threshold. The earlier raw-cost +2.56 flip does not survive
   translation into advantage space.
2. Threshold immateriality: {3,5,8} and structure-derived per-env budgets all
   gate-equivalent; removing it simplifies the objective and preserves
   cross-policy comparability (no budget concept foreign to the released
   scaffold).
3. lam=2 retained by the span-matching rule (window estimates 1.86-2.76);
   lam=1 shown to reduce the basis to t2's original (rank-tracks iterations
   0.88 vs sub-calls 0.41 -- the diagnosed imbalance); functional form (ln vs
   ln^2 vs hybrid) proven immaterial after span matching (group min-max
   normalization absorbs shape; S>100 tail gets strongest pressure, 72% down,
   under every form).

Run consequence: the B=5 attempt (launched 13:39, <=1 training step + step-0
eval) is DISCARDED (outputs/qwen3-30b-t3-b5-discarded-20260717); training
restarts from step 0. All gates G0-G3, stop rules S1-S6, and predictions
P-1..P-4 apply unchanged to the restarted run. S1 zero-invariance audit is
unaffected (zero-neutralization unchanged). The first-5-free spot-check in the
advisor loop is superseded: rollouts with S=0 should show cost == iterations;
any S>0 rollout shows cost == iterations + 2*ln(1+S).

### A1 code note (2026-07-17, post-launch)
The operator implementation was simplified to the thresholdless registered form:
the B parameter and the max(0, S-B) hinge were removed from
_cost_iterations_ln_excess, _cost, and adaptive_group_advantage; the config's
B kwarg was removed accordingly. Behavior is bit-identical to the launched
configuration (B=0 made the hinge the identity; formula equivalence verified at
S in {0,1,3,5,20,800} to machine precision, and the 9-test suite passes on the
new form). The live process holds the launch-time code in memory; any auto-resume
loads this simplified code with the matching config, computing the same numbers.

## S3 adjudication and Amendment A2 (2026-07-19, step-80 boundary, USER-RATIFIED)

At the step-80 window gate (steps 1-79, canonical variant: 563 complete groups,
18.8% partial-group prune disclosed, frozen lambda=2 enforced), S3 FIRED on
condition 4 alone: prod_vs_zero_paired point -0.0053, bootstrap CI
[-0.0216, +0.0105], n=11 qualifying pairs. Conditions 1,2,3,5,6 PASSED with
margins; shadow-lambda 1.5453 inside [0.93, 3.71] (S2 PASS).

RULING (user-ratified, canary-conditioned as pre-agreed): CONTINUE. Grounds:
(a) condition 4 is a point-sign rule on n=11 whose CI contains the value it was
registered as passing with (+0.010, itself with a zero-crossing CI at n=7);
(b) the mechanism guarantee it proxies (abstention never advantaged) is enforced
bit-exactly by the every-step zero-invariance audit, unviolated across ~2,600
rollouts; (c) the independent transfer canary printed CLEAR at the same boundary
(delegation share flat at .28 for three consecutive prints, accuracy .40
joint-best, benign zero-sub phenotype), and all BC+ tripwires cleared with
best-ever accuracy.

AMENDMENT A2: condition 4 is reformulated from a per-boundary point-sign stop
rule to (i) a per-boundary DESCRIPTIVE line (point + 95% CI + n, reported at
100/120/140/160/180), and (ii) a single registered RULING at step 200 on the
accumulated full-run qualifying pairs (expected n ~ 25-30), failing only if the
95% CI excludes zero on the negative side. S3 continues to govern conditions
1,2,3,5,6 unchanged. Artifacts: outputs/advisor/gate_t3_79/.

## Amendment A3 (2026-07-19, step ~87, pre-outcome, USER-RATIFIED): expanded offline replication on the contract environments

Registered BEFORE any t3 offline result exists. The post-run offline evaluation
expands replication where the no-regression ruling is decided:

- CONTRACT ENVIRONMENTS (OOLONG-trec, BrowseComp+): SIX independent repetitions
  for t3-final and control-final (was three). Expected effect: per-cell clustered
  SE shrinks from ~0.06 to ~0.04, difference SE from ~0.08 to ~0.055, giving the
  eps=0.05 no-regression margin a decision band comparable to its own width.
- Transfer suites (CodeQA, OOLONG-Pairs) and all other policies (base, released,
  step-120 checkpoints): three repetitions, unchanged.
- Protocol otherwise identical: one machine, one serving stack, same env configs,
  same scorers, pooled with question-clustered SEs. Runtime re-measured for ALL
  policies in the same pass (ai16 scale; studio-era absolute times not spliced).

Rationale: control's contract-env cells (.500 trec, .431 BC+) are consistent
across existing reps (not single-rep luck: trec per-rep .52/.46/.52), so the
binding constraint on the ruling is difference-SE, not point instability.
Registered now to preclude post-hoc repetition-shopping optics.

## Amendment A4 (2026-07-19, pre-outcome): sub-call tail metrics join the offline reporting set

The offline evaluation reports, per suite and per policy, in addition to the
existing D_ops set (mean iterations, mean sub-calls, p95 scaffold cost, fatal
rate): sub-call quantiles p90/p95/p99 and maximum (no fixed threshold — the
distribution reported as such, with the CCDF figure carrying the full shape),
plus one relative share: the fraction of rollouts exceeding the BASE policy's
per-suite sub-call p95 ("share above the untrained model's own worst 5%").
The base-relative threshold is task-calibrated and carries no arbitrary
constant; a single fixed cutoff cannot serve suites whose base p95 differs by
an order of magnitude (trec 3,209 vs BC+ 378 in the existing data). Rationale: the deployment motivation (ch01) is the delegation tail
specifically; the scaffold-cost p95 under-expresses it because the log term
compresses extreme sub-call counts (a 16,722-call rollout and a 494-call rollout
differ by ~3.5 cost units). Registered before any t3 offline result exists;
applies to every policy in the pass (base, released, control, t3, step-120
checkpoints) so the released policy's tail (trec p95 3,593, max 16,722 in the
existing three-rep data) is reported on the same footing. Thesis surface:
tbl:res-tail gains these rows (or a companion sub-call tail table) at population
time; the CCDF figure (fig:tail-ccdf) already visualizes the full distributions.

## S4 adjudication annex (2026-07-20, written BEFORE the step-120 print; USER-DIRECTED)

Context: the canary delegating share reached exactly 0.20 at step 100 — the fire
threshold's boundary value (chosen as t2's terminal share) — with the benign
phenotype at its strongest (zero-sub rollouts: 12 REPL iterations median, cap
0.05, accuracy 0.40 = canary joint-best; both contract-env evals at run records).
The share rule and the pathology it proxies have visibly decoupled.

Pre-committed adjudication for step 120 and after:
1. The letter stands: a print below 0.20 FIRES S4. No post-hoc reinterpretation
   of the threshold.
2. FIRING NEVER STOPS TRAINING AUTOMATICALLY. A fired rule = loud escalation
   (terminal + push notification) and continued training while the user
   adjudicates. Training stops ONLY on explicit user confirmation (standing
   user directive, 2026-07-20).
3. The adjudication evidence is pre-named: (a) zero-sub phenotype panel
   (iterations / REPL activity / cap rate), (b) canary accuracy vs base,
   (c) BC+ delegation-share coupling. All-benign -> expected ruling is
   documented-continue with the claim-4 rescope (delegation SHARE not
   preserved on far transfer; delegation CAPABILITY and accuracy preserved)
   and formal promotion of the delegation floor in the next-run design.
   Any malignant marker (cap rising, accuracy below base, REPL work absent)
   -> stopping becomes the default recommendation put to the user.
