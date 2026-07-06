# Pre-registered post-A/B amendment: advantage-layer discovery credit

Registered 2026-07-06, BEFORE the next-gen A/B arms run (control launching
today, beta_max=0.0; treatment beta_max=0.15). Nothing in this amendment is
active during the A/B: the arms run the committed `adaptive_group` unchanged.

## Problem (measured)

Binary terminal reward leaves all-wrong groups with zero advantage: on the
multienv-20 smoke, 42% of BC+ groups (31/74) and 8% of oolong groups carried
no gradient. Dissection shows the waste is not operational (0 groups died of
coordinated overflow) but selectional: 32-39% of dead groups contain a sibling
whose transcript SURFACED the gold answer without selecting/finalizing it
(12/31 BC+-only early measurement; 12/37 both-env full-smoke measurement).

## Proposed mechanism (advantage layer only — reward untouched)

In `adaptive_group_advantage`, for groups with zero correct rollouts (today:
all shaped = 0, zero-advantage filtered), set

    shaped_i = DELTA * discovered_i          (DELTA = 0.2)

where `discovered_i` = the gold answer string appears in rollout i's
transcript (computable offline from saved nodes; if adopted in-run, a ~25-line
telemetry digest restores the in-run metric). Group-mean centering then
creates within-dead-group gradient toward finalize-what-you-found — the
selection/synthesis behavior that is now fully inside the trainer's loss
window (verified 100% credit reach on the naked env).

## Dominance guarantee (P1-style)

DELTA < 1 - beta_max ensures every correct rollout in any group outranks every
discovered-but-wrong rollout in any group: correct-valid shaped >= 1 -
beta_max = 0.85 > DELTA = 0.2 > 0 = undiscovered-wrong. The lexicographic
ordering (correct > discovered-wrong > undiscovered-wrong) cannot invert
correctness. The credit fires ONLY in groups where no correct rollout exists,
so it never interacts with the cost lever (which requires >= 2 valid).

## Expected effect size (from the smoke)

Rescues ~32-39% of dead groups: 48 rollouts (of 608) would have gained
gradient at DELTA=0.2 in the 19-step smoke. On BC+ that converts ~13-17% of
all groups from waste to selection-teaching signal.

## Evaluation plan

Post-A/B: replay `tools/discovery_credit_offline.py` on both completed arms
to quantify the prize on 200-step data; if adopted, it enters as a THIRD arm
or a follow-up run, never retrofitted into the A/B comparison. Guard against
gaming (transcript stuffing of candidate strings) by requiring the discovery
string to be non-trivial (len >= 3, judge-verified variant optional) and by
the oracle rescorer.
