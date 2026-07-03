# EfficientRLM — pristine upstream baselines

Created 2026-07-03. This directory holds **unmodified upstream** copies of the two
dependencies, cloned directly from their public remotes and pinned to exact commits.
The purpose is a provenance-clean starting point: all EfficientRLM changes should land
as **tracked commits / patches on top of these baselines**, so "what we changed and why"
is always recoverable (unlike the earlier fork, whose undocumented edits led to an
inaccurate "byte-for-byte unmodified" claim).

## Contents

| Component | Upstream remote | Pinned ref | Commit | Verified pristine |
|---|---|---|---|---|
| `rlm/` | https://github.com/alexzhang13/rlm.git | commit `156fd72` | `156fd725411b9cae822f5920a6cbf102a5473baa` | clean tree; `training/src/rlm_train/env.py` = 335 lines, **no** root/sub prompt budget or windowing (released baseline); `rlms` 0.1.2 |
| `prime-rl/` | https://github.com/PrimeIntellect-ai/prime-rl.git | tag `v0.6.1.dev14` | `d507aeafc0afaf365f6cdb5b9f791fe686a3a3cb` | clean tree; dispatcher has **no** `PRIME_RL_ROLLOUT_TIMEOUT_S` (i.e. lacks the local eval-zombie patch); shallow clone of the tag |

Both pinned commits match the exact baseline the prior work (`.research/ERLM-main`)
was built on, minus the local modifications — so this is the same baseline, verified clean.

## Rules

- **Never modify `prime-rl/` source in place.** Treatment/behavior changes go through the
  per-environment advantage hook, environment config, or wrappers/env vars. If a trainer
  change is unavoidable, keep it as a named, committed patch on top of `v0.6.1.dev14` and
  document it here.
- Record every deviation from upstream in this file (component, commit, what, why).

## Repository layout

This is a **single vendored git repository** (the two upstreams' nested `.git`
directories were removed after verification). The **initial commit is the pristine
upstream baseline** — so `git diff <initial-commit>..HEAD` is the exact, complete
record of every EfficientRLM modification to upstream. That audit anchor is the whole
point: provenance is recoverable from `git log`, not from memory.
