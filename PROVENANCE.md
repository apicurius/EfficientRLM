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

- **`prime-rl/` is frozen — no modifications, period** (owner decision 2026-07-04).
  Treatment/behavior changes go through the per-environment advantage hook, environment
  config, or wrappers/env vars only.
- **Core `rlm/` (outside `rlm/training/`) is frozen** with exactly ONE accepted deviation:
  the prompt budget adaptation in `rlm/rlm/utils/prompts.py` (see table below). No further
  core changes are accepted.
- Record every deviation from upstream in this file (component, commit, what, why).

## Deviations from upstream

| Component | What | Why |
|---|---|---|
| `rlm/rlm/utils/prompts.py` | 3 budget hunks (user-directed, 2026-07-04): ORCHESTRATOR_ADDENDUM per-prompt ceiling "~100K characters" → "~12K tokens (roughly 36K–48K characters)"; brute-force heuristic "~20 × 100K chars" → "~20 × 12K-token chunks"; metadata line "~100k tokens at once" → "keep under ~12k tokens". Byte-identical to the fork the live A/B run trained on. | The training scaffold (`rlm/training/src/rlm_train/proxy.py`) rejects sub-prompts above 12k estimated tokens / 36k chars; upstream's capacity claims steer the policy into deterministic rejections (reject-loop). Prompt guidance must match enforcement. |

## Repository layout

This is a **single vendored git repository** (the two upstreams' nested `.git`
directories were removed after verification). The **initial commit is the pristine
upstream baseline** — so `git diff <initial-commit>..HEAD` is the exact, complete
record of every EfficientRLM modification to upstream. That audit anchor is the whole
point: provenance is recoverable from `git log`, not from memory.
