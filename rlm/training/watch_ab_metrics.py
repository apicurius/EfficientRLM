#!/usr/bin/env python3
"""Cron-safe metrics watcher for the control arm: rewards, entropy, grad norm.

Disk-first (trainer.log + run_default/rollouts/step_N/train_rollouts.jsonl are
canonical; wandb is a cross-check surface, not a dependency). Appends alerts to
<OUT>/VERIFY/METRICS_ALERTS.log and a one-line snapshot per invocation to
<OUT>/VERIFY/METRICS_STATUS.log. Stateless across steps except a small JSON
state file used to avoid re-firing the same alert.

Thresholds (calibrated 2026-07-07):
- grad norm: healthy 30B-LoRA band 0.02-0.07 (gradnorm-early-signal rule);
  WARN > 0.09, ALERT > 0.12 (launch tripwire), TREND alert on sustained rise
  because upward drift is the earliest instability signal.
- entropy: attempt-1 baseline ~0.40 at step 20; WARN < 0.25 or > 1.0,
  ALERT < 0.15 (collapse), WARN on drop below 50% of rolling median.
- reward/all/mean: attempt-1 baseline mean 0.39, single-step range
  0.125-0.6875, rolling(3) never below ~0.2; WARN on a 0.0 step,
  ALERT when rolling(3) < 0.10 (dead arm), WARN when rolling(5) falls
  below 50% of the run median after step 10.
"""
import json
import os
import re
import statistics
import sys
import time

OUT = sys.argv[1] if len(sys.argv) > 1 else \
    "/scratch/omeerdogan23/erlm/.research/EfficientRLM/outputs/qwen3-30b-ab-control-multienv-200step"
TRAINER_LOG = os.path.join(OUT, "logs", "trainer.log")
ROLLOUTS = os.path.join(OUT, "run_default", "rollouts")
VERIFY = os.path.join(OUT, "VERIFY")
ALERTS = os.path.join(VERIFY, "METRICS_ALERTS.log")
STATUS = os.path.join(VERIFY, "METRICS_STATUS.log")
STATE = os.path.join(VERIFY, "metrics_state.json")

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(
    r"Step (\d+) \|[^|]*\| Loss ([\d.eE+-]+) \| Entropy ([\d.eE+-]+)"
    r" \| Mismatch KL ([\d.eE+-]+) \| Grad\. Norm ([\d.eE+-]+)"
)


def now():
    return time.strftime("%F %T")


def load_state():
    try:
        with open(STATE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"fired": []}


def emit(alerts, level, tag, step, msg):
    key = f"{tag}:{step}"
    alerts.append((key, f"[{now()}] {level} {tag} step={step} {msg}"))


def trainer_series():
    """[(step, loss, entropy, gradnorm)] — dedupe repeated steps (resume) by
    keeping the LAST occurrence, in step order."""
    seen = {}
    try:
        with open(TRAINER_LOG, errors="replace") as f:
            for line in f:
                m = STEP_RE.search(ANSI.sub("", line))
                if m:
                    s = int(m.group(1))
                    seen[s] = (float(m.group(2)), float(m.group(3)), float(m.group(5)))
    except OSError:
        return []
    return [(s, *seen[s]) for s in sorted(seen)]


def reward_series():
    """[(step, mean_reward)] from train_rollouts.jsonl, skipping steps whose
    JSONL is still being written (mtime < 60s)."""
    out = []
    try:
        dirs = os.listdir(ROLLOUTS)
    except OSError:
        return out
    for d in dirs:
        if not d.startswith("step_"):
            continue
        p = os.path.join(ROLLOUTS, d, "train_rollouts.jsonl")
        try:
            if time.time() - os.path.getmtime(p) < 60:
                continue
            vals = []
            with open(p) as f:
                for line in f:
                    row = json.loads(line)
                    r = (row.get("rewards") or {}).get("reward")
                    if r is not None:
                        vals.append(float(r))
            if vals:
                out.append((int(d.split("_")[1]), statistics.mean(vals)))
        except (OSError, ValueError):
            continue
    return sorted(out)


def check(trainer, rewards):
    alerts = []
    # --- grad norm + entropy (trainer stream) ---
    for i, (step, loss, ent, gn) in enumerate(trainer):
        if step == 0:
            continue  # cold-start step: grad spike expected (0.1475 on this run), not a signal
        if gn > 0.12:
            emit(alerts, "ALERT", "GRADNORM_SPIKE", step, f"grad_norm={gn:.4f} > 0.12 tripwire")
        elif gn > 0.09:
            emit(alerts, "WARN", "GRADNORM_HIGH", step, f"grad_norm={gn:.4f} above healthy band (0.02-0.07)")
        if ent < 0.15:
            emit(alerts, "ALERT", "ENTROPY_COLLAPSE", step, f"entropy={ent:.4f} < 0.15")
        elif ent < 0.25:
            emit(alerts, "WARN", "ENTROPY_LOW", step, f"entropy={ent:.4f} < 0.25 (baseline ~0.40)")
        elif ent > 1.0:
            emit(alerts, "WARN", "ENTROPY_SPIKE", step, f"entropy={ent:.4f} > 1.0")
        hist = [t[2] for t in trainer[max(0, i - 10):i]]
        if len(hist) >= 5 and ent < 0.5 * statistics.median(hist):
            emit(alerts, "WARN", "ENTROPY_DROP", step,
                 f"entropy={ent:.4f} < 50% of rolling median {statistics.median(hist):.4f}")
    # sustained grad-norm rise: last 3 all above band AND mean(last5) >= 2x mean(prev10)
    gns = [t[3] for t in trainer]
    if len(gns) >= 15:
        last5, prev10 = gns[-5:], gns[-15:-5]
        if (all(g > 0.07 for g in gns[-3:])
                and statistics.mean(last5) >= 2 * statistics.mean(prev10)):
            emit(alerts, "ALERT", "GRADNORM_TREND", trainer[-1][0],
                 f"mean(last5)={statistics.mean(last5):.4f} >= 2x mean(prev10)={statistics.mean(prev10):.4f}, last3 all > 0.07")
    # --- reward (orchestrator stream) ---
    rs = [r for _, r in rewards]
    for i, (step, r) in enumerate(rewards):
        if r == 0.0:
            emit(alerts, "WARN", "REWARD_ZERO", step, "mean reward exactly 0.0")
        if i >= 2 and statistics.mean(rs[i - 2:i + 1]) < 0.10:
            emit(alerts, "ALERT", "REWARD_DEAD", step,
                 f"rolling(3)={statistics.mean(rs[i - 2:i + 1]):.3f} < 0.10 (baseline mean 0.39)")
        if i >= 10:
            med = statistics.median(rs[: i + 1])
            roll5 = statistics.mean(rs[i - 4:i + 1])
            if roll5 < 0.5 * med:
                emit(alerts, "WARN", "REWARD_DROP", step,
                     f"rolling(5)={roll5:.3f} < 50% of run median {med:.3f}")
    return alerts


def main():
    os.makedirs(VERIFY, exist_ok=True)
    state = load_state()
    fired = set(state.get("fired", []))
    trainer = trainer_series()
    rewards = reward_series()

    fresh = [(k, line) for k, line in check(trainer, rewards) if k not in fired]
    if fresh:
        with open(ALERTS, "a") as f:
            for _, line in fresh:
                f.write(line + "\n")
                print(line)
        fired.update(k for k, _ in fresh)

    if trainer:
        s, loss, ent, gn = trainer[-1]
        r = f"{rewards[-1][1]:.3f}@{rewards[-1][0]}" if rewards else "n/a"
        snap = (f"[{now()}] step={s} loss={loss:.4f} entropy={ent:.4f} "
                f"grad_norm={gn:.4f} reward_mean={r} alerts_total={len(fired)}")
    else:
        snap = f"[{now()}] no trainer steps parsed yet (rollout steps={len(rewards)})"
    with open(STATUS, "a") as f:
        f.write(snap + "\n")

    state["fired"] = sorted(fired)
    with open(STATE, "w") as f:
        json.dump(state, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
