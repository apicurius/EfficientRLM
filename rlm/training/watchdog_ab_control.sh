#!/bin/bash
# Cron-safe watchdog for the EfficientRLM control arm. The supervisor tmux
# session lives ON ai16 (compute node), not the login node — the nightly 01:00
# login-node reaper kills login tmux servers (proven Jul 5: treatment tmux died
# 01:00:02, cron recreated it 01:15:01), but never touches compute nodes.
# This watchdog runs from login-node cron purely as a backstop: if the ai16
# session somehow dies, recreate it there via ssh (self-authorized keypair,
# ~/.ssh/id_ed25519). Idempotent: safe to run by hand and every 15 min.
#
# Install (crontab -e):
#   */15 * * * * /scratch/omeerdogan23/erlm/.research/EfficientRLM/rlm/training/watchdog_ab_control.sh >> /scratch/omeerdogan23/erlm/outputs/ab-watchdog.log 2>&1
#
# Remove the cron line once ARM_DONE exists.
set -u

TMUX=/opt/ohpc/pub/apps/tmux/3.5/bin/tmux
NODE=ai16
EFF=/scratch/omeerdogan23/erlm/.research/EfficientRLM
OUT="$EFF/outputs/qwen3-30b-ab-control-multienv-200step"
SESSION="ab-control"
# Fixed log path (not timestamped): the supervisor appends, and any tail -f
# watching the launch log keeps streaming across restarts.
CONSOLE_LOG="$EFF/outputs/ab-control-console-fresh-20260707.log"

rnode() { timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=10 "$NODE" "$@"; }

[ -e "$OUT/ARM_DONE" ] && exit 0
# Supervisor gave up after repeated fast failures — do NOT resurrect it into a
# crash-loop; a human investigates and removes ARM_FAILED to re-arm.
[ -e "$OUT/ARM_FAILED" ] && exit 0

# Supervisor session alive on the compute node — nothing to do.
rnode "$TMUX has-session -t $SESSION" 2>/dev/null && exit 0

# ssh itself failing is not proof the arm is dead (node busy, sshd hiccup).
# Distinguish via the console log: the run logs every ~10 s, so a log quieter
# than 20 min with no reachable session means genuinely down.
if ! rnode true 2>/dev/null; then
  AGE=$(( $(date +%s) - $(stat -c %Y "$CONSOLE_LOG" 2>/dev/null || echo 0) ))
  if [ "$AGE" -lt 1200 ]; then
    exit 0
  fi
  echo "[watchdog] $(date '+%F %T') control: ssh $NODE unreachable AND log stale ${AGE}s — cannot recover remotely"
  exit 1
fi

# Session gone but a supervisor/srun may survive outside tmux on the node.
# Never start a second training on the same GPUs.
if rnode "pgrep -u $USER -f '[s]upervise_ab_control.sh|[r]un_ab_control.sh'" >/dev/null 2>&1; then
  echo "[watchdog] $(date '+%F %T') control: supervisor alive on $NODE outside tmux — not relaunching"
  exit 0
fi
# A leftover login-node supervisor (pre-migration layout) also counts.
if pgrep -u "$USER" -f '[s]upervise_ab_control.sh|[r]un_ab_control.sh' >/dev/null 2>&1; then
  echo "[watchdog] $(date '+%F %T') control: supervisor alive on login node — not relaunching"
  exit 0
fi

echo "[watchdog] $(date '+%F %T') control: creating tmux session $SESSION on $NODE"
rnode "$TMUX new-session -d -s $SESSION \"SMOKE_LOG=$CONSOLE_LOG $EFF/rlm/training/supervise_ab_control.sh\""
