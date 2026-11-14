#!/usr/bin/env bash
# on_speak.sh — fire the mandatory triad for one Grok Bot spoken turn.
#
# Lawrence rule: every time Grok Bot speaks (SendToUser), it counts as a task.
# This hook assigns a task_id and runs fire_triad.sh (tombstone ∥ covenant ∥ L-lens over Anthropic's Jacobian Lens).
# No degrade mode: any leg failure exits nonzero, and a triad that fails after
# its tombstone entry was written is recorded as failed (see below).
#
# Usage:
#   ./on_speak.sh "exact words spoken to the user" [optional-title]
# Or:
#   SPOKEN_TEXT=... ./on_speak.sh
#   echo "..." | ./on_speak.sh --stdin
#
# Env overrides: same as fire_triad.sh (JLENS_VENV, AI_MEMORY_ROOT, TASK_ID, ...)
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
FIRE="$DIR/fire_triad.sh"

if [[ "${1:-}" == "--stdin" ]]; then
  SPOKEN_TEXT="$(cat)"
  TITLE="${2:-speak-turn}"
elif [[ -n "${SPOKEN_TEXT:-}" ]]; then
  TITLE="${1:-speak-turn}"
else
  SPOKEN_TEXT="${1:-}"
  TITLE="${2:-speak-turn}"
fi

if [[ -z "$SPOKEN_TEXT" ]]; then
  echo "Usage: $0 \"spoken text\" [title]   OR   echo text | $0 --stdin [title]" >&2
  exit 1
fi

# gpt2 context is small; keep a stable prefix + truncated speech for the lens prompt
TRUNC="$SPOKEN_TEXT"
if ((${#TRUNC} > 480)); then
  TRUNC="${TRUNC:0:480}…"
fi
PROMPT="Grok Bot speaking to Lawrence: ${TRUNC}"

TASK="Assistant speak-turn: deliver user-visible message"
DONE="Spoke to user via SendToUser; triad logged under shared task_id"
NOTES="hook=on_speak.sh; speak_chars=${#SPOKEN_TEXT}"

# The recorded outcome comes from what happened: the exit status of the triad
# this hook runs. fire_triad.sh writes its tombstone entry part-way through
# (after the lens leg, before the covenant writes, the verification and the
# publish step), so that entry can only say what the outcome is IF the triad
# completes. When the triad exits nonzero after that entry was written, this
# hook appends a second entry for the same task with outcome "failed" and the
# exit status. The log is append-only, so the earlier entry is corrected, not
# rewritten. A triad that fails before its entry is written (the lens leg)
# still appends nothing.
OUTCOME="success"   # what fire_triad.sh's entry says; true only if the triad completes
REPO_ROOT="$(cd "$DIR/.." && pwd)"
TOMBSTONE_APPEND="${TOMBSTONE_APPEND:-$REPO_ROOT/tombstone/append.sh}"
export TOMBSTONE_LOG="${TOMBSTONE_LOG:-$REPO_ROOT/tombstone/tombstone.md}"
export TASK_ID="${TASK_ID:-$(date -u +%Y%m%d-%H%M%S)-$$}"
entries_for_task() { grep -c -F "task_id=${TASK_ID};" "$TOMBSTONE_LOG" 2>/dev/null || true; }
ENTRIES_BEFORE="$(entries_for_task)"

export TITLE TASK DONE OUTCOME PROMPT
export MISTAKES="${MISTAKES:-none}"
export LESSONS="${LESSONS:-Every speak-turn fires triad; no degrade mode}"
export NOTES

RC=0
"$FIRE" "$TITLE" "$TASK" "$DONE" "$OUTCOME" "$PROMPT" "$MISTAKES" "$LESSONS" "$NOTES" || RC=$?
if [[ "$RC" -ne 0 && "$(entries_for_task)" != "$ENTRIES_BEFORE" ]]; then
  "$TOMBSTONE_APPEND" "$TITLE" "$TASK" \
    "fire_triad.sh exited ${RC} after its tombstone entry was written; the triad did not complete" \
    "failed" "$MISTAKES" "$LESSONS" \
    "hook=on_speak.sh; triad_failed_for=${TASK_ID}; fire_triad_exit=${RC}; corrects the earlier entry for this task, which said ${OUTCOME}"
fi
exit "$RC"
