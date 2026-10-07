#!/usr/bin/env bash
# on_speak.sh — fire the mandatory triad for one Grok Bot spoken turn.
#
# Lawrence rule: every time Grok Bot speaks (SendToUser), it counts as a task.
# This hook assigns a task_id and runs fire_triad.sh (tombstone ∥ covenant ∥ JLens).
# No degrade mode: any leg failure exits nonzero.
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
OUTCOME="success"
NOTES="hook=on_speak.sh; speak_chars=${#SPOKEN_TEXT}"

export TITLE TASK DONE OUTCOME PROMPT
export MISTAKES="${MISTAKES:-none}"
export LESSONS="${LESSONS:-Every speak-turn fires triad; no degrade mode}"
export NOTES

exec "$FIRE" "$TITLE" "$TASK" "$DONE" "$OUTCOME" "$PROMPT" "$MISTAKES" "$LESSONS" "$NOTES"
