#!/usr/bin/env bash
# on_speak.sh — LSpace GATE then mandatory triad for one Grok Bot spoken turn.
#
# Lawrence rule: every speak is a task. Operating mode: outbound LSpace gate on
# FULL speak text BEFORE triad/SendToUser success path.
# - ALLOW  → fire_triad with lspace receipt (proxy_jlens demoted, not gate)
# - BLOCK  → write FLAG receipt, optional flag triad, exit 3 (do NOT claim speak success)
#
# Usage:
#   ./on_speak.sh "exact words spoken to the user" [optional-title]
#   SPOKEN_TEXT=... ./on_speak.sh
#   echo "..." | ./on_speak.sh --stdin
set -euo pipefail

# One-release aliases: prefer LSPACE_*; accept JSPACE_* if caller still sets it
if [[ -z "${LSPACE_BODY_FILE:-}" && -n "${JSPACE_BODY_FILE:-}" ]]; then
  export LSPACE_BODY_FILE="$JSPACE_BODY_FILE"
fi
if [[ -z "${LSPACE_BODY:-}" && -n "${JSPACE_BODY:-}" ]]; then
  export LSPACE_BODY="$JSPACE_BODY"
fi
if [[ -z "${LSPACE_SNAP_PY:-}" && -n "${JSPACE_SNAP_PY:-}" ]]; then
  export LSPACE_SNAP_PY="$JSPACE_SNAP_PY"
fi

DIR="$(cd "$(dirname "$0")" && pwd)"
FIRE="$DIR/fire_triad.sh"
GATE="$DIR/lspace/pre_speak_gate.sh"
JLENS_VENV="${JLENS_VENV:-/workspace/jlens-venv}"
RUN_DIR="${RUN_DIR:-$DIR/runs}"
PUT_COVENANT="${PUT_COVENANT:-$DIR/put_covenant_memory.sh}"
export AI_MEMORY_ROOT="${AI_MEMORY_ROOT:-$DIR/ai_memory_root}"
# Further demote: proxy_jlens OFF by default (never gate; optional continuity only)
export SKIP_PROXY_JLENS="${SKIP_PROXY_JLENS:-1}"

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

TASK_ID="${FORCE_TASK_ID:-$(date -u +%Y%m%d-%H%M%S)-$$}"
SNAP_DIR="$RUN_DIR/$TASK_ID"
mkdir -p "$SNAP_DIR"
GATE_RECEIPT="$SNAP_DIR/lspace_gate_outbound.json"
export TASK_ID

# --- Operating gate on FULL text (no 480 truncate) ---
echo "=== on_speak LSpace outbound gate task_id=$TASK_ID speak_chars=${#SPOKEN_TEXT} ==="
set +e
"$GATE" "$SPOKEN_TEXT" "$GATE_RECEIPT"
GATE_EC=$?
set -e

if [[ "$GATE_EC" -eq 2 ]]; then
  echo "LSPACE_GATE_BLOCK task_id=$TASK_ID receipt=$GATE_RECEIPT" >&2
  FLAG_MSG="LSPACE GATE BLOCK — withheld speak. See receipt $GATE_RECEIPT"
  if [[ -f "$GATE_RECEIPT" ]]; then
    FLAG_MSG="$FLAG_MSG $(python3 -c "import json;d=json.load(open('$GATE_RECEIPT'));print(d.get('flag_message') or d.get('decision'))" 2>/dev/null || true)"
  fi
  # Fire a FLAG triad (not a successful speak) with gate receipt as lspace body
  export TITLE="lspace-gate-FLAG"
  export TASK="LSpace operating gate BLOCKED outbound speak"
  export DONE="Withheld problematic content; flag receipt logged"
  export OUTCOME="blocked"
  export PROMPT="LSPACE_GATE_FLAG: ${FLAG_MSG}"
  export MISTAKES="gate_block"
  export LESSONS="Silent intent or tombstone contradiction must not be spoken"
  export NOTES="hook=on_speak.sh; gate=BLOCK; speak_chars=${#SPOKEN_TEXT}; gate_receipt=$GATE_RECEIPT"
  export LSPACE_BODY_FILE="$GATE_RECEIPT"
  export SKIP_PROXY_JLENS=1
  export FULL_SPEAK_FILE="$SNAP_DIR/withheld_speak.txt"
  printf '%s' "$SPOKEN_TEXT" > "$FULL_SPEAK_FILE"
  set +e
  "$FIRE" "$TITLE" "$TASK" "$DONE" "$OUTCOME" "$PROMPT" "$MISTAKES" "$LESSONS" "$NOTES"
  set -e
  echo "$FLAG_MSG" >&2
  exit 3
fi

if [[ "$GATE_EC" -ne 0 ]]; then
  echo "LSPACE_GATE_ERROR exit=$GATE_EC" >&2
  exit 3
fi

# ALLOW: full speak for LSpace leg; demoted proxy may still get a short prompt for continuity
PROMPT="Grok Bot speaking to Lawrence: ${SPOKEN_TEXT}"
TASK="Assistant speak-turn: deliver user-visible message"
DONE="Spoke to user via SendToUser; triad logged under shared task_id"
OUTCOME="success"
NOTES="hook=on_speak.sh; speak_chars=${#SPOKEN_TEXT}; lspace_gate=ALLOW; gate_receipt=$GATE_RECEIPT"

export TITLE TASK DONE OUTCOME PROMPT
export MISTAKES="${MISTAKES:-none}"
export LESSONS="${LESSONS:-Every speak-turn fires triad; LSpace gate is operating mode}"
export NOTES
export LSPACE_BODY_FILE="$GATE_RECEIPT"
export FULL_SPEAK_FILE="$SNAP_DIR/full_speak.txt"
printf '%s' "$SPOKEN_TEXT" > "$FULL_SPEAK_FILE"

exec "$FIRE" "$TITLE" "$TASK" "$DONE" "$OUTCOME" "$PROMPT" "$MISTAKES" "$LESSONS" "$NOTES"
