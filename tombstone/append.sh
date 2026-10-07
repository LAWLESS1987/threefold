#!/usr/bin/env bash
# Append one tombstone entry. Usage:
#   ./append.sh "short-title" "task" "done" "outcome" "mistakes" "lessons" ["notes"]
# Or with env vars: TITLE TASK DONE OUTCOME MISTAKES LESSONS NOTES
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="${TOMBSTONE_LOG:-$DIR/tombstone.md}"

TITLE="${1:-${TITLE:-}}"
TASK="${2:-${TASK:-}}"
DONE="${3:-${DONE:-}}"
OUTCOME="${4:-${OUTCOME:-}}"
MISTAKES="${5:-${MISTAKES:-none}}"
LESSONS="${6:-${LESSONS:-none}}"
NOTES="${7:-${NOTES:-}}"

if [[ -z "$TITLE" || -z "$TASK" || -z "$DONE" || -z "$OUTCOME" ]]; then
  echo "Usage: $0 \"short-title\" \"task\" \"done\" \"outcome\" [\"mistakes\"] [\"lessons\"] [\"notes\"]" >&2
  echo "outcome must be one of: success | partial | failed | blocked" >&2
  exit 1
fi

case "$OUTCOME" in
  success|partial|failed|blocked) ;;
  *) echo "Invalid outcome: $OUTCOME (use success|partial|failed|blocked)" >&2; exit 1 ;;
esac

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
ID="$(date -u +%Y%m%d-%H%M%S)"

{
  echo ""
  echo "### ${TS} | ${TITLE}"
  echo "- id: ${ID}"
  echo "- task: ${TASK}"
  echo "- done: ${DONE}"
  echo "- outcome: ${OUTCOME}"
  echo "- mistakes: ${MISTAKES}"
  echo "- lessons: ${LESSONS}"
  if [[ -n "$NOTES" ]]; then
    echo "- notes: ${NOTES}"
  else
    echo "- notes: none"
  fi
} >> "$LOG"

echo "Appended entry ${ID} to $LOG"
