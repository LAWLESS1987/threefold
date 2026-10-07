#!/usr/bin/env bash
# fire_triad.sh — mandatory three-leg triad (tombstone + covenant + real JLens)
#
# Usage:
#   ./fire_triad.sh TITLE TASK DONE OUTCOME PROMPT [MISTAKES] [LESSONS] [NOTES]
# Or env: TITLE TASK DONE OUTCOME PROMPT MISTAKES LESSONS NOTES TASK_ID
#
# Legs (all required; any failure => exit nonzero):
#   (a) tombstone append with JLens digest in notes
#   (b) covenant puts (task-{id} + jlens-{id} type:jlens snapshot)
#   (c) real Anthropic jlens.JacobianLens.apply snapshot
#
# Defaults resolve relative to REPO_ROOT (parent of triad/).
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$DIR/.." && pwd)"
TOMBSTONE_APPEND="${TOMBSTONE_APPEND:-$REPO_ROOT/tombstone/append.sh}"
PUT_COVENANT="${PUT_COVENANT:-$DIR/put_covenant_memory.sh}"
JLENS_VENV="${JLENS_VENV:-${REPO_ROOT}/.venv}"
# Prefer repo-local scripts; allow override for shared venvs.
JLENS_SNAPSHOT_PY="${JLENS_SNAPSHOT_PY:-$REPO_ROOT/jlens_snapshot.py}"
JLENS_LENS="${JLENS_LENS:-$REPO_ROOT/gpt2_jacobian_lens.pt}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs}"
export AI_MEMORY_ROOT="${AI_MEMORY_ROOT:-$REPO_ROOT/data/ai_memory_root}"
export COVENANT_SRC="${COVENANT_SRC:-$REPO_ROOT/vendor/covenant}"

TITLE="${1:-${TITLE:-}}"
TASK="${2:-${TASK:-}}"
DONE="${3:-${DONE:-}}"
OUTCOME="${4:-${OUTCOME:-}}"
PROMPT="${5:-${PROMPT:-}}"
MISTAKES="${6:-${MISTAKES:-none}}"
LESSONS="${7:-${LESSONS:-none}}"
NOTES="${8:-${NOTES:-}}"

if [[ -z "$TITLE" || -z "$TASK" || -z "$DONE" || -z "$OUTCOME" || -z "$PROMPT" ]]; then
  echo "Usage: $0 TITLE TASK DONE OUTCOME PROMPT [MISTAKES] [LESSONS] [NOTES]" >&2
  exit 1
fi

TASK_ID="${TASK_ID:-$(date -u +%Y%m%d-%H%M%S)-$$}"
SNAP_DIR="$RUN_DIR/$TASK_ID"
mkdir -p "$SNAP_DIR"
SNAP_JSON="$SNAP_DIR/jlens_snapshot.json"
SNAP_DIGEST="$SNAP_DIR/jlens_digest.txt"

echo "=== fire_triad task_id=$TASK_ID ==="
echo "prompt=$PROMPT"
echo "REPO_ROOT=$REPO_ROOT"

# Resolve python: JLENS_VENV, else python3 on PATH
if [[ -x "$JLENS_VENV/bin/python" ]]; then
  PYTHON="$JLENS_VENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo "No Python found (set JLENS_VENV or install python3)" >&2
  exit 1
fi

# --- Leg (c) FIRST so digest can go into tombstone notes ---
echo "--- leg c: JLens apply ---"
if [[ ! -f "$JLENS_SNAPSHOT_PY" ]]; then
  echo "Missing jlens_snapshot.py at $JLENS_SNAPSHOT_PY" >&2
  exit 1
fi
if [[ ! -f "$JLENS_LENS" ]]; then
  echo "Missing lens weights at $JLENS_LENS" >&2
  echo "Run: scripts/fetch_lens.sh" >&2
  exit 1
fi

"$PYTHON" "$JLENS_SNAPSHOT_PY" \
  --task-id "$TASK_ID" \
  --prompt "$PROMPT" \
  --out "$SNAP_JSON" \
  --digest "$SNAP_DIGEST" \
  --lens "$JLENS_LENS"
JLENS_DIGEST="$(head -1 "$SNAP_DIGEST")"
JLENS_BODY="$(cat "$SNAP_JSON")"

# --- Leg (a) tombstone with JLens digest in notes ---
echo "--- leg a: tombstone ---"
if [[ ! -x "$TOMBSTONE_APPEND" ]]; then
  echo "Missing tombstone append at $TOMBSTONE_APPEND" >&2
  exit 1
fi
COMBINED_NOTES="task_id=${TASK_ID}; jlens_digest=${JLENS_DIGEST}"
if [[ -n "$NOTES" ]]; then
  COMBINED_NOTES="${COMBINED_NOTES}; ${NOTES}"
fi
"$TOMBSTONE_APPEND" "$TITLE" "$TASK" "$DONE" "$OUTCOME" "$MISTAKES" "$LESSONS" "$COMBINED_NOTES"

# --- Leg (b) covenant puts ---
echo "--- leg b: covenant ---"
if [[ ! -x "$PUT_COVENANT" ]]; then
  echo "Missing put_covenant_memory.sh at $PUT_COVENANT" >&2
  exit 1
fi
TASK_BODY="task_id=${TASK_ID}
title=${TITLE}
task=${TASK}
done=${DONE}
outcome=${OUTCOME}
prompt=${PROMPT}
jlens_digest=${JLENS_DIGEST}
"
JLENS_BODY="$JLENS_BODY" "$PUT_COVENANT" "$TASK_ID" "$TASK_BODY" \
  "triad task memory for ${TASK_ID}"

# Proof: all three legs share task_id
echo "=== triad proof ==="
echo "task_id=$TASK_ID"
echo "jlens_snapshot=$SNAP_JSON"
rg -n --fixed-strings "$TASK_ID" "$REPO_ROOT/tombstone/tombstone.md" | tail -5 || {
  echo "PROOF FAIL: task_id not in tombstone" >&2
  exit 1
}
rg -n --fixed-strings "$TASK_ID" "$AI_MEMORY_ROOT" -g '*.md' | head -20 || {
  echo "PROOF FAIL: task_id not in covenant memories" >&2
  exit 1
}
rg -n --fixed-strings "$TASK_ID" "$SNAP_JSON" || {
  echo "PROOF FAIL: task_id not in jlens snapshot" >&2
  exit 1
}
echo "FIRE_TRIAD_OK task_id=$TASK_ID (tombstone + covenant + real jlens.apply)"
