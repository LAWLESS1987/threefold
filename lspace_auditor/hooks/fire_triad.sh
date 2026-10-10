#!/usr/bin/env bash
# fire_triad.sh — mandatory triad: tombstone + covenant + LSpace auditor
# (demoted proxy_jlens optional continuity leg — NOT the gate, NEVER called LSpace)
#
# Usage:
#   ./fire_triad.sh TITLE TASK DONE OUTCOME PROMPT [MISTAKES] [LESSONS] [NOTES]
# Env: TASK_ID, LSPACE_BODY_FILE, FULL_SPEAK_FILE, SKIP_PROXY_JLENS, ...
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
AGENT_ROOT="$(cd "$DIR/.." && pwd)"
TOMBSTONE_APPEND="${TOMBSTONE_APPEND:-$AGENT_ROOT/tombstone/append.sh}"
PUT_COVENANT="${PUT_COVENANT:-$DIR/put_covenant_memory.sh}"
JLENS_VENV="${JLENS_VENV:-/workspace/jlens-venv}"
# DEMOTTED proxy — not LSpace
PROXY_JLENS_PY="${JLENS_SNAPSHOT_PY:-/workspace/jlens-demo/jlens_snapshot.py}"
JLENS_LENS="${JLENS_LENS:-/workspace/jlens-demo/gpt2_jacobian_lens.pt}"
LSPACE_SNAP_PY="${LSPACE_SNAP_PY:-$DIR/lspace/lspace_snapshot.py}"
RUN_DIR="${RUN_DIR:-$DIR/runs}"
export AI_MEMORY_ROOT="${AI_MEMORY_ROOT:-$DIR/ai_memory_root}"

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
LSPACE_JSON="$SNAP_DIR/lspace_snapshot.json"
LSPACE_DIGEST_F="$SNAP_DIR/lspace_digest.txt"
PROXY_JSON="$SNAP_DIR/proxy_jlens_snapshot.json"
PROXY_DIGEST_F="$SNAP_DIR/proxy_jlens_digest.txt"

echo "=== fire_triad task_id=$TASK_ID ==="

if [[ ! -x "$JLENS_VENV/bin/python" ]]; then
  echo "Missing jlens venv python at $JLENS_VENV/bin/python" >&2
  exit 1
fi

# --- Leg (c) LSpace auditor FIRST (full speak when available) ---
echo "--- leg c: LSpace speak-text auditor ---"
SPEAK_FOR_LSPACE="$PROMPT"
if [[ -n "${FULL_SPEAK_FILE:-}" && -f "${FULL_SPEAK_FILE}" ]]; then
  SPEAK_FOR_LSPACE="$(cat "$FULL_SPEAK_FILE")"
fi
# If gate already produced a receipt, keep it and still write activation snapshot
if [[ -n "${LSPACE_BODY_FILE:-}" && -f "${LSPACE_BODY_FILE}" ]]; then
  cp "${LSPACE_BODY_FILE}" "$SNAP_DIR/lspace_gate_receipt.json"
fi

SPEAK_TMP=$(mktemp)
printf '%s' "$SPEAK_FOR_LSPACE" > "$SPEAK_TMP"
"$JLENS_VENV/bin/python" "$LSPACE_SNAP_PY" \
  --task-id "$TASK_ID" \
  --prompt-file "$SPEAK_TMP" \
  --out "$LSPACE_JSON" \
  --digest "$LSPACE_DIGEST_F"
rm -f "$SPEAK_TMP"
LSPACE_DIGEST="$(head -1 "$LSPACE_DIGEST_F")"
# Keep snapshot on disk only — never load into env (ARG_MAX)
unset LSPACE_BODY 2>/dev/null || true
export LSPACE_BODY_FILE="$LSPACE_JSON"

# Demoted proxy_jlens (continuity only; truncated prompt OK; NOT gate; NOT LSpace)
PROXY_DIGEST="proxy_jlens=SKIPPED"
if [[ "${SKIP_PROXY_JLENS:-0}" != "1" ]]; then
  echo "--- leg c2: DEMOTTED proxy_jlens (not LSpace) ---"
  PROXY_PROMPT="$PROMPT"
  if ((${#PROXY_PROMPT} > 480)); then
    PROXY_PROMPT="${PROXY_PROMPT:0:480}…"
  fi
  set +e
  "$JLENS_VENV/bin/python" "$PROXY_JLENS_PY" \
    --task-id "$TASK_ID" \
    --prompt "$PROXY_PROMPT" \
    --out "$PROXY_JSON" \
    --digest "$PROXY_DIGEST_F" \
    --lens "$JLENS_LENS"
  PROXY_EC=$?
  set -e
  if [[ "$PROXY_EC" -eq 0 && -f "$PROXY_DIGEST_F" ]]; then
    PROXY_DIGEST="$(head -1 "$PROXY_DIGEST_F")"
  else
    echo "WARN: demoted proxy_jlens failed (non-fatal); LSpace leg is authoritative" >&2
    PROXY_DIGEST="proxy_jlens=FAILED"
  fi
else
  echo "--- leg c2: proxy_jlens skipped (SKIP_PROXY_JLENS=1) ---"
fi

# --- Leg (a) tombstone ---
echo "--- leg a: tombstone ---"
if [[ ! -x "$TOMBSTONE_APPEND" ]]; then
  echo "Missing tombstone append at $TOMBSTONE_APPEND" >&2
  exit 1
fi
COMBINED_NOTES="task_id=${TASK_ID}; lspace_digest=${LSPACE_DIGEST}; proxy_jlens_digest=${PROXY_DIGEST}"
if [[ -n "$NOTES" ]]; then
  COMBINED_NOTES="${COMBINED_NOTES}; ${NOTES}"
fi
"$TOMBSTONE_APPEND" "$TITLE" "$TASK" "$DONE" "$OUTCOME" "$MISTAKES" "$LESSONS" "$COMBINED_NOTES"

# --- Leg (b) covenant puts: task + lspace (+ optional demoted proxy) ---
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
lspace_digest=${LSPACE_DIGEST}
proxy_jlens_digest=${PROXY_DIGEST}
"
export LSPACE_BODY_FILE="$LSPACE_JSON"
unset LSPACE_BODY 2>/dev/null || true
if [[ -f "$PROXY_JSON" ]]; then
  export PROXY_JLENS_BODY_FILE="$PROXY_JSON"
  export JLENS_BODY_FILE="$PROXY_JSON"
fi
TASK_BODY_FILE="$SNAP_DIR/task_body.txt"
printf '%s' "$TASK_BODY" > "$TASK_BODY_FILE"
BODY_FILE="$TASK_BODY_FILE" "$PUT_COVENANT" "$TASK_ID" "" \
  "triad task memory for ${TASK_ID}"

echo "=== triad proof ==="
echo "task_id=$TASK_ID"
echo "lspace_snapshot=$LSPACE_JSON"
rg -n --fixed-strings "$TASK_ID" "$AGENT_ROOT/tombstone/tombstone.md" | tail -5 || {
  echo "PROOF FAIL: task_id not in tombstone" >&2
  exit 1
}
rg -n --fixed-strings "$TASK_ID" "$AI_MEMORY_ROOT" -g '*.md' | head -20 || {
  echo "PROOF FAIL: task_id not in covenant memories" >&2
  exit 1
}
rg -n --fixed-strings "$TASK_ID" "$LSPACE_JSON" || {
  echo "PROOF FAIL: task_id not in lspace snapshot" >&2
  exit 1
}
echo "FIRE_TRIAD_OK task_id=$TASK_ID (tombstone + covenant + lspace auditor; proxy demoted)"
