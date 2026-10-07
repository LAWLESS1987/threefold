#!/usr/bin/env bash
# Put covenant memories into the PRIVATE AI_MEMORY_ROOT.
# Never writes into vendor/covenant. Memories stay outside the public clone.
#
# Usage:
#   ./put_covenant_memory.sh TASK_ID BODY [DESCRIPTION]
# Env:
#   AI_MEMORY_ROOT, COVENANT_SRC, AGENT_NAME, MEMORY_TYPE
#   JLENS_BODY — if set, also writes jlens-{TASK_ID} (type:jlens snapshot)
#                CLI enum only allows user|feedback|project|reference;
#                we use --type project and stamp type:jlens in body+description.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$DIR/.." && pwd)"

# Prefer COVENANT_SRC; else vendor/covenant under repo root.
COVENANT_SRC="${COVENANT_SRC:-$REPO_ROOT/vendor/covenant}"
MAIN="$COVENANT_SRC/ai_memory_system/main.py"
export AI_MEMORY_ROOT="${AI_MEMORY_ROOT:-$REPO_ROOT/data/ai_memory_root}"
AGENT_NAME="${AGENT_NAME:-grok-bot}"
MEMORY_TYPE="${MEMORY_TYPE:-project}"

TASK_ID="${1:-${TASK_ID:-}}"
BODY="${2:-${BODY:-${DECISION_BODY:-}}}"
DESCRIPTION="${3:-${DESCRIPTION:-triad decision for ${TASK_ID:-}}}"

if [[ -z "$TASK_ID" || -z "$BODY" ]]; then
  echo "Usage: $0 TASK_ID BODY [DESCRIPTION]" >&2
  exit 1
fi

if [[ ! -f "$MAIN" ]]; then
  echo "Covenant main.py not found at $MAIN" >&2
  echo "Clone it: git clone https://github.com/LAWLESS1987/covenant \"$REPO_ROOT/vendor/covenant\"" >&2
  echo "Or set COVENANT_SRC to an existing checkout." >&2
  exit 1
fi

mkdir -p "$AI_MEMORY_ROOT"

safe_name() { printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '-'; }

MEM_NAME="$(safe_name "task-${TASK_ID}")"

echo "AI_MEMORY_ROOT=$AI_MEMORY_ROOT"
python3 "$MAIN" --root "$AI_MEMORY_ROOT" put "$MEM_NAME" \
  --description "$DESCRIPTION" \
  --type "$MEMORY_TYPE" \
  --body "$BODY" \
  --agent "$AGENT_NAME"

if [[ -n "${JLENS_BODY:-}" ]]; then
  JLENS_NAME="$(safe_name "jlens-${TASK_ID}")"
  # Real CLI type enum has no 'jlens'; stamp type:jlens in description+body.
  python3 "$MAIN" --root "$AI_MEMORY_ROOT" put "$JLENS_NAME" \
    --description "type:jlens full Jacobian-lens snapshot for ${TASK_ID}" \
    --type project \
    --body "$JLENS_BODY" \
    --agent "$AGENT_NAME"
  echo "Wrote covenant memories: $MEM_NAME and $JLENS_NAME under $AI_MEMORY_ROOT"
else
  echo "Wrote covenant memory: $MEM_NAME under $AI_MEMORY_ROOT"
fi
