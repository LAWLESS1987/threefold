#!/usr/bin/env bash
# Put covenant memories into AI_MEMORY_ROOT (public clone of LAWLESS1987/threefold-memory).
# Never writes into covenant-src/ code tree. After each put, commit+push the data repo.
#
# Usage:
#   ./put_covenant_memory.sh TASK_ID BODY [DESCRIPTION]
# Env:
#   AI_MEMORY_ROOT, COVENANT_SRC, AGENT_NAME, MEMORY_TYPE, GH_TOKEN
#   JLENS_BODY — if set, also writes jlens-{TASK_ID} (type:jlens snapshot)
#                CLI enum only allows user|feedback|project|reference;
#                we use --type project and stamp type:jlens in body+description.
#   OBS_BODY — retired; ignored (self-observation is not a triad leg)
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
COVENANT_SRC="${COVENANT_SRC:-$DIR/covenant-src}"
MAIN="$COVENANT_SRC/ai_memory_system/main.py"
export AI_MEMORY_ROOT="${AI_MEMORY_ROOT:-$DIR/ai_memory_root}"
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

# Sync public AI_MEMORY_ROOT to GitHub when it is a git checkout of threefold-memory
sync_public_memory_root() {
  local root="$AI_MEMORY_ROOT"
  if [[ ! -d "$root/.git" ]]; then
    echo "AI_MEMORY_ROOT is not a git repo; skip push ($root)"
    return 0
  fi
  git -C "$root" add -A
  if git -C "$root" diff --cached --quiet; then
    echo "No memory changes to commit"
    return 0
  fi
  git -C "$root" -c user.email="${GIT_AUTHOR_EMAIL:-grok-bot@lawless.local}" \
    -c user.name="${GIT_AUTHOR_NAME:-Grok Bot}" \
    commit -m "triad memory: ${TASK_ID} ($(date -u +%Y-%m-%dT%H:%MZ))"
  # Prefer GH_TOKEN HTTPS if present; else existing remote
  if [[ -n "${GH_TOKEN:-}" ]]; then
    git -C "$root" push "https://x-access-token:${GH_TOKEN}@github.com/LAWLESS1987/threefold-memory.git" HEAD:main
  else
    git -C "$root" push origin HEAD:main
  fi
  echo "Pushed AI_MEMORY_ROOT to LAWLESS1987/threefold-memory"
}

sync_public_memory_root
