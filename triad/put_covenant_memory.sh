#!/usr/bin/env bash
# Put covenant memories into AI_MEMORY_ROOT (public clone of LAWLESS1987/threefold-memory).
# Never writes into covenant-src/ code tree.
#
# Writing and publishing are separate steps. A put only writes. The commit and
# push of the data repo is its own call (--publish), and fire_triad.sh makes
# that call only after triad/verify_triad.py has passed, so a triad that does
# not verify is not pushed by that run.
#
# Usage:
#   ./put_covenant_memory.sh TASK_ID BODY [DESCRIPTION]     write, never push
#   ./put_covenant_memory.sh --publish TASK_ID              commit+push the data repo
# Env:
#   AI_MEMORY_ROOT, COVENANT_SRC, AGENT_NAME, MEMORY_TYPE, GH_TOKEN
#   JLENS_BODY — if set, also writes jlens-{TASK_ID} (type:jlens snapshot)
#                CLI enum only allows user|feedback|project|reference;
#                we use --type project and stamp type:jlens in body+description.
#   JLENS_BODY_FILE / LLENS_BODY_FILE — the upstream JLens capture and the
#                L-lens artifact, read from files (they can exceed one
#                command-line argument); written as jlens-{TASK_ID} and
#                llens-{TASK_ID} through triad/put_memory_file.py, which makes
#                the same MemoryStore.put call main.py makes.
#   PYTHON_COV — python for covenant's memory code (default python3)
#   OBS_BODY — retired; ignored (self-observation is not a triad leg)
#   MEMORY_INTENT, JLENS_INTENT, LLENS_INTENT, MEMORY_BENEFICIARY,
#   MEMORY_DIVERGENCE, MEMORY_DIVERGES_FROM — covenant's intent provenance,
#                which it requires on every put: why the record is written
#                (one sentence, not a restatement of the body), for whom, and
#                what it conflicts with ("none", or a claim plus the name of
#                the record it diverges from). The defaults below say what a
#                triad record is for; set these when a record is written for
#                another reason. Covenant refuses a put whose intent restates
#                its body, and the script then exits nonzero.
#   GH_TOKEN — used by --publish only. It reaches git through the environment
#                and is never part of a remote address or of any command's
#                arguments (see sync_public_memory_root).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
COVENANT_SRC="${COVENANT_SRC:-$DIR/covenant-src}"
MAIN="$COVENANT_SRC/ai_memory_system/main.py"
export AI_MEMORY_ROOT="${AI_MEMORY_ROOT:-$DIR/ai_memory_root}"
AGENT_NAME="${AGENT_NAME:-grok-bot}"
MEMORY_TYPE="${MEMORY_TYPE:-project}"
MEMORY_INTENT="${MEMORY_INTENT:-Lets later auditors locate the tombstone entry plus both lens artifacts by task_id, then confirm all three legs agree.}"
JLENS_INTENT="${JLENS_INTENT:-Preserves the upstream measurement beside the task record so later verifiers can recompute the digest and compare it against the tombstone entry.}"
LLENS_INTENT="${LLENS_INTENT:-Keeps the provenance package beside the task record so later verifiers can recompute the digest and the summaries without trusting the writer.}"
MEMORY_BENEFICIARY="${MEMORY_BENEFICIARY:-whoever later audits this task: the operator or a future agent}"
MEMORY_DIVERGENCE="${MEMORY_DIVERGENCE:-none}"
MEMORY_DIVERGES_FROM="${MEMORY_DIVERGES_FROM:-}"

PUBLISH_ONLY=""
if [[ "${1:-}" == "--publish" ]]; then
  PUBLISH_ONLY=1
  shift
fi

TASK_ID="${1:-${TASK_ID:-}}"
BODY="${2:-${BODY:-${DECISION_BODY:-}}}"
DESCRIPTION="${3:-${DESCRIPTION:-triad decision for ${TASK_ID:-}}}"

if [[ -z "$TASK_ID" || ( -z "$PUBLISH_ONLY" && -z "$BODY" ) ]]; then
  echo "Usage: $0 TASK_ID BODY [DESCRIPTION]" >&2
  echo "       $0 --publish TASK_ID" >&2
  exit 1
fi

if [[ -z "$PUBLISH_ONLY" ]]; then
  if [[ ! -f "$MAIN" ]]; then
    echo "Covenant main.py not found at $MAIN" >&2
    exit 1
  fi

  mkdir -p "$AI_MEMORY_ROOT"

  safe_name() { printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '-'; }

  MEM_NAME="$(safe_name "task-${TASK_ID}")"

  echo "AI_MEMORY_ROOT=$AI_MEMORY_ROOT"
  "${PYTHON_COV:-python3}" "$MAIN" --root "$AI_MEMORY_ROOT" put "$MEM_NAME" \
    --description "$DESCRIPTION" \
    --type "$MEMORY_TYPE" \
    --body "$BODY" \
    --agent "$AGENT_NAME" \
    --intent "$MEMORY_INTENT" --beneficiary "$MEMORY_BENEFICIARY" \
    --divergence "$MEMORY_DIVERGENCE" --diverges-from "$MEMORY_DIVERGES_FROM"

  PUT_FILE="$DIR/put_memory_file.py"
  WROTE_EXTRA=""
  if [[ -n "${JLENS_BODY_FILE:-}" ]]; then
    JLENS_NAME="$(safe_name "jlens-${TASK_ID}")"
    "${PYTHON_COV:-python3}" "$PUT_FILE" --root "$AI_MEMORY_ROOT" --name "$JLENS_NAME" \
      --description "type:jlens upstream Jacobian Lens capture (anthropics/jacobian-lens, Apache-2.0) for ${TASK_ID}" \
      --type project --agent "$AGENT_NAME" --body-file "$JLENS_BODY_FILE" --covenant-src "$COVENANT_SRC" \
      --intent "$JLENS_INTENT" --beneficiary "$MEMORY_BENEFICIARY" \
      --divergence "$MEMORY_DIVERGENCE" --diverges-from "$MEMORY_DIVERGES_FROM"
    WROTE_EXTRA="$WROTE_EXTRA $JLENS_NAME"
  fi
  if [[ -n "${LLENS_BODY_FILE:-}" ]]; then
    LLENS_NAME="$(safe_name "llens-${TASK_ID}")"
    "${PYTHON_COV:-python3}" "$PUT_FILE" --root "$AI_MEMORY_ROOT" --name "$LLENS_NAME" \
      --description "type:llens Threefold L-lens observability/provenance artifact built on the JLens capture jlens-${TASK_ID}" \
      --type project --agent "$AGENT_NAME" --body-file "$LLENS_BODY_FILE" --covenant-src "$COVENANT_SRC" \
      --intent "$LLENS_INTENT" --beneficiary "$MEMORY_BENEFICIARY" \
      --divergence "$MEMORY_DIVERGENCE" --diverges-from "$MEMORY_DIVERGES_FROM"
    WROTE_EXTRA="$WROTE_EXTRA $LLENS_NAME"
  fi

  if [[ -n "$WROTE_EXTRA" ]]; then
    echo "Wrote covenant memories: $MEM_NAME$WROTE_EXTRA under $AI_MEMORY_ROOT"
  elif [[ -n "${JLENS_BODY:-}" ]]; then
    JLENS_NAME="$(safe_name "jlens-${TASK_ID}")"
    # Real CLI type enum has no 'jlens'; stamp type:jlens in description+body.
    "${PYTHON_COV:-python3}" "$MAIN" --root "$AI_MEMORY_ROOT" put "$JLENS_NAME" \
      --description "type:jlens full Jacobian-lens snapshot for ${TASK_ID}" \
      --type project \
      --body "$JLENS_BODY" \
      --agent "$AGENT_NAME" \
      --intent "$JLENS_INTENT" --beneficiary "$MEMORY_BENEFICIARY" \
      --divergence "$MEMORY_DIVERGENCE" --diverges-from "$MEMORY_DIVERGES_FROM"
    echo "Wrote covenant memories: $MEM_NAME and $JLENS_NAME under $AI_MEMORY_ROOT"
  else
    echo "Wrote covenant memory: $MEM_NAME under $AI_MEMORY_ROOT"
  fi
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
  # Prefer GH_TOKEN HTTPS if present; else existing remote. The token is handed
  # to git through the environment (GIT_CONFIG_COUNT/KEY/VALUE, git 2.31+) as an
  # extra header for this one repository. It is never part of a remote address
  # and never an argument of any command: printf and export are shell builtins,
  # and base64 reads standard input.
  if [[ -n "${GH_TOKEN:-}" ]]; then
    local remote="https://github.com/LAWLESS1987/threefold-memory.git"
    local n="${GIT_CONFIG_COUNT:-0}" basic
    basic="$(printf 'x-access-token:%s' "$GH_TOKEN" | base64 | tr -d '\r\n')"
    (
      export GIT_CONFIG_COUNT="$((n + 1))"
      export "GIT_CONFIG_KEY_${n}=http.${remote}.extraheader"
      export "GIT_CONFIG_VALUE_${n}=Authorization: Basic ${basic}"
      git -C "$root" push "$remote" HEAD:main
    )
  else
    git -C "$root" push origin HEAD:main
  fi
  echo "Pushed AI_MEMORY_ROOT to LAWLESS1987/threefold-memory"
}

if [[ -n "$PUBLISH_ONLY" ]]; then
  sync_public_memory_root
fi
