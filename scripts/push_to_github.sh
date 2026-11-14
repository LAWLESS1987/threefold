#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
OWNER=LAWLESS1987
NAME=threefold
# Create if missing
if ! gh repo view "$OWNER/$NAME" >/dev/null 2>&1; then
  gh repo create "$OWNER/$NAME" --public --description "Mandatory three-leg agent memory: tombstone || covenant || JLens (shared task_id, no degrade)" --source . --remote origin --push
else
  # An existing origin is never removed or rewritten. It is added only when
  # there is none, and the script stops if it points anywhere but this repository.
  if CURRENT="$(git remote get-url origin 2>/dev/null)"; then
    CURRENT="${CURRENT%/}"
    CURRENT="${CURRENT%.git}"
    case "${CURRENT,,}" in
      "https://github.com/${OWNER,,}/$NAME" | "git@github.com:${OWNER,,}/$NAME" | "ssh://git@github.com/${OWNER,,}/$NAME") ;;
      *)
        echo "origin does not point to https://github.com/$OWNER/$NAME; it is left as it is and nothing is pushed." >&2
        echo "See it with: git remote get-url origin. Change it yourself if that is what you want, then run this again." >&2
        exit 1
        ;;
    esac
  else
    git remote add origin "https://github.com/$OWNER/$NAME.git"
  fi
  git push -u origin main
fi
echo "PUSH_OK https://github.com/$OWNER/$NAME"
