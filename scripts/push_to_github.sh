#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
OWNER=LAWLESS1987
NAME=triad
# Create if missing
if ! gh repo view "$OWNER/$NAME" >/dev/null 2>&1; then
  gh repo create "$OWNER/$NAME" --public --description "Mandatory three-leg agent memory: tombstone || covenant || JLens (shared task_id, no degrade)" --source . --remote origin --push
else
  git remote remove origin 2>/dev/null || true
  git remote add origin "https://github.com/$OWNER/$NAME.git"
  git push -u origin main
fi
echo "PUSH_OK https://github.com/$OWNER/$NAME"
