#!/usr/bin/env bash
# Regenerate the public demo: fetch lens if missing, then either
#   MODE=apply  (default) — run demo_apply.py and require SUCCESS
#   MODE=triad            — run fire_triad.sh with fixed TASK_ID/PROMPT
#
# Fixed demo values (illustrative / reproducible enough to prove SUCCESS):
#   TASK_ID=20261007-020907-13423
#   PROMPT="Fact: The currency used in the country shaped like a boot is"
#
# Env overrides:
#   JLENS_VENV  — python venv with torch/transformers/jlens
#   MODE        — apply | triad
#   COVENANT_SRC / AI_MEMORY_ROOT — for MODE=triad
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TASK_ID="${TASK_ID:-20261007-020907-13423}"
PROMPT="${TRIAD_PROMPT:-Fact: The currency used in the country shaped like a boot is}"
MODE="${MODE:-apply}"

# Prefer explicit JLENS_VENV, then repo .venv, then common shared path
if [[ -z "${JLENS_VENV:-}" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    JLENS_VENV="$REPO_ROOT/.venv"
  elif [[ -x /workspace/jlens-venv/bin/python ]]; then
    JLENS_VENV=/workspace/jlens-venv
  else
    JLENS_VENV="$REPO_ROOT/.venv"
  fi
fi
export JLENS_VENV

echo "=== seed_demo MODE=$MODE TASK_ID=$TASK_ID ==="
echo "REPO_ROOT=$REPO_ROOT"
echo "JLENS_VENV=$JLENS_VENV"

# 1) Lens
"$REPO_ROOT/scripts/fetch_lens.sh"
export JLENS_LENS="${JLENS_LENS:-$REPO_ROOT/gpt2_jacobian_lens.pt}"

# Resolve python
if [[ -x "$JLENS_VENV/bin/python" ]]; then
  PYTHON="$JLENS_VENV/bin/python"
elif [[ -x "$JLENS_VENV/Scripts/python.exe" ]]; then
  PYTHON="$JLENS_VENV/Scripts/python.exe"   # a Windows venv
else
  PYTHON="$(command -v python3)"
fi
echo "PYTHON=$PYTHON"

if [[ "$MODE" == "apply" ]]; then
  echo "--- demo_apply.py ---"
  TRIAD_PROMPT="$PROMPT" JLENS_LENS="$JLENS_LENS" "$PYTHON" "$REPO_ROOT/demo_apply.py" | tee /tmp/triad_seed_demo_apply.log
  grep -n -F "SUCCESS: lens.apply returned real top-k tokens" /tmp/triad_seed_demo_apply.log
  echo "SEED_DEMO_OK mode=apply"
  exit 0
fi

if [[ "$MODE" == "triad" ]]; then
  # Ensure covenant is available
  export COVENANT_SRC="${COVENANT_SRC:-$REPO_ROOT/vendor/covenant}"
  if [[ ! -f "$COVENANT_SRC/ai_memory_system/main.py" ]]; then
    echo "Cloning covenant into vendor/covenant ..."
    mkdir -p "$REPO_ROOT/vendor"
    git clone --depth 1 https://github.com/LAWLESS1987/covenant "$COVENANT_SRC"
  fi
  export AI_MEMORY_ROOT="${AI_MEMORY_ROOT:-$REPO_ROOT/data/ai_memory_root}"
  mkdir -p "$AI_MEMORY_ROOT"

  echo "--- fire_triad.sh ---"
  export TASK_ID PROMPT JLENS_VENV JLENS_LENS
  export JLENS_SNAPSHOT_PY="$REPO_ROOT/jlens_snapshot.py"
  "$REPO_ROOT/triad/fire_triad.sh" \
    "jlens-triad-e2e" \
    "Prove real Anthropic jacobian-lens as triad third leg" \
    "Installed gpt2 lens, wired fire_triad with tombstone+covenant+jlens.apply" \
    "success" \
    "$PROMPT" \
    "none" \
    "Third leg must be real jlens.apply not self-observation" \
    "seed-demo" | tee /tmp/triad_seed_demo_triad.log
  grep -n -F "FIRE_TRIAD_OK" /tmp/triad_seed_demo_triad.log
  echo "SEED_DEMO_OK mode=triad task_id=$TASK_ID"
  exit 0
fi

echo "Unknown MODE=$MODE (use apply|triad)" >&2
exit 1
