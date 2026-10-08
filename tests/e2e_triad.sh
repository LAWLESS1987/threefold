#!/usr/bin/env bash
# e2e_triad.sh — the whole triad on a real model, in a scratch directory, driven both ways.
#
# Runs triad/on_speak.sh with the tombstone log, the memory root and the run
# directory all inside a fresh temp dir. It never touches tombstone/tombstone.md,
# and never pushes: the scratch memory root is not a git repo, so the push step
# is skipped. Then it breaks the triad on purpose, and each break must fail:
#   1. a tombstone entry citing another L-lens artifact  -> verify_triad exits 1
#   2. the L-lens memory removed from covenant            -> verify_triad exits 1
#   3. a lens file that is not a lens                     -> fire_triad exits
#      nonzero BEFORE the tombstone is written (fail closed, nothing appended)
#
# Needs: the venv (JLENS_VENV, default <repo>/.venv), the lens file, the model,
# rg, and covenant's ai_memory_system (COVENANT_SRC).
# Prints E2E_TRIAD_OK when every step behaved as stated.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
export JLENS_VENV="${JLENS_VENV:-$REPO_ROOT/.venv}"
export JLENS_LENS="${JLENS_LENS:-$REPO_ROOT/gpt2_jacobian_lens.pt}"
export COVENANT_SRC="${COVENANT_SRC:?set COVENANT_SRC to a covenant checkout}"
export TOMBSTONE_LOG="$SCRATCH/tombstone.md"
export AI_MEMORY_ROOT="$SCRATCH/memory"
export RUN_DIR="$SCRATCH/runs"
if [[ -x "$JLENS_VENV/bin/python" ]]; then PY="$JLENS_VENV/bin/python"; else PY="$JLENS_VENV/Scripts/python.exe"; fi
printf '# scratch tombstone log\n' > "$TOMBSTONE_LOG"
mkdir -p "$AI_MEMORY_ROOT"
SAMPLE_BEFORE="$(sha256sum "$REPO_ROOT/tombstone/tombstone.md" | cut -c1-64)"

fail() { echo "E2E FAIL: $*" >&2; exit 1; }

echo "=== 1. a real speak-turn fires all three legs and verifies ==="
export TASK_ID="e2e-$(date -u +%Y%m%d-%H%M%S)-$$"
OUT="$("$REPO_ROOT/triad/on_speak.sh" "The covenant keeps its record append-only, and every claim cites the check that backs it." "e2e")" \
  || { printf '%s\n' "$OUT"; fail "on_speak.sh exited nonzero on a clean run"; }
printf '%s\n' "$OUT" | grep -E "SNAPSHOT_OK|LLENS_OK|LLENS_VERIFY_OK|VERIFY_TRIAD_OK|FIRE_TRIAD_OK|Wrote covenant"
printf '%s\n' "$OUT" | grep -q "^VERIFY_TRIAD_OK task_id=$TASK_ID" || fail "no VERIFY_TRIAD_OK"
printf '%s\n' "$OUT" | grep -q "^FIRE_TRIAD_OK task_id=$TASK_ID" || fail "no FIRE_TRIAD_OK"
for m in task jlens llens; do [[ -f "$AI_MEMORY_ROOT/$m-$TASK_ID.md" ]] || fail "covenant memory $m-$TASK_ID absent"; done
V=("$PY" "$REPO_ROOT/triad/verify_triad.py" --task-id "$TASK_ID" --run-dir "$RUN_DIR/$TASK_ID"
   --tombstone-log "$TOMBSTONE_LOG" --memory-root "$AI_MEMORY_ROOT")
"${V[@]}" >/dev/null || fail "verify_triad fails on the clean triad"

echo "=== 2. break: the tombstone cites another L-lens artifact ==="
cp "$TOMBSTONE_LOG" "$SCRATCH/tombstone.bak"
sed -i -E "s/llens_sha256=[0-9a-f]{64}/llens_sha256=$(printf '0%.0s' {1..64})/" "$TOMBSTONE_LOG"
if "${V[@]}" > "$SCRATCH/v2.txt"; then fail "verify_triad passed with a wrong tombstone digest"; fi
grep "FAIL  tombstone: llens_sha256 agrees" "$SCRATCH/v2.txt" || fail "the wrong digest was not the named failure"
cp "$SCRATCH/tombstone.bak" "$TOMBSTONE_LOG"
"${V[@]}" >/dev/null || fail "restoring the tombstone did not restore green"

echo "=== 3. break: the L-lens memory is gone from covenant ==="
mv "$AI_MEMORY_ROOT/llens-$TASK_ID.md" "$SCRATCH/llens.bak"
if "${V[@]}" > "$SCRATCH/v3.txt"; then fail "verify_triad passed without the L-lens memory"; fi
grep "FAIL  covenant: llens-$TASK_ID present" "$SCRATCH/v3.txt" || fail "the missing memory was not the named failure"
mv "$SCRATCH/llens.bak" "$AI_MEMORY_ROOT/llens-$TASK_ID.md"
"${V[@]}" >/dev/null || fail "restoring the memory did not restore green"

echo "=== 4. break: the lens leg fails; nothing may be appended ==="
LINES_BEFORE="$(wc -l < "$TOMBSTONE_LOG")"
printf 'not a lens\n' > "$SCRATCH/bad_lens.pt"
if JLENS_LENS="$SCRATCH/bad_lens.pt" TASK_ID="e2e-bad-$$" "$REPO_ROOT/triad/on_speak.sh" "x" "e2e-bad" > "$SCRATCH/bad.txt" 2>&1; then
  fail "fire_triad succeeded with a broken lens"
fi
[[ "$(wc -l < "$TOMBSTONE_LOG")" == "$LINES_BEFORE" ]] || fail "the tombstone was written although the lens leg failed"
[[ ! -e "$AI_MEMORY_ROOT/task-e2e-bad-$$.md" ]] || fail "a covenant memory was written although the lens leg failed"
echo "lens leg failed as it should; tombstone and covenant untouched"

[[ "$(sha256sum "$REPO_ROOT/tombstone/tombstone.md" | cut -c1-64)" == "$SAMPLE_BEFORE" ]] || fail "the repo's tombstone sample changed"
echo "E2E_TRIAD_OK task_id=$TASK_ID"
