#!/usr/bin/env bash
# Inbound LSpace silent-intent probe before formulating a reply.
set -euo pipefail
JLENS_VENV="${JLENS_VENV:-/workspace/jlens-venv}"
GATE_PY="${GATE_PY:-$(cd "$(dirname "$0")/.." && pwd)/gate.py}"
TEXT="${1:-}"
RECEIPT="${2:-}"
if [[ -z "$TEXT" ]]; then
  echo "Usage: $0 \"operator inbound text\" [receipt.json]" >&2
  exit 3
fi
TMP=$(mktemp)
printf '%s' "$TEXT" > "$TMP"
ARGS=(inbound --text-file "$TMP")
if [[ -n "$RECEIPT" ]]; then
  ARGS+=(--receipt "$RECEIPT")
fi
set +e
"$JLENS_VENV/bin/python" "$GATE_PY" "${ARGS[@]}"
EC=$?
set -e
rm -f "$TMP"
exit $EC
