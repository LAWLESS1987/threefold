#!/usr/bin/env bash
# Download the pretrained gpt2-small Jacobian lens from Hugging Face
# (neuronpedia/jacobian-lens) into the repo root.
#
# Source path on the Hub:
#   gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt
#
# Usage:
#   ./scripts/fetch_lens.sh
#   OUT=/path/to/gpt2_jacobian_lens.pt ./scripts/fetch_lens.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-$REPO_ROOT/gpt2_jacobian_lens.pt}"
REPO_ID="${HF_REPO_ID:-neuronpedia/jacobian-lens}"
HF_PATH="${HF_LENS_PATH:-gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt}"

if [[ -f "$OUT" ]]; then
  echo "Lens already present: $OUT ($(du -h "$OUT" | awk '{print $1}'))"
  exit 0
fi

echo "Fetching $REPO_ID / $HF_PATH -> $OUT"

# Prefer huggingface-cli / huggingface_hub if available
if command -v huggingface-cli >/dev/null 2>&1; then
  TMPDIR="$(mktemp -d)"
  huggingface-cli download "$REPO_ID" "$HF_PATH" --local-dir "$TMPDIR" --local-dir-use-symlinks False
  mv "$TMPDIR/$HF_PATH" "$OUT"
  rm -rf "$TMPDIR"
elif python3 -c "import huggingface_hub" 2>/dev/null; then
  python3 - <<PY
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download(repo_id="${REPO_ID}", filename="${HF_PATH}")
shutil.copy2(path, "${OUT}")
print(f"copied {path} -> ${OUT}")
PY
else
  echo "Need huggingface_hub (pip install huggingface_hub) or huggingface-cli" >&2
  exit 1
fi

ls -lh "$OUT"
echo "FETCH_LENS_OK path=$OUT"
