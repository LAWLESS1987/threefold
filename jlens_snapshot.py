#!/usr/bin/env python3
"""Run real Anthropic jlens.JacobianLens.apply and write a triad snapshot.

Usage:
  jlens_snapshot.py --task-id TID --prompt "..." --out SNAPSHOT_PATH [--digest DIGEST_PATH]

Exits nonzero if apply fails or top-k is empty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import transformers
import jlens

MODEL_ID = "openai-community/gpt2"
# Default: lens next to this script (after scripts/fetch_lens.sh)
DEFAULT_LENS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpt2_jacobian_lens.pt")
TOP_K = 5


def topk_tokens(logits: torch.Tensor, tok, k: int = TOP_K) -> list[str]:
    ids = logits.topk(k).indices.tolist()
    return [tok.decode([t]) for t in ids]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True, help="JSON snapshot path")
    ap.add_argument("--digest", default="", help="optional short digest text path")
    ap.add_argument("--lens", default=DEFAULT_LENS)
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()

    t0 = time.perf_counter()
    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model)
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf, tok)

    lens = jlens.JacobianLens.load(args.lens)

    layers = [
        model.n_layers // 4,
        model.n_layers // 2,
        (model.n_layers * 3) // 4,
        model.n_layers - 2,
    ]
    layers = [L for L in layers if L in lens.jacobians]
    if not layers:
        layers = list(lens.source_layers)[:4]

    lens_logits, model_logits, _ = lens.apply(
        model, args.prompt, layers=layers, positions=[-2]
    )

    per_layer = {}
    lines = []
    for layer in layers:
        toks = topk_tokens(lens_logits[layer][0], tok)
        if not toks:
            print(f"ERROR: empty top-k at layer {layer}", file=sys.stderr)
            return 1
        per_layer[str(layer)] = toks
        lines.append(f"L{layer}: {toks}")

    model_top = topk_tokens(model_logits[0], tok)
    body = {
        "type": "jlens",
        "task_id": args.task_id,
        "model_id": args.model,
        "lens_path": args.lens,
        "prompt": args.prompt,
        "layers": layers,
        "positions": [-2],
        "top_k": TOP_K,
        "jlens_top_k": per_layer,
        "model_final_top_k": model_top,
        "apply_s": round(time.perf_counter() - t0, 3),
        "mechanism": "anthropics/jacobian-lens JacobianLens.apply (real per-layer logits)",
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2)
        f.write("\n")

    digest = (
        f"jlens task_id={args.task_id} model={args.model} "
        f"layers={layers} topk={{{'; '.join(lines)}}}"
    )
    digest_hash = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()
    ).hexdigest()[:16]
    digest_line = f"{digest} digest={digest_hash}"

    if args.digest:
        with open(args.digest, "w", encoding="utf-8") as f:
            f.write(digest_line + "\n")
            for line in lines:
                f.write(line + "\n")

    print(digest_line)
    for line in lines:
        print(line)
    print(f"SNAPSHOT_OK path={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
