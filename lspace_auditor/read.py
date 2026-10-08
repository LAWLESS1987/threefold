#!/usr/bin/env python3
"""Multi-position JacobianLens read on FULL prompt (speak-text LSpace auditor).

lens_ℓ(h) = unembed(J_ℓ @ h). Never claims Grok Bot residuals.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    MIDDLE_LAYERS,
    MAX_SEQ_LEN,
    SILENT_LEXICON,
    last_window_text,
    load_model_and_lens,
    select_positions,
    token_id,
    topk_tokens,
)


def read_activations(
    prompt: str,
    *,
    layers: list[int] | None = None,
    stride: int = 4,
    tail: int = 16,
    top_k: int = 5,
    max_seq_len: int = MAX_SEQ_LEN,
) -> dict:
    model, lens, tok, _hf = load_model_and_lens()
    layers = layers or [L for L in MIDDLE_LAYERS if L in lens.source_layers]
    window_text, tok_start, tok_end = last_window_text(prompt, tok, max_seq_len)
    t0 = time.perf_counter()
    lens_logits, model_logits, input_ids = lens.apply(
        model,
        window_text,
        layers=layers,
        positions=None,  # all positions in window; we subsample after
        max_seq_len=max_seq_len,
        use_jacobian=True,
    )
    seq_len = int(input_ids.shape[-1]) if hasattr(input_ids, "shape") else len(input_ids[0])
    # input_ids may be 1d or 2d depending on encode
    if hasattr(input_ids, "ndim") and input_ids.ndim == 2:
        seq_len = int(input_ids.shape[-1])
    elif hasattr(input_ids, "shape"):
        seq_len = int(input_ids.shape[0]) if input_ids.ndim == 1 else int(input_ids.shape[-1])
    else:
        seq_len = len(input_ids)

    positions = select_positions(seq_len, stride=stride, tail=tail)
    # lens_logits[layer] is [seq_len, vocab] when positions=None
    per_layer_pos: dict[str, dict[str, list[str]]] = {}
    silent_ranks: dict[str, dict[str, dict[str, int | None]]] = {}

    lexicon_ids = {w: token_id(tok, w) for w in SILENT_LEXICON}

    for layer in layers:
        logits = lens_logits[layer]  # [seq, V]
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        layer_key = str(layer)
        per_layer_pos[layer_key] = {}
        silent_ranks[layer_key] = {}
        for p in positions:
            row = logits[p]
            toks = topk_tokens(row, tok, k=top_k)
            per_layer_pos[layer_key][str(p)] = toks
            # rank of each silent lexicon token (0 = top)
            ranks: dict[str, int | None] = {}
            # argsort descending
            order = row.argsort(descending=True)
            rank_of = {int(tid): int(r) for r, tid in enumerate(order.tolist())}
            for w, tid in lexicon_ids.items():
                ranks[w] = rank_of.get(tid) if tid is not None else None
            silent_ranks[layer_key][str(p)] = ranks

    # model final at selected positions
    if model_logits.ndim == 1:
        model_logits = model_logits.unsqueeze(0)
    model_pos = {str(p): topk_tokens(model_logits[p], tok, k=top_k) for p in positions}

    return {
        "type": "lspace",
        "instrument": "speak-text-lspace-auditor",
        "not_grok_bot_residuals": True,
        "model_id": "openai-community/gpt2",
        "prompt_chars": len(prompt),
        "window_chars": len(window_text),
        "token_window": [tok_start, tok_end],
        "seq_len": seq_len,
        "layers": layers,
        "positions": positions,
        "top_k": top_k,
        "jlens_top_k_by_layer_pos": per_layer_pos,
        "model_final_top_k_by_pos": model_pos,
        "silent_lexicon_ranks": silent_ranks,
        "apply_s": round(time.perf_counter() - t0, 3),
        "mechanism": (
            "anthropics/jacobian-lens JacobianLens.apply multi-pos; "
            "speak-text auditor (NOT Grok Bot residuals; NOT demoted proxy)"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="LSpace multi-pos read (speak-text auditor)")
    ap.add_argument("--prompt", default="", help="full prompt text")
    ap.add_argument("--prompt-file", default="", help="read prompt from file")
    ap.add_argument("--out", default="", help="write JSON snapshot")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--tail", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8", errors="replace") as f:
            prompt = f.read()
    if not prompt:
        print("need --prompt or --prompt-file", file=sys.stderr)
        return 2
    body = read_activations(prompt, stride=args.stride, tail=args.tail, top_k=args.top_k)
    text = json.dumps(body, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
