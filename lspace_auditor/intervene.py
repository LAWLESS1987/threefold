#!/usr/bin/env python3
"""Write ops on the open auditor model: steer / ablate / swap concept vectors.

Proves causality on GPT-2 (spider→ant style). Not Grok Bot internals.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import concept_direction, load_model_and_lens, topk_tokens
from jlens.hooks import ActivationRecorder


def _forward_logits(hf, input_ids: torch.Tensor) -> torch.Tensor:
    out = hf(input_ids)
    return out.logits[0, -1].detach().float()  # next-token at last pos


def _lens_topk_at(model, lens, hf, tok, input_ids, layer: int, pos: int, top_k: int = 5):
    """Manual transport+unembed at one layer/pos after a hooked forward."""
    with ActivationRecorder(model.layers, at=[layer, model.n_layers - 1]) as rec:
        model.forward(input_ids)
        h = rec.activations[layer][0, pos].float()
    transported = lens.transport(h, layer)
    logits = model.unembed(transported).float()
    return topk_tokens(logits, tok, k=top_k)


def intervene(
    prompt: str,
    *,
    op: str,
    concept_a: str,
    concept_b: str = "",
    layer: int = 6,
    alpha: float = 8.0,
    top_k: int = 5,
) -> dict:
    """op in {steer, ablate, swap}. Steering applied at `layer` last position."""
    model, lens, tok, hf = load_model_and_lens()
    if layer not in lens.source_layers:
        layer = sorted(lens.source_layers)[len(lens.source_layers) // 2]

    input_ids = model.encode(prompt, max_length=1024)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    seq_len = int(input_ids.shape[-1])
    pos = seq_len - 1

    dir_a = concept_direction(hf, tok, concept_a)
    dir_b = concept_direction(hf, tok, concept_b) if concept_b else None

    # BEFORE
    before_next = topk_tokens(_forward_logits(hf, input_ids), tok, k=top_k)
    before_lens = _lens_topk_at(model, lens, hf, tok, input_ids, layer, pos, top_k)

    def make_hook(operation: str):
        def hook(module, inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            # tensor: [batch, seq, d]
            h = tensor.clone()
            d = dir_a.to(h.device, dtype=h.dtype)
            if operation == "steer":
                h[:, pos, :] = h[:, pos, :] + alpha * d
            elif operation == "ablate":
                # subtract projection onto concept direction
                v = h[:, pos, :]
                proj = (v * d).sum(dim=-1, keepdim=True) * d
                h[:, pos, :] = v - proj
            elif operation == "swap":
                assert dir_b is not None
                db = dir_b.to(h.device, dtype=h.dtype)
                v = h[:, pos, :]
                proj_a = (v * d).sum(dim=-1, keepdim=True) * d
                h[:, pos, :] = v - proj_a + alpha * db
            else:
                raise ValueError(operation)
            if torch.is_tensor(output):
                return h
            return (h,) + tuple(output[1:])

        return hook

    handle = model.layers[layer].register_forward_hook(make_hook(op))
    try:
        after_next = topk_tokens(_forward_logits(hf, input_ids), tok, k=top_k)
        after_lens = _lens_topk_at(model, lens, hf, tok, input_ids, layer, pos, top_k)
    finally:
        handle.remove()

    changed = before_next != after_next or before_lens != after_lens
    return {
        "type": "lspace_intervene",
        "op": op,
        "concept_a": concept_a,
        "concept_b": concept_b or None,
        "layer": layer,
        "pos": pos,
        "alpha": alpha,
        "prompt": prompt,
        "before_next_topk": before_next,
        "after_next_topk": after_next,
        "before_lens_topk": before_lens,
        "after_lens_topk": after_lens,
        "causality_signal": changed,
        "not_grok_bot_residuals": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The animal that spun the web was a")
    ap.add_argument("--op", choices=["steer", "ablate", "swap"], default="swap")
    ap.add_argument("--concept-a", default="spider")
    ap.add_argument("--concept-b", default="ant")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=20.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    t0 = time.perf_counter()
    body = intervene(
        args.prompt,
        op=args.op,
        concept_a=args.concept_a,
        concept_b=args.concept_b,
        layer=args.layer,
        alpha=args.alpha,
    )
    body["apply_s"] = round(time.perf_counter() - t0, 3)
    text = json.dumps(body, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return 0 if body["causality_signal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
