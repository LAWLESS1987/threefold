#!/usr/bin/env python3
"""Score silent-intent lexicon tokens high in lens ranks but absent from surface text."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SILENT_LEXICON
from read import read_activations

# Rank threshold: token in top-N of vocab at a middle-layer position counts as "lit"
DEFAULT_RANK_THRESH = 50
# Need this many (layer,pos) hits to flag
DEFAULT_MIN_HITS = 3


def surface_contains(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE) is not None


def score_silent_intent(
    prompt: str,
    *,
    snapshot: dict | None = None,
    rank_thresh: int = DEFAULT_RANK_THRESH,
    min_hits: int = DEFAULT_MIN_HITS,
) -> dict:
    snap = snapshot or read_activations(prompt)
    surface = prompt
    per_word: dict[str, dict] = {}
    flagged: list[str] = []

    ranks = snap.get("silent_lexicon_ranks") or {}
    for word in SILENT_LEXICON:
        hits = []
        best_rank = None
        for layer, posmap in ranks.items():
            for pos, wranks in posmap.items():
                r = wranks.get(word)
                if r is None:
                    continue
                if best_rank is None or r < best_rank:
                    best_rank = r
                if r < rank_thresh:
                    hits.append({"layer": int(layer), "pos": int(pos), "rank": r})
        on_surface = surface_contains(surface, word)
        silent = (not on_surface) and len(hits) >= min_hits
        per_word[word] = {
            "on_surface": on_surface,
            "best_rank": best_rank,
            "hits_below_thresh": len(hits),
            "hit_samples": hits[:8],
            "silent_flag": silent,
        }
        if silent:
            flagged.append(word)

    return {
        "type": "lspace_silent_intent",
        "rank_thresh": rank_thresh,
        "min_hits": min_hits,
        "flagged": flagged,
        "blocked": bool(flagged),
        "per_word": per_word,
        "prompt_chars": len(prompt),
        "note": (
            "Silent = lexicon token lit in Jacobian-lens ranks on open GPT-2 auditor "
            "but not present as substring/word in surface text. Not Grok Bot residuals."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="")
    ap.add_argument("--prompt-file", default="")
    ap.add_argument("--snapshot", default="", help="optional prior read JSON")
    ap.add_argument("--rank-thresh", type=int, default=DEFAULT_RANK_THRESH)
    ap.add_argument("--min-hits", type=int, default=DEFAULT_MIN_HITS)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt = f.read()
    snap = None
    if args.snapshot:
        with open(args.snapshot, encoding="utf-8") as f:
            snap = json.load(f)
    if not prompt and not snap:
        print("need prompt", file=sys.stderr)
        return 2
    if not prompt and snap:
        prompt = snap.get("prompt") or ""
    body = score_silent_intent(
        prompt, snapshot=snap, rank_thresh=args.rank_thresh, min_hits=args.min_hits
    )
    text = json.dumps(body, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return 0 if not body["blocked"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
