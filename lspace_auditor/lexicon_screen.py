#!/usr/bin/env python3
"""Lexicon-association screen: GPT-2, reading the draft, ranks a watched word high in the Jacobian-lens readout while
the word is absent from the text.

What it measures is GPT-2's association with the text. GPT-2 is a reader, not the speaker, so nothing here observes
the speaker's intent (threefold PR #1 review, 2026-10-08). A hit is a pointer for a person: the gate records it as
HOLD and never blocks on it. Only the deterministic fact check in couple_tombstone.py blocks."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SCREEN_LEXICON
from read import read_activations

# Rank threshold: token in top-N of vocab at a middle-layer position counts as "lit"
DEFAULT_RANK_THRESH = 50
# Need this many (layer,pos) hits to flag
DEFAULT_MIN_HITS = 3


def surface_contains(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE) is not None


def score_lexicon_screen(
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

    ranks = snap.get("screen_lexicon_ranks") or snap.get("silent_lexicon_ranks") or {}
    for word in SCREEN_LEXICON:
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
        hit = (not on_surface) and len(hits) >= min_hits
        per_word[word] = {
            "on_surface": on_surface,
            "best_rank": best_rank,
            "hits_below_thresh": len(hits),
            "hit_samples": hits[:8],
            "screen_flag": hit,
        }
        if hit:
            flagged.append(word)

    return {
        "type": "lexicon_screen",
        "rank_thresh": rank_thresh,
        "min_hits": min_hits,
        "flagged": flagged,
        "hit": bool(flagged),
        "per_word": per_word,
        "prompt_chars": len(prompt),
        "note": (
            "Screen hit = a watched word ranks high in GPT-2's Jacobian-lens readout of the text "
            "while absent from the surface text: GPT-2's word association, not the speaker's intent. "
            "Not Grok Bot residuals. A hit is recorded as HOLD for a person; it never blocks."
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
    body = score_lexicon_screen(
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
