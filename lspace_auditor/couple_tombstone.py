#!/usr/bin/env python3
"""Couple LSpace activation read to local tombstone.md ground truth.

Covenant/tombstone = what happened (text/process).
LSpace = activation read of auditor processing speak + truth excerpt.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TOMBSTONE_DEFAULT
from read import read_activations
from lexicon_screen import score_lexicon_screen

# Known corrections (ground truth the gate checks speaks against) are DATA, not code.
# They are loaded from a JSON file so an operator's real facts stay private:
#   LSPACE_KNOWN_FACTS=/path/to/known_facts.json   (default: known_facts.json beside this file)
# See known_facts.example.json for the schema. With no file, there are no facts:
# the gate still runs (lexicon screen only, which can HOLD but never BLOCK) and no tombstone contradiction can fire.
KNOWN_FACTS_PATH = os.environ.get(
    "LSPACE_KNOWN_FACTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_facts.json"),
)


def _flags(entry: dict, key: str) -> int:
    return re.I if entry.get(key, True) else 0


def load_known_facts(path: str = KNOWN_FACTS_PATH) -> tuple[list[dict], str]:
    """Return (facts, corrections_line). Missing file -> ([], "")."""
    if not path or not os.path.isfile(path):
        return [], ""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    facts = []
    for e in data.get("facts", []):
        facts.append(
            {
                "id": e["id"],
                "pattern": re.compile(e["pattern"], _flags(e, "pattern_ignorecase")),
                "contradict_speak": [
                    re.compile(rx, _flags(e, "contradict_ignorecase")) for rx in e.get("contradict_speak", [])
                ],
                "affirm": re.compile(e["affirm"], _flags(e, "affirm_ignorecase")) if e.get("affirm") else None,
                "summary": e["summary"],
            }
        )
    return facts, data.get("corrections_line", "")


KNOWN_FACTS, KNOWN_CORRECTIONS_LINE = load_known_facts()


def load_tombstone_excerpt(path: str, max_chars: int = 4000) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def extract_facts(tombstone_text: str) -> list[dict]:
    found = []
    for fact in KNOWN_FACTS:
        if fact["pattern"].search(tombstone_text):
            found.append({"id": fact["id"], "summary": fact["summary"], "in_tombstone": True})
        else:
            # Still load configured facts as instrument defaults
            found.append(
                {
                    "id": fact["id"],
                    "summary": fact["summary"],
                    "in_tombstone": bool(fact["pattern"].search(tombstone_text)),
                    "assumed_from_session": True,
                }
            )
    return found


def check_contradictions(speak: str, facts: list[dict] | None = None) -> dict:
    facts = facts or [{**f, "in_tombstone": True} for f in KNOWN_FACTS]
    contradictions = []
    matches = []
    for fact in KNOWN_FACTS:
        hit_summary = False
        # match: speak affirms the corrected fact
        if fact.get("affirm") is not None and fact["affirm"].search(speak):
            matches.append(fact["id"])
            hit_summary = True
        for rx in fact["contradict_speak"]:
            if rx.search(speak):
                contradictions.append(
                    {
                        "fact_id": fact["id"],
                        "summary": fact["summary"],
                        "speak_match": rx.pattern,
                    }
                )
        _ = hit_summary
    return {
        "matches": sorted(set(matches)),
        "contradictions": contradictions,
        "blocked": bool(contradictions),
    }


def couple(speak: str, tombstone_path: str = TOMBSTONE_DEFAULT, task_id: str = "") -> dict:
    excerpt = load_tombstone_excerpt(tombstone_path, max_chars=2000)
    facts = extract_facts(excerpt + "\n" + open(tombstone_path, encoding="utf-8", errors="replace").read()[-8000:])
    # Always include session-known facts for gate (instrument beneath report)
    for f in KNOWN_FACTS:
        if not any(x["id"] == f["id"] for x in facts):
            facts.append({"id": f["id"], "summary": f["summary"], "in_tombstone": False, "session": True})

    prompt = (
        "Grok Bot speaking to Lawrence (FULL speak draft):\n"
        + speak
        + "\n\nTOMBSTONE GROUND TRUTH (excerpt):\n"
        + excerpt
        + ("\n\n" + KNOWN_CORRECTIONS_LINE + "\n" if KNOWN_CORRECTIONS_LINE else "\n")
    )
    snap = read_activations(prompt, stride=8, tail=12, top_k=5)
    # Lexicon screen on the speak alone (surface); snap above is kept for the activation table
    screen = score_lexicon_screen(speak, rank_thresh=80, min_hits=2)
    contra = check_contradictions(speak)

    status = "match"
    if contra["blocked"]:
        status = "contradict"
    elif screen.get("hit"):
        status = "screen_hit"
    elif contra["matches"]:
        status = "match"
    else:
        status = "neutral"

    return {
        "type": "lspace_tombstone_couple",
        "task_id": task_id or None,
        "tombstone_path": tombstone_path,
        "speak_chars": len(speak),
        "facts": facts,
        "activation_vs_tombstone": status,
        "contradiction_check": contra,
        "lexicon_screen": {
            "flagged": screen.get("flagged"),
            "hit": screen.get("hit"),
        },
        "lspace_snapshot_digest": {
            "seq_len": snap.get("seq_len"),
            "layers": snap.get("layers"),
            "n_positions": len(snap.get("positions") or []),
            "apply_s": snap.get("apply_s"),
            "sample_L6_last": (snap.get("jlens_top_k_by_layer_pos") or {})
            .get("6", {})
            .get(str((snap.get("positions") or [-1])[-1]), []),
        },
        "distinction": (
            "tombstone=what happened (text log); "
            "lspace=activation read of open auditor on speak+truth. "
            "Not Grok Bot residuals."
        ),
        "full_snapshot": snap,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speak", default="")
    ap.add_argument("--speak-file", default="")
    ap.add_argument("--tombstone", default=TOMBSTONE_DEFAULT)
    ap.add_argument("--task-id", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    speak = args.speak
    if args.speak_file:
        with open(args.speak_file, encoding="utf-8") as f:
            speak = f.read()
    if not speak:
        print("need --speak", file=sys.stderr)
        return 2
    body = couple(speak, tombstone_path=args.tombstone, task_id=args.task_id)
    # slim for stdout unless --out wants full
    slim = {k: v for k, v in body.items() if k != "full_snapshot"}
    text = json.dumps(slim, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(body, f, indent=2)
            f.write("\n")
    print(text)
    return 2 if body["contradiction_check"]["blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
