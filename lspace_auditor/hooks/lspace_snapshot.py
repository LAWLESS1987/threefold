#!/usr/bin/env python3
"""Thin wrapper: write lspace_snapshot.json for a full speak via the auditor's read.py (LSPACE_DIR)."""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.environ.get("LSPACE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from read import read_activations
from silent_intent import score_silent_intent

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--prompt-file", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--digest", default="")
    args = ap.parse_args()
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt = f.read()
    snap = read_activations(prompt, stride=6, tail=16, top_k=5)
    snap["task_id"] = args.task_id
    silent = score_silent_intent(prompt, snapshot=snap)
    snap["silent_intent_summary"] = {
        "flagged": silent.get("flagged"),
        "blocked": silent.get("blocked"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2)
        f.write("\n")
    last = str((snap.get("positions") or [-1])[-1])
    sample = []
    for L in (snap.get("layers") or [])[:4]:
        toks = (snap.get("jlens_top_k_by_layer_pos") or {}).get(str(L), {}).get(last, [])
        sample.append(f"L{L}: {toks}")
    digest = (
        f"lspace task_id={args.task_id} seq={snap.get('seq_len')} "
        f"npos={len(snap.get('positions') or [])} silent={silent.get('flagged')} "
        f"sample={{{'; '.join(sample)}}}"
    )
    if args.digest:
        with open(args.digest, "w", encoding="utf-8") as f:
            f.write(digest + "\n")
    print(digest)
    print(f"LSPACE_SNAPSHOT_OK path={args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
