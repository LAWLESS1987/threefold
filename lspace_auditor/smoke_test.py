#!/usr/bin/env python3
"""Smoke: multi-pos read, lexicon screen, intervene swap, tombstone couple, gate demos (BLOCK / HOLD / ALLOW)."""
from __future__ import annotations

import json
import os
import sys
import traceback

if hasattr(sys.stdout, "reconfigure"):  # cp1252 consoles (Windows) cannot print the arrows in the log
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE_DIR = os.environ.get("LSPACE_SMOKE_DIR", _HERE)
REPORT = os.path.join(SMOKE_DIR, "SMOKE_REPORT.md")
OUT_DIR = os.path.join(SMOKE_DIR, "smoke_out")
os.makedirs(OUT_DIR, exist_ok=True)

# Generic fixtures only: the smoke never reads an operator's real facts, tombstone or live state
# unless the caller points these variables at them on purpose.
os.environ.setdefault("LSPACE_KNOWN_FACTS", os.path.join(_HERE, "known_facts.example.json"))
os.environ.setdefault("LSPACE_AUTOTUNE_DIR", os.path.join(OUT_DIR, "state"))
os.environ.setdefault("LSPACE_HOLDS", os.path.join(OUT_DIR, "holds.jsonl"))
_EXAMPLE_TOMBSTONE = os.path.join(OUT_DIR, "tombstone_example.md")
if "LSPACE_TOMBSTONE" not in os.environ:
    with open(_EXAMPLE_TOMBSTONE, "w", encoding="utf-8") as _f:
        _f.write("## example task\n- outcome: the example nightly ran 11 builds; all green.\n")
    os.environ["LSPACE_TOMBSTONE"] = _EXAMPLE_TOMBSTONE


def main() -> int:
    lines: list[str] = []
    ok = True

    def log(s: str = "") -> None:
        lines.append(s)
        print(s)

    log("# LSpace smoke report")
    log()
    log("Instrument: **speak-text LSpace auditor** (open GPT-2 + JacobianLens).")
    log("**Not** Grok Bot residuals. Demoted proxy_jlens is not used as gate.")
    log()

    # --- (c) multi-pos read ---
    try:
        from read import read_activations

        speak = (
            "Status: the example nightly ran 11 builds. "
            "All checks are green and the report is filed."
        )
        snap = read_activations(speak, stride=4, tail=16, top_k=5)
        with open(f"{OUT_DIR}/read.json", "w") as f:
            json.dump(snap, f, indent=2)
        log("## (c) Multi-position read")
        log(f"- seq_len={snap['seq_len']} layers={snap['layers']} n_pos={len(snap['positions'])}")
        log(f"- apply_s={snap['apply_s']} prompt_chars={snap['prompt_chars']} (no 480 truncate)")
        log("- sample table (layer → last-pos top-5):")
        last = str(snap["positions"][-1])
        for L in snap["layers"]:
            toks = snap["jlens_top_k_by_layer_pos"][str(L)].get(last, [])
            log(f"  - L{L}@{last}: {toks}")
        log()
    except Exception:
        ok = False
        log("## (c) Multi-position read FAILED")
        log("```")
        log(traceback.format_exc())
        log("```")

    # --- lexicon screen (GPT-2's word association while reading the draft; not the speaker's intent) ---
    try:
        from lexicon_screen import score_lexicon_screen

        screen = score_lexicon_screen(speak, snapshot=snap)
        with open(f"{OUT_DIR}/lexicon_screen.json", "w") as f:
            json.dump(screen, f, indent=2)
        log("## Lexicon-screen scores")
        log(f"- flagged={screen['flagged']} hit={screen['hit']}")
        for w, v in screen["per_word"].items():
            log(
                f"  - {w}: on_surface={v['on_surface']} best_rank={v['best_rank']} "
                f"hits={v['hits_below_thresh']} screen_flag={v['screen_flag']}"
            )
        log()
    except Exception:
        ok = False
        log("## Lexicon screen FAILED")
        log("```")
        log(traceback.format_exc())
        log("```")

    # --- intervene swap ---
    try:
        from intervene import intervene

        iv = intervene(
            "The small creature that spun the web in the garden was a",
            op="swap",
            concept_a="spider",
            concept_b="ant",
            layer=6,
            alpha=25.0,
        )
        with open(f"{OUT_DIR}/intervene_swap.json", "w") as f:
            json.dump(iv, f, indent=2)
        log("## Steer/ablate/swap causality (spider→ant)")
        log(f"- before next: {iv['before_next_topk']}")
        log(f"- after next:  {iv['after_next_topk']}")
        log(f"- before lens: {iv['before_lens_topk']}")
        log(f"- after lens:  {iv['after_lens_topk']}")
        log(f"- causality_signal={iv['causality_signal']}")
        if not iv["causality_signal"]:
            # try stronger alpha / steer
            iv2 = intervene(
                "The animal was a",
                op="steer",
                concept_a="ant",
                concept_b="",
                layer=6,
                alpha=40.0,
            )
            log(f"- fallback steer ant: before={iv2['before_next_topk']} after={iv2['after_next_topk']} changed={iv2['causality_signal']}")
            if not iv2["causality_signal"]:
                ok = False
                log("- WARN: no token flip observed (still recorded)")
        log()
    except Exception:
        ok = False
        log("## Intervene FAILED")
        log("```")
        log(traceback.format_exc())
        log("```")

    # --- (d) tombstone couple ---
    try:
        from couple_tombstone import couple

        # Correct speak should match
        good = couple(speak)
        with open(f"{OUT_DIR}/couple_good.json", "w") as f:
            json.dump({k: v for k, v in good.items() if k != "full_snapshot"}, f, indent=2)
        # Bad speak: contradicts the example fact (12 builds, not 11)
        bad_speak = "The example nightly ran 12 builds and two were skipped."
        bad = couple(bad_speak)
        with open(f"{OUT_DIR}/couple_bad.json", "w") as f:
            json.dump({k: v for k, v in bad.items() if k != "full_snapshot"}, f, indent=2)
        log("## (d) Tombstone coupling")
        log(f"- good speak status={good['activation_vs_tombstone']} contradictions={good['contradiction_check']['contradictions']}")
        log(f"- bad speak status={bad['activation_vs_tombstone']}")
        for c in bad["contradiction_check"]["contradictions"]:
            log(f"  - CONTRADICT {c['fact_id']}: {c['summary']}")
        if not bad["contradiction_check"]["blocked"]:
            ok = False
            log("- FAIL: expected tombstone contradictions on bad speak")
        log(f"- digest good L6 sample: {good.get('lspace_snapshot_digest')}")
        log()
    except Exception:
        ok = False
        log("## Tombstone couple FAILED")
        log("```")
        log(traceback.format_exc())
        log("```")

    # --- gate demos ---
    try:
        from gate import gate_inbound, gate_outbound

        log("## Gate smoke")
        inbound = gate_inbound(
            "Please confirm the example nightly ran 11 builds.",
            receipt_path=f"{OUT_DIR}/gate_inbound.json",
        )
        log(f"- inbound decision={inbound['decision']} elapsed={inbound['elapsed_s']}s sample={inbound['lspace']['sample_mid']}")

        outbound = gate_outbound(
            speak,
            receipt_path=f"{OUT_DIR}/gate_outbound_allow.json",
            skip_full_couple_snapshot=True,
        )
        log(f"- outbound ALLOW path decision={outbound['decision']} chars={outbound['text_chars']} elapsed={outbound['elapsed_s']}s")

        held_text = "This report is complete and accurate."
        hold_screen = gate_outbound(
            held_text,
            receipt_path=f"{OUT_DIR}/gate_outbound_screen_hold.json",
            force_screen_hit=True,
            skip_full_couple_snapshot=True,
        )
        log(f"- forced lexicon-screen HOLD decision={hold_screen['decision']} flag={hold_screen.get('flag_message')}")

        block_tomb = gate_outbound(
            "Correction: there were 12 builds in the example nightly.",
            receipt_path=f"{OUT_DIR}/gate_outbound_tomb_block.json",
            skip_full_couple_snapshot=True,
        )
        log(f"- tombstone-contradiction BLOCK decision={block_tomb['decision']}")
        log(f"  reasons={block_tomb['reasons']}")
        if hold_screen["decision"] != "HOLD" or block_tomb["decision"] != "BLOCK":
            ok = False
            log("- FAIL: expected HOLD on the forced screen hit and BLOCK on the tombstone contradiction")
        import hashlib
        holds_path = os.environ["LSPACE_HOLDS"]
        holds_text = open(holds_path, encoding="utf-8").read() if os.path.exists(holds_path) else ""
        rows = [json.loads(l) for l in holds_text.splitlines() if l.strip()]
        want = hashlib.sha256(held_text.encode("utf-8")).hexdigest()
        held_rows = [r for r in rows if r.get("decision") == "HOLD" and r.get("draft_sha256") == want]
        log(f"- holds.jsonl: {len(rows)} row(s); the HOLD is recorded by hash: {bool(held_rows)}; "
            f"no draft text in the file: {held_text not in holds_text}; label left for a person: "
            f"{bool(held_rows) and held_rows[-1].get('label') is None}")
        if not held_rows or held_text in holds_text or held_rows[-1].get("label") is not None:
            ok = False
            log("- FAIL: a HOLD must be recorded by its hash, with no text and an empty label")
        if not any(r.get("decision") == "BLOCK" for r in rows):
            ok = False
            log("- FAIL: the BLOCK was not recorded in holds.jsonl")
        if outbound["decision"] != "ALLOW":
            log(f"- WARN: clean outbound not ALLOW ({outbound['decision']})")
        log()
    except Exception:
        ok = False
        log("## Gate FAILED")
        log("```")
        log(traceback.format_exc())
        log("```")

    log("## Verdict")
    log(f"- smoke_ok={ok}")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"WROTE {REPORT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
