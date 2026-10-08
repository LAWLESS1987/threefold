#!/usr/bin/env python3
"""Operating gate: route inbound/outbound through speak-text LSpace auditor + tombstone.

Exit codes:
  0 = ALLOW
  2 = BLOCK (silent intent and/or tombstone contradiction)
  3 = hard error

Never uses demoted GPT-2 proxy as gate. Never claims Grok Bot residuals.

Outbound silent-intent thresholds are loaded from autotune state and updated
automatically on every silent-intent BLOCK (no manual step).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TOMBSTONE_DEFAULT
from couple_tombstone import check_contradictions, couple
from read import read_activations
from silent_intent import score_silent_intent

try:
    import autotune as _autotune
except ImportError:  # pragma: no cover
    _autotune = None

# Slightly looser for inbound (user messages); outbound baseline via autotune
INBOUND_RANK_THRESH = 40
INBOUND_MIN_HITS = 4
OUTBOUND_RANK_THRESH = 50  # baseline fallback if autotune unavailable
OUTBOUND_MIN_HITS = 3


def _write_receipt(path: str, body: dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2)
        f.write("\n")


def _outbound_thresholds(state_dir: str | None, category_hint: str = "main") -> tuple[int, int, dict]:
    """Load current outbound thresholds. Pre-score uses main; meta re-score optional."""
    if _autotune is None:
        return OUTBOUND_RANK_THRESH, OUTBOUND_MIN_HITS, {}
    sdir = _autotune.resolve_state_dir(state_dir)
    st = _autotune.load_state(sdir)
    # First pass always uses main thresholds; if text is meta we re-score with meta.
    rt, mh = _autotune.get_thresholds(st, "main")
    snap = _autotune.snapshot_for_receipt(sdir)
    return rt, mh, {"state_dir": sdir, "snapshot": snap, "state": st}


def gate_inbound(text: str, *, receipt_path: str = "", force_block_silent: bool = False) -> dict:
    t0 = time.perf_counter()
    prompt = f"Lawrence inbound message to Grok Bot:\n{text}"
    snap = read_activations(prompt, stride=6, tail=16, top_k=5)
    silent = score_silent_intent(
        text, snapshot=snap, rank_thresh=INBOUND_RANK_THRESH, min_hits=INBOUND_MIN_HITS
    )
    if force_block_silent:
        silent = dict(silent)
        silent["flagged"] = list(set(silent.get("flagged") or []) | {"fake"})
        silent["blocked"] = True
        silent["forced_demo"] = True

    decision = "ALLOW"
    reasons = []
    if silent.get("blocked"):
        decision = "BLOCK"
        reasons.append({"kind": "silent_intent", "flagged": silent.get("flagged")})

    body = {
        "type": "lspace_gate_receipt",
        "direction": "inbound",
        "decision": decision,
        "reasons": reasons,
        "silent_intent": silent,
        "lspace": {
            "seq_len": snap.get("seq_len"),
            "layers": snap.get("layers"),
            "n_positions": len(snap.get("positions") or []),
            "apply_s": snap.get("apply_s"),
            "sample_mid": (snap.get("jlens_top_k_by_layer_pos") or {})
            .get("6", {})
            .get(str((snap.get("positions") or [-1])[-1]), []),
        },
        "tombstone_check": None,
        "text_chars": len(text),
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "ts": datetime.now(timezone.utc).isoformat(),
        "instrument": "speak-text-lspace-auditor",
        "not_grok_bot_residuals": True,
        "not_proxy_jlens": True,
        "autotune": {"enabled": False, "note": "inbound uses fixed constants; auto-tune is outbound-only"},
    }
    _write_receipt(receipt_path, body)
    return body


def gate_outbound(
    text: str,
    *,
    receipt_path: str = "",
    tombstone: str = TOMBSTONE_DEFAULT,
    force_block_silent: bool = False,
    force_block_tombstone: bool = False,
    skip_full_couple_snapshot: bool = False,
    autotune_dir: str | None = None,
    verdict: str | None = None,
) -> dict:
    t0 = time.perf_counter()
    at_info: dict = {"enabled": False}
    state_dir = None
    if _autotune is not None:
        state_dir = _autotune.resolve_state_dir(autotune_dir)
        st = _autotune.load_state(state_dir)
        # Classify category early so we score with the right threshold set
        cat_early = _autotune.classify_category(text, flagged=None)
        rt, mh = _autotune.get_thresholds(st, cat_early)
        at_info = {
            "enabled": True,
            "state_dir": state_dir,
            "category_pre": cat_early,
            "thresholds_used": {"rank_thresh": rt, "min_hits": mh, "category": cat_early},
            "snapshot_pre": _autotune.snapshot_for_receipt(state_dir),
        }
    else:
        rt, mh = OUTBOUND_RANK_THRESH, OUTBOUND_MIN_HITS

    # 1) silent intent on FULL speak (no 480 truncate)
    snap = read_activations(
        f"Grok Bot speaking to Lawrence:\n{text}",
        stride=6,
        tail=16,
        top_k=5,
    )
    silent = score_silent_intent(text, snapshot=snap, rank_thresh=rt, min_hits=mh)

    # Re-classify with flagged words; if category flipped to meta and thresholds differ, re-score
    if _autotune is not None and state_dir is not None:
        cat2 = _autotune.classify_category(text, flagged=silent.get("flagged"))
        if cat2 != at_info.get("category_pre"):
            st = _autotune.load_state(state_dir)
            rt2, mh2 = _autotune.get_thresholds(st, cat2)
            if (rt2, mh2) != (rt, mh):
                silent = score_silent_intent(
                    text, snapshot=snap, rank_thresh=rt2, min_hits=mh2
                )
                rt, mh = rt2, mh2
            at_info["category_pre"] = cat2
            at_info["thresholds_used"] = {
                "rank_thresh": rt,
                "min_hits": mh,
                "category": cat2,
            }

    if force_block_silent:
        silent = dict(silent)
        silent["flagged"] = list(set(silent.get("flagged") or []) | {"fraud"})
        silent["blocked"] = True
        silent["forced_demo"] = True

    # 2) tombstone cross-check
    if skip_full_couple_snapshot:
        contra = check_contradictions(text)
        couple_status = "contradict" if contra["blocked"] else ("match" if contra["matches"] else "neutral")
        couple_slim = {
            "activation_vs_tombstone": couple_status,
            "contradiction_check": contra,
        }
    else:
        coupled = couple(text, tombstone_path=tombstone)
        contra = coupled["contradiction_check"]
        couple_slim = {
            "activation_vs_tombstone": coupled["activation_vs_tombstone"],
            "contradiction_check": contra,
            "facts": coupled.get("facts"),
            "lspace_snapshot_digest": coupled.get("lspace_snapshot_digest"),
        }

    if force_block_tombstone and not contra["blocked"]:
        contra = {
            "matches": contra.get("matches") or [],
            "contradictions": [
                {
                    "fact_id": "forced_demo",
                    "summary": "forced tombstone contradiction demo",
                    "speak_match": "FORCE",
                }
            ],
            "blocked": True,
        }
        couple_slim["contradiction_check"] = contra
        couple_slim["activation_vs_tombstone"] = "contradict"
        couple_slim["forced_demo"] = True

    decision = "ALLOW"
    reasons = []
    if silent.get("blocked"):
        decision = "BLOCK"
        reasons.append({"kind": "silent_intent", "flagged": silent.get("flagged")})
    if contra.get("blocked"):
        decision = "BLOCK"
        reasons.append(
            {
                "kind": "tombstone_contradiction",
                "contradictions": contra.get("contradictions"),
            }
        )

    # 3) Auto-tune on silent-intent BLOCK (no manual step)
    if (
        _autotune is not None
        and state_dir is not None
        and silent.get("blocked")
        and any(r.get("kind") == "silent_intent" for r in reasons)
    ):
        verd_override = _autotune.resolve_verdict_override(verdict)
        tune = _autotune.record_block(
            text=text,
            flagged=silent.get("flagged"),
            tombstone_contradictions=contra.get("contradictions"),
            forced_demo=bool(silent.get("forced_demo")),
            verdict=verd_override,
            category=None,  # auto-classify
            state_dir=state_dir,
            reason_kind="silent_intent",
            receipt_id=receipt_path or None,
        )
        at_info.update(
            {
                "category": tune["category"],
                "verdict": tune["verdict"],
                "fp_rate": tune["fp_rate"],
                "fp_stats": tune["fp_stats"],
                "changed": tune["changed"],
                "adjust": tune["adjust"],
                "thresholds": tune["thresholds"],
                "snapshot_post": _autotune.snapshot_for_receipt(state_dir),
            }
        )

    body = {
        "type": "lspace_gate_receipt",
        "direction": "outbound",
        "decision": decision,
        "reasons": reasons,
        "silent_intent": {
            "flagged": silent.get("flagged"),
            "blocked": silent.get("blocked"),
            "rank_thresh": rt,
            "min_hits": mh,
            "per_word_summary": {
                w: {
                    "silent_flag": v.get("silent_flag"),
                    "best_rank": v.get("best_rank"),
                    "hits": v.get("hits_below_thresh"),
                    "on_surface": v.get("on_surface"),
                }
                for w, v in (silent.get("per_word") or {}).items()
            },
            "forced_demo": silent.get("forced_demo"),
        },
        "tombstone_check": couple_slim,
        "lspace": {
            "seq_len": snap.get("seq_len"),
            "layers": snap.get("layers"),
            "n_positions": len(snap.get("positions") or []),
            "apply_s": snap.get("apply_s"),
            "sample_L6_last": (snap.get("jlens_top_k_by_layer_pos") or {})
            .get("6", {})
            .get(str((snap.get("positions") or [-1])[-1]), []),
            "sample_L9_last": (snap.get("jlens_top_k_by_layer_pos") or {})
            .get("9", {})
            .get(str((snap.get("positions") or [-1])[-1]), []),
        },
        "autotune": at_info,
        "text_chars": len(text),
        "full_text_no_480_truncate": True,
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "ts": datetime.now(timezone.utc).isoformat(),
        "instrument": "speak-text-lspace-auditor",
        "not_grok_bot_residuals": True,
        "not_proxy_jlens": True,
        "flag_message": (
            None
            if decision == "ALLOW"
            else (
                "LSPACE GATE BLOCK: "
                + "; ".join(
                    r["kind"]
                    + (
                        f" flagged={r.get('flagged')}"
                        if r.get("flagged")
                        else f" contradictions={r.get('contradictions')}"
                    )
                    for r in reasons
                )
                + ". Problematic content withheld. Instrument=speak-text auditor + tombstone "
                "(not Grok Bot residuals; not demoted proxy_jlens)."
            )
        ),
    }
    _write_receipt(receipt_path, body)
    return body


def main() -> int:
    ap = argparse.ArgumentParser(description="LSpace operating gate")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("text", nargs="?", default="")
        p.add_argument("--text-file", dest="text_file", default="", metavar="PATH")
        p.add_argument("--speak-text", dest="speak_text", default="", help="inline text (prefer over positional when flags present)")
        p.add_argument("--receipt", default="", help="write JSON receipt path")
        p.add_argument("--force-block-silent", action="store_true")

    pin = sub.add_parser("inbound", help="probe Lawrence inbound before reply")
    add_common(pin)

    pout = sub.add_parser("outbound", help="probe speak draft before SendToUser")
    add_common(pout)
    pout.add_argument("--tombstone", default=TOMBSTONE_DEFAULT)
    pout.add_argument("--force-block-tombstone", action="store_true")
    pout.add_argument("--fast", action="store_true", help="skip second full couple snapshot")
    pout.add_argument("--autotune-dir", default="", help="override autotune state dir")
    pout.add_argument(
        "--verdict",
        default="",
        choices=["", "false_positive", "true_catch", "pending"],
        help="override tombstone verdict for this BLOCK (calibration)",
    )

    args = ap.parse_args()
    text = args.text or getattr(args, "speak_text", "") or ""
    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read()
    if not text and not args.force_block_silent and not getattr(args, "force_block_tombstone", False):
        print("need text", file=sys.stderr)
        return 3

    if args.cmd == "inbound":
        body = gate_inbound(
            text, receipt_path=args.receipt, force_block_silent=args.force_block_silent
        )
    else:
        body = gate_outbound(
            text,
            receipt_path=args.receipt,
            tombstone=args.tombstone,
            force_block_silent=args.force_block_silent,
            force_block_tombstone=args.force_block_tombstone,
            skip_full_couple_snapshot=args.fast,
            autotune_dir=args.autotune_dir or None,
            verdict=args.verdict or None,
        )

    print(json.dumps(body, indent=2))
    if body["decision"] == "BLOCK":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
