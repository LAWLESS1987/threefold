#!/usr/bin/env python3
"""Automatic self-tuning of LSpace silent-intent thresholds from observed FP rate.

Threshold semantics (IMPORTANT):
  score_silent_intent flags a word when rank r < rank_thresh at enough (layer,pos)
  hits. Therefore:
    - HIGHER rank_thresh  => MORE sensitive => MORE blocks / more FPs
    - LOWER  rank_thresh  => HARDER to block => FEWER FPs
    - HIGHER min_hits     => HARDER to block => FEWER FPs
    - LOWER  min_hits     => EASIER to block => MORE sensitive

To reduce FPs when rate > upper: decrease rank_thresh and/or increase min_hits.
To restore sensitivity when rate < lower: increase rank_thresh and/or decrease min_hits.
Never cross safety floors/ceilings that would disable the gate.

Outbound auto-tunes only (inbound keeps fixed constants). Meta-text BLOCKs use a
separate rolling window + thresholds so status-report FPs do not poison main.
"""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Defaults / bounds
# ---------------------------------------------------------------------------

STATE_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state")

MAIN_BASELINE = {"rank_thresh": 50, "min_hits": 3}
META_BASELINE = {"rank_thresh": 35, "min_hits": 5}  # harder to false-block status

WINDOW = 20
FP_UPPER = 0.40
FP_LOWER = 0.10
MIN_SAMPLES = 5
RANK_FLOOR = 15
RANK_CEILING = 80
HITS_FLOOR = 2
HITS_CEILING = 8
STEP_RANK = 5
STEP_HITS = 1

# Lexicon subset that commonly poisons meta/status speaks when on_surface
META_LEXICON_SUBSET = frozenset({"error", "omit", "missing", "wrong"})

# Self-referential / gate-status markers
_META_MARKERS = re.compile(
    r"\b("
    r"lspace|jspace|triad|tombstone|silent[-_ ]?intent|autotune|"
    r"rank_thresh|min_hits|threshold|receipt|gate|"
    r"block|allow|flag(?:ged)?|verdict|false[-_ ]?positive|true[-_ ]?catch|"
    r"speak[-_ ]?text|auditor|outbound|inbound|"
    r"smoke|status\s+report|instrument"
    r")\b",
    re.I,
)

_lock = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_state() -> dict:
    return {
        "version": 1,
        "main": {
            "rank_thresh": MAIN_BASELINE["rank_thresh"],
            "min_hits": MAIN_BASELINE["min_hits"],
            "baseline": dict(MAIN_BASELINE),
        },
        "meta": {
            "rank_thresh": META_BASELINE["rank_thresh"],
            "min_hits": META_BASELINE["min_hits"],
            "baseline": dict(META_BASELINE),
        },
        "bounds": {
            "window": WINDOW,
            "fp_upper": FP_UPPER,
            "fp_lower": FP_LOWER,
            "min_samples": MIN_SAMPLES,
            "rank_floor": RANK_FLOOR,
            "rank_ceiling": RANK_CEILING,
            "hits_floor": HITS_FLOOR,
            "hits_ceiling": HITS_CEILING,
            "step_rank": STEP_RANK,
            "step_hits": STEP_HITS,
        },
        "updated_ts": None,
        "note": (
            "Outbound silent-intent auto-tune. "
            "Lower rank_thresh / higher min_hits => fewer FPs. "
            "Instrument=speak-text-lspace-auditor (not Grok residuals)."
        ),
    }


def state_paths(state_dir: str) -> dict[str, str]:
    return {
        "state": os.path.join(state_dir, "autotune_state.json"),
        "block_log": os.path.join(state_dir, "block_log.jsonl"),
        "threshold_changes": os.path.join(state_dir, "threshold_changes.jsonl"),
    }


def load_state(state_dir: str = STATE_DIR_DEFAULT) -> dict:
    path = state_paths(state_dir)["state"]
    if not os.path.isfile(path):
        st = default_state()
        save_state(st, state_dir)
        return st
    with open(path, encoding="utf-8") as f:
        st = json.load(f)
    # fill missing keys from defaults
    base = default_state()
    for k, v in base.items():
        if k not in st:
            st[k] = v
    for cat in ("main", "meta"):
        for kk, vv in base[cat].items():
            st.setdefault(cat, {}).setdefault(kk, vv)
    return st


def save_state(state: dict, state_dir: str = STATE_DIR_DEFAULT) -> None:
    os.makedirs(state_dir, exist_ok=True)
    state = dict(state)
    state["updated_ts"] = _utc_now()
    path = state_paths(state_dir)["state"]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def get_thresholds(state: dict, category: str = "main") -> tuple[int, int]:
    cat = state.get(category) or state["main"]
    return int(cat["rank_thresh"]), int(cat["min_hits"])


# ---------------------------------------------------------------------------
# Category + verdict classification
# ---------------------------------------------------------------------------

def is_meta_text(text: str, flagged: list[str] | None = None) -> bool:
    """Detect self-referential status / gate-about-itself speaks.

    True when text clearly discusses the gate/LSpace/triad/tombstone/silent-intent
    machinery, OR when flagged words are only from META_LEXICON_SUBSET and the
    speak looks like a status report (markers present).
    """
    if not text:
        return False
    if _META_MARKERS.search(text):
        return True
    flagged = list(flagged or [])
    if flagged and set(flagged).issubset(META_LEXICON_SUBSET):
        # status-ish prose mentioning those words in a reporting context
        if re.search(
            r"\b(report|status|blocked|withheld|lexicon|flag|threshold|receipt)\b",
            text,
            re.I,
        ):
            return True
    return False


def classify_category(
    text: str,
    *,
    flagged: list[str] | None = None,
    explicit: str | None = None,
) -> str:
    if explicit in ("main", "meta"):
        return explicit
    return "meta" if is_meta_text(text, flagged) else "main"


def classify_verdict(
    *,
    category: str,
    silent_blocked: bool,
    tombstone_contradictions: list | None,
    forced_demo: bool = False,
    explicit: str | None = None,
) -> str:
    """Map a silent-intent BLOCK to false_positive / true_catch / pending.

    Calibration heuristic (matches the FP pattern verified in operation):
      - explicit override wins
      - tombstone contradictions present => true_catch
      - forced security demo with non-meta silent flags => true_catch
      - meta category silent BLOCK with no contradictions => meta_false_positive
        (stored as false_positive under category=meta)
      - silent BLOCK + neutral/no contradictions + not forced real catch => false_positive
      - else pending
    """
    if explicit in ("false_positive", "true_catch", "pending"):
        return explicit
    contras = tombstone_contradictions or []
    if contras:
        return "true_catch"
    if forced_demo and category == "main":
        # Forced silent demo for security — count as true_catch so it does not
        # inflate FP rate and loosen the gate incorrectly.
        return "true_catch"
    if not silent_blocked:
        return "pending"
    # silent BLOCK, no contradictions: default FP (verified pattern)
    return "false_positive"


# ---------------------------------------------------------------------------
# Rolling FP rate + adjust
# ---------------------------------------------------------------------------

def compute_fp_rate(verdicts: list[str]) -> dict:
    """Compute FP rate over verdicts that are decided (ignore pending)."""
    decided = [v for v in verdicts if v in ("false_positive", "true_catch")]
    n = len(decided)
    fps = sum(1 for v in decided if v == "false_positive")
    rate = (fps / n) if n else 0.0
    return {
        "n_decided": n,
        "n_fp": fps,
        "n_true": n - fps,
        "fp_rate": rate,
        "n_pending": sum(1 for v in verdicts if v == "pending"),
        "n_raw": len(verdicts),
    }


def maybe_adjust(
    rank_thresh: int,
    min_hits: int,
    fp_rate: float,
    n_samples: int,
    *,
    fp_upper: float = FP_UPPER,
    fp_lower: float = FP_LOWER,
    min_samples: int = MIN_SAMPLES,
    rank_floor: int = RANK_FLOOR,
    rank_ceiling: int = RANK_CEILING,
    hits_floor: int = HITS_FLOOR,
    hits_ceiling: int = HITS_CEILING,
    step_rank: int = STEP_RANK,
    step_hits: int = STEP_HITS,
    category: str = "main",
) -> dict:
    """Return adjustment decision. Max one step per call (one BLOCK event).

    When fp_rate > upper: move toward fewer FPs (rank_thresh -= step, else min_hits += 1).
    When fp_rate < lower and enough samples: move toward baseline sensitivity
    (rank_thresh += step, else min_hits -= 1), clamped to floors/ceilings.
    """
    old_rt, old_mh = rank_thresh, min_hits
    new_rt, new_mh = rank_thresh, min_hits
    action = "none"
    reason = "no_change"

    if n_samples < min_samples:
        return {
            "changed": False,
            "action": "none",
            "reason": f"insufficient_samples ({n_samples}<{min_samples})",
            "category": category,
            "fp_rate": fp_rate,
            "n_samples": n_samples,
            "old": {"rank_thresh": old_rt, "min_hits": old_mh},
            "new": {"rank_thresh": new_rt, "min_hits": new_mh},
        }

    if fp_rate > fp_upper:
        # Fewer FPs: prefer lowering rank_thresh first, then raising min_hits
        if rank_thresh - step_rank >= rank_floor:
            new_rt = rank_thresh - step_rank
            action = "harden_rank"
            reason = f"fp_rate {fp_rate:.3f} > upper {fp_upper}"
        elif min_hits + step_hits <= hits_ceiling:
            new_mh = min_hits + step_hits
            action = "harden_hits"
            reason = f"fp_rate {fp_rate:.3f} > upper {fp_upper} (rank at floor)"
        else:
            action = "none"
            reason = f"fp_rate {fp_rate:.3f} > upper but already at harden limits"
    elif fp_rate < fp_lower:
        # More sensitive (toward baseline): raise rank_thresh first, then lower min_hits
        if rank_thresh + step_rank <= rank_ceiling:
            new_rt = rank_thresh + step_rank
            action = "loosen_rank"
            reason = f"fp_rate {fp_rate:.3f} < lower {fp_lower}"
        elif min_hits - step_hits >= hits_floor:
            new_mh = min_hits - step_hits
            action = "loosen_hits"
            reason = f"fp_rate {fp_rate:.3f} < lower {fp_lower} (rank at ceiling)"
        else:
            action = "none"
            reason = f"fp_rate {fp_rate:.3f} < lower but already at loosen limits"
    else:
        reason = f"fp_rate {fp_rate:.3f} within [{fp_lower}, {fp_upper}]"

    changed = (new_rt != old_rt) or (new_mh != old_mh)
    return {
        "changed": changed,
        "action": action if changed else "none",
        "reason": reason,
        "category": category,
        "fp_rate": fp_rate,
        "n_samples": n_samples,
        "old": {"rank_thresh": old_rt, "min_hits": old_mh},
        "new": {"rank_thresh": new_rt, "min_hits": new_mh},
    }


def _read_recent_verdicts(
    block_log_path: str, category: str, window: int
) -> list[str]:
    if not os.path.isfile(block_log_path):
        return []
    rows: list[dict] = []
    with open(block_log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    # silent-intent BLOCKs for this category only (newest last)
    filtered = [
        r
        for r in rows
        if r.get("category") == category
        and r.get("reason_kind") == "silent_intent"
    ]
    recent = filtered[-window:]
    return [r.get("verdict", "pending") for r in recent]


def _append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")


def record_block(
    *,
    text: str,
    flagged: list[str] | None,
    tombstone_contradictions: list | None = None,
    forced_demo: bool = False,
    verdict: str | None = None,
    category: str | None = None,
    state_dir: str = STATE_DIR_DEFAULT,
    reason_kind: str = "silent_intent",
    receipt_id: str | None = None,
) -> dict:
    """Log one BLOCK, update rolling FP rate, maybe step thresholds. Pure-ish API.

    Returns snapshot: thresholds, fp_rate, category, verdict, changed, adjust.
    Thread-safe via module lock.
    """
    with _lock:
        st = load_state(state_dir)
        paths = state_paths(state_dir)
        bounds = st.get("bounds") or default_state()["bounds"]

        cat = classify_category(text, flagged=flagged, explicit=category)
        verd = classify_verdict(
            category=cat,
            silent_blocked=True,
            tombstone_contradictions=tombstone_contradictions,
            forced_demo=forced_demo,
            explicit=verdict,
        )

        rt, mh = get_thresholds(st, cat)
        log_row = {
            "ts": _utc_now(),
            "category": cat,
            "verdict": verd,
            "reason_kind": reason_kind,
            "flagged": list(flagged or []),
            "forced_demo": bool(forced_demo),
            "n_contradictions": len(tombstone_contradictions or []),
            "rank_thresh_used": rt,
            "min_hits_used": mh,
            "text_chars": len(text or ""),
            "text_preview": (text or "")[:160],
            "receipt_id": receipt_id,
        }
        _append_jsonl(paths["block_log"], log_row)

        window = int(bounds.get("window", WINDOW))
        verdicts = _read_recent_verdicts(paths["block_log"], cat, window)
        rate_info = compute_fp_rate(verdicts)

        adjust = maybe_adjust(
            rt,
            mh,
            rate_info["fp_rate"],
            rate_info["n_decided"],
            fp_upper=float(bounds.get("fp_upper", FP_UPPER)),
            fp_lower=float(bounds.get("fp_lower", FP_LOWER)),
            min_samples=int(bounds.get("min_samples", MIN_SAMPLES)),
            rank_floor=int(bounds.get("rank_floor", RANK_FLOOR)),
            rank_ceiling=int(bounds.get("rank_ceiling", RANK_CEILING)),
            hits_floor=int(bounds.get("hits_floor", HITS_FLOOR)),
            hits_ceiling=int(bounds.get("hits_ceiling", HITS_CEILING)),
            step_rank=int(bounds.get("step_rank", STEP_RANK)),
            step_hits=int(bounds.get("step_hits", STEP_HITS)),
            category=cat,
        )

        changed = bool(adjust["changed"])
        if changed:
            st[cat]["rank_thresh"] = adjust["new"]["rank_thresh"]
            st[cat]["min_hits"] = adjust["new"]["min_hits"]
            save_state(st, state_dir)
            change_row = {
                "ts": _utc_now(),
                "category": cat,
                "fp_rate": rate_info["fp_rate"],
                "n_samples": rate_info["n_decided"],
                "action": adjust["action"],
                "reason": adjust["reason"],
                "old": adjust["old"],
                "new": adjust["new"],
            }
            _append_jsonl(paths["threshold_changes"], change_row)
        else:
            # still touch updated_ts lightly? skip — only on change or load
            pass

        # reload after possible change
        st = load_state(state_dir)
        rt2, mh2 = get_thresholds(st, cat)
        main_rt, main_mh = get_thresholds(st, "main")
        meta_rt, meta_mh = get_thresholds(st, "meta")

        return {
            "category": cat,
            "verdict": verd,
            "fp_rate": rate_info["fp_rate"],
            "fp_stats": rate_info,
            "changed": changed,
            "adjust": adjust,
            "thresholds": {
                "category": cat,
                "rank_thresh": rt2,
                "min_hits": mh2,
                "main": {"rank_thresh": main_rt, "min_hits": main_mh},
                "meta": {"rank_thresh": meta_rt, "min_hits": meta_mh},
            },
            "log_row": log_row,
        }


def snapshot_for_receipt(state_dir: str = STATE_DIR_DEFAULT) -> dict:
    st = load_state(state_dir)
    main_rt, main_mh = get_thresholds(st, "main")
    meta_rt, meta_mh = get_thresholds(st, "meta")
    return {
        "main": {"rank_thresh": main_rt, "min_hits": main_mh},
        "meta": {"rank_thresh": meta_rt, "min_hits": meta_mh},
        "bounds": st.get("bounds"),
        "updated_ts": st.get("updated_ts"),
    }


def resolve_state_dir(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    return os.environ.get("LSPACE_AUTOTUNE_DIR", STATE_DIR_DEFAULT)


def resolve_verdict_override(explicit: str | None = None) -> str | None:
    if explicit in ("false_positive", "true_catch", "pending"):
        return explicit
    env = os.environ.get("LSPACE_AUTOTUNE_VERDICT", "").strip()
    if env in ("false_positive", "true_catch", "pending"):
        return env
    return None
