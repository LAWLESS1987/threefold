#!/usr/bin/env python3
"""Smoke: simulated FP streak raises the bar (fewer FPs); meta does not poison main.

Uses a temp state dir — does NOT clobber the live state/ directory.
Pure autotune functions; no GPT-2 required.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPORT = os.path.join(os.environ.get("LSPACE_SMOKE_DIR", os.path.dirname(os.path.abspath(__file__))), "AUTOTUNE_SMOKE.md")


def main() -> int:
    lines: list[str] = []
    ok = True

    def log(s: str = "") -> None:
        lines.append(s)
        print(s)

    log("# LSpace autotune smoke")
    log()
    log("Instrument: **speak-text LSpace auditor** auto-tune (pure functions).")
    log("**Not** Grok Bot residuals. Temp state dir only.")
    log()

    tmp = tempfile.mkdtemp(prefix="lspace_autotune_smoke_")
    try:
        import autotune as at

        # --- init ---
        st = at.load_state(tmp)
        main_rt0, main_mh0 = at.get_thresholds(st, "main")
        meta_rt0, meta_mh0 = at.get_thresholds(st, "meta")
        log("## Initial thresholds")
        log(f"- main: rank_thresh={main_rt0} min_hits={main_mh0}")
        log(f"- meta: rank_thresh={meta_rt0} min_hits={meta_mh0}")
        log(f"- bounds: {json.dumps(st['bounds'])}")
        log()

        if (main_rt0, main_mh0) != (50, 3) or (meta_rt0, meta_mh0) != (35, 5):
            ok = False
            log("- FAIL: unexpected baselines")

        # --- (c) FP streak on main: should harden (rank_thresh down and/or min_hits up) ---
        log("## Simulated main false-positive streak")
        n_fp = 10
        last = None
        for i in range(n_fp):
            last = at.record_block(
                text=f"Ordinary report paragraph {i} with no gate jargon.",
                flagged=["fraud"],
                tombstone_contradictions=[],
                forced_demo=False,
                verdict="false_positive",
                category="main",
                state_dir=tmp,
            )
            log(
                f"- block {i+1}: fp_rate={last['fp_rate']:.3f} "
                f"changed={last['changed']} action={last['adjust']['action']} "
                f"rt={last['thresholds']['rank_thresh']} mh={last['thresholds']['min_hits']}"
            )

        st2 = at.load_state(tmp)
        main_rt1, main_mh1 = at.get_thresholds(st2, "main")
        meta_rt1, meta_mh1 = at.get_thresholds(st2, "meta")
        log()
        log(f"- after streak: main rt={main_rt1} mh={main_mh1} (was {main_rt0}/{main_mh0})")
        log(f"- meta unchanged check: rt={meta_rt1} mh={meta_mh1} (was {meta_rt0}/{meta_mh0})")

        hardened = (main_rt1 < main_rt0) or (main_mh1 > main_mh0)
        if not hardened:
            ok = False
            log("- FAIL: expected main thresholds to harden (fewer FPs)")
        else:
            log("- PASS: main thresholds moved toward fewer FPs "
                f"(rank_thresh {main_rt0}->{main_rt1}, min_hits {main_mh0}->{main_mh1})")

        if (meta_rt1, meta_mh1) != (meta_rt0, meta_mh0):
            ok = False
            log("- FAIL: main FP streak should not change meta thresholds")
        else:
            log("- PASS: meta thresholds untouched by main streak")

        # threshold_changes must include triggering rate
        ch_path = at.state_paths(tmp)["threshold_changes"]
        changes = []
        if os.path.isfile(ch_path):
            with open(ch_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        changes.append(json.loads(line))
        log(f"- threshold_changes rows={len(changes)}")
        if not changes:
            ok = False
            log("- FAIL: expected at least one threshold change log row")
        else:
            for c in changes:
                log(f"  - {c['action']} fp_rate={c['fp_rate']:.3f} "
                    f"{c['old']} -> {c['new']} reason={c['reason']}")
            if any(c.get("fp_rate") is None for c in changes):
                ok = False
                log("- FAIL: change log missing fp_rate")
            else:
                log("- PASS: threshold_changes log includes triggering fp_rate")
            if any(c.get("category") != "main" for c in changes):
                ok = False
                log("- FAIL: expected only main category in this streak's changes")

        # --- meta BLOCKs do not change main ---
        log()
        log("## Meta-text BLOCKs isolated from main")
        main_before = at.get_thresholds(at.load_state(tmp), "main")
        meta_before = at.get_thresholds(at.load_state(tmp), "meta")
        for i in range(8):
            at.record_block(
                text=(
                    f"LSpace gate status report {i}: silent-intent BLOCK on "
                    f"lexicon words error/omit/missing; receipt threshold autotune note."
                ),
                flagged=["error", "omit"],
                tombstone_contradictions=[],
                forced_demo=False,
                verdict="false_positive",
                category="meta",
                state_dir=tmp,
            )
        main_after = at.get_thresholds(at.load_state(tmp), "main")
        meta_after = at.get_thresholds(at.load_state(tmp), "meta")
        log(f"- main before={main_before} after={main_after}")
        log(f"- meta before={meta_before} after={meta_after}")
        if main_after != main_before:
            ok = False
            log("- FAIL: meta BLOCKs must not change main thresholds")
        else:
            log("- PASS: main thresholds unchanged by meta streak")
        # meta should harden if FP rate high enough
        meta_hardened = (meta_after[0] < meta_before[0]) or (meta_after[1] > meta_before[1])
        if not meta_hardened:
            # 8 samples with window 20, min_samples 5, all FP => rate=1.0 > 0.40
            ok = False
            log("- FAIL: expected meta thresholds to harden on meta FP streak")
        else:
            log("- PASS: meta thresholds hardened independently "
                f"(rt {meta_before[0]}->{meta_after[0]}, mh {meta_before[1]}->{meta_after[1]})")

        # --- classify helpers ---
        log()
        log("## Category / verdict helpers")
        assert at.is_meta_text("LSpace gate BLOCK receipt: threshold autotune")
        assert not at.is_meta_text("The animal was a cat in the garden.")
        v = at.classify_verdict(
            category="main",
            silent_blocked=True,
            tombstone_contradictions=[],
            forced_demo=False,
        )
        assert v == "false_positive", v
        v2 = at.classify_verdict(
            category="main",
            silent_blocked=True,
            tombstone_contradictions=[{"fact_id": "x"}],
            forced_demo=False,
        )
        assert v2 == "true_catch", v2
        v3 = at.classify_verdict(
            category="main",
            silent_blocked=True,
            tombstone_contradictions=[],
            forced_demo=True,
        )
        assert v3 == "true_catch", v3
        log("- PASS: is_meta_text + classify_verdict heuristics")

        # rate math
        r = at.compute_fp_rate(["false_positive"] * 4 + ["true_catch"] * 1)
        assert abs(r["fp_rate"] - 0.8) < 1e-9
        log(f"- PASS: compute_fp_rate example={r}")

        # maybe_adjust unit
        adj = at.maybe_adjust(50, 3, 0.5, 10)
        assert adj["changed"] and adj["new"]["rank_thresh"] == 45
        log(f"- PASS: maybe_adjust harden_rank -> {adj['new']}")
        adj2 = at.maybe_adjust(15, 3, 0.5, 10)  # at floor
        assert adj2["changed"] and adj2["new"]["min_hits"] == 4
        log(f"- PASS: maybe_adjust harden_hits at rank floor -> {adj2['new']}")

    except Exception:
        ok = False
        log("## EXCEPTION")
        log("```")
        log(traceback.format_exc())
        log("```")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        log()
        log(f"(temp state dir removed: {tmp})")

    log()
    log("## Verdict")
    log(f"- smoke_ok={ok}")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"WROTE {REPORT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
