#!/usr/bin/env python3
"""Verify one triad: all three legs exist for one task_id, and their links agree.

Fails closed. An absent artifact, an unreadable one, a task_id that differs, or a digest that differs is a FAIL,
and the exit status is 1. Checks, in order:

  lens leg   runs/<task_id>/jlens_snapshot.json (the JLens capture), jlens_raw.npz, jlens_vocab.json and
             llens.json exist. The capture's raw arrays and vocabulary file match the digests recorded at capture
             time. The L-lens artifact cites this capture, and its derived section recomputes exactly from the raw
             arrays (llens.verify_llens).
  tombstone  an entry in the tombstone log whose notes carry task_id=<id>, jlens_capture_sha256=<the capture's
             digest> and llens_sha256=<the L-lens artifact's digest>.
  covenant   memories task-<id>, jlens-<id> and llens-<id> exist under AI_MEMORY_ROOT. The task memory carries the
             same task_id and both digests. The jlens and llens memories hold bodies whose canonical digests equal
             the capture's and the artifact's.

What it cannot see. It checks that the records agree with each other and with the stored arrays. It does not re-run
the model, so it cannot tell whether the arrays are what upstream would return today. For that, re-run
jlens_snapshot.py with the recorded configuration and compare llens_capture_time.full_tensor_sha256. It also does
not check covenant's own hash chain; covenant's verify_chain does that.

Usage:
  verify_triad.py --task-id ID [--run-dir DIR] [--tombstone-log PATH] [--memory-root DIR]
Defaults: --run-dir $RUN_DIR/<id> or <repo>/runs/<id>; --tombstone-log $TOMBSTONE_LOG or
<repo>/tombstone/tombstone.md; --memory-root $AI_MEMORY_ROOT.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import llens  # noqa: E402

HEX64 = r"[0-9a-f]{64}"


def safe_name(s: str) -> str:
    """put_covenant_memory.sh's safe_name: every character outside [A-Za-z0-9._-] becomes '-'."""
    return re.sub(r"[^A-Za-z0-9._-]", "-", s)


def tombstone_entries(text: str) -> list:
    """Each '### ' section of the tombstone log as {field: value}."""
    out = []
    for block in re.split(r"(?m)^### ", text)[1:]:
        fields = {"heading": block.splitlines()[0].strip()}
        for line in block.splitlines()[1:]:
            m = re.match(r"^- ([a-z_]+): ?(.*)$", line)
            if m:
                fields[m.group(1)] = m.group(2)
        out.append(fields)
    return out


def note_value(notes: str, key: str):
    m = re.search(r"(?:^|;\s*)%s=([^;]*)" % re.escape(key), notes or "")
    return m.group(1).strip() if m else None


def memory_body(root: str, name: str):
    """The body of covenant memory <name> (text after the closing frontmatter fence), or None."""
    path = os.path.join(root, safe_name(name) + ".md")
    if not os.path.isfile(path):
        return None
    text = open(path, encoding="utf-8").read()
    m = re.match(r"^---\n.*?\n---\n(.*)$", text, re.S)
    return (m.group(1) if m else text).strip()


def verify(task_id: str, run_dir: str, tombstone_log: str, memory_root: str) -> list:
    out = []

    def check(ok, name, why=""):
        out.append((bool(ok), name, why))
        return bool(ok)

    cap_p, raw_p, ll_p = (os.path.join(run_dir, f) for f in ("jlens_snapshot.json", "jlens_raw.npz", "llens.json"))
    present = [check(os.path.isfile(p), "lens leg: %s present" % os.path.basename(p), "missing: %s" % p)
               for p in (cap_p, raw_p, ll_p)]
    cap_sha = ll_sha = None
    if all(present):
        try:
            capture = json.load(open(cap_p, encoding="utf-8"))
            art = json.load(open(ll_p, encoding="utf-8"))
            raw = llens.load_raw(raw_p)
            vocab = llens.load_vocab(cap_p, capture)
        except Exception as e:  # noqa: BLE001 -- unreadable is a failure, named
            check(False, "lens leg: artifacts readable", "%s: %s" % (type(e).__name__, e))
        else:
            check(capture.get("task_id") == task_id, "lens leg: capture task_id",
                  "capture says %r" % capture.get("task_id"))
            out.extend((ok, "lens leg: " + name, why) for ok, name, why in llens.verify_llens(art, capture, raw, vocab))
            cap_sha, ll_sha = llens.digest(capture), llens.digest(art)

    # tombstone
    if check(os.path.isfile(tombstone_log), "tombstone: log present", "missing: %s" % tombstone_log):
        entries = [e for e in tombstone_entries(open(tombstone_log, encoding="utf-8").read())
                   if note_value(e.get("notes"), "task_id") == task_id]
        if check(entries, "tombstone: an entry for this task_id", "no entry's notes carry task_id=%s" % task_id):
            e = entries[-1]
            for key, want in (("jlens_capture_sha256", cap_sha), ("llens_sha256", ll_sha)):
                got = note_value(e.get("notes"), key)
                check(want is not None and got == want, "tombstone: %s agrees" % key,
                      "entry has %r, lens leg gives %r" % (got, want))

    # covenant
    if check(memory_root and os.path.isdir(memory_root), "covenant: memory root present",
             "AI_MEMORY_ROOT %r is not a directory" % memory_root):
        body = memory_body(memory_root, "task-" + task_id)
        if check(body is not None, "covenant: task-%s present" % task_id, "no such memory"):
            lines = dict(l.split("=", 1) for l in body.splitlines() if "=" in l)
            check(lines.get("task_id") == task_id, "covenant: task memory task_id", "has %r" % lines.get("task_id"))
            for key, want in (("jlens_capture_sha256", cap_sha), ("llens_sha256", ll_sha)):
                check(want is not None and lines.get(key) == want, "covenant: task memory %s agrees" % key,
                      "memory has %r, lens leg gives %r" % (lines.get(key), want))
        for prefix, want in (("jlens-", cap_sha), ("llens-", ll_sha)):
            b = memory_body(memory_root, prefix + task_id)
            if check(b is not None, "covenant: %s%s present" % (prefix, task_id), "no such memory"):
                try:
                    got = llens.digest(json.loads(b))
                except ValueError as e:
                    got = "unparseable (%s)" % e
                check(want is not None and got == want, "covenant: %s%s body digest agrees" % (prefix, task_id),
                      "memory body gives %r, lens leg gives %r" % (got, want))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--tombstone-log", default=os.environ.get("TOMBSTONE_LOG")
                    or os.path.join(REPO_ROOT, "tombstone", "tombstone.md"))
    ap.add_argument("--memory-root", default=os.environ.get("AI_MEMORY_ROOT", ""))
    a = ap.parse_args(argv)
    run_dir = a.run_dir or os.path.join(os.environ.get("RUN_DIR") or os.path.join(REPO_ROOT, "runs"), a.task_id)
    results = verify(a.task_id, run_dir, a.tombstone_log, a.memory_root)
    for ok, name, why in results:
        print("  %s  %s%s" % ("PASS" if ok else "FAIL", name, "" if ok else "  -- " + why))
    bad = sum(1 for r in results if not r[0])
    print("VERIFY_TRIAD_OK task_id=%s checks=%d" % (a.task_id, len(results)) if not bad
          else "VERIFY_TRIAD_FAIL task_id=%s failed=%d of %d" % (a.task_id, bad, len(results)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
