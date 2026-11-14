"""The triad writers, the publish order and the push script, without a model.

These run the real scripts: triad/put_memory_file.py, triad/put_covenant_memory.sh, triad/fire_triad.sh,
triad/on_speak.sh and scripts/push_to_github.sh. The memory store is covenant's own ai_memory_system, from a real
checkout (COVENANT_SRC, else vendor/covenant; CI fetches the commit named in .github/workflows/tests.yml), writing
into a temporary memory root.

Two things are stand-ins, and only these:
  - the capture. jlens_snapshot.py needs torch, the model and the lens, so a small script writes the synthetic
    capture tests/test_llens.py uses. Everything after it (llens.py, the tombstone, covenant, verify_triad.py) is
    the real code.
  - git and gh. A recorder named `git` (and one named `gh`) stands first on PATH and writes down its arguments. No
    test here can reach a remote, and the temporary memory root is not a real repository.

Every test but one was red on the code before the change it covers and is green after it. The one that was green
before, test_on_speak_appends_nothing_when_the_lens_leg_fails, guards behaviour the change must keep.
"""
from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "tests")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "triad"))
import verify_triad  # noqa: E402

BASH = shutil.which("bash")
CANARY = "canary-0123456789-placeholder-not-a-credential"
ZEROS = "0" * 64
CLEAN = ("TASK_ID", "TITLE", "TASK", "DONE", "OUTCOME", "PROMPT", "NOTES", "MISTAKES", "LESSONS", "BODY",
         "DECISION_BODY", "DESCRIPTION", "JLENS_BODY", "JLENS_BODY_FILE", "LLENS_BODY_FILE", "SPOKEN_TEXT", "GH_TOKEN",
         "MEMORY_INTENT", "JLENS_INTENT", "LLENS_INTENT", "MEMORY_BENEFICIARY", "MEMORY_DIVERGENCE",
         "MEMORY_DIVERGES_FROM", "MEMORY_TYPE", "AGENT_NAME", "PUT_COVENANT", "VERIFY_PY", "LLENS_PY",
         "GIT_AUTHOR_EMAIL", "GIT_AUTHOR_NAME")

FAKE_GIT = r"""#!/usr/bin/env bash
# A recorder, not git: every argument on its own line, then the configuration handed over in the environment.
{
  echo "CALL"
  for a in "$@"; do printf 'ARG %s\n' "$a"; done
  n="${GIT_CONFIG_COUNT:-0}"; i=0
  while (( i < n )); do
    k="GIT_CONFIG_KEY_$i"; v="GIT_CONFIG_VALUE_$i"
    printf 'CFG %s=%s\n' "${!k:-}" "${!v:-}"
    i=$((i + 1))
  done
} >> "$FAKE_GIT_LOG"
if [[ "${1:-}" == "remote" && "${2:-}" == "get-url" ]]; then
  if [[ -n "${FAKE_GIT_ORIGIN:-}" ]]; then printf '%s\n' "$FAKE_GIT_ORIGIN"; exit 0; fi
  exit 2
fi
for a in "$@"; do if [[ "$a" == "diff" ]]; then exit 1; fi; done   # "something is staged"
exit 0
"""

FAKE_GH = r"""#!/usr/bin/env bash
{ echo "CALL"; for a in "$@"; do printf 'ARG %s\n' "$a"; done; } >> "$FAKE_GH_LOG"
exit 0
"""

STUB_CAPTURE = r"""
import argparse, json, os, sys
sys.path.insert(0, %(root)r)
sys.path.insert(0, %(tests)r)
import numpy as np
import llens
from test_llens import VOCAB, synthetic
ap = argparse.ArgumentParser()
for name in ("--task-id", "--prompt", "--out", "--raw", "--digest", "--lens"):
    ap.add_argument(name)
a = ap.parse_args()
if os.environ.get("STUB_CAPTURE_FAILS"):
    print("ERROR: the capture failed (on purpose)", file=sys.stderr)
    raise SystemExit(3)
cap, raw = synthetic(a.task_id)
json.dump(cap, open(a.out, "w", encoding="utf-8"), indent=1)
np.savez_compressed(a.raw, **raw)
json.dump(VOCAB, open(os.path.join(os.path.dirname(a.raw), "jlens_vocab.json"), "w", encoding="utf-8"), indent=0)
open(a.digest, "w", encoding="utf-8").write("jlens synthetic capture\n")
print("JLENS_CAPTURE_SHA256=%%s" %% llens.digest(cap))
"""

TAMPER_APPEND = r"""#!/usr/bin/env bash
# The real tombstone append, except that the notes cite another L-lens artifact. verify_triad.py must refuse it.
notes="$(printf '%%s' "${7:-}" | sed -E 's/llens_sha256=[0-9a-f]{64}/llens_sha256=%(zeros)s/')"
exec "%(append)s" "$1" "$2" "$3" "$4" "$5" "$6" "$notes"
"""


def px(p) -> str:
    """A path both bash and a Windows python accept."""
    return str(p).replace("\\", "/")


def write_script(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.chmod(path, 0o755)
    return path


def covenant_src() -> str:
    src = os.environ.get("COVENANT_SRC") or os.path.join(ROOT, "vendor", "covenant")
    if not os.path.isfile(os.path.join(src, "ai_memory_system", "main.py")):
        pytest.fail("covenant's ai_memory_system is not at %r. Fetch covenant into vendor/covenant (the commit and the "
                    "commands are in .github/workflows/tests.yml) or set COVENANT_SRC." % src)
    return src


@pytest.fixture
def box(tmp_path):
    """A scratch triad: its own tombstone log, memory root and run directory, a python wrapper for the scripts, the
    stand-in capture, and the recorders first on PATH."""
    assert BASH, "bash is needed to run the triad scripts"
    fakebin = tmp_path / "fakebin"
    venv = tmp_path / "venv" / "bin"
    for d in (fakebin, venv, tmp_path / "memory"):
        d.mkdir(parents=True)
    write_script(fakebin / "git", FAKE_GIT)
    write_script(fakebin / "gh", FAKE_GH)
    write_script(venv / "python", '#!/usr/bin/env bash\nexec "%s" -B "$@"\n' % px(sys.executable))
    stub = tmp_path / "stub_capture.py"
    stub.write_text(STUB_CAPTURE % {"root": px(ROOT), "tests": px(TESTS)}, encoding="utf-8")
    (tmp_path / "lens.pt").write_text("a stand-in file; the stand-in capture does not read it\n", encoding="utf-8")
    (tmp_path / "tombstone.md").write_text("# scratch tombstone log\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in CLEAN and not k.startswith("GIT_CONFIG_")}
    env.update(
        PATH=str(fakebin) + os.pathsep + env.get("PATH", ""),
        FAKE_GIT_LOG=px(tmp_path / "git.log"), FAKE_GH_LOG=px(tmp_path / "gh.log"),
        COVENANT_SRC=px(covenant_src()), AI_MEMORY_ROOT=px(tmp_path / "memory"),
        TOMBSTONE_LOG=px(tmp_path / "tombstone.md"), RUN_DIR=px(tmp_path / "runs"),
        JLENS_VENV=px(tmp_path / "venv"), PYTHON_COV=px(venv / "python"),
        JLENS_SNAPSHOT_PY=px(stub), JLENS_LENS=px(tmp_path / "lens.pt"),
    )
    # The recorder must be the git the scripts find. If it is not, stop before any script runs.
    probe = subprocess.run([BASH, "-c", "git probe-that-the-recorder-is-first"], env=env, capture_output=True, text=True)
    assert probe.returncode == 0 and "probe-that-the-recorder-is-first" in read(tmp_path / "git.log"), \
        "the recorder is not the git on PATH: %s" % probe.stderr
    os.remove(tmp_path / "git.log")

    class Box:
        pass

    b = Box()
    b.tmp, b.env, b.memory, b.log = tmp_path, env, tmp_path / "memory", tmp_path / "tombstone.md"
    return b


def read(path) -> str:
    return open(path, encoding="utf-8").read() if os.path.isfile(path) else ""


def run(box, script, *args, **env):
    e = dict(box.env)
    e.update(env)
    cmd = [px(script)] + list(args)
    if not str(script).endswith(".py"):
        cmd = [BASH] + cmd
    else:
        cmd = [sys.executable, "-B"] + cmd
    r = subprocess.run(cmd, env=e, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=600)
    r.both = r.stdout + "\n--- stderr ---\n" + r.stderr
    return r


def git_calls(box):
    """[[argument, ...], ...] for every call the recorder saw, and every configuration line it was handed."""
    calls, cfg = [], []
    for line in read(box.tmp / "git.log").splitlines():
        if line == "CALL":
            calls.append([])
        elif line.startswith("ARG "):
            calls[-1].append(line[4:])
        elif line.startswith("CFG "):
            cfg.append(line[4:])
    return calls, cfg


def frontmatter(box, name) -> str:
    text = read(box.memory / (name + ".md"))
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert m, "no memory %s under the scratch root" % name
    return m.group(1)


def make_git_checkout(box):
    """The scripts publish only when the memory root has a .git directory. This one is empty: nothing can be pushed."""
    (box.memory / ".git").mkdir()


def tamper_append(box):
    return px(write_script(box.tmp / "tamper_append.sh",
                           TAMPER_APPEND % {"zeros": ZEROS, "append": px(os.path.join(ROOT, "tombstone", "append.sh"))}))


PUT_FILE = os.path.join(ROOT, "triad", "put_memory_file.py")
PUT_SH = os.path.join(ROOT, "triad", "put_covenant_memory.sh")
FIRE = os.path.join(ROOT, "triad", "fire_triad.sh")
ON_SPEAK = os.path.join(ROOT, "triad", "on_speak.sh")
PROVENANCE = ["--intent", "Lets a later verifier recompute this digest.",
              "--beneficiary", "whoever audits the task later", "--divergence", "none"]


# --------------------------------------------------------------------------- K1: the writers against real covenant

def put_file(box, name, body, *extra):
    f = box.tmp / (name + ".body.txt")
    f.write_text(body, encoding="utf-8")
    return run(box, PUT_FILE, "--root", px(box.memory), "--name", name, "--description", "a test record",
               "--type", "project", "--agent", "test", "--body-file", px(f), "--covenant-src", box.env["COVENANT_SRC"],
               *extra)


def test_put_memory_file_writes_a_record_that_carries_its_provenance(box):
    r = put_file(box, "jlens-t-file-1", '{"measured": [1, 2, 3]}', *PROVENANCE)
    assert r.returncode == 0, r.both
    fm = frontmatter(box, "jlens-t-file-1")
    assert "intent: Lets a later verifier recompute this digest." in fm
    assert "beneficiary: whoever audits the task later" in fm and "divergence: none" in fm


def test_put_memory_file_is_refused_when_the_intent_restates_the_body(box):
    r = put_file(box, "t-file-2", "the user prefers dark roast coffee", "--intent", "the user prefers dark roast",
                 "--beneficiary", "whoever audits the task later", "--divergence", "none")
    assert r.returncode != 0 and "restates" in r.both
    assert not os.path.exists(box.memory / "t-file-2.md")


def test_put_memory_file_passes_diverges_from_through(box):
    assert put_file(box, "t-file-3", "the reading was 4", *PROVENANCE).returncode == 0
    args = ["--intent", "Corrects the earlier figure so a reader does not rely on it.",
            "--beneficiary", "whoever audits the task later", "--divergence", "the earlier reading"]
    r = put_file(box, "t-file-4", "the reading was 5", *args, "--diverges-from", "t-file-3")
    assert r.returncode == 0, r.both
    assert "diverges_from: t-file-3" in frontmatter(box, "t-file-4")
    r = put_file(box, "t-file-5", "the reading was 6", *args, "--diverges-from", "no-such-record")
    assert r.returncode != 0 and not os.path.exists(box.memory / "t-file-5.md")


def test_the_shell_writer_puts_all_three_memories_with_provenance(box):
    cap = box.tmp / "cap.json"
    cap.write_text('{"type": "jlens", "task_id": "t-sh-1", "rows": [1, 2]}', encoding="utf-8")
    art = box.tmp / "art.json"
    art.write_text('{"type": "llens", "task_id": "t-sh-1", "derived": {"n": 2}}', encoding="utf-8")
    r = run(box, PUT_SH, "t-sh-1", "task_id=t-sh-1\ntitle=a test\noutcome=success\n", "triad task memory for t-sh-1",
            JLENS_BODY_FILE=px(cap), LLENS_BODY_FILE=px(art))
    assert r.returncode == 0, r.both
    intents = set()
    for name in ("task-t-sh-1", "jlens-t-sh-1", "llens-t-sh-1"):
        fm = frontmatter(box, name)
        assert "divergence: none" in fm and re.search(r"(?m)^\s*beneficiary: \S", fm), name
        intents.add(re.search(r"(?m)^\s*intent: (.+)$", fm).group(1))
    assert len(intents) == 3, "each of the three records says what it is for"


def test_the_shell_writer_takes_provenance_from_the_environment(box):
    r = run(box, PUT_SH, "t-sh-2", "task_id=t-sh-2\n", "triad task memory for t-sh-2",
            JLENS_BODY='{"type": "jlens", "task_id": "t-sh-2"}',
            MEMORY_INTENT="Shows a reviewer which run produced the figure.", JLENS_INTENT="Holds the reading for a rerun.",
            MEMORY_BENEFICIARY="the reviewer of run t-sh-2")
    assert r.returncode == 0, r.both
    assert "intent: Shows a reviewer which run produced the figure." in frontmatter(box, "task-t-sh-2")
    fm = frontmatter(box, "jlens-t-sh-2")   # the inline JLENS_BODY branch
    assert "intent: Holds the reading for a rerun." in fm and "beneficiary: the reviewer of run t-sh-2" in fm


# --------------------------------------------------------------------------- K2: publish after verification only

def test_the_writer_alone_never_publishes(box):
    make_git_checkout(box)
    r = run(box, PUT_SH, "t-sh-3", "task_id=t-sh-3\n", "triad task memory for t-sh-3", GH_TOKEN=CANARY)
    assert r.returncode == 0, r.both
    assert git_calls(box)[0] == [], "writing a memory ran git: writing and publishing are separate steps"


def test_a_verified_triad_is_published_after_its_verification(box):
    make_git_checkout(box)
    r = run(box, FIRE, "t", "a test task", "ran the writers", "success", "a b", TASK_ID="t-fire-0")
    assert r.returncode == 0, r.both
    assert "FIRE_TRIAD_OK task_id=t-fire-0" in r.stdout
    calls, _ = git_calls(box)
    assert ["-C", box.env["AI_MEMORY_ROOT"], "push", "origin", "HEAD:main"] in calls, "the verified triad was not published"
    assert r.stdout.index("VERIFY_TRIAD_OK") < r.stdout.index("Pushed AI_MEMORY_ROOT"), "published before verification"


def test_the_token_is_in_no_git_argument(box):
    make_git_checkout(box)
    r = run(box, FIRE, "t", "a test task", "ran the writers", "success", "a b", GH_TOKEN=CANARY, TASK_ID="t-fire-1")
    assert r.returncode == 0, r.both
    calls, cfg = git_calls(box)
    assert any("push" in c for c in calls), "the verified triad was not published"
    encoded = base64.b64encode(("x-access-token:" + CANARY).encode()).decode()
    for c in calls:
        for arg in c:
            assert CANARY not in arg and encoded not in arg, "the token is in a git argument: %r" % c[:4]
            assert "@github.com" not in arg, "a remote address carries a user part: %r" % c[:4]
    assert any("extraheader" in line and encoded in line for line in cfg), \
        "the credential did not reach git through the environment"


def test_a_triad_that_does_not_verify_is_not_published(box):
    make_git_checkout(box)
    r = run(box, FIRE, "t", "a test task", "ran the writers", "success", "a b", GH_TOKEN=CANARY, TASK_ID="t-fire-2",
            TOMBSTONE_APPEND=tamper_append(box))
    assert r.returncode != 0, r.both
    assert "FAIL  tombstone: llens_sha256 agrees" in r.stdout and "VERIFY_TRIAD_FAIL" in r.stdout, r.both
    assert "FIRE_TRIAD_OK" not in r.stdout
    assert git_calls(box)[0] == [], "git ran although the verification failed"


def test_without_a_git_checkout_nothing_is_published(box):
    r = run(box, FIRE, "t", "a test task", "ran the writers", "success", "a b", TASK_ID="t-fire-3")
    assert r.returncode == 0, r.both
    assert "FIRE_TRIAD_OK task_id=t-fire-3" in r.stdout and git_calls(box)[0] == []


# --------------------------------------------------------------------------- K3: the outcome is what happened

def entries(box):
    return verify_triad.tombstone_entries(read(box.log))


def test_on_speak_records_success_when_the_triad_completes(box):
    r = run(box, ON_SPEAK, "The record is append-only.", "t", TASK_ID="t-speak-1")
    assert r.returncode == 0, r.both
    assert [e["outcome"] for e in entries(box)] == ["success"]


def test_on_speak_records_failed_when_the_triad_fails_after_its_entry(box):
    r = run(box, ON_SPEAK, "The record is append-only.", "t", TASK_ID="t-speak-2", TOMBSTONE_APPEND=tamper_append(box))
    assert r.returncode != 0, r.both
    es = entries(box)
    assert es and es[-1]["outcome"] == "failed", "the last word on a failed triad is %r" % (es[-1]["outcome"] if es else None)
    assert "t-speak-2" in es[-1]["notes"] and "fire_triad_exit=" in es[-1]["notes"]
    # the correction is not mistaken for the triad's own entry by the verifier
    assert verify_triad.note_value(es[-1]["notes"], "task_id") is None


def test_on_speak_appends_nothing_when_the_lens_leg_fails(box):
    before = read(box.log)
    r = run(box, ON_SPEAK, "The record is append-only.", "t", TASK_ID="t-speak-3", STUB_CAPTURE_FAILS="1")
    assert r.returncode != 0, r.both
    assert read(box.log) == before and not os.path.exists(box.memory / "task-t-speak-3.md")


# --------------------------------------------------------------------------- K4: the push script

def push_script(box, **env):
    """scripts/push_to_github.sh, run from a copy in a scratch tree so that its `cd` lands outside this checkout."""
    scripts = box.tmp / "repo" / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "scripts", "push_to_github.sh"), scripts / "push_to_github.sh")
    r = run(box, scripts / "push_to_github.sh", **env)
    calls = git_calls(box)[0]
    return r, calls, read(box.tmp / "gh.log")


def test_push_script_names_this_repository_and_adds_origin_only_when_there_is_none(box):
    r, calls, gh = push_script(box)
    assert r.returncode == 0, r.both
    assert "ARG LAWLESS1987/threefold\n" in gh and "LAWLESS1987/triad" not in gh + r.both
    assert ["remote", "add", "origin", "https://github.com/LAWLESS1987/threefold.git"] in calls
    assert not any(c[:2] == ["remote", "remove"] for c in calls), "an origin was removed"
    assert ["push", "-u", "origin", "main"] in calls


@pytest.mark.parametrize("origin", ["https://github.com/LAWLESS1987/threefold.git", "https://github.com/lawless1987/threefold",
                                    "git@github.com:LAWLESS1987/threefold.git"])
def test_push_script_leaves_an_existing_origin_alone(box, origin):
    r, calls, _ = push_script(box, FAKE_GIT_ORIGIN=origin)
    assert r.returncode == 0, r.both
    assert not any(c[:2] in (["remote", "remove"], ["remote", "add"], ["remote", "set-url"]) for c in calls)
    assert ["push", "-u", "origin", "main"] in calls


def test_push_script_stops_when_origin_points_somewhere_else(box):
    r, calls, _ = push_script(box, FAKE_GIT_ORIGIN="https://example.invalid/someone/else.git")
    assert r.returncode != 0 and "origin" in r.stderr, r.both
    assert not any(c[:1] == ["push"] or c[:2] in (["remote", "remove"], ["remote", "add"], ["remote", "set-url"])
                   for c in calls), "the script went on although origin points somewhere unexpected"


# --------------------------------------------------------------------------- K5: the README installs what it measured

def test_readme_installs_the_lens_library_at_the_commit_it_names():
    text = read(os.path.join(ROOT, "README.md"))
    named = re.search(r"jlens \S+ at ([0-9a-f]{7,40})\b", text)
    assert named, "the README no longer names the upstream commit its results were measured at"
    installs = re.findall(r"git\+https://github\.com/anthropics/jacobian-lens([^\s\"']*)", text)
    assert installs, "the README no longer shows how to install the lens library"
    assert all(i == "@" + named.group(1) for i in installs), \
        "the README installs %r but states results at %s" % (installs, named.group(1))
