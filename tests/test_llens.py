"""L-lens and triad verification, without a model: synthetic captures, then each link broken on purpose.

Every check is driven both ways. A clean artifact passes, and one specific tamper must fail with a named reason. A
verifier that has only been seen passing has not been observed.
"""
from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "triad"))
import llens  # noqa: E402
import verify_triad  # noqa: E402

TASK = "20261008-000000-1"


def synthetic(task_id=TASK, baseline=True):
    """2 layers x 3 positions x K=2, with known answers:
    final top ids per position: [5,6], [7,8], [9,1]
    jacobian layer 0: [5,0], [0,7], [2,3]   -> top1 agree 1 (pos 0); final top1 in top-K: pos0, pos1 -> 2
    jacobian layer 1: [5,6], [7,8], [1,9]   -> top1 agree 2; in top-K 3; overlap 2+2+2 = 6
    first layer the final top1 is in top-K: pos0 -> 0, pos1 -> 0, pos2 -> 1
    """
    raw = {"input_ids": np.array([50256, 11, 12], dtype=np.int64),
           "layers": np.array([0, 1], dtype=np.int64), "positions": np.array([0, 1, 2], dtype=np.int64),
           "final_top_ids": np.array([[5, 6], [7, 8], [9, 1]], dtype=np.int64),
           "final_top_logits": np.array([[3, 2], [3, 2], [3, 2]], dtype=np.float32),
           "jacobian_top_ids": np.array([[[5, 0], [0, 7], [2, 3]], [[5, 6], [7, 8], [1, 9]]], dtype=np.int64),
           "jacobian_top_logits": np.ones((2, 3, 2), dtype=np.float32)}
    if baseline:
        raw["baseline_top_ids"] = np.array([[[0, 1], [2, 3], [4, 5]], [[5, 0], [7, 0], [9, 0]]], dtype=np.int64)
        raw["baseline_top_logits"] = np.ones((2, 3, 2), dtype=np.float32)
    cap = {"schema": llens.CAPTURE_SCHEMA, "type": "jlens", "task_id": task_id,
           "subject_model": {"id": "synthetic", "revision": None, "note": "test"},
           "input": {"prompt": "a b", "input_ids": [50256, 11, 12]},
           "readout": {"layers": [0, 1], "positions": [0, 1, 2], "top_k": 2, "baseline": baseline},
           "vocab": {"file": "jlens_vocab.json"},
           "llens_capture_time": {"array_sha256": {k: llens.array_sha256(v) for k, v in raw.items()},
                                  "vocab_sha256": llens.digest(VOCAB)}}
    return cap, raw


VOCAB = {str(i): "t%d" % i for i in range(20)} | {"50256": "<|endoftext|>"}


def roundtrip(obj):
    return json.loads(json.dumps(obj))


# --------------------------------------------------------------------------- derivation

def test_derived_values_are_the_hand_computed_ones():
    cap, raw = synthetic()
    d = llens.derive(raw)
    j = d["jacobian"]["per_layer"]
    assert j["0"] == {"top1_agree": 1, "final_top1_in_topk": 2, "topk_overlap": 2, "n_positions": 3, "k": 2}
    assert j["1"] == {"top1_agree": 2, "final_top1_in_topk": 3, "topk_overlap": 6, "n_positions": 3, "k": 2}
    assert d["jacobian"]["first_layer_final_top1_in_topk"] == [0, 0, 1]
    assert d["baseline"]["per_layer"]["0"]["top1_agree"] == 0
    assert d["baseline"]["first_layer_final_top1_in_topk"] == [1, 1, 1]


def test_no_baseline_means_no_baseline_section():
    cap, raw = synthetic(baseline=False)
    assert "baseline" not in llens.derive(raw)
    assert all(ok for ok, _, _ in llens.verify_capture(cap, raw))


def test_artifact_says_what_it_is_and_credits_upstream():
    cap, raw = synthetic()
    art = llens.build(cap, raw)
    assert art["type"] == "llens" and art["task_id"] == TASK
    assert "not a replacement name for the Jacobian Lens" in art["what"]
    assert art["upstream_credit"]["source"] == "https://github.com/anthropics/jacobian-lens"
    assert art["upstream_credit"]["license"] == "Apache-2.0"
    assert art["inputs"]["jlens_capture_sha256"] == llens.digest(cap)


# --------------------------------------------------------------------------- hashing

def test_canonical_digest_ignores_whitespace_and_key_order_but_not_content():
    a = {"b": 1, "a": [1, 2]}
    assert llens.digest(a) == llens.digest(json.loads(json.dumps(a, indent=4)))
    assert llens.digest(a) != llens.digest({"b": 1, "a": [2, 1]})


def test_array_digest_sees_dtype_and_shape():
    x = np.arange(6, dtype=np.int64)
    assert llens.array_sha256(x) != llens.array_sha256(x.reshape(2, 3))
    assert llens.array_sha256(x) != llens.array_sha256(x.astype(np.int32))
    assert llens.array_sha256(x) == llens.array_sha256(x.copy())


# --------------------------------------------------------------------------- verify, both ways

def test_clean_capture_and_artifact_verify():
    cap, raw = synthetic()
    art = roundtrip(llens.build(cap, raw))
    assert all(ok for ok, _, _ in llens.verify_llens(art, cap, raw))


def failing(results):
    return [name for ok, name, _ in results if not ok]


def test_a_changed_raw_value_fails_its_digest():
    cap, raw = synthetic()
    art = roundtrip(llens.build(cap, raw))
    raw["jacobian_top_ids"][1, 2, 0] = 4
    assert "raw jacobian_top_ids digest" in failing(llens.verify_llens(art, cap, raw))


def test_a_changed_derived_section_fails_recompute():
    cap, raw = synthetic()
    art = roundtrip(llens.build(cap, raw))
    art["derived"]["jacobian"]["per_layer"]["1"]["top1_agree"] = 3
    assert failing(llens.verify_llens(art, cap, raw)) == ["derived recomputes from raw"]


def test_an_edited_capture_is_no_longer_the_one_cited():
    cap, raw = synthetic()
    art = roundtrip(llens.build(cap, raw))
    cap["input"]["prompt"] = "a c"
    assert failing(llens.verify_llens(art, cap, raw)) == ["llens cites this capture"]


def test_a_different_task_id_fails():
    cap, raw = synthetic()
    art = roundtrip(llens.build(cap, raw))
    art["task_id"] = "other"
    assert "llens task_id agrees" in failing(llens.verify_llens(art, cap, raw))


def test_a_shape_that_disagrees_with_the_readout_fails():
    cap, raw = synthetic()
    cap["readout"]["top_k"] = 3
    assert "raw jacobian_top_ids shape" in failing(llens.verify_capture(cap, raw))


def test_build_refuses_a_capture_that_does_not_verify():
    cap, raw = synthetic()
    raw["final_top_ids"][0, 0] = 1
    with pytest.raises(ValueError, match="does not verify"):
        llens.build(cap, raw)


# --------------------------------------------------------------------------- the triad, end to end on disk

def write_memory(root, name, body):
    with open(os.path.join(root, name + ".md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("---\nname: %s\ndescription: test\nmetadata:\n  type: project\n---\n\n%s\n" % (name, body))


def triad(tmp_path, task_id=TASK):
    cap, raw = synthetic(task_id)
    art = llens.build(cap, raw)
    run = tmp_path / "runs" / task_id
    run.mkdir(parents=True)
    json.dump(cap, open(run / "jlens_snapshot.json", "w", encoding="utf-8"), indent=1)
    np.savez_compressed(run / "jlens_raw.npz", **raw)
    json.dump(art, open(run / "llens.json", "w", encoding="utf-8"), indent=1)
    json.dump(VOCAB, open(run / "jlens_vocab.json", "w", encoding="utf-8"), indent=0)
    cs, ls = llens.digest(cap), llens.digest(art)
    log = tmp_path / "tombstone.md"
    log.write_text("# log\n\n### 2026-10-08T00:00:00Z | t\n- id: 1\n- task: t\n- outcome: success\n"
                   "- notes: task_id=%s; jlens_capture_sha256=%s; llens_sha256=%s; jlens_digest=jlens x=1\n"
                   % (task_id, cs, ls), encoding="utf-8")
    mem = tmp_path / "mem"
    mem.mkdir()
    write_memory(mem, "task-" + task_id, "task_id=%s\ntitle=t\njlens_capture_sha256=%s\nllens_sha256=%s" % (task_id, cs, ls))
    write_memory(mem, "jlens-" + task_id, json.dumps(cap, indent=2))
    write_memory(mem, "llens-" + task_id, json.dumps(art, indent=2))
    return {"run": str(run), "log": str(log), "mem": str(mem), "cap_sha": cs, "ll_sha": ls}


def run_verify(t, task_id=TASK):
    return verify_triad.verify(task_id, t["run"], t["log"], t["mem"])


def test_a_complete_triad_verifies(tmp_path):
    res = run_verify(triad(tmp_path))
    assert failing(res) == [] and len(res) > 20


def test_verifier_main_exits_1_on_failure_and_0_when_clean(tmp_path):
    t = triad(tmp_path)
    args = ["--task-id", TASK, "--run-dir", t["run"], "--tombstone-log", t["log"], "--memory-root", t["mem"]]
    assert verify_triad.main(args) == 0
    os.remove(os.path.join(t["mem"], "llens-%s.md" % TASK))
    assert verify_triad.main(args) == 1


def test_no_tombstone_entry_for_the_task_fails(tmp_path):
    t = triad(tmp_path)
    open(t["log"], "w", encoding="utf-8").write("# log\n")
    assert "tombstone: an entry for this task_id" in failing(run_verify(t))


def test_a_tombstone_entry_citing_another_artifact_fails(tmp_path):
    t = triad(tmp_path)
    text = open(t["log"], encoding="utf-8").read().replace(t["ll_sha"], "0" * 64)
    open(t["log"], "w", encoding="utf-8").write(text)
    assert failing(run_verify(t)) == ["tombstone: llens_sha256 agrees"]


def test_a_missing_covenant_memory_fails(tmp_path):
    t = triad(tmp_path)
    os.remove(os.path.join(t["mem"], "jlens-%s.md" % TASK))
    assert failing(run_verify(t)) == ["covenant: jlens-%s present" % TASK]


def test_a_covenant_task_memory_under_another_id_fails(tmp_path):
    t = triad(tmp_path)
    p = os.path.join(t["mem"], "task-%s.md" % TASK)
    text = open(p, encoding="utf-8").read()
    open(p, "w", encoding="utf-8").write(text.replace("task_id=%s" % TASK, "task_id=x"))
    assert failing(run_verify(t)) == ["covenant: task memory task_id"]


def test_an_altered_covenant_llens_body_fails(tmp_path):
    t = triad(tmp_path)
    art = json.load(open(os.path.join(t["run"], "llens.json"), encoding="utf-8"))
    art["what"] = art["what"] + " (edited)"
    write_memory(t["mem"], "llens-" + TASK, json.dumps(art))
    assert failing(run_verify(t)) == ["covenant: llens-%s body digest agrees" % TASK]


def test_a_missing_lens_artifact_fails_closed(tmp_path):
    t = triad(tmp_path)
    os.remove(os.path.join(t["run"], "llens.json"))
    f = failing(run_verify(t))
    assert "lens leg: llens.json present" in f
    assert "tombstone: llens_sha256 agrees" in f  # no artifact, so nothing can agree with the tombstone


def test_reformatting_a_memory_body_is_not_a_change(tmp_path):
    t = triad(tmp_path)
    cap = json.load(open(os.path.join(t["run"], "jlens_snapshot.json"), encoding="utf-8"))
    write_memory(t["mem"], "jlens-" + TASK, json.dumps(cap, separators=(",", ":"), sort_keys=True))
    assert failing(run_verify(t)) == []


def test_raw_npz_swapped_for_another_tasks_fails(tmp_path):
    t = triad(tmp_path)
    cap2, raw2 = synthetic("other-task")
    raw2["jacobian_top_ids"][0, 0, 1] = 19
    np.savez_compressed(os.path.join(t["run"], "jlens_raw.npz"), **raw2)
    assert "lens leg: raw jacobian_top_ids digest" in failing(run_verify(t))


def test_the_artifact_holds_no_decoded_vocabulary(tmp_path):
    cap, raw = synthetic()
    text = json.dumps(llens.build(cap, raw, VOCAB))
    assert not any(w in text for w in VOCAB.values())


def test_show_decodes_from_the_vocab_file_only():
    cap, raw = synthetic()
    r = llens.readable(raw, VOCAB, position=0, n_tokens=2)
    assert r["final_top_tokens"] == ["t5", "t6"] and r["jacobian_top_tokens_by_layer"]["1"] == ["t5", "t6"]


def test_a_changed_vocab_file_fails(tmp_path):
    t = triad(tmp_path)
    v = dict(VOCAB, **{"5": "changed"})
    json.dump(v, open(os.path.join(t["run"], "jlens_vocab.json"), "w", encoding="utf-8"))
    assert failing(run_verify(t)) == ["lens leg: vocab digest"]


def test_a_missing_vocab_file_fails_closed(tmp_path):
    t = triad(tmp_path)
    os.remove(os.path.join(t["run"], "jlens_vocab.json"))
    assert "lens leg: artifacts readable" in failing(run_verify(t))
