#!/usr/bin/env python3
"""Does a stored capture hold what upstream returns? An independent re-run, compared cell by cell.

For a capture made by jlens_snapshot.py, this loads the same model and lens, calls upstream
jlens.JacobianLens.apply the way the upstream README does (all requested layers in ONE call, which
jlens_snapshot.py does not do), and checks:
  - the stored top-K ids and logits equal the direct call's, exactly, for every layer and position;
  - the stored final-layer top-K equals the direct call's model_logits top-K;
  - each full [positions, vocab] tensor hashes to the full_tensor_sha256 L-lens recorded at capture time;
  - the use_jacobian=False baseline, likewise.
Exit 0 only if everything matches. Needs torch, transformers, jlens, the model and the lens file.

Usage: check_against_upstream.py RUN_DIR [--lens PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import transformers
import jlens

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import llens  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--lens", default="")
    a = ap.parse_args(argv)
    cap = json.load(open(os.path.join(a.run_dir, "jlens_snapshot.json"), encoding="utf-8"))
    raw = llens.load_raw(os.path.join(a.run_dir, cap["raw"]["file"]))
    ro = cap["readout"]
    lens_path = a.lens or cap["lens"]["path"]
    if llens_sha(lens_path) != cap["lens"]["sha256"]:
        print("FAIL lens file differs from the one the capture recorded")
        return 1
    hf = transformers.AutoModelForCausalLM.from_pretrained(cap["subject_model"]["id"],
                                                           revision=cap["subject_model"].get("revision"))
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(cap["subject_model"]["id"],
                                                     revision=cap["subject_model"].get("revision"))
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(lens_path)
    k, bad, n = ro["top_k"], 0, 0
    fams = [("jacobian", True)] + ([("baseline", False)] if ro["baseline"] else [])
    for fam, use_j in fams:
        lens_logits, model_logits, ids = lens.apply(model, cap["input"]["prompt"], layers=ro["layers"],
                                                    positions=ro["positions"], max_seq_len=cap["input"]["max_seq_len"],
                                                    use_jacobian=use_j)
        n += 1
        if ids[0].tolist() != cap["input"]["input_ids"]:
            print("FAIL %s: input ids differ" % fam)
            bad += 1
        for li, layer in enumerate(ro["layers"]):
            v, i = lens_logits[layer].topk(k, dim=-1)
            ok_ids = np.array_equal(i.numpy(), raw[fam + "_top_ids"][li])
            ok_val = np.array_equal(v.numpy().astype(np.float32), raw[fam + "_top_logits"][li])
            ok_full = (llens.array_sha256(lens_logits[layer].numpy().astype(np.float32))
                       == cap["llens_capture_time"]["full_tensor_sha256"][fam][str(layer)])
            n += 3
            if not (ok_ids and ok_val and ok_full):
                bad += 1
                print("FAIL %s layer %d: ids %s, logits %s, full-tensor digest %s" % (fam, layer, ok_ids, ok_val, ok_full))
        v, i = model_logits.topk(k, dim=-1)
        ok = (np.array_equal(i.numpy(), raw["final_top_ids"]) and np.array_equal(v.numpy(), raw["final_top_logits"])
              and llens.array_sha256(model_logits.numpy().astype(np.float32))
              == cap["llens_capture_time"]["full_tensor_sha256"]["final"])
        n += 1
        if not ok:
            bad += 1
            print("FAIL %s: final-layer top-K or digest differs" % fam)
    print("UPSTREAM_MATCH %d of %d comparisons" % (n - bad, n) if not bad else "UPSTREAM_MISMATCH %d" % bad)
    return 1 if bad else 0


def llens_sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
