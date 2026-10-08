#!/usr/bin/env python3
"""L-lens: Threefold's observability and provenance layer around upstream Jacobian Lens measurements.

L-lens is not the Jacobian Lens, and it is not a new name for it. The Jacobian Lens (JLens) is Anthropic's
mechanistic-interpretability instrument (https://github.com/anthropics/jacobian-lens, Apache-2.0). It linearly
transports a residual-stream vector at one layer and position into the final-layer basis, then decodes it with the
model's own unembedding into logits over the vocabulary. `jlens_snapshot.py` runs it and stores what it returned
("raw").

L-lens is the part Threefold adds around those measurements. It does four things, all of which are ours:

  1. Hashing. It hashes every stored raw array, and the full [positions, vocab] tensors that are too large to keep
     (at capture time). It also hashes the capture record in canonical JSON, so a later reader can tell whether
     anything changed.
  2. Derivation. It derives a small set of integer summaries from the stored raw arrays (DEFINITIONS below). Every
     one can be recomputed exactly from those arrays, and `verify_llens` does recompute them.
  3. Packaging. It puts the summaries and the digests into one artifact under the triad's task_id. The artifact
     holds token ids, never decoded vocabulary strings. `llens.py show` decodes them for a reader from the
     capture's hashed vocabulary file.
  4. Linking. The tombstone entry and the covenant task memory for that task_id cite the artifact's digest, and
     `triad/verify_triad.py` fails if any of the links is absent or disagrees.

What none of it is: a reading of thoughts, beliefs, intent or truth. A JLens readout says which tokens a layer's
residual is disposed to make the model predict, under a lens fitted on a text corpus. The subject is the model named
in the capture (by default GPT-2 small reading the prompt text), not the agent that wrote the text. The derived
summaries describe how each layer's top-K list relates to the model's own final top-K at the same position. That is
an observability statistic about one model reading one text.

Usage:
  llens.py build  --capture CAPTURE_JSON --raw RAW_NPZ --out LLENS_JSON   (prints LLENS_SHA256=<hex>)
  llens.py verify --capture CAPTURE_JSON --raw RAW_NPZ [--llens LLENS_JSON]
  llens.py show   --capture CAPTURE_JSON --raw RAW_NPZ [--position P] [--n N]
The vocabulary file is read from the capture's directory (capture["vocab"]["file"]).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time

import numpy as np

SCHEMA = "threefold.llens/1"
CAPTURE_SCHEMA = "threefold.jlens-capture/2"

STATEMENT = (
    "L-lens is the Threefold observability and provenance layer built around upstream Jacobian Lens measurements "
    "(anthropics/jacobian-lens, Apache-2.0). It is not a replacement name for the Jacobian Lens and adds no "
    "measurement of its own. Everything under 'derived' is computed by L-lens from the stored raw arrays of the "
    "capture this artifact cites. Nothing here is a reading of thoughts, beliefs, intent or ground truth."
)

UPSTREAM_CREDIT = {
    "instrument": "Jacobian Lens (JLens)",
    "by": "Anthropic",
    "source": "https://github.com/anthropics/jacobian-lens",
    "license": "Apache-2.0",
    "paper": "https://transformer-circuits.pub/2026/workspace/index.html",
}

DEFINITIONS = {
    "family": "jacobian = upstream JacobianLens.apply with use_jacobian on; baseline = the same call with "
              "use_jacobian off (upstream's vanilla logit-lens baseline); final = the model's own final-layer "
              "logits at the same positions, returned by the same calls.",
    "top1_agree": "for one layer: the number of positions whose lens top-1 token id equals the final top-1 id.",
    "final_top1_in_topk": "for one layer: the number of positions whose final top-1 id appears anywhere in the "
                          "lens top-K list.",
    "topk_overlap": "for one layer: the sum over positions of |lens top-K ids AND final top-K ids|, out of "
                    "n_positions * K.",
    "first_layer_final_top1_in_topk": "for one position: the first layer, in readout order, whose lens top-K list "
                                      "contains the final top-1 id; null if none does. Limited to the stored top-K: "
                                      "a token ranked K+1 counts as absent.",
}


# ----------------------------------------------------------------------------- hashing (L-lens)

def canonical_json(obj) -> bytes:
    """One byte string per JSON value: sorted keys, no whitespace, UTF-8. Whitespace or key order in a file or a
    memory body therefore cannot change a digest."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(obj) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def array_sha256(a: np.ndarray) -> str:
    """sha256 over dtype, shape and little-endian C-order bytes, so a reshaped or re-typed copy does not match."""
    a = np.ascontiguousarray(a)
    a = a.astype(a.dtype.newbyteorder("<"), copy=False)
    h = hashlib.sha256()
    h.update(("%s|%s|" % (a.dtype.str, ",".join(str(d) for d in a.shape))).encode("ascii"))
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def load_raw(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k]) for k in z.files}


def load_vocab(capture_path: str, capture: dict):
    """The capture's vocabulary file ({id: decoded string}), or None when the capture names none."""
    name = (capture.get("vocab") or {}).get("file")
    if not name:
        return None
    import os
    with open(os.path.join(os.path.dirname(os.path.abspath(capture_path)), name), encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- derivation (L-lens)

def _family_stats(ids: np.ndarray, final: np.ndarray, layers: list) -> dict:
    """ids: [L, n, K] lens top-K ids; final: [n, K] the model's own top-K ids at the same positions."""
    n, k = final.shape
    per_layer, first = {}, [None] * n
    for li, layer in enumerate(layers):
        lens_ids = ids[li]
        agree = int(np.sum(lens_ids[:, 0] == final[:, 0]))
        in_topk = [bool(final[p, 0] in lens_ids[p]) for p in range(n)]
        overlap = int(sum(len(set(lens_ids[p].tolist()) & set(final[p].tolist())) for p in range(n)))
        per_layer[str(layer)] = {"top1_agree": agree, "final_top1_in_topk": int(sum(in_topk)),
                                 "topk_overlap": overlap, "n_positions": n, "k": k}
        for p in range(n):
            if first[p] is None and in_topk[p]:
                first[p] = int(layer)
    return {"per_layer": per_layer, "first_layer_final_top1_in_topk": first}


def derive(raw: dict) -> dict:
    """The derived section: a pure function of the stored raw arrays, so verify can recompute it exactly."""
    layers = [int(x) for x in raw["layers"].tolist()]
    out = {"definitions": DEFINITIONS, "layers": layers, "positions": [int(x) for x in raw["positions"].tolist()],
           "jacobian": _family_stats(raw["jacobian_top_ids"], raw["final_top_ids"], layers)}
    if "baseline_top_ids" in raw:
        out["baseline"] = _family_stats(raw["baseline_top_ids"], raw["final_top_ids"], layers)
    return out


def readable(raw: dict, vocab: dict, position: int = -1, n_tokens: int = 5) -> dict:
    """A human view of one position, decoded with the capture's vocabulary file (the subject model's own tokenizer
    output at capture time). Presentation only: it is not stored in the artifact, and nothing is computed here."""
    word = lambda i: (vocab or {}).get(str(int(i)), "<id %d>" % int(i))  # noqa: E731
    p = position % int(raw["final_top_ids"].shape[0])
    layers = [int(x) for x in raw["layers"].tolist()]
    out = {"position": int(raw["positions"][p]),
           "jacobian_top_tokens_by_layer": {str(l): [word(i) for i in raw["jacobian_top_ids"][li, p, :n_tokens]]
                                            for li, l in enumerate(layers)},
           "final_top_tokens": [word(i) for i in raw["final_top_ids"][p, :n_tokens]]}
    if "baseline_top_ids" in raw:
        out["baseline_top_tokens_by_layer"] = {str(l): [word(i) for i in raw["baseline_top_ids"][li, p, :n_tokens]]
                                               for li, l in enumerate(layers)}
    return out


def build(capture: dict, raw: dict, vocab=None) -> dict:
    problems = [why for ok, _, why in verify_capture(capture, raw, vocab) if not ok]
    if problems:
        raise ValueError("capture does not verify; L-lens will not package it: " + "; ".join(problems))
    task_id = capture["task_id"]
    return {
        "schema": SCHEMA,
        "type": "llens",
        "task_id": task_id,
        "what": STATEMENT,
        "upstream_credit": UPSTREAM_CREDIT,
        "subject": {"model_id": capture["subject_model"]["id"],
                    "revision": capture["subject_model"].get("revision"),
                    "note": capture["subject_model"].get("note")},
        "inputs": {"jlens_capture_sha256": digest(capture), "jlens_capture_schema": capture.get("schema"),
                   "raw_array_sha256": capture["llens_capture_time"]["array_sha256"]},
        "derived": derive(raw),
        "links": {"task_id": task_id,
                  "tombstone": "the tombstone entry whose notes carry task_id=%s and llens_sha256" % task_id,
                  "covenant": ["task-%s" % task_id, "jlens-%s" % task_id, "llens-%s" % task_id]},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ----------------------------------------------------------------------------- verification

def verify_capture(capture: dict, raw: dict, vocab=None) -> list:
    """[(ok, check, why)] -- the capture's own consistency: its arrays match the digests it recorded, their shapes
    match the readout it declares, and (when given) the vocabulary file is the one it recorded."""
    out = []

    def check(ok, name, why=""):
        out.append((bool(ok), name, why))

    if vocab is not None:
        check(digest(vocab) == (capture.get("llens_capture_time") or {}).get("vocab_sha256"), "vocab digest",
              "the vocabulary file is not the one recorded at capture time")

    check(capture.get("schema") == CAPTURE_SCHEMA, "capture schema", "schema is %r" % capture.get("schema"))
    check(bool(capture.get("task_id")), "capture task_id", "no task_id")
    recorded = (capture.get("llens_capture_time") or {}).get("array_sha256") or {}
    check(set(recorded) == set(raw), "raw array set",
          "recorded %s, file has %s" % (sorted(recorded), sorted(raw)))
    for name in sorted(set(recorded) & set(raw)):
        check(array_sha256(raw[name]) == recorded[name], "raw %s digest" % name, "%s changed since capture" % name)
    ro = capture.get("readout") or {}
    try:
        L, n, K = len(ro["layers"]), len(ro["positions"]), int(ro["top_k"])
        shapes = {"jacobian_top_ids": (L, n, K), "jacobian_top_logits": (L, n, K),
                  "final_top_ids": (n, K), "final_top_logits": (n, K), "layers": (L,), "positions": (n,)}
        if ro.get("baseline"):
            shapes.update(baseline_top_ids=(L, n, K), baseline_top_logits=(L, n, K))
        for name, shape in shapes.items():
            check(name in raw and tuple(raw[name].shape) == shape, "raw %s shape" % name,
                  "%s is %s, readout says %s" % (name, tuple(raw[name].shape) if name in raw else None, shape))
        check("layers" in raw and [int(x) for x in raw["layers"]] == list(ro["layers"]), "raw layers agree",
              "the layers array is not the declared readout")
        check("positions" in raw and [int(x) for x in raw["positions"]] == list(ro["positions"]),
              "raw positions agree", "the positions array is not the declared readout")
    except (KeyError, TypeError, ValueError) as e:
        check(False, "readout declared", "%s: %s" % (type(e).__name__, e))
    return out


def verify_llens(llens: dict, capture: dict, raw: dict, vocab=None) -> list:
    out = verify_capture(capture, raw, vocab)

    def check(ok, name, why=""):
        out.append((bool(ok), name, why))

    check(llens.get("schema") == SCHEMA, "llens schema", "schema is %r" % llens.get("schema"))
    check(llens.get("task_id") == capture.get("task_id"), "llens task_id agrees",
          "llens %r, capture %r" % (llens.get("task_id"), capture.get("task_id")))
    check((llens.get("inputs") or {}).get("jlens_capture_sha256") == digest(capture), "llens cites this capture",
          "the capture's digest is not the one L-lens recorded")
    check(llens.get("derived") == json.loads(json.dumps(derive(raw))), "derived recomputes from raw",
          "the stored derived section is not what the raw arrays give")
    return out


def _report(results) -> int:
    bad = [r for r in results if not r[0]]
    for ok, name, why in results:
        print("  %s  %s%s" % ("PASS" if ok else "FAIL", name, "" if ok else "  -- " + why))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--capture", required=True)
    b.add_argument("--raw", required=True)
    b.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--capture", required=True)
    v.add_argument("--raw", required=True)
    v.add_argument("--llens", default="")
    s = sub.add_parser("show")
    s.add_argument("--capture", required=True)
    s.add_argument("--raw", required=True)
    s.add_argument("--position", type=int, default=-1, help="index into the readout's positions (default last)")
    s.add_argument("--n", type=int, default=5)
    a = ap.parse_args(argv)

    with open(a.capture, encoding="utf-8") as f:
        capture = json.load(f)
    raw = load_raw(a.raw)
    try:
        vocab = load_vocab(a.capture, capture)
    except OSError as e:
        print("LLENS_FAIL the capture names a vocabulary file that cannot be read: %s" % e, file=sys.stderr)
        return 1
    if a.cmd == "show":
        print(json.dumps(readable(raw, vocab, a.position, a.n), indent=1, ensure_ascii=False))
        return 0
    if a.cmd == "build":
        try:
            art = build(capture, raw, vocab)
        except ValueError as e:
            print("LLENS_FAIL %s" % e, file=sys.stderr)
            return 1
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(art, f, indent=1, sort_keys=True, ensure_ascii=False)
            f.write("\n")
        print("LLENS_SHA256=%s" % digest(art))
        print("LLENS_OK path=%s" % a.out)
        return 0
    if a.llens:
        with open(a.llens, encoding="utf-8") as f:
            rc = _report(verify_llens(json.load(f), capture, raw, vocab))
    else:
        rc = _report(verify_capture(capture, raw, vocab))
    print("LLENS_VERIFY_OK" if rc == 0 else "LLENS_VERIFY_FAIL")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
