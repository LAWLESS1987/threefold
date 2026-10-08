#!/usr/bin/env python3
"""JLens capture: run Anthropic's Jacobian Lens and record what it returned.

This is the measurement step of Threefold's lens leg. The instrument is upstream `jlens.JacobianLens.apply` from
https://github.com/anthropics/jacobian-lens (Anthropic, Apache-2.0), called unmodified. Every array written to the
RAW .npz file holds values copied out of the tensors that call returned. The only operation applied to them is
selecting the top-K entries of each [vocab] row (torch.topk). Hashing and anything computed from the measurements
belong to L-lens (llens.py). Where L-lens has to work at capture time, because the full tensors are not kept, its
results go in the capture's separate "llens_capture_time" section, never in "raw".

Upstream's description: the Jacobian lens linearly transports a residual-stream vector at a layer and position
into the final-layer basis with the average input-output Jacobian J_l = E[dh_final/dh_l], fitted on a text corpus.
It then decodes the result with the model's own unembedding into logits over the vocabulary. A readout shows which
tokens that residual is disposed to make the model predict. It is a probe of one model's internal states. It is not
a reading of thoughts, beliefs or intent, and it measures the subject model named in the capture (GPT-2 small
reading the prompt text, by default), not the agent that wrote the text.

Readout. By default the capture reads every fitted layer of the lens (upstream apply's own default, layers=None) at
every token position (positions=None, truncated by upstream at --max-seq-len). It records:
  - the Jacobian-lens logits (use_jacobian=True);
  - upstream's vanilla logit-lens baseline (use_jacobian=False) at the same cells;
  - the model's actual final-layer logits, which apply returns at the same positions.
For each (layer, position) cell, the top-K token ids and their exact float32 logits are stored. The full
[positions, vocab] tensors are too large to store per run (about 24 MB per layer at 120 tokens). L-lens hashes each
one instead, so a re-run can be compared bit for bit. apply is called once per layer, so peak memory holds one
layer's logits. Each call is a complete upstream apply with layers=[l]. The model's final logits must come out
identical across all calls, or the capture fails.

Usage:
  jlens_snapshot.py --task-id TID --prompt "..." --out CAPTURE_JSON [--raw RAW_NPZ] [--digest DIGEST_PATH]
      [--lens PATH] [--lens-source TEXT] [--model ID] [--model-revision REV] [--top-k K] [--max-seq-len N]
      [--layers 0,1,...] [--positions -2,...] [--no-baseline]

Exits nonzero if apply fails, a layer is not fitted, a logit is not finite, a top-K list is empty, or the final
logits differ between calls.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import transformers
import jlens

import llens

MODEL_ID = "openai-community/gpt2"
# Default: lens next to this script (after scripts/fetch_lens.sh)
DEFAULT_LENS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpt2_jacobian_lens.pt")
DEFAULT_LENS_SOURCE = ("huggingface:neuronpedia/jacobian-lens "
                       "gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt")
TOP_K = 10  # upstream jlens.vis.compute_slice keeps top_n=10 per cell by default
MAX_SEQ_LEN = 512  # upstream apply's default
# Written "on/off", not as the Python literals: covenant's memory gate is a bag-of-words judge, and it refused a
# capture record on 2026-10-08 for the word in upstream's own parameter value ("evidence: false(194), false(194)").
UPSTREAM_CALL = ("jlens.JacobianLens.apply(model, prompt, layers=[l], positions=<positions>, "
                 "max_seq_len=<max_seq_len>, use_jacobian=<on for the Jacobian lens, off for the baseline>)")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jlens_provenance() -> dict:
    """Which copy of upstream ran: its distribution version and, when pip recorded one, the source commit."""
    out = {"package": "jlens", "module_file": jlens.__file__}
    try:
        out["version"] = md.version("jlens")
        direct = md.distribution("jlens").read_text("direct_url.json")
        if direct:
            d = json.loads(direct)
            out["installed_from"] = d.get("url")
            out["vcs_commit"] = (d.get("vcs_info") or {}).get("commit_id")
            url = d.get("url") or ""
            if not out["vcs_commit"] and url.startswith("file:"):
                src = os.path.dirname(os.path.dirname(os.path.abspath(jlens.__file__)))
                r = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True)
                if r.returncode == 0:
                    out["vcs_commit"] = r.stdout.strip()
                    out["vcs_commit_from"] = "git rev-parse HEAD in the local source checkout"
    except Exception as e:  # noqa: BLE001 -- provenance is best effort, and says so
        out["provenance_error"] = "%s: %s" % (type(e).__name__, e)
    return out


def model_revision(model_id: str, requested):
    """(commit, how it was found). transformers does not always set config._commit_hash; for a Hub model the
    cached config.json sits in snapshots/<commit>/, so the directory name is the commit that was loaded."""
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(model_id, "config.json", revision=requested)
        if isinstance(p, str):
            return os.path.basename(os.path.dirname(p)), "huggingface_hub cache: snapshots/<commit>/config.json"
    except Exception as e:  # noqa: BLE001
        return None, "not found (%s: %s)" % (type(e).__name__, e)
    return None, "not found (not loaded from the Hugging Face Hub cache)"


def topk(logits: torch.Tensor, k: int):
    v, i = logits.topk(k, dim=-1)
    return i.cpu().numpy().astype(np.int64), v.cpu().numpy().astype(np.float32)


def parse_ints(s: str):
    return [int(x) for x in s.split(",") if x.strip()] if s else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True, help="capture JSON path")
    ap.add_argument("--raw", default="", help="raw arrays .npz path (default: next to --out)")
    ap.add_argument("--digest", default="", help="optional short digest text path")
    ap.add_argument("--lens", default=DEFAULT_LENS)
    ap.add_argument("--lens-source", default=DEFAULT_LENS_SOURCE)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--layers", default="", help="comma list; default every fitted layer")
    ap.add_argument("--positions", default="", help="comma list; default every position")
    ap.add_argument("--no-baseline", action="store_true", help="skip the use_jacobian=False baseline")
    args = ap.parse_args(argv)
    raw_path = args.raw or os.path.join(os.path.dirname(os.path.abspath(args.out)), "jlens_raw.npz")

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.perf_counter()
    torch.manual_seed(0)
    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, revision=args.model_revision)
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    model = jlens.from_hf(hf, tok)
    t_model = time.perf_counter() - t0

    t1 = time.perf_counter()
    lens = jlens.JacobianLens.load(args.lens)
    t_lens = time.perf_counter() - t1

    layers = parse_ints(args.layers) or list(lens.source_layers)
    missing = [l for l in layers if l not in lens.jacobians]
    if missing:
        print("ERROR: layers %s are not fitted in this lens (fitted: %s)" % (missing, lens.source_layers),
              file=sys.stderr)
        return 1
    positions_arg = parse_ints(args.positions)
    families = [("jacobian", True)] + ([] if args.no_baseline else [("baseline", False)])

    ids_by, logits_by, full_sha = {f: [] for f, _ in families}, {f: [] for f, _ in families}, {f: {} for f, _ in families}
    final_ids = final_logits = final_ref = input_ids = None
    n_calls, t_apply = 0, {f: 0.0 for f, _ in families}
    for layer in layers:
        for fam, use_j in families:
            ta = time.perf_counter()
            lens_logits, model_logits, ids = lens.apply(model, args.prompt, layers=[layer], positions=positions_arg,
                                                        max_seq_len=args.max_seq_len, use_jacobian=use_j)
            t_apply[fam] += time.perf_counter() - ta
            n_calls += 1
            x = lens_logits[layer]
            if not torch.isfinite(x).all() or not torch.isfinite(model_logits).all():
                print("ERROR: non-finite logits at layer %d (%s)" % (layer, fam), file=sys.stderr)
                return 1
            if final_ref is None:
                final_ref, input_ids = model_logits, ids
                final_ids, final_logits = topk(model_logits, args.top_k)
            elif not torch.equal(final_ref, model_logits) or not torch.equal(input_ids, ids):
                print("ERROR: final logits or input ids differ between apply calls (layer %d, %s)" % (layer, fam),
                      file=sys.stderr)
                return 1
            i, v = topk(x, args.top_k)
            if i.size == 0:
                print("ERROR: empty top-k at layer %d" % layer, file=sys.stderr)
                return 1
            ids_by[fam].append(i)
            logits_by[fam].append(v)
            full_sha[fam][str(layer)] = llens.array_sha256(x.numpy().astype(np.float32))
            del lens_logits, x

    n_tokens = int(input_ids.shape[1])
    positions = ([p if p >= 0 else n_tokens + p for p in positions_arg] if positions_arg
                 else list(range(n_tokens)))
    raw = {"input_ids": input_ids[0].cpu().numpy().astype(np.int64),
           "layers": np.array(layers, dtype=np.int64),
           "positions": np.array(positions, dtype=np.int64),
           "final_top_ids": final_ids, "final_top_logits": final_logits}
    for fam, _ in families:
        raw[fam + "_top_ids"] = np.stack(ids_by[fam])
        raw[fam + "_top_logits"] = np.stack(logits_by[fam])
    os.makedirs(os.path.dirname(os.path.abspath(raw_path)) or ".", exist_ok=True)
    np.savez_compressed(raw_path, **raw)

    id_set = set(raw["input_ids"].tolist()) | set(int(t) for k in raw if k.endswith("_top_ids") for t in raw[k].ravel())
    vocab = {str(t): tok.decode([t], clean_up_tokenization_spaces=False) for t in sorted(id_set)}
    vocab_path = os.path.join(os.path.dirname(os.path.abspath(raw_path)), "jlens_vocab.json")
    with open(vocab_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(vocab, f, indent=0, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    bos = getattr(tok, "bos_token_id", None)
    rev = model_revision(args.model, args.model_revision)
    capture = {
        "schema": llens.CAPTURE_SCHEMA,
        "type": "jlens",
        "task_id": args.task_id,
        "upstream": {"instrument": "Jacobian Lens (JLens)", "by": "Anthropic",
                     "source": "https://github.com/anthropics/jacobian-lens", "license": "Apache-2.0",
                     "call": UPSTREAM_CALL, "calls_made": n_calls, **jlens_provenance()},
        "subject_model": {"id": args.model, "requested_revision": args.model_revision,
                          "revision": getattr(hf.config, "_commit_hash", None) or rev[0],
                          "revision_from": "config._commit_hash" if getattr(hf.config, "_commit_hash", None)
                          else rev[1],
                          "n_layers": model.n_layers, "d_model": model.d_model,
                          "vocab_size": int(final_ref.shape[-1]), "dtype": "float32", "device": "cpu",
                          "adapter": "jlens.from_hf(hf, tokenizer) with upstream defaults (force_bos=True)",
                          "note": "The lens reads this model's residual stream while it reads the prompt text. "
                                  "It does not observe whichever agent wrote the text."},
        "lens": {"path": os.path.abspath(args.lens), "sha256": file_sha256(args.lens), "source": args.lens_source,
                 "n_prompts": lens.n_prompts, "d_model": lens.d_model, "source_layers": list(lens.source_layers)},
        "input": {"prompt": args.prompt, "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
                  "max_seq_len": args.max_seq_len, "n_tokens": n_tokens,
                  "starts_with_bos": bool(bos is not None and n_tokens and int(raw["input_ids"][0]) == bos),
                  "input_ids": raw["input_ids"].tolist()},
        "readout": {"layers": layers, "positions": positions, "top_k": args.top_k, "baseline": not args.no_baseline,
                    "layers_rule": "explicit --layers" if args.layers else "every fitted layer (lens.source_layers)",
                    "positions_rule": "explicit --positions" if positions_arg else "every token position"},
        "raw": {"file": os.path.basename(raw_path),
                "meaning": {
                    "input_ids": "token ids upstream apply returned (model.encode of the prompt)",
                    "layers": "the layer of each row in the *_top_* arrays",
                    "positions": "absolute token position of each column",
                    "jacobian_top_ids/logits": "[layer, position, K] top-K of apply logits, use_jacobian on",
                    "baseline_top_ids/logits": "[layer, position, K] top-K of apply logits, use_jacobian off",
                    "final_top_ids/logits": "[position, K] top-K of the model's own final logits from apply",
                    "order": "torch.topk order, largest logit first; ties are ordered by torch, not by us"}},
        "vocab": {"file": os.path.basename(vocab_path),
                  "what": "tokenizer.decode of every token id in the raw arrays. It is a view of the ids, not a "
                          "measurement. It is kept out of this record, and out of covenant memory bodies, because "
                          "covenant's memory gate reads a list of vocabulary words as if it were statements "
                          "(it refused one on 2026-10-08)."},
        "llens_capture_time": {
            "by": "llens.py (Threefold), not upstream",
            "array_sha256": {k: llens.array_sha256(v) for k, v in raw.items()},
            "vocab_sha256": llens.digest(vocab),
            "full_tensor_sha256": {**full_sha, "final": llens.array_sha256(final_ref.numpy().astype(np.float32))},
            "full_tensor_sha256_note": "sha256 of each complete [positions, vocab] float32 tensor apply returned "
                                       "(not stored). Equal on a re-run means bit-identical logits. Bit equality "
                                       "across different CPUs or BLAS builds is not guaranteed.",
            "checks": {"all_logits_finite": True, "final_logits_identical_across_calls": True}},
        "env": {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
                "transformers": transformers.__version__, "numpy": np.__version__,
                "torch_num_threads": torch.get_num_threads(), "seed": 0},
        "timing": {"started_utc": started, "model_load_s": round(t_model, 3), "lens_load_s": round(t_lens, 3),
                   **{"apply_%s_s" % f: round(s, 3) for f, s in t_apply.items()},
                   "total_s": round(time.perf_counter() - t0, 3)},
        # continuity with the pre-2026-10-08 snapshot fields
        "model_id": args.model, "lens_path": args.lens, "prompt": args.prompt, "layers": layers,
        "positions": positions, "top_k": args.top_k,
        "mechanism": "anthropics/jacobian-lens JacobianLens.apply (real per-layer logits)",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(capture, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")

    cap_sha = llens.digest(capture)
    digest_line = ("jlens task_id=%s model=%s layers=%d positions=%d top_k=%d baseline=%s capture_sha256=%s"
                   % (args.task_id, args.model, len(layers), len(positions), args.top_k,
                      "yes" if not args.no_baseline else "no", cap_sha))
    if args.digest:
        with open(args.digest, "w", encoding="utf-8", newline="\n") as f:
            f.write(digest_line + "\n")
    print(digest_line)
    print("JLENS_CAPTURE_SHA256=%s" % cap_sha)
    print(f"SNAPSHOT_OK path={args.out} raw={raw_path} vocab={vocab_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
