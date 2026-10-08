#!/usr/bin/env python3
"""Prove jlens.JacobianLens.apply returns real top-k tokens on CPU.

Model: openai-community/gpt2 (~124M)
Lens:  pretrained gpt2-small lens (fetch via scripts/fetch_lens.sh)
"""
from __future__ import annotations

import os
try:
    import resource
except ImportError:  # Windows has no resource module; RSS is then reported as nan
    resource = None
import time
import traceback

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import transformers
import jlens

MODEL_ID = "openai-community/gpt2"
LENS_PATH = os.environ.get(
    "JLENS_LENS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpt2_jacobian_lens.pt"),
)
PROMPT = os.environ.get(
    "TRIAD_PROMPT",
    "Fact: The currency used in the country shaped like a boot is",
)
TOP_K = 5


def rss_mb() -> float:
    if resource is None:
        return float("nan")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def topk_tokens(logits: torch.Tensor, tok, k: int = TOP_K) -> list[str]:
    ids = logits.topk(k).indices.tolist()
    return [tok.decode([t]) for t in ids]


def main() -> None:
    t0 = time.perf_counter()
    print("=== jlens CPU demo ===")
    print(f"python torch={torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"transformers={transformers.__version__}")
    print(f"jlens={jlens.__file__}")
    print(f"model_id={MODEL_ID}")
    print(f"lens=JacobianLens.load({LENS_PATH!r})")
    print(f"rss_start_mb={rss_mb():.1f}")
    print()

    if not os.path.isfile(LENS_PATH):
        raise SystemExit(
            f"Missing lens at {LENS_PATH}. Run scripts/fetch_lens.sh first."
        )

    print("Loading model (CPU, float32)...")
    t_load = time.perf_counter()
    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID)
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = jlens.from_hf(hf, tok)
    print(f"  model n_layers={model.n_layers} d_model={model.d_model}")
    print(f"  load_s={time.perf_counter() - t_load:.1f} rss_mb={rss_mb():.1f}")
    print()

    print("Loading pre-fitted lens...")
    t_lens = time.perf_counter()
    lens = jlens.JacobianLens.load(LENS_PATH)
    print(f"  {lens}")
    print(f"  lens_load_s={time.perf_counter() - t_lens:.1f} rss_mb={rss_mb():.1f}")
    print()

    layers = [
        model.n_layers // 4,
        model.n_layers // 2,
        (model.n_layers * 3) // 4,
        model.n_layers - 2,
    ]
    layers = [L for L in layers if L in lens.jacobians]
    if not layers:
        layers = lens.source_layers[:: max(1, len(lens.source_layers) // 4)][:4]

    print(f"prompt={PROMPT!r}")
    print(f"layers={layers} positions=[-2]")
    print()

    t_apply = time.perf_counter()
    lens_logits, model_logits, _ = lens.apply(
        model, PROMPT, layers=layers, positions=[-2]
    )
    apply_s = time.perf_counter() - t_apply
    print(f"apply_s={apply_s:.2f} rss_mb={rss_mb():.1f}")
    print()

    print("--- top-k from lens.apply (Jacobian lens) ---")
    for layer in layers:
        toks = topk_tokens(lens_logits[layer][0], tok)
        print(f"L{layer:>3} J-lens top-{TOP_K}: {toks}")

    print("--- model final logits top-k ---")
    print(f"model top-{TOP_K}: {topk_tokens(model_logits[0], tok)}")
    print()
    print(f"TOTAL_s={time.perf_counter() - t0:.1f} peak_rss_mb={rss_mb():.1f}")
    print("SUCCESS: lens.apply returned real top-k tokens")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
