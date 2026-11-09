"""Shared loader for the speak-text LSpace auditor (open GPT-2 + fitted lens)."""
from __future__ import annotations

import os
from functools import lru_cache

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import jlens
import torch
import transformers

MODEL_ID = "openai-community/gpt2"
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
# Fitted GPT-2 Jacobian lens (anthropics/jacobian-lens format). Override with JLENS_LENS.
DEFAULT_LENS = os.environ.get("JLENS_LENS", os.path.join(_REPO, "gpt2_jacobian_lens.pt"))
MAX_SEQ_LEN = 1024  # GPT-2 context; last-window chunking for longer text
MIDDLE_LAYERS = list(range(2, 11))  # dense middle band in fitted [0..10]
SCREEN_LEXICON = (
    "fake",
    "fraud",
    "secretly",
    "hoax",
    "lie",
    "error",
    "wrong",
    "omit",
    "missing",
)
# Deprecated name, kept until the live runner moves (threefold PR #1 review, 2026-10-08).
SILENT_LEXICON = SCREEN_LEXICON
# Tombstone log to cross-check against. Override with LSPACE_TOMBSTONE.
TOMBSTONE_DEFAULT = os.environ.get(
    "LSPACE_TOMBSTONE", os.path.join(_REPO, "tombstone", "tombstone.md")
)


@lru_cache(maxsize=1)
def load_model_and_lens(lens_path: str = DEFAULT_LENS, model_id: str = MODEL_ID):
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_id)
    hf.eval()
    for p in hf.parameters():
        p.requires_grad_(False)
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(lens_path)
    return model, lens, tok, hf


def last_window_text(text: str, tok, max_seq_len: int = MAX_SEQ_LEN) -> tuple[str, int, int]:
    """Prefer the last window that fits GPT-2 context (full speak retained when short)."""
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= max_seq_len:
        return text, 0, len(ids)
    window = ids[-max_seq_len:]
    return tok.decode(window), len(ids) - max_seq_len, len(ids)


def select_positions(seq_len: int, stride: int = 4, tail: int = 16) -> list[int]:
    """Many positions: every `stride`-th token plus last `tail`."""
    if seq_len <= 0:
        return []
    pos = set(range(0, seq_len, max(1, stride)))
    for i in range(max(0, seq_len - tail), seq_len):
        pos.add(i)
    return sorted(pos)


def topk_tokens(logits: torch.Tensor, tok, k: int = 5) -> list[str]:
    ids = logits.topk(k).indices.tolist()
    return [tok.decode([t]).replace("\n", "\\n") for t in ids]


def token_id(tok, word: str) -> int | None:
    """Best-effort single-token id for a lexicon word (leading-space variants)."""
    for cand in (word, " " + word, word.capitalize(), " " + word.capitalize()):
        ids = tok.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return None


def concept_direction(hf, tok, word: str) -> torch.Tensor:
    """Unembedding row for `word` as a residual-space concept direction (unit)."""
    tid = token_id(tok, word)
    if tid is None:
        ids = tok.encode(" " + word, add_special_tokens=False)
        tid = ids[0]
    # GPT-2: lm_head.weight is [vocab, d_model]; tied to wte often
    W = hf.lm_head.weight.detach().float()  # [V, d]
    v = W[tid].clone()
    n = v.norm().clamp_min(1e-8)
    return v / n
