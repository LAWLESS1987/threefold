# Speaking-model internals probe (summary)

**Date:** 2026-10-07. The raw probe output is not published (host environment details).

## (1) What was probed

- Process list for local inference servers (vLLM, TGI, ollama, llama.cpp, etc.)
- Listening ports for model servers
- Local disk for Grok / xAI / safetensors / pytorch weights
- Env vars related to MODEL / LOGIT / LOGPROB / HIDDEN / RESIDUAL / INFERENCE
- CUDA availability in the auditor's torch venv
- Local reference docs for activation/logprob APIs

## (2) What was found

| Signal | Result |
|--------|--------|
| Grok Bot residual stream / hidden states | **Not exposed** |
| Speaking-model logits / logprobs API | **Not exposed** |
| Local Grok / xAI weights | **None on disk** |
| Local inference server for chat model | **None** |
| CUDA | **False** (CPU torch only) |
| Open activations on this box | GPT-2 + pretrained `gpt2_jacobian_lens.pt` only |

**Conclusion:** The conversation / speaking model runs in a **closed host runtime** outside the agent box. The box receives text (user messages / drafts); it cannot hook the model that authored the reply.

## (3) Closest faithful upgrade (what we wire as primary)

1. **Primary LSpace leg (implemented):** speak-text auditor — open-weights GPT-2 + real `JacobianLens.apply`, multi-position, full untruncated speak, silent-intent lexicon, steer/ablate/swap demos, tombstone couple, operating gate. Honest label: `speak-text-lspace-auditor` / `not_grok_bot_residuals: true`.
2. **Not available without host change:** true Anthropic-style J-space on Grok Bot would need the host to expose residual hooks or at least per-token logprobs for the speaking forward pass, plus a fitted Jacobian lens for that model.
3. **Intermediate upgrade path (if host adds API):** prefer **logprobs on the speaking completion** as a weak surface proxy (still not J-space); then residual hooks + fitted J-lens as the real close.

## (4) Gate / proxy policy

- Gate uses **only** the LSpace auditor + tombstone cross-check.
- GPT-2 **proxy_jlens** (old ≤480 / pos −2 / L3,6,9,10 top-5) is **demoted**, optional continuity only, **never** the gate, **never** called LSpace.
