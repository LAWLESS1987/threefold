# LSpace Accessibility — what is and is not available

## NOT accessible (do not claim otherwise)
- **Grok Bot / conversation-model residual stream**: closed runtime; no hooks into the chat model.
- Anthropic Claude-class J-space on the speaking model itself.
- Full activation tensors persisted for the chat model; concept swaps inside Grok Bot.

## IS accessible
- Full speak text passed to `on_speak.sh` / SendToUser (no 480-char truncate on the LSpace path).
- Local covenant `tombstone.md` (ground truth of what happened — text/process log).
- Covenant `AI_MEMORY_ROOT` (threefold-memory).
- Open-weights **GPT-2** + pretrained `gpt2_jacobian_lens.pt` via `anthropics/jacobian-lens`
  (installed into the auditor's venv).
- `ActivationRecorder` / forward hooks for steer / ablate / swap on that open model.

## What this build IS
A **speak-text LSpace auditor**: multi-position Jacobian-lens reads of an open model
processing the **full speak draft** (+ optional tombstone excerpt), with a lexicon-association
screen (HOLD-and-record), write-ops demos, and an operating **gate** that blocks a speak only on a tombstone contradiction.

Honest name: **speak-text LSpace auditor** — **not** Grok Bot’s own J-space.

## Naming rule (hard)
- Call this system **LSpace** / `lspace_*` / type:`lspace`.
- The old GPT-2 top-k triad snapshot (`jlens_snapshot.py`, L3/6/9/10, pos −2, ≤480-char; on threefold main since
  2406473 it is the full capture L-lens is built on)
  is **DEMOTTED proxy** — **never call it LSpace**. Prefer `proxy_jlens`.
