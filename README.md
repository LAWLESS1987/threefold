# threefold

**Tombstone ∥ Covenant ∥ L-lens**: an experimental AI observability and provenance apparatus.

Threefold links three things under one task identity: mechanistic measurements, persistent structured memory,
and an append-only execution record. **L-lens** is the Threefold observability layer built around upstream
**Jacobian Lens (JLens)** measurements. It is not a replacement name for the Jacobian Lens. JLens is Anthropic's
instrument ([anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens), Apache-2.0), and Threefold
runs it unmodified.

```
Anthropic Jacobian Lens          L-lens (ours)                         Threefold linkage
jlens.JacobianLens.apply   →    llens.py: hashes the capture,     →   one task_id, three legs:
(run by jlens_snapshot.py;      derives summaries from the raw         tombstone entry  (cites both digests)
 its outputs stored as raw)     arrays, packages and links them        covenant memories task-/jlens-/llens-
                                                                       verified by triad/verify_triad.py
```

Every completed task fires **all three** legs under one shared `task_id`. There is no degrade path. If any leg
fails, or the legs do not agree, the triad fails.

| Leg | What it is | Artifacts |
|-----|------------|-----------|
| **Tombstone** | Append-only run log | `tombstone/tombstone.md` (or `$TOMBSTONE_LOG`); the notes carry `task_id`, `jlens_capture_sha256`, `llens_sha256` |
| **Covenant** | Structured AI memories, written by [covenant](https://github.com/LAWLESS1987/covenant)'s `ai_memory_system` | `task-<id>` (both digests), `jlens-<id>` (the JLens capture), `llens-<id>` (the L-lens artifact) under `AI_MEMORY_ROOT` |
| **Lens** | Upstream JLens measurements (`jlens_snapshot.py`), and the L-lens layer around them (`llens.py`) | `runs/<id>/jlens_snapshot.json`, `jlens_raw.npz`, `jlens_vocab.json`, `llens.json` |

Self-observation is **not** a triad leg.

## The Jacobian Lens: what it measures, and what it does not

These are upstream's terms. The lens linearly transports a residual-stream vector at layer `l` and one token
position into the final-layer basis, using the average input-output Jacobian `J_l = E[∂h_final/∂h_l]` fitted on a
text corpus. It then decodes the result with the model's own unembedding into logits over the vocabulary. A
readout says which tokens that residual is disposed to make the model predict.

It is a mechanistic probing technique that produces layer-level predictive information about **one subject
model's** internal states. Threefold makes no claim that it reads thoughts, beliefs or intent, detects
consciousness, or establishes semantic ground truth.

**The subject is not the agent.** By default the subject model is GPT-2 small (`openai-community/gpt2`), reading
the text of a task (for a speak-turn, the words that were spoken). The lens observes GPT-2's residual stream
while it reads that text. It does not observe whatever agent wrote the text. Every capture says so in
`subject_model.note`.

## What a capture records (`jlens_snapshot.py`)

The defaults follow upstream's own API. `JacobianLens.apply` reads **every fitted layer** when `layers=None`
and **every token position** when `positions=None`. Upstream's slice view (`jlens.vis.compute_slice`) keeps the
top 10 tokens per (position, layer) cell. The capture does the same for:

- the Jacobian-lens logits (`use_jacobian` on);
- upstream's vanilla **logit-lens baseline** (`use_jacobian` off) at the same cells, so the effect of the
  Jacobian transport can be compared against plain unembedding;
- the model's own final-layer logits at the same positions.

For the bundled GPT-2 lens that is all 11 fitted layers (0–10; layer 11 is the model's output) at every
position. The earlier version of this script read 4 layers at position −2. Those cells are a subset of the new
readout and give the same tokens.

| Field | Where |
|-------|-------|
| task_id | `task_id` |
| model and version | `subject_model.id`, `subject_model.revision` (Hub commit), `n_layers`, `d_model`, `vocab_size`, dtype, device |
| lens and its source | `lens.source`, `lens.sha256` (of the .pt file), `lens.n_prompts`, `lens.source_layers` |
| upstream library | `upstream.source`, `upstream.version`, `upstream.vcs_commit`, `upstream.call` |
| prompt / input | `input.prompt`, `input.prompt_sha256`, `input.input_ids`, `input.n_tokens`, `input.starts_with_bos` |
| layers and positions | `readout.layers`, `readout.positions`, `readout.top_k`, `readout.baseline` |
| raw JLens outputs | `jlens_raw.npz`: `jacobian_top_ids/logits`, `baseline_top_ids/logits` `[layer, position, K]`, `final_top_ids/logits` `[position, K]`, `input_ids`, `layers`, `positions` |
| timestamps, runtime | `timing.started_utc`, `model_load_s`, `apply_jacobian_s`, `apply_baseline_s`, `total_s` |
| environment | `env`: python, torch, transformers, numpy, platform, thread count, seed |
| digests | `llens_capture_time.array_sha256` (every raw array), `full_tensor_sha256` (every complete `[positions, vocab]` tensor), `vocab_sha256` |

**Raw versus derived.** The `.npz` holds only values copied out of the tensors upstream returned. The one
operation applied is `torch.topk` selecting each cell's top K. Hashes are L-lens's work. Where they must be
taken at capture time (the full tensors are too large to keep, about 24 MB per layer at 120 tokens), they sit in
the separate `llens_capture_time` section, never under raw. Decoded token strings are a tokenizer view of the
ids, not a measurement. They live in `jlens_vocab.json` (hashed), and `python llens.py show` prints them.

## L-lens (`llens.py`)

L-lens consumes a capture and writes `llens.json`:

- `inputs.jlens_capture_sha256`: the canonical-JSON sha256 of the capture it was built from, plus the digest
  of every raw array;
- `derived`: integer summaries computed **only** from the stored raw arrays, so they can be recomputed
  exactly. For the Jacobian lens and the baseline, each layer gets `top1_agree`, `final_top1_in_topk` and
  `topk_overlap` (each against the model's own final top-K), and each position gets
  `first_layer_final_top1_in_topk`. The definitions are written into every artifact;
- `upstream_credit`, `subject`, `links` (task_id and the names of the tombstone/covenant records), `what` (the
  statement of what L-lens is and is not).

The derived numbers are observability statistics about one model reading one text. `first_layer_…` is limited
to the stored top-K: a token ranked K+1 counts as absent.

## Verification

`python llens.py verify --capture … --raw … --llens …` checks the lens leg on its own: the array digests,
the vocabulary digest, shapes against the declared readout, that the artifact cites this capture, and that
`derived` recomputes exactly.

`python triad/verify_triad.py --task-id ID` checks the whole triad: the lens leg as above, a tombstone entry
whose notes carry this task_id **and** both digests, and covenant memories `task-/jlens-/llens-<id>` whose
contents give the same task_id and the same canonical digests. `fire_triad.sh` runs it after every triad. Any
FAIL exits 1.

**What verification cannot see.** It checks that the records agree with each other and with the stored arrays.
It does not re-run the model. To check that the arrays are what upstream returns, re-run with the recorded
configuration and compare `full_tensor_sha256`; `tests/check_against_upstream.py` does that. Bit equality
across different CPUs or BLAS builds is not guaranteed. Covenant's own hash chain is checked by covenant
(`verify_chain`), not here. Verification runs after the legs are written. A failure marks the triad failed, and
it does not unwrite append-only records.

## Tests

```bash
python -m pytest tests/test_llens.py            # no model needed: derivation, hashing, every link broken on purpose
python tests/check_against_upstream.py runs/ID  # model + lens: the stored capture vs a direct upstream apply call
COVENANT_SRC=… tests/e2e_triad.sh               # model + lens + covenant: a real speak-turn in a scratch dir, then broken 3 ways
```

Results when this was committed (2026-10-08; Windows 11, CPU, Python 3.12.10, torch 2.14.1+cpu,
transformers 5.19.0, jlens 0.1.0 at 581d398):

- `tests/test_llens.py`: 26 passed.
- `check_against_upstream.py` on the seed prompt: 70 of 70 comparisons equal. The stored top-K ids and
  logits, and every full-tensor digest, match a direct upstream call that reads all layers at once.
- `e2e_triad.sh`: a speak-turn verified with 44 checks. A wrong tombstone digest and a missing L-lens memory
  each failed with that named reason, and restoring each restored green. A broken lens file failed the triad
  before anything was appended.
- `scripts/seed_demo.sh`: `MODE=apply` and `MODE=triad` both OK (scratch tombstone and memory root).
- A 480-character speak-turn (106 positions): 12 s total, raw arrays 135 KB, memory bodies 10 KB (capture) and
  9 KB (L-lens).
- Not exercised: a Linux run, `scripts/push_to_github.sh`, and pushing to the live `threefold-memory`
  repository (the tests use a scratch memory root that is not a git repo).

## Known limits

- One subject model (GPT-2 small) and one pretrained lens (fitted on 277 wikitext prompts; hosted by Neuronpedia). The
  readout is only as good as that lens. Upstream's paper uses lenses fitted on ~1000 prompts.
- Only the top K of each cell is stored. Full tensors are hashed, not kept.
- Covenant's memory gate is a bag-of-words judge. It refused a capture whose text contained upstream's literal
  parameter value for the baseline, and another that listed decoded vocabulary. So records say "use_jacobian
  off", and decoded strings stay out of memory bodies. A spoken text the gate refuses fails the triad, closed.
- The live runner that writes `lspace-`/`jspace-` files to threefold-memory runs code that is not in this
  repository. This repository describes and tests only what is in it.

## CPU notes

- Model: `openai-community/gpt2` (~124M, 12 layers). Fits comfortably on CPU.
- **No GPU required.** float32 CPU only.
- Lens: pretrained gpt2-small weights from Hugging Face `neuronpedia/jacobian-lens`
  (`gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt`), fetched by `scripts/fetch_lens.sh`.
  The 13 MB `.pt` file is **not** shipped in this repo (size / GitHub limits).
- Prefer Python **3.12**. Install **CPU** torch first.

## Setup

```bash
git clone https://github.com/LAWLESS1987/threefold
cd threefold

python3.12 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
pip install -r requirements.txt
pip install "jlens @ git+https://github.com/anthropics/jacobian-lens"

# Covenant (memory CLI) — do not vendor the whole tree into git history
git clone https://github.com/LAWLESS1987/covenant vendor/covenant
# or: export COVENANT_SRC=/path/to/covenant

./scripts/fetch_lens.sh         # PYTHON=… to choose the interpreter (default python3)
./scripts/seed_demo.sh          # MODE=apply (default): proves SUCCESS via demo_apply.py
# MODE=triad ./scripts/seed_demo.sh   # full three-leg fire with fixed TASK_ID
```

On Windows (Git Bash), the scripts also find a venv's `Scripts/python.exe`. Paths written inside heredocs are
not translated for Windows Python, so give `fetch_lens.sh` a Windows-style `OUT=C:/…` there.

Memories default to `./data/ai_memory_root` (gitignored). Override with `AI_MEMORY_ROOT`, the tombstone log
with `TOMBSTONE_LOG`, and the run directory with `RUN_DIR`.

## How seed works

`scripts/seed_demo.sh` uses fixed demo values:

- `TASK_ID=20261007-020907-13423`
- `PROMPT="Fact: The currency used in the country shaped like a boot is"`

Default `MODE=apply` fetches the lens (if missing) and runs `demo_apply.py`, expecting the line:

`SUCCESS: lens.apply returned real top-k tokens`

`MODE=triad` additionally clones covenant if needed and runs `triad/fire_triad.sh`, expecting `FIRE_TRIAD_OK`.

The sample entry in `tombstone/tombstone.md` is illustrative of that same `task_id` format only. It predates
L-lens, so it carries the older `jlens_digest` note and no `llens_sha256`.

## Layout

```
threefold/
  README.md
  requirements.txt
  jlens_snapshot.py        # JLens capture: runs upstream apply, stores what it returned
  llens.py                 # L-lens: hashing, derived summaries, packaging, verify, show
  demo_apply.py
  tombstone/
    append.sh
    tombstone.md           # SAMPLE only
  triad/
    fire_triad.sh          # the three legs, then verify_triad.py
    on_speak.sh
    put_covenant_memory.sh
    put_memory_file.py     # covenant MemoryStore.put with the body read from a file
    verify_triad.py        # cross-leg verification, fails closed
  scripts/
    fetch_lens.sh
    seed_demo.sh
  tests/
    test_llens.py
    check_against_upstream.py
    e2e_triad.sh
```

## License / upstream

- **Jacobian Lens** (the instrument): [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens),
  © Anthropic PBC, Apache-2.0. It is installed as a dependency, not copied into this repository. Paper:
  [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html).
- Pretrained lens weights: [neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens)
- Subject model: [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)
- Covenant memory CLI: [LAWLESS1987/covenant](https://github.com/LAWLESS1987/covenant)

## Speak-turn hook

Every Grok Bot user-visible reply is a triad task. Run:

```bash
./triad/on_speak.sh "exact words spoken" "short-title"
```

This calls `fire_triad.sh` (all three legs required, then verified).

## Public memory root

Live data: https://github.com/LAWLESS1987/threefold-memory

```bash
git clone https://github.com/LAWLESS1987/threefold-memory
export AI_MEMORY_ROOT=$PWD/threefold-memory
```
