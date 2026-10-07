# threefold

**Tombstone ∥ Covenant ∥ JLens** — a mandatory three-leg memory/observability system for agent runs.

Every completed task must fire **all three** legs under one shared `task_id`. There is no degrade path: if any leg fails, the triad fails.

| Leg | What it is | Artifact |
|-----|------------|----------|
| **Tombstone** | Append-only public-facing run log | `tombstone/tombstone.md` |
| **Covenant** | Private structured AI memories | `AI_MEMORY_ROOT` (default `./data/ai_memory_root`, gitignored) |
| **JLens** | Real Anthropic `jlens.JacobianLens.apply` snapshot | `runs/<task_id>/jlens_snapshot.json` |

Self-observation is **not** a triad leg.

## Architecture

```
                    task_id (shared)
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
     tombstone.md    covenant puts    jlens snapshot
     (append.sh)   (put_covenant_*)   (jlens_snapshot.py)
          │               │               │
          └───────────────┴───────────────┘
                          │
                   fire_triad.sh
```

`fire_triad.sh` runs **JLens first** so the digest can be stamped into tombstone notes and covenant bodies, then proves the same `task_id` appears in all three stores.

## CPU notes

- Model: `openai-community/gpt2` (~124M, 12 layers). Fits comfortably on CPU (~1GB peak RSS for the demo).
- **No GPU required.** float32 CPU only.
- Lens: pretrained gpt2-small weights from Hugging Face `neuronpedia/jacobian-lens`
  (`gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt`), fetched by `scripts/fetch_lens.sh`.
  The ~12MB `.pt` file is **not** shipped in this repo (size / GitHub limits).
- Prefer Python **3.12**. Install **CPU** torch first.

## Setup

```bash
git clone https://github.com/LAWLESS1987/threefold
cd triad

python3.12 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
pip install -r requirements.txt
pip install "jlens @ git+https://github.com/anthropics/jacobian-lens"

# Covenant (private memory CLI) — do not vendor the whole tree into git history
git clone https://github.com/LAWLESS1987/covenant vendor/covenant
# or: export COVENANT_SRC=/path/to/covenant

./scripts/fetch_lens.sh
./scripts/seed_demo.sh          # MODE=apply (default): proves SUCCESS via demo_apply.py
# MODE=triad ./scripts/seed_demo.sh   # full three-leg fire with fixed TASK_ID
```

Private memories default to `./data/ai_memory_root` (gitignored). Override with `AI_MEMORY_ROOT`.

## How seed works

`scripts/seed_demo.sh` uses fixed demo values:

- `TASK_ID=20261007-020907-13423`
- `PROMPT="Fact: The currency used in the country shaped like a boot is"`

Default `MODE=apply` fetches the lens (if missing) and runs `demo_apply.py`, expecting the line:

`SUCCESS: lens.apply returned real top-k tokens`

`MODE=triad` additionally clones covenant if needed and runs `triad/fire_triad.sh`, expecting `FIRE_TRIAD_OK`.

The sample entry in `tombstone/tombstone.md` is illustrative of that same `task_id` format only — no private history.

## Layout

```
triad/
  README.md
  requirements.txt
  .gitignore
  jlens_snapshot.py
  demo_apply.py
  tombstone/
    append.sh
    tombstone.md          # SAMPLE only
  triad/
    fire_triad.sh
    put_covenant_memory.sh
  scripts/
    fetch_lens.sh
    seed_demo.sh
```

## License / upstream

- Jacobian lens library: [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)
- Pretrained lens weights: [neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens)
- Covenant memory CLI: [LAWLESS1987/covenant](https://github.com/LAWLESS1987/covenant)

## Speak-turn hook

Every Grok Bot user-visible reply is a triad task. Run:

```bash
./triad/on_speak.sh "exact words spoken" "short-title"
```

This calls `fire_triad.sh` (all three legs required).
