# Speak-text LSpace auditor (public, redacted copy)

Honest name: activations of an **open GPT-2** model reading the **full speak draft**
(+ tombstone context), via Anthropic `JacobianLens` (`anthropics/jacobian-lens`,
Apache-2.0, used as a dependency — no code vendored). **Not** Grok Bot’s residual stream.

This is the earlier **LSpace** auditor. **L-lens** (`../llens.py`) is ours and is built
around Anthropic’s **J-lens**; this tree is to be reconciled with L-lens. See
`PUBLISH_REVIEW.md` in the PR for what differs from the live copy.

## Run

```bash
pip install -r ../requirements.txt   # then: pip install "jlens @ git+https://github.com/anthropics/jacobian-lens"
../scripts/fetch_lens.sh                    # puts gpt2_jacobian_lens.pt at the repo root (the default)
export LSPACE_TOMBSTONE=../tombstone/tombstone.md
export LSPACE_KNOWN_FACTS=./known_facts.json   # optional; copy known_facts.example.json
python read.py --prompt-file speak.txt --out snap.json
python silent_intent.py --prompt-file speak.txt
python intervene.py --op swap --concept-a spider --concept-b ant
python couple_tombstone.py --speak-file speak.txt
python gate.py inbound "Operator message…"
python gate.py outbound "Full speak draft…"
LSPACE_SMOKE_DIR=/tmp/lspace_smoke python smoke_test.py      # generic fixtures
LSPACE_SMOKE_DIR=/tmp/lspace_smoke python autotune_smoke.py  # no GPT-2 needed
```

Gate exit: `0` ALLOW, `2` BLOCK, `3` error.

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `JLENS_LENS` | `<repo>/gpt2_jacobian_lens.pt` | pretrained lens |
| `LSPACE_TOMBSTONE` | `<repo>/tombstone/tombstone.md` | tombstone for coupling |
| `LSPACE_KNOWN_FACTS` | `lspace_auditor/known_facts.json` (gitignored) | operator ground-truth corrections; absent = none |
| `LSPACE_AUTOTUNE_DIR` | `lspace_auditor/state/` (gitignored) | autotune thresholds + logs |
| `LSPACE_SMOKE_DIR` | `lspace_auditor/` | where smoke reports are written |
| `LSPACE_DIR` | `lspace_auditor/` | used by `hooks/lspace_snapshot.py` |

## Known facts

Tombstone contradiction rules are **data**, not code: `known_facts.json`
(schema in `known_facts.example.json`). Real facts are operator ground truth
and stay private. Without the file the gate still runs (silent-intent only).

## Hooks

`hooks/` are reference wrappers from the live runner (inbound/outbound gate, an
`on_speak.sh` that gates before the triad, `fire_triad.sh`). They default to
`/workspace/...` paths with env overrides and do **not** replace `../triad/`.

## Naming
- This tree = **LSpace** auditor (earlier name); **L-lens** = ours; **J-lens** = Anthropic’s.
- `jlens_snapshot.py` (GPT-2 top-k triad snapshot) = **demoted proxy_jlens** — never the gate.
