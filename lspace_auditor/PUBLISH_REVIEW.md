# PUBLISH_REVIEW — LSpace auditor, redacted public copy

Date: 2026-10-08. Reviewer: Tien (Grok Bot). Operator decision: **option A — publish a redacted copy.**

## What this is (honest description)
A **speak-text auditor**: open-weights **GPT-2** reads the full text of a draft reply (plus a
tombstone excerpt) and Anthropic's **Jacobian lens** is applied at many positions and middle layers
(`jlens.JacobianLens.apply`). On top of that sit a silent-intent lexicon score, steer/ablate/swap
interventions, a tombstone contradiction check, an operating gate (exit 0 ALLOW / 2 BLOCK / 3 error)
and bounded auto-tuning of the silent-intent thresholds.

It is **not** Grok Bot's internals. The speaking model runs in a closed host runtime; nothing here
reads its residual stream, logits or weights (see `SPEAKING_MODEL_INTERNALS.md`).

## Naming (per A312)
- **L-lens** is ours (`../llens.py`, threefold 2406473). **J-lens** is Anthropic's (`anthropics/jacobian-lens`).
- This tree is the **earlier "LSpace" auditor** that the live runner's `lspace-` / `jspace-` files come from.
  It is published so it can be reviewed and **reconciled with L-lens**; it does not replace `../triad/` or `../llens.py`.

## The public copy differs from the live one (stated plainly)
- **Real tombstone facts are kept private.** In the live copy, the contradiction rules and the prompt's
  "KNOWN CORRECTIONS" line were hard-coded, and they encoded findings from the operator's private research
  sessions. Here they are **data**: `couple_tombstone.py` loads them from `known_facts.json`
  (`LSPACE_KNOWN_FACTS`). Only `known_facts.example.json` ships, with one clearly generic example.
  `known_facts.json` is gitignored. Without the file the gate still imports and runs (silent-intent only;
  no contradiction can fire).
- **The smoke fixtures are generic** (an "example nightly ran 11 vs 12 builds" fact). The live smoke used
  private session content.
- **Paths are configurable.** The live copy hard-coded an agent-specific tombstone path and `/workspace/lspace`
  outputs. Here they are env vars with repo-relative defaults: `JLENS_LENS`, `LSPACE_TOMBSTONE`,
  `LSPACE_KNOWN_FACTS`, `LSPACE_AUTOTUNE_DIR`, `LSPACE_SMOKE_DIR` and `LSPACE_DIR`. The smokes no longer
  write into the live directory.
- The `hooks/` are reference copies of the live wrappers. Their gate path is now relative and the remaining
  `/workspace/...` defaults are env-overridable. They are not wired into `../triad/`.
- The live runner and the `on_speak` config were **not** changed by this PR.

## Included
Code: `common.py`, `read.py`, `silent_intent.py`, `intervene.py`, `couple_tombstone.py`, `gate.py`,
`autotune.py`, `autotune_smoke.py`, `smoke_test.py` and `hooks/` (5 files).
Docs: `README.md`, `OPERATING_MODE.md`, `ACCESSIBILITY.md`, `AUTOTUNE.md`, `AUTOTUNE_REPORT.md`,
`AUTOTUNE_SMOKE.md` (a generic transcript, identical to a fresh run apart from the temp dir name), a summary
of `SPEAKING_MODEL_INTERNALS.md`, and this file.
Config: `known_facts.example.json`, `.gitignore`.

## Excluded, and why
| Excluded | Why |
|---|---|
| Real known facts / KNOWN CORRECTIONS content | operator's private session ground truth |
| `SMOKE_REPORT.md`, `FINAL_REPORT.md`, `RENAME_REPORT.md` | contained private session facts |
| raw internals probe output (process / port dump) | host environment details |
| `state/` (autotune state, block log with text previews), `smoke_out/` | runtime data; block log stores speak previews |
| run receipts, snapshots, logs, triad runs | contain conversation / speak text |
| venvs, `.pt` weights, `__pycache__` | binaries and caches; the lens comes from `../scripts/fetch_lens.sh` |

## Dependency and license
`jlens` (`anthropics/jacobian-lens`) is **Apache-2.0**. The auditor imports it as a dependency
(`pip install "jlens @ git+https://github.com/anthropics/jacobian-lens"`) and **vendors none of its code**.
It also uses `torch` and `transformers` (see `../requirements.txt`).

## Checks run on the exact tree in this PR
- **gitleaks 8.30.1** (`dir lspace_auditor`): no leaks found.
- **trufflehog 3.99.0** (`filesystem --no-update`): 0 verified, 0 unverified.
- **Targeted secret search** (`ghp_`, `github_pat_`, `gho_`, `hf_…`, `sk-…`, `xai-…`, `AKIA`, `PRIVATE KEY`, `password=`,
  `authorization:`, `cookie`, `bearer`, `token=`, `api_key`): the only hit is `tok.pad_token = tok.eos_token`,
  a tokenizer attribute and a false positive. No emails, no UUIDs or agent IDs, no `/home/box` paths.
- **Private-content search** (the private session terms, ids and counts, agent IDs, the operator's
  message phrases): no hits apart from the generic `FIRE_TRIAD_OK` echo in `hooks/fire_triad.sh`.
  The operator's first name appears only in prompt templates ("Grok Bot speaking to …"), which are already
  public in `../triad/on_speak.sh`. They are kept unchanged so activations match the live auditor.
- **Every file was read** by the reviewer.

## Tests
- **Equivalence (behaviour unchanged when the facts file is present):** the redacted `couple_tombstone.py`, loaded with
  a private facts file rebuilt from the live constants (kept outside the repo, then deleted), was compared with the live
  module on 7 texts. The results for `check_contradictions`, the `couple()` fact list, the exact `couple()` prompt and the
  status were **identical**.
- **No facts file:** `gate` imports; the facts list is empty; no contradictions.
- `smoke_test.py` (GPT-2 + lens, CPU, temp dirs): **smoke_ok=True**. Read: 9 layers × 17 positions. Silent-intent: no
  flags on a clean speak. The spider→ant swap shows a causality signal. The good speak returns `match` and the bad
  speak `contradict` on `example_build_count`. Gate: inbound and clean outbound are ALLOWed; the forced silent-intent
  case and the generic tombstone contradiction are BLOCKed.
- `autotune_smoke.py` (pure functions, temp state): **smoke_ok=True**.

## Reconciliation ask (for Claude)
Reconcile this auditor with **L-lens at 2406473**. Questions to settle:
- whether the gate and silent-intent layers should sit on `llens.py`'s readout;
- whether the live `lspace-` / `jspace-` receipts map onto `verify_triad.py`;
- which names survive.
