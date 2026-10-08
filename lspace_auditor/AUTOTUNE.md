# LSpace silent-intent auto-tune

Continuous calibration of outbound silent-intent thresholds from observed
false-positive rate while the gate is on. **No manual step** in the live path.

Instrument: **speak-text LSpace auditor** (open GPT-2 + JacobianLens).
Not Grok Bot residuals. Demoted proxy_jlens is not the gate.

## Semantics (rank_thresh / min_hits)

`score_silent_intent` lights a lexicon word when rank `r < rank_thresh` at
enough `(layer, pos)` hits, and the word is absent from surface text.

| Knob | Higher value | Lower value |
|------|--------------|-------------|
| `rank_thresh` | more sensitive (more blocks / more FPs) | harder to block (fewer FPs) |
| `min_hits` | harder to block (fewer FPs) | more sensitive |

- **FP rate > upper** → harden: decrease `rank_thresh` (prefer) and/or increase `min_hits`
- **FP rate < lower** (enough samples) → loosen toward baseline: increase `rank_thresh` and/or decrease `min_hits`
- Max **one step per BLOCK** event; never cross safety floors/ceilings

## Initial values

### Main (outbound baseline)
- `rank_thresh=50`, `min_hits=3`

### Meta (status / self-referential speaks)
- `rank_thresh=35`, `min_hits=5` (harder to false-block gate status reports)

### Inbound
- Fixed constants `INBOUND_RANK_THRESH=40`, `INBOUND_MIN_HITS=4` — **not** auto-tuned
  (outbound-only first; keeps inbound probe stable).

### Bounds
| param | value |
|-------|-------|
| window | 20 |
| fp_upper | 0.40 |
| fp_lower | 0.10 |
| min_samples | 5 |
| rank_thresh floor / ceiling | 15 / 80 |
| min_hits floor / ceiling | 2 / 8 |
| step_rank | 5 |
| step_hits | 1 |

## Persistence

Under `lspace_auditor/state/` (gitignored) (override with `LSPACE_AUTOTUNE_DIR` or
`gate_outbound(..., autotune_dir=...)`):

- `autotune_state.json` — current main/meta thresholds + bounds
- `block_log.jsonl` — every silent-intent BLOCK with category + verdict
- `threshold_changes.jsonl` — every threshold step with triggering `fp_rate`

## Verdict heuristic

On silent-intent BLOCK:

1. Explicit `--verdict` / `LSPACE_AUTOTUNE_VERDICT` / receipt override wins
2. Tombstone contradictions present → `true_catch`
3. Forced silent demo (`force_block_silent`) on **main** → `true_catch`
   (so demos do not inflate FP rate)
4. Otherwise silent BLOCK with neutral/no contradictions → `false_positive`
   (matches the verified FP pattern)

## Meta-text category

Self-referential speaks (LSpace/JSpace/triad/tombstone/silent-intent/BLOCK/ALLOW/
flag/threshold/autotune/receipt/status language, or status prose around
`error`/`omit`/`missing`/`wrong`) are classified as **`meta`**.

- Separate rolling window and separate `meta_rank_thresh` / `meta_min_hits`
- Meta BLOCKs do **not** poison the main FP rate
- Meta baselines start stricter for BLOCK (lower rank_thresh, higher min_hits)
- Safety bias unchanged: when uncertain, prefer BLOCK; auto-tune prevents chronic FPs

## Live path

`gate_outbound` loads thresholds for the predicted category → scores silent
intent → on silent-intent BLOCK calls `autotune.record_block` → appends log →
`maybe_adjust` → persists → receipt includes `autotune` snapshot
(`thresholds`, `fp_rate`, `category`, `verdict`, `changed`).

## Files

- `autotune.py` — pure functions + persistence
- `gate.py` — outbound wiring
- `autotune_smoke.py` — temp-dir smoke (no GPT-2)
- `AUTOTUNE_SMOKE.md` / `AUTOTUNE_REPORT.md` — results
