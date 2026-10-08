# LSpace auto-tune report

Instrument: **speak-text LSpace auditor**. Not Grok Bot residuals. Proxy demoted.

## (a) What was implemented and where

| File | Role |
|------|------|
| `autotune.py` | Pure calibration: `classify_category`, `classify_verdict`, `compute_fp_rate`, `maybe_adjust`, `record_block`, state I/O |
| `gate.py` | Outbound loads autotune thresholds; on silent-intent BLOCK runs `record_block` automatically; receipt `autotune` key; CLI `--autotune-dir` / `--verdict`. Inbound unchanged (fixed constants). |
| `state/autotune_state.json` | Live thresholds (seeded at baselines; gitignored, not published) |
| `state/block_log.jsonl` | Append-only BLOCK log (created on first live BLOCK; not published) |
| `state/threshold_changes.jsonl` | Append-only threshold change log (not published) |
| `autotune_smoke.py` | Temp-dir smoke; does not clobber live state |
| `AUTOTUNE.md` | Design + bounds |
| `AUTOTUNE_SMOKE.md` | Smoke transcript |
| `AUTOTUNE_REPORT.md` | This report |

No LSpace rename. Proxy stays demoted.

## (b) Initial threshold values and bounds

**Main outbound:** `rank_thresh=50`, `min_hits=3`  
**Meta:** `rank_thresh=35`, `min_hits=5`  
**Inbound:** fixed 40 / 4 (not auto-tuned)

**Bounds:** window=20, fp_upper=0.40, fp_lower=0.10, min_samples=5,  
rank floor/ceiling=15/80, hits floor/ceiling=2/8, step_rank=5, step_hits=1,  
max one step per BLOCK.

**Direction:** higher `rank_thresh` = more sensitive; to cut FPs, **lower**
`rank_thresh` and/or **raise** `min_hits`.

## (c) Smoke test — simulated FP streak raises the bar

`python autotune_smoke.py` → **smoke_ok=True**

- Seeded 10 main `false_positive` silent-intent BLOCKs in a temp state dir.
- After min_samples=5, each further BLOCK stepped `harden_rank`:  
  **rank_thresh 50 → 20** (min_hits stayed 3). Fewer FPs = harder to light.
- `threshold_changes.jsonl` logged each step with `fp_rate=1.000` and old→new.
- Meta thresholds stayed at 35/5 during the main streak.

## (d) Meta-text category handling

- `is_meta_text` / `classify_category` detect gate/status self-reference
  (LSpace, triad, tombstone, silent-intent, BLOCK/ALLOW, threshold, autotune,
  receipt, …) and status prose around meta lexicon subset
  `{error, omit, missing, wrong}`.
- Meta BLOCKs use a **separate** rolling window and `meta_*` thresholds.
- Smoke: 8 meta FP BLOCKs hardened meta **35→15** while **main stayed at 20/3**.
- Verdicts on meta silent BLOCKs with no tombstone contradictions default to
  `false_positive` under `category=meta` (do not poison main FP rate).
- Safety bias: uncertain → BLOCK; auto-tune on both categories prevents chronic
  false blocks on status reports without disabling the main security bar.

## Live gate

`gate_outbound` uses `lspace_auditor/state/` by default (`LSPACE_AUTOTUNE_DIR`
override). Forced silent demos classify as `true_catch` on main so they do not
falsely inflate the FP rate. Override via `--verdict` or env
`LSPACE_AUTOTUNE_VERDICT`.
