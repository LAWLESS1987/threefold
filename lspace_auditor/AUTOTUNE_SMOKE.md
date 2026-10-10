# LSpace autotune smoke

Instrument: **speak-text LSpace auditor** auto-tune (pure functions).
**Not** Grok Bot residuals. Temp state dir only.

## Initial thresholds
- main: rank_thresh=50 min_hits=3
- meta: rank_thresh=35 min_hits=5
- bounds: {"window": 20, "fp_upper": 0.4, "fp_lower": 0.1, "min_samples": 5, "rank_floor": 15, "rank_ceiling": 80, "hits_floor": 2, "hits_ceiling": 8, "step_rank": 5, "step_hits": 1}

## Simulated main false-positive streak
- block 1: fp_rate=1.000 changed=False action=none rt=50 mh=3
- block 2: fp_rate=1.000 changed=False action=none rt=50 mh=3
- block 3: fp_rate=1.000 changed=False action=none rt=50 mh=3
- block 4: fp_rate=1.000 changed=False action=none rt=50 mh=3
- block 5: fp_rate=1.000 changed=True action=harden_rank rt=45 mh=3
- block 6: fp_rate=1.000 changed=True action=harden_rank rt=40 mh=3
- block 7: fp_rate=1.000 changed=True action=harden_rank rt=35 mh=3
- block 8: fp_rate=1.000 changed=True action=harden_rank rt=30 mh=3
- block 9: fp_rate=1.000 changed=True action=harden_rank rt=25 mh=3
- block 10: fp_rate=1.000 changed=True action=harden_rank rt=20 mh=3

- after streak: main rt=20 mh=3 (was 50/3)
- meta unchanged check: rt=35 mh=5 (was 35/5)
- PASS: main thresholds moved toward fewer FPs (rank_thresh 50->20, min_hits 3->3)
- PASS: meta thresholds untouched by main streak
- threshold_changes rows=6
  - harden_rank fp_rate=1.000 {'rank_thresh': 50, 'min_hits': 3} -> {'rank_thresh': 45, 'min_hits': 3} reason=fp_rate 1.000 > upper 0.4
  - harden_rank fp_rate=1.000 {'rank_thresh': 45, 'min_hits': 3} -> {'rank_thresh': 40, 'min_hits': 3} reason=fp_rate 1.000 > upper 0.4
  - harden_rank fp_rate=1.000 {'rank_thresh': 40, 'min_hits': 3} -> {'rank_thresh': 35, 'min_hits': 3} reason=fp_rate 1.000 > upper 0.4
  - harden_rank fp_rate=1.000 {'rank_thresh': 35, 'min_hits': 3} -> {'rank_thresh': 30, 'min_hits': 3} reason=fp_rate 1.000 > upper 0.4
  - harden_rank fp_rate=1.000 {'rank_thresh': 30, 'min_hits': 3} -> {'rank_thresh': 25, 'min_hits': 3} reason=fp_rate 1.000 > upper 0.4
  - harden_rank fp_rate=1.000 {'rank_thresh': 25, 'min_hits': 3} -> {'rank_thresh': 20, 'min_hits': 3} reason=fp_rate 1.000 > upper 0.4
- PASS: threshold_changes log includes triggering fp_rate

## Meta-text BLOCKs isolated from main
- main before=(20, 3) after=(20, 3)
- meta before=(35, 5) after=(15, 5)
- PASS: main thresholds unchanged by meta streak
- PASS: meta thresholds hardened independently (rt 35->15, mh 5->5)

## Category / verdict helpers
- PASS: is_meta_text + classify_verdict heuristics
- PASS: compute_fp_rate example={'n_decided': 5, 'n_fp': 4, 'n_true': 1, 'fp_rate': 0.8, 'n_pending': 0, 'n_raw': 5}
- PASS: maybe_adjust harden_rank -> {'rank_thresh': 45, 'min_hits': 3}
- PASS: maybe_adjust harden_hits at rank floor -> {'rank_thresh': 15, 'min_hits': 4}

(temp state dir removed: /tmp/lspace_autotune_smoke_1t5mdgm9)

## Verdict
- smoke_ok=True
