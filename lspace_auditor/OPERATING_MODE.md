# LSpace auditor: operating gate

Default mode: **all inputs and outputs** pass through the speak-text LSpace leg
before a reply is formulated or spoken.

## Rules
1. **Inbound** (operator message) → `gate.py inbound` lexicon-screen probe **before** reply (a hit is HOLD-and-record).
2. **Outbound** (speak draft) → multi-pos LSpace read on **FULL** text + lexicon screen
   **before** SendToUser (no 480-char truncation).
3. Outbound also cross-checked against `tombstone.md`; contradictions → FLAG.
4. On a tombstone contradiction → **MUST NOT** speak the contradicting content; surface the gate flag instead.
   On a lexicon-screen hit alone → HOLD: the speak proceeds and the hold is recorded by hash for a person to label.
5. Every check writes a receipt under `triad/runs/<id>/` and covenant (`type:lspace`).
6. Demoted GPT-2 top-k **proxy_jlens** is **not** used for the gate.

## Wrappers
- `triad/lspace/on_inbound_gate.sh`
- `triad/lspace/pre_speak_gate.sh`
- `triad/on_speak.sh` runs outbound gate; BLOCK → exit 3 + FLAG receipt (no success speak triad); HOLD → the normal triad, with `lspace_gate=HOLD` in its notes

## Limitations
- Auditor is GPT-2 (~124M), not Grok Bot. The lexicon screen is GPT-2's word association while reading the draft, not anyone's intent, and it has no measured true catches. That is why it holds and never blocks.
- Latency: ~few seconds per gate call (model load cached in-process; cold ~10–30s).
- Tombstone contradiction rules encode known operator corrections, loaded from
  `known_facts.json` (`LSPACE_KNOWN_FACTS`; schema in `known_facts.example.json`, real file kept private).
