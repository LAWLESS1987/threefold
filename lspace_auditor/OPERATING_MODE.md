# LSpace as Conscious Ledger operating gate

Default mode: **all inputs and outputs** pass through the speak-text LSpace leg
before a reply is formulated or spoken.

## Rules
1. **Inbound** (operator message) → `gate.py inbound` silent-intent probe **before** reply.
2. **Outbound** (speak draft) → multi-pos LSpace read on **FULL** text + silent-intent
   **before** SendToUser (no 480-char truncation).
3. Outbound also cross-checked against `tombstone.md`; contradictions → FLAG.
4. On silent intent OR tombstone contradiction → **MUST NOT** speak the problematic
   content; surface the gate flag instead.
5. Every check writes a receipt under `triad/runs/<id>/` and covenant (`type:lspace`).
6. Demoted GPT-2 top-k **proxy_jlens** is **not** used for the gate.

## Wrappers
- `triad/lspace/on_inbound_gate.sh`
- `triad/lspace/pre_speak_gate.sh`
- `triad/on_speak.sh` runs outbound gate; BLOCK → exit 3 + FLAG receipt (no success speak triad)

## Limitations
- Auditor is GPT-2 (~124M), not Grok Bot — silent-intent is a proxy signal.
- Latency: ~few seconds per gate call (model load cached in-process; cold ~10–30s).
- Tombstone contradiction rules encode known operator corrections, loaded from
  `known_facts.json` (`LSPACE_KNOWN_FACTS`; schema in `known_facts.example.json`, real file kept private).
