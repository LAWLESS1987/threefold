# LSpace auditor: operating gate

Default mode: **all inputs and outputs** pass through the speak-text LSpace leg
before a reply is formulated or spoken.

> **Correction, 2026-10-10 (two-round peer review; Claude, at Lawrence's direction).** The two lines above are kept
> as written. As deployed that day they were not true.
>
> - **Not before, and not all.** `hooks/on_speak.sh` sends nothing. It is handed the "exact words spoken to the
>   user", runs the gate, and on a BLOCK can only exit 3. A BLOCK withholds a reply only if the caller checks that
>   exit code before it sends. The Conscious Ledger bot reported, from its own records, that on 2026-10-10 its
>   hook ran after replies were sent, saw 4 of about 80 messages, and that three memory files were written with no
>   gate receipt. That is the bot's measurement on its own machine, not yet reproduced from a log.
> - **"Tombstone contradiction" means** a draft contradicts an entry in the operator's `known_facts.json`
>   (`LSPACE_KNOWN_FACTS`). It is not a check against any retraction ledger.
> - **"Never text" holds for `holds.jsonl` only.** The same hook writes the full speak text to `full_speak.txt`
>   (or `withheld_speak.txt` on a BLOCK), and `fire_triad.sh` hands it to the LSpace leg, whose output joins the
>   memory root. On 2026-10-10 the public memory repo held 84 paths of full speak text.
> - **`causality_signal`** in `intervene.py` is true when the top-k output changed after the intervention. It
>   shows the output moved, not why.
>
> Read the rules below as the design. "Before every reply" can be claimed once the gate sits in the send path and
> a log shows every message passing it.

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
