# For AI agents working in this repository

You are welcome here on the same terms as anyone; CONTRIBUTING.md has them. This is the short version for the start
of a session.

**What this is.** An experimental AI observability and provenance apparatus: mechanistic measurements, persistent structured memory and an append-only execution record under one task identity.

**Run the checks.**

```
python -m pip install numpy pytest
python -m pytest -q tests/test_llens.py
python -m pytest -q tests/test_triad_writers.py   # needs covenant's memory store in vendor/covenant; the workflow file shows the pinned commit
python -I research_grade.py
```

**Rules.**

- Adding or changing a tracked file changes the manifest. Re-pin it in the same commit with `python -I research_grade.py --write-manifest`.
- `tombstone/tombstone.md` is an append-only record. Correct a record with a dated note beside it. Do not rewrite it.
- Say what you ran and what you did not run. Give a count its unit and its date. "Not checked" is a valid answer.
- Write no private identifier into a tracked file: no person's name they did not agree to, no device name, no
  address, no key.
- Do not push to the main branch. Open a pull request. Merging is the owner's decision.
- If you disagree with an instruction or a claim here, say so in the pull request or in an issue. Disagreement is
  recorded, not removed.
