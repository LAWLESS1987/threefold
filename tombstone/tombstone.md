# Tombstone log (SAMPLE)

Public sample only. Real deployments append locally; do not commit private entries.

Format: each entry is a markdown subsection with `id`, `task`, `done`, `outcome`,
`mistakes`, `lessons`, and `notes`. Triad runs stamp `task_id=...` and
`jlens_digest=...` into notes so the three legs share one id. Since 2026-10-08 they
also stamp `jlens_capture_sha256=...` and `llens_sha256=...`, which
`triad/verify_triad.py` checks against the lens leg and covenant. The entry below
predates that and is kept as written.

### 2026-10-07T02:09:13Z | jlens-triad-e2e
- id: 20261007-020913
- task: Prove real Anthropic jacobian-lens as triad third leg
- done: Installed gpt2 lens, wired fire_triad with tombstone+covenant+jlens.apply
- outcome: success
- mistakes: none
- lessons: Third leg must be real jlens.apply not self-observation
- notes: task_id=20261007-020907-13423; jlens_digest=jlens task_id=20261007-020907-13423 model=openai-community/gpt2 layers=[3, 6, 9, 10] topk={L3: [' boot', 'boot', 'Boot', ' Boot', ' boots']; L6: ['boot', ' boot', 'Boot', ' Boot', ' boots']; L9: ['strap', 'loader', 'boot', 'legged', ' boot']; L10: ['strap', 'stra', 'legged', 'loader', 'leg']} digest=82a9a0b5be4cfbaa; sample-demo-entry
