# Eval set v2

Labelled questions for measuring retrieval, the relevance gate and follow-up handling.
Format: [schema.md](schema.md). Design and targets: `docs/roadmap-q4-2026.md` §3.
Numbers go in `docs/eval-results.md`; the person reading them decides settings, these files
and scripts only produce the numbers.

| File | What it is | Who edits it |
|---|---|---|
| `schema.md` | The record format and slice targets. | Humans. |
| `seed.json` | Every drafted or hand-written question, with `review` = `pending` / `accepted` / `rejected`. | The drafter appends; reviewers edit. |
| `train.json` | Accepted records, the 70 % used for tuning and ablations. | **Generated only** (`split_eval_set`). |
| `test.json` | Accepted records, the 30 % held out. | **Generated only**. Never used for tuning; read it only to report. |

## Adding questions

1. Draft from the corpus. `sample_corpus_pages --n 150` writes a stratified page sample
   (git-ignored `scripts/_corpus_sample.json`); `draft_eval_questions` turns each page into
   1-2 Bengali questions with `review: "pending"`. It prints the OpenAI cost estimate first and
   needs `--yes`. Drafts that reuse six consecutive words of the page are dropped automatically.
2. Hand-write what the drafter cannot: Banglish and English **variants** (`variant_of` the `bn`
   record, same `expected_row_keys`), **follow-up pairs** (`previous_turn` + a follow-up that
   contains an anaphor and names no book), **on-theme negatives** (Islamic topics the corpus does
   not cover) and **off-topic / adversarial** messages. Use the id prefixes in the schema and
   `review: "pending"`.
3. Keep each `bn` record's `category` and `source_db` from the page (the drafter fills them); the
   validator reports the shares so the set stays stratified.

## Reviewing

Open `seed.json`, read each pending record against its page (`notes` says what the page says),
then set `review` to `accepted` or `rejected`, editing the question if needed. Reject when the
question reuses the page's wording, names the book, needs more than the page to answer, or is
ambiguous. Nothing is accepted automatically. Then:

```bash
uv run python -m scripts.validate_eval_set scripts/eval/seed.json
```

It exits 1 on hard errors (unknown ids, keys missing from Chroma, follow-ups without an
anaphor, variants with different keys, ...) and prints the slice shares against the targets.

## Splitting

```bash
uv run python -m scripts.split_eval_set          # seed.json -> train.json + test.json
```

Deterministic (seed 20261025), 70/30, stratified by slice and category; a `bn` record and its
variants stay on the same side. Re-run after every review session and commit all three files
together. Do not edit `train.json` / `test.json` by hand.

**Rule:** `test.json` is never used to choose a threshold, a flag or a prompt. Tune on
`train.json`, then report `test.json` once per change in `docs/eval-results.md`.

## Running the reports

```bash
uv run python -m scripts.eval_retrieval scripts/eval/test.json --k 1,3,5,10   # recall, MRR, per slice / category
uv run python -m scripts.tune_threshold scripts/eval/train.json              # gate sweep, per slice / category
uv run python -m scripts.eval_followups scripts/eval/test.json               # follow-up slice, with vs without condensation
```

All three need the app environment, `.env`, and the index (GPU mode recommended; see
`docs/evaluation.md`). The nightly workflow (`.github/workflows/eval-nightly.yml`) runs
`eval_retrieval` on `test.json` against a staging index and fails on a regression against the
last row of `docs/eval-results.md`.
