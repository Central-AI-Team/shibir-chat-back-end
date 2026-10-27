# Eval results

One row per run worth keeping: a baseline, a setting change, a model change. The nightly
workflow compares against the **last row whose slice is `all`** (recall@5 and gate precision
columns), so append, never reorder. Numbers come from `scripts/eval_retrieval.py` on
`scripts/eval/test.json` unless the notes say otherwise; percentages, one decimal.

Gate precision/recall: of "grounded" at `MIN_RERANK_SCORE`, answerable = positive
(`eval_retrieval._gate_counts`, the same rule as `tune_threshold._metrics_at`).

| date | commit | eval-set version | slice | recall@5 | MRR | gate precision | gate recall | notes |
|---|---|---|---|---|---|---|---|---|

No row yet: `scripts/eval/test.json` does not exist until the drafted questions have been
reviewed and split (2026-10-25: 0 accepted records). The v1 file
`scripts/retrieval_eval_questions.json` (40 questions, slice `legacy`) is not a held-out set and
is not recorded here.
