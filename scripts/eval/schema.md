# Eval-set record schema (v2)

One JSON list per file (`seed.json`, `train.json`, `test.json`). Every record has every
field below; use `null`, not absence. `scripts/validate_eval_set.py` enforces the rules.
The v1 file `scripts/retrieval_eval_questions.json` keeps its old shape
(`query` / `relevant_ids` / `lang`); the eval scripts read both.

| Field | Type | Meaning and rule |
|---|---|---|
| `id` | string | Unique across the file. `<prefix>-<nnnn>` with the slice prefix: `bn`, `bl`, `en`, `fu`, `neg`, `off`. |
| `slice` | string | One of `answerable_bn`, `answerable_banglish`, `answerable_en`, `followup`, `unanswerable_on_theme`, `off_topic`. Decides the language and `answerable` value (below). |
| `category` | string or null | The corpus `categories.name` of the pages in `expected_row_keys` (for example `কুরআন ও তাফসীর`). `null` for the two negative slices. |
| `source_db` | string or null | Where the expected pages come from: `tarun`, `nobin`, `tafheem`, `books_sql`, `shaheed`, `cs-post`, `pp-article`, `central-leadership`. `null` for negatives. |
| `language` | string | `bn`, `banglish` or `en`. Fixed by the slice: `answerable_bn` and `followup` are `bn`; `answerable_banglish` is `banglish`; `answerable_en` is `en`; negatives may be any of the three. |
| `question` | string | The message sent to the pipeline, exactly as a user would type it. Must not reuse a sentence of the expected page verbatim (the drafter rejects n-gram overlap; reviewers reject by eye). |
| `previous_turn` | string or null | `followup` only: the first-turn question of the pair. The follow-up harness runs it live to build the history, then sends `question`. `null` for every other slice. A follow-up must contain an anaphor (`intent_classifier.ANAPHORA`) and must not name the expected page's book. |
| `expected_row_keys` | list of string | `page_<id>` / `article_<id>` (`app/rag/ingest.py` row keys), the unit of relevance. Each must exist in the live Chroma collection and must not be a known-bad page. Empty exactly when `answerable` is `false`; several keys when several pages answer. |
| `answerable` | bool | `true` for the four answerable slices, `false` for `unanswerable_on_theme` and `off_topic`. |
| `variant_of` | string or null | `answerable_banglish` / `answerable_en` only: the `id` of the `answerable_bn` record this translates. The variant must carry the same `expected_row_keys`. `null` elsewhere. |
| `review` | string | `pending` (drafted, unreviewed), `accepted`, `rejected`. Only `accepted` records reach `train.json` / `test.json`. |
| `notes` | string | Free text: what the page says, why the question is hard, reviewer remarks. May be empty. |

## Slice targets (250 questions; `docs/roadmap-q4-2026.md` §3)

| slice | share | target |
|---|---|---|
| `answerable_bn` | 40 % | 100 |
| `answerable_banglish` | 15 % | 38 |
| `answerable_en` | 5 % | 12 |
| `followup` | 15 % | 38 |
| `unanswerable_on_theme` | 15 % | 37 |
| `off_topic` | 10 % | 25 |

Within `answerable_bn`, stratify by `category` and `source_db` in proportion to the eligible
corpus pool (`scripts/sample_corpus_pages.py` does the sampling). The validator prints the
observed shares against the targets; drift is a warning, not an error.

## Example

```json
{
  "id": "bn-0007",
  "slice": "answerable_bn",
  "category": "কুরআন ও তাফসীর",
  "source_db": "tafheem",
  "language": "bn",
  "question": "সূরা ফাতিহায় 'সিরাতুল মুসতাকীম' বলতে কোন পথকে বোঝানো হয়েছে?",
  "previous_turn": null,
  "expected_row_keys": ["page_16394"],
  "answerable": true,
  "variant_of": null,
  "review": "accepted",
  "notes": "page explains the straight path as the path of those favoured, not of the astray"
}
```
