# Evaluation

Retrieval, reranking, prompts and model routing all affect answer quality in ways unit tests do
not catch. The scripts in `scripts/` measure that quality against labelled question sets. Run the
relevant ones before merging any change to the retrieval pipeline, the prompts or the model
configuration. [CONTRIBUTING.md](../CONTRIBUTING.md) lists which checks apply to which kind of
change.

All scripts run from the repository root, in the app's environment, with a working `.env`:

```bash
uv run python -m scripts.<script_name> [arguments]
```

> **Resource note.** Scripts that run retrieval load `bge-m3` and the reranker (about 6 GB of
> RAM) unless `GPU_SERVICE_URL` is set. On a machine with 8 GB or less, use GPU mode or run them
> on a larger host. Do not run them at the same time as `app.rag.ingest`: both open the same
> Chroma directory.

## Question sets

| File | Used by | Shape |
|---|---|---|
| `scripts/eval/seed.json`, `train.json`, `test.json` | `eval_retrieval`, `tune_threshold`, `eval_followups`, the nightly workflow | **v2** ([scripts/eval/schema.md](../scripts/eval/schema.md)): `id`, `slice`, `category`, `source_db`, `language`, `question`, `previous_turn`, `expected_row_keys`, `answerable`, `variant_of`, `review`, `notes`. `seed.json` is everything drafted; the split files hold accepted records only and are generated, never edited. `test.json` is held out: report on it, never tune on it. Workflow: [scripts/eval/README.md](../scripts/eval/README.md). |
| `scripts/retrieval_eval_questions.json` | `eval_retrieval`, `tune_threshold`, `eval_responses`, `eval_ragas` | **v1**: `{"query", "answerable", "relevant_ids": ["page_8219", ...]}`. Hand-labelled, 40 questions; kept as is and read as slice `legacy`. |
| `scripts/retrieval_eval_questions.example.json` | Template | The same shape, with a `_README` explaining how to label. |
| `scripts/eval_questions.example.json` | `tune_threshold`, `eval_responses` (default) | `{"query", "answerable"}`. Copy it; do not edit it in place. |
| `scripts/chat_intent_test_questions.json` | `eval_intent_routing`, `eval_task_routing` | Messages with their expected `/chat` mode. |

`relevant_ids` are Chroma `row_key` values (`page_<id>` or `article_<id>`), so one question can
match a whole page rather than a single chunk.

To extend the labelled set (details in [scripts/eval/README.md](../scripts/eval/README.md)):

1. `sample_corpus_pages --n 150`: a reproducible, spread-out sample of published pages.
2. `draft_eval_questions`: 1-2 Bengali questions per sampled page from the LLM (task
   `eval_draft`), written to `scripts/eval/seed.json` as `review: "pending"`. Prints the cost
   estimate first and needs `--yes`. Drafts reusing the page's wording are dropped.
3. Review by hand: set `review` to `accepted` or `rejected`. Add variants, follow-up pairs and
   negatives by hand. `eval_retrieval --label` still shows top candidates for unlabelled
   questions.
4. `validate_eval_set [file] [--no-chroma]`: every schema rule, live `row_key` existence,
   known-bad pages, near-duplicates, and the slice shares against the roadmap targets. Exits 1
   on hard errors.
5. `split_eval_set`: accepted records to `train.json` / `test.json`, 70/30, stratified by slice
   and category, fixed seed, variants kept with their `bn` record.

## Scripts

### Retrieval

| Script | What it answers |
|---|---|
| `eval_retrieval [file] [--k 1,3,5,10] [--no-rewrite] [--ablation]` | Are the right pages retrieved at all? Reports recall@k, hit rate and MRR, separately for the Chroma candidate pool and the reranked top-k. Each miss is classed as a retrieval loss or a rerank loss. No answers are generated. On a v2 file it also prints the same numbers, plus the gate confusion at `MIN_RERANK_SCORE`, **per slice and per category** (`breakdowns` in the JSON report). |
| `tune_threshold [file] [--report out.json]` | Where should `MIN_RERANK_SCORE` sit? Compares top rerank scores of answerable and unanswerable questions and suggests a cutoff. Use at least 30 questions (about 20 answerable, 10 not). On a v2 file it adds the confusion counts per slice and per category at the current, balanced and conservative thresholds. The choice is a human's: the output is a table, not a setting. |
| `eval_followups [file] [--k ...]` | Follow-up slice: recall@k / MRR of the follow-up question **with** condensation and carry-over (`classify_intent` + `condense` + `retrieve_stages(extra_row_keys=...)`) against the raw follow-up, after running `previous_turn` live to build the history. Exits with a message when the pipeline has no `history` parameter. |
| `eval_query_expansion` | Do more query-rewrite variants (`MAX_VARIANTS`) improve recall enough to justify the extra embed and search cost? |
| `eval_chunking` | Compares chunk size and overlap settings against the labelled set. Builds temporary indexes and is slow. |

### Routing and models

| Script | What it answers |
|---|---|
| `eval_intent_routing [file]` | Does `classify_intent()` pick the expected mode? |
| `eval_task_routing [--candidates ...]` | Can a cheaper model handle a mechanical task (`intent`, `rewrite`) without more parse failures or accuracy loss? Required before adding any entry to `MODEL_BY_TASK`. |
| `eval_generation_ab` | A/B test of generation models on Bengali answer quality, judged by a model from a different family. Needs `GROQ_API_KEY`. |

### End-to-end

| Script | What it answers |
|---|---|
| `eval_responses [file]` | Full pipeline: does each answerable question get a grounded answer with sources, and each unanswerable one a refusal with none? Writes an HTML report to `eval_reports/`. |
| `view_chat_responses [file]` | Runs messages through the real `/chat` handler and renders every response for manual reading. |
| `eval_ragas capture` / `score` | RAGAS metrics (faithfulness, answer relevancy, context precision and recall). Two phases: `capture` runs in the main environment; `score` runs in the separate `venv-ragas/` environment (`requirements-ragas.txt`). `score` prints a cost and time estimate and needs `--yes` to proceed. |

### Debugging

| Script | Purpose |
|---|---|
| `debug_pipeline` | Walks one query through every stage (normalize, intent, rewrite, embed, retrieve, rerank, gate, answer) and prints the intermediate results. |

## Recording results and the nightly run

`docs/eval-results.md` holds one row per run worth keeping (date, commit, eval-set version,
slice, recall@5, MRR, gate precision, gate recall, notes). `.github/workflows/eval-nightly.yml`
runs `eval_retrieval` on `scripts/eval/test.json` against a **staging** copy of the index every
night (and on `workflow_dispatch`), and fails when recall@5 drops by more than 2 points or gate
precision by more than 1 point against the last `all` row of that table. It never runs on PRs:
it costs OpenAI calls and needs the index. Repository secrets it expects, all placeholders until
set: `OPENAI_API_KEY`, `EVAL_GPU_SERVICE_URL`, `EVAL_GPU_API_KEY`, `EVAL_DATABASE_URL`,
`EVAL_CHROMA_ARCHIVE_URL` (a URL to a tarball of the staging `chroma_db/`).

**Citation validity** is checked on every production `generate_answer()` call
(`app/rag/generator.citation_labels_valid`: every `[n]`, Bengali or ASCII digits, within
`1..len(sources)`; no `[n]` when there are no sources). A failure is logged as a warning and
recorded as `citation_valid` on the request trace. The answer is never changed.

## Lessons learned

- **Check rewrites across repeated runs, not one call.** Query rewriting is non-deterministic.
  Routing `rewrite` to `openai/gpt-oss-20b` passed a single-run evaluation but misspelled key
  Banglish terms in 7 of 18 repeated rewrites. That dropped top rerank scores from about 0.92 to
  0.01, and the change was reverted to `gpt-5-mini`.
- **Separate retrieval from generation.** A wrong answer is either a retrieval miss or a
  generation error. Run `eval_retrieval` before blaming the prompt.
- **Re-tune the gate after changing the reranker or chunking.** `MIN_RERANK_SCORE` is calibrated
  on sigmoid scores from `bge-reranker-v2-m3` over the current chunking.
