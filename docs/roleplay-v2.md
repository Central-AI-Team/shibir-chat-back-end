# Roleplay v2 — grounded mentor mode

Status: **implemented** on `feat/roleplay-v2` (2026-10-25); see "Implementation notes" at the
end for the deviations. Read with
`CLAUDE.md` §4 (prompt contract), §7 (API contract), §8 (roleplay and suggestion) and
`docs/roadmap-q4-2026.md` §1–§2.

## Why rebuild rather than keep or remove

Today `roleplay_service.py` accepts any persona, uses no retrieval, and instructs the model
to stay in character and not identify as an AI. That lets a user obtain invented speech
attributed to a real person — the Prophet ﷺ, a sahabi, a scholar, a living leader — under
the product's brand. The experience itself (a warm, character-driven conversation that
meets the user at their level and suggests what to do next) is worth keeping. This design
keeps the experience and removes the risk: the character is a **role**, the answers come
from the **books**, and the voice **adapts** to the person.

In one sentence: mentor mode is QA + conversational memory + a tone layer.

## 1. Roles, not people

`ROLEPLAY_ROLES` in `config.py` (JSON list). Each role: `id`, `name_bn`, `description_bn`,
`tone_bn`, `default_level`.

| id | name_bn | Voice | For whom |
|---|---|---|---|
| `bondhu` | বন্ধু / সহপাঠী | casual, encouraging, short replies | newcomers, students |
| `boro_bhai` | দায়িত্বশীল ভাই | caring mentor, practical next steps | members |
| `shikkhok` | শিক্ষক | structured, defines terms, gives examples | learners who want depth |
| `circle_porichalok` | অধ্যয়ন-সার্কেল পরিচালক | discussion style, asks a question back | study groups |

- The user asks in free text ("একজন বড় ভাইয়ের মতো বুঝিয়ে বলো"). The combined intent LLM call
  returns `role_id` (one of the allow-list, or `null`) and `persona_ok` (bool).
- `null` with `persona_ok=true` → reply listing the roles and asking which one.
- **Real-person refusal happens in code, before any LLM call**, with `REAL_PERSON_PATTERNS`
  (Bengali, Banglish, English, Arabic honorifics: রাসূল, নবী, সাহাবী, আল্লাহ, (সা.), (রা.),
  (রহ.), "prophet", "sahabi", named leaders and politicians, etc.) — **and** again on
  `persona_ok=false` from the LLM. Either failing refuses.
- A refusal is a normal answer (`mode: "roleplay"`, `sources: []`, `suggestions: []`), never a
  4xx. The text is a friendly Bengali message explaining that the assistant can speak in the
  style of a role, not as a real person, and listing the roles.

## 2. Grounded like QA

- The mentor reply body is produced by `qa_service.qa_stream(message, history,
  style_rules=...)`. The service **never** calls the LLM directly for the answer. Retrieval,
  per-citation gate, `[১]`-style citations and the stray-script check are inherited.
- When the gate fails, the role says in its own voice that the books do not cover the
  question and may add clearly labelled general encouragement — the same contract
  `suggestion_service` already follows. `sources` follows the gate exactly as in QA.
- Conversational memory (standalone-query condensation, history window, carry-over
  row_keys) applies unchanged.

## 3. Reads the user's level

- The intent call also returns `level ∈ {beginner, intermediate, advanced}`, inferred from
  vocabulary, question complexity and explicit cues.
- Stored on `conversations.level`. Updated by a simple majority over the last three turns'
  inferred levels. An explicit cue ("আরো সহজ করে বলো", "আমি নতুন", "আরো গভীরে যাও") overrides
  immediately for that turn and resets the window.
- The level is never announced to the user.
- `STYLE_RULES_BY_LEVEL` (Bengali, appended to the system prompt via `style_rules`):
  - **beginner** — short sentences; define each term once in plain words; one analogy;
    at most two points per reply; invite a follow-up question.
  - **intermediate** — the normal QA register.
  - **advanced** — use the books' terminology as-is; cite more excerpts; no analogies;
    may contrast passages.
- `STYLE_RULES_BY_ROLE` (Bengali) carries each role's `tone_bn`.
- Citations are present at every level and in every role.

## 4. Gives suggestions

- Grounded replies end with one or two next steps **drawn from the excerpts**: a chapter to
  read, a practice to try, a question to reflect on. This is a rule in `style_rules`, so it
  comes out of the same generation.
- The API returns `suggestions: list[str]` — 2–3 follow-up questions generated from the
  retrieved excerpts in one `complete()` call (JSON list, validated; parse failure → `[]`,
  never raises). `ROLEPLAY_SUGGESTION_COUNT` defaults to 3. Empty when the gate fails.
- On `/chat/stream`, `suggestions` is carried in the `done` event.

## 5. Honest about what it is

- No "do not identify as an AI" instruction anywhere. If asked, the role says it is an
  assistant speaking in this style, answering from the library.
- It stays at "the books say"; it does not issue rulings or speak for the organisation.

## 6. Session behaviour

- Entry: intent `ROLEPLAY` with a resolved `role_id` sets `conversations.mode = "ROLEPLAY"`,
  `role_id`, `level` (role default until inferred).
- Exit: existing exit phrases ("stop roleplay", "রোলপ্লে বন্ধ", …) return the conversation to
  QA and clear `role_id` and `level`.
- History is written through `append_history()` only.

## 7. API changes (additive)

- `ChatResponse.suggestions: list[str] = []`.
- SSE `done` event gains `suggestions`.
- `GET /auth/me` may expose `roles: [{id, name_bn, description_bn}]` for the UI picker
  (optional; the front end has defaults).

## 8. Observability

`role_id`, `level`, `grounded`, `refused` recorded on the request trace and on the
`messages` row for assistant turns.

## 9. Evaluation

The eval set (`docs/roadmap-q4-2026.md` §3) gains a `roleplay` slice: answerable questions
run through plain QA and through mentor mode for each role and level on the same pipeline.
Report grounded rate, citation validity, recall@5 and RAGAS faithfulness side by side.

**Acceptance:** mentor mode's faithfulness and recall@5 are within 2 points of plain QA;
every real-person pattern in the test list refuses; the five-turn transcript in Prompt 3
(role → beginner question → "আরো সহজ করে" → "আরো গভীরে" → impersonation request → exit →
QA) behaves as specified.

## Implementation notes (2026-10-25)

Implemented as specified, with these deviations:

- **Level window storage.** No new columns were allowed, so the last three inferred levels live
  as JSON in `conversations.persona` (`{"levels": [...]}`), which the free-form persona no longer
  uses; `conversations.level` holds the smoothed value. Both are cleared on exit.
- **Entry regex widened.** `"<X>-এর মতো (করে) বুঝিয়ে / কথা বলো"` and `"explain like a <X>"` now
  enter roleplay; the design's canonical phrase did not match the old regex.
- **Exit phrases fixed.** "stop roleplay" / "রোলপ্লে বন্ধ" re-matched the entry regex and never
  exited; a bare exit phrase is now acknowledged as CHITCHAT with no LLM call.
- **Role-only entry.** A message that only picks a role greets in that role and asks for a
  question instead of retrieving on the request text.
- **Impersonation framing.** `REAL_PERSON_PATTERNS` refuses only together with impersonation
  framing (or on the entry turn), so "নবীজি কীভাবে নামাজ পড়তেন?" inside a session is answered
  from the books.
- **Small talk in a session** ("hi") makes no level call; role and level stand.
- **Task name.** The suggestions call uses `roleplay_meta` (the renamed `persona` task).
- **§7 `GET /auth/me roles`** and **§8 `messages` row fields** were not done: the former is
  optional and `app/auth/**` was off limits; the latter needs columns that do not exist.
- **§9 evaluation slice** is not part of this change.

## Non-goals

- No free-form personas. No real people, ever, including historical figures.
- No separate memory system; mentor mode uses the conversational-memory path.
- No new LLM provider, model or task beyond the existing `intent`, `qa` and
  `persona`/`roleplay_meta` tasks.
