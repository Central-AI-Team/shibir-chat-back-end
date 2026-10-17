"""Detect and repair stray-script letters in generated answers.

Incident 2026-10-11: gpt-5-mini sometimes switches script in the middle of a
Bengali word ("নিয়মিত দینی পড়াশুনা", Cyrillic inside "সিস্টেমатিক", Thai or
Armenian letters, Korean/Hebrew/Devanagari inside a quoted Arabic verse). The
corpus does not contain these strings. A prompt rule ("write Bengali words in
Bengali script") and reasoning_effort=low both failed to reduce it (8/18 ->
10/18 answers affected on the same three questions), and temperature cannot be
lowered on this model (CLAUDE.md §5), so the answer is checked in code.

Allowed: Bengali and Latin letters anywhere; Arabic only as a quotation, i.e. a
run of at least two consecutive Arabic-only words (Quranic/hadith text copied
from an excerpt), or the ﷺ-style ligatures. Everything else is "stray": any
other script, and Arabic letters mixed into a Bengali/Latin word or standing
alone.
"""

from __future__ import annotations

import re
import unicodedata

_KEEP = {"BENGALI", "LATIN"}
_ALLOWED = _KEEP | {"ARABIC"}
_NEUTRAL = {"COMBINING", "MODIFIER"}
_TOKEN = re.compile(r"[^\s—–]+")  # em/en dashes split words: "আল্লাহ—إِنَّ" is two tokens
# Arabic ligature symbols such as ﷺ / ﷻ (U+FDF0-FDFF) are one-token honorifics.
_LIGATURES = re.compile(r"[ﷰ-﷿]")


def _script(ch: str) -> str | None:
    """First word of the Unicode name for letters/marks; None for everything else."""
    if unicodedata.category(ch)[0] not in "LM":
        return None
    word = unicodedata.name(ch, "").split(" ", 1)[0]
    return None if word in _NEUTRAL or not word else word


def _scripts(token: str) -> set[str]:
    return {s for s in map(_script, token) if s}


def _token_verdicts(text: str) -> list[tuple[re.Match, set[str], bool]]:
    """(token, scripts, is_stray) for every whitespace-separated token."""
    tokens = [(m, _scripts(m.group())) for m in _TOKEN.finditer(text)]
    # Length of the run of consecutive Arabic-only tokens each one belongs to.
    # Tokens without letters (punctuation, digits, [১]) neither extend nor
    # break a run, so '"إِنَّ صَلَاتِي" -' still counts as one quotation.
    # A token with Arabic plus a corrupting foreign letter still belongs to the quotation.
    arabicish = [bool(sc) and "ARABIC" in sc and not sc & _KEEP for _, sc in tokens]
    run_len = [0] * len(tokens)
    run: list[int] = []
    for i, (_, sc) in enumerate(tokens + [(None, {"END"})]):
        if not sc:
            continue
        if i < len(tokens) and arabicish[i]:
            run.append(i)
            continue
        for j in run:
            run_len[j] = len(run)
        run = []
    out = []
    for i, (m, sc) in enumerate(tokens):
        if not sc:
            stray = False
        elif sc - _ALLOWED:
            stray = True
        elif "ARABIC" in sc and len(sc) > 1:
            stray = True
        elif sc == {"ARABIC"}:
            stray = run_len[i] < 2 and not _LIGATURES.fullmatch(m.group().strip("()[]\"'“”.,;:"))
        else:
            stray = False
        out.append((m, sc, stray))
    return out


def count_stray_tokens(text: str) -> int:
    return sum(1 for _, _, stray in _token_verdicts(text) if stray)


def strip_stray(text: str) -> str:
    """Remove stray letters: foreign letters from mixed tokens, lone Arabic words."""
    pieces, last = [], 0
    for m, sc, stray in _token_verdicts(text):
        pieces.append(text[last : m.start()])
        last = m.end()
        tok = m.group()
        if not stray:
            pieces.append(tok)
        elif sc == {"ARABIC"}:
            continue  # a lone Arabic word with nothing Bengali around it: drop it
        else:
            keep = _KEEP if sc & _KEEP else {"ARABIC"}
            pieces.append("".join(c for c in tok if _script(c) is None or _script(c) in keep))
    pieces.append(text[last:])
    return re.sub(r"[ \t]{2,}", " ", "".join(pieces))
