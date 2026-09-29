"""HTML -> clean text, junk detection and exact-duplicate fingerprints.

Ported (standard library only) from shibir-chat-gpu-service/ingestion/text.py
and pipeline.py. Chunking is deliberately NOT ported: app/rag/chunker.py owns
that for everything that reaches Chroma.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import zlib
from html.parser import HTMLParser

_SKIP_TAGS = {"head", "style", "script", "title", "noscript", "iframe", "svg"}
_BLOCK_TAGS = {
    "p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "hr", "pre",
}  # fmt: skip


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


# Keep ZWNJ/ZWJ (U+200C/U+200D): they change Bengali conjunct rendering.
_INVISIBLE_RE = re.compile(r"[​‎‏⁠﻿­]")
_SPACES_RE = re.compile(r"[ \t  -   　]+")
_NEWLINES_RE = re.compile(r"\n\s*\n+")


def html_to_text(html: str) -> str:
    if "<" in html:
        parser = _TextExtractor()
        parser.feed(html)
        parser.close()
        text = "".join(parser.parts)
    else:
        text = html
    text = unicodedata.normalize("NFC", text)
    text = _INVISIBLE_RE.sub("", text)
    text = _SPACES_RE.sub(" ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(ln for ln in lines)
    return _NEWLINES_RE.sub("\n\n", text).strip()


def is_repetitive(text: str, threshold: float = 0.12) -> bool:
    """True for spam-like text (e.g. one sentence pasted many times).

    Normal prose compresses to ~30-45% with zlib; heavy repetition goes far lower.
    """
    data = text.encode("utf-8")
    if len(data) < 400:
        return False
    return len(zlib.compress(data, 9)) / len(data) < threshold


# Sentence end: Bengali danda/double danda, ?, !, or '.' followed by whitespace.
_SENT_RE = re.compile(r"(?<=[।॥?!])\s+|(?<=\.)\s+(?=\S)")


def _split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(paragraph) if s.strip()]


def drop_repeated_sentences(text: str, min_len: int = 20) -> str:
    """Remove sentences already seen earlier in the same document.

    Some posts contain their whole body pasted twice; this keeps one copy.
    Short sentences (< min_len chars, e.g. "হ্যাঁ।") are always kept.
    """
    seen: set[str] = set()
    paras = []
    for para in text.split("\n\n"):
        kept = []
        for sent in _split_sentences(para.replace("\n", " ")):
            key = " ".join(sent.split()).casefold()
            if len(key) >= min_len:
                if key in seen:
                    continue
                seen.add(key)
            kept.append(sent)
        if kept:
            paras.append(" ".join(kept))
    return "\n\n".join(paras)


def _fingerprint(text: str) -> str:
    """Exact-duplicate key: whitespace-collapsed, case-folded SHA-1."""
    return hashlib.sha1(" ".join(text.split()).casefold().encode()).hexdigest()
