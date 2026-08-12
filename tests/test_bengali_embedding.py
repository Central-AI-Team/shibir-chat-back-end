"""Regression test for the MiniLM -> bge-m3 migration.

The original bug: all-MiniLM-L6-v2's WordPiece vocabulary has no Bengali
codepoints, so every Bengali word tokenized to [UNK] and all documents
collapsed to nearly the same embedding vector -- retrieval was effectively
random. These assertions would have caught that regression immediately.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.rag.chunker import chunk_text, normalize
from app.rag.embedder import _model

BENGALI_TEXT = "নামাজের গুরুত্ব ও ফজিলত"


@pytest.fixture(scope="module")
def model():
    return _model()


def test_bengali_tokenizer_has_no_unk(model):
    tokenizer = model.tokenizer
    unk_token = tokenizer.unk_token
    tokens = tokenizer.tokenize(BENGALI_TEXT)
    assert tokens, "tokenizer produced no tokens for Bengali input"
    assert unk_token not in tokens, (
        f"Bengali text tokenized to {unk_token!r}; the model's vocabulary "
        "does not cover Bengali codepoints"
    )


def test_unrelated_bengali_sentences_are_not_collapsed(model):
    sentence_a = "নামাজের গুরুত্ব ও ফজিলত ইসলামে অপরিসীম।"
    sentence_b = "ফুটবল খেলায় গোল করার জন্য দক্ষ খেলোয়াড় প্রয়োজন।"

    vectors = model.encode(
        [sentence_a, sentence_b],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    similarity = float(np.dot(vectors[0], vectors[1]))
    assert similarity < 0.75, (
        f"unrelated sentences scored {similarity:.3f} cosine similarity; "
        "embeddings may be collapsing like the old MiniLM model did (~0.95)"
    )


def test_chunk_text_splits_long_bengali_string():
    long_text = ("নামাজের গুরুত্ব ও ফজিলত ইসলামে অপরিসীম। " * 100)[:3000]
    chunks = chunk_text(long_text)
    assert len(chunks) > 1, "3000-char input should split into multiple chunks"
    assert all(len(c) < 1100 for c in chunks), (
        "every chunk must stay under 1100 chars"
    )


def test_normalize_is_idempotent():
    text = "নামাজের গুরুত্ব ও ফজিলত  \n\n\n\nইসলামে অপরিসীম।"
    once = normalize(text)
    twice = normalize(once)
    assert once == twice
