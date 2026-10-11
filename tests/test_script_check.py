"""Stray-script check on generated answers (app/rag/script_check.py).

Samples are real gpt-5-mini outputs from the 2026-10-11 glitch runs.
"""

from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.rag import generator
from app.rag.script_check import count_stray_tokens, strip_stray

VERSE = "إِنَّ صَلَاتِي وَنُسُكِي وَمَحْيَايَ وَمَمَاتِي لِلَّهِ رَبِّ الْعَالَمِينَ"


@pytest.mark.parametrize(
    "text",
    [
        "ইসলামী নেতৃত্বের গুণাবলি [১] এবং সততা [২]।",
        "Answer in English with [১] and Islamic terms like salah.",
        f'কুরআনে বলা হয়েছে: "{VERSE}" (সূরা আনআম: ১৬২)',
        "রাসূল ﷺ বলেছেন [৩]",
        "",
    ],
)
def test_clean_text_is_untouched(text):
    assert count_stray_tokens(text) == 0
    assert strip_stray(text) == text


@pytest.mark.parametrize(
    "text, expected",
    [
        ("নিয়মিত দینی পড়াশুনা বন্ধ", "নিয়মিত দ পড়াশুনা বন্ধ"),  # Arabic inside a Bengali word
        ("সিস্টেমатিক পরিকল্পনা", "সিস্টেমিক পরিকল্পনা"),  # Cyrillic
        ("বাস্তবে عمل ও নিষ্ঠা", "বাস্তবে ও নিষ্ঠা"),  # lone Arabic word
        ("আল্লাহর رضার জন্য", "আল্লাহর ার জন্য"),
        ("তাই շաբաթে সময় রাখুন", "তাই ে সময় রাখুন"),  # Armenian
    ],
)
def test_stray_letters_detected_and_stripped(text, expected):
    assert count_stray_tokens(text) >= 1
    cleaned = strip_stray(text)
    assert count_stray_tokens(cleaned) == 0
    assert "ی" not in cleaned and "عمل" not in cleaned and "а" not in cleaned
    assert cleaned == expected


def test_foreign_letters_inside_a_quoted_verse_are_removed_but_verse_kept():
    corrupted = VERSE.replace("لِلَّهِ", "لِلَّهِ대상으로")
    assert count_stray_tokens(f'"{corrupted}"') == 1
    cleaned = strip_stray(f'"{corrupted}"')
    assert count_stray_tokens(cleaned) == 0
    assert "رَبِّ الْعَالَمِينَ" in cleaned and "대" not in cleaned


def _fake_complete(outputs):
    calls = []

    def fake(task, messages, **kw):
        calls.append(task)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=outputs[len(calls) - 1]))])

    fake.calls = calls
    return fake


def test_generate_answer_regenerates_once_and_keeps_clean_retry(monkeypatch):
    fake = _fake_complete(["নিয়মিত دینی পড়া", "নিয়মিত দ্বীনি পড়া"])
    monkeypatch.setattr(generator, "complete", fake)
    assert generator.generate_answer("q", []) == "নিয়মিত দ্বীনি পড়া"
    assert len(fake.calls) == 2


def test_generate_answer_strips_when_retry_is_also_bad(monkeypatch):
    fake = _fake_complete(["নিয়মিত دینی পড়া", "নিয়মিত دینی সাহিত্য"])
    monkeypatch.setattr(generator, "complete", fake)
    out = generator.generate_answer("q", [])
    assert len(fake.calls) == 2 and count_stray_tokens(out) == 0


def test_generate_answer_no_retry_when_clean(monkeypatch):
    fake = _fake_complete(["পরিষ্কার উত্তর [১]"])
    monkeypatch.setattr(generator, "complete", fake)
    assert generator.generate_answer("q", []) == "পরিষ্কার উত্তর [১]"
    assert len(fake.calls) == 1


def test_retries_zero_strips_without_regenerating(monkeypatch):
    fake = _fake_complete(["নিয়মিত دینی পড়া"])
    monkeypatch.setattr(generator, "complete", fake)
    monkeypatch.setattr(settings, "answer_script_retries", 0)
    assert count_stray_tokens(generator.generate_answer("q", [])) == 0
    assert len(fake.calls) == 1


def test_corrupted_word_does_not_break_the_quotation_run():
    text = 'কুরআন: "مِنْ أَحَبَّ للهकلپق وَأَنْفُسَكُمْ" ও'
    assert count_stray_tokens(text) == 1  # only the Devanagari-corrupted word
    cleaned = strip_stray(text)
    assert "وَأَنْفُسَكُمْ" in cleaned and "क" not in cleaned


def test_verse_quoted_right_after_a_dash_is_kept():
    text = f"আল্লাহ বলেন—{VERSE}—এটাই সত্য"
    assert count_stray_tokens(text) == 0
    assert strip_stray(text) == text
