"""
test_query_rewriter.py
======================
query_rewriter.py ফাইলের দুটি অংশ আলাদাভাবে test করে:

  Part A — detect_language()  : LLM ছাড়া, শুধু pure Python logic (display/logging utility)
  Part B — expand_query()     : LLM mock করে, single-Bengali-translation flow end-to-end

চালানোর command:
  uv run pytest tests/test_query_rewriter.py -v
অথবা:
  pytest tests/test_query_rewriter.py -v
"""

from __future__ import annotations

import unittest.mock as mock

import pytest


# ══════════════════════════════════════════════════════════════════════════════
# Fixture: query_rewriter module লোড করো — app.core.llm mock করা হয়েছে, কারণ
# তার জন্য .env + API key দরকার।
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def qr():
    """
    query_rewriter module import করে return করে।
    app.core.llm stub করা হয়েছে।
    """
    import importlib
    import sys

    # Minimal stubs — শুধু যা query_rewriter.py import করে। MonkeyPatch
    # teardown-এ আসল module ফিরিয়ে দেয় (বা মুছে দেয়), নইলে পরের test file-গুলো
    # (যেমন test_tracing.py) MagicMock পেত।
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "app.core.llm", mock.MagicMock())
        mp.setitem(sys.modules, "app.rag.chunker", mock.MagicMock(normalize=lambda x: x.strip()))

        # এখন নিরাপদে import করা যাবে
        yield importlib.import_module("app.rag.query_rewriter")


# ══════════════════════════════════════════════════════════════════════════════
# PART A — detect_language() tests  (LLM call হয় না; expand_query() এখন এটা
# ব্যবহার করে না, কিন্তু display/logging-এর জন্য এখনও exported utility)
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectLanguage:
    """detect_language() সঠিক route return করছে কিনা।"""

    # ── Bengali ───────────────────────────────────────────────────────────────

    def test_pure_bengali_script(self, qr):
        assert qr.detect_language("নামাযের গুরুত্ব কী") == "BENGALI"

    def test_bengali_with_arabic_term(self, qr):
        assert qr.detect_language("ইসলামে রোযার বিধান কী") == "BENGALI"

    def test_bengali_tawba(self, qr):
        assert qr.detect_language("তওবার শর্ত কয়টি") == "BENGALI"

    def test_bengali_long_question(self, qr):
        assert qr.detect_language("নামাজ পড়ার সঠিক নিয়ম কী এবং কতটি ফরজ") == "BENGALI"

    # ── Banglish ──────────────────────────────────────────────────────────────

    def test_classic_banglish(self, qr):
        assert qr.detect_language("namajer gurutto ki") == "BANGLISH"

    def test_banglish_fasting(self, qr):
        assert qr.detect_language("roza rakhar niyom ki") == "BANGLISH"

    def test_banglish_wuzu(self, qr):
        assert qr.detect_language("wuzu korar sothik niyom") == "BANGLISH"

    def test_banglish_short(self, qr):
        assert qr.detect_language("namaz ki") == "BANGLISH"

    def test_banglish_dua(self, qr):
        assert qr.detect_language("Allah ke ki vabe dakbo") == "BANGLISH"

    def test_banglish_quran(self, qr):
        assert qr.detect_language("quran tela wat korar niyom") == "BANGLISH"

    # ── English / Arabic ──────────────────────────────────────────────────────

    def test_english_prayer(self, qr):
        assert qr.detect_language("what is the importance of prayer") == "ENGLISH_ARABIC"

    def test_english_wudu(self, qr):
        assert qr.detect_language("how to perform wudu correctly") == "ENGLISH_ARABIC"

    def test_arabic_pure(self, qr):
        assert qr.detect_language("ما هي أهمية الصلاة") == "ENGLISH_ARABIC"

    def test_arabic_wuzu(self, qr):
        assert qr.detect_language("كيف أتوضأ") == "ENGLISH_ARABIC"

    def test_english_with_islamic_noun_only(self, qr):
        # "salat importance" — Islamic noun আছে কিন্তু Banglish connector নেই
        # এটা English হওয়া উচিত, Banglish নয়
        assert qr.detect_language("salat importance") == "ENGLISH_ARABIC"

    def test_english_zakat(self, qr):
        assert qr.detect_language("what is zakat in Islam") == "ENGLISH_ARABIC"

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_empty_string(self, qr):
        # Empty হলে fallback ENGLISH_ARABIC
        result = qr.detect_language("")
        assert result in ("BENGALI", "BANGLISH", "ENGLISH_ARABIC")

    def test_mixed_bengali_english_mostly_bengali(self, qr):
        # বেশিরভাগ Bengali script → BENGALI
        assert qr.detect_language("নামাজ prayer করার নিয়ম") == "BENGALI"


# ══════════════════════════════════════════════════════════════════════════════
# PART B — expand_query() end-to-end tests  (LLM mock করা)
#
# নতুন সহজ contract: input বাংলা/Banglish/ইংরেজি/আরবি যাই হোক, একটাই LLM call
# তাকে plain বাংলায় রূপান্তর করে, এবং expand_query() একটি 1-element tuple
# ফেরত দেয়: (bengali_translation,)। কোনো route branching, variants, বা
# preserved-original field নেই।
# ══════════════════════════════════════════════════════════════════════════════

def _make_llm_response(content: str):
    """LLM যা plain text return করবে তার mock তৈরি করে।"""
    fake_choice = mock.MagicMock()
    fake_choice.message.content = content
    fake_response = mock.MagicMock()
    fake_response.choices = [fake_choice]
    return fake_response


class TestExpandQuery:
    """expand_query() পুরো flow ঠিকঠাক কাজ করছে কিনা।"""

    def test_bengali_translation_returned(self, qr):
        """LLM যা বাংলা টেক্সট ফেরত দেয়, তাই result-এ থাকবে।"""
        fake_resp = _make_llm_response("নামাযের গুরুত্ব কী")
        with mock.patch.object(qr, "complete", return_value=fake_resp):
            qr.expand_query.cache_clear()
            result = qr.expand_query("namajer gurutto ki")

        assert result == ("নামাযের গুরুত্ব কী",)

    def test_bengali_input_also_goes_through_llm(self, qr):
        """ইতিমধ্যে বাংলা input হলেও একই single LLM call flow ব্যবহার হয়।"""
        fake_resp = _make_llm_response("নামাযের গুরুত্ব কী")
        with mock.patch.object(qr, "complete", return_value=fake_resp):
            qr.expand_query.cache_clear()
            result = qr.expand_query("নামাযের গুরুত্ব কী")

        assert result == ("নামাযের গুরুত্ব কী",)

    def test_max_variants_param_accepted_but_has_no_effect(self, qr):
        """max_variants শুধু call-site compatibility-র জন্য accepted, কোনো effect নেই।"""
        fake_resp = _make_llm_response("নামাযের গুরুত্ব কী")
        with mock.patch.object(qr, "complete", return_value=fake_resp):
            qr.expand_query.cache_clear()
            result = qr.expand_query("namajer gurutto ki", max_variants=2)

        assert result == ("নামাযের গুরুত্ব কী",)

    # ── Fallback behavior ─────────────────────────────────────────────────────

    def test_llm_failure_falls_back_to_raw_query(self, qr):
        """
        LLM call fail করলে raw query return হবে।
        Pipeline কখনো break হবে না।
        """
        with mock.patch.object(qr, "complete", side_effect=Exception("API down")):
            qr.expand_query.cache_clear()
            qr._reset_fallback_count()
            result = qr.expand_query("নামাযের গুরুত্ব কী")

        assert result == ("নামাযের গুরুত্ব কী",), "Fallback-এ raw query return হওয়া উচিত"
        assert qr.get_fallback_count() == 1, "Fallback counter increment হয়নি"

    def test_llm_empty_content_falls_back(self, qr):
        """LLM খালি content ফেরত দিলেও crash হবে না, raw query দিয়ে fallback হবে।"""
        fake_resp = _make_llm_response("")
        with mock.patch.object(qr, "complete", return_value=fake_resp):
            qr.expand_query.cache_clear()
            qr._reset_fallback_count()
            result = qr.expand_query("namajer gurutto ki")

        assert result == ("namajer gurutto ki",)
        assert qr.get_fallback_count() == 1

    def test_empty_query_returns_empty_tuple(self, qr):
        """Empty query → empty tuple।"""
        qr.expand_query.cache_clear()
        result = qr.expand_query("   ")
        assert result == ()

    # ── Return type ───────────────────────────────────────────────────────────

    def test_expand_query_returns_tuple(self, qr):
        """Return type সবসময় tuple হবে (downstream embed_texts() এর জন্য)।"""
        fake_resp = _make_llm_response("নামায")
        with mock.patch.object(qr, "complete", return_value=fake_resp):
            qr.expand_query.cache_clear()
            result = qr.expand_query("নামায")
        assert isinstance(result, tuple), f"tuple হওয়া উচিত, পেয়েছি {type(result)}"

    def test_expand_query_all_elements_are_strings(self, qr):
        """tuple-এর প্রতিটি element str হবে।"""
        fake_resp = _make_llm_response("নামায")
        with mock.patch.object(qr, "complete", return_value=fake_resp):
            qr.expand_query.cache_clear()
            result = qr.expand_query("নামায")
        assert all(isinstance(q, str) for q in result), f"সব element str হওয়া উচিত: {result}"
