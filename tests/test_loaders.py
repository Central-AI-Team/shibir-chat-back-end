"""app/loaders: ported from shibir-chat-gpu-service/tests/test_ingestion.py
(the chunking / SQLite-book / pipeline tests are not, since that code wasn't
ported), plus _fingerprint and read_cs_posts."""

from app.loaders.sources import DumpStats, iter_mysql_values, read_cs_posts, read_pp_articles
from app.loaders.text import (
    _fingerprint,
    drop_repeated_sentences,
    html_to_text,
    is_repetitive,
)

# ---------- MySQL dump parsing ----------

DUMP = r"""
CREATE TABLE `posts` (`id` bigint, `title` text, `post` longtext);
INSERT INTO `posts` (`id`, `title`, `post`) VALUES
(1, 'It\'s ok', '<p>line1\nline2 \"q\"</p>'),
(2, 'NULLs', NULL);
INSERT INTO `other` (`id`, `x`) VALUES
(9, 'ignored');
INSERT INTO `posts` (`id`, `title`, `post`) VALUES
;\">truncated fragment of a damaged row'),
(3, 'বাংলা', 'লেখা (বন্ধনী), কমা');
ALTER TABLE `posts`
  ADD PRIMARY KEY (`id`);
"""


def test_mysql_values_parses_escapes_nulls_and_skips_damage():
    stats = DumpStats()
    rows = [vals for _, vals in iter_mysql_values(DUMP, "posts", stats)]
    assert rows == [
        ["1", "It's ok", '<p>line1\nline2 "q"</p>'],
        ["2", "NULLs", None],
        ["3", "বাংলা", "লেখা (বন্ধনী), কমা"],
    ]
    assert stats.rows == 3
    assert stats.skipped == 1  # the truncated fragment only, not the ALTER lines


def test_pp_articles_maps_16_column_rows_positionally(tmp_path):
    vals16 = [
        "7", "'slug'", "'Real Title'", "NULL", "NULL", "'24-05-2023'", "'t.jpg'", "'i.jpg'",
        "'<p>Body</p>'", "2197", "NULL", "'perspectivebd.com'", "1", "0", "'c'", "'u'",
    ]  # fmt: skip
    dump = (
        "INSERT INTO `articles` (`id`, `title`, `published_date`, `description`, `video_url`)"
        " VALUES\n"
        "(2, 'Five', '22-10-2022', 'desc', 'https://v.test/1'),\n"
        f"({', '.join(vals16)});\n"
    )
    path = tmp_path / "a.sql"
    path.write_text(dump, encoding="utf-8")
    docs = list(read_pp_articles(path))
    assert [(d.doc_id, d.title, d.html) for d in docs] == [
        ("2", "Five", "desc"),
        ("7", "Real Title", "<p>Body</p>"),
    ]
    assert docs[0].meta["video_url"] == "https://v.test/1"
    assert docs[1].meta == {"date": "24-05-2023", "byline": "perspectivebd.com", "video_url": ""}


def test_read_cs_posts_maps_columns_and_skips_wrong_width(tmp_path):
    dump = (
        "INSERT INTO `posts` (`id`, `title`, `writer`, `designation`, `post`, `news_date`)"
        " VALUES\n"
        "(1, 'শিরোনাম', 'লেখক', NULL, '<p>লেখা</p>', '2024-01-17'),\n"
        "(2, 'short row', 'x');\n"
    )
    path = tmp_path / "p.sql"
    path.write_text(dump, encoding="utf-8")
    stats = DumpStats()
    [doc] = read_cs_posts(path, stats)
    assert (doc.doc_id, doc.title, doc.html) == ("1", "শিরোনাম", "<p>লেখা</p>")
    assert doc.meta == {"writer": "লেখক", "designation": "", "date": "2024-01-17"}
    assert stats.skipped == 1


# ---------- text cleaning ----------


def test_html_to_text_drops_head_style_and_keeps_blocks():
    html = (
        "<html><head><title>T</title><style>p{color:red}</style></head>"
        "<body><p>প্রথম&nbsp;অনুচ্ছেদ।</p><div>Second&amp;para</div><script>x()</script></body></html>"
    )
    assert html_to_text(html) == "প্রথম অনুচ্ছেদ।\n\nSecond&para"


def test_html_to_text_keeps_zwnj_and_strips_zero_width_space():
    assert html_to_text("র‌য​়") == "র‌য়"


def test_drop_repeated_sentences_removes_pasted_twice_body():
    body = "এটি প্রথম বাক্য যা বেশ লম্বা। এটি দ্বিতীয় বাক্য যা বেশ লম্বা।"
    assert drop_repeated_sentences(f"{body}\n\n{body}") == body


def test_is_repetitive_catches_spam_not_prose():
    assert is_repetitive("Best Travel Guide Books 2020 " * 40)
    prose = " ".join(f"Sentence number {i} talks about topic {i * 7 % 13}." for i in range(60))
    assert not is_repetitive(prose[:300])  # too short to judge


def test_fingerprint_ignores_whitespace_and_case_only():
    assert _fingerprint("Hello   World\n বাংলা") == _fingerprint("hello world বাংলা")
    assert _fingerprint("hello world") != _fingerprint("hello world!")
