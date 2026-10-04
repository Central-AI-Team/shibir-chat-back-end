"""Excerpt labels in the prompt must use Bengali digits (CLAUDE.md §4, rule 3).

The model cites with whatever labels it sees, so ASCII labels ([1], [2])
produced ASCII citations even though the prompt asks for [১], [২].
"""

from app.rag.generator import format_context
from app.schemas.query import Citation


def _citation(n: int) -> Citation:
    return Citation(book=f"বই {n}", chapter="অধ্যায়", source_db="tarun", content=f"অংশ {n}")


def test_excerpts_are_labelled_with_bengali_digits(monkeypatch):
    # This test is about labels, not the context_top_k prompt limit.
    monkeypatch.setattr('app.rag.generator.settings.context_top_k', 20)
    context = format_context([_citation(n) for n in range(1, 13)])

    for label in ("[১]", "[২]", "[৯]", "[১০]", "[১২]"):
        assert f"{label} বই:" in context
    assert "[1]" not in context and "[10]" not in context


def test_suggestion_service_uses_the_same_labels():
    from app.services import suggestion_service

    assert suggestion_service.format_context is format_context
