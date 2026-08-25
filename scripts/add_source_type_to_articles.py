"""
One-time migration: adds source_type / source_ref / source_metadata columns
to the articles table, so it can hold content from PDF/Word docs, web
articles, other SQL databases, and CSV/spreadsheets -- not just the original
web-article content. Matches the Article model in app/db/models.py.

Usage:
    python -m scripts.add_source_type_to_articles

Safe to re-run: uses ADD COLUMN IF NOT EXISTS, so already-migrated databases
are left untouched.
"""
from sqlalchemy import text

from app.db.session import engine

STATEMENTS = [
    "ALTER TABLE articles ADD COLUMN IF NOT EXISTS source_type VARCHAR(30) "
    "NOT NULL DEFAULT 'web_article'",
    "ALTER TABLE articles ADD COLUMN IF NOT EXISTS source_ref VARCHAR(500)",
    "ALTER TABLE articles ADD COLUMN IF NOT EXISTS source_metadata JSONB",
]


def main() -> None:
    with engine.begin() as conn:
        for statement in STATEMENTS:
            conn.execute(text(statement))

    print(
        "articles table updated: added source_type (VARCHAR(30), default "
        "'web_article'), source_ref (VARCHAR(500)), source_metadata (JSONB)."
    )


if __name__ == "__main__":
    main()
