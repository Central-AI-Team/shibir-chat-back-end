"""Settings.

CHANGED defaults:
  embedding_model_name  all-MiniLM-L6-v2 -> BAAI/bge-m3   (the critical one)
  top_k                 3 -> 5
NEW:
  reranker_model_name, fetch_k, min_similarity, min_rerank_score
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str
    gemini_model: str = "gemini-flash-latest"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

    # all-MiniLM-L6-v2 cannot tokenize Bengali -- every word became [UNK].
    embedding_model_name: str = "BAAI/bge-m3"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"

    database_url: str = "postgresql+psycopg2://user:password@localhost:5432/shibir_chat"

    chroma_persist_dir: str = "chroma_db"
    chroma_collection_name: str = "documents_bge_m3"  # new name = new index

    top_k: int = 5        # chunks sent to Gemini after reranking
    fetch_k: int = 25     # candidates pulled from Chroma before reranking

    # Cheap pre-filter on cosine similarity (1 - distance). Loose on purpose;
    # the reranker does the real filtering.
    min_similarity: float = 0.25

    # The real "do we have an answer?" gate. bge-reranker-v2-m3 logits: roughly
    # >2 clearly relevant, 0 to 2 borderline, <0 unrelated. TUNE THIS on ~30 of
    # your own questions -- too high and it refuses valid questions, too low and
    # it hallucinates from noise.
    min_rerank_score: float = 0.0

    tarun_db_path: str = "data/Tarun_Associate.db"
    nobin_db_path: str = "data/Nobin_Associate.db"

    host: str = "0.0.0.0"
    port: int = 9200

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()