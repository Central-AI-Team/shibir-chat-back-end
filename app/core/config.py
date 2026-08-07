from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str
    gemini_model: str = "gemini-flash-latest"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

    embedding_model_name: str = "all-MiniLM-L6-v2"

    chroma_persist_dir: str = "chroma_db"
    chroma_collection_name: str = "documents"

    top_k: int = 3

    tarun_db_path: str = "data/Tarun_Associate.db"
    nobin_db_path: str = "data/Nobin_Associate.db"

    host: str = "0.0.0.0"
    port: int = 9200

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
