from sentence_transformers import SentenceTransformer

from app.core.config import settings

_model = SentenceTransformer(settings.embedding_model_name)


def embed_text(text: str):
    return _model.encode(text).tolist()
