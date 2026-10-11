import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import router
from app.core import tracing
from app.core.config import settings


logger = logging.getLogger(__name__)


def _warmup_models() -> None:
    """Embed, search and rerank a dummy query so the first real request does
    not pay model-load / Chroma-open / GPU cold-start cost. Never raises."""
    try:
        from app.rag.chroma_client import get_collection
        from app.rag.embedder import embed_texts
        from app.rag.reranker import rerank

        vector = embed_texts(["warmup"])
        get_collection().query(query_embeddings=vector, n_results=1)
        rerank("warmup", ["warmup"], top_n=1)
        logger.info("warmup complete")
    except Exception:
        logger.warning("warmup failed; the first request will be slower", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pay the langfuse SDK import + client-init cost now, not on the first
    # /chat request. No-op when tracing is disabled; never raises.
    tracing.init()
    # Background, not awaited: a cold GPU service can take ~2 min and readiness
    # (health checks, deploy.sh) must not wait on it. Keep a reference so the
    # task is not garbage-collected mid-run.
    warmup = None
    if settings.warmup_on_startup:
        warmup = asyncio.get_running_loop().run_in_executor(None, _warmup_models)
    yield
    # Flush any buffered Langfuse events on graceful shutdown. No-op when
    # tracing is disabled; never raises.
    tracing.shutdown()


# Docs are opt-in (settings.enable_docs): in production the API schema is not
# published, and the reverse proxy 404s these paths as a second layer.
_docs_kwargs = (
    {}
    if settings.enable_docs
    else {"docs_url": None, "redoc_url": None, "openapi_url": None}
)
app = FastAPI(title="Shibir Chat Backend", lifespan=lifespan, **_docs_kwargs)

# Cross-origin access for the browser front-end. The anonymous owner cookie
# (app/api/deps.py) needs allow_credentials=True; Settings rejects "*" in
# cors_allow_origins because browsers refuse credentials with a wildcard.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)

app.include_router(router)
