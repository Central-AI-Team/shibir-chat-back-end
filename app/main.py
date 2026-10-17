import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth_routes import router as auth_router
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


def create_auth_tables() -> None:
    """Create users / auth_sessions / guest_usage if missing (checkfirst; never
    alters an existing table). Without them every guest chat and login is a 503
    (the auth code fails closed), and nothing else creates them: there are no
    migrations. docs/sql/auth_tables.sql is the same schema for manual use.
    Never raises: a database that is down at startup only logs."""
    try:
        from app.auth import open_db
        from app.db.base import Base
        from app.db.models import AuthSession, GuestUsage, User

        with open_db() as db:
            Base.metadata.create_all(
                db.get_bind(), tables=[User.__table__, AuthSession.__table__, GuestUsage.__table__]
            )
    except Exception:
        logger.warning(
            "could not create the auth tables; guest chat and login return 503 "
            "until the database is reachable and they exist", exc_info=True,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Awaited, but bounded: a database that hangs must not block startup (the
    # deploy health check gives up after 180 s).
    try:
        await asyncio.wait_for(asyncio.get_running_loop().run_in_executor(None, create_auth_tables), 20)
    except asyncio.TimeoutError:
        logger.warning("creating the auth tables timed out; continuing startup")
    # Pay the langfuse SDK import + client-init cost now, not on the first
    # /chat request. No-op when tracing is disabled; never raises.
    tracing.init()
    # Background, not awaited: a cold GPU service can take ~2 min and readiness
    # (health checks, deploy.sh) must not wait on it. Keep a reference (on the
    # app, so ruff's F841 does not flag it) so the task is not garbage-collected
    # mid-run.
    app.state.warmup = None
    if settings.warmup_on_startup:
        app.state.warmup = asyncio.get_running_loop().run_in_executor(None, _warmup_models)
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
# Retry-After is not a CORS-safelisted response header, so without
# expose_headers a cross-origin front-end cannot read it from a 429.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
    expose_headers=["Retry-After"],
)

app.include_router(router)
app.include_router(auth_router)
