from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import router
from app.core import tracing
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pay the langfuse SDK import + client-init cost now, not on the first
    # /chat request. No-op when tracing is disabled; never raises.
    tracing.init()
    yield
    # Flush any buffered Langfuse events on graceful shutdown. No-op when
    # tracing is disabled; never raises.
    tracing.shutdown()


app = FastAPI(title="Shibir Chat Backend", lifespan=lifespan)

# Cross-origin access for the browser front-end. No cookies are used, so
# allow_credentials stays False and a plain origin allow-list is enough
# (see settings.cors_allow_origins).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(router)
