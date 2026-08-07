from fastapi import FastAPI

from app.api.router import router

app = FastAPI(title="Shibir Chat Backend")
app.include_router(router)
