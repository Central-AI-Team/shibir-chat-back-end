FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /usr/local/bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock .
RUN uv sync --locked --no-install-project --no-dev

COPY . .

RUN uv sync --locked --no-dev

EXPOSE 9200

# Behind the reverse proxy uvicorn must trust X-Forwarded-* only from the proxy.
# From inside the container the proxy is the Docker bridge gateway, so
# docker-compose.yml overrides this env var (an explicit --forwarded-allow-ips
# flag would take precedence over it and defeat that override).
ENV FORWARDED_ALLOW_IPS=127.0.0.1

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9200", "--proxy-headers"]
