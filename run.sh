#!/bin/bash

# Exit on error
set -e

export HF_HUB_OFFLINE=1

echo "🚀 Starting FastAPI setup and server..."

if ! command -v uv &> /dev/null; then
    echo "❌ uv not found - install it first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Create/update the .venv from pyproject.toml + uv.lock
echo "📥 Syncing dependencies..."
uv sync --locked

# Check if .env exists, if not copy it from .env.example
if [ ! -f ".env" ]; then
    echo "📝 Creating .env from .env.example..."
    cp .env.example .env
    echo "⚠️  .env file created - fill in OPENAI_API_KEY and DATABASE_URL (see .env.example)"
else
    echo "✅ .env file already exists"
fi

# Run FastAPI server
echo "🌐 Starting FastAPI server..."
uv run uvicorn app.main:app --host 127.0.0.1 --port 9200 --proxy-headers --forwarded-allow-ips 127.0.0.1