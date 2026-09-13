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

# Check if .env exists, if not create template
if [ ! -f ".env" ]; then
    echo "📝 Creating .env template..."
    cat > .env << 'EOF'
# Environment variables for FastAPI app
# Add your configuration here
# Example:
# DATABASE_URL=sqlite:///./test.db
# SECRET_KEY=your-secret-key-here
# API_KEY=your-api-key
EOF
    echo "⚠️  .env file created - please configure your environment variables"
else
    echo "✅ .env file already exists"
fi

# Run FastAPI server
echo "🌐 Starting FastAPI server..."
uv run uvicorn app.main:app --host 0.0.0.0 --port 9200