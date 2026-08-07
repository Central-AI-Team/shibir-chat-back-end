#!/bin/bash

# Exit on error
set -e

export HF_HUB_OFFLINE=1

echo "🚀 Starting FastAPI setup and server..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found - skipping dependency installation"
fi

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
uvicorn app.main:app --host 0.0.0.0 --port 9200