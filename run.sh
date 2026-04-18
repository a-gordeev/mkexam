#!/bin/bash
set -e

if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "  Created .env — open it and set your GEMINI_API_KEY, then run again."
    exit 1
fi

if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d .venv ]; then
    uv venv .venv
fi

uv pip install -r requirements.txt

PORT=${1:-5000}
echo ""
echo "  Starting mkexam at http://localhost:$PORT"
.venv/bin/python app.py --port "$PORT"
