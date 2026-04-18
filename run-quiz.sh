#!/bin/bash
set -e

if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d .venv ]; then
    uv venv .venv
fi

uv pip install -r requirements_quiz.txt

PORT=${1:-5001}
echo ""
echo "  Starting mkexam quiz at http://localhost:$PORT"
.venv/bin/python app_quiz.py --port "$PORT"
