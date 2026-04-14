#!/bin/bash
# ──────────────────────────────────────────────
# Book GPT — Setup Script
# Installs all dependencies for local execution
# ──────────────────────────────────────────────

set -e

echo "📚 Book GPT — Setup"
echo "════════════════════════════════════════"

# 1. Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# 2. spaCy language model
echo ""
echo "🧠 Downloading spaCy language model (en_core_web_sm)..."
python -m spacy download en_core_web_sm

# 3. Ollama model
echo ""
echo "🤖 Pulling Ollama model (llama3.1)..."
echo "   Make sure Ollama is installed: https://ollama.ai"
ollama pull llama3.1

echo ""
echo "════════════════════════════════════════"
echo "✅ Setup complete!"
echo ""
echo "To start Book GPT:"
echo "  1. Run: ollama serve"
echo "  2. Run: python server.py"
echo "════════════════════════════════════════"
