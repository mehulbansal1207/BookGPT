#!/bin/bash
set -e

echo "========================================="
echo "  📚 Book GPT — Starting Up"
echo "========================================="

# 1. Start Ollama server in background
echo ""
echo "🚀 Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# 2. Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to initialize..."
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Ollama failed to start after 60 seconds"
        exit 1
    fi
    sleep 2
done

# 3. Pull the LLM model
echo ""
echo "📥 Pulling phi3 model (this takes ~2 min on first start)..."
ollama pull phi3
echo "✅ phi3 model ready!"

# 4. Start Book GPT Flask server on port 7860
echo ""
echo "========================================="
echo "  📚 Book GPT is live on port 7860"
echo "========================================="
exec python server.py --port 7860
