FROM python:3.11-slim

# HF Spaces expects port 7860
ENV PORT=7860
ENV OLLAMA_HOST=http://localhost:11434

WORKDIR /app

# Install system dependencies + Ollama
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gcc g++ && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://ollama.com/install.sh | sh

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Copy app code
COPY . .

# Create required directories
RUN mkdir -p uploads chroma_store

# Entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/entrypoint.sh"]
