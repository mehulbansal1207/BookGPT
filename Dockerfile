FROM python:3.11-slim

ENV PORT=7860
ENV OLLAMA_HOST=0.0.0.0:11434

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama binary directly (no install script)
RUN curl -fsSL -o /usr/local/bin/ollama https://ollama.com/download/ollama-linux-amd64 && \
    chmod +x /usr/local/bin/ollama

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
