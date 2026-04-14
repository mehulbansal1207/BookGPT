import os
import gc
import logging
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

from rag.chunk_filter import filter_chunks
from rag.parsers import parse_file

logger = logging.getLogger(__name__)

# ── Embedding model registry ────────────────────────────────────────────────
EMBEDDING_MODELS = {
    "fast":     {"name": "all-MiniLM-L6-v2",       "dims": 384,  "label": "Fast (MiniLM)"},
    "balanced": {"name": "BAAI/bge-large-en-v1.5",  "dims": 1024, "label": "Balanced (BGE)"},
    "ollama":   {"name": "nomic-embed-text",         "dims": 768,  "label": "Ollama (Nomic)"},
}

EMBEDDER_CACHE = {}   # cache all loaded models simultaneously
_chroma_client = None


# ── OllamaEmbedder (wraps Ollama's /api/embed endpoint) ─────────────────────

class OllamaEmbedder:
    """
    Drop-in replacement for SentenceTransformer that uses Ollama's
    embedding API (nomic-embed-text) over HTTP.
    Exposes the same .encode() interface.
    """

    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model
        logger.info("OllamaEmbedder initialised (model=%s)", model)

    def encode(self, texts, normalize_embeddings: bool = True, **kwargs):
        """
        Embed a list of texts (or a single string) via Ollama API.
        Returns np.ndarray of shape (N, dim).
        """
        import ollama as ollama_client

        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            resp = ollama_client.embed(model=self.model, input=text)
            vec = resp["embeddings"][0]
            embeddings.append(vec)

        result = np.array(embeddings, dtype=np.float32)

        if normalize_embeddings:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            result = result / norms

        return result


# ── Embedder accessors ──────────────────────────────────────────────────────

def get_embedder(model_name: str = "balanced"):
    """
    Get (or load) an embedder by name.  Models are cached in EMBEDDER_CACHE.

    model_name: "fast" | "balanced" | "ollama"
    """
    if model_name in EMBEDDER_CACHE:
        return EMBEDDER_CACHE[model_name]

    if model_name == "ollama":
        embedder = OllamaEmbedder()
    else:
        info = EMBEDDING_MODELS.get(model_name, EMBEDDING_MODELS["balanced"])
        hf_name = info["name"]
        logger.info("Loading embedding model %s (%s)…", model_name, hf_name)
        embedder = SentenceTransformer(hf_name)
        logger.info("Embedding model %s loaded.", model_name)

    EMBEDDER_CACHE[model_name] = embedder
    return embedder


def release_embedder(model_name: str | None = None):
    """
    Unload one or all embedders from RAM.
    If model_name is None, release ALL cached embedders.
    """
    global EMBEDDER_CACHE
    targets = [model_name] if model_name else list(EMBEDDER_CACHE.keys())

    for name in targets:
        obj = EMBEDDER_CACHE.pop(name, None)
        if obj is not None and not isinstance(obj, OllamaEmbedder):
            del obj

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    logger.info("Embedder(s) released: %s", targets)


# ── ChromaDB ────────────────────────────────────────────────────────────────

def get_chroma_client() -> chromadb.PersistentClient:
    """Singleton ChromaDB PersistentClient stored at ./chroma_store."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path="./chroma_store")
    return _chroma_client


# ── Chunking ────────────────────────────────────────────────────────────────

def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Split extracted text into chunks.
    Returns: list of {text: str, page: int}
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "(?<=\\. )", " ", ""],
        is_separator_regex=True,
        chunk_size=600,
        chunk_overlap=80,
    )

    chunks = []
    for page_data in pages:
        text = page_data["text"]
        page_num = page_data["page"]
        page_chunks = splitter.split_text(text)
        for chunk in page_chunks:
            chunk = chunk.strip()
            if chunk:
                chunks.append({
                    "text": chunk,
                    "page": page_num
                })
    return chunks


# ── Main ingestion pipeline ─────────────────────────────────────────────────

def ingest(file_path: str, collection_name: str,
           embed_model: str = "balanced") -> dict:
    """
    Orchestrates ingestion: parse -> chunk -> filter -> embed -> store.

    Args:
        file_path:       path to the uploaded file
        collection_name: base ChromaDB collection name (model suffix appended)
        embed_model:     "fast" | "balanced" | "ollama"

    Returns: dict with num_pages, num_chunks, embeddings, chunks,
             filter_stats, embed_model, collection_name.
    """
    pages = parse_file(file_path)
    raw_chunks = chunk_pages(pages)

    # The actual collection name includes the embedding model
    full_collection = f"{collection_name}_{embed_model}"

    if not raw_chunks:
        return {
            "num_pages": len(pages),
            "num_chunks": 0,
            "embeddings": np.array([]),
            "chunks": [],
            "filter_stats": {
                "total": 0, "accepted": 0,
                "rejected_short": 0, "rejected_numeric": 0,
                "rejected_duplicate": 0,
            },
            "embed_model": embed_model,
            "collection_name": full_collection,
        }

    # ── Quality filtering (before batch embedding) ───────────────────────
    embedder = get_embedder(embed_model)
    filter_result = filter_chunks(raw_chunks, embedder)
    chunks = filter_result["accepted"]
    filter_stats = filter_result["stats"]

    logger.info(
        "Filtered %d -> %d chunks: %s",
        filter_stats["total"],
        filter_stats["accepted"],
        filter_stats,
    )

    if not chunks:
        return {
            "num_pages": len(pages),
            "num_chunks": 0,
            "embeddings": np.array([]),
            "chunks": [],
            "filter_stats": filter_stats,
            "embed_model": embed_model,
            "collection_name": full_collection,
        }

    # ── Embed accepted chunks ────────────────────────────────────────────
    texts = [chunk["text"] for chunk in chunks]
    t0 = time.perf_counter()
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    embed_ms = (time.perf_counter() - t0) * 1000
    logger.info("Embedded %d chunks in %.0f ms with %s", len(texts), embed_ms, embed_model)

    # ── Store in ChromaDB ────────────────────────────────────────────────
    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection(name=full_collection)
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"page": chunk["page"]} for chunk in chunks]
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
    except Exception as e:
        logger.error("ChromaDB storage failed: %s", e)
        raise RuntimeError(
            f"Failed to store embeddings in ChromaDB: {e}\n"
            "Make sure the ./chroma_store directory is writable."
        ) from e

    return {
        "num_pages": len(pages),
        "num_chunks": len(chunks),
        "embeddings": embeddings,
        "chunks": chunks,
        "filter_stats": filter_stats,
        "embed_model": embed_model,
        "collection_name": full_collection,
    }
