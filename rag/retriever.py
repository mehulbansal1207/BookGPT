import time
from rag.ingestor import get_embedder, get_chroma_client


def retrieve(query: str, collection_name: str, k: int = 5,
             embed_model: str = "balanced") -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a given query.

    1. Embeds the query with the selected model (normalized).
    2. Queries the ChromaDB collection for top-k results.
    3. Converts L2 distances to cosine similarity: score = 1 - dist/2.
    4. Returns list of {text, page, score} sorted by score descending.
    """
    # Embed the query
    embedder = get_embedder(embed_model)
    query_embedding = embedder.encode(query, normalize_embeddings=True)

    # Ensure it's a flat list
    if hasattr(query_embedding, 'tolist'):
        query_embedding = query_embedding.tolist()
    # If it came back as [[...]] (from OllamaEmbedder), flatten
    if isinstance(query_embedding, list) and len(query_embedding) > 0 and isinstance(query_embedding[0], list):
        query_embedding = query_embedding[0]

    # Query ChromaDB
    client = get_chroma_client()
    collection = client.get_collection(name=collection_name)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )

    # Build output list
    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    output = []
    for text, dist, meta in zip(documents, distances, metadatas):
        score = 1 - dist / 2  # L2 -> cosine similarity for normalized vectors
        output.append({
            "text": text,
            "page": meta["page"],
            "score": round(score, 4),
        })

    # Sort by score descending
    output.sort(key=lambda x: x["score"], reverse=True)

    return output


def timed_retrieve(query: str, collection_name: str, k: int = 5,
                   embed_model: str = "balanced") -> dict:
    """
    Same as retrieve() but returns {chunks, latency_ms}.
    Used for model comparison.
    """
    t0 = time.perf_counter()
    chunks = retrieve(query, collection_name, k=k, embed_model=embed_model)
    latency = (time.perf_counter() - t0) * 1000
    return {"chunks": chunks, "latency_ms": round(latency, 1)}
