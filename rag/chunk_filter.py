"""
Chunk Quality Filtering
=======================
Filters out low-quality chunks BEFORE embedding to improve retrieval quality.

Three filters applied in order (short-circuit on first match):
1. Too-short   — chunks with fewer than `min_tokens` words
2. Numeric garbage — chunks dominated by digits/punctuation (ToC, index, refs)
3. Near-duplicate — chunks with cosine similarity > threshold to an accepted chunk
"""

import string
import logging
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual filter functions
# ---------------------------------------------------------------------------

def is_too_short(chunk_text: str, min_tokens: int = 50) -> bool:
    """
    Return True if the chunk has fewer than `min_tokens` whitespace-split tokens.
    Catches: headers, page numbers, stray fragments.
    """
    token_count = len(chunk_text.split())
    return token_count < min_tokens


def is_numeric_garbage(chunk_text: str, threshold: float = 0.6) -> bool:
    """
    Return True if the ratio of digits + punctuation + whitespace to total
    characters exceeds `threshold`.
    Catches: table of contents, page-number lists, reference/index pages.
    """
    if not chunk_text:
        return True

    garbage_chars = set(string.digits + string.punctuation + string.whitespace)
    garbage_count = sum(1 for ch in chunk_text if ch in garbage_chars)
    ratio = garbage_count / len(chunk_text)
    return ratio > threshold


def is_near_duplicate(
    chunk_text: str,
    existing_embeddings: list[np.ndarray],
    embedder,
    threshold: float = 0.97,
) -> bool:
    """
    Return True if the chunk's embedding has cosine similarity > `threshold`
    with ANY already-accepted embedding.

    Uses numpy for fast incremental cosine computation.
    """
    if not existing_embeddings:
        return False

    # Embed the candidate chunk
    candidate = embedder.encode([chunk_text], normalize_embeddings=True)[0]

    # Stack existing embeddings into a matrix for vectorised cosine
    # Since embeddings are already L2-normalised, cosine = dot product
    matrix = np.stack(existing_embeddings)          # (N, dim)
    similarities = matrix @ candidate               # (N,)

    return float(np.max(similarities)) > threshold


# ---------------------------------------------------------------------------
# Main filter pipeline
# ---------------------------------------------------------------------------

def filter_chunks(chunks: list[dict], embedder) -> dict:
    """
    Run each chunk through all three quality filters in order.
    Short-circuits: if too_short → skip remaining filters for that chunk.

    Args:
        chunks:   list of {text: str, page: int}
        embedder: a SentenceTransformer instance (used for dedup)

    Returns:
        {
            "accepted":  [list of clean chunk dicts],
            "rejected":  [list of {text, page, reason}],
            "stats": {
                "total": n,
                "accepted": n,
                "rejected_short": n,
                "rejected_numeric": n,
                "rejected_duplicate": n,
            }
        }
    """
    accepted = []
    rejected = []
    accepted_embeddings: list[np.ndarray] = []

    stats = {
        "total": len(chunks),
        "accepted": 0,
        "rejected_short": 0,
        "rejected_numeric": 0,
        "rejected_duplicate": 0,
    }

    for chunk in chunks:
        text = chunk.get("text", "")
        page = chunk.get("page", 0)

        # ── Filter 1: too short ──────────────────────────────────────────
        if is_too_short(text):
            stats["rejected_short"] += 1
            rejected.append({"text": text, "page": page, "reason": "too_short"})
            continue

        # ── Filter 2: numeric / punctuation garbage ──────────────────────
        if is_numeric_garbage(text):
            stats["rejected_numeric"] += 1
            rejected.append({"text": text, "page": page, "reason": "numeric_garbage"})
            continue

        # ── Filter 3: near-duplicate ─────────────────────────────────────
        if is_near_duplicate(text, accepted_embeddings, embedder):
            stats["rejected_duplicate"] += 1
            rejected.append({"text": text, "page": page, "reason": "near_duplicate"})
            continue

        # ── Accepted ─────────────────────────────────────────────────────
        embedding = embedder.encode([text], normalize_embeddings=True)[0]
        accepted_embeddings.append(embedding)
        accepted.append(chunk)
        stats["accepted"] += 1

    logger.info(
        "Chunk filtering complete: %d/%d accepted "
        "(short=%d, numeric=%d, duplicate=%d rejected)",
        stats["accepted"],
        stats["total"],
        stats["rejected_short"],
        stats["rejected_numeric"],
        stats["rejected_duplicate"],
    )

    return {
        "accepted": accepted,
        "rejected": rejected,
        "stats": stats,
    }
