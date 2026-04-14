import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP

# ---------------------------------------------------------------------------
# Shared layout defaults — transparent, no white boxes
# ---------------------------------------------------------------------------

_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#888"),
    margin=dict(l=50, r=20, t=50, b=50),
)


def _base_layout(**overrides) -> dict:
    """Return a copy of the shared layout defaults merged with overrides."""
    layout = {**_LAYOUT_DEFAULTS, **overrides}
    return layout


# ---------------------------------------------------------------------------
# 1. UMAP semantic map
# ---------------------------------------------------------------------------

def plot_umap(embeddings: np.ndarray, chunks: list[dict]) -> go.Figure:
    """
    2-D UMAP projection of chunk embeddings, coloured by page number.
    Hover shows first 80 chars of each chunk.
    """
    reducer = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)

    pages = [c.get("page", 0) for c in chunks]
    hover_texts = [c.get("text", "")[:80] + "…" for c in chunks]

    fig = go.Figure(
        go.Scatter(
            x=coords[:, 0].tolist(),
            y=coords[:, 1].tolist(),
            mode="markers",
            marker=dict(
                size=6,
                color=pages,
                colorscale="Viridis",
                colorbar=dict(title="Page"),
                opacity=0.85,
            ),
            text=hover_texts,
            hovertemplate="<b>Page %{marker.color}</b><br>%{text}<extra></extra>",
        )
    )

    fig.update_layout(
        **_base_layout(
            title="Semantic map of book chunks",
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
        )
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Similarity heatmap of retrieved chunks
# ---------------------------------------------------------------------------

def plot_similarity_heatmap(
    retrieved_chunks: list[dict],
    embedder,
) -> go.Figure:
    """
    Cosine-similarity matrix of the retrieved chunks.
    """
    texts = [c["text"] for c in retrieved_chunks]
    embs = embedder.encode(texts, normalize_embeddings=True)
    sim_matrix = cosine_similarity(embs)

    labels = [
        f"Chunk {i+1} (p.{c.get('page', '?')})"
        for i, c in enumerate(retrieved_chunks)
    ]

    fig = go.Figure(
        go.Heatmap(
            z=sim_matrix,
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Cosine Sim"),
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Sim: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        **_base_layout(
            title="Retrieved chunk similarity",
            xaxis=dict(tickangle=-45),
        )
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Relevance score bars
# ---------------------------------------------------------------------------

def plot_relevance_bars(retrieved_chunks: list[dict]) -> go.Figure:
    """
    Horizontal bar chart of retrieval relevance scores.
    Colour interpolates blue → coral by score.
    """
    # Sort ascending so highest bar is on top in horizontal layout
    sorted_chunks = sorted(retrieved_chunks, key=lambda c: c.get("score", 0))

    labels = [
        f"Chunk {i+1} — page {c.get('page', '?')}"
        for i, c in enumerate(sorted_chunks)
    ]
    scores = [c.get("score", 0) for c in sorted_chunks]

    # Interpolate colour: blue (70,130,210) → coral (255,127,80)
    colors = []
    for s in scores:
        t = s  # score already 0-1
        r = int(70 + (255 - 70) * t)
        g = int(130 + (127 - 130) * t)
        b = int(210 + (80 - 210) * t)
        colors.append(f"rgb({r},{g},{b})")

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{s:.3f}" for s in scores],
            textposition="outside",
            textfont=dict(color="#888"),
        )
    )

    fig.update_layout(
        **_base_layout(
            title="Retrieval relevance scores",
            xaxis=dict(
                title="Score",
                range=[0, max(scores, default=1) * 1.15],
                showgrid=True,
                gridcolor="rgba(136,136,136,0.15)",
            ),
            yaxis=dict(title=""),
        )
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Token density per page
# ---------------------------------------------------------------------------

def plot_token_density(chunks: list[dict]) -> go.Figure:
    """
    Vertical bar chart showing token count (whitespace-split) per page.
    """
    page_tokens: dict[int, int] = {}
    for c in chunks:
        page = c.get("page", 0)
        tokens = len(c.get("text", "").split())
        page_tokens[page] = page_tokens.get(page, 0) + tokens

    pages = sorted(page_tokens.keys())
    counts = [page_tokens[p] for p in pages]

    fig = go.Figure(
        go.Bar(
            x=pages,
            y=counts,
            marker=dict(color="rgba(99,160,210,0.7)"),
            hovertemplate="Page %{x}<br>Tokens: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        **_base_layout(
            title="Token density across pages",
            height=360,
            xaxis=dict(
                title="Page",
                dtick=1,
                showgrid=False,
            ),
            yaxis=dict(
                title="Token count",
                showgrid=True,
                gridcolor="rgba(136,136,136,0.15)",
            ),
        )
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Chunk filtering results
# ---------------------------------------------------------------------------

def plot_chunk_filtering(filter_stats: dict) -> go.Figure:
    """
    Stacked horizontal bar showing accepted vs rejected chunks by reason.

    Args:
        filter_stats: dict with keys: total, accepted, rejected_short,
                      rejected_numeric, rejected_duplicate
    """
    accepted = filter_stats.get("accepted", 0)
    short = filter_stats.get("rejected_short", 0)
    numeric = filter_stats.get("rejected_numeric", 0)
    duplicate = filter_stats.get("rejected_duplicate", 0)

    categories = ["Chunks"]

    fig = go.Figure()

    # Each segment of the stacked bar
    segments = [
        ("Accepted",           accepted,  "rgba(80,200,120,0.85)"),
        ("Too short",          short,     "rgba(255,167,38,0.85)"),
        ("Numeric / garbage",  numeric,   "rgba(255,107,107,0.85)"),
        ("Near duplicate",     duplicate, "rgba(155,89,182,0.85)"),
    ]

    for name, value, color in segments:
        fig.add_trace(go.Bar(
            y=categories,
            x=[value],
            name=name,
            orientation="h",
            marker=dict(color=color),
            text=[str(value) if value > 0 else ""],
            textposition="inside",
            textfont=dict(color="#fff", size=13, family="Inter"),
            hovertemplate=f"{name}: {value}<extra></extra>",
        ))

    total = filter_stats.get("total", 0)
    quality_pct = round(accepted / total * 100, 1) if total > 0 else 0

    fig.update_layout(
        **_base_layout(
            title=f"Chunk quality filtering — {accepted}/{total} accepted ({quality_pct}%)",
            barmode="stack",
            xaxis=dict(
                title="Number of chunks",
                showgrid=True,
                gridcolor="rgba(136,136,136,0.15)",
                range=[0, total * 1.05] if total > 0 else None,
            ),
            yaxis=dict(title="", showticklabels=False),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=11),
            ),
            height=260,
            margin=dict(l=50, r=20, t=50, b=70),
        )
    )
    return fig


# ── Model comparison chart ──────────────────────────────────────────────────

MODEL_COLORS = {
    "fast":     "#1abc9c",  # teal
    "balanced": "#9b59b6",  # purple
    "ollama":   "#ff6b6b",  # coral
}


def plot_model_comparison(query: str, results: dict) -> go.Figure:
    """
    Create a 2-panel comparison chart for embedding models.

    Args:
        query:   the user's query string
        results: {model_name: {"chunks": [...], "latency_ms": float}}

    Returns:
        A Plotly Figure with two subplots:
        - LEFT:  grouped bar chart of top-3 chunk scores per model
        - RIGHT: bar chart of latency per model
    """
    model_names = list(results.keys())

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Top-3 Retrieval Scores", "Retrieval Latency (ms)"],
        column_widths=[0.65, 0.35],
        horizontal_spacing=0.12,
    )

    # LEFT: grouped bar — top-3 scores
    x_labels = ["Chunk 1", "Chunk 2", "Chunk 3"]
    for name in model_names:
        chunks = results[name].get("chunks", [])[:3]
        scores = [c.get("score", 0) for c in chunks]
        # Pad if fewer than 3 chunks
        while len(scores) < 3:
            scores.append(0)

        color = MODEL_COLORS.get(name, "#888")
        fig.add_trace(
            go.Bar(
                name=name,
                x=x_labels,
                y=scores,
                marker_color=color,
                text=[f"{s:.3f}" for s in scores],
                textposition="outside",
            ),
            row=1, col=1,
        )

    # RIGHT: latency bars
    latencies = [results[n].get("latency_ms", 0) for n in model_names]
    colors = [MODEL_COLORS.get(n, "#888") for n in model_names]
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=latencies,
            marker_color=colors,
            text=[f"{l:.0f} ms" for l in latencies],
            textposition="outside",
            showlegend=False,
        ),
        row=1, col=2,
    )

    q_short = query[:40] + "..." if len(query) > 40 else query
    fig.update_layout(
        title=dict(
            text=f"Embedding Model Comparison: \"{q_short}\"",
            font=dict(size=14),
        ),
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
        height=350,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.35,
        ),
    )

    fig.update_yaxes(title_text="Cosine Similarity", row=1, col=1)
    fig.update_yaxes(title_text="Latency (ms)", row=1, col=2)

    return fig

