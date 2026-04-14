import spacy
from collections import defaultdict
from itertools import combinations

import networkx as nx
from pyvis.network import Network

from rag.ingestor import get_embedder
from rag.retriever import retrieve

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_nlp = None

VALID_ENT_TYPES = {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT",
                   "WORK_OF_ART", "LAW", "NORP", "FAC", "LOC"}

NODE_COLORS = {
    "PERSON": "#FF7F7F",   # coral
    "ORG":    "#4A90D9",   # blue
    "GPE":    "#50C878",   # green
}
DEFAULT_NODE_COLOR = "#9B59B6"  # purple


def _load_nlp():
    """Singleton spaCy loader."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. "
                "Run: python -m spacy download en_core_web_sm"
            )
    return _nlp


def _normalize(text: str) -> str:
    """Lowercase and strip a concept string."""
    return text.lower().strip()


def _extract_concepts(doc) -> list[tuple[str, str]]:
    """
    Extract meaningful concepts from a spaCy Doc.
    Returns list of (normalized_name, type_label).
    """
    concepts = []

    # Named entities
    for ent in doc.ents:
        if ent.label_ in VALID_ENT_TYPES:
            name = _normalize(ent.text)
            if len(name) > 2:
                concepts.append((name, ent.label_))

    # Noun chunks (labelled as CONCEPT)
    for nc in doc.noun_chunks:
        name = _normalize(nc.text)
        # Skip very short / pronoun-only chunks
        if len(name) > 2 and name not in {"it", "its", "this", "that", "they",
                                            "them", "these", "those", "which",
                                            "what", "who", "whom", "the"}:
            concepts.append((name, "CONCEPT"))

    return concepts


# ---------------------------------------------------------------------------
# 1. build_graph
# ---------------------------------------------------------------------------

def build_graph(chunks: list[dict]) -> nx.Graph:
    """
    Build an undirected weighted co-occurrence graph from chunks.

    Nodes  — unique concepts with {type, frequency, pages}.
    Edges  — concepts co-occurring in the same chunk; weight = count.
    """
    nlp = _load_nlp()
    G = nx.Graph()

    for chunk in chunks:
        text = chunk.get("text", "")
        page = chunk.get("page", 0)
        doc = nlp(text)

        concepts = _extract_concepts(doc)
        # Deduplicate within a single chunk (keep first type seen)
        seen = {}
        for name, ctype in concepts:
            if name not in seen:
                seen[name] = ctype

        unique_concepts = list(seen.items())  # [(name, type), ...]

        # Update nodes
        for name, ctype in unique_concepts:
            if G.has_node(name):
                G.nodes[name]["frequency"] += 1
                if page not in G.nodes[name]["pages"]:
                    G.nodes[name]["pages"].append(page)
                # Keep the most specific type (prefer named-ent over CONCEPT)
                if ctype != "CONCEPT":
                    G.nodes[name]["type"] = ctype
            else:
                G.add_node(name, type=ctype, frequency=1, pages=[page])

        # Update edges — every pair of concepts in this chunk
        names = [n for n, _ in unique_concepts]
        for a, b in combinations(names, 2):
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)

    return G


# ---------------------------------------------------------------------------
# 2. graph_retrieve
# ---------------------------------------------------------------------------

def graph_retrieve(
    query: str,
    graph: nx.Graph,
    chunks: list[dict],
    embedder=None,
    top_n: int = 3,
) -> list[dict]:
    """
    Retrieve chunks via concept-graph traversal.

    1. Extract concepts from the query.
    2. Match them to graph nodes (exact + fuzzy substring).
    3. 2-hop BFS to collect neighbor concepts.
    4. Score chunks by how many neighbor concepts they contain.
    5. Return top_n results.
    """
    nlp = _load_nlp()
    doc = nlp(query)
    query_concepts = _extract_concepts(doc)
    query_names = {name for name, _ in query_concepts}

    # Also add lowered query tokens as fallback matches
    query_tokens = {_normalize(tok.text) for tok in doc if not tok.is_stop and len(tok.text) > 2}
    query_names |= query_tokens

    # Match to graph nodes: exact + fuzzy substring
    matched_nodes = set()
    all_nodes = set(graph.nodes())
    for qn in query_names:
        for node in all_nodes:
            if qn == node or qn in node or node in qn:
                matched_nodes.add(node)

    if not matched_nodes:
        return []

    # 2-hop BFS from matched nodes
    neighbor_concepts = set(matched_nodes)
    for node in matched_nodes:
        if node not in graph:
            continue
        # Hop 1
        for n1 in graph.neighbors(node):
            neighbor_concepts.add(n1)
            # Hop 2
            for n2 in graph.neighbors(n1):
                neighbor_concepts.add(n2)

    # Score each chunk by how many neighbor concepts it contains
    scored = []
    for chunk in chunks:
        text_lower = chunk["text"].lower()
        matched = [c for c in neighbor_concepts if c in text_lower]
        if matched:
            scored.append({
                "text": chunk["text"],
                "page": chunk.get("page", 0),
                "score": len(matched),
                "matched_concepts": matched,
            })

    # Normalize scores to 0-1
    if scored:
        max_score = max(s["score"] for s in scored)
        if max_score > 0:
            for s in scored:
                s["score"] = round(s["score"] / max_score, 4)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]


# ---------------------------------------------------------------------------
# 3. combined_retrieve
# ---------------------------------------------------------------------------

def combined_retrieve(
    query: str,
    collection_name: str,
    graph: nx.Graph,
    chunks: list[dict],
    k: int = 5,
    embed_model: str = "balanced",
) -> list[dict]:
    """
    Merge vector retrieval and graph retrieval into a single ranked list.

    - Vector retrieve: top k  (scores already 0-1)
    - Graph retrieve:  top 3  (scores normalized 0-1)
    - Dedup by text; boost score +0.15 if chunk appears in both.
    - Return up to k+2 results.
    """
    # Vector retrieval
    vec_results = retrieve(query, collection_name, k=k, embed_model=embed_model)

    # Graph retrieval
    graph_results = graph_retrieve(query, graph, chunks, top_n=3)

    # Merge into a dict keyed by text
    merged: dict[str, dict] = {}

    for r in vec_results:
        key = r["text"]
        merged[key] = {
            "text": r["text"],
            "page": r.get("page", 0),
            "score": r.get("score", 0.0),
            "matched_concepts": [],
            "_sources": {"vector"},
        }

    for r in graph_results:
        key = r["text"]
        if key in merged:
            # Present in both — boost
            merged[key]["score"] += 0.15
            merged[key]["matched_concepts"] = r.get("matched_concepts", [])
            merged[key]["_sources"].add("graph")
        else:
            merged[key] = {
                "text": r["text"],
                "page": r.get("page", 0),
                "score": r.get("score", 0.0),
                "matched_concepts": r.get("matched_concepts", []),
                "_sources": {"graph"},
            }

    # Build final list, drop internal _sources key
    final = []
    for item in merged.values():
        final.append({
            "text": item["text"],
            "page": item["page"],
            "score": round(item["score"], 4),
            "matched_concepts": item["matched_concepts"],
        })

    final.sort(key=lambda x: x["score"], reverse=True)
    return final[: k + 2]


# ---------------------------------------------------------------------------
# 4. export_graph_html
MAX_RENDER_NODES = 80


def export_graph_html(
    graph: nx.Graph,
    output_path: str = "graph.html",
) -> str:
    """
    Render the concept graph as an interactive HTML file using pyvis.

    - Limits rendering to the top MAX_RENDER_NODES nodes by degree centrality
      to prevent browser freezes on large graphs.
    - Disables physics simulation after initial stabilization so the browser
      stays responsive.

    Returns the HTML string (also writes to output_path).
    """
    # ── FIX 1: Limit to top N nodes by degree ────────────────────────────
    if graph.number_of_nodes() > MAX_RENDER_NODES:
        top_nodes = sorted(
            graph.nodes,
            key=lambda n: graph.degree(n),
            reverse=True,
        )[:MAX_RENDER_NODES]
        render_graph = graph.subgraph(top_nodes).copy()
    else:
        render_graph = graph

    # ── Build pyvis network ──────────────────────────────────────────────
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=False,
    )

    # FIX 2: Barnes-Hut layout with stabilization
    net.barnes_hut(
        spring_length=120,
        spring_strength=0.02,
        damping=0.09,
    )

    # Degree centrality for node sizing (computed on render_graph)
    centrality = nx.degree_centrality(render_graph)

    # Max edge weight for normalization
    max_weight = max(
        (d.get("weight", 1) for _, _, d in render_graph.edges(data=True)),
        default=1,
    )

    # Add nodes
    for node, data in render_graph.nodes(data=True):
        ntype = data.get("type", "CONCEPT")
        freq = data.get("frequency", 1)
        pages = data.get("pages", [])

        color = NODE_COLORS.get(ntype, DEFAULT_NODE_COLOR)
        size = centrality.get(node, 0) * 40
        size = max(10, min(size, 60))

        label = node.title()
        title = (
            f"<b>{label}</b><br>"
            f"Type: {ntype}<br>"
            f"Frequency: {freq}<br>"
            f"Pages: {', '.join(str(p) for p in sorted(pages))}"
        )

        net.add_node(node, label=label, title=title,
                     color=color, size=size)

    # Add edges
    for src, dst, data in render_graph.edges(data=True):
        weight = data.get("weight", 1)
        width = 1 + (weight / max_weight) * 4 if max_weight > 0 else 1
        net.add_edge(src, dst, value=width,
                     title=f"Co-occurrences: {weight}")

    # Configure stabilization
    net.set_options("""{
        "physics": {
            "enabled": true,
            "stabilization": {
                "enabled": true,
                "iterations": 200,
                "fit": true
            }
        }
    }""")

    # Write to file
    net.save_graph(output_path)

    # ── FIX 2b: Inject JS to kill physics after stabilization ────────────
    with open(output_path, "r", encoding="utf-8") as f:
        html_string = f.read()

    # Insert script right before </body>
    disable_physics_js = """
    <script>
      if (typeof network !== 'undefined') {
        network.once("stabilized", function() {
          network.setOptions({ physics: { enabled: false } });
        });
      }
    </script>
    """
    html_string = html_string.replace("</body>", disable_physics_js + "</body>")

    # Write back the modified HTML
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_string)

    return html_string
