---
title: BookGPT
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 📚 Book GPT — Local Graph-Augmented RAG

**Ask questions about any book using a hybrid retrieval engine that combines vector search with concept-graph traversal — fully local, no APIs, no data leaves your machine.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/LLM-Ollama-black?logo=ollama)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧠 What Makes This Novel

Most RAG (Retrieval-Augmented Generation) systems rely on **flat vector search** — they embed chunks, embed the query, and return the top-k nearest neighbors. This works well for direct lookups ("What is X?") but breaks down when the answer requires **connecting concepts across distant parts of a book** ("How does character A's decision in chapter 3 affect the outcome in chapter 9?").

**Book GPT adds a concept co-occurrence graph** built with spaCy NER and NetworkX:

| Approach | How It Works | Strength |
|---|---|---|
| **Vector Search** | Cosine similarity between query and chunk embeddings | Great for semantic matches |
| **Graph Search** | 2-hop BFS from query concepts through a co-occurrence graph | Finds structurally related chunks that vectors miss |
| **Book GPT (Hybrid)** | Merges both result sets, boosts chunks found by both methods | Best of both worlds |

The concept graph connects entities (people, places, organizations) and noun phrases that appear together in the same chunk. When a query mentions "trade policy," the graph can surface chunks about "tariffs," "WTO," or "economic sanctions" even if they aren't semantically similar in embedding space — because they co-occur with overlapping concepts in the book.

> Inspired by [Microsoft's GraphRAG](https://github.com/microsoft/graphrag) paper, but built **entirely locally** with open-source tools. No cloud APIs, no OpenAI, no data exfiltration.

---

## 🏗 Architecture

```
┌─────────────────────── INGESTION PIPELINE ───────────────────────┐
│                                                                  │
│   PDF ──► PyMuPDF ──► Page Text ──► LangChain Splitter ──► Chunks│
│              Parse         │           (600 char, 80 overlap)    │
│                            │                                     │
│                            ▼                                     │
│                   BGE-large-en-v1.5                              │
│                    (embed chunks)                                │
│                       │        │                                 │
│                       ▼        ▼                                 │
│                  ChromaDB    spaCy NLP                           │
│                 (vectors)   (NER + noun chunks)                  │
│                                │                                 │
│                                ▼                                 │
│                         NetworkX Graph                           │
│                    (concept co-occurrence)                       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────────── QUERY PIPELINE ───────────────────────────┐
│                                                                  │
│   Query                                                          │
│     │                                                            │
│     ├──► Embed ──► ChromaDB top-k ──────────────┐                │
│     │              (vector search)               │               │
│     │                                            ▼               │
│     └──► spaCy ──► Match Nodes ──► 2-hop BFS ──► Merge + Rank    │
│          (concept    (exact +        (graph       (boost +0.15   │
│           extract)    fuzzy)          search)      if in both)   │
│                                                      │           │
│                                                      ▼           │
│                                                 Ollama LLM       │
│                                                (llama3.1 /       │
│                                                 mistral / phi3)  │
│                                                      │           │
│                                                      ▼           │
│                                            Streamed Answer       │
│                                          with page citations     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Tool | Purpose | Why Chosen |
|---|---|---|
| **PyMuPDF (fitz)** | PDF text extraction | Fast, handles complex layouts, pure Python |
| **LangChain** | Text chunking | `RecursiveCharacterTextSplitter` with paragraph + sentence boundaries |
| **BAAI/bge-large-en-v1.5** | Embedding model | Top-ranking open embedding model, supports normalization |
| **ChromaDB** | Vector store | Lightweight, persistent, no server needed |
| **spaCy (en_core_web_sm)** | NER + noun chunk extraction | Fast CPU inference, accurate entity recognition |
| **NetworkX** | Concept graph | Mature graph library with BFS, centrality metrics |
| **Ollama** | Local LLM inference | Runs llama3.1/mistral/phi3 locally, zero config |
| **Flask** | Web Server | Lightweight backend for the single-page application |
| **Plotly** | Analytics visualizations | Interactive charts, transparent backgrounds |
| **pyvis** | Graph visualization | Interactive HTML network diagrams |
| **UMAP** | Dimensionality reduction | Visualize embedding clusters in 2D |

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running

### Quick Setup

```bash
# Clone the project
cd book-gpt

# Run the setup script
chmod +x setup.sh
./setup.sh
```

The `setup.sh` script handles everything:
1. Installs Python dependencies from `requirements.txt`
2. Downloads the spaCy language model (`en_core_web_sm`)
3. Pulls the default LLM (`llama3.1`) via Ollama

### Manual Setup

If you prefer to install manually:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
ollama pull llama3.1
```

---

## 📖 Usage

### Start the App

```bash
# Make sure Ollama is running
ollama serve

# In another terminal, launch Book GPT (Flask server)
cd book-gpt
python server.py
```

### Workflow

1. **Upload** a PDF book via the sidebar
2. **Click "Ingest Book"** — the app parses, chunks, embeds, and builds the concept graph
3. **Ask questions** in the chat tab — answers stream in with page citations
4. **Explore** the concept graph and analytics tabs

---

## 📊 Visualizations

Book GPT includes **5 interactive visualizations** across two tabs:

### 🕸 Concept Graph Tab

| Visualization | Description |
|---|---|
| **Interactive Concept Graph** | Pyvis network diagram — nodes are concepts (color-coded by type: coral=Person, blue=Org, green=Place, purple=Other), edges are co-occurrence links. Node size scales with degree centrality. Dark background, physics-enabled layout. Hover for details. |

### 📊 Analytics Tab (2×2 grid)

| Plot | What It Shows |
|---|---|
| **Retrieval Relevance Scores** | Horizontal bar chart of chunk scores from the last query. Blue → coral color gradient by relevance. |
| **Retrieved Chunk Similarity** | Cosine similarity heatmap between retrieved chunks. Reveals redundancy or diversity in retrieval. |
| **Token Density Across Pages** | Bar chart of token count per page. Identifies content-heavy vs. sparse pages (images, diagrams). |
| **Semantic Map (UMAP)** | 2D UMAP projection of all chunk embeddings, colored by page number. Shows semantic clusters in the book. |

---

## 💬 Sample Queries

Here are example questions demonstrating what **graph-augmented retrieval adds** over plain vector search:

### 1. Cross-chapter reasoning
> **"How does the author's definition of intelligence in chapter 2 relate to the creativity discussion in chapter 7?"**
>
> 🔍 **Vector search** finds chunks near "intelligence" OR "creativity" but may miss the conceptual bridge.
> 🕸 **Graph search** traverses from "intelligence" → co-occurring concepts → "creativity" neighbors, surfacing linking chunks that mention both domains.

### 2. Entity-relationship queries
> **"What role does Dr. Chen play in the research project, and who are her collaborators?"**
>
> 🔍 **Vector search** finds chunks mentioning "Dr. Chen" by semantic similarity.
> 🕸 **Graph search** follows the "dr. chen" node through co-occurrence edges to find all connected person/org nodes — surfacing collaborator mentions even in chapters where Dr. Chen isn't named directly.

### 3. Thematic analysis
> **"What are the economic consequences discussed throughout the book?"**
>
> 🔍 **Vector search** returns chunks closest to "economic consequences" in embedding space.
> 🕸 **Graph search** expands from "economic" through 2 hops, reaching nodes like "unemployment," "gdp," "trade deficit," "inflation" — pulling in thematically related chunks that pure similarity misses.

---

## 🔮 Future Work

- **Hierarchical graph communities** — Use Leiden clustering to detect concept communities, generating chapter-level and book-level summaries (closer to full GraphRAG community reports).
- **Multi-book knowledge base** — Support ingesting multiple books into the same graph, enabling cross-book queries like "Compare how Author A and Author B discuss free will."
- **Adaptive retrieval** — Automatically decide whether a query benefits from graph traversal or pure vector search based on query type classification, reducing latency for simple factual lookups.

---

## 📁 Project Structure

```
book-gpt/
├── server.py               # Flask Server & API (chat, graph, analytics)
├── requirements.txt        # Python dependencies
├── setup.sh                # One-command install script
├── README.md
├── rag/
│   ├── __init__.py
│   ├── ingestor.py         # PDF parse → chunk → embed → ChromaDB
│   ├── retriever.py        # Vector search with cosine scoring
│   ├── generator.py        # Ollama LLM with streaming
│   ├── concept_graph.py    # spaCy NER → NetworkX graph → hybrid retrieval
│   └── visualizer.py       # Plotly charts (UMAP, heatmap, bars, density)
└── chroma_store/           # Persistent vector database (auto-created)
```

---

