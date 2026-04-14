"""
Book GPT -- Flask Backend
========================
API server for the Graph-Augmented RAG application.
All heavy models (embedder, spaCy, ChromaDB) are loaded once at startup.
"""

import os
import re
import io
import uuid
import json
import tempfile
import logging
import threading
from datetime import datetime

from flask import (
    Flask, render_template, request, jsonify, Response, stream_with_context,
    session, send_file,
)
from flask_session import Session
from werkzeug.utils import secure_filename

from rag.ingestor import (
    ingest, get_embedder, release_embedder, get_chroma_client,
    EMBEDDING_MODELS,
)
from rag.retriever import retrieve, timed_retrieve
from rag.generator import stream_generate
from rag.concept_graph import build_graph, combined_retrieve, export_graph_html
from rag.parsers import ALLOWED_EXTENSIONS
from rag.report_generator import generate_report
from rag.visualizer import (
    plot_umap,
    plot_similarity_heatmap,
    plot_relevance_bars,
    plot_token_density,
    plot_chunk_filtering,
    plot_model_comparison,
)

# -- Logging ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
logger = logging.getLogger("book-gpt")

# -- Flask app ----------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB max upload

# -- Upload directory ---------------------------------------------------------
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -- Session config (server-side filesystem) ----------------------------------
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "book-gpt-secret-key-change-me")
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = os.path.join(tempfile.gettempdir(), "book_gpt_sessions")
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True
Session(app)

# -- Constants ----------------------------------------------------------------
MAX_HISTORY_ENTRIES = 8   # 4 exchanges = 4 user + 4 assistant
MAX_PREV_CONTEXTS = 3     # Keep last 3 retrieved chunk sets

# -- In-memory state ----------------------------------------------------------
jobs = {}          # {job_id: {status, num_pages, num_chunks, num_graph_nodes, error}}
book_state = {     # Persists the current book's data
    "ingest_result": None,
    "collection_name": None,       # base name (without model suffix)
    "graph": None,
    "graph_html": None,
    "model": "phi3",               # LLM model
    "embed_model": "balanced",     # current embedding model
    "indexed_models": [],          # list of embed models that have been indexed
    "qa_pairs": [],                # Q&A pairs for report export (stored server-side)
}


# ============================================================================
# ROUTES
# ============================================================================

@app.route("/")
def index():
    """Serve the single-page app."""
    return render_template("index.html")


# -- POST /ingest -------------------------------------------------------------
def _run_ingestion(job_id: str, file_path: str, collection_name: str,
                   embed_model: str):
    """Background worker for file ingestion."""
    logger.info("Ingestion worker started — file_path=%s, exists=%s",
                file_path, os.path.exists(file_path))

    if not os.path.exists(file_path):
        logger.error("File does not exist at path: %s", file_path)
        jobs[job_id] = {"status": "error",
                        "error": f"File not found: {file_path}"}
        return

    try:
        result = ingest(file_path, collection_name, embed_model=embed_model)
        graph = build_graph(result["chunks"])
        graph_html = export_graph_html(graph, output_path="graph.html")

        book_state["ingest_result"] = result
        book_state["collection_name"] = collection_name
        book_state["graph"] = graph
        book_state["graph_html"] = graph_html
        book_state["embed_model"] = embed_model

        # Track which models have been indexed
        if embed_model not in book_state["indexed_models"]:
            book_state["indexed_models"].append(embed_model)

        # Release SentenceTransformer embedders to free RAM for Ollama LLM
        release_embedder()

        jobs[job_id] = {
            "status": "done",
            "num_pages": result["num_pages"],
            "num_chunks": result["num_chunks"],
            "num_graph_nodes": graph.number_of_nodes(),
            "num_graph_edges": graph.number_of_edges(),
            "filter_stats": result.get("filter_stats"),
            "embed_model": embed_model,
        }
        logger.info("Ingestion complete: %s", jobs[job_id])
    except Exception as e:
        logger.error("Ingestion failed: %s", e, exc_info=True)
        jobs[job_id] = {"status": "error", "error": str(e)}
    finally:
        try:
            os.unlink(file_path)
            logger.info("Cleaned up uploaded file: %s", file_path)
        except OSError:
            pass


@app.route("/ingest", methods=["POST"])
def ingest_route():
    """Accept file upload (PDF, EPUB, TXT, DOCX, MD) and start background ingestion."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected"}), 400

    original_name = file.filename
    ext = os.path.splitext(original_name)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return jsonify({"error": f"Unsupported format '{ext}'. Supported: {supported}"}), 400

    # Save uploaded file to the uploads/ directory with a unique safe name
    safe_name = secure_filename(original_name)
    if not safe_name:
        safe_name = f"upload_{uuid.uuid4().hex[:8]}{ext}"
    unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    file.save(file_path)
    logger.info("File saved: %s (size=%d bytes, exists=%s)",
                file_path, os.path.getsize(file_path), os.path.exists(file_path))

    # Build collection name from the original filename (sanitised)
    base_name = os.path.splitext(original_name)[0].replace(" ", "_").lower()
    collection_name = re.sub(r"[^a-zA-Z0-9._-]", "", base_name)
    collection_name = collection_name.strip("._-")[:50] or "book"

    # LLM model
    model = request.form.get("model", "phi3")
    book_state["model"] = model

    # Embedding model
    embed_model = request.form.get("embed_model", "balanced")
    if embed_model not in EMBEDDING_MODELS:
        embed_model = "balanced"

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "processing"}

    # Clear conversation history when a new book is ingested
    session.pop("history", None)
    session.pop("retrieved_contexts", None)
    session.pop("qa_pairs", None)

    thread = threading.Thread(
        target=_run_ingestion,
        args=(job_id, file_path, collection_name, embed_model),
        daemon=True,
    )
    thread.start()

    return jsonify({"status": "processing", "job_id": job_id})


# -- GET /ingest/status/<job_id> ----------------------------------------------
@app.route("/ingest/status/<job_id>")
def ingest_status(job_id):
    """Poll ingestion status."""
    job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job ID"}), 404
    return jsonify(job)


# -- POST /reset --------------------------------------------------------------
@app.route("/reset", methods=["POST"])
def reset_route():
    """Clear all server-side state so a new book can be ingested."""
    # Delete ALL ChromaDB collections for this book
    base = book_state.get("collection_name")
    if base:
        client = get_chroma_client()
        for m in list(book_state.get("indexed_models", [])):
            coll_name = f"{base}_{m}"
            try:
                client.delete_collection(name=coll_name)
                logger.info("Deleted ChromaDB collection: %s", coll_name)
            except Exception as e:
                logger.warning("Could not delete collection '%s': %s", coll_name, e)

    # Reset in-memory state
    book_state["ingest_result"] = None
    book_state["collection_name"] = None
    book_state["graph"] = None
    book_state["graph_html"] = None
    book_state["indexed_models"] = []
    book_state["qa_pairs"] = []

    # Clear session conversation history
    session.pop("history", None)
    session.pop("retrieved_contexts", None)

    jobs.clear()
    logger.info("Server state reset -- ready for a new book.")
    return jsonify({"status": "ok"})


# -- POST /clear_history ------------------------------------------------------
@app.route("/clear_history", methods=["POST"])
def clear_history():
    """Reset conversation history without clearing the ingested book."""
    session.pop("history", None)
    session.pop("retrieved_contexts", None)
    session.pop("qa_pairs", None)
    logger.info("Conversation history cleared.")
    return jsonify({"status": "ok"})


# -- POST /export_report ------------------------------------------------------
@app.route("/export_report", methods=["POST"])
def export_report_route():
    """Generate and return a PDF report of the current Q&A session."""
    qa_pairs = book_state.get("qa_pairs", [])
    if not qa_pairs:
        return jsonify({"error": "No questions asked yet. Ask a question in the chat first."}), 400

    # Determine book title from collection name or fallback
    collection_name = book_state.get("collection_name", "Untitled Book")
    book_title = collection_name.replace("_", " ").title()

    session_data = {
        "book_title": book_title,
        "questions_and_answers": qa_pairs,
    }

    pdf_bytes = generate_report(session_data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"book_gpt_report_{timestamp}.pdf"

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


# -- POST /query --------------------------------------------------------------
@app.route("/query", methods=["POST"])
def query_route():
    """Stream LLM answer as SSE (Server-Sent Events)."""
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    use_graph = data.get("use_graph", True)
    k = int(data.get("k", 5))
    model = data.get("model", book_state.get("model", "phi3"))
    embed_model = data.get("embed_model", book_state.get("embed_model", "balanced"))

    if not query:
        return jsonify({"error": "Empty query"}), 400
    if book_state["ingest_result"] is None:
        return jsonify({"error": "No book ingested yet"}), 400

    # Determine which collection to query
    base_name = book_state["collection_name"]
    collection_name = f"{base_name}_{embed_model}"

    # Check if the requested embed model has been indexed
    if embed_model not in book_state.get("indexed_models", []):
        return jsonify({"error": f"Book not indexed with '{embed_model}' model. Please re-ingest."}), 400

    graph = book_state["graph"]
    all_chunks = book_state["ingest_result"]["chunks"]

    # Retrieve
    if use_graph and graph is not None:
        retrieved = combined_retrieve(query, collection_name, graph, all_chunks,
                                      k=k, embed_model=embed_model)
        source_label = "Vector + Graph"
    else:
        retrieved = retrieve(query, collection_name, k=k, embed_model=embed_model)
        source_label = "Vector only"

    # Release embedder before LLM call
    release_embedder()

    # -- Conversation memory --------------------------------------------------
    history = session.setdefault("history", [])
    retrieved_contexts = session.setdefault("retrieved_contexts", [])

    # Build previous-context summary for the LLM (last 3 chunk sets)
    prev_context_summary = ""
    if retrieved_contexts:
        parts = []
        for i, ctx in enumerate(retrieved_contexts[-MAX_PREV_CONTEXTS:], 1):
            chunk_texts = [c.get("text", "")[:200] for c in ctx[:3]]
            parts.append(
                f"[Previous retrieval {i}]: " + " | ".join(chunk_texts)
            )
        prev_context_summary = "\n".join(parts)

    # Take last MAX_HISTORY_ENTRIES messages (4 exchanges)
    recent_history = history[-MAX_HISTORY_ENTRIES:]

    # If we have previous context, inject it as a system-level hint
    context_hint_msg = []
    if prev_context_summary:
        context_hint_msg = [{
            "role": "system",
            "content": (
                "Previously retrieved passages (for reference only, "
                "prioritize current context):\n" + prev_context_summary
            ),
        }]

    # Full history to pass to the generator:
    # context_hint (optional) + recent conversation exchanges
    full_history = context_hint_msg + recent_history

    # Save retrieved chunks to session for future context
    retrieved_contexts.append([
        {"text": c["text"][:400], "page": c.get("page", 0)}
        for c in retrieved
    ])
    # Keep only last MAX_PREV_CONTEXTS sets
    if len(retrieved_contexts) > MAX_PREV_CONTEXTS:
        session["retrieved_contexts"] = retrieved_contexts[-MAX_PREV_CONTEXTS:]

    def generate_sse():
        # First event: sources metadata
        sources_data = []
        for chunk in retrieved:
            sources_data.append({
                "text": chunk["text"][:400],
                "page": chunk.get("page", 0),
                "score": round(chunk.get("score", 0), 4),
                "matched_concepts": chunk.get("matched_concepts", []),
            })
        yield f"data: {json.dumps({'type': 'sources', 'label': source_label, 'chunks': sources_data})}\n\n"

        # Stream tokens, collecting full response
        full_response = []
        for token in stream_generate(
            query, retrieved, model=model, history=full_history
        ):
            full_response.append(token)
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        # After streaming completes, save to conversation history
        assistant_text = "".join(full_response)
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": assistant_text})

        # Trim to last MAX_HISTORY_ENTRIES
        if len(history) > MAX_HISTORY_ENTRIES:
            session["history"] = history[-MAX_HISTORY_ENTRIES:]

        # Save QA pair + sources for report export (server-side, not session)
        book_state.setdefault("qa_pairs", []).append({
            "question": query,
            "answer": assistant_text,
            "sources": sources_data,
        })

        session.modified = True

        # End event
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(
        stream_with_context(generate_sse()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# -- GET /compare_models ------------------------------------------------------
@app.route("/compare_models")
def compare_models_route():
    """
    Run the same query through all indexed embedding models and return
    comparison data (top-3 chunks + latency per model) plus a Plotly chart.
    """
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify({"error": "Missing ?query= parameter"}), 400

    base_name = book_state.get("collection_name")
    if not base_name:
        return jsonify({"error": "No book ingested"}), 400

    indexed = book_state.get("indexed_models", [])
    if not indexed:
        return jsonify({"error": "No models indexed"}), 400

    results = {}
    for model_name in indexed:
        coll = f"{base_name}_{model_name}"
        try:
            data = timed_retrieve(query, coll, k=3, embed_model=model_name)
            results[model_name] = data
        except Exception as e:
            logger.warning("Compare failed for %s: %s", model_name, e)
            results[model_name] = {"chunks": [], "latency_ms": 0, "error": str(e)}

    # Release embedders after comparison
    release_embedder()

    # Build Plotly chart
    try:
        fig = plot_model_comparison(query, results)
        chart_json = json.loads(fig.to_json())
    except Exception as e:
        logger.warning("Comparison chart failed: %s", e)
        chart_json = None

    return jsonify({
        "query": query,
        "results": {
            name: {
                "chunks": [
                    {"text": c["text"][:300], "page": c.get("page", "?"), "score": c["score"]}
                    for c in d.get("chunks", [])
                ],
                "latency_ms": d.get("latency_ms", 0),
            }
            for name, d in results.items()
        },
        "chart": chart_json,
        "indexed_models": indexed,
    })


# -- GET /graph ---------------------------------------------------------------
@app.route("/graph")
def graph_route():
    """Return the pyvis interactive graph HTML."""
    if book_state["graph_html"] is None:
        return "<p style='color:#888;text-align:center;padding:40px;'>Ingest a book first.</p>"
    return book_state["graph_html"]


# -- GET /graph/stats ---------------------------------------------------------
@app.route("/graph/stats")
def graph_stats():
    """Return graph metrics as JSON."""
    import networkx as nx
    g = book_state.get("graph")
    if g is None:
        return jsonify({"nodes": 0, "edges": 0, "top_concepts": []})

    centrality = nx.degree_centrality(g)
    top5 = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]

    return jsonify({
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "top_concepts": [
            {
                "name": name.title(),
                "centrality": round(score, 4),
                "type": g.nodes[name].get("type", "CONCEPT"),
                "frequency": g.nodes[name].get("frequency", 0),
            }
            for name, score in top5
        ],
    })


# -- GET /analytics -----------------------------------------------------------
@app.route("/analytics")
def analytics_route():
    """Return Plotly chart JSON for client-side rendering."""
    result = book_state.get("ingest_result")
    if result is None:
        return jsonify({"error": "No book ingested"}), 400

    all_chunks = result["chunks"]
    all_embeddings = result["embeddings"]

    charts = {}

    # Token density (always available)
    try:
        fig = plot_token_density(all_chunks)
        charts["token_density"] = json.loads(fig.to_json())
    except Exception as e:
        logger.warning("Token density chart failed: %s", e)



    # Chunk filtering chart (always available after ingestion)
    filter_stats = result.get("filter_stats")
    if filter_stats:
        try:
            fig = plot_chunk_filtering(filter_stats)
            charts["chunk_filtering"] = json.loads(fig.to_json())
        except Exception as e:
            logger.warning("Chunk filtering chart failed: %s", e)

    return jsonify(charts)


# -- GET /analytics/query -----------------------------------------------------
@app.route("/analytics/query")
def analytics_query_route():
    """Return query-specific analytics (relevance bars + heatmap)."""
    return jsonify({"info": "Use POST /analytics/query with chunks data"})


# -- GET /status --------------------------------------------------------------
@app.route("/status")
def status_route():
    """System status for the sidebar."""
    result = book_state.get("ingest_result")
    graph = book_state.get("graph")

    # Check Ollama
    ollama_ok = False
    try:
        import urllib.request
        req = urllib.request.urlopen("http://localhost:11434", timeout=2)
        ollama_ok = req.status == 200
    except Exception:
        pass

    return jsonify({
        "ollama": ollama_ok,
        "model": book_state.get("model", "phi3"),
        "embed_model": book_state.get("embed_model", "balanced"),
        "indexed_models": book_state.get("indexed_models", []),
        "ingested": result is not None,
        "num_chunks": result["num_chunks"] if result else 0,
        "num_pages": result["num_pages"] if result else 0,
        "graph_nodes": graph.number_of_nodes() if graph else 0,
        "graph_edges": graph.number_of_edges() if graph else 0,
    })


# ============================================================================
# STARTUP
# ============================================================================
def _kill_stale_server(port: int = 5000):
    """Kill any existing process listening on the given port (Windows only)."""
    import subprocess, signal
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                pid = int(parts[-1])
                if pid > 0 and pid != os.getpid():
                    logger.warning(
                        "Port %d is occupied by PID %d — killing stale process...",
                        port, pid,
                    )
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except OSError:
                        pass
                    import time
                    time.sleep(1)
    except Exception as e:
        logger.debug("Port cleanup skipped: %s", e)


if __name__ == "__main__":
    _kill_stale_server(5000)
    logger.info("Starting Book GPT server...")
    logger.info("Pre-loading ChromaDB client...")
    get_chroma_client()
    logger.info("Server ready at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
