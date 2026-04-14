/* ═══════════════════════════════════════════════════════════════════════════
   Book GPT — Client-Side Logic
   Vanilla JS — no frameworks, no jQuery
   ═══════════════════════════════════════════════════════════════════════════ */

// ── DOM refs ────────────────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dropZone     = $("#dropZone");
const fileInput    = $("#fileInput");
const fileName     = $("#fileName");
const ingestBtn    = $("#ingestBtn");
const progressCont = $("#progressContainer");
const progressBar  = $("#progressBar");
const chatMessages = $("#chatMessages");
const chatInput    = $("#chatInput");
const sendBtn      = $("#sendBtn");
const kSlider      = $("#kSlider");
const kValue       = $("#kValue");
const modelSelect  = $("#modelSelect");
const useGraphCb   = $("#useGraph");
const newBookBtn   = $("#newBookBtn");
const clearHistoryBtn = $("#clearHistoryBtn");
const chatHeader   = $("#chatHeader");
const fileTypeBadge = $("#fileTypeBadge");
const exportReportBtn = $("#exportReportBtn");
const reindexBanner = $("#reindexBanner");
const reindexBtn    = $("#reindexBtn");
const compareModelsBtn = $("#compareModelsBtn");
const compareModal  = $("#compareModal");
const closeCompareModal = $("#closeCompareModal");
const compareChartContainer = $("#compareChartContainer");
const compareDetails = $("#compareDetails");

const ALLOWED_EXTENSIONS = [".pdf", ".epub", ".txt", ".docx", ".md"];
const BADGE_COLORS = {
    ".pdf":  "#ff6b6b",
    ".epub": "#9b59b6",
    ".txt":  "#1abc9c",
    ".docx": "#3498db",
    ".md":   "#2ecc71",
};

let selectedFile   = null;
let isStreaming     = false;
let selectedEmbedModel = "balanced";
let indexedModels  = [];
let lastQuery      = "";

// ── Tabs ────────────────────────────────────────────────────────────────────
const tabBtns    = $$(".tab-btn");
const tabPanels  = $$(".tab-panel");
const tabIndicator = $("#tabIndicator");

function activateTab(tabName) {
    tabBtns.forEach(b => b.classList.toggle("active", b.dataset.tab === tabName));
    tabPanels.forEach(p => p.classList.toggle("active", p.id === `tab-${tabName}`));
    updateIndicator();

    // Lazy-load tab content
    if (tabName === "graph")     loadGraph();
    if (tabName === "analytics") {
        loadAnalytics();
        if (analyticsLoaded) {
            setTimeout(() => {
                window.dispatchEvent(new Event("resize"));
            }, 50);
        }
    }
}

function updateIndicator() {
    const active = $(".tab-btn.active");
    if (active && tabIndicator) {
        tabIndicator.style.left   = active.offsetLeft + "px";
        tabIndicator.style.width  = active.offsetWidth + "px";
    }
}

tabBtns.forEach(btn => btn.addEventListener("click", () => activateTab(btn.dataset.tab)));
window.addEventListener("resize", updateIndicator);
requestAnimationFrame(updateIndicator);

// ── Slider ──────────────────────────────────────────────────────────────────
kSlider.addEventListener("input", () => { kValue.textContent = kSlider.value; });

// ── File Upload (Drag & Drop) ───────────────────────────────────────────────
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file && isAllowedFile(file.name)) {
        selectFile(file);
    } else {
        showToast("Unsupported format. Use PDF, EPUB, TXT, DOCX, or MD.", "error");
    }
});

fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) selectFile(fileInput.files[0]);
});

function getFileExtension(name) {
    const i = name.lastIndexOf(".");
    return i >= 0 ? name.slice(i).toLowerCase() : "";
}

function isAllowedFile(name) {
    return ALLOWED_EXTENSIONS.includes(getFileExtension(name));
}

function selectFile(file) {
    selectedFile = file;
    fileName.textContent = file.name;
    ingestBtn.disabled = false;

    // Show color-coded file type badge
    const ext = getFileExtension(file.name);
    const label = ext.replace(".", "").toUpperCase();
    const color = BADGE_COLORS[ext] || "#888";
    fileTypeBadge.textContent = label;
    fileTypeBadge.style.background = color;
    fileTypeBadge.style.display = "inline-block";
}

// ── Ingest ──────────────────────────────────────────────────────────────────
ingestBtn.addEventListener("click", startIngestion);

async function startIngestion() {
    if (!selectedFile) return;

    ingestBtn.disabled = true;
    ingestBtn.textContent = "⏳ Processing…";
    progressCont.style.display = "block";
    progressBar.classList.add("indeterminate");

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("model", modelSelect.value);
    formData.append("embed_model", selectedEmbedModel);

    try {
        const res  = await fetch("/ingest", { method: "POST", body: formData });
        const data = await res.json();

        if (data.error) {
            showToast(data.error, "error");
            resetIngestBtn();
            return;
        }

        // Poll for completion
        pollIngestion(data.job_id);
    } catch (err) {
        showToast("Upload failed: " + err.message, "error");
        resetIngestBtn();
    }
}

function pollIngestion(jobId) {
    const interval = setInterval(async () => {
        try {
            const res  = await fetch(`/ingest/status/${jobId}`);
            const data = await res.json();

            if (data.status === "done") {
                clearInterval(interval);
                progressBar.classList.remove("indeterminate");
                progressBar.style.width = "100%";

                showToast(
                    `✅ ${data.num_pages} pages · ${data.num_chunks} chunks · ${data.num_graph_nodes} graph nodes`,
                    "success"
                );

                chatInput.disabled = false;
                sendBtn.disabled   = false;
                ingestBtn.textContent = "✅ Ingested";
                newBookBtn.style.display = "block";
                chatHeader.style.display = "flex";

                // Reset graph & analytics so they reload for the new book
                graphLoaded = false;
                analyticsLoaded = false;

                // Clear welcome message
                const welcome = $(".welcome-message");
                if (welcome) welcome.remove();

                // Refresh status
                fetchStatus();

                setTimeout(() => {
                    progressCont.style.display = "none";
                    progressBar.style.width = "0%";
                }, 2000);

            } else if (data.status === "error") {
                clearInterval(interval);
                showToast("Ingestion failed: " + (data.error || "Unknown error"), "error");
                resetIngestBtn();
            }
        } catch (err) {
            clearInterval(interval);
            showToast("Polling failed: " + err.message, "error");
            resetIngestBtn();
        }
    }, 2000);
}

function resetIngestBtn() {
    ingestBtn.disabled = selectedFile === null;
    ingestBtn.textContent = "🚀 Ingest Book";
    progressCont.style.display = "none";
    progressBar.classList.remove("indeterminate");
    progressBar.style.width = "0%";
}

// ── Chat ────────────────────────────────────────────────────────────────────
chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !isStreaming) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener("click", () => {
    if (!isStreaming) sendMessage();
});

function sendMessage() {
    const query = chatInput.value.trim();
    if (!query) return;

    // Add user message
    appendMessage("user", query);
    chatInput.value = "";
    lastQuery = query;

    // Start streaming response
    streamResponse(query);
}

function appendMessage(role, content, extras = {}) {
    const row  = document.createElement("div");
    row.className = `message-row ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "👤" : "🤖";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    if (extras.label) {
        const lbl = document.createElement("div");
        lbl.className = "retrieval-label";
        lbl.textContent = extras.label;
        bubble.appendChild(lbl);
    }

    const textEl = document.createElement("div");
    textEl.className = "message-text";
    textEl.innerHTML = formatMarkdown(content);
    bubble.appendChild(textEl);

    row.appendChild(avatar);
    row.appendChild(bubble);
    chatMessages.appendChild(row);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return { row, bubble, textEl };
}

function streamResponse(query) {
    isStreaming = true;
    sendBtn.disabled   = true;
    chatInput.disabled = true;

    // Create assistant bubble with typing indicator
    const { row, bubble, textEl } = appendMessage("assistant", "");
    textEl.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';

    let fullText  = "";
    let sources   = [];
    let label     = "";

    const body = JSON.stringify({
        query,
        use_graph: useGraphCb.checked,
        k: parseInt(kSlider.value),
        model: modelSelect.value,
        embed_model: selectedEmbedModel,
    });

    fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
    }).then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        function read() {
            reader.read().then(({ done, value }) => {
                if (done) {
                    finishStream();
                    return;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop(); // keep incomplete line

                for (const line of lines) {
                    if (!line.startsWith("data: ")) continue;
                    const jsonStr = line.slice(6);
                    try {
                        const evt = JSON.parse(jsonStr);

                        if (evt.type === "sources") {
                            sources = evt.chunks || [];
                            label = evt.label || "";
                        } else if (evt.type === "token") {
                            if (fullText === "") {
                                textEl.innerHTML = ""; // clear typing dots
                                // Add retrieval label
                                if (label) {
                                    const lbl = document.createElement("div");
                                    lbl.className = "retrieval-label";
                                    lbl.textContent = `${label} · ${sources.length} chunks`;
                                    bubble.insertBefore(lbl, textEl);
                                }
                            }
                            fullText += evt.content;
                            textEl.innerHTML = formatMarkdown(fullText);
                            chatMessages.scrollTop = chatMessages.scrollHeight;
                        } else if (evt.type === "done") {
                            finishStream();
                            return;
                        }
                    } catch (e) {
                        // skip malformed JSON
                    }
                }

                read();
            }).catch(err => {
                textEl.innerHTML = formatMarkdown(fullText || "⚠️ Stream error: " + err.message);
                finishStream();
            });
        }

        read();
    }).catch(err => {
        textEl.innerHTML = "⚠️ Connection error: " + err.message;
        finishStream();
    });

    function finishStream() {
        isStreaming = false;
        sendBtn.disabled   = false;
        chatInput.disabled = false;
        chatInput.focus();

        // Show export button now that we have at least 1 QA
        if (exportReportBtn) exportReportBtn.style.display = "inline-flex";
        if (compareModelsBtn) compareModelsBtn.disabled = false;

        // Add sources accordion
        if (sources.length > 0) {
            const toggle = document.createElement("button");
            toggle.className = "sources-toggle";
            toggle.innerHTML = "📎 Sources ▸";

            const content = document.createElement("div");
            content.className = "sources-content";
            content.innerHTML = sources.map((s, i) =>
                `<div class="source-item">` +
                `<div class="source-meta">Page ${s.page} · score ${s.score.toFixed(3)}</div>` +
                (s.matched_concepts && s.matched_concepts.length > 0
                    ? `<div class="source-concepts">Concepts: ${s.matched_concepts.join(", ")}</div>`
                    : "") +
                `<div>${escapeHtml(s.text)}</div>` +
                `</div>`
            ).join("");

            toggle.addEventListener("click", () => {
                content.classList.toggle("open");
                toggle.innerHTML = content.classList.contains("open")
                    ? "📎 Sources ▾" : "📎 Sources ▸";
            });

            bubble.appendChild(toggle);
            bubble.appendChild(content);
        }
    }
}

// ── Graph Tab ───────────────────────────────────────────────────────────────
let graphLoaded = false;

function loadGraph() {
    if (graphLoaded) return;
    const iframe = $("#graphFrame");
    iframe.src = "/graph";
    graphLoaded = true;

    // Load stats
    fetch("/graph/stats")
        .then(r => r.json())
        .then(data => {
            $("#statNodes").textContent = data.nodes || "—";
            $("#statEdges").textContent = data.edges || "—";

            if (data.top_concepts && data.top_concepts.length > 0) {
                const html = `
                    <table class="concept-table">
                        <thead><tr><th>Concept</th><th>Type</th><th>Centrality</th><th>Freq</th></tr></thead>
                        <tbody>
                            ${data.top_concepts.map(c =>
                                `<tr><td>${escapeHtml(c.name)}</td><td>${c.type}</td>` +
                                `<td>${c.centrality}</td><td>${c.frequency}</td></tr>`
                            ).join("")}
                        </tbody>
                    </table>`;
                $("#topConcepts").innerHTML = html;
            }
        })
        .catch(() => {});
}

// ── Analytics Tab ───────────────────────────────────────────────────────────
let analyticsLoaded = false;

function loadAnalytics() {
    if (analyticsLoaded) return;

    fetch("/analytics")
        .then(r => r.json())
        .then(data => {
            if (data.error) return;

            const plotConfig = {
                responsive: true,
                displayModeBar: false,
            };

            // Only override colors — preserve margin/height/legend from server
            function applyDarkTheme(layout) {
                layout.paper_bgcolor = "rgba(0,0,0,0)";
                layout.plot_bgcolor = "rgba(0,0,0,0)";
                if (!layout.font) layout.font = {};
                layout.font.color = "#999";
                layout.font.family = "Inter";
            }

            // Helper: prepare a container div with explicit height for Plotly
            function prepareContainer(parentEl, height) {
                parentEl.innerHTML = "";
                const wrapper = document.createElement("div");
                wrapper.style.width = "100%";
                wrapper.style.minHeight = (height || 400) + "px";
                wrapper.style.position = "relative";
                parentEl.appendChild(wrapper);
                return wrapper;
            }

            // Chunk filtering chart
            if (data.chunk_filtering) {
                const cf = data.chunk_filtering;
                applyDarkTheme(cf.layout);
                const parentEl = $("#chart-chunk-filtering");
                const chartHeight = cf.layout.height || 260;
                const wrapper = prepareContainer(parentEl, chartHeight);
                Plotly.newPlot(wrapper, cf.data, cf.layout, plotConfig);

                // Extract stats for the summary header
                const title = cf.layout.title;
                if (title) {
                    const titleText = typeof title === "object" ? title.text : title;
                    const match = titleText.match(/(\d+)\/(\d+) accepted \(([0-9.]+)%\)/);
                    if (match) {
                        const summaryEl = $("#qualitySummary");
                        summaryEl.innerHTML = `✅ <strong>${match[1]} of ${match[2]}</strong> chunks accepted (<strong>${match[3]}%</strong> quality rate)`;
                        summaryEl.style.display = "block";
                    }
                }
            }

            // Token density chart
            if (data.token_density) {
                const td = data.token_density;
                applyDarkTheme(td.layout);
                const parentEl = $("#chart-token-density");
                const chartHeight = td.layout.height || 360;
                const wrapper = prepareContainer(parentEl, chartHeight);
                Plotly.newPlot(wrapper, td.data, td.layout, plotConfig);
            }

            analyticsLoaded = true;

            // Force Plotly to recalculate dimensions after layout settles
            setTimeout(() => {
                window.dispatchEvent(new Event("resize"));
            }, 100);
        })
        .catch(() => {});
}

// ── Status polling ──────────────────────────────────────────────────────────
function fetchStatus() {
    fetch("/status")
        .then(r => r.json())
        .then(data => {
            const ollamaDot = $("#ollamaDot");
            const ollamaVal = $("#ollamaStatus");
            if (data.ollama) {
                ollamaDot.className = "status-dot";
                ollamaVal.textContent = data.model || "connected";
            } else {
                ollamaDot.className = "status-dot error";
                ollamaVal.textContent = "offline";
            }

            const chunkVal = $("#chunkStatus");
            chunkVal.textContent = data.ingested
                ? `${data.num_chunks} chunks`
                : "—";

            const graphVal = $("#graphStatus");
            graphVal.textContent = data.graph_nodes > 0
                ? `${data.graph_nodes} nodes, ${data.graph_edges} edges`
                : "—";

            // Update status dots
            const dots = $$(".status-dot");
            if (data.ingested && dots[1]) dots[1].className = "status-dot";
            if (data.graph_nodes > 0 && dots[2]) dots[2].className = "status-dot";

            // Track indexed models
            if (data.indexed_models) indexedModels = data.indexed_models;
        })
        .catch(() => {});
}

// Poll status every 10 seconds
fetchStatus();
setInterval(fetchStatus, 10000);

// ── New Book (Reset) ────────────────────────────────────────────────────────
newBookBtn.addEventListener("click", async () => {
    newBookBtn.disabled = true;
    newBookBtn.textContent = "⏳ Resetting…";

    try {
        const res = await fetch("/reset", { method: "POST" });
        const data = await res.json();

        if (data.status === "ok") {
            // Reset UI
            selectedFile = null;
            fileName.textContent = "";
            fileInput.value = "";
            fileTypeBadge.style.display = "none";
            fileTypeBadge.textContent = "";
            ingestBtn.disabled = true;
            ingestBtn.textContent = "🚀 Ingest Book";
            chatInput.disabled = true;
            sendBtn.disabled = true;
            newBookBtn.style.display = "none";
            chatHeader.style.display = "none";
            if (exportReportBtn) exportReportBtn.style.display = "none";
            if (reindexBanner) reindexBanner.style.display = "none";
            indexedModels = [];
            lastQuery = "";
            if (compareModelsBtn) compareModelsBtn.disabled = true;

            // Clear chat
            chatMessages.innerHTML = `
                <div class="welcome-message">
                    <div class="welcome-icon">📚</div>
                    <h2>Welcome to Book GPT</h2>
                    <p>Upload a PDF and ingest it to start asking questions.<br>
                    Your answers are powered by local LLMs with concept-graph retrieval.</p>
                </div>`;

            // Reset graph & analytics (must happen before DOM access)
            graphLoaded = false;
            analyticsLoaded = false;
            $("#graphFrame").src = "about:blank";
            $("#statNodes").textContent = "—";
            $("#statEdges").textContent = "—";
            $("#topConcepts").innerHTML = "";

            const tokenDensity = $("#chart-token-density");
            if (tokenDensity) tokenDensity.innerHTML = '<div class="chart-placeholder">Ingest a book and run a query to see analytics.</div>';
            const chunkFiltering = $("#chart-chunk-filtering");
            if (chunkFiltering) chunkFiltering.innerHTML = '<div class="chart-placeholder">Chunk quality filtering results will appear here.</div>';
            const qualitySummary = $("#qualitySummary");
            if (qualitySummary) { qualitySummary.style.display = "none"; qualitySummary.innerHTML = ""; }

            // Reset status panel
            $("#chunkStatus").textContent = "—";
            $("#graphStatus").textContent = "—";

            showToast("Ready for a new book!", "success");
        } else {
            showToast("Reset failed.", "error");
        }
    } catch (err) {
        showToast("Reset error: " + err.message, "error");
    } finally {
        newBookBtn.disabled = false;
        newBookBtn.textContent = "📖 New Book";
    }
});

// ── Clear Conversation History ──────────────────────────────────────────────
clearHistoryBtn.addEventListener("click", async () => {
    clearHistoryBtn.disabled = true;

    try {
        const res = await fetch("/clear_history", { method: "POST" });
        const data = await res.json();

        if (data.status === "ok") {
            // Add session separator with timestamp
            addSessionSeparator();
            if (exportReportBtn) exportReportBtn.style.display = "none";
            showToast("Conversation history cleared.", "info");
        } else {
            showToast("Failed to clear history.", "error");
        }
    } catch (err) {
        showToast("Error: " + err.message, "error");
    } finally {
        clearHistoryBtn.disabled = false;
    }
});

function addSessionSeparator() {
    const sep = document.createElement("div");
    sep.className = "session-separator";
    const now = new Date();
    const time = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    sep.textContent = `New session · ${time}`;
    chatMessages.appendChild(sep);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ── Export Report ───────────────────────────────────────────────────────────
if (exportReportBtn) {
    exportReportBtn.addEventListener("click", async () => {
        const originalText = exportReportBtn.innerHTML;
        exportReportBtn.disabled = true;
        exportReportBtn.innerHTML = `
            <svg class="spin" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
            </svg>
            Generating…`;

        try {
            const res = await fetch("/export_report", { method: "POST" });

            if (!res.ok) {
                const err = await res.json().catch(() => ({ error: "Export failed" }));
                showToast(err.error || "Export failed.", "error");
                return;
            }

            // Download the PDF blob
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            // Try to get filename from Content-Disposition, fallback
            const cd = res.headers.get("Content-Disposition") || "";
            const match = cd.match(/filename="?([^"]+)"?/);
            a.download = match ? match[1] : "book_gpt_report.pdf";
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);

            showToast("Report downloaded!", "success");
        } catch (err) {
            showToast("Export error: " + err.message, "error");
        } finally {
            exportReportBtn.disabled = false;
            exportReportBtn.innerHTML = originalText;
        }
    });
}

// -- Embedding Model Switcher ------------------------------------------------
const embedRadios = document.querySelectorAll('input[name="embedModel"]');
embedRadios.forEach(radio => {
    radio.addEventListener("change", () => {
        selectedEmbedModel = radio.value;
        // Show/hide reindex banner
        if (indexedModels.length > 0 && !indexedModels.includes(selectedEmbedModel)) {
            if (reindexBanner) reindexBanner.style.display = "flex";
        } else {
            if (reindexBanner) reindexBanner.style.display = "none";
        }
    });
});

// Re-index with the currently selected embed model
if (reindexBtn) {
    reindexBtn.addEventListener("click", () => {
        if (!selectedFile && !ingestBtn.textContent.includes("Ingested")) {
            showToast("Upload a file first.", "error");
            return;
        }
        if (reindexBanner) reindexBanner.style.display = "none";
        showToast(`Re-indexing with ${selectedEmbedModel} model...`, "info");
        startIngestion();
    });
}

// -- Compare Models ----------------------------------------------------------
if (compareModelsBtn) {
    compareModelsBtn.addEventListener("click", async () => {
        const q = lastQuery;
        if (!q) {
            showToast("Run a query first.", "error");
            return;
        }

        compareModelsBtn.disabled = true;
        compareModelsBtn.textContent = "Comparing...";

        try {
            const res = await fetch(`/compare_models?query=${encodeURIComponent(q)}`);
            const data = await res.json();

            if (data.error) {
                showToast(data.error, "error");
                return;
            }

            // Show modal BEFORE plotting so Plotly can calculate dimensions
            if (compareModal) compareModal.style.display = "flex";

            // Render Plotly chart in modal
            if (data.chart && compareChartContainer) {
                compareChartContainer.innerHTML = "";
                Plotly.newPlot(compareChartContainer, data.chart.data, data.chart.layout, {
                    responsive: true,
                    displayModeBar: false,
                });
            }

            // Build details table
            if (compareDetails) {
                let html = '<div class="compare-table-wrap">';
                for (const [model, info] of Object.entries(data.results)) {
                    html += `<div class="compare-model-card">`;
                    html += `<h4>${model} <span class="compare-latency">${info.latency_ms.toFixed(0)} ms</span></h4>`;
                    html += `<ol class="compare-chunks">`;
                    for (const c of info.chunks) {
                        html += `<li><span class="compare-score">${c.score.toFixed(3)}</span> p.${c.page} &mdash; ${escapeHtml(c.text.slice(0, 100))}...</li>`;
                    }
                    html += `</ol></div>`;
                }
                html += '</div>';
                compareDetails.innerHTML = html;
            }

        } catch (err) {
            showToast("Compare failed: " + err.message, "error");
        } finally {
            compareModelsBtn.disabled = false;
            compareModelsBtn.textContent = "\uD83D\uDCCA Compare Embedding Models";
        }
    });
}

// Modal close
if (closeCompareModal) {
    closeCompareModal.addEventListener("click", () => {
        if (compareModal) compareModal.style.display = "none";
    });
}
if (compareModal) {
    compareModal.addEventListener("click", (e) => {
        if (e.target === compareModal) compareModal.style.display = "none";
    });
}

// -- Utilities ---------------------------------------------------------------
function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

function formatMarkdown(text) {
    if (!text) return "";
    // Basic markdown: bold, italic, code, line breaks
    return escapeHtml(text)
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.*?)\*/g, "<em>$1</em>")
        .replace(/`(.*?)`/g, "<code>$1</code>")
        .replace(/\n/g, "<br>");
}

function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}
