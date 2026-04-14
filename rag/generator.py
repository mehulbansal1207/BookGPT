import ollama
from ollama._types import ResponseError

SYSTEM_PROMPT = (
    "You are a knowledgeable reading assistant. You MUST synthesize fluent, "
    "natural language answers from the context chunks provided below.\n\n"
    "CRITICAL RULES:\n"
    "1. ALWAYS write your answer as complete, well-formed sentences in your "
    "own words. NEVER copy-paste raw text, table fragments, bullet lists, "
    "or formatting from the context chunks.\n"
    "2. If the context contains tables or structured data, extract the "
    "relevant facts and restate them naturally. For example, if a table "
    "shows 'AWS | 2006', write 'AWS was founded in 2006' — do NOT "
    "reproduce the table.\n"
    "3. If the answer is a single fact like a date or number, embed it in "
    "a full explanatory sentence with surrounding context.\n"
    "4. When the answer spans multiple chunks, reason across all of them "
    "to produce a coherent, unified response.\n"
    "5. Cite page numbers inline like (p.12) when referencing specific "
    "information.\n"
    "6. If the context only partially covers the question, reason from "
    "what IS available rather than refusing. State what is and isn't "
    "covered.\n"
    "7. NEVER fabricate facts not present in the provided context.\n"
    "8. Use the context chunks as EVIDENCE to support your answer — "
    "the chunks are NOT the answer itself. Your job is to interpret "
    "and explain.\n"
    "9. You have access to recent conversation history. Use it to "
    "understand references like 'he', 'it', 'that concept', 'the "
    "previous answer'. Always prioritize the book context over "
    "conversation history for factual claims.\n"
)


def _build_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a single context string.
    Each chunk is prefixed with its page number and separated by '---'.
    """
    sections = []
    for chunk in chunks:
        page = chunk.get("page", "?")
        text = chunk.get("text", "")
        sections.append(f"[Page {page}] {text}")
    return "\n---\n".join(sections)


def _build_user_prompt(query: str, context: str) -> str:
    """Build the full user message with context + question."""
    return (
        f"CONTEXT (use as evidence, do NOT copy raw text):\n{context}\n\n"
        f"---\n\n"
        f"QUESTION: {query}\n\n"
        f"Remember: Answer in your own words as fluent sentences. "
        f"Do NOT paste tables or raw chunk text."
    )


def generate(
    query: str,
    chunks: list[dict],
    model: str = "llama3.1",
    history: list[dict] | None = None,
) -> str:
    """
    Generate a response using Ollama (non-streaming).

    Args:
        query:   The user's question.
        chunks:  Retrieved context chunks [{text, page, score}, ...].
        model:   Ollama model name.
        history: Optional list of prior {"role": ..., "content": ...} messages.

    Returns:
        The model's response string.
    """
    context = _build_context(chunks)
    user_prompt = _build_user_prompt(query, context)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = ollama.chat(
            model=model,
            messages=messages,
        )
    except ConnectionError:
        return (
            "⚠️ **Ollama is not running.** "
            "Start it with: `ollama serve`"
        )
    except ResponseError as e:
        if "memory" in str(e).lower() or e.status_code == 500:
            return (
                "⚠️ **Not enough memory to run this model.** "
                "Close other applications to free RAM, or try a smaller "
                "model like `phi3` from the sidebar."
            )
        raise

    return response["message"]["content"]


def stream_generate(
    query: str,
    chunks: list[dict],
    model: str = "llama3.1",
    history: list[dict] | None = None,
):
    """
    Streaming variant — yields tokens one at a time for Flask SSE
    or similar real-time streaming display.

    Args:
        query:   The user's question.
        chunks:  Retrieved context chunks [{text, page, score}, ...].
        model:   Ollama model name.
        history: Optional list of prior {"role": ..., "content": ...} messages.

    Yields:
        str: Individual token strings as they arrive.
    """
    context = _build_context(chunks)
    user_prompt = _build_user_prompt(query, context)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    try:
        stream = ollama.chat(
            model=model,
            messages=messages,
            stream=True,
        )
    except ConnectionError:
        yield (
            "⚠️ **Ollama is not running.** "
            "Start it with: `ollama serve`"
        )
        return
    except ResponseError as e:
        if "memory" in str(e).lower() or e.status_code == 500:
            yield (
                "⚠️ **Not enough memory to run this model.** "
                "Close other applications to free RAM, or try a smaller "
                "model like `phi3` from the sidebar."
            )
            return
        raise

    try:
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield token
    except ResponseError as e:
        if "memory" in str(e).lower() or e.status_code == 500:
            yield (
                "\n\n⚠️ **Model ran out of memory mid-generation.** "
                "Try a smaller model like `phi3`."
            )
        else:
            raise
