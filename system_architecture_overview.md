# Project Architecture: Book GPT

This document provides a simple, high-level explanation of how the Book GPT system works. You can use this to explain the technical flow to your professor.

---

## 1. The Core Concept: RAG
This project uses **RAG (Retrieval-Augmented Generation)**. 
Instead of sending a massive book directly to an AI (which would be too expensive or exceed its memory limit), we:
1. **Retrieve** only the most relevant snippets from the book.
2. **Augment** the user's question with those snippets.
3. **Generate** an answer based *only* on that evidence.

---

## 2. Component Breakdown

### A. The Ingestion Pipeline (The "Reading" Phase)
When you upload a file, the system goes through these steps:
- **Parsers**: The system supports PDF, EPUB, DOCX, and TXT. These scripts "strip" the formatting and extract the raw text and page numbers.
- **Chunking**: Large books are broken into small "chunks" (about 600 characters each). This makes it easier for the AI to find specific facts.
- **Quality Filtering**: We use a custom filter to throw away "garbage" chunks (like table of contents, page numbers, or numeric tables) so the AI doesn't get confused by meaningless data.

### B. Embedding Models (The "Translation" Phase)
Computer programs can't compare "words" very well, but they are great at comparing "numbers."
- **What they do**: An Embedding Model turns a text chunk into a long list of numbers (a **Vector**). 
- **The Secret**: Chunks with similar *meanings* end up with similar *numbers*.
- **Our Options**: We provide three "flavors":
    - **Fast**: Small and quick.
    - **Balanced**: High accuracy but slower.
    - **Ollama**: Uses an external server to do the math.

### C. ChromaDB (The "Bookshelf")
**ChromaDB** is a **Vector Database**. Think of it as a specialized bookshelf where books aren't stored by title, but by *meaning*. When we have a question, ChromaDB quickly finds the chunks whose "numbers" (vectors) most closely match the question's numbers.

### D. The Concept Graph (The "Intelligence" Layer)
This is a unique feature of this project. While ChromaDB finds text, the **Concept Graph** builds a "mind map" of the book. 
- It uses **NLP (Natural Language Processing)** to find key entities (people, places, technical terms).
- It creates links between them.
- When you ask a question, the system looks at the map to see if there are related concepts that the text search might have missed.

### E. The LLM - Large Language Model (The "Writer")
Once we have the best chunks and the graph data, we send them to the **LLM** (via **Ollama**).
- **Models used**: `phi3`, `mistral`, or `llama3.1`.
- **Role**: It acts as a professional researcher. it reads the provided chunks and writes a fluent, natural-sounding answer. It is told: *"Only answer using the context I gave you. Do not make things up."*

---

## 3. The Path of a Query (Step-by-Step)

1. **User Asks**: "How do carbohydrates affect energy?"
2. **Embedding**: The question is turned into a vector (numbers).
3. **Retrieval**: ChromaDB finds the top 5 chunks about "carbohydrates" and "energy."
4. **Graph Search**: The Concept Graph identifies that "Glucose" is related to "Carbohydrates" and adds context about glucose.
5. **Prompting**: We build a "Super Prompt" for the AI:
   > "Here is context from the book: [Chunk 1], [Chunk 2]... Based on this, answer the user's question: How do carbohydrates affect energy?"
6. **Streaming**: The LLM (Ollama) generates the answer word-by-word, which streams back to your screen in real-time.

---

## 4. Why this is better than "Standard" AI
- **No Hallucinations**: Since the AI is locked to the provided text, it won't "guess" facts.
- **Privacy**: Everything runs **locally**. Your books and questions never leave your computer.
- **Transparency**: The system provides **Sources** (page numbers) for every claim it makes.
- **Exportable**: The Q&A session can be exported as a clean **PDF Report** for study or grading.
