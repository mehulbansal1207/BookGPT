"""
Multi-Format File Parsers
=========================
Each parser returns the same structure: list of {page: int, text: str}
so the rest of the pipeline (chunking, filtering, embedding) is format-agnostic.

Supported: PDF, EPUB, TXT, DOCX, MD
"""

import os
import re
import logging
from html.parser import HTMLParser

logger = logging.getLogger(__name__)

# ── Allowed extensions (used by the router and the server) ───────────────────
ALLOWED_EXTENSIONS = {".pdf", ".epub", ".txt", ".docx", ".md"}


# ---------------------------------------------------------------------------
# HTML tag stripper (for EPUB)
# ---------------------------------------------------------------------------

class _HTMLStripper(HTMLParser):
    """Simple HTML→plain-text converter."""

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str):
        self.parts.append(data)

    def get_text(self) -> str:
        return "".join(self.parts)


def _strip_html(html_text: str) -> str:
    stripper = _HTMLStripper()
    stripper.feed(html_text)
    return stripper.get_text()


# ---------------------------------------------------------------------------
# 1. PDF  (PyMuPDF / fitz)
# ---------------------------------------------------------------------------

def parse_pdf(file_path: str) -> list[dict]:
    """Extract text page by page from a PDF."""
    import fitz  # PyMuPDF

    doc = fitz.open(file_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        if len(text.strip()) < 50:
            logger.warning(
                "Page %d has only %d chars — likely a scanned/image page.",
                page_num + 1,
                len(text.strip()),
            )
        pages.append({"page": page_num + 1, "text": text})
    doc.close()
    return pages


# ---------------------------------------------------------------------------
# 2. EPUB  (ebooklib)
# ---------------------------------------------------------------------------

def parse_epub(file_path: str) -> list[dict]:
    """
    Extract text chapter by chapter from an EPUB.
    Each chapter is treated as a 'page'.
    """
    import ebooklib
    from ebooklib import epub

    book = epub.read_epub(file_path, options={"ignore_ncx": True})
    pages = []
    chapter_num = 0

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        raw_html = item.get_content().decode("utf-8", errors="replace")
        text = _strip_html(raw_html).strip()
        if len(text) < 20:
            continue  # skip near-empty chapters (cover images, etc.)
        chapter_num += 1
        pages.append({"page": chapter_num, "text": text})

    if not pages:
        logger.warning("EPUB produced 0 chapters — file may be image-only.")

    return pages


# ---------------------------------------------------------------------------
# 3. TXT  (plain text)
# ---------------------------------------------------------------------------

_TXT_PAGE_SIZE = 3000  # chars per synthetic "page"


def parse_txt(file_path: str) -> list[dict]:
    """
    Read a plain text file, splitting into synthetic pages of ~3000 chars.
    Tries UTF-8 first, falls back to Latin-1.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Cannot decode text file: {file_path}")

    pages = []
    for i in range(0, len(content), _TXT_PAGE_SIZE):
        chunk = content[i : i + _TXT_PAGE_SIZE]
        if chunk.strip():
            pages.append({"page": len(pages) + 1, "text": chunk})

    return pages


# ---------------------------------------------------------------------------
# 4. DOCX  (python-docx)
# ---------------------------------------------------------------------------

_DOCX_PAGE_SIZE = 3000  # chars per synthetic "page"


def parse_docx(file_path: str) -> list[dict]:
    """
    Extract text from a .docx file, grouping paragraphs into ~3000-char pages.
    Heading structure is preserved in text.
    """
    from docx import Document

    doc = Document(file_path)
    pages = []
    current_text = ""
    page_num = 1

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            current_text += "\n"
            continue

        # Preserve heading structure
        if para.style and para.style.name.startswith("Heading"):
            prefix = "#" * min(int(para.style.name[-1]) if para.style.name[-1].isdigit() else 1, 4)
            text = f"\n{prefix} {text}\n"

        current_text += text + "\n"

        if len(current_text) >= _DOCX_PAGE_SIZE:
            pages.append({"page": page_num, "text": current_text.strip()})
            current_text = ""
            page_num += 1

    # Flush remaining text
    if current_text.strip():
        pages.append({"page": page_num, "text": current_text.strip()})

    return pages


# ---------------------------------------------------------------------------
# 5. Markdown
# ---------------------------------------------------------------------------

def parse_md(file_path: str) -> list[dict]:
    """
    Read a markdown file, splitting on ## headings.
    Each section becomes one 'page'.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Cannot decode markdown file: {file_path}")

    # Split on ## headings (keep the heading with its section)
    sections = re.split(r"(?=^## )", content, flags=re.MULTILINE)

    pages = []
    for section in sections:
        text = section.strip()
        if text:
            pages.append({"page": len(pages) + 1, "text": text})

    # If no ## headings were found, return the whole file as one page
    if not pages and content.strip():
        pages.append({"page": 1, "text": content.strip()})

    return pages


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_PARSERS = {
    ".pdf":  parse_pdf,
    ".epub": parse_epub,
    ".txt":  parse_txt,
    ".docx": parse_docx,
    ".md":   parse_md,
}


def parse_file(file_path: str) -> list[dict]:
    """
    Detect format from file extension and call the right parser.
    Returns: list of {page: int, text: str}
    """
    ext = os.path.splitext(file_path)[1].lower()

    parser = _PARSERS.get(ext)
    if parser is None:
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file format: '{ext}'. "
            f"Supported formats: {supported}"
        )

    logger.info("Parsing %s file: %s", ext.upper(), os.path.basename(file_path))
    pages = parser(file_path)
    logger.info("Parsed %d pages/sections from %s", len(pages), os.path.basename(file_path))
    return pages
