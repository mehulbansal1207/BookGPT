"""
PDF Report Generator
====================
Exports the Q&A session as a clean, downloadable PDF using fpdf2.
"""

from datetime import datetime
from fpdf import FPDF


# ── Unicode -> ASCII mapping for Helvetica (Latin-1) compatibility ───────────
_UNICODE_REPLACEMENTS = {
    "\u2014": "--",     # em-dash
    "\u2013": "-",      # en-dash
    "\u2018": "'",      # left single quote
    "\u2019": "'",      # right single quote
    "\u201c": '"',      # left double quote
    "\u201d": '"',      # right double quote
    "\u2022": "*",      # bullet
    "\u2026": "...",    # ellipsis
    "\u00b7": "*",      # middle dot
    "\u2192": "->",     # right arrow
    "\u2190": "<-",     # left arrow
    "\u2264": "<=",     # less-than-or-equal
    "\u2265": ">=",     # greater-than-or-equal
    "\u2260": "!=",     # not-equal
    "\u00a0": " ",      # non-breaking space
}


def _safe_text(text: str) -> str:
    """Replace Unicode characters unsupported by Helvetica with ASCII equivalents."""
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    # Final safety: encode to latin-1 replacing any remaining outliers
    return text.encode("latin-1", errors="replace").decode("latin-1")


class _ReportPDF(FPDF):
    """Custom PDF with header/footer for the Book GPT report."""

    def __init__(self, book_title: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.book_title = book_title
        self.set_auto_page_break(auto=True, margin=25)

    # ── Header ───────────────────────────────────────────────────────────
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, "Book GPT - Session Report", align="L")
        self.set_font("Helvetica", "", 8)
        self.cell(0, 6, datetime.now().strftime("%Y-%m-%d  %H:%M"), align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    # ── Footer ───────────────────────────────────────────────────────────
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(160, 160, 160)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def generate_report(session_data: dict) -> bytes:
    """
    Generate a PDF report from the session data.

    Args:
        session_data: {
            "book_title": str,
            "questions_and_answers": [
                {
                    "question": str,
                    "answer": str,
                    "sources": [{"page": int, "text": str, "score": float}]
                }
            ]
        }

    Returns:
        PDF file content as bytes.
    """
    book_title = _safe_text(session_data.get("book_title", "Untitled Book"))
    qa_list = session_data.get("questions_and_answers", [])

    pdf = _ReportPDF(book_title)
    pdf.alias_nb_pages()
    pdf.add_page()

    # ── Cover section ────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 10, book_title, align="L")
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 6, f"Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Total questions: {len(qa_list)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Divider
    pdf.set_draw_color(200, 200, 200)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(8)

    # ── Q&A blocks ───────────────────────────────────────────────────────
    for i, qa in enumerate(qa_list, 1):
        question = _safe_text(qa.get("question", ""))
        answer = _safe_text(qa.get("answer", "").replace("**", "").replace("__", ""))
        sources = qa.get("sources", [])

        # -- Question header (accent background) --
        pdf.set_fill_color(240, 238, 255)  # #f0eeff
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(50, 42, 100)
        q_text = f"Q{i}: {question}"
        pdf.multi_cell(0, 8, q_text, fill=True, align="L")
        pdf.ln(3)

        # -- Answer text --
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(50, 50, 50)
        pdf.multi_cell(0, 6.5, answer, align="L")
        pdf.ln(3)

        # -- Sources --
        if sources:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(0, 6, "Sources:", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)

            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(130, 130, 130)
            for src in sources:
                page = src.get("page", "?")
                score = src.get("score", 0)
                text_preview = _safe_text(src.get("text", "")[:120].replace("\n", " "))
                source_line = f"  * Page {page} -- score: {score:.3f}  \"{text_preview}...\""
                # Indent by shifting x
                old_x = pdf.get_x()
                pdf.set_x(old_x + 10)
                pdf.multi_cell(pdf.w - pdf.r_margin - pdf.get_x(), 5, source_line, align="L")
                pdf.ln(1)

        # Gap between Q&A blocks
        pdf.ln(8)

        # Separator between Q&A blocks (except last)
        if i < len(qa_list):
            y = pdf.get_y()
            pdf.set_draw_color(220, 220, 220)
            pdf.dashed_line(pdf.l_margin + 20, y, pdf.w - pdf.r_margin - 20, y, dash_length=2, space_length=2)
            pdf.ln(6)

    return pdf.output()
