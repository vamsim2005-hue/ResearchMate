"""
PDF Processor Module for ResearchMate
Extracts selectable text and document metadata from academic research paper PDFs,
preserving page numbering and physical boundaries.
"""

from pathlib import Path
from typing import List, Dict, Any
import pymupdf


def extract_pdf_text(pdf_path: str | Path) -> List[Dict[str, Any]]:
    """
    Extract non-empty text pages from a PDF.
    Returns a list of dicts with keys:
      - document: str (filename)
      - page: int (1-based page number)
      - text: str (raw extracted text)
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF files are supported, got: {pdf_path.name}")

    pages = []

    with pymupdf.open(pdf_path) as doc:
        if doc.is_encrypted:
            raise ValueError(f"Encrypted PDFs are not supported: {pdf_path.name}")

        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({
                    "document": pdf_path.name,
                    "page": page_number,
                    "text": text
                })

    if not pages:
        raise ValueError(
            f"No selectable text found in {pdf_path.name}. The document may be scanned or image-only."
        )

    return pages


def get_pdf_info(pdf_path: str | Path) -> Dict[str, Any]:
    """Return basic metadata about a PDF file."""
    pdf_path = Path(pdf_path)
    with pymupdf.open(pdf_path) as doc:
        meta = doc.metadata or {}
        return {
            "filename": pdf_path.name,
            "filepath": str(pdf_path.resolve()),
            "total_pages": len(doc),
            "title": meta.get("title") or pdf_path.stem.replace("_", " "),
            "author": meta.get("author") or "Unknown Author",
            "subject": meta.get("subject") or "",
            "file_size_kb": round(pdf_path.stat().st_size / 1024, 2)
        }
