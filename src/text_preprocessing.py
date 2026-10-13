"""
Text Preprocessing and Intra-Page Chunking Module for ResearchMate
Cleans scientific text and splits documents into page-grounded passages.
"""

import re
from typing import List, Dict, Any


def clean_text(text: str) -> str:
    """
    Cleans raw PDF text while preserving scientific punctuation, formulas,
    and citations.
    """
    if not text:
        return ""
    # Remove null bytes and control chars
    text = text.replace("\x00", " ")
    text = text.replace("\r", "\n")
    # Fix broken hyphenated line breaks (e.g., 'para-\ngraph' -> 'paragraph')
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Collapse multiple inline whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(pages: List[Dict[str, Any]], chunk_size: int = 350, overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Splits page text into intra-page chunks. Chunks NEVER span across
    multiple PDF pages, ensuring exact page provenance for every citation.

    Parameters:
      pages: list of dicts with {"document": str, "page": int, "text": str}
      chunk_size: number of words per chunk (default 350, ~450 tokens)
      overlap: number of overlapping words between consecutive chunks

    Returns:
      list of dicts with {"chunk_id": str, "document": str, "page": int, "text": str, "word_count": int}
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and less than chunk_size.")

    chunks = []
    chunk_counter = 0
    step = chunk_size - overlap

    for page_info in pages:
        doc_name = page_info["document"]
        page_num = page_info["page"]
        text = clean_text(page_info["text"])
        words = text.split()

        if not words:
            continue

        # If page is shorter than chunk_size, keep it as a single chunk
        if len(words) <= chunk_size:
            chunk_counter += 1
            chunks.append({
                "chunk_id": f"chunk_{chunk_counter}",
                "document": doc_name,
                "page": page_num,
                "text": " ".join(words),
                "word_count": len(words)
            })
            continue

        for start in range(0, len(words), step):
            selected = words[start:start + chunk_size]
            if not selected:
                continue

            chunk_counter += 1
            chunks.append({
                "chunk_id": f"chunk_{chunk_counter}",
                "document": doc_name,
                "page": page_num,
                "text": " ".join(selected),
                "word_count": len(selected)
            })

            if start + chunk_size >= len(words):
                break

    return chunks
