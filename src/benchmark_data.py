"""
Benchmark Data Generator and Loader for ResearchMate
Provides curated and automated evaluation test sets for Information Retrieval (IR)
and RAG faithfulness benchmarking.
"""

from pathlib import Path
from typing import List, Dict, Any
import json
import re


def generate_benchmark_queries_from_chunks(chunks: List[Dict[str, Any]], num_queries: int = 15) -> List[Dict[str, Any]]:
    """
    Synthesizes grounded evaluation test queries from extracted chunks with verified
    ground-truth relevant chunk IDs.
    """
    queries = []
    step = max(1, len(chunks) // (num_queries + 1))

    for i in range(0, min(len(chunks), num_queries * step), step):
        target_chunk = chunks[i]
        text = target_chunk["text"]

        # Extract salient terms or first question-worthy sentence
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 35]
        if not sentences:
            continue

        target_sentence = sentences[0]
        # Formulate a natural research query based on the sentence
        words = target_sentence.split()[:8]
        query_text = " ".join(words)

        queries.append({
            "id": f"q_{len(queries) + 1}",
            "question": f"What does the paper state regarding {query_text}?",
            "relevant_chunk_ids": [target_chunk["chunk_id"]],
            "document": target_chunk.get("document", ""),
            "page": target_chunk.get("page", 1),
            "expected_evidence": target_sentence[:180]
        })

        if len(queries) >= num_queries:
            break

    return queries


def save_benchmark_queries(queries: List[Dict[str, Any]], filepath: str | Path):
    """Saves benchmark queries to JSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2)


def load_benchmark_queries(filepath: str | Path) -> List[Dict[str, Any]]:
    """Loads benchmark queries from JSON."""
    filepath = Path(filepath)
    if not filepath.exists():
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
