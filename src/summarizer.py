"""
Summarization Module for ResearchMate
Provides both fast extractive scientific summarization and transformer-based
abstractive summarization.
"""

from typing import List, Dict, Any
import re
from collections import Counter

_abstractive_pipeline = None


def generate_extractive_summary(pages: List[Dict[str, Any]], num_sentences: int = 5) -> str:
    """
    Fast, reliable extractive summary that extracts the most salient sentences
    from the title/abstract/introduction (early pages) and conclusion (late pages).
    """
    if not pages:
        return "No text available to summarize."

    # Prioritize first 2 pages (Abstract + Intro) and last 2 pages (Discussion + Conclusion)
    candidate_text = ""
    for p in pages[:2]:
        candidate_text += "\n" + p.get("text", "")
    if len(pages) > 3:
        for p in pages[-2:]:
            candidate_text += "\n" + p.get("text", "")

    # Clean sentences
    raw_sentences = re.split(r'(?<=[.!?])\s+', candidate_text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 30 and len(s.strip()) < 300]

    if not sentences:
        return "Insufficient text to extract sentences."

    # Score sentences by term frequency
    words = re.findall(r'\b[a-zA-Z]{3,}\b', candidate_text.lower())
    stop_words = set(["this", "that", "with", "from", "have", "were", "been", "which", "their", "there", "also", "these"])
    freq = Counter([w for w in words if w not in stop_words])

    def score_sentence(sent: str) -> float:
        sent_words = re.findall(r'\b[a-zA-Z]{3,}\b', sent.lower())
        if not sent_words:
            return 0.0
        # Boost sentences containing key academic indicator phrases
        boost = 1.0
        lower_s = sent.lower()
        if any(indicator in lower_s for indicator in ["we propose", "in this paper", "we demonstrate", "our findings", "we find", "conclude", "in conclusion"]):
            boost = 1.5
        return (sum(freq.get(w, 0) for w in sent_words) / len(sent_words)) * boost

    scored = sorted([(score_sentence(s), s) for s in sentences], key=lambda x: x[0], reverse=True)
    selected = [s for score, s in scored[:num_sentences]]

    # Reorder by original appearance
    ordered = [s for s in sentences if s in selected][:num_sentences]
    return " ".join(ordered)


def generate_abstractive_summary(text: str, max_length: int = 150, min_length: int = 40) -> str:
    """
    Transformer-based abstractive summarization using DistilBART.
    Falls back to extractive summary if model fails or cannot be loaded.
    """
    global _abstractive_pipeline
    try:
        from transformers import pipeline
        import torch

        if _abstractive_pipeline is None:
            device = 0 if torch.cuda.is_available() else -1
            _abstractive_pipeline = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                device=device
            )

        truncated = text[:3000]
        res = _abstractive_pipeline(
            truncated,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        return res[0]["summary_text"].strip()
    except Exception as e:
        # Fallback to simple extractive snippet
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return " ".join([s.strip() for s in sentences[:4] if len(s.strip()) > 20])
