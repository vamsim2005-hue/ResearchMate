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


_summarizer_tokenizer = None
_summarizer_model = None
DEFAULT_SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"


def get_abstractive_model(model_name: str = DEFAULT_SUMMARIZATION_MODEL):
    """Lazily load Hugging Face pretrained Seq2Seq summarization model."""
    global _summarizer_tokenizer, _summarizer_model
    if _summarizer_model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        _summarizer_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _summarizer_model.eval()
        if torch.cuda.is_available():
            _summarizer_model = _summarizer_model.cuda()
    return _summarizer_tokenizer, _summarizer_model


def generate_abstractive_summary(
    text: str,
    max_length: int = 150,
    min_length: int = 40,
    model_name: str = DEFAULT_SUMMARIZATION_MODEL
) -> str:
    """
    Transformer-based abstractive summarization using Hugging Face pretrained models
    (e.g., sshleifer/distilbart-cnn-12-6 or facebook/bart-large-cnn).
    Properly handles token truncation and fallback.
    """
    if not text or not text.strip():
        return "No text available for abstractive summarization."

    try:
        import torch
        tokenizer, model = get_abstractive_model(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Tokenize with proper max_length truncation
        inputs = tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        )
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=2,
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )

        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
        if summary_text:
            return summary_text
        else:
            raise ValueError("Empty summary produced by transformer.")

    except Exception as e:
        # Fallback to high-scoring extractive sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        valid = [s.strip() for s in sentences if len(s.strip()) > 25]
        return " ".join(valid[:4]) if valid else text[:250].strip()

