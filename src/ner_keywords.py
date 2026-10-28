"""
Named Entity Recognition and Keyword Extraction Module for ResearchMate
Extracts salient scientific terms, methods, organizations, and keywords from academic text.
"""

from typing import List, Dict, Any
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer

_nlp = None


def get_spacy_nlp():
    """Lazily load spaCy model."""
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            _nlp = spacy.blank("en")
    return _nlp


def extract_keywords(texts: List[str], top_k: int = 15) -> List[Dict[str, Any]]:
    """
    Extracts top keywords and n-grams using TF-IDF across the supplied text passages.
    Returns a list of dicts: [{"keyword": str, "score": float}, ...]
    """
    cleaned = [t.strip() for t in texts if t and t.strip()]
    if not cleaned:
        return []

    # Stopwords specific to academic publications
    custom_stopwords = [
        "et", "al", "figure", "table", "section", "paper", "author", "authors",
        "study", "results", "method", "proposed", "using", "used", "based", "data"
    ]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_df=0.85,
            min_df=1,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]{2,}\b"
        )
        tfidf_matrix = vectorizer.fit_transform(cleaned)
        feature_names = vectorizer.get_feature_names_out()
        mean_scores = tfidf_matrix.mean(axis=0).A1

        # Filter out custom academic boilerplate words
        keyword_scores = []
        for word, score in zip(feature_names, mean_scores):
            if word.lower() not in custom_stopwords and not word.isnumeric():
                keyword_scores.append({"keyword": word, "score": round(float(score), 4)})

        keyword_scores.sort(key=lambda x: x["score"], reverse=True)
        return keyword_scores[:top_k]

    except Exception as e:
        # Fallback to simple frequency counter if vectorizer fails
        words = re.findall(r"\b[a-zA-Z]{4,}\b", " ".join(cleaned).lower())
        stop_words_set = set(["this", "that", "with", "from", "have", "were", "been", "which", "their", "there"])
        filtered = [w for w in words if w not in stop_words_set and w not in custom_stopwords]
        counts = Counter(filtered).most_common(top_k)
        return [{"keyword": w, "score": round(c / max(1, len(filtered)), 4)} for w, c in counts]


_transformer_ner_pipeline = None


def get_transformer_ner(model_name: str = "dslim/bert-base-NER"):
    """Lazily load Hugging Face pretrained transformer NER pipeline."""
    global _transformer_ner_pipeline
    if _transformer_ner_pipeline is None:
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        _transformer_ner_pipeline = pipeline(
            "token-classification",
            model=model_name,
            aggregation_strategy="simple",
            device=device
        )
    return _transformer_ner_pipeline


def extract_entities_transformer(
    text: str,
    model_name: str = "dslim/bert-base-NER",
    top_k: int = 20,
    max_chunk_chars: int = 1500
) -> List[Dict[str, Any]]:
    """
    Extracts named entities using a pretrained Hugging Face Transformer (dslim/bert-base-NER).
    Splits long research text into token-safe chunks, merges entities, and aggregates counts and scores.
    """
    if not text or not text.strip():
        return []

    try:
        ner_pipe = get_transformer_ner(model_name)
    except Exception as e:
        # Fallback to spaCy if transformer fails to load
        return extract_entities_spacy(text, top_k=top_k)

    # Chunk text to stay well within BERT's 512-token context window
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\.\s+", text) if p.strip()]
    chunks = []
    current_chunk = ""
    for p in paragraphs:
        if len(current_chunk) + len(p) + 2 < max_chunk_chars:
            current_chunk += (" " + p) if current_chunk else p
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = p[:max_chunk_chars]
        if len(chunks) >= 12:  # Process up to 12 representative passages
            break
    if current_chunk and len(chunks) < 12:
        chunks.append(current_chunk)

    if not chunks:
        chunks = [text[:max_chunk_chars]]

    label_human_map = {
        "PER": "PERSON",
        "ORG": "ORGANIZATION",
        "LOC": "LOCATION",
        "MISC": "ACADEMIC / MISC"
    }

    entity_records: Dict[str, Dict[str, Any]] = {}

    for chunk in chunks:
        try:
            preds = ner_pipe(chunk)
            for item in preds:
                word = item.get("word", "").strip().strip(".,;:()[]\"'")
                # Clean up subwords or short noise
                if len(word) < 2 or word.isdigit() or word.lower() in {"the", "and", "for", "with"}:
                    continue

                raw_group = item.get("entity_group", "MISC")
                label = label_human_map.get(raw_group, raw_group)
                score = float(item.get("score", 0.0))

                key = (word, label)
                if key not in entity_records:
                    entity_records[key] = {
                        "entity": word,
                        "label": label,
                        "count": 0,
                        "total_score": 0.0,
                        "model": "Neural Model (BERT-NER)"
                    }
                entity_records[key]["count"] += 1
                entity_records[key]["total_score"] += score
        except Exception:
            continue

    if not entity_records:
        return extract_entities_spacy(text, top_k=top_k)

    results = []
    for rec in entity_records.values():
        avg_score = rec["total_score"] / max(1, rec["count"])
        results.append({
            "entity": rec["entity"],
            "label": rec["label"],
            "count": rec["count"],
            "confidence": round(avg_score, 4),
            "model": "Neural Model (BERT-NER)"
        })

    # Sort by frequency, then confidence
    results.sort(key=lambda x: (x["count"], x["confidence"]), reverse=True)
    return results[:top_k]


def extract_entities_spacy(text: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Extracts named entities using spaCy en_core_web_sm.
    """
    if not text or not text.strip():
        return []

    nlp = get_spacy_nlp()
    doc = nlp(text[:100000])

    entity_counts = Counter()
    label_map = {}
    target_labels = {"ORG", "PERSON", "GPE", "PRODUCT", "WORK_OF_ART", "EVENT", "LAW", "NORP"}

    for ent in doc.ents:
        clean_ent = ent.text.strip().replace("\n", " ")
        if len(clean_ent) > 2 and ent.label_ in target_labels:
            entity_counts[clean_ent] += 1
            label_map[clean_ent] = ent.label_

    results = []
    for ent_text, count in entity_counts.most_common(top_k):
        results.append({
            "entity": ent_text,
            "label": label_map.get(ent_text, "MISC"),
            "count": count,
            "confidence": 0.85,
            "model": "spaCy"
        })

    return results


def extract_entities(text: str, top_k: int = 20, method: str = "transformer") -> List[Dict[str, Any]]:
    """
    Extracts named entities from research text.
    Methods:
      - 'transformer': Hugging Face pretrained BERT-NER (dslim/bert-base-NER) [Default]
      - 'spacy': spaCy statistical pipeline
    """
    if method == "transformer":
        return extract_entities_transformer(text, top_k=top_k)
    else:
        return extract_entities_spacy(text, top_k=top_k)

