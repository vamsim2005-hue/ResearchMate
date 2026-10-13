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


def extract_entities(text: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Extracts named entities (Organizations, People, Locations, Metrics, Datasets)
    from research text using spaCy.
    """
    if not text or not text.strip():
        return []

    nlp = get_spacy_nlp()
    # Process up to 100,000 characters to prevent memory blowout
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
            "count": count
        })

    return results
