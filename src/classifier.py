"""
Pretrained Transformer Zero-Shot Classifier Module for ResearchMate
Uses Hugging Face cross-encoder/nli-deberta-v3-small (or MNLI transformers)
to categorize academic papers by scientific discipline, methodology, and thematic focus.
"""

from typing import List, Dict, Any, Union, Optional
import torch

DEFAULT_ZERO_SHOT_MODEL = "cross-encoder/nli-deberta-v3-small"
_zero_shot_pipeline = None

# Pre-defined academic taxonomies
DISCIPLINE_TAXONOMY = [
    "Human Rights, Law & Ethics",
    "Computer Science & Artificial Intelligence",
    "Biomedicine, Healthcare & Life Sciences",
    "Social, Economic & Political Science",
    "Environmental & Climate Science",
    "Physics, Mathematics & Engineering"
]

METHODOLOGY_TAXONOMY = [
    "Empirical & Quantitative Study",
    "Policy Analysis & Legal Governance",
    "Theoretical & Conceptual Framework",
    "System Design, Implementation & Benchmark",
    "Survey & Systematic Literature Review"
]

THEME_TAXONOMY = [
    "International Governance & Treaty Obligations",
    "Ethical & Social Implications of Technology",
    "Algorithm Optimization & Performance",
    "Public Health & Medical Intervention",
    "Sustainability & Environmental Conservation"
]


def get_zero_shot_classifier(model_name: str = DEFAULT_ZERO_SHOT_MODEL):
    """Lazily load Hugging Face zero-shot classification pipeline."""
    global _zero_shot_pipeline
    if _zero_shot_pipeline is None:
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        _zero_shot_pipeline = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device
        )
    return _zero_shot_pipeline


def classify_text(
    text: str,
    candidate_labels: List[str],
    multi_label: bool = False,
    hypothesis_template: str = "This research paper is about {}."
) -> Dict[str, Any]:
    """
    Classifies a text passage into candidate labels using a pretrained Hugging Face Transformer.
    Returns: {"labels": [...], "scores": [...], "top_label": str, "top_score": float}
    """
    if not text or not text.strip() or not candidate_labels:
        return {"labels": [], "scores": [], "top_label": "Unknown", "top_score": 0.0}

    classifier = get_zero_shot_classifier()
    # Use first ~1500 chars (title, abstract, intro) for fast, highly accurate classification
    sample = text[:1500].strip()

    try:
        res = classifier(
            sample,
            candidate_labels=candidate_labels,
            multi_label=multi_label,
            hypothesis_template=hypothesis_template
        )
        scores = [round(float(s), 4) for s in res["scores"]]
        labels = list(res["labels"])
        return {
            "labels": labels,
            "scores": scores,
            "top_label": labels[0] if labels else "Unknown",
            "top_score": scores[0] if scores else 0.0,
            "ranking": [
                {"label": l, "score": s, "percentage": round(s * 100, 1)}
                for l, s in zip(labels, scores)
            ]
        }
    except Exception as e:
        return {
            "labels": candidate_labels,
            "scores": [1.0 / len(candidate_labels)] * len(candidate_labels),
            "top_label": candidate_labels[0],
            "top_score": 0.5,
            "error": str(e),
            "ranking": []
        }


def classify_paper(
    pages_or_text: Union[str, List[Dict[str, Any]]],
    model_name: str = DEFAULT_ZERO_SHOT_MODEL
) -> Dict[str, Any]:
    """
    Performs full zero-shot multi-dimensional classification of a research paper:
    1. Research Discipline
    2. Methodology / Study Type
    3. Thematic Focus
    """
    if isinstance(pages_or_text, list):
        # Extract title and abstract from first 2-3 pages
        text = " ".join([p.get("text", "") for p in pages_or_text[:3]])
    else:
        text = str(pages_or_text)

    discipline_res = classify_text(
        text,
        candidate_labels=DISCIPLINE_TAXONOMY,
        multi_label=False,
        hypothesis_template="This scientific paper belongs to the academic field of {}."
    )

    methodology_res = classify_text(
        text,
        candidate_labels=METHODOLOGY_TAXONOMY,
        multi_label=False,
        hypothesis_template="The research methodology used in this work is a {}."
    )

    theme_res = classify_text(
        text,
        candidate_labels=THEME_TAXONOMY,
        multi_label=True,
        hypothesis_template="This research addresses {}."
    )

    return {
        "model": f"Hugging Face ({model_name})",
        "primary_discipline": discipline_res["top_label"],
        "discipline_confidence": discipline_res["top_score"],
        "disciplines": discipline_res["ranking"],
        "primary_methodology": methodology_res["top_label"],
        "methodology_confidence": methodology_res["top_score"],
        "methodologies": methodology_res["ranking"],
        "themes": theme_res["ranking"][:3]
    }
