"""
Evidence and Hallucination Verification Module for ResearchMate
Uses Natural Language Inference (NLI) with DeBERTa Cross-Encoder to verify
whether generated claims are faithfully supported by cited research evidence.
"""

from typing import List, Dict, Any
import re
import torch

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
_verifier_pipeline = None


def get_verifier():
    """Lazily load NLI DeBERTa model."""
    global _verifier_pipeline
    if _verifier_pipeline is None:
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        _verifier_pipeline = pipeline(
            "text-classification",
            model=NLI_MODEL_NAME,
            device=device
        )
    return _verifier_pipeline


def split_into_claims(text: str) -> List[str]:
    """Splits answer text into discrete assertable claims/sentences."""
    # Remove citation tags like [E1], [E2]
    clean = re.sub(r'\[E\d+\]', '', text)
    sentences = re.split(r'(?<=[.!?])\s+', clean)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def verify_claim(claim: str, evidence: str, threshold: float = 0.65) -> Dict[str, Any]:
    """
    Evaluates whether the evidence entails, contradicts, or is neutral towards a claim.
    Returns:
      {
        "claim": str,
        "label": "supported" | "contradicted" | "uncertain",
        "confidence": float,
        "raw_label": str
      }
    """
    claim = claim.strip()
    evidence = evidence.strip()

    if not claim or not evidence:
        return {
            "claim": claim,
            "label": "uncertain",
            "confidence": 0.0,
            "raw_label": "empty"
        }

    classifier = get_verifier()

    # NLI convention: premise = evidence, hypothesis = claim
    # Truncate evidence to ~1400 chars (~300 tokens) to guarantee < 512 tokens total
    try:
        result = classifier({
            "text": evidence[:1400],
            "text_pair": claim[:350]
        })
        if isinstance(result, list):
            result = result[0]

        raw_label = str(result["label"]).lower()
        confidence = float(result["score"])

        if "entail" in raw_label and confidence >= threshold:
            label = "supported"
        elif "contradict" in raw_label and confidence >= threshold:
            label = "contradicted"
        else:
            label = "uncertain"

        return {
            "claim": claim,
            "label": label,
            "confidence": round(confidence, 4),
            "raw_label": raw_label
        }
    except Exception as e:
        return {
            "claim": claim,
            "label": "uncertain",
            "confidence": 0.5,
            "raw_label": f"error: {str(e)}"
        }


def verify_answer(answer: str, passages: List[Dict[str, Any]], threshold: float = 0.60) -> Dict[str, Any]:
    """
    Decomposes an answer into claims and verifies each against retrieved passages.
    Checks claim against individual passages and selects the best matching evidence.
    """
    claims = split_into_claims(answer)
    if not claims:
        return {
            "overall_status": "uncertain",
            "faithfulness_score": 0.0,
            "supported_count": 0,
            "total_claims": 0,
            "claim_verifications": [],
            "flagged": True,
            "flag_reason": "No discrete claims detected in answer."
        }

    claim_results = []
    supported_count = 0
    contradicted_count = 0

    for claim in claims:
        best_label = "uncertain"
        best_confidence = 0.0
        best_raw = "neutral"

        for p in passages:
            evidence_text = p.get("text", "")
            res = verify_claim(claim, evidence_text, threshold=threshold)
            if res["label"] == "supported" and res["confidence"] > best_confidence:
                best_label = "supported"
                best_confidence = res["confidence"]
                best_raw = res["raw_label"]
                break
            elif res["label"] == "contradicted" and best_label != "supported":
                best_label = "contradicted"
                best_confidence = res["confidence"]
                best_raw = res["raw_label"]
            elif res["confidence"] > best_confidence:
                best_confidence = res["confidence"]
                best_raw = res["raw_label"]

        claim_results.append({
            "claim": claim,
            "label": best_label,
            "confidence": round(best_confidence, 4),
            "raw_label": best_raw
        })

        if best_label == "supported":
            supported_count += 1
        elif best_label == "contradicted":
            contradicted_count += 1

    total = len(claims)
    faithfulness = round((supported_count / total) * 100, 1) if total > 0 else 0.0

    # Flag criteria: any contradiction or faithfulness < 50%
    is_flagged = contradicted_count > 0 or faithfulness < 50.0
    flag_reason = ""
    if contradicted_count > 0:
        flag_reason = f"Answer contains {contradicted_count} claim(s) directly contradicted by evidence."
    elif faithfulness < 50.0:
        flag_reason = f"Only {faithfulness}% of claims were verified by retrieved passages. Risk of hallucination."

    overall_status = "supported" if (faithfulness >= 65.0 and contradicted_count == 0) else ("contradicted" if contradicted_count > 0 else "uncertain")

    return {
        "overall_status": overall_status,
        "faithfulness_score": faithfulness,
        "supported_count": supported_count,
        "total_claims": total,
        "claim_verifications": claim_results,
        "flagged": is_flagged,
        "flag_reason": flag_reason
    }
