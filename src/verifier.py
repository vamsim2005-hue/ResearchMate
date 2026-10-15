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

    # 1. Fast direct containment / high lexical overlap check
    clean_claim = re.sub(r'[^\w\s]', '', claim.lower()).strip()
    clean_evidence = re.sub(r'[^\w\s]', '', evidence.lower())
    if clean_claim and clean_claim in clean_evidence:
        return {
            "claim": claim,
            "label": "supported",
            "confidence": 0.999,
            "raw_label": "exact_entailment"
        }

    # 2. Localize the most relevant context window in evidence
    evidence_sentences = re.split(r'(?<=[.!?])\s+', evidence)
    claim_words = set(re.findall(r'\b\w{4,}\b', claim.lower()))
    
    best_sent = ""
    best_overlap = 0
    for s in evidence_sentences:
        s_words = set(re.findall(r'\b\w{4,}\b', s.lower()))
        overlap = len(claim_words & s_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sent = s.strip()

    # Use the best matching sentence or slice around it as premise
    if best_sent and len(best_sent) > 20:
        premise = best_sent[:600]
    else:
        premise = evidence[:800]

    classifier = get_verifier()

    # NLI convention: premise = evidence, hypothesis = claim
    try:
        result = classifier({
            "text": premise,
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
        found_supported = False
        best_supported_conf = 0.0
        best_contradicted_conf = 0.0
        highest_conf = 0.0
        highest_raw = "neutral"

        for p in passages:
            evidence_text = p.get("text", "")
            res = verify_claim(claim, evidence_text, threshold=threshold)
            c_conf = res.get("confidence", 0.0)
            c_label = res.get("label", "uncertain")

            if c_label == "supported":
                found_supported = True
                if c_conf > best_supported_conf:
                    best_supported_conf = c_conf
                break  # Verified by at least one cited passage
            elif c_label == "contradicted":
                if c_conf > best_contradicted_conf:
                    best_contradicted_conf = c_conf

            if c_conf > highest_conf:
                highest_conf = c_conf
                highest_raw = res.get("raw_label", "neutral")

        if found_supported:
            final_label = "supported"
            final_conf = best_supported_conf
            final_raw = "entailment"
        elif best_contradicted_conf > 0:
            final_label = "contradicted"
            final_conf = best_contradicted_conf
            final_raw = "contradiction"
        else:
            final_label = "uncertain"
            final_conf = highest_conf
            final_raw = highest_raw

        claim_results.append({
            "claim": claim,
            "label": final_label,
            "confidence": round(final_conf, 4),
            "raw_label": final_raw
        })

        if final_label == "supported":
            supported_count += 1
        elif final_label == "contradicted":
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
