"""
Grounded RAG Pipeline Module for ResearchMate
Orchestrates multi-stage retrieval, evidence-grounded prompt assembly,
intelligent question comprehension, generative answer synthesis with citations [E1], [E2],
and NLI claim-level verification.
"""

import os
from typing import List, Dict, Any, Optional
import torch
import re

DEFAULT_LOCAL_MODEL = "sshleifer/distilbart-cnn-12-6"
DEFAULT_QA_MODEL = "distilbert/distilbert-base-cased-distilled-squad"

_seq2seq_model = None
_seq2seq_tokenizer = None
_qa_model = None
_qa_tokenizer = None


def get_seq2seq_generator(model_name: str = DEFAULT_LOCAL_MODEL):
    """Lazily load Hugging Face pretrained Seq2Seq model (e.g. distilbart-cnn)."""
    global _seq2seq_model, _seq2seq_tokenizer
    if _seq2seq_model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _seq2seq_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _seq2seq_model.eval()
        if torch.cuda.is_available():
            _seq2seq_model = _seq2seq_model.cuda()
    return _seq2seq_tokenizer, _seq2seq_model


def get_qa_pipeline(model_name: str = DEFAULT_QA_MODEL):
    """Lazily load Hugging Face pretrained extractive QA model."""
    global _qa_model, _qa_tokenizer
    if _qa_model is None:
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        _qa_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        _qa_model.eval()
        if torch.cuda.is_available():
            _qa_model = _qa_model.cuda()
    return _qa_tokenizer, _qa_model


def clean_sentence_for_qa(text: str) -> str:
    """Cleans noisy headers, bullet points, and special characters from extracted sentences."""
    cleaned = text.strip()
    # Remove leading heading prefixes
    cleaned = re.sub(r'^(?:abstract|introduction|conclusion|summary|background|discussion|\d+[\.\)]\s+)\s*[-:–—]?\s*', '', cleaned, flags=re.IGNORECASE)
    # Remove leading numbers or bullet characters
    cleaned = re.sub(r'^[•\-\*–—\d\.\s]+\s*', '', cleaned)
    # Clean up excessive punctuation or whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def synthesize_cross_encoder_answer(
    question: str,
    passages: List[Dict[str, Any]],
    reranker=None,
    min_score: float = -1.8
) -> Dict[str, Any]:
    """
    Synthesizes a cohesive, direct answer to the researcher's question by applying
    joint Cross-Encoder attention scoring (question, sentence) over all retrieved passages.
    Filters boilerplate, selects most informative factual sentences, and formats with citations.
    """
    if not passages:
        return {
            "answer": "The provided paper passages do not contain sufficient evidence to answer this question.",
            "confidence": 0.0,
            "citations": [],
            "model": "Cross-Encoder Semantic Comprehension"
        }

    # Lazy-load reranker if not provided
    if reranker is None:
        from sentence_transformers import CrossEncoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

    candidate_sentences = []
    # Filter out non-informative boilerplate and metadata lines
    boilerplate_patterns = [
        r"^by\s+[A-Z]",
        r"senior lecturer at",
        r"independent policy researcher",
        r"table of contents",
        r"all rights reserved",
        r"isbn\s+",
        r"issn\s+",
        r"downloaded from",
        r"page\s+\d+\s+of",
        r"south centre and swiss commission",
        r"author details",
        r"conflict of interest"
    ]

    for i, p in enumerate(passages, start=1):
        context = p.get("text", "")
        raw_sents = re.split(r'(?<=[.!?])\s+', context)
        for s in raw_sents:
            s_clean = clean_sentence_for_qa(s)
            # Ensure candidate has substantive length and is not a fragment or footer
            if len(s_clean) < 35 or len(s_clean) > 450:
                continue
            # Avoid boilerplate
            if any(re.search(bp, s_clean, re.IGNORECASE) for bp in boilerplate_patterns):
                continue
            # Skip standalone rhetorical questions unless user specifically asks a meta-question
            if s_clean.endswith("?") and not ("question" in question.lower() or "challenge" in question.lower()):
                continue

            candidate_sentences.append({
                "sentence": s_clean,
                "passage_idx": i,
                "document": p.get("document", "the paper"),
                "page": p.get("page", 1)
            })

    if not candidate_sentences:
        return {
            "answer": "The provided paper passages do not contain sufficient evidence to answer this question.",
            "confidence": 0.0,
            "citations": [],
            "model": "Cross-Encoder Semantic Comprehension"
        }

    # Joint Cross-Encoder scoring: captures deep semantic relationship between question and candidate sentence
    pairs = [[question, cs["sentence"]] for cs in candidate_sentences]
    scores = reranker.predict(pairs)

    ranked_items = []
    for cs, score in zip(candidate_sentences, scores):
        ranked_items.append((float(score), cs))

    ranked_items.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate and pick the top most relevant sentences
    top_entries = []
    seen_texts = set()
    for score, item in ranked_items:
        # Check normalized prefix to prevent redundancy
        normalized = re.sub(r'\W+', '', item["sentence"][:60].lower())
        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)

        if score >= min_score or len(top_entries) == 0:
            top_entries.append((score, item))

        if len(top_entries) >= 3:
            break

    if not top_entries or (len(top_entries) == 1 and top_entries[0][0] < -4.0):
        return {
            "answer": "The provided paper passages do not contain sufficient evidence to answer this question.",
            "confidence": 0.0,
            "citations": [],
            "model": "Cross-Encoder Semantic Comprehension"
        }

    # Synthesize answer with citations
    answer_parts = []
    for score, item in top_entries:
        p_idx = item["passage_idx"]
        answer_parts.append(f"{item['sentence']} [E{p_idx}]")

    synthesized_answer = " ".join(answer_parts)
    top_score = top_entries[0][0]
    # Normalize score to 0..1 confidence range approximately
    confidence = round(1.0 / (1.0 + float(torch.exp(torch.tensor(-top_score)).item())), 4)

    return {
        "answer": synthesized_answer,
        "confidence": confidence,
        "best_passage_idx": top_entries[0][1]["passage_idx"],
        "top_score": round(top_score, 3),
        "model": "Cross-Encoder Semantic Comprehension"
    }


def generate_gemini_answer(
    question: str,
    passages: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash"
) -> Dict[str, Any]:
    """
    Generates an evidence-grounded academic answer using Google Gemini Generative AI (via google.genai).
    Produces complete, well-reasoned answers referencing exact citations [E1], [E2].
    """
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not resolved_key or not resolved_key.strip():
        return {
            "answer": "Gemini API Key is not configured. Please enter your Gemini API Key in the left sidebar to enable Gemini Generative Comprehension, or select '🧠 Cross-Encoder Semantic Comprehension (Local)'.",
            "confidence": 0.0,
            "model": "Google Gemini (Key Missing)"
        }

    if not passages:
        return {
            "answer": "The provided paper passages do not contain sufficient evidence to answer this question.",
            "confidence": 0.0,
            "model": f"Google Gemini ({model_name})"
        }

    try:
        from google import genai
        client = genai.Client(api_key=resolved_key.strip())

        evidence_blocks = []
        for i, p in enumerate(passages, start=1):
            evidence_blocks.append(
                f"[E{i}] Document: {p.get('document', 'Paper')} (Page {p.get('page', '?')})\n"
                f"Content: {p.get('text', '').strip()}"
            )
        evidence_str = "\n\n".join(evidence_blocks)

        prompt = (
            "You are ResearchMate, an expert academic research assistant analyzing scientific publications.\n"
            "Answer the researcher's question accurately, thoroughly, and clearly based ONLY on the evidence passages provided below.\n\n"
            "Academic Rules:\n"
            "1. Answer Comprehension: Carefully understand the question and directly answer what was asked.\n"
            "2. Evidence Grounding: Rely strictly on facts in the provided evidence passages. Do not extrapolate, hallucinate, or assume unstated facts.\n"
            "3. Provenance Citations: Attach citations like [E1], [E2] immediately after each claim to link it directly to its source passage.\n"
            "4. Insufficient Evidence: If the provided passages do not contain enough facts to answer the question, explicitly state: 'The provided paper passages do not contain sufficient evidence to answer this question.'\n\n"
            f"Evidence Passages:\n{evidence_str}\n\n"
            f"Question: {question}\n\n"
            "Academic Answer:"
        )

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            ans_text = response.text.strip()
        except Exception:
            # Fallback to gemini-1.5-flash if gemini-2.5-flash is not available in region
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
            ans_text = response.text.strip()
            model_name = "gemini-1.5-flash"

        return {
            "answer": ans_text,
            "confidence": 0.98,
            "model": f"Google Gemini ({model_name})"
        }
    except Exception as e:
        return {
            "answer": f"Error calling Gemini API: {str(e)}. Please check your API key or use the local Cross-Encoder Semantic engine.",
            "confidence": 0.0,
            "model": "Google Gemini (Error)"
        }


def synthesize_seq2seq_answer(
    question: str,
    passages: List[Dict[str, Any]],
    model_name: str = DEFAULT_LOCAL_MODEL
) -> Dict[str, Any]:
    """
    Local generative question answering using Hugging Face Seq2Seq model (distilbart-cnn-12-6).
    """
    if not passages:
        return {
            "answer": "The provided paper passages do not contain sufficient evidence to answer this question.",
            "confidence": 0.0,
            "model": f"Local Seq2Seq ({model_name})"
        }

    try:
        tokenizer, model = get_seq2seq_generator(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Combine top passages
        combined_context = " ".join([p.get("text", "") for p in passages[:3]])
        input_text = f"Question: {question} Context: {combined_context}"

        inputs = tokenizer(
            input_text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        )
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                inputs["input_ids"],
                max_length=200,
                min_length=30,
                num_beams=2,
                early_stopping=True,
                no_repeat_ngram_size=3
            )

        gen_answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        # Append provenance citation
        formatted_answer = f"{gen_answer} [E1]"

        return {
            "answer": formatted_answer,
            "confidence": 0.85,
            "model": f"Local Seq2Seq ({model_name})"
        }
    except Exception as e:
        return {
            "answer": "",
            "confidence": 0.0,
            "model": f"Local Seq2Seq ({model_name})",
            "error": str(e)
        }


def extract_qa_transformer(
    question: str,
    passages: List[Dict[str, Any]],
    model_name: str = DEFAULT_QA_MODEL
) -> Dict[str, Any]:
    """
    Hugging Face Extractive QA Transformer (DistilBERT SQuAD).
    Extracts the highest-confidence answer span directly from evidence text and links
    it to its provenance citation [E1], [E2].
    """
    if not passages:
        return {
            "answer": "The provided paper passages do not contain sufficient evidence to answer this question.",
            "confidence": 0.0,
            "best_passage_idx": None,
            "extracted_span": "",
            "model": f"Neural Model ({model_name})"
        }

    tokenizer, model = get_qa_pipeline(model_name)

    best_score = -1e9
    best_answer = ""
    best_idx = 1
    best_passage = passages[0]
    best_full_sent = ""

    boilerplate_patterns = [
        r"^by\s+[A-Z]",
        r"senior lecturer at",
        r"independent policy researcher",
        r"table of contents",
        r"all rights reserved",
        r"isbn\s+",
        r"issn\s+",
        r"downloaded from",
        r"author details",
        r"conflict of interest"
    ]

    for i, p in enumerate(passages, start=1):
        context = p.get("text", "").strip()
        if not context:
            continue

        try:
            inputs = tokenizer(
                question,
                context,
                max_length=512,
                truncation="only_second",
                return_tensors="pt"
            )
            if device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]

            seq_ids = inputs.sequence_ids(0) if hasattr(inputs, "sequence_ids") else None

            if seq_ids is not None:
                sub_start_logits = start_logits.clone()
                sub_end_logits = end_logits.clone()
                for idx in range(len(start_logits)):
                    if seq_ids[idx] != 1:
                        sub_start_logits[idx] = -1e9
                        sub_end_logits[idx] = -1e9
                start_idx = torch.argmax(sub_start_logits).item()
                end_idx = torch.argmax(sub_end_logits).item()
            else:
                start_idx = torch.argmax(start_logits).item()
                end_idx = torch.argmax(end_logits).item()

            if end_idx >= start_idx and (end_idx - start_idx) < 50:
                answer_tokens = inputs["input_ids"][0][start_idx: end_idx + 1]
                span_text = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

                start_prob = torch.softmax(start_logits, dim=-1)[start_idx].item()
                end_prob = torch.softmax(end_logits, dim=-1)[end_idx].item()
                score = start_prob * end_prob

                # Skip uninformative or tiny spans
                if len(span_text) > 3 and not span_text.lower().startswith(("[cls]", "[sep]", "<s>")):
                    # Check containing sentence
                    sentences = re.split(r'(?<=[.!?])\s+', context)
                    cand_sent = ""
                    for s in sentences:
                        if span_text in s and len(s.strip()) > 30:
                            cand_sent = clean_sentence_for_qa(s)
                            break

                    # Reject if containing sentence is boilerplate
                    if cand_sent and any(re.search(bp, cand_sent, re.IGNORECASE) for bp in boilerplate_patterns):
                        continue

                    if score > best_score:
                        best_score = score
                        best_answer = span_text
                        best_idx = i
                        best_passage = p
                        best_full_sent = cand_sent
        except Exception:
            continue

    if best_answer and best_score > 0.05 and best_full_sent:
        formatted_answer = f"{best_full_sent} [E{best_idx}]"

        return {
            "answer": formatted_answer,
            "confidence": round(float(best_score), 4),
            "best_passage_idx": best_idx,
            "extracted_span": best_answer,
            "model": f"Neural Model ({model_name})"
        }
    else:
        # Graceful fallback to Cross-Encoder Semantic Comprehension
        return synthesize_cross_encoder_answer(question, passages)


class ResearchRAG:
    def __init__(
        self,
        retriever,
        llm_model_name: str = DEFAULT_LOCAL_MODEL,
        qa_model_name: str = DEFAULT_QA_MODEL,
        gemini_api_key: Optional[str] = None
    ):
        self.retriever = retriever
        self.llm_model_name = llm_model_name
        self.qa_model_name = qa_model_name
        self.gemini_api_key = gemini_api_key

    def build_prompt(self, question: str, passages: List[Dict[str, Any]]) -> str:
        """Constructs a strict academic provenance prompt."""
        evidence_blocks = []
        for i, p in enumerate(passages, start=1):
            evidence_blocks.append(
                f"[E{i}] Document: {p.get('document', 'Unknown')} (Page {p.get('page', '?')})\n"
                f"Content: {p.get('text', '')}"
            )

        evidence_str = "\n\n".join(evidence_blocks)

        system_prompt = (
            "You are ResearchMate, an academic assistant specialized in analyzing scientific papers.\n"
            "Rules:\n"
            "1. Answer the question using ONLY the provided evidence passages.\n"
            "2. Cite evidence using tags like [E1], [E2] directly next to each claim.\n"
            "3. If the passages do not contain enough facts to answer, explicitly state: 'The provided paper passages do not contain sufficient evidence to answer this question.'\n"
            "4. Never extrapolate, hallucinate, or assume unstated facts.\n"
        )

        user_content = f"Evidence Passages:\n{evidence_str}\n\nQuestion: {question}\n\nAcademic Answer:"
        return f"{system_prompt}\n\n{user_content}"

    def synthesize_fallback_answer(self, question: str, passages: List[Dict[str, Any]]) -> str:
        """
        Synthesizes direct answer using the Cross-Encoder reranker.
        Falls back to semantic sentence selection.
        """
        if not passages:
            return "The provided paper passages do not contain sufficient evidence to answer this question."

        try:
            reranker = getattr(self.retriever, "get_reranker", lambda: None)()
            res = synthesize_cross_encoder_answer(question, passages, reranker=reranker)
            if res.get("answer"):
                return res["answer"]
        except Exception:
            pass

        # Fallback to first informative sentence
        for i, p in enumerate(passages, start=1):
            sents = re.split(r'(?<=[.!?])\s+', p.get("text", ""))
            for s in sents:
                clean_s = clean_sentence_for_qa(s)
                if len(clean_s) > 40 and not clean_s.lower().startswith(("table", "figure", "page")):
                    return f"{clean_s} [E{i}]"

        return f"According to the retrieved paper text, {passages[0]['text'][:200].strip()}... [E1]"

    def answer_question(
        self,
        question: str,
        retrieval_method: str = "hybrid_rerank",
        engine: Optional[str] = None,
        top_k: int = 4,
        verify: bool = True,
        gemini_api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Full end-to-end question answering pipeline:
        Retrieval -> QA Inference / Comprehension Synthesis -> Evidence Verification.
        Available Engines:
          - 'gemini': Google Gemini GenAI (Recommended, deep academic comprehension)
          - 'semantic_synthesizer': Cross-Encoder Semantic Comprehension (Local default, fast and accurate)
          - 'seq2seq_transformer': Local Generative Transformer (DistilBART)
          - 'hf_transformer': Extractive Span Reader (DistilBERT SQuAD)
          - 'synthesis': Fast Lexical Sentence Synthesizer
        """

        # 1. Retrieve evidence passages
        if retrieval_method == "bm25":
            passages = self.retriever.search_bm25(question, top_k=top_k)
        elif retrieval_method == "dense":
            passages = self.retriever.search_dense(question, top_k=top_k)
        elif retrieval_method == "hybrid":
            passages = self.retriever.search_hybrid(question, top_k=top_k)
        elif retrieval_method == "hybrid_rerank":
            candidates = self.retriever.search_hybrid(question, top_k=max(12, top_k * 3), candidate_k=25)
            passages = self.retriever.rerank(question, candidates, top_k=top_k)
        else:
            raise ValueError(f"Unknown retrieval method: {retrieval_method}")

        # 2. Select appropriate QA Engine
        api_key = gemini_api_key or self.gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if engine is None:
            if api_key:
                selected_engine = "gemini"
            else:
                selected_engine = "semantic_synthesizer"
        else:
            selected_engine = engine

        qa_meta = {}
        raw_answer = ""

        # 3. Generate answer based on selected engine
        if selected_engine == "gemini":
            gemini_res = generate_gemini_answer(question, passages, api_key=api_key)
            raw_answer = gemini_res["answer"]
            qa_meta = gemini_res
            if not raw_answer or "is not configured" in raw_answer:
                # Fall back to local semantic synthesizer if Gemini is unconfigured or failed
                local_res = synthesize_cross_encoder_answer(
                    question,
                    passages,
                    reranker=getattr(self.retriever, "get_reranker", lambda: None)()
                )
                if not raw_answer:
                    raw_answer = local_res["answer"]
                    qa_meta = local_res
                else:
                    qa_meta["fallback_local"] = local_res["answer"]

        elif selected_engine in ("semantic_synthesizer", "synthesis"):
            reranker = getattr(self.retriever, "get_reranker", lambda: None)()
            sem_res = synthesize_cross_encoder_answer(question, passages, reranker=reranker)
            raw_answer = sem_res["answer"]
            qa_meta = sem_res

        elif selected_engine == "seq2seq_transformer":
            seq_res = synthesize_seq2seq_answer(question, passages, model_name=self.llm_model_name)
            if seq_res.get("answer"):
                raw_answer = seq_res["answer"]
                qa_meta = seq_res
            else:
                raw_answer = self.synthesize_fallback_answer(question, passages)
                qa_meta = {"model": "Semantic Synthesizer Fallback"}

        elif selected_engine == "hf_transformer":
            try:
                qa_res = extract_qa_transformer(question, passages, model_name=self.qa_model_name)
                if qa_res.get("answer"):
                    raw_answer = qa_res["answer"]
                    qa_meta = qa_res
                else:
                    # Fallback to semantic cross-encoder
                    sem_res = synthesize_cross_encoder_answer(
                        question,
                        passages,
                        reranker=getattr(self.retriever, "get_reranker", lambda: None)()
                    )
                    raw_answer = sem_res["answer"]
                    qa_meta = sem_res
            except Exception as e:
                raw_answer = self.synthesize_fallback_answer(question, passages)
                qa_meta = {"error": str(e), "model": f"Neural Model ({self.qa_model_name})"}

        else:
            raw_answer = self.synthesize_fallback_answer(question, passages)
            qa_meta = {"model": "Cross-Encoder Semantic Synthesizer"}

        # 4. Extract citations and resolve provenance
        citation_tags = re.findall(r'\[E(\d+)\]', raw_answer)
        cited_indices = sorted(list(set(int(tag) for tag in citation_tags)))

        citations = []
        for idx in cited_indices:
            if 1 <= idx <= len(passages):
                p = passages[idx - 1]
                citations.append({
                    "tag": f"[E{idx}]",
                    "document": p.get("document", "Unknown"),
                    "page": p.get("page", 1),
                    "chunk_id": p.get("chunk_id", ""),
                    "snippet": p.get("text", "")[:200] + "..."
                })

        # If answer has no citations but passages exist, cite top passage [E1]
        if not citations and passages and "not contain sufficient evidence" not in raw_answer:
            citations.append({
                "tag": "[E1]",
                "document": passages[0].get("document", "Unknown"),
                "page": passages[0].get("page", 1),
                "chunk_id": passages[0].get("chunk_id", ""),
                "snippet": passages[0].get("text", "")[:200] + "..."
            })
            if not raw_answer.endswith("[E1]"):
                raw_answer = f"{raw_answer} [E1]"

        # 5. NLI Verification Check
        verification_result = None
        if verify and "not contain sufficient evidence" not in raw_answer:
            from .verifier import verify_answer
            verification_result = verify_answer(raw_answer, passages)

        return {
            "question": question,
            "answer": raw_answer,
            "engine": selected_engine,
            "qa_metadata": qa_meta,
            "retrieval_method": retrieval_method,
            "passages": passages,
            "citations": citations,
            "verification": verification_result
        }
