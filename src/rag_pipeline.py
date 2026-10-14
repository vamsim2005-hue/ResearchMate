"""
Grounded RAG Pipeline Module for ResearchMate
Orchestrates multi-stage retrieval, evidence-grounded prompt assembly,
generative answer synthesis with citations [E1], [E2], and NLI verification.
"""

from typing import List, Dict, Any, Optional
import torch
import re

DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_QA_MODEL = "distilbert/distilbert-base-cased-distilled-squad"

_generator_model = None
_generator_tokenizer = None
_qa_model = None
_qa_tokenizer = None


def get_qa_pipeline(model_name: str = DEFAULT_QA_MODEL):
    """Lazily load Hugging Face pretrained Question Answering transformer."""
    global _qa_model, _qa_tokenizer
    if _qa_model is None:
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        _qa_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        _qa_model.eval()
        if torch.cuda.is_available():
            _qa_model = _qa_model.cuda()
    return _qa_tokenizer, _qa_model


def extract_qa_transformer(
    question: str,
    passages: List[Dict[str, Any]],
    model_name: str = DEFAULT_QA_MODEL
) -> Dict[str, Any]:
    """
    Applies a pretrained Hugging Face Question-Answering Transformer
    (e.g., distilbert/distilbert-base-cased-distilled-squad) across retrieved passages.
    Extracts the highest-confidence answer span directly from the evidence text and links
    it to its provenance citation [E1], [E2], etc.
    """
    if not passages:
        return {
            "answer": "The provided paper passages do not contain sufficient evidence to answer this question.",
            "confidence": 0.0,
            "best_passage_idx": None,
            "extracted_span": "",
            "model": f"Hugging Face ({model_name})"
        }

    tokenizer, model = get_qa_pipeline(model_name)

    best_score = -1e9
    best_answer = ""
    best_idx = 1
    best_passage = passages[0]
    best_full_sent = ""

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

            # Mask out CLS token (index 0) or question tokens
            # Find sequence ids: only consider context tokens
            seq_ids = inputs.sequence_ids(0) if hasattr(inputs, "sequence_ids") else None

            start_idx = torch.argmax(start_logits).item()
            end_idx = torch.argmax(end_logits).item()

            if seq_ids is not None:
                # Ensure start and end indices are within context tokens (sequence_id == 1)
                valid_indices = [idx for idx, s_id in enumerate(seq_ids) if s_id == 1]
                if valid_indices:
                    # Filter logits to valid context tokens
                    sub_start_logits = start_logits.clone()
                    sub_end_logits = end_logits.clone()
                    for idx in range(len(start_logits)):
                        if seq_ids[idx] != 1:
                            sub_start_logits[idx] = -1e9
                            sub_end_logits[idx] = -1e9
                    start_idx = torch.argmax(sub_start_logits).item()
                    end_idx = torch.argmax(sub_end_logits).item()

            if end_idx >= start_idx and (end_idx - start_idx) < 50:
                answer_tokens = inputs["input_ids"][0][start_idx: end_idx + 1]
                span_text = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

                # Approximate confidence via softmax over start and end
                start_prob = torch.softmax(start_logits, dim=-1)[start_idx].item()
                end_prob = torch.softmax(end_logits, dim=-1)[end_idx].item()
                score = start_prob * end_prob

                # Check if non-trivial answer
                if len(span_text) > 3 and not span_text.lower().startswith(("[cls]", "[sep]", "<s>")):
                    if score > best_score:
                        best_score = score
                        best_answer = span_text
                        best_idx = i
                        best_passage = p

                        # Locate sentence containing the span for context
                        sentences = re.split(r'(?<=[.!?])\s+', context)
                        for s in sentences:
                            if span_text in s:
                                best_full_sent = s.strip()
                                break
        except Exception:
            continue

    if best_answer and best_score > 0.005:
        # Construct grounded, cited response
        if best_full_sent and best_full_sent != best_answer:
            formatted_answer = f"According to {best_passage.get('document', 'the paper')}, {best_full_sent} [E{best_idx}]"
        else:
            formatted_answer = f"The paper indicates that {best_answer}. [E{best_idx}]"

        return {
            "answer": formatted_answer,
            "confidence": round(float(best_score), 4),
            "best_passage_idx": best_idx,
            "extracted_span": best_answer,
            "model": f"Hugging Face ({model_name})"
        }
    else:
        return {
            "answer": "",
            "confidence": 0.0,
            "best_passage_idx": None,
            "extracted_span": "",
            "model": f"Hugging Face ({model_name})"
        }


def get_generator(model_name: str = DEFAULT_LOCAL_MODEL):
    """Lazily load generative LLM and tokenizer."""
    global _generator_model, _generator_tokenizer
    if _generator_model is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        _generator_tokenizer = AutoTokenizer.from_pretrained(model_name)
        if torch.cuda.is_available():
            _generator_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            _generator_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            )
        _generator_model.eval()
    return _generator_tokenizer, _generator_model


class ResearchRAG:
    def __init__(
        self,
        retriever,
        llm_model_name: str = DEFAULT_LOCAL_MODEL,
        qa_model_name: str = DEFAULT_QA_MODEL
    ):
        self.retriever = retriever
        self.llm_model_name = llm_model_name
        self.qa_model_name = qa_model_name
        self.use_neural_llm = False
        self.use_transformer_qa = True  # Use Hugging Face QA Transformer by default

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
        Instant fallback synthesizer: extracts and combines the most direct factual
        sentences from top passages when large causal LLM weights are offline.
        """
        if not passages:
            return "The provided paper passages do not contain sufficient evidence to answer this question."

        q_terms = set(re.findall(r'\b\w{3,}\b', question.lower()))
        answer_parts = []

        for i, p in enumerate(passages[:3], start=1):
            sentences = re.split(r'(?<=[.!?])\s+', p.get("text", ""))
            scored_sentences = []
            for s in sentences:
                s_words = set(re.findall(r'\b\w{3,}\b', s.lower()))
                overlap = len(s_words & q_terms)
                if overlap > 0 and len(s) > 25:
                    scored_sentences.append((overlap, s.strip()))

            if scored_sentences:
                scored_sentences.sort(key=lambda x: x[0], reverse=True)
                top_sent = scored_sentences[0][1]
                answer_parts.append(f"{top_sent} [E{i}]")

        if answer_parts:
            return " ".join(answer_parts)
        else:
            first_sent = passages[0]["text"][:200].strip()
            return f"According to the retrieved paper text, {first_sent}... [E1]"

    def generate_neural_answer(self, prompt: str, max_new_tokens: int = 250) -> str:
        """Generates answer using loaded causal LLM."""
        tokenizer, model = get_generator(self.llm_model_name)
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return gen_text.strip()

    def answer_question(
        self,
        question: str,
        retrieval_method: str = "hybrid_rerank",
        engine: Optional[str] = None,
        top_k: int = 4,
        verify: bool = True
    ) -> Dict[str, Any]:
        """
        Full end-to-end question answering pipeline:
        Retrieval -> QA Inference / Generation -> Evidence Verification.
        Engines:
          - 'hf_transformer': Pretrained Hugging Face QA Transformer (distilbert-base-cased-distilled-squad)
          - 'neural_llm': Causal/Seq2Seq LLM
          - 'synthesis': Lexical sentence overlap synthesizer
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

        # 2. Generate answer
        selected_engine = engine or ("hf_transformer" if self.use_transformer_qa else ("neural_llm" if self.use_neural_llm else "synthesis"))
        qa_meta = {}

        if selected_engine == "hf_transformer":
            try:
                qa_res = extract_qa_transformer(question, passages, model_name=self.qa_model_name)
                qa_meta = qa_res
                if qa_res.get("answer"):
                    raw_answer = qa_res["answer"]
                else:
                    raw_answer = self.synthesize_fallback_answer(question, passages)
            except Exception as e:
                raw_answer = self.synthesize_fallback_answer(question, passages)
                qa_meta = {"error": str(e), "model": f"Hugging Face ({self.qa_model_name})"}
        elif selected_engine == "neural_llm":
            try:
                prompt = self.build_prompt(question, passages)
                raw_answer = self.generate_neural_answer(prompt)
                qa_meta = {"model": f"Hugging Face ({self.llm_model_name})"}
            except Exception:
                raw_answer = self.synthesize_fallback_answer(question, passages)
                qa_meta = {"model": "Synthesis Fallback"}
        else:
            raw_answer = self.synthesize_fallback_answer(question, passages)
            qa_meta = {"model": "Provenance Sentence Synthesizer"}


        # 3. Extract citations
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

        # 4. Verification Check
        verification_result = None
        if verify:
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

