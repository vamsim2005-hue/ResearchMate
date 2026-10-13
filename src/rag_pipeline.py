"""
Grounded RAG Pipeline Module for ResearchMate
Orchestrates multi-stage retrieval, evidence-grounded prompt assembly,
generative answer synthesis with citations [E1], [E2], and NLI verification.
"""

from typing import List, Dict, Any, Optional
import torch
import re

DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
_generator_model = None
_generator_tokenizer = None


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
    def __init__(self, retriever, llm_model_name: str = DEFAULT_LOCAL_MODEL):
        self.retriever = retriever
        self.llm_model_name = llm_model_name
        self.use_neural_llm = False  # Set to True when model weights are cached/requested

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
        top_k: int = 4,
        verify: bool = True
    ) -> Dict[str, Any]:
        """
        Full end-to-end question answering pipeline:
        Retrieval -> Prompt Assembly -> Generation -> Evidence Verification.
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
        prompt = self.build_prompt(question, passages)
        if self.use_neural_llm:
            try:
                raw_answer = self.generate_neural_answer(prompt)
            except Exception:
                raw_answer = self.synthesize_fallback_answer(question, passages)
        else:
            raw_answer = self.synthesize_fallback_answer(question, passages)

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
            "retrieval_method": retrieval_method,
            "passages": passages,
            "citations": citations,
            "verification": verification_result
        }
