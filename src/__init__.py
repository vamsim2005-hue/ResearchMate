"""ResearchMate NLP and RAG Package"""
from .pdf_processor import extract_pdf_text, get_pdf_info
from .text_preprocessing import clean_text, chunk_pages
from .ner_keywords import extract_keywords, extract_entities, extract_entities_transformer
from .classifier import classify_paper, classify_text
from .summarizer import generate_extractive_summary, generate_abstractive_summary
from .retriever import ResearchRetriever
from .verifier import verify_claim, verify_answer
from .rag_pipeline import ResearchRAG, extract_qa_transformer
from .evaluation import evaluate_retrieval, hit_at_k, recall_at_k, reciprocal_rank, ndcg_at_k

__all__ = [
    "extract_pdf_text",
    "get_pdf_info",
    "clean_text",
    "chunk_pages",
    "extract_keywords",
    "extract_entities",
    "extract_entities_transformer",
    "classify_paper",
    "classify_text",
    "generate_extractive_summary",
    "generate_abstractive_summary",
    "ResearchRetriever",
    "verify_claim",
    "verify_answer",
    "ResearchRAG",
    "extract_qa_transformer",
    "evaluate_retrieval",
    "hit_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "ndcg_at_k"
]

