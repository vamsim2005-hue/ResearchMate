"""
Integration test for ResearchMate complete pipeline.
Tests PDF extraction, chunking, all 4 retrieval methods, RAG Q&A with citations,
NLI verification, and IR evaluation metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pdf_processor import extract_pdf_text, get_pdf_info
from src.text_preprocessing import chunk_pages
from src.ner_keywords import extract_keywords, extract_entities, extract_entities_transformer
from src.classifier import classify_paper
from src.summarizer import generate_extractive_summary, generate_abstractive_summary
from src.retriever import ResearchRetriever
from src.rag_pipeline import ResearchRAG, extract_qa_transformer
from src.verifier import verify_claim, verify_answer
from src.evaluation import evaluate_retrieval
from src.benchmark_data import generate_benchmark_queries_from_chunks


def test_pipeline():
    print("=== Step 1: Ingestion & Preprocessing ===")
    pdf_path = Path("data/pdfs/RP163_The-Human-Right-to-Science_EN.pdf")
    assert pdf_path.exists(), f"PDF not found: {pdf_path}"

    info = get_pdf_info(pdf_path)
    print(f"Loaded: {info['filename']} ({info['total_pages']} pages)")

    pages = extract_pdf_text(pdf_path)
    # Test on first 12 pages for fast verification
    test_pages = pages[:12]
    chunks = chunk_pages(test_pages, chunk_size=300, overlap=40)
    print(f"Generated {len(chunks)} chunks across {len(test_pages)} pages.")
    assert len(chunks) > 0

    print("\n=== Step 2: NLP Features & Pretrained Transformers ===")
    kw = extract_keywords([c["text"] for c in chunks[:10]], top_k=6)
    print(f"Top Keywords: {[k['keyword'] for k in kw]}")

    print("Testing Hugging Face Pretrained Transformer NER (dslim/bert-base-NER)...")
    tf_ent = extract_entities_transformer(test_pages[1]["text"], top_k=5)
    print(f"Transformer NER Entities: {tf_ent}")
    assert len(tf_ent) > 0, "Transformer NER returned no entities"

    print("Testing Hugging Face Zero-Shot Classifier (cross-encoder/nli-deberta-v3-small)...")
    clf_res = classify_paper(test_pages[:3])
    print(f"Primary Discipline: {clf_res['primary_discipline']} (Confidence: {clf_res['discipline_confidence']:.2f})")
    print(f"Methodology: {clf_res['primary_methodology']} (Confidence: {clf_res['methodology_confidence']:.2f})")
    assert "primary_discipline" in clf_res

    summary = generate_extractive_summary(test_pages, num_sentences=3)
    print(f"Extractive Summary: {summary[:100]}...")

    print("Testing Hugging Face Abstractive Summarization (sshleifer/distilbart-cnn-12-6)...")
    abs_summary = generate_abstractive_summary(test_pages[0]["text"][:600], max_length=60, min_length=15)
    print(f"Abstractive Summary: {abs_summary[:120]}...")
    assert len(abs_summary) > 10

    print("\n=== Step 3: Retrieval Indexing & Comparison ===")
    retriever = ResearchRetriever(chunks)

    query = "What is the role of UNESCO in human rights and science?"
    all_results = retriever.search_all_methods(query, top_k=3)

    for method_name, res_list in all_results.items():
        print(f"\nMethod: {method_name}")
        for r in res_list:
            print(f"  - [{r['chunk_id']} | p.{r['page']} | Score: {r['score']}] {r['text'][:90]}...")

    print("\n=== Step 4: Hugging Face QA Transformer & Citations ===")
    rag = ResearchRAG(retriever)
    rag_result = rag.answer_question(query, retrieval_method="hybrid_rerank", engine="hf_transformer", top_k=3, verify=False)
    print(f"Question: {rag_result['question']}")
    print(f"Engine: {rag_result.get('engine')}")
    print(f"Answer: {rag_result['answer']}")
    print(f"Citations: {rag_result['citations']}")
    assert len(rag_result["answer"]) > 0


    print("\n=== Step 5: NLI Evidence Verification ===")
    test_claim = "UNESCO plays a significant role in science policy."
    evidence_text = chunks[0]["text"]
    claim_ver = verify_claim(test_claim, evidence_text)
    print(f"Claim: '{test_claim}'")
    print(f"Verification: {claim_ver['label']} (Confidence: {claim_ver['confidence']})")

    full_ver = verify_answer(rag_result["answer"], rag_result["passages"])
    print(f"Answer Faithfulness: {full_ver['faithfulness_score']}% (Flagged: {full_ver['flagged']})")

    print("\n=== Step 6: Benchmark & Quantitative Evaluation ===")
    bench_queries = generate_benchmark_queries_from_chunks(chunks, num_queries=6)
    print(f"Generated {len(bench_queries)} benchmark queries.")

    eval_df = evaluate_retrieval(retriever, bench_queries, methods=["bm25", "dense", "hybrid"], k_eval=3)
    print("\nEvaluation Benchmark Results:")
    print(eval_df.to_string(index=False))

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(results_dir / "retrieval_benchmark_results.csv", index=False)
    print(f"\nResults successfully saved to {results_dir / 'retrieval_benchmark_results.csv'}")

    print("\n>>> ALL PIPELINE TESTS PASSED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    test_pipeline()
