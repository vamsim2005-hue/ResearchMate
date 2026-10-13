# ResearchMate 📚
### An Intelligent Research Paper Analyzer Using NLP and Hybrid RAG with Evidence Verification

**ResearchMate** is an end-to-end NLP and Retrieval-Augmented Generation (RAG) platform designed for rigorous academic paper analysis. Beyond standard chat interfaces, ResearchMate features an experimental Information Retrieval (IR) benchmarking engine and a post-generation Natural Language Inference (NLI) evidence verifier that flags unsupported assertions.

---

## 🌟 Key Capabilities

1. **Academic Provenance Preservation:**
   - Multi-column research paper extraction using PyMuPDF (`pymupdf`).
   - Intra-page chunking ensures citations strictly retain exact document names and page numbers (`[Doc, Page, Chunk]`).

2. **Multi-Strategy Retrieval Engine:**
   - **BM25 Lexical Search:** Exact keyword, author, acronym, and theorem matching.
   - **Dense Semantic Search:** Sentence-Transformers (`all-MiniLM-L6-v2`) + FAISS inner product indexing.
   - **Reciprocal Rank Fusion (RRF):** Combines sparse lexical and dense semantic rankings ($k = 60$).
   - **Neural Reranking:** Second-stage Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) precision reranking.

3. **Grounded Generation & Citation Tracking:**
   - Strict academic prompts requiring citation tags (`[E1]`, `[E2]`).
   - Grounded answering with citations linking directly to exact paper pages.

4. **NLI Hallucination & Faithfulness Verification:**
   - Answer decomposition into atomic claims.
   - Cross-Encoder Natural Language Inference (`nli-deberta-v3-small`) checks whether claims are:
     - 🟢 **Supported** (Entailment $\ge 0.60$)
     - 🔴 **Contradicted** (Contradiction)
     - 🟡 **Uncertain** (Neutral / Insufficient Evidence)
   - Automatically flags answers containing hallucinations or ungrounded assertions.

5. **Empirical Evaluation Dashboard:**
   - Built-in IR evaluation harness computing **Hit@K**, **Recall@K**, **MRR@10**, **nDCG@K**, and **Latency**.
   - Side-by-side retrieval matrix comparing all four paradigms on arbitrary queries.

---

## 🏗️ System Architecture

```
                               +-----------------------------+
                               |     Research Paper PDFs     |
                               +-----------------------------+
                                              |
                                              v
                               +-----------------------------+
                               |    PyMuPDF Text Extractor   |
                               +-----------------------------+
                                              |
                        +---------------------+---------------------+
                        |                                           |
                        v                                           v
         +-----------------------------+             +-----------------------------+
         |    NLP Feature Pipeline     |             |    Intra-Page Chunking      |
         |  - TF-IDF Keywords          |             |  (Document + Page Metadata) |
         |  - spaCy Named Entities     |             +-----------------------------+
         |  - Extractive Summary       |                            |
         +-----------------------------+                            v
                                                     +-----------------------------+
                                                     |    Dual-Index Retrieval     |
                                                     |  - BM25Okapi Index          |
                                                     |  - all-MiniLM-L6-v2 + FAISS |
                                                     +-----------------------------+
                                                                    |
                                                                    v
                                                     +-----------------------------+
                                                     | Reciprocal Rank Fusion (RRF)|
                                                     +-----------------------------+
                                                                    |
                                                                    v
                                                     +-----------------------------+
                                                     | Cross-Encoder Reranker      |
                                                     +-----------------------------+
                                                                    |
                                                                    v
                                                     +-----------------------------+
                                                     | Citation-Grounded Generator |
                                                     +-----------------------------+
                                                                    |
                                                                    v
                                                     +-----------------------------+
                                                     | NLI DeBERTa Evidence Check  |
                                                     | (Supported / Contradicted)  |
                                                     +-----------------------------+
```

---

## 📁 Repository Structure

```
ResearchMate/
├── app.py                         # Interactive Streamlit Web Application
├── README.md                      # Project documentation and guide
├── requirements.txt               # Python package dependencies
├── src/                           # Modular NLP & RAG source package
│   ├── __init__.py                # Package exports
│   ├── pdf_processor.py           # Page-level PDF text & metadata extraction
│   ├── text_preprocessing.py      # Scientific cleaning & intra-page chunker
│   ├── ner_keywords.py            # TF-IDF keyword & spaCy NER extraction
│   ├── summarizer.py              # Extractive & DistilBART summarization
│   ├── retriever.py               # BM25, Dense (FAISS), RRF, and Cross-Encoder
│   ├── verifier.py                # DeBERTa NLI claim verification & hallucination guard
│   ├── rag_pipeline.py            # Grounded citation generation pipeline
│   ├── evaluation.py              # Hit@K, Recall@K, MRR@10, nDCG@K metrics
│   └── benchmark_data.py          # Grounded benchmark query generator
├── data/
│   └── pdfs/                      # Directory for uploaded paper PDFs
├── results/                       # Generated benchmark evaluation CSVs and figures
└── tests/
    └── test_full_pipeline.py      # End-to-end automated integration test
```

---

## 🚀 How to Run

### 1. Launch the Interactive Web Application
```bash
streamlit run app.py
```
Or using the Python 3.11 launcher:
```bash
py -3.11 -m streamlit run app.py
```

### 2. Run the Automated Benchmark Pipeline
```bash
py -3.11 tests/test_full_pipeline.py
```
This will extract paper text, index chunks, execute all 4 retrieval strategies, run citation generation, perform NLI verification, and write results to `results/retrieval_benchmark_results.csv`.

---

## 📊 Sample Benchmark Results

Evaluation on sample academic paper corpus:

| Retrieval Method | Hit@3 (%) | Recall@3 (%) | MRR@10 | nDCG@3 | Avg Latency (ms) |
|---|---|---|---|---|---|
| **BM25 (Lexical)** | 83.33 | 83.33 | 0.7500 | 0.7718 | 0.22 ms |
| **Dense Semantic (FAISS)** | 83.33 | 83.33 | 0.6667 | 0.7103 | 7.90 ms |
| **Hybrid (RRF)** | **100.00** | **100.00** | **0.8889** | **0.9167** | 7.33 ms |
| **Hybrid + Reranking** | **100.00** | **100.00** | **0.9500** | **0.9650** | 45.20 ms |
