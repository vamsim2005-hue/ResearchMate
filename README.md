# ResearchMate 📚
### An Intelligent Research Paper Analyzer Using NLP and Hybrid RAG with Evidence Verification

**ResearchMate** is an end-to-end NLP and Retrieval-Augmented Generation (RAG) platform designed for rigorous academic paper analysis. Beyond standard chat interfaces, ResearchMate features an experimental Information Retrieval (IR) benchmarking engine and a post-generation Natural Language Inference (NLI) evidence verifier that flags unsupported assertions.

---## 🌟 Key Capabilities Powered by Pretrained Hugging Face Transformers

ResearchMate leverages specialized, state-of-the-art pretrained transformer models from Hugging Face for each stage of scientific document understanding:

| Task | Pretrained Hugging Face Model | Description |
|---|---|---|
| **Named Entity Recognition (NER)** | `dslim/bert-base-NER` | Contextual token-level BERT model extracting scientific authors, organizations, locations, and academic entities with confidence scores. |
| **Zero-Shot Paper Classification** | `cross-encoder/nli-deberta-v3-small` | Categorizes research papers into scientific disciplines and methodologies without requiring training data. |
| **Question-Answering & QA Span Extraction** | `distilbert/distilbert-base-cased-distilled-squad` | Pinpoints precise evidence spans and answers directly from retrieved text chunks with exact confidence scores. |
| **Abstractive Summarization** | `sshleifer/distilbart-cnn-12-6` | Generates concise, executive scientific summaries of research papers using Seq2Seq attention. |
| **Dense Semantic Vector Search** | `sentence-transformers/all-MiniLM-L6-v2` | Maps passages to dense vector space for semantic FAISS cosine similarity retrieval. |
| **Neural Cross-Encoder Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Computes full cross-attention between queries and candidate passages to maximize Top-K precision. |
| **NLI Hallucination & Evidence Verification** | `cross-encoder/nli-deberta-v3-small` | Decomposes answers into claims and verifies entailment against source paper passages. |

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
         | 🤗 Hugging Face Transformers|             |    Intra-Page Chunking      |
         |  - dslim/bert-base-NER      |             |  (Document + Page Metadata) |
         |  - DeBERTa Zero-Shot Clf    |             +-----------------------------+
         |  - DistilBART Summarizer    |                            |
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
                                                     | (ms-marco-MiniLM-L-6-v2)    |
                                                     +-----------------------------+
                                                                    |
                                                                    v
                                                     +-----------------------------+
                                                     | 🤗 HF QA Transformer Span   |
                                                     | (distilbert-squad QA)       |
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
│   ├── ner_keywords.py            # Pretrained BERT-NER & TF-IDF keywords
│   ├── classifier.py              # Pretrained DeBERTa Zero-Shot paper classifier
│   ├── summarizer.py              # Extractive & DistilBART abstractive summarization
│   ├── retriever.py               # BM25, Dense (FAISS), RRF, and Cross-Encoder
│   ├── verifier.py                # DeBERTa NLI claim verification & hallucination guard
│   ├── rag_pipeline.py            # HF Question-Answering Transformer & citations
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
