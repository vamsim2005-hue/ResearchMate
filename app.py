"""
ResearchMate — Intelligent Research Paper Analyzer
Web Application built with Streamlit, NLP, Hybrid Retrieval, and RAG Evidence Verification.
"""

import os
import sys
from pathlib import Path
import time
import json
import torch
try:
    torch.classes.__path__ = []
except Exception:
    pass

import pandas as pd
import streamlit as st

# Add current workspace to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pdf_processor import extract_pdf_text, get_pdf_info
from src.text_preprocessing import chunk_pages
from src.ner_keywords import extract_keywords, extract_entities, extract_entities_transformer
from src.classifier import classify_paper
from src.summarizer import generate_extractive_summary, generate_abstractive_summary
from src.retriever import ResearchRetriever
from src.rag_pipeline import ResearchRAG
from src.verifier import verify_answer
from src.evaluation import evaluate_retrieval
from src.benchmark_data import generate_benchmark_queries_from_chunks

# Page configuration
st.set_page_config(
    page_title="ResearchMate | Intelligent Research Paper Analyzer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# Direct Gemini API Configuration (gemini-api-dev)
# ==========================================================
# Configure your Gemini API key directly here or pass via GEMINI_API_KEY env variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Custom CSS for modern Generative UI aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Gradient Brand Header */
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 40%, #8b5cf6 70%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
        letter-spacing: -0.025em;
    }

    .sub-header {
        color: #64748b;
        font-size: 1.0rem;
        margin-bottom: 1.6rem;
        font-weight: 450;
    }

    /* Modern Card Container */
    .ui-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .ui-card:hover {
        box-shadow: 0 8px 30px -4px rgba(15, 23, 42, 0.08);
    }

    /* Badges & Pills */
    .hf-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #fde68a;
        margin-bottom: 8px;
    }

    .model-pill {
        background-color: #eef2ff;
        color: #4338ca;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid #c7d2fe;
    }

    .badge-entity {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        margin: 2px 4px 4px 0;
    }

    /* Evidence & Citation Box */
    .retrieval-box {
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 12px;
        border-left: 4px solid #3b82f6;
        border-top: 1px solid #f1f5f9;
        border-right: 1px solid #f1f5f9;
        border-bottom: 1px solid #f1f5f9;
    }

    .citation-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 10px;
        padding: 12px 16px;
        margin-top: 10px;
        font-size: 0.88rem;
    }

    .citation-tag {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        padding: 2px 7px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 6px;
    }

    /* Verification Status Pills */
    .status-supported {
        background-color: #d1fae5;
        color: #065f46;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border: 1px solid #6ee7b7;
    }
    .status-uncertain {
        background-color: #fef3c7;
        color: #92400e;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border: 1px solid #fcd34d;
    }
    .status-contradicted {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border: 1px solid #fca5a5;
    }

    /* Streamlit Tab and Button Polish */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f8fafc;
        padding: 6px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #2563eb !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = GEMINI_API_KEY
if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = {}  # {filename: {"pages": [...], "info": {...}}}
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag" not in st.session_state:
    st.session_state.rag = None
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []


# Sidebar: PDF Ingestion & Settings
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/books.png", width=64)
    st.title("ResearchMate")
    st.caption("Intelligent RAG Analyzer for Academic Papers")

    st.markdown("---")
    st.subheader("📁 Upload Research Papers")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload academic papers (.pdf). Multi-column papers are supported."
    )

    pdf_dir = Path("data/pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Process uploads
    if uploaded_files:
        for uploaded_file in uploaded_files:
            dest_path = pdf_dir / uploaded_file.name
            with open(dest_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

    # Scan existing PDFs in data/pdfs/
    existing_pdfs = sorted(list(pdf_dir.glob("*.pdf")))
    st.markdown(f"**Found {len(existing_pdfs)} paper(s) in repository.**")

    # Chunking configuration
    with st.expander("⚙️ Chunking & Indexing Settings"):
        chunk_size = st.slider("Chunk size (words)", min_value=150, max_value=600, value=300, step=50)
        overlap = st.slider("Chunk overlap (words)", min_value=20, max_value=120, value=40, step=10)

    # Google Gemini GenAI Status (configured directly via code)
    if GEMINI_API_KEY:
        st.markdown("""
        <div style="background:#ecfdf5; border:1px solid #a7f3d0; border-radius:12px; padding:10px 14px; margin-top:10px; margin-bottom:12px;">
            <div style="font-weight:700; color:#065f46; font-size:0.84rem; display:flex; align-items:center; gap:6px;">
                <span>✨</span> Gemini 3.8 Flash Active
            </div>
            <div style="font-size:0.75rem; color:#047857; margin-top:2px;">Direct API configuration active</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#fef3c7; border:1px solid #fde68a; border-radius:12px; padding:10px 14px; margin-top:10px; margin-bottom:12px;">
            <div style="font-weight:700; color:#92400e; font-size:0.84rem;">
                ℹ️ Local Neural Models Active
            </div>
            <div style="font-size:0.75rem; color:#b45309; margin-top:2px;">Using Cross-Encoder / DeBERTa</div>
        </div>
        """, unsafe_allow_html=True)

    # Index button
    if st.button("🚀 Process & Build Index", type="primary", use_container_width=True) or (existing_pdfs and not st.session_state.chunks):
        with st.spinner("Extracting text and building dual BM25 + FAISS index..."):
            all_pages = []
            all_chunks = []
            pdf_data = {}

            for pdf_path in existing_pdfs:
                try:
                    info = get_pdf_info(pdf_path)
                    pages = extract_pdf_text(pdf_path)
                    pdf_data[pdf_path.name] = {"info": info, "pages": pages}
                    all_pages.extend(pages)
                except Exception as e:
                    st.error(f"Error processing {pdf_path.name}: {e}")

            if all_pages:
                all_chunks = chunk_pages(all_pages, chunk_size=chunk_size, overlap=overlap)
                st.session_state.pdf_data = pdf_data
                st.session_state.chunks = all_chunks

                # Initialize retriever & RAG
                retriever = ResearchRetriever(all_chunks)
                rag = ResearchRAG(retriever, gemini_api_key=st.session_state.get("gemini_api_key"))

                st.session_state.retriever = retriever
                st.session_state.rag = rag
                st.success(f"Indexed {len(pdf_data)} paper(s) into {len(all_chunks)} chunks!")

    if st.session_state.chunks:
        st.info(f"📊 **Active Corpus:**\n- Papers: {len(st.session_state.pdf_data)}\n- Total Chunks: {len(st.session_state.chunks)}")



# Header
st.markdown('<div class="main-header">ResearchMate: Intelligent Research Paper Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Evaluating Lexical vs. Dense vs. Hybrid Retrieval with Grounded Citations & NLI Hallucination Verification</div>', unsafe_allow_html=True)

if not st.session_state.chunks:
    st.warning("👈 Please upload one or more research-paper PDFs in the sidebar and click **Process & Build Index** to begin.")
    st.stop()


# Main Tabs
tabs = st.tabs([
    "📄 Paper Intelligence",
    "🔍 Retrieval Comparison",
    "💬 Grounded Q&A & Verification",
    "📊 Evaluation Dashboard"
])


# ==========================================================
# TAB 1: PAPER INTELLIGENCE
# ==========================================================
with tabs[0]:
    st.subheader("Document Insights & Metadata")
    paper_options = list(st.session_state.pdf_data.keys())
    selected_paper = st.selectbox("Select Paper to Inspect:", paper_options)

    if selected_paper:
        doc_entry = st.session_state.pdf_data[selected_paper]
        info = doc_entry["info"]
        pages = doc_entry["pages"]

        # Metric stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Pages", info["total_pages"])
        col2.metric("File Size", f"{info['file_size_kb']} KB")
        col3.metric("Extracted Pages", len(pages))
        doc_chunks = [c for c in st.session_state.chunks if c["document"] == selected_paper]
        col4.metric("Total Chunks", len(doc_chunks))

        st.markdown("---")

        # Zero-Shot Transformer Paper Classification
        st.markdown("### 🎯 Zero-Shot Academic Classification")
        st.caption("Powered by DeBERTa (`cross-encoder/nli-deberta-v3-small`)")

        class_cache_key = f"classification_{selected_paper}"
        if class_cache_key not in st.session_state:
            with st.spinner("Classifying paper discipline & methodology with Zero-Shot DeBERTa..."):
                st.session_state[class_cache_key] = classify_paper(pages)

        clf_data = st.session_state[class_cache_key]

        c_clf1, c_clf2 = st.columns(2)
        with c_clf1:
            st.markdown(f"**Primary Discipline:** `{clf_data['primary_discipline']}`")
            st.progress(clf_data['discipline_confidence'])
            st.caption(f"Confidence: {clf_data['discipline_confidence']:.1%}")

        with c_clf2:
            st.markdown(f"**Research Methodology:** `{clf_data['primary_methodology']}`")
            st.progress(clf_data['methodology_confidence'])
            st.caption(f"Confidence: {clf_data['methodology_confidence']:.1%}")

        st.markdown("---")

        c_left, c_right = st.columns([1, 1])

        with c_left:
            st.markdown("#### 📝 Executive Paper Summary")
            st.markdown('<span class="hf-badge">distilbart-cnn-12-6</span>', unsafe_allow_html=True)
            summary_type = st.radio("Summary Mode:", ["Fast Extractive", "Neural Abstractive (DistilBART)"], horizontal=True)

            if summary_type == "Fast Extractive":
                extractive_sum = generate_extractive_summary(pages, num_sentences=4)
                st.write(extractive_sum)
            else:
                with st.spinner("Generating abstractive summary with DistilBART..."):
                    first_text = " ".join([p["text"] for p in pages[:3]])
                    abstractive_sum = generate_abstractive_summary(first_text)
                    st.write(abstractive_sum)

            st.markdown("#### 🏷️ Salient Keywords (TF-IDF)")
            keywords = extract_keywords([c["text"] for c in doc_chunks], top_k=12)
            if keywords:
                kw_html = " ".join([f'<span class="badge-entity" style="background:#eff6ff; color:#1e40af; border-color:#bfdbfe;">#{k["keyword"]}</span>' for k in keywords[:12]])
                st.markdown(f'<div style="margin-bottom:8px;">{kw_html}</div>', unsafe_allow_html=True)
                with st.expander("View Keywords Table", expanded=False):
                    kw_df = pd.DataFrame(keywords)
                    st.dataframe(kw_df, use_container_width=True, hide_index=True)

        with c_right:
            st.markdown("#### 🔬 Named Entities Recognition")
            ner_mode = st.radio(
                "NER Model:",
                ["Neural Model (dslim/bert-base-NER)", "spaCy (Statistical)"],
                horizontal=True
            )

            all_text_sample = " ".join([p["text"] for p in pages[:4]])
            if "Neural Model" in ner_mode or "dslim" in ner_mode:
                with st.spinner("Extracting entities with BERT-NER..."):
                    entities = extract_entities_transformer(all_text_sample, top_k=15)
            else:
                entities = extract_entities(all_text_sample, top_k=15, method="spacy")

            if entities:
                badges_html = " ".join([f'<span class="badge-entity">{e["entity"]} <span style="opacity:0.75; font-size:0.7rem;">({e.get("type", "NER")})</span></span>' for e in entities[:15]])
                st.markdown(f'<div style="margin-bottom:8px;">{badges_html}</div>', unsafe_allow_html=True)
                with st.expander("View Entities Table", expanded=False):
                    ent_df = pd.DataFrame(entities)
                    st.dataframe(ent_df, use_container_width=True, hide_index=True)
            else:
                st.info("No salient named entities detected.")

            st.markdown("#### 📖 Browse Extracted Page Text")
            page_nums = [p["page"] for p in pages]
            sel_page_num = st.selectbox("Select Page Number:", page_nums)
            chosen_page_text = next((p["text"] for p in pages if p["page"] == sel_page_num), "")
            st.text_area("Page Text Preview", chosen_page_text, height=220)



# ==========================================================
# TAB 2: MULTI-STRATEGY RETRIEVAL COMPARISON
# ==========================================================
with tabs[1]:
    st.subheader("Side-by-Side Multi-Strategy Retrieval Benchmark")
    st.caption("Compare BM25 (Lexical), Dense Semantic (FAISS), Hybrid (RRF), and Cross-Encoder Reranking.")

    search_query = st.text_input(
        "Enter keyword search or natural-language research question:",
        value="What is the primary contribution and methodology proposed?"
    )
    top_k_slider = st.slider("Passages to retrieve per method (Top-K):", min_value=1, max_value=8, value=3)

    if st.button("🔎 Search Across All 4 Methods", type="primary"):
        retriever = st.session_state.retriever
        with st.spinner("Executing BM25, Dense, Hybrid, and Cross-Encoder Reranking in parallel..."):
            all_res = retriever.search_all_methods(search_query, top_k=top_k_slider)

        cols = st.columns(4)
        methods = list(all_res.keys())

        for idx, col in enumerate(cols):
            m_name = methods[idx]
            res_list = all_res[m_name]

            method_icons = {
                "BM25": "📝 BM25 (Lexical)",
                "Dense": "🌐 Dense (FAISS)",
                "Hybrid": "⚡ Hybrid (RRF)",
                "Hybrid + Rerank": "⭐ Cross-Encoder"
            }
            with col:
                st.markdown(f"#### {method_icons.get(m_name, m_name)}")
                if not res_list:
                    st.warning("No matches found.")
                for rank, item in enumerate(res_list, start=1):
                    badge_style = "background:#dbeafe; color:#1e40af;" if rank == 1 else "background:#f1f5f9; color:#475569;"
                    with st.expander(f"#{rank} [Page {item['page']}] Score: {item['score']}", expanded=(rank == 1)):
                        st.markdown(f'<span style="padding:2px 8px; border-radius:6px; font-size:0.75rem; font-weight:700; {badge_style}">Rank #{rank}</span> <span style="font-size:0.8rem; color:#64748b;">Chunk #{item["chunk_id"]}</span>', unsafe_allow_html=True)
                        st.markdown(f"**Document:** `{item['document']}`")
                        st.markdown(f'<div class="retrieval-box" style="margin-top:6px; font-size:0.88rem; line-height:1.6;">{item["text"]}</div>', unsafe_allow_html=True)


# ==========================================================
# TAB 3: GROUNDED Q&A & EVIDENCE VERIFICATION
# ==========================================================
with tabs[2]:
    st.subheader("Citation-Grounded Question Answering & NLI Hallucination Verification")
    st.caption("Answers are constrained strictly to retrieved paper passages, with exact page citations and claim-by-claim verification.")

    qa_col1, qa_col2, qa_col3 = st.columns([2.5, 1.2, 1.3])
    with qa_col1:
        question_input = st.text_input(
            "Ask a specific question about the paper(s):",
            value="What are the key findings regarding human rights and scientific freedom?"
        )
    with qa_col2:
        method_choice = st.selectbox(
            "Retrieval Strategy:",
            ["hybrid_rerank", "hybrid", "dense", "bm25"],
            format_func=lambda x: {
                "hybrid_rerank": "Hybrid + Cross-Encoder Rerank (Recommended)",
                "hybrid": "Hybrid Search (RRF)",
                "dense": "Dense Semantic (FAISS)",
                "bm25": "BM25 (Keyword Lexical)"
            }[x]
        )
    has_gemini = bool(st.session_state.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY"))
    engine_options = ["gemini", "semantic_synthesizer", "seq2seq_transformer", "hf_transformer"] if has_gemini else ["semantic_synthesizer", "gemini", "seq2seq_transformer", "hf_transformer"]

    with qa_col3:
        engine_choice = st.selectbox(
            "QA Engine:",
            engine_options,
            format_func=lambda x: {
                "gemini": "✨ Google Gemini (gemini-3.8-flash - Recommended)",
                "semantic_synthesizer": "🧠 Cross-Encoder Semantic Comprehension (Local)",
                "seq2seq_transformer": "🤖 Local Generative Transformer (DistilBART)",
                "hf_transformer": "🔍 Extractive Span Reader (DistilBERT SQuAD)"
            }[x]
        )

    verify_claims_toggle = st.checkbox("Run NLI Evidence Verification (DeBERTa)", value=True)

    if st.button("💡 Generate Answer & Verify", type="primary"):
        rag = st.session_state.rag
        with st.spinner("Retrieving evidence, synthesizing answer with citations, and verifying claims..."):
            result = rag.answer_question(
                question=question_input,
                retrieval_method=method_choice,
                engine=engine_choice,
                top_k=3,
                verify=verify_claims_toggle,
                gemini_api_key=st.session_state.get("gemini_api_key")
            )

        st.markdown("---")
        st.markdown("### 📝 Synthesized Academic Answer")
        
        # Display engine badge & confidence if available
        meta = result.get("qa_metadata", {})
        model_name = meta.get("model", "Neural Model")
        conf = meta.get("confidence", 0.0)
        span = meta.get("extracted_span", "")
        
        c_badge1, c_badge2 = st.columns([1, 1])
        with c_badge1:
            st.markdown(f'<span class="hf-badge">🤖 Engine: {model_name}</span>', unsafe_allow_html=True)
        with c_badge2:
            if conf > 0:
                st.markdown(f'<span class="model-pill">🎯 Span Confidence: {conf:.1%}</span>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="ui-card" style="border-left: 4px solid #3b82f6; background: #ffffff; padding: 18px; margin-top: 10px;">
            <div style="font-size: 0.96rem; line-height: 1.7; color: #1e293b;">
                {result['answer']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if span:
            with st.expander("🔍 Extracted Evidence Span Details"):
                st.markdown(f"**Exact Span:** `{span}`")
                st.markdown(f"**Confidence:** `{conf:.2%}`")
                st.markdown(f"**Passage Index:** `[E{meta.get('best_passage_idx', 1)}]`")

        # Citations section
        st.markdown("#### 📚 Supporting Evidence Citations")
        if result["citations"]:
            for cit in result["citations"]:
                with st.expander(f"📌 Citation {cit['tag']}: {cit['document']} — Page {cit['page']}"):
                    st.markdown(f"**Chunk ID:** `{cit['chunk_id']}`")
                    st.write(cit["snippet"])
        else:
            st.info("No explicit passage tags in synthesized response.")


        # Verification section
        if result["verification"]:
            st.markdown("---")
            st.markdown("### 🛡️ NLI Hallucination & Faithfulness Verification")
            v_res = result["verification"]

            v_col1, v_col2, v_col3 = st.columns(3)
            with v_col1:
                status = v_res["overall_status"]
                if status == "supported":
                    st.markdown('<div class="status-supported">✅ FAITHFULLY SUPPORTED</div>', unsafe_allow_html=True)
                elif status == "contradicted":
                    st.markdown('<div class="status-contradicted">❌ CONTRADICTED BY EVIDENCE</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-uncertain">⚠️ UNCERTAIN / LOW EVIDENCE</div>', unsafe_allow_html=True)

            with v_col2:
                st.metric("Faithfulness Score", f"{v_res['faithfulness_score']}%")

            with v_col3:
                st.metric("Verified Claims", f"{v_res['supported_count']} / {v_res['total_claims']}")

            if v_res["flagged"]:
                st.warning(f"🚨 **Hallucination Alert:** {v_res['flag_reason']}")

            st.markdown("##### Claim-by-Claim Entailment Breakdown:")
            claims_data = []
            for c in v_res["claim_verifications"]:
                claims_data.append({
                    "Atomic Claim": c["claim"],
                    "Status": c["label"].upper(),
                    "Confidence": f"{c['confidence']:.2f}"
                })
            st.dataframe(pd.DataFrame(claims_data), use_container_width=True, hide_index=True)


# ==========================================================
# TAB 4: EVALUATION DASHBOARD
# ==========================================================
with tabs[3]:
    st.subheader("Quantitative Information Retrieval (IR) Evaluation")
    st.caption("Empirical benchmark comparing Hit@K, Recall@K, MRR@10, and Latency across the active paper corpus.")

    st.markdown("""
    This evaluation suite generates grounded test queries with known chunk targets to evaluate retrieval quality:
    - **Hit@K:** Did the retriever return at least one relevant passage in the top-K?
    - **Recall@K:** What fraction of target evidence passages were retrieved in the top-K?
    - **MRR@10:** Mean Reciprocal Rank (1/rank of the first relevant result).
    - **nDCG@K:** Normalized Discounted Cumulative Gain accounting for position.
    """)

    num_eval_queries = st.slider("Number of benchmark questions to evaluate:", min_value=4, max_value=20, value=8)

    if st.button("🧪 Run IR Evaluation Benchmark", type="primary"):
        with st.spinner("Generating test questions and running benchmark evaluation across all 4 strategies..."):
            bench_queries = generate_benchmark_queries_from_chunks(
                st.session_state.chunks,
                num_queries=num_eval_queries
            )

            eval_df = evaluate_retrieval(
                st.session_state.retriever,
                bench_queries,
                methods=["bm25", "dense", "hybrid", "hybrid_rerank"],
                k_eval=5
            )

            st.session_state.eval_results = eval_df

    if "eval_results" in st.session_state:
        df = st.session_state.eval_results
        st.markdown("### 🏆 Comparative Benchmark Results")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Matplotlib visualization
        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            st.markdown("#### Accuracy Metrics (Hit@5 & Recall@5)")
            st.bar_chart(df.set_index("Method")[["Hit@5", "Recall@5"]])

        with c_chart2:
            st.markdown("#### Ranking Quality (MRR@10)")
            st.bar_chart(df.set_index("Method")[["MRR@10"]])

        st.markdown("#### Latency Profiling (ms)")
        st.line_chart(df.set_index("Method")[["Avg Latency (ms)"]])

        # Download CSV
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Benchmark Results CSV for Report",
            data=csv_data,
            file_name="researchmate_evaluation_results.csv",
            mime="text/csv"
        )
