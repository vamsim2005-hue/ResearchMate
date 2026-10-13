"""
Multi-Stage Hybrid Retriever Module for ResearchMate
Implements:
  1. BM25 Lexical Retrieval (Rank-BM25)
  2. Dense Vector Retrieval (SentenceTransformers + FAISS)
  3. Reciprocal Rank Fusion (RRF Hybrid)
  4. Cross-Encoder Neural Reranking
"""

import re
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


class ResearchRetriever:
    def __init__(self, chunks: List[Dict[str, Any]], embedding_model_name: str = EMBED_MODEL_NAME):
        if not chunks:
            raise ValueError("At least one text chunk is required to initialize the retriever.")

        self.chunks = chunks
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Initialize BM25 Lexical Index
        tokenized_corpus = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. Initialize Dense Vector Index
        self.embedder = SentenceTransformer(embedding_model_name, device=self.device)
        corpus_texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(
            corpus_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        self.embeddings = np.asarray(embeddings, dtype="float32")

        self.dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(self.embeddings)

        # 3. Lazy-loaded Reranker
        self._reranker: Optional[CrossEncoder] = None

    def get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(RERANK_MODEL_NAME, device=self.device)
        return self._reranker

    def search_bm25(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search chunks using BM25Okapi scoring."""
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0 and len(top_indices) > top_k:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = round(score, 4)
            chunk["method"] = "BM25"
            results.append(chunk)

        return results

    def search_dense(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search chunks using Dense embeddings and FAISS cosine similarity."""
        query_vec = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        query_vec = np.asarray(query_vec, dtype="float32")

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = round(float(score), 4)
            chunk["method"] = "Dense (MiniLM)"
            results.append(chunk)

        return results

    def search_hybrid(self, query: str, top_k: int = 5, candidate_k: int = 20, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """
        Combines BM25 and Dense search using Reciprocal Rank Fusion (RRF).
        Formula: RRF_score(d) = sum(1 / (rrf_k + rank_m(d)))
        """
        bm25_results = self.search_bm25(query, top_k=candidate_k)
        dense_results = self.search_dense(query, top_k=candidate_k)

        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        # Accumulate BM25 ranks
        for rank, res in enumerate(bm25_results, start=1):
            cid = res["chunk_id"]
            chunk_map[cid] = res
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

        # Accumulate Dense ranks
        for rank, res in enumerate(dense_results, start=1):
            cid = res["chunk_id"]
            chunk_map[cid] = res
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        results = []
        for cid in sorted_cids[:top_k]:
            chunk = dict(chunk_map[cid])
            chunk["score"] = round(rrf_scores[cid], 5)
            chunk["method"] = "Hybrid (RRF)"
            results.append(chunk)

        return results

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate passages using a Cross-Encoder.
        """
        if not candidates:
            return []

        reranker = self.get_reranker()
        pairs = [[query, c["text"]] for c in candidates]
        scores = reranker.predict(pairs)

        ranked = []
        for chunk, score in zip(candidates, scores):
            c = dict(chunk)
            c["score"] = round(float(score), 4)
            c["method"] = "Hybrid + Reranking"
            ranked.append(c)

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    def search_all_methods(self, query: str, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Runs all four retrieval methods simultaneously for side-by-side comparison.
        """
        bm25_res = self.search_bm25(query, top_k=top_k)
        dense_res = self.search_dense(query, top_k=top_k)
        hybrid_res = self.search_hybrid(query, top_k=top_k, candidate_k=max(20, top_k * 2))

        # Candidate pool for reranker
        hybrid_candidates = self.search_hybrid(query, top_k=max(15, top_k * 2), candidate_k=30)
        reranked_res = self.rerank(query, hybrid_candidates, top_k=top_k)

        return {
            "BM25 (Lexical)": bm25_res,
            "Dense Semantic (FAISS)": dense_res,
            "Hybrid (RRF)": hybrid_res,
            "Hybrid + Reranking": reranked_res
        }
