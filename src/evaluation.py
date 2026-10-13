"""
Information Retrieval (IR) Evaluation Module for ResearchMate
Evaluates retrieval strategies (BM25, Dense, Hybrid, Rerank) using standard metrics:
Hit@K, Recall@K, MRR@10, nDCG@K, and Latency profiling.
"""

import time
import math
from typing import List, Dict, Any
import pandas as pd
import numpy as np


def hit_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
    """Returns 1.0 if at least one relevant item is in the top-k retrieved items, else 0.0."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return 1.0 if any(cid in relevant for cid in retrieved_ids[:k]) else 0.0


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
    """Fraction of relevant items successfully retrieved in top-k."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    retrieved = set(retrieved_ids[:k])
    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str], max_k: int = 10) -> float:
    """Computes Reciprocal Rank (1/rank of first relevant item within max_k)."""
    relevant = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids[:max_k], start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
    """Computes Normalized Discounted Cumulative Gain at k (binary relevance)."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    dcg = 0.0
    for i, cid in enumerate(retrieved_ids[:k]):
        if cid in relevant:
            dcg += 1.0 / math.log2(i + 2)

    # Ideal DCG
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval(
    retriever,
    benchmark_queries: List[Dict[str, Any]],
    methods: List[str] = None,
    k_eval: int = 5
) -> pd.DataFrame:
    """
    Evaluates retriever across multiple methods on a benchmark question set.
    Each benchmark item has:
      {"question": str, "relevant_chunk_ids": list[str]}
    """
    if methods is None:
        methods = ["bm25", "dense", "hybrid", "hybrid_rerank"]

    rows = []

    for method in methods:
        hit_scores = []
        recall_scores = []
        rr_scores = []
        ndcg_scores = []
        latencies = []

        for item in benchmark_queries:
            query = item["question"]
            relevant_ids = item["relevant_chunk_ids"]

            start_t = time.perf_counter()
            if method == "bm25":
                results = retriever.search_bm25(query, top_k=k_eval)
            elif method == "dense":
                results = retriever.search_dense(query, top_k=k_eval)
            elif method == "hybrid":
                results = retriever.search_hybrid(query, top_k=k_eval, candidate_k=max(20, k_eval * 2))
            elif method == "hybrid_rerank":
                candidates = retriever.search_hybrid(query, top_k=max(15, k_eval * 3), candidate_k=30)
                results = retriever.rerank(query, candidates, top_k=k_eval)
            else:
                raise ValueError(f"Unknown retrieval method: {method}")
            elapsed_ms = (time.perf_counter() - start_t) * 1000

            retrieved_ids = [r["chunk_id"] for r in results]

            hit_scores.append(hit_at_k(retrieved_ids, relevant_ids, k=k_eval))
            recall_scores.append(recall_at_k(retrieved_ids, relevant_ids, k=k_eval))
            rr_scores.append(reciprocal_rank(retrieved_ids, relevant_ids, max_k=10))
            ndcg_scores.append(ndcg_at_k(retrieved_ids, relevant_ids, k=k_eval))
            latencies.append(elapsed_ms)

        rows.append({
            "Method": method.replace("_", " + ").upper(),
            f"Hit@{k_eval}": round(float(np.mean(hit_scores)) * 100, 2),
            f"Recall@{k_eval}": round(float(np.mean(recall_scores)) * 100, 2),
            "MRR@10": round(float(np.mean(rr_scores)), 4),
            f"nDCG@{k_eval}": round(float(np.mean(ndcg_scores)), 4),
            "Avg Latency (ms)": round(float(np.mean(latencies)), 2)
        })

    return pd.DataFrame(rows)
