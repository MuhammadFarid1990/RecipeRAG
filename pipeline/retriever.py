"""
RecipeRAG — Hybrid Retriever
Combines dense (FAISS) and sparse (BM25) retrieval via Reciprocal Rank Fusion.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Tuple
import json
import os

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_DIR  = "pipeline/index"


class HybridRetriever:
    def __init__(self):
        self.model   = SentenceTransformer(MODEL_NAME)
        self.index   = None
        self.bm25    = None
        self.recipes = []

    def load(self):
        """Load pre-built FAISS index and BM25 corpus."""
        self.index = faiss.read_index(f"{INDEX_DIR}/recipes.faiss")
        with open(f"{INDEX_DIR}/recipes.json") as f:
            self.recipes = json.load(f)
        tokenized = [r["search_text"].lower().split() for r in self.recipes]
        self.bm25 = BM25Okapi(tokenized)
        return self

    def dense_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Dense search using sentence embeddings + FAISS."""
        q_emb = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(q_emb.astype(np.float32), top_k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]

    def sparse_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Sparse search using BM25."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[int, float]],
        sparse_results: List[Tuple[int, float]],
        k: int = 60,
        dense_weight: float = 0.7,
    ) -> List[Tuple[int, float]]:
        """Combine dense and sparse rankings using RRF."""
        rrf_scores = {}

        for rank, (idx, _) in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + dense_weight * (1 / (k + rank + 1))

        for rank, (idx, _) in enumerate(sparse_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (1 - dense_weight) * (1 / (k + rank + 1))

        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def search(self, query: str, top_k: int = 5, filters: dict = None) -> List[dict]:
        """Full hybrid search pipeline."""
        dense  = self.dense_search(query,  top_k=20)
        sparse = self.sparse_search(query, top_k=20)
        fused  = self.reciprocal_rank_fusion(dense, sparse)

        results = []
        for idx, score in fused:
            recipe = self.recipes[idx]

            # Apply metadata filters
            if filters:
                if "max_calories" in filters and recipe.get("calories", 999) > filters["max_calories"]:
                    continue
                if "max_time" in filters and recipe.get("total_time_mins", 999) > filters["max_time"]:
                    continue
                if "dietary" in filters and filters["dietary"] not in recipe.get("tags", []):
                    continue

            results.append({**recipe, "rrf_score": round(score, 4)})
            if len(results) >= top_k:
                break

        return results
