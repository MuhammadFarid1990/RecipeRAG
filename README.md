# RecipeRAG

> A hybrid RAG (Retrieval-Augmented Generation) pipeline for recipe retrieval and recommendation — combining dense vector search with sparse BM25 retrieval.

---

## What it does

RecipeRAG takes a natural-language query ("something quick with chicken and lemon that's under 500 calories") and returns the most relevant recipes from the corpus, ranked by a hybrid dense+sparse score. Claude then generates a personalized recommendation with substitutions, timing tips, and why it fits the query.

---

## Why hybrid RAG?

Dense-only retrieval (embeddings) is great for semantic similarity but misses exact keyword matches. Sparse retrieval (BM25) is great for keywords but misses semantic meaning. Hybrid RAG gets the best of both.

```
Query: "quick chicken lemon under 500 cal"
         │
    ┌────┴────┐
    │         │
  Dense     Sparse
 (embeddings) (BM25)
    │         │
    └────┬────┘
         │  Reciprocal Rank Fusion
         ▼
   Top-K candidates
         │
   Claude re-ranks
   + generates response
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     FastAPI server                       │
└───────────────────────────┬─────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
   Dense retrieval    Sparse retrieval    Metadata filter
   (sentence-        (BM25 via           (calories, time,
    transformers)     rank-bm25)          dietary flags)
         │                  │                  │
         └──────────────────┼──────────────────┘
                            │ Reciprocal Rank Fusion
                            ▼
                    Top-K candidates
                            │
                      Claude re-rank
                    + generate response
```

---

## Tech stack

| Layer | Tech |
|-------|------|
| Dense retrieval | sentence-transformers (all-MiniLM-L6-v2) |
| Sparse retrieval | rank-bm25 |
| Vector store | FAISS |
| LLM | Claude (Anthropic SDK) |
| API | FastAPI |
| Data | ~50K recipes from public recipe dataset |

---

## Quickstart

```bash
git clone https://github.com/MuhammadFarid1990/RecipeRAG
cd RecipeRAG
pip install -r requirements.txt

# Index the recipe corpus (first run only, ~5 min)
python pipeline/index.py

# Start the API
uvicorn api.main:app --reload

# Query
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "quick chicken lemon under 500 calories", "top_k": 5}'
```

---

## Project structure

```
RecipeRAG/
├── pipeline/
│   ├── index.py          # Build FAISS index + BM25 corpus
│   ├── retriever.py      # Hybrid retrieval with RRF
│   └── reranker.py       # Claude-based re-ranking
├── api/
│   ├── main.py           # FastAPI endpoints
│   └── schemas.py        # Request/response models
├── data/
│   └── load.py           # Recipe dataset loader
├── requirements.txt
└── README.md
```

---

## About the builder

**Muhammad Farid** — MS Business Analytics & AI @ UT Dallas.

[Portfolio](https://muhammadfarid1990.github.io) · [GitHub](https://github.com/MuhammadFarid1990)

Built with [Claude](https://claude.ai).
