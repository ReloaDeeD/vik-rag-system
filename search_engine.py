"""Гибридный поиск: векторный + BM25 + RRF + re-ranking.
Единая точка входа для generate_answer.py и app.py.
"""
import json
import sqlite3
import urllib.request
from sentence_transformers import CrossEncoder
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

# --- Константы ---
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"
CHROMA_PATH = "data/chroma"
DB_PATH = "data/metadata.db"
TOP_K_RECALL = 20
TOP_K_FUSION = 10
TOP_K_FINAL = 5
RRF_K = 60


@dataclass
class Hit:
    chunk_id: int
    content: str
    title: str
    chunk_index: int
    score: float
    source: str = ""
    vec_dist: float = -1.0
    locator: str = ""         


# --- Ленивые глобалы ---
_bm25 = None
_bm25_ids = None
_chroma_coll = None
_sql = None


def _load_all():
    global _bm25, _bm25_ids, _chroma_coll, _sql
    if _sql is None:
        # Добавляем check_same_thread=False для совместимости со Streamlit
        _sql = sqlite3.connect(DB_PATH, check_same_thread=False)
    if _chroma_coll is None:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _chroma_coll = client.get_or_create_collection("corp_docs", embedding_function=ef)
    if _bm25 is None:
        rows = _sql.execute("SELECT id, content FROM chunks").fetchall()
        _bm25_ids = [r[0] for r in rows]
        tokenized = [r[1].lower().split() for r in rows]
        _bm25 = BM25Okapi(tokenized)


def _vector_search(question: str, top_k: int) -> list[Hit]:
    result = _chroma_coll.query(query_texts=[question], n_results=top_k)
    hits = []
    if not result["documents"] or not result["documents"][0]:
        return hits
        
    for doc, dist, meta in zip(
        result["documents"][0], result["distances"][0], result["metadatas"][0]
    ):
        if meta is None:  # <-- ЗАЩИТА: если метаданных нет, пропускаем
            continue
            
        hits.append(
            Hit(
                chunk_id=meta.get("chunk_id", 0),
                content=doc,
                title=meta.get("title", "Unknown"),
                chunk_index=meta.get("chunk_index", 0),
                score=1.0 / (1.0 + dist),
                source="vector",
                vec_dist=dist,
                locator=meta.get("locator", ""),
            )
        )
    return hits


def _bm25_search(question: str, top_k: int) -> list[Hit]:
    tokens = question.lower().split()
    scores = _bm25.get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    hits = []
    for idx in ranked:
        chunk_id = _bm25_ids[idx]
        row = _sql.execute(
            "SELECT c.content, c.chunk_index, d.title, c.locator "
            "FROM chunks c "
            "JOIN documents d ON c.document_id = d.id "
            "WHERE c.id = ?",
            (chunk_id,),
        ).fetchone()
        if row:
            hits.append(
                Hit(
                    chunk_id=chunk_id,
                    content=row[0],
                    title=row[2],
                    chunk_index=row[1],
                    score=float(scores[idx]),
                    source="bm25",
                    locator=row[3] or "",       
                )
            )
    return hits


def _rrf_fusion(vector_hits: list[Hit], bm25_hits: list[Hit], top_k: int) -> list[Hit]:
    agg: dict[int, Hit] = {}
    for rank, h in enumerate(vector_hits):
        agg[h.chunk_id] = Hit(
            chunk_id=h.chunk_id,
            content=h.content,
            title=h.title,
            chunk_index=h.chunk_index,
            score=1.0 / (RRF_K + rank + 1),
            source="vector",
            vec_dist=h.vec_dist,
            locator=h.locator,                  
        )
    for rank, h in enumerate(bm25_hits):
        if h.chunk_id in agg:
            agg[h.chunk_id].score += 1.0 / (RRF_K + rank + 1)
            agg[h.chunk_id].source = "hybrid"
            if not agg[h.chunk_id].locator:
                agg[h.chunk_id].locator = h.locator   
        else:
            agg[h.chunk_id] = Hit(
                chunk_id=h.chunk_id,
                content=h.content,
                title=h.title,
                chunk_index=h.chunk_index,
                score=1.0 / (RRF_K + rank + 1),
                source="bm25",
                locator=h.locator,              
            )
    return sorted(agg.values(), key=lambda x: x.score, reverse=True)[:top_k]


_cross_encoder = None

def _rerank(question: str, hits: list[Hit], top_k: int) -> list[Hit]:
    """Переранжирование с помощью Cross-Encoder (быстро и точно)."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("BAAI/bge-reranker-base")
        
    if len(hits) <= 1:
        return hits
        
    pairs = [(question, h.content) for h in hits]
    
    scores = _cross_encoder.predict(pairs)
    
    for h, score in zip(hits, scores):
        h.score = float(score)
        
    ranked = sorted(hits, key=lambda x: x.score, reverse=True)
    
    for h in ranked:
        h.source = "rerank-cross-encoder"
        
    return ranked[:top_k]


def search(question: str, mode: str = "hybrid_rerank") -> list[Hit]:
    """
    Универсальная точка входа для поиска.
    Режимы:
    - "vector": только векторный поиск
    - "bm25": только BM25
    - "hybrid": векторный + BM25 + RRF
    - "hybrid_rerank": векторный + BM25 + RRF + переранжирование LLM (по умолчанию)
    """
    _load_all()
    
    if mode == "vector":
        return _vector_search(question, TOP_K_FINAL)
        
    elif mode == "bm25":
        return _bm25_search(question, TOP_K_FINAL)
        
    elif mode == "hybrid":
        v = _vector_search(question, TOP_K_RECALL)
        b = _bm25_search(question, TOP_K_RECALL)
        v = [h for h in v if h.score >= 1.0 / (1.0 + 0.7)]
        b = [h for h in b if h.score > 0.0]
        if not v and not b:
            return []
        return _rrf_fusion(v, b, TOP_K_FINAL)
        
    else:  # hybrid_rerank
        v = _vector_search(question, TOP_K_RECALL)
        b = _bm25_search(question, TOP_K_RECALL)
        v = [h for h in v if h.score >= 1.0 / (1.0 + 0.7)]
        b = [h for h in b if h.score > 0.0]
        if not v and not b:
            return []
        fused = _rrf_fusion(v, b, TOP_K_FUSION)
        return _rerank(question, fused, TOP_K_FINAL)

# Оставляем для обратной совместимости со старым кодом (app.py, generate_answer.py)
def hybrid_search(question: str) -> list[Hit]:
    return search(question, mode="hybrid_rerank")