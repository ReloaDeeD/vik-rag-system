"""Гибридный поиск: векторный + BM25 + RRF + re-ranking.
Единая точка входа для generate_answer.py и app.py.
"""
import json
import sqlite3
import urllib.request
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
    for doc, dist, meta in zip(
        result["documents"][0], result["distances"][0], result["metadatas"][0]
    ):
        hits.append(
            Hit(
                chunk_id=meta["chunk_id"],
                content=doc,
                title=meta["title"],
                chunk_index=meta["chunk_index"],
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


def _rerank(question: str, hits: list[Hit], top_k: int) -> list[Hit]:
    if len(hits) <= 1:
        return hits
    listing = "\n".join(f"{i}. {h.content[:400]}" for i, h in enumerate(hits, 1))
    prompt = (
        "Ниже вопрос и пронумерованные фрагменты текста. "
        "Расставь номера фрагментов в порядке убывания полезности для ответа на вопрос. "
        "Ответь ТОЛЬКО номерами через запятую, без других слов.\n"
        f"Вопрос: {question}\nФрагменты:\n{listing}\nПорядок:"
    )
    payload = json.dumps({
        "model": RERANK_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = json.loads(resp.read().decode("utf-8"))["response"]
        order = [int(x) for x in raw.replace(".", ",").split(",") if x.strip().isdigit()]
        order = [n for n in order if 1 <= n <= len(hits)]
        seen = set()
        unique_order = []
        for n in order:
            if n not in seen:
                seen.add(n)
                unique_order.append(n)
        unique_order += [i for i in range(1, len(hits) + 1) if i not in seen]
        ranked = [hits[n - 1] for n in unique_order]
    except Exception:
        ranked = hits
    for new_rank, h in enumerate(ranked):
        h.score = 1.0 / (new_rank + 1)
        h.source = "rerank-llm"
    return ranked[:top_k]


def hybrid_search(question: str) -> list[Hit]:
    """Единая точка входа: векторный + BM25 → пороги → RRF → re-ranking."""
    _load_all()
    v = _vector_search(question, TOP_K_RECALL)
    b = _bm25_search(question, TOP_K_RECALL)
    v = [h for h in v if h.score >= 1.0 / (1.0 + 0.7)]
    b = [h for h in b if h.score > 0.0]
    if not v and not b:
        return []
    fused = _rrf_fusion(v, b, TOP_K_FUSION)
    return _rerank(question, fused, TOP_K_FINAL)