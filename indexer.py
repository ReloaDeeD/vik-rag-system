"""
Единый индексатор.

Прогоняет документ по всему конвейеру:
файл -> текст -> чанки -> SQL (метаданные) + ChromaDB (векторы) под одинаковыми id.
"""
import hashlib
import sqlite3
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

DB_PATH = Path("data/metadata.db")
CHROMA_PATH = Path("data/chroma")
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
ALLOWED = {".txt", ".md"}


def chunk_text(text: str, max_size: int = 600) -> list[str]:
    """Режет документ на смысловые куски: по абзацам (пустым строкам).
    Слишком длинный абзац дочерпает символами с перекрытием."""
    chunks = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_size:
            chunks.append(para)
        else:
            step = max_size - 100
            chunks.extend(para[i:i + max_size] for i in range(0, len(para), step))
    return chunks


def connect():
    """Открывает оба хранилища: SQL и векторное."""
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    sql = sqlite3.connect(DB_PATH)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    coll = client.get_or_create_collection("corp_docs", embedding_function=ef)
    return sql, coll


def index_document(path: Path, sql, coll) -> str:
    """Индексирует один файл: отпечаток -> SQL -> ChromaDB."""
    text = path.read_text(encoding="utf-8")
    finger = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cur = sql.cursor()

    # 1. Файл не менялся с прошлой индексации — делать нечего
    row = cur.execute("SELECT id FROM documents WHERE content_hash = ?", (finger,)).fetchone()
    if row:
        return f"{path.name}: без изменений (id {row[0]}), пропускаю"

    # 2. Файл знакомый, но изменился — удаляем старую версию из ОБЕИХ баз
    old = cur.execute("SELECT id FROM documents WHERE source_path = ?", (str(path),)).fetchone()
    if old:
        old_id = old[0]
        old_chunks = cur.execute("SELECT id FROM chunks WHERE document_id = ?", (old_id,)).fetchall()
        coll.delete(ids=[f"chunk_{r[0]}" for r in old_chunks])
        cur.execute("DELETE FROM chunks WHERE document_id = ?", (old_id,))
        cur.execute("DELETE FROM documents WHERE id = ?", (old_id,))

    # 3. Вставляем документ в SQL
    cur.execute(
        "INSERT INTO documents (title, source_path, format, content_hash) VALUES (?, ?, ?, ?)",
        (path.name, str(path), path.suffix.lstrip("."), finger),
    )
    doc_id = cur.lastrowid

    # 4. Чанки: сначала в SQL (там они получают id), затем в ChromaDB под теми же id
    sql_ids, texts = [], []
    for i, chunk in enumerate(chunk_text(text)):
        cur.execute(
            "INSERT INTO chunks (document_id, chunk_index, content, char_count) VALUES (?, ?, ?, ?)",
            (doc_id, i, chunk, len(chunk)),
        )
        sql_ids.append(cur.lastrowid)
        texts.append(chunk)

    coll.upsert(
        ids=[f"chunk_{i}" for i in sql_ids],
        documents=texts,
        metadatas=[{"chunk_id": i, "document_id": doc_id, "chunk_index": n, "title": path.name}
                   for n, i in enumerate(sql_ids)],
    )
    sql.commit()
    return f"{path.name}: проиндексирован, чанков: {len(sql_ids)} (id документа {doc_id})"


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs")
    sql, coll = connect()
    files = [target] if target.is_file() else sorted(p for p in target.iterdir() if p.suffix.lower() in ALLOWED)
    for f in files:
        print(index_document(f, sql, coll))
    n_docs = sql.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    n_chunks = sql.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"---\nИтого: {n_docs} док., {n_chunks} чанков в SQL; {coll.count()} векторов в ChromaDB")
    sql.close()