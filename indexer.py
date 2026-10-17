import hashlib
import sqlite3
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# Импортируем наш новый парсер
from document_parser import parse_document, ParsedBlock

DB_PATH = Path("data/metadata.db")
CHROMA_PATH = Path("data/chroma")
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
ALLOWED = {".txt", ".md", ".pdf", ".docx"} # <-- Добавили форматы

def chunk_blocks(blocks: list[ParsedBlock], max_size: int = 600) -> list[tuple[str, str]]:
    """Режет блоки на чанки. Возвращает список кортежей (текст_чанка, локатор)."""
    chunks = []
    for block in blocks:
        para = block.text
        if len(para) <= max_size:
            chunks.append((para, block.locator))
        else:
            step = max_size - 100
            for i in range(0, len(para), step):
                chunk_part = para[i:i + max_size]
                chunks.append((chunk_part, block.locator))
    return chunks

def connect():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    sql = sqlite3.connect(DB_PATH)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    coll = client.get_or_create_collection("corp_docs", embedding_function=ef)
    return sql, coll

def index_document(path: Path, sql, coll) -> str:
    # 1. Считаем хэш по БАЙТАМ файла, а не по тексту (надежнее для PDF)
    file_bytes = path.read_bytes()
    finger = hashlib.sha256(file_bytes).hexdigest()
    cur = sql.cursor()

    # 2. Проверка идемпотентности
    row = cur.execute("SELECT id FROM documents WHERE content_hash = ?", (finger,)).fetchone()
    if row:
        return f"{path.name}: без изменений (id {row[0]}), пропускаю"

    # 3. Удаление старой версии
    old = cur.execute("SELECT id FROM documents WHERE source_path = ?", (str(path),)).fetchone()
    if old:
        old_id = old[0]
        old_chunks = cur.execute("SELECT id FROM chunks WHERE document_id = ?", (old_id,)).fetchall()
        if old_chunks:
            coll.delete(ids=[f"chunk_{r[0]}" for r in old_chunks])
        cur.execute("DELETE FROM chunks WHERE document_id = ?", (old_id,))
        cur.execute("DELETE FROM documents WHERE id = ?", (old_id,))

    # 4. Парсинг документа
    try:
        blocks = parse_document(path)
    except Exception as e:
        return f"{path.name}: ОШИБКА - {e}"

    if not blocks:
        return f"{path.name}: пропущен (нет текста)"

    # 5. Вставка документа в SQL
    cur.execute(
        "INSERT INTO documents (title, source_path, format, content_hash) VALUES (?, ?, ?, ?)",
        (path.name, str(path), path.suffix.lstrip("."), finger),
    )
    doc_id = cur.lastrowid

    # 6. Чанкинг и запись в SQL + Chroma
    sql_ids, texts, metadatas = [], [], []
    
    # Получаем список (текст, локатор)
    chunks_with_locs = chunk_blocks(blocks)
    
    for i, (chunk_text, locator) in enumerate(chunks_with_locs):
        cur.execute(
            "INSERT INTO chunks (document_id, chunk_index, content, char_count, locator) VALUES (?, ?, ?, ?, ?)",
            (doc_id, i, chunk_text, len(chunk_text), locator),
        )
        chunk_id = cur.lastrowid
        sql_ids.append(chunk_id)
        texts.append(chunk_text)
        metadatas.append({
            "chunk_id": chunk_id, 
            "document_id": doc_id, 
            "chunk_index": i, 
            "title": path.name,
            "locator": locator  # <-- Передаем в ChromaDB для поиска
        })

    coll.upsert(
        ids=[f"chunk_{i}" for i in sql_ids],
        documents=texts,
        metadatas=metadatas,
    )
    sql.commit()
    return f"{path.name}: проиндексирован, чанков: {len(sql_ids)} (id документа {doc_id})"

if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs")
    sql, coll = connect()
    
    if target.is_file():
        files = [target] if target.suffix.lower() in ALLOWED else []
    else:
        files = sorted(p for p in target.iterdir() if p.suffix.lower() in ALLOWED)
        
    for f in files:
        print(index_document(f, sql, coll))
        
    n_docs = sql.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    n_chunks = sql.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"---\nИтого: {n_docs} док., {n_chunks} чанков в SQL; {coll.count()} векторов в ChromaDB")
    sql.close()