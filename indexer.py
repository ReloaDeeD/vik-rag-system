import hashlib
import sqlite3
import sys
import re
import argparse
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from document_parser import parse_document, ParsedBlock

DB_PATH = Path("data/metadata.db")
CHROMA_PATH = Path("data/chroma")
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
ALLOWED = {".txt", ".md", ".pdf", ".docx"} 

def chunk_blocks(blocks: list[ParsedBlock], min_size: int = 150, max_size: int = 400) -> list[tuple[str, str]]:
    # ... (оставляем твой код чанкинга без изменений) ...
    chunks = []
    current_text = ""
    current_locs = []

    def flush_chunk():
        nonlocal current_text, current_locs
        if not current_text.strip():
            return
        if len(current_locs) == 1:
            loc_str = current_locs[0]
        else:
            unique_locs = list(dict.fromkeys(current_locs))
            loc_str = f"{unique_locs[0]} – {unique_locs[-1]}" if len(unique_locs) > 1 else unique_locs[0]

        if len(current_text) > max_size:
            sentences = re.split(r'(?<=[.!?])\s+', current_text)
            temp_text = ""
            for sent in sentences:
                if len(temp_text) + len(sent) > max_size and temp_text:
                    chunks.append((temp_text.strip(), loc_str))
                    temp_text = sent + " "
                else:
                    temp_text += sent + " "
            if temp_text.strip():
                chunks.append((temp_text.strip(), loc_str))
        else:
            chunks.append((current_text.strip(), loc_str))
        current_text = ""
        current_locs = []

    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        if len(current_text) + len(text) > max_size and len(current_text) >= min_size:
            flush_chunk()
        current_text += text + " "
        current_locs.append(block.locator)

    flush_chunk()
    return chunks

def connect():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    sql = sqlite3.connect(DB_PATH)
    
    # Читаем схему из папки sql
    schema_path = Path("sql/schema.sql")
    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            sql.executescript(f.read())
        sql.commit()
    else:
        print(f"ВНИМАНИЕ: Файл {schema_path} не найден!")
    
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    coll = client.get_or_create_collection("corp_docs", embedding_function=ef)
    return sql, coll

def index_document(path: Path, sql, coll, force: bool = False) -> str:
    file_bytes = path.read_bytes()
    finger = hashlib.sha256(file_bytes).hexdigest()
    cur = sql.cursor()

    if not force:
        row = cur.execute("SELECT id FROM documents WHERE content_hash = ?", (finger,)).fetchone()
        if row:
            return f"{path.name}: без изменений (id {row[0]}), пропускаю"

    old = cur.execute("SELECT id FROM documents WHERE source_path = ?", (str(path),)).fetchone()
    if old:
        old_id = old[0]
        old_chunks = cur.execute("SELECT id FROM chunks WHERE document_id = ?", (old_id,)).fetchall()
        if old_chunks:
            try:
                coll.delete(ids=[f"chunk_{r[0]}" for r in old_chunks])
            except Exception:
                pass
        cur.execute("DELETE FROM chunks WHERE document_id = ?", (old_id,))
        cur.execute("DELETE FROM documents WHERE id = ?", (old_id,))

    try:
        blocks = parse_document(path)
    except Exception as e:
        return f"{path.name}: ОШИБКА - {e}"

    if not blocks:
        return f"{path.name}: пропущен (нет текста)"

    cur.execute(
        "INSERT INTO documents (title, source_path, format, content_hash) VALUES (?, ?, ?, ?)",
        (path.name, str(path), path.suffix.lstrip("."), finger),
    )
    doc_id = cur.lastrowid

    sql_ids, texts, metadatas = [], [], []
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
            "locator": locator  
        })

    coll.upsert(
        ids=[f"chunk_{i}" for i in sql_ids],
        documents=texts,
        metadatas=metadatas,
    )
    sql.commit()
    return f"{path.name}: проиндексирован, чанков: {len(sql_ids)}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="docs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    target = Path(args.path)
    sql, coll = connect()
    
    if target.is_file():
        files = [target] if target.suffix.lower() in ALLOWED else []
    else:
        files = sorted(p for p in target.iterdir() if p.suffix.lower() in ALLOWED)
        
    for f in files:
        print(index_document(f, sql, coll, force=args.force))
        
    n_docs = sql.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    n_chunks = sql.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"---\nИтого: {n_docs} док., {n_chunks} чанков в SQL; {coll.count()} векторов в ChromaDB")
    sql.close()