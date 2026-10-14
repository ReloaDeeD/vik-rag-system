"""Поиск с указанием источника: вопрос -> фрагмент + где он найден."""
import sqlite3
import sys

import chromadb
from chromadb.utils import embedding_functions

THRESHOLD = 0.7

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
sql = sqlite3.connect("data/metadata.db")
client = chromadb.PersistentClient(path="data/chroma")
coll = client.get_or_create_collection("corp_docs", embedding_function=ef)

question = sys.argv[1] if len(sys.argv) > 1 else "что будет за плохой пароль?"
result = coll.query(query_texts=[question], n_results=3)

found = False
for doc, dist, meta in zip(result["documents"][0], result["distances"][0], result["metadatas"][0]):
    if dist > THRESHOLD:
        continue
    found = True
    row = sql.execute("SELECT title FROM documents WHERE id = ?", (meta["document_id"],)).fetchone()
    print(f"{dist:.3f} | источник: {row[0]}, фрагмент {meta['chunk_index']}")
    print(f"       {doc}\n")

if not found:
    print("В документах нет ответа на этот вопрос.")