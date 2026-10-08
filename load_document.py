import hashlib
import sqlite3
from pathlib import Path

conn = sqlite3.connect("data/metadata.db")
cur = conn.cursor()

path = Path("docs/reglament.txt")
text = path.read_text(encoding="utf-8")
finger = hashlib.sha256(text.encode("utf-8")).hexdigest()   # "отпечаток пальца" файла

# если файл с таким отпечатком уже загружали — не дублируем
row = cur.execute("SELECT id FROM documents WHERE content_hash = ?", (finger,)).fetchone()
if row:
    print("Этот документ уже загружен, id =", row[0])
else:
    cur.execute(
        "INSERT INTO documents (title, source_path, format, content_hash) VALUES (?, ?, ?, ?)",
        (path.name, str(path), path.suffix.lstrip("."), finger),
    )
    doc_id = cur.lastrowid          # id только что вставленной строки

    chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
    for i, chunk in enumerate(chunks):
        cur.execute(
            "INSERT INTO chunks (document_id, chunk_index, content, char_count) VALUES (?, ?, ?, ?)",
            (doc_id, i, chunk, len(chunk)),
        )
    conn.commit()
    print(f"Документ загружен, id = {doc_id}, чанков: {len(chunks)}")

# проверка обычными SELECT
for doc_id, title, fmt in cur.execute("SELECT id, title, format FROM documents"):
    print("Документ:", doc_id, title, fmt)
for idx, cnt in cur.execute("SELECT chunk_index, char_count FROM chunks ORDER BY chunk_index"):
    print(f"  чанк {idx}: {cnt} символов")
conn.close()