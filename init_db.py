import sqlite3
from pathlib import Path

DB_PATH = Path("data/metadata.db")
SCHEMA_PATH = Path("sql/schema.sql")

DB_PATH.parent.mkdir(parents=True, exist_ok=True)   # папка data/, если её нет

conn = sqlite3.connect(DB_PATH)                     
conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
conn.commit()

tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Созданы таблицы:", [t[0] for t in tables])
conn.close()