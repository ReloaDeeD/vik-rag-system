import sqlite3
conn = sqlite3.connect("data/metadata.db")
try:
    conn.execute("ALTER TABLE query_log ADD COLUMN feedback INTEGER DEFAULT 0")
    conn.commit()
    print("Колонка feedback успешно добавлена!")
except sqlite3.OperationalError:
    print("Колонка уже существует.")
conn.close()
