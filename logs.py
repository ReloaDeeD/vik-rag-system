import sqlite3
sql = sqlite3.connect("data/metadata.db")
rows = sql.execute(
    "SELECT rowid, asked_at, question, answer, feedback "
    "FROM query_log ORDER BY rowid DESC LIMIT 10"
).fetchall()
sql.close()

for rid, asked, q, a, fb in rows:
    print(f"--- #{rid} | {asked} | оценка: {fb}")
    print(f"Вопрос: {q}")
    print(f"Ответ:  {a}")
    print()