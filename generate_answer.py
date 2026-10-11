"""Шаг 6: генерация ответа по контексту (RAG целиком).
Вопрос -> векторный поиск -> контекст -> локальная LLM (Ollama) -> ответ со ссылками.
"""
import json
import sqlite3
import sys
import urllib.request

import chromadb
from chromadb.utils import embedding_functions

THRESHOLD = 0.7
MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT = """Ты — справочный ассистент компании. Отвечай ТОЛЬКО по контексту ниже.
Правила:
1. Не выдумывай факты: если в контексте нет ответа, напиши ровно: "В предоставленных документах нет ответа на этот вопрос."
2. Отвечай кратко и по сути (2–4 предложения).
3. В конце ответа перечисли номера источников в квадратных скобках, например: [1], [2].

Контекст:
{context}

Вопрос: {question}
Ответ:"""


def search(question: str):
    """Векторный поиск с порогом — тот же, что в search_with_sources.py."""
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    client = chromadb.PersistentClient(path="data/chroma")
    coll = client.get_or_create_collection("corp_docs", embedding_function=ef)
    result = coll.query(query_texts=[question], n_results=3)
    hits = []
    for doc, dist, meta in zip(result["documents"][0], result["distances"][0], result["metadatas"][0]):
        if dist <= THRESHOLD:
            hits.append((doc, dist, meta))
    return hits


def ask_llm(prompt: str) -> str:
    """Обращение к локальному серверу Ollama по HTTP."""
    payload = json.dumps({"model": MODEL, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))["response"]


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "сколько дней длится отпуск?"
    print(f"Вопрос: {question}\n")

    hits = search(question)
    if not hits:
        print("Поиск: релевантных фрагментов нет (все расстояния выше порога 0.7).")
        print("Ответ системы: в документах нет ответа на этот вопрос.")
        sys.exit(0)

    context = "\n".join(
        f"[{n}] ({m['title']}, фрагмент {m['chunk_index']}, расстояние {d:.3f}) {t}"
        for n, (t, d, m) in enumerate(hits, start=1)
    )
    print("Найдено поиском:")
    print(context, "\n")

    answer = ask_llm(PROMPT.format(context=context, question=question))
    print("Ответ LLM:")
    print(answer)

    sql = sqlite3.connect("data/metadata.db")
    sql.execute(
        "INSERT INTO query_log (question, answer, source_chunk_ids, best_distance) VALUES (?, ?, ?, ?)",
        (question, answer, json.dumps([m["chunk_id"] for _, _, m in hits], ensure_ascii=False), hits[0][1]),
    )
    sql.commit()
    print("\nЗапись добавлена в query_log.")
    sql.close()