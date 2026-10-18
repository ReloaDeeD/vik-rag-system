"""Шаг 6: генерация ответа по контексту (RAG целиком).
Вопрос -> гибридный поиск -> контекст -> локальная LLM (Ollama) -> ответ со ссылками.
"""
import json
import sqlite3
import sys
import urllib.request

from search_engine import hybrid_search

MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT = """Ты — справочный ассистент компании.
Отвечай ТОЛЬКО по контексту ниже. Правила:
1. Внимательно прочитай контекст. Если в нём есть информация для ответа — сформулируй ответ по контексту (2–4 предложения) и в конце перечисли номера источников в квадратных скобках, например: [1], [2].
2. Отказывайся ТОЛЬКО тогда, когда в контексте действительно нет информации, нужной для ответа. Фраза отказа: "В предоставленных документах нет ответа на этот вопрос." Без номеров источников.
3. Не используй собственные знания вне контекста и не выдумывай факты.

Контекст:
{context}

Вопрос: {question}
Ответ:"""


def ask_llm(prompt: str) -> str:
    """Обращение к локальному серверу Ollama по HTTP."""
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))["response"]


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "сколько дней длится отпуск?"
    print(f"Вопрос: {question}\n")

    hits = hybrid_search(question)
    if not hits:
        print("Поиск: оба контура молчат — релевантных фрагментов нет.")
        print("Ответ системы: в документах нет ответа на этот вопрос.")
        sys.exit(0)

    # <-- ИЗМЕНЕНИЕ: используем locator, если он есть, иначе chunk_index
    context = "\n".join(
        f"[{n}] ({h.title}, {h.locator or f'фрагмент {h.chunk_index}'}) {h.content}"
        for n, h in enumerate(hits, start=1)
    )
    print("Найдено гибридным поиском:")
    print(context, "\n")

    try:
        answer = ask_llm(PROMPT.format(context=context, question=question))
    except Exception as e:
        print(f"Ошибка генерации ответа: {e}")
        print("Возможно, Ollama не запущена. Запустите 'ollama serve' в отдельном терминале.")
        sys.exit(1)

    print("Ответ LLM:")
    print(answer)

    sql = sqlite3.connect("data/metadata.db")
    sql.execute(
        "INSERT INTO query_log (question, answer, source_chunk_ids, best_distance) "
        "VALUES (?, ?, ?, ?)",
        (
            question,
            answer,
            json.dumps([h.chunk_id for h in hits], ensure_ascii=False),
            hits[0].vec_dist if hits[0].vec_dist >= 0 else None,
        ),
    )
    sql.commit()
    print("\nЗапись добавлена в query_log.")
    sql.close()