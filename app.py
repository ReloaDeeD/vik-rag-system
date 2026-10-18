"""Шаг 7: веб-интерфейс RAG-ассистента на Streamlit.
Запуск: streamlit run app.py
"""
import json
import sqlite3
import urllib.request
from pathlib import Path

import streamlit as st

from indexer import connect, index_document   # <-- НОВОЕ: импорт индексатора
from search_engine import hybrid_search

MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"
UPLOAD_DIR = Path("docs")
UPLOAD_DIR.mkdir(exist_ok=True)

PROMPT = """Ты — справочный ассистент компании.
Отвечай ТОЛЬКО по контексту ниже. Правила:
1. Внимательно прочитай контекст. Если в нём есть информация для ответа — сформулируй ОТВЕТ СЛОВАМИ (2–4 предложения), основанный на контексте.
2. ЗАПРЕЩЕНО отвечать только номером источника или только "[1]". Ответ должен содержать текст.
3. В конце ответа перечисли номера источников в квадратных скобках, например: [1], [2].
4. Отказывайся ТОЛЬКО тогда, когда в контексте действительно нет информации. Фраза отказа: "В предоставленных документах нет ответа на этот вопрос." Без номеров источников.
5. Не используй собственные знания вне контекста и не выдумывай факты.

Контекст:
{context}

Вопрос: {question}
Ответ:"""


def ask_llm(prompt: str) -> str:
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


def log_query(question: str, answer: str, hits: list, best_dist: float | None):
    sql = sqlite3.connect("data/metadata.db")
    sql.execute(
        "INSERT INTO query_log (question, answer, source_chunk_ids, best_distance) "
        "VALUES (?, ?, ?, ?)",
        (
            question,
            answer,
            json.dumps([h.chunk_id for h in hits], ensure_ascii=False),
            best_dist,
        ),
    )
    sql.commit()
    sql.close()


# --- Интерфейс ---
st.set_page_config(page_title="RAG-ассистент", layout="centered")
st.title("RAG-ассистент по корпоративной документации")
st.caption("Локальный гибридный поиск с генерацией ответа. Данные не покидают контур предприятия.")

# --- Блок загрузки документов ---
with st.expander("📂 Загрузить документы (PDF, DOCX, TXT, MD)", expanded=False):
    uploaded = st.file_uploader(
        "Выберите файлы",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded:
        sql, coll = connect()
        for file in uploaded:
            save_path = UPLOAD_DIR / file.name
            save_path.write_bytes(file.getbuffer())
            try:
                msg = index_document(save_path, sql, coll)
                st.success(msg)
            except Exception as e:
                st.error(f"{file.name}: ошибка индексации — {e}")
        sql.close()

# --- Блок вопросов ---
with st.form("qa_form"):
    question = st.text_input(
        "Ваш вопрос:", placeholder="например, сколько дней длится отпуск?"
    )
    submitted = st.form_submit_button("Найти ответ")

if submitted and question.strip():
    with st.spinner("Ищем и формулируем ответ…"):
        hits = hybrid_search(question)
        answer = None
        if hits:
            # <-- ИЗМЕНЕНИЕ: используем locator в контексте
            context = "\n".join(
                f"[{n}] ({h.title}, {h.locator or f'фрагмент {h.chunk_index}'}) {h.content}"
                for n, h in enumerate(hits, start=1)
            )
            answer = ask_llm(PROMPT.format(context=context, question=question))
            best_dist = hits[0].vec_dist if hits[0].vec_dist >= 0 else None
            log_query(question, answer, hits, best_dist)
        st.session_state["last"] = {"question": question, "hits": hits, "answer": answer}

if "last" in st.session_state:
    res = st.session_state["last"]
    hits, answer = res["hits"], res["answer"]
    st.subheader(f"Вопрос: {res['question']}")

    if not hits:
        st.warning("В документах нет релевантных фрагментов на этот вопрос.")
    elif answer and "нет ответа на этот вопрос" in answer.lower():
        st.warning(answer)
    else:
        st.success(answer)

    if hits:
        with st.expander(f"Найдено фрагментов: {len(hits)}"):
            for n, h in enumerate(hits, start=1):
                dist_text = f"расстояние {h.vec_dist:.3f}" if h.vec_dist >= 0 else "BM25"
                # <-- ИЗМЕНЕНИЕ: показываем locator в UI
                loc = h.locator or f"фрагмент {h.chunk_index}"
                st.markdown(
                    f"**[{n}]** `{h.title}`, {loc} — {dist_text}, src={h.source}"
                )
                st.info(h.content)

with st.expander("Журнал последних запросов"):
    sql = sqlite3.connect("data/metadata.db")
    rows = sql.execute(
        "SELECT asked_at, question, best_distance FROM query_log "
        "ORDER BY asked_at DESC LIMIT 10"
    ).fetchall()
    sql.close()
    for asked, q, d in rows:
        d_text = f"{d:.3f}" if d is not None else "—"
        st.write(f"`{asked}` | {d_text} | {q}")