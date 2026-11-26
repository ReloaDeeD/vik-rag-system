"""Веб-интерфейс RAG-ассистента (Streamlit).
Запуск: streamlit run app.py
"""
import json
import sqlite3
import urllib.request
from pathlib import Path

import streamlit as st

from indexer import connect, index_document
from search_engine import hybrid_search

from prompts import RAG_PROMPT, REWRITE_PROMPT

# --- Константы ---
MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"
DB_PATH = "data/metadata.db"
UPLOAD_DIR = Path("docs")
UPLOAD_DIR.mkdir(exist_ok=True)



# --- Вспомогательные функции ---
def ask_llm(prompt: str, temperature: float = 0.2) -> str:
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))["response"]


def log_query(question: str, answer: str, hits: list, best_dist: float | None) -> int:
    """Пишет запрос в журнал и возвращает id записи (для голосования)."""
    sql = sqlite3.connect(DB_PATH)
    cur = sql.execute(
        "INSERT INTO query_log (question, answer, source_chunk_ids, best_distance) "
        "VALUES (?, ?, ?, ?)",
        (question, answer, json.dumps([h.chunk_id for h in hits], ensure_ascii=False), best_dist),
    )
    sql.commit()
    qid = cur.lastrowid
    sql.close()
    return qid


def save_feedback(query_id: int, value: int):
    """Сохраняет лайк (1) или дизлайк (-1) в журнал."""
    sql = sqlite3.connect(DB_PATH)
    sql.execute("UPDATE query_log SET feedback = ? WHERE rowid = ?", (value, query_id))
    sql.commit()
    sql.close()


def get_db_stats():
    try:
        sql = sqlite3.connect(DB_PATH)
        docs_count = sql.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks_count = sql.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        sql.close()
        return docs_count, chunks_count
    except Exception:
        return 0, 0


def render_sources(hits: list):
    """Единая отрисовка источников: и в живом ответе, и в истории."""
    if not hits:
        return
    with st.expander("Источники"):
        for n, h in enumerate(hits, 1):
            st.markdown(
                f"**[{n}]** `{h.title}`, {h.locator or f'фрагмент {h.chunk_index}'}\n\n{h.content}"
            )


def handle_rag_query(prompt: str, history_messages: list):
    META = ("какие документы", "список документов", "что есть в базе", "какие файлы")
    if any(p in prompt.lower() for p in META):
        sql = sqlite3.connect(DB_PATH)
        rows = sql.execute("SELECT title, format FROM documents ORDER BY title").fetchall()
        sql.close()
        listing = "\n".join(f"- {t} ({f})" for t, f in rows)
        return f"В базе знаний сейчас {len(rows)} документ(ов):\n{listing}", [], prompt, None
    """Единая точка обработки запроса. Возвращает (answer, hits, rewritten, query_id)."""
    history_str = "\n".join(f"{m['role']}: {m['content']}" for m in history_messages[-5:-1])
    try:
        rewritten_query = ask_llm(REWRITE_PROMPT.format(history=history_str, question=prompt))
    except Exception:
        rewritten_query = prompt  # LLM недоступна — ищем исходным вопросом

    final_hits = hybrid_search(rewritten_query)[:5]
    if not final_hits:
        return ("В предоставленных документах нет релевантной информации для ответа на этот вопрос.",
                [], rewritten_query, None)

    context = "\n".join(
        f"[{n}] ({h.title}, {h.locator or f'фрагмент {h.chunk_index}'}) {h.content}"
        for n, h in enumerate(final_hits, start=1)
    )
    chat_history = "\n".join(f"Пользователь: {m['content']}" for m in history_messages[-3:] if m['role'] == 'user')
    try:
        answer = ask_llm(RAG_PROMPT.format(history=chat_history, context=context, question=prompt))
    except Exception as e:
        return (f"Локальная LLM недоступна: {e}. Запустите Ollama и повторите вопрос.",
                final_hits, rewritten_query, None)

    best_dist = final_hits[0].vec_dist if final_hits[0].vec_dist >= 0 else None
    qid = log_query(prompt, answer, final_hits, best_dist)
    return answer, final_hits, rewritten_query, qid


# --- Инициализация состояния ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "voted" not in st.session_state:
    st.session_state.voted = set()

# --- Интерфейс ---
st.set_page_config(page_title="CorpRAG AI", layout="wide")

with st.sidebar:
    st.title("Настройки")
    with st.expander("Загрузить документы", expanded=False):
        uploaded = st.file_uploader("Выберите файлы", type=["pdf", "docx", "txt", "md"],
                                    accept_multiple_files=True)
        if uploaded:
            sql, coll = connect()
            for file in uploaded:
                save_path = UPLOAD_DIR / file.name
                save_path.write_bytes(file.getbuffer())
                try:
                    st.toast(index_document(save_path, sql, coll))
                except Exception as e:
                    st.error(f"Ошибка {file.name}: {e}")
            sql.close()
    st.divider()
    st.subheader("База знаний")
    docs, chunks = get_db_stats()
    col1, col2 = st.columns(2)
    col1.metric("Документы", docs)
    col2.metric("Фрагменты", chunks)
    st.divider()
    if st.button("Очистить историю чата", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("Корпоративный RAG-ассистент")
st.markdown("Ваш интеллектуальный гид по внутренней документации компании.")

# --- Быстрые вопросы ---
st.markdown("### Быстрые вопросы")
quick_queries = ["Сколько дней длится отпуск?", "Как оформить отпуск?",
                 "Какая политика паролей?", "Как передать конфиденциальные данные?"]
cols = st.columns(len(quick_queries))
for i, q in enumerate(quick_queries):
    if cols[i].button(q, use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("assistant"):
            with st.status("Обработка запроса...", expanded=True) as status:
                st.write("Подготовка запроса и поиск...")
                answer, hits, rewritten, qid = handle_rag_query(q, st.session_state.messages)
                st.write(f"Запрос: {rewritten}")
                status.update(label="Ответ готов", state="complete", expanded=False)
            st.markdown(answer)
            render_sources(hits)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "hits": hits, "query_id": qid}
        )
        st.rerun()

st.divider()

# --- История переписки (один цикл, без вложенных) ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] != "assistant":
            continue

        render_sources(message.get("hits") or [])

        qid = message.get("query_id")
        if qid is None:
            continue
        if qid in st.session_state.voted:
            st.caption("Оценка учтена")
            continue

        col1, col2, _ = st.columns([0.05, 0.05, 0.9])
        if col1.button("👍", key=f"up_{qid}"):
            save_feedback(qid, 1)
            st.session_state.voted.add(qid)
            st.rerun()
        if col2.button("👎", key=f"down_{qid}"):
            save_feedback(qid, -1)
            st.session_state.voted.add(qid)
            st.rerun()

# --- Ввод через чат ---
if prompt := st.chat_input("Задайте вопрос по документации..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.status("Обработка запроса...", expanded=True) as status:
            st.write("Подготовка запроса и поиск...")
            answer, hits, rewritten, qid = handle_rag_query(prompt, st.session_state.messages)
            st.write(f"Запрос: {rewritten}")
            status.update(label="Ответ готов", state="complete", expanded=False)
        st.markdown(answer)
        render_sources(hits)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "hits": hits, "query_id": qid}
    )