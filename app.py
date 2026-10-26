import json
import sqlite3
from pathlib import Path
import streamlit as st

from indexer import connect, index_document
from search_engine import hybrid_search

# --- Константы ---
MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"
UPLOAD_DIR = Path("docs")
UPLOAD_DIR.mkdir(exist_ok=True)

# --- Промпты ---
RAG_PROMPT = """Ты — справочный ассистент компании.
Отвечай максимально точно, используя предоставленный контекст и историю переписки.

ПРАВИЛА:
1. Если в контексте есть информация, которая отвечает на вопрос (даже если слова немного отличаются или в вопросе есть опечатки), обязательно сформулируй ответ.
2. ЦИТИРОВАНИЕ: Ставь номер источника [N] сразу после того факта, который ты взял из этого источника.
3. Используй источник [N] только если в его тексте содержится эта информация.
4. Если в контексте действительно нет никакой информации по теме, отвечай: "В предоставленных документах нет ответа на этот вопрос".
5. Не выдумывай факты, которых нет в предоставленном тексте.

История переписки:
{history}

Контекст из документов:
{context}

Вопрос пользователя: {question}
Ответ:"""

REWRITE_PROMPT = """Ты — модуль подготовки поисковых запросов. 
Твоя задача: переписать последний вопрос пользователя так, чтобы он стал полноценным, понятным, грамматически правильным и независимым от контекста предыдущих сообщений. 
Обязательно исправляй опечатки и ошибки в словах.

Пример:
История: "Сколько дней длится отпуск?" -> "28 дней"
Вопрос: "а как его офмить?"
Результат: "Как оформить ежегодный оплачиваемый отпуск?"

История переписки:
{history}

Последний вопрос: {question}
Результат (только текст перефразированного вопроса):"""

# --- Вспомогательные функции ---
def ask_llm(prompt: str, temperature: float = 0.1) -> str:
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }).encode("utf-8")
    
    import urllib.request
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
        (question, answer, json.dumps([h.chunk_id for h in hits], ensure_ascii=False), best_dist),
    )
    sql.commit()
    sql.close()

def get_db_stats():
    try:
        sql = sqlite3.connect("data/metadata.db")
        docs_count = sql.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks_count = sql.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        sql.close()
        return docs_count, chunks_count
    except:
        return 0, 0

def handle_rag_query(prompt, history_messages):
    """Единая точка обработки запроса для всех типов ввода."""
    # 1. Перефразирование
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history_messages[-5:-1]])
    rewritten_query = ask_llm(REWRITE_PROMPT.format(history=history_str, question=prompt))
    
    # 2. Поиск (Без фильтра по дистанции, доверяем Реренкеру)
    all_hits = hybrid_search(rewritten_query)
    final_hits = all_hits[:5]

    if final_hits:
        context = "\n".join(
            f"[{n}] ({h.title}, {h.locator or f'фрагмент {h.chunk_index}'}) {h.content}"
            for n, h in enumerate(final_hits, start=1)
        )
        chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in history_messages[-5:]])
        full_prompt = RAG_PROMPT.format(history=chat_history, context=context, question=prompt)
        answer = ask_llm(full_prompt)
        
        best_dist = final_hits[0].vec_dist if final_hits[0].vec_dist >= 0 else None
        log_query(prompt, answer, final_hits, best_dist)
        return answer, final_hits, rewritten_query
    else:
        return "В предоставленных документах нет релевантной информации для ответа на этот вопрос.", [], rewritten_query

# --- Инициализация состояния ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Интерфейс ---
st.set_page_config(page_title="CorpRAG AI", layout="wide")

with st.sidebar:
    st.title("Настройки")
    with st.expander("Загрузить документы", expanded=False):
        uploaded = st.file_uploader("Выберите файлы", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
        if uploaded:
            sql, coll = connect()
            for file in uploaded:
                save_path = UPLOAD_DIR / file.name
                save_path.write_bytes(file.getbuffer())
                try:
                    msg = index_document(save_path, sql, coll)
                    st.toast(msg)
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

# Блок быстрых вопросов
st.markdown("### Быстрые вопросы")
quick_queries = ["Сколько дней длится отпуск?", "Как оформить отпуск?", "Какая политика паролей?", "Как передать конфиденциальные данные?"]
cols = st.columns(len(quick_queries))
for i, q in enumerate(quick_queries):
    if cols[i].button(q, use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": q})
        # Включаем логику ответа прямо здесь, вызывая функцию handle_rag_query
        with st.chat_message("assistant"):
            with st.status("Обработка запроса...", expanded=True) as status:
                st.write("Подготовка запроса и поиск...")
                answer, hits, rewritten = handle_rag_query(q, st.session_state.messages)
                st.write(f"Запрос: {rewritten}")
                status.update(label="Ответ готов", state="complete", expanded=False)
            st.markdown(answer)
            if hits:
                with st.expander("Источники"):
                    for n, h in enumerate(hits, start=1):
                        st.markdown(f"**[{n}]** `{h.title}`, {h.locator} \n\n {h.content}")
        st.session_state.messages.append({"role": "assistant", "content": answer, "hits": hits})
        st.rerun()

st.divider()

# Отображение истории
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "hits" in message and message["hits"]:
            with st.expander("Источники"):
                for n, h in enumerate(message["hits"], start=1):
                    st.markdown(f"**[{n}]** `{h.title}`, {h.locator} \n\n {h.content}")

# Прием ввода через чат
if prompt := st.chat_input("Задайте вопрос по документации..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Обработка запроса...", expanded=True) as status:
            st.write("Подготовка запроса и поиск...")
            answer, hits, rewritten = handle_rag_query(prompt, st.session_state.messages)
            st.write(f"Запрос: {rewritten}")
            status.update(label="Ответ готов", state="complete", expanded=False)
        st.markdown(answer)
        if hits:
            with st.expander("Источники"):
                for n, h in enumerate(hits, start=1):
                    st.markdown(f"**[{n}]** `{h.title}`, {h.locator} \n\n {h.content}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "hits": hits})
