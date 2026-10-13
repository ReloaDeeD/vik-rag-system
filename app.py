"""Шаг 7: веб-интерфейс RAG-ассистента на Streamlit.
Запуск: streamlit run app.py
"""
import json
import sqlite3
import chromadb
import streamlit as st
from chromadb.utils import embedding_functions

# --- Константы, что в generate_answer.py ---
import urllib.request
THRESHOLD = 0.7
MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

PROMPT = """Ты — справочный ассистент компании. Отвечай ТОЛЬКО по контексту ниже.
Правила:
1. Внимательно прочитай контекст. Если в нём есть информация для ответа — сформулируй ответ по контексту (2–4 предложения) и в конце перечисли номера источников в квадратных скобках, например: [1], [2].
2. Отказывайся ТОЛЬКО тогда, когда в контексте действительно нет информации, нужной для ответа. Фраза отказа: "В предоставленных документах нет ответа на этот вопрос." Без номеров источников.
3. Не используй собственные знания вне контекста и не выдумывай факты.

Контекст:
{context}

Вопрос: {question}
Ответ:"""


def search(question: str):
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path="data/chroma")
    coll = client.get_or_create_collection("corp_docs", embedding_function=ef)
    result = coll.query(query_texts=[question], n_results=3)
    hits = []
    for doc, dist, meta in zip(result["documents"][0], result["distances"][0], result["metadatas"][0]):
        if dist <= THRESHOLD:
            hits.append((doc, dist, meta))
    return hits


def ask_llm(prompt: str) -> str:
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},   # стабильные фактические ответы, без дрейфа
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))["response"]


def log_query(question: str, answer: str, hits: list, best_dist: float):
    sql = sqlite3.connect("data/metadata.db")
    sql.execute(
        "INSERT INTO query_log (question, answer, source_chunk_ids, best_distance) VALUES (?, ?, ?, ?)",
        (question, answer,
         json.dumps([m["chunk_id"] for _, _, m in hits], ensure_ascii=False),
         best_dist),
    )
    sql.commit()
    sql.close()


# --- Интерфейс ---
st.set_page_config(page_title="RAG-ассистент", layout="centered")
st.title("RAG-ассистент по корпоративной документации")
st.caption("Локальный поиск с генерацией ответа. Данные не покидают контур предприятия.")

with st.form("qa_form"):
    question = st.text_input("Ваш вопрос:", placeholder="например, сколько дней длится отпуск?")
    submitted = st.form_submit_button("Найти ответ")

if submitted and question.strip():
    with st.spinner("Ищем и формулируем ответ…"):
        hits = search(question)
        answer = None
        if hits:
            context = "\n".join(
                f"[{n}] ({m['title']}, фрагмент {m['chunk_index']}) {t}"
                for n, (t, d, m) in enumerate(hits, start=1)
            )
            answer = ask_llm(PROMPT.format(context=context, question=question))
            log_query(question, answer, hits, hits[0][1])   # пишем в журнал ТОЛЬКО по кнопке
    st.session_state["last"] = {"question": question, "hits": hits, "answer": answer}

if "last" in st.session_state:
    res = st.session_state["last"]
    hits, answer = res["hits"], res["answer"]

    st.subheader(f"Вопрос: {res['question']}")
    # 1. Ответ — всегда сверху и сразу виден
    if not hits:
        st.warning("В документах нет релевантных фрагментов на этот вопрос.")
    elif answer and "нет ответа на этот вопрос" in answer.lower():
        st.warning(answer)          # отказ — жёлтым, а не зелёным
    else:
        st.success(answer)          # ответ — зелёным

    # 2. Источники — свёрнутыми под ответом
    if hits:
        with st.expander(f"Найдено фрагментов: {len(hits)}"):
            for n, (doc, dist, meta) in enumerate(hits, start=1):
                st.markdown(f"**[{n}]** `{meta['title']}`, фрагмент {meta['chunk_index']} — расстояние **{dist:.3f}**")
                st.info(doc)

    # 3. Журнал — в самом низу
    with st.expander("Журнал последних запросов"):
        sql = sqlite3.connect("data/metadata.db")
        rows = sql.execute(
            "SELECT asked_at, question, best_distance FROM query_log ORDER BY asked_at DESC LIMIT 10"
        ).fetchall()
        sql.close()
        for asked, q, d in rows:
            st.write(f"`{asked}` | {d:.3f} | {q}")