import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
client = chromadb.PersistentClient(path="data/chroma")
text = Path("docs/reglament.txt").read_text(encoding="utf-8")
docs = [p.strip() for p in text.split("\n\n") if p.strip()]
ids = [f"p{i}" for i in range(len(docs))]
coll = client.get_or_create_collection("reglament", embedding_function=ef)
coll.upsert(ids=ids, documents=docs)
result = coll.query(query_texts=["что будет за плохой пароль?"], n_results=3)
for doc, dist in zip(result["documents"][0], result["distances"][0]):
    print(f"{dist:.3f} | {doc}")

best_dist = result["distances"][0][0]
best_doc = result["documents"][0][0]

if best_dist > 0.9:
    print("В документах нет ответа на этот вопрос.")
else:
    print(f"Ответ: {best_doc} (расстояние {best_dist:.3f})")