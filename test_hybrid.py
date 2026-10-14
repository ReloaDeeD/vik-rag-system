"""Сравнение: чистый вектор vs гибрид на кейсе 'сколько дней будет отпуск?'"""
from search_engine import hybrid_search, _load_all, _vector_search

_load_all()
question = "сколько дней будет отпуск?"

print("=== Чистый векторный поиск (топ-3) ===")
for i, h in enumerate(_vector_search(question, 3), 1):
    print(f"{i}. [{h.score:.3f}] {h.title} фрагм. {h.chunk_index}: {h.content[:80]}...")

print("\n=== Гибрид: BM25 + вектор + RRF + re-ranking (топ-3) ===")
for i, h in enumerate(hybrid_search(question), 1):
    print(f"{i}. [{h.score:.3f}, src={h.source}] {h.title} фрагм. {h.chunk_index}: {h.content[:80]}...")