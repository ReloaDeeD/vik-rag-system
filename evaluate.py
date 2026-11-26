"""Оценка качества RAG-системы. Честные метрики с учетом морфологии и фактов."""
import argparse
import json
import re
import urllib.request
import snowballstemmer

from search_engine import search
from prompts import RAG_PROMPT

MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"
stemmer = snowballstemmer.stemmer('russian')

# Строгий судья для проверки галлюцинаций
FAITHFULNESS_PROMPT = """Оцени, подтверждается ли ответ текстом контекста.
Если ответ содержит фразу "нет ответа", но вопрос ожидает ответа, ставь 0.0.
Если ответ содержит факты (цифры, сроки), которых НЕТ в контексте, ставь 0.0.
Если все факты в ответе есть в контексте, ставь 1.0.
Ответь ТОЛЬКО числом: 1.0 или 0.0.

Контекст: {context}
Вопрос: {question}
Ответ: {answer}
Число:"""

def ask_llm(prompt: str, temperature: float = 0.0) -> str:
    payload = json.dumps({
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))["response"].strip()
    except Exception as e:
        print(f"  [Ошибка LLM: {e}]")
        return ""

def parse_score(text: str) -> float:
    match = re.search(r'(0[.,]\d+|1[.,]0*|1|0)', text)
    if match:
        return max(0.0, min(1.0, float(match.group(1).replace(',', '.'))))
    return 0.5

def is_refusal(answer: str) -> bool:
    answer_lower = answer.lower()
    refusals = ['нет ответа', 'нет информации', 'не содержится', 'не могу', 'отсутствует', 'не указано']
    return any(r in answer_lower for r in refusals)

def get_stems(text: str) -> set:
    """Разбивает текст на слова, убирает мусор и возвращает основы слов (стемы)."""
    words = re.findall(r'\b[а-яёa-z]{4,}\b', text.lower())
    return set(stemmer.stemWords(words))

def evaluate_context_precision(hits, expected_answer: str, category: str) -> float:
    """Проверяет, попали ли нужные факты в найденные чанки."""
    if category == "ловушка" or not expected_answer or expected_answer.lower() == "нет ответа":
        return 1.0 if not hits else 0.0  # Для ловушек: либо пусто (1.0), либо мусор (0.0)
    
    if not hits:
        return 0.0
        
    # Ищем ключевые слова из ожидаемого ответа внутри найденных чанков
    context_text = " ".join([h.content for h in hits])
    expected_stems = get_stems(expected_answer)
    context_stems = get_stems(context_text)
    
    if not expected_stems: return 1.0
    overlap = len(expected_stems & context_stems)
    return 1.0 if (overlap / len(expected_stems)) >= 0.3 else 0.0

def evaluate_faithfulness(question: str, context: str, answer: str, category: str, expected_answer: str) -> float:
    """Оценивает, есть ли в ответе галлюцинации."""
    refusal = is_refusal(answer)
    
    if category == "ловушка":
        return 1.0 if refusal else 0.0  # Если на ловушку ответили - это галлюцинация
    
    if not expected_answer or expected_answer.lower() == "нет ответа":
        return 1.0 if refusal else 0.0
        
    if refusal:
        return 0.0  # Отказ, когда ответ ЕСТЬ в документах - это провал

    prompt = FAITHFULNESS_PROMPT.format(context=context, question=question, answer=answer)
    return parse_score(ask_llm(prompt))

def evaluate_answer_relevancy(question: str, answer: str, expected_answer: str, category: str) -> float:
    """Оценивает, отвечает ли бот на вопрос."""
    refusal = is_refusal(answer)
    
    if category == "ловушка":
        return 1.0 if refusal else 0.0

    if not expected_answer or expected_answer.lower() == "нет ответа":
        return 1.0 if refusal else 0.0

    if refusal:
        return 0.0  # Отказ на нормальный вопрос

    # Быстрая проверка по стемам
    expected_stems = get_stems(expected_answer)
    answer_stems = get_stems(answer)
    
    if expected_stems:
        overlap = len(expected_stems & answer_stems)
        # Если совпало больше 40% ключевых слов из эталона - ответ релевантен
        if (overlap / len(expected_stems)) >= 0.4:
            return 1.0

    # Если по словам не совпало, спрашиваем LLM
    prompt = f"Отвечает ли этот ответ на вопрос?\nВопрос: {question}\nОтвет: {answer}\nОтветь только числом 1.0 или 0.0:"
    return parse_score(ask_llm(prompt))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="hybrid_rerank", 
                        choices=["vector", "bm25", "hybrid", "hybrid_rerank"])
    args = parser.parse_args()
    mode = args.mode

    print(f"Загрузка датасета... Режим поиска: [{mode.upper()}]")
    with open("eval_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Загружено {len(dataset)} вопросов.\n")

    results = []
    for i, item in enumerate(dataset, 1):
        question = item["question"]
        expected_sources = item.get("expected_sources", [])
        expected_answer = item.get("expected_answer", "")
        category = item.get("category", "unknown")
        
        print(f"[{i}/{len(dataset)}] {category}: {question}")
        
        hits = search(question, mode=mode)
        
        if not hits:
            answer = "В предоставленных документах нет ответа на этот вопрос."
        else:
            context = "\n".join(
                f"[{n}] ({h.title}, {h.locator or f'фрагмент {h.chunk_index}'}) {h.content}"
                for n, h in enumerate(hits, start=1)
            )
            answer = ask_llm(RAG_PROMPT.format(context=context, question=question))
            
            # Обрезаем ответ, если модель начала писать "[N] (vacation.md...)"
            if answer.startswith("[") and "]" in answer[:5]:
                pass # Оставляем как есть, если это формат ссылки

        faith = evaluate_faithfulness(question, context if hits else "", answer, category, expected_answer)
        prec = evaluate_context_precision(hits, expected_answer, category)
        rel = evaluate_answer_relevancy(question, answer, expected_answer, category)
        
        results.append({
            "question": question, "category": category, "answer": answer,
            "faithfulness": faith, "context_precision": prec, "answer_relevancy": rel
        })
        
        print(f"  → faith={faith:.2f}, prec={prec:.2f}, rel={rel:.2f}")
        print(f"    ответ: {answer[:100]}...\n")

    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_prec = sum(r["context_precision"] for r in results) / len(results)
    avg_rel = sum(r["answer_relevancy"] for r in results) / len(results)

    print("=" * 60)
    print(f"ИТОГОВЫЕ МЕТРИКИ [{mode.upper()}]:")
    print(f"  Faithfulness:        {avg_faith:.3f}")
    print(f"  Context Precision:   {avg_prec:.3f}")
    print(f"  Answer Relevancy:    {avg_rel:.3f}")
    print("=" * 60)

    with open(f"eval_results_{mode}.json", "w", encoding="utf-8") as f:
        json.dump({"mode": mode, "metrics": {"faith": avg_faith, "prec": avg_prec, "rel": avg_rel}, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены в eval_results_{mode}.json")

if __name__ == "__main__":
    main()