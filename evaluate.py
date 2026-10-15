"""Шаг 9: оценка качества RAG-системы на тестовом датасете.
Метрики: faithfulness, context precision, answer relevancy.
"""
import json
import sys
from search_engine import hybrid_search

MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT = """Ты — справочный ассистент компании. Отвечай ТОЛЬКО по контексту ниже.
Правила:
1. Внимательно прочитай контекст. Если в нём есть информация для ответа — сформулируй ОТВЕТ СЛОВАМИ (2–4 предложения), основанный на контексте.
2. ЗАПРЕЩЕНО отвечать только номером источника или только "[1]". Ответ должен содержать текст.
3. В конце ответа перечисли номера источников в квадратных скобках, например: [1], [2].
4. Отказывайся ТОЛЬКО тогда, когда в контексте действительно нет информации. Фраза отказа: "В предоставленных документах нет ответа на этот вопрос." Без номеров источников.
5. Не используй собственные знания вне контекста и не выдумывай факты.

Контекст:
{context}

Вопрос: {question}
Ответ:"""

FAITHFULNESS_PROMPT = """Оцени, насколько ответ основан на контексте (нет ли галлюцинаций).
Контекст:
{context}

Вопрос: {question}

Ответ: {answer}

Все ли факты в ответе подтверждаются контекстом? Ответь ТОЛЬКО ОДНИМ числом от 0 до 1, где 1 = полностью основан на контексте, 0 = полностью выдуман. Никаких пояснений.
Число:"""

RELEVANCY_PROMPT = """Оцени, насколько ответ релевантен вопросу.
Вопрос: {question}

Ответ: {answer}

Отвечает ли ответ на вопрос? Ответь ТОЛЬКО ОДНИМ числом от 0 до 1, где 1 = полностью релевантен, 0 = не релевантен. Никаких пояснений.
Число:"""


def ask_llm(prompt: str, temperature: float = 0.1) -> str:
    import urllib.request
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
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
    """Извлекает число 0-1 из текста LLM."""
    import re
    # Ищем число с точкой или запятой
    match = re.search(r'(0[.,]\d+|1[.,]0*|1|0)', text)
    if match:
        val = float(match.group(1).replace(',', '.'))
        return max(0.0, min(1.0, val))
    # Если нет числа, проверяем на "да"/"нет"
    text_lower = text.lower()
    if any(word in text_lower for word in ['да', 'yes', 'полностью', 'relevant', 'релевантен']):
        return 1.0
    if any(word in text_lower for word in ['нет', 'no', 'не', 'not relevant', 'нерелевантен']):
        return 0.0
    return 0.5  # дефолт, если не распарсилось


def is_refusal(answer: str) -> bool:
    """Проверяет, является ли ответ честным отказом."""
    answer_lower = answer.lower()
    refusal_patterns = [
        'нет ответа на этот вопрос',
        'нет информации о',
        'не содержится информации',
        'не могу предоставить',
        'в документах отсутствует',
        'не указано',
    ]
    return any(pattern in answer_lower for pattern in refusal_patterns)


def evaluate_faithfulness(question: str, context: str, answer: str) -> float:
    """LLM-as-judge: насколько ответ основан на контексте."""
    if is_refusal(answer):
        return 1.0  # честный отказ = высокая faithfulness
    prompt = FAITHFULNESS_PROMPT.format(context=context, question=question, answer=answer)
    response = ask_llm(prompt, temperature=0.0)
    score = parse_score(response)
    # Отладка: если низкий балл, показываем, что ответила LLM
    if score < 0.5:
        print(f"    [DEBUG faithfulness] LLM ответила: '{response[:100]}...'")
    return score


def evaluate_context_precision(hits, expected_sources: list) -> float:
    """Проверка: есть ли в топ-5 фрагментов ожидаемые источники?"""
    if not expected_sources:
        return 1.0
    if not hits:
        return 0.0
    found_sources = set(h.title for h in hits)
    expected_set = set(expected_sources)
    return len(found_sources & expected_set) / len(expected_set)


def evaluate_answer_relevancy(question: str, answer: str) -> float:
    """LLM-as-judge: насколько ответ релевантен вопросу."""
    if is_refusal(answer):
        return 1.0
    prompt = RELEVANCY_PROMPT.format(question=question, answer=answer)
    response = ask_llm(prompt, temperature=0.0)
    score = parse_score(response)
    # Отладка: если низкий балл, показываем, что ответила LLM
    if score < 0.5:
        print(f"    [DEBUG relevancy] LLM ответила: '{response[:100]}...'")
    return score


def main():
    print("Загрузка датасета...")
    with open("eval_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Загружено {len(dataset)} вопросов.\n")

    results = []
    for i, item in enumerate(dataset, 1):
        question = item["question"]
        expected_sources = item.get("expected_sources", [])
        category = item.get("category", "unknown")
        
        print(f"[{i}/{len(dataset)}] {category}: {question}")
        
        # Поиск
        hits = hybrid_search(question)
        
        if not hits:
            if not expected_sources:
                faithfulness = 1.0
                context_precision = 1.0
                answer_relevancy = 1.0
                answer = "В предоставленных документах нет ответа на этот вопрос."
            else:
                faithfulness = 0.0
                context_precision = 0.0
                answer_relevancy = 0.0
                answer = "[ОШИБКА: нет фрагментов]"
            
            results.append({
                "question": question,
                "category": category,
                "answer": answer,
                "faithfulness": faithfulness,
                "context_precision": context_precision,
                "answer_relevancy": answer_relevancy,
                "hits": []
            })
            print(f"  → нет фрагментов (faithfulness={faithfulness:.2f}, precision={context_precision:.2f}, relevancy={answer_relevancy:.2f})\n")
            continue
        
        # Генерация ответа
        context = "\n".join(
            f"[{n}] ({h.title}, фрагмент {h.chunk_index}) {h.content}"
            for n, h in enumerate(hits, start=1)
        )
        answer = ask_llm(PROMPT.format(context=context, question=question))
        
        # Метрики
        faithfulness = evaluate_faithfulness(question, context, answer)
        context_precision = evaluate_context_precision(hits, expected_sources)
        answer_relevancy = evaluate_answer_relevancy(question, answer)
        
        results.append({
            "question": question,
            "category": category,
            "answer": answer,
            "faithfulness": faithfulness,
            "context_precision": context_precision,
            "answer_relevancy": answer_relevancy,
            "hits": [{"title": h.title, "chunk_index": h.chunk_index, "source": h.source} for h in hits]
        })
        
        print(f"  → faithfulness={faithfulness:.2f}, precision={context_precision:.2f}, relevancy={answer_relevancy:.2f}")
        print(f"    ответ: {answer[:100]}...\n")

    # Средние метрики
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_context_precision = sum(r["context_precision"] for r in results) / len(results)
    avg_answer_relevancy = sum(r["answer_relevancy"] for r in results) / len(results)

    print("=" * 60)
    print("ИТОГОВЫЕ МЕТРИКИ:")
    print(f"  Faithfulness:        {avg_faithfulness:.3f} (цель: ≥0.85)")
    print(f"  Context Precision:   {avg_context_precision:.3f} (цель: ≥0.80)")
    print(f"  Answer Relevancy:    {avg_answer_relevancy:.3f}")
    print("=" * 60)

    # Сохранение результатов
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "faithfulness": avg_faithfulness,
                "context_precision": avg_context_precision,
                "answer_relevancy": avg_answer_relevancy
            },
            "results": results
        }, f, ensure_ascii=False, indent=2)
    print("\nРезультаты сохранены в eval_results.json")


if __name__ == "__main__":
    main()