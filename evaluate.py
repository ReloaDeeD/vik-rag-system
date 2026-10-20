"""Шаг 9: оценка качества RAG-системы на тестовом датасете.
Метрики: faithfulness, context precision, answer relevancy.
Поддерживает сравнение режимов: vector, bm25, hybrid, hybrid_rerank.
"""
import argparse
import json
import sys
import urllib.request
import re

from search_engine import search

MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT = """Ты — справочный ассистент компании. Отвечай ТОЛЬКО по контексту ниже.
Правила:
1. Внимательно прочитай контекст. Если в нём есть информация для ответа — сформулируй ОТВЕТ СЛОВАМИ (2–4 предложения), основанный на контексте.
2. ЗАПРЕЩЕНО отвечать только номером источника или только "[1]". Ответ должен содержать текст.
3. В конце ответа перечисли номера источников в квадратных скобках, например: [1], [2].
4. Внимательно ищи ответ в контексте! Если вопрос спрашивает про комбинацию клавиш, сроки, числа или адреса — выпиши их из контекста. Отказывайся ("В предоставленных документах нет ответа..."), только если ты прочитал весь контекст и информации там точно нет.
5. Не используй собственные знания вне контекста и не выдумывай факты.

Контекст:
{context}

Вопрос: {question}
Ответ:"""

FAITHFULNESS_PROMPT = """Оцени, насколько ответ основан на контексте (нет ли галлюцинаций).
Если ответ содержит фразу "В предоставленных документах нет ответа", ставь 1.0.
В остальных случаях: если все факты в ответе подтверждаются контекстом, ставь 1.0. Если есть выдумка — 0.0.
Ответь ТОЛЬКО числом: 1.0 или 0.0.

Контекст: {context}
Вопрос: {question}
Ответ: {answer}
Число:"""

RELEVANCY_PROMPT = """Оцени, отвечает ли ответ на вопрос.
Если ответ содержит фразу "В предоставленных документах нет ответа", но вопрос ожидает ответа (например, спрашивают факт из документа), ставь 0.0.
В остальных случаях: если ответ по смыслу решает вопрос (даже если он короткий), ставь 1.0.
Ответь ТОЛЬКО числом: 1.0 или 0.0.

Вопрос: {question}
Ответ: {answer}
Число:"""


def ask_llm(prompt: str, temperature: float = 0.1) -> str:
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
    match = re.search(r'(0[.,]\d+|1[.,]0*|1|0)', text)
    if match:
        val = float(match.group(1).replace(',', '.'))
        return max(0.0, min(1.0, val))
    text_lower = text.lower()
    if any(word in text_lower for word in ['да', 'yes', 'полностью', 'relevant', 'релевантен']): return 1.0
    if any(word in text_lower for word in ['нет', 'no', 'не', 'not relevant', 'нерелевантен']): return 0.0
    return 0.5


def is_refusal(answer: str) -> bool:
    answer_lower = answer.lower()
    refusal_patterns = [
        'нет ответа на этот вопрос', 'нет информации о', 'не содержится информации',
        'не могу предоставить', 'в документах отсутствует', 'не указано',
    ]
    return any(pattern in answer_lower for pattern in refusal_patterns)


def evaluate_faithfulness(question: str, context: str, answer: str) -> float:
    if is_refusal(answer): return 1.0
    prompt = FAITHFULNESS_PROMPT.format(context=context, question=question, answer=answer)
    response = ask_llm(prompt, temperature=0.0)
    return parse_score(response)


def evaluate_context_precision(hits, expected_sources: list) -> float:
    # Если вопрос-ловушка (нет ожидаемых источников)
    if not expected_sources:
        # Если система ничего не нашла — отлично (1.0). Если нашла — плохо (0.0)
        return 1.0 if not hits else 0.0
    
    if not hits:
        return 0.0
        
    found_sources = set(h.title for h in hits)
    expected_set = set(expected_sources)
    return len(found_sources & expected_set) / len(expected_set)


def evaluate_answer_relevancy(question: str, answer: str, expected_answer: str = "", category: str = "") -> float:
    # 1. Если это ловушка и система честно отказалась — это идеальный ответ (1.0)
    if category == "ловушка" and is_refusal(answer):
        return 1.0
        
    # 2. Если ожидается реальный ответ
    if expected_answer and expected_answer.lower() != "нет ответа":
        # Если система отказалась отвечать на реальный вопрос — это провал (0.0)
        if is_refusal(answer):
            return 0.0
            
        # Прямая проверка ключевых слов (Fast-path)
        exp_lower = expected_answer.lower()
        ans_lower = answer.lower()
        keywords = [w for w in exp_lower.split() if len(w) > 3] 
        if keywords:
            matches = sum(1 for w in keywords if w in ans_lower)
            if matches / len(keywords) >= 0.4: # Если совпало хотя бы 40% ключевых слов
                return 1.0

    # 3. Если прямой проверки недостаточно, спрашиваем LLM
    prompt = RELEVANCY_PROMPT.format(question=question, answer=answer)
    response = ask_llm(prompt, temperature=0.0)
    return parse_score(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="hybrid_rerank", 
                        choices=["vector", "bm25", "hybrid", "hybrid_rerank"],
                        help="Режим поиска для оценки")
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
        category = item.get("category", "unknown")
        
        print(f"[{i}/{len(dataset)}] {category}: {question}")
        
        # Поиск с выбранным режимом
        hits = search(question, mode=mode)
        
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
                "question": question, "category": category, "answer": answer,
                "faithfulness": faithfulness, "context_precision": context_precision,
                "answer_relevancy": answer_relevancy, "hits": []
            })
            print(f"  → нет фрагментов (faith={faithfulness:.2f}, prec={context_precision:.2f}, rel={answer_relevancy:.2f})\n")
            continue
        
        # Генерация ответа
        context = "\n".join(
            f"[{n}] ({h.title}, {h.locator or f'фрагмент {h.chunk_index}'}) {h.content}"
            for n, h in enumerate(hits, start=1)
        )
        answer = ask_llm(PROMPT.format(context=context, question=question))
        
        # Метрики
        faithfulness = evaluate_faithfulness(question, context, answer)
        context_precision = evaluate_context_precision(hits, expected_sources)
        expected_answer = item.get("expected_answer", "")
        answer_relevancy = evaluate_answer_relevancy(question, answer, item.get("expected_answer", ""), category)
        
        results.append({
            "question": question, "category": category, "answer": answer,
            "faithfulness": faithfulness, "context_precision": context_precision,
            "answer_relevancy": answer_relevancy,
            "hits": [{"title": h.title, "chunk_index": h.chunk_index, "source": h.source} for h in hits]
        })
        
        print(f"  → faith={faithfulness:.2f}, prec={context_precision:.2f}, rel={answer_relevancy:.2f}")
        print(f"    ответ: {answer[:100]}...\n")

    # Средние метрики
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_context_precision = sum(r["context_precision"] for r in results) / len(results)
    avg_answer_relevancy = sum(r["answer_relevancy"] for r in results) / len(results)

    print("=" * 60)
    print(f"ИТОГОВЫЕ МЕТРИКИ [{mode.upper()}]:")
    print(f"  Faithfulness:        {avg_faithfulness:.3f}")
    print(f"  Context Precision:   {avg_context_precision:.3f}")
    print(f"  Answer Relevancy:    {avg_answer_relevancy:.3f}")
    print("=" * 60)

    # Сохранение результатов (файл зависит от режима)
    out_file = f"eval_results_{mode}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "mode": mode,
            "metrics": {
                "faithfulness": avg_faithfulness,
                "context_precision": avg_context_precision,
                "answer_relevancy": avg_answer_relevancy
            },
            "results": results
        }, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены в {out_file}")


if __name__ == "__main__":
    main()