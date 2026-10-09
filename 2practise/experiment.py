import json
import random
import numpy as np
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# Направляем стандартный клиент OpenAI на локальный сервер Ollama
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama', # Ollama не проверяет ключ, но SDK требует строку
)
# Выбираем скачанную модель (qwen2.5:0.5b или llama3)
MODEL_NAME = "qwen2.5:0.5b"

print("=========================================")
print(f"ЧАСТЬ A: Эксперимент по выбору инструмента (Модель: {MODEL_NAME})")
print("=========================================")

# 1. Сбор пула из 100 инструментов
# base_tools = [
#     {"name": "ocr_init_task", "description": "Создает задачу на OCR. Принимает file_path (.jpg, .png, .pdf)."},
#     {"name": "ocr_estimate_width", "description": "Оценивает среднюю ширину символов. Требует handle."},
#     {"name": "ocr_extract_text", "description": "Возвращает распознанный текст (concise/detailed). Требует handle."},
#     {"name": "ocr_get_bboxes", "description": "Возвращает количество bounding boxes. Требует handle."}
# ]
# similar_tools = [{"name": f"cv_tool_{i}", "description": f"Обработка изображений, сегментация, маски, CV. Вариант {i}."} for i in range(20)]
# random_tools = [{"name": f"random_tool_{i}", "description": f"Абстрактная утилита для работы с БД или API. Вариант {i}."} for i in range(76)]
# all_tools = base_tools + similar_tools + random_tools


# Базовые инструменты (остаются теми же)
base_tools = [
    {"name": "ocr_init_task", "description": "Создает задачу на OCR. Принимает file_path (.jpg, .png, .pdf)."},
    {"name": "ocr_estimate_width", "description": "Оценивает среднюю ширину символов. Требует handle."},
    {"name": "ocr_extract_text", "description": "Возвращает распознанный текст (concise/detailed). Требует handle."},
    {"name": "ocr_get_bboxes", "description": "Возвращает количество bounding boxes. Требует handle."}
]

# "Враждебные" инструменты-двойники, которые сбивают с толку
adversarial_tools = [
    {"name": "init_video_task", "description": "Создает задачу на обработку. Принимает file_path (.mp4, .avi)."},
    {"name": "ocr_extract_metadata", "description": "Возвращает только служебные метаданные документа, без самого текста. Требует handle."},
    {"name": "text_extractor_pro", "description": "Возвращает распознанный текст. Работает только со ссылками на веб-страницы (URL)."},
    {"name": "estimate_image_width", "description": "Оценивает ширину самой картинки в пикселях. Требует file_path."},
    {"name": "get_all_boxes", "description": "Возвращает bounding boxes для поиска лиц на фото. Требует image_id."}
]
# Размножаем их, чтобы заполнить пул
similar_tools = [{"name": f"{t['name']}_{i}", "description": t["description"]} for i in range(4) for t in adversarial_tools]
random_tools = [{"name": f"random_tool_{i}", "description": f"Абстрактная утилита для работы с БД или API. Вариант {i}."} for i in range(76)]

all_tools = base_tools + similar_tools + random_tools

# 2. Подготовка 30 запросов
# queries = []
# for i in range(30):
#     target = base_tools[i % 4]
#     queries.append({
#         "query_id": i,
#         "text": f"Мне нужно выполнить следующую задачу: {target['description'].split('.')[0].lower()}.",
#         "target_tool": target["name"]
#     })

# Человеческие запросы, в которых нет прямых слов из описаний
human_queries = [
    {"target_tool": "ocr_init_task", "text": "Я тут сфоткал чек на телефон, хочу закинуть его в систему на чтение."},
    {"target_tool": "ocr_estimate_width", "text": "Нам нужно понять, насколько крупный шрифт используется в этом скане."},
    {"target_tool": "ocr_extract_text", "text": "Вытащи мне сумму и позиции товаров из загруженного инвойса."},
    {"target_tool": "ocr_get_bboxes", "text": "Сколько всего отдельных букв или слов нашла нейронка на этой странице?"},
    {"target_tool": "ocr_init_task", "text": "Загрузи пдфку с договором в пайплайн."},
]
# Размножаем до 30 штук с уникальными ID
queries = []
for i in range(30):
    q = human_queries[i % 5].copy()
    q["query_id"] = i
    queries.append(q)

print("Загрузка sentence-transformers...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
tool_embeddings = embedder.encode([t["description"] for t in all_tools])

def get_top_5_tools(query_text, current_tools):
    q_emb = embedder.encode([query_text])
    curr_emb = embedder.encode([t["description"] for t in current_tools])
    sims = cosine_similarity(q_emb, curr_emb)[0]
    top_indices = np.argsort(sims)[-5:][::-1]
    return [current_tools[i] for i in top_indices], top_indices

def run_selection(query, available_tools):
    prompt = f"Выбери один инструмент для задачи: '{query}'. Доступны: {json.dumps(available_tools, ensure_ascii=False)}. Верни только точное имя инструмента без лишнего текста."
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.0
        )
        usage = response.usage
        tokens = usage.prompt_tokens if usage else 0
        return response.choices[0].message.content.strip(), tokens
    except Exception as e:
        print(f"Ошибка API: {e}")
        return "", 0

results_a = []
N_values = [5, 20, 50, 100]
RUNS = 3

print("Начинаем прогон запросов к LLM (зайдет несколько минут)...")
for N in N_values:
    current_tools = all_tools[:N]
    for q in queries:
        top_5, top_idx = get_top_5_tools(q["text"], current_tools)
        recall_5 = 1 if any(current_tools[i]["name"] == q["target_tool"] for i in top_idx) else 0
        
        for run_id in range(RUNS):
            # Вариант (A)
            selected_A, tokens_A = run_selection(q["text"], current_tools)
            is_correct_A = 1 if q["target_tool"] in selected_A else 0
            results_a.append({"N": N, "method": "Full Context", "query_id": q["query_id"], "is_correct": is_correct_A, "recall_5": np.nan, "tokens": tokens_A})
            
            # Вариант (B)
            selected_B, tokens_B = run_selection(q["text"], top_5)
            is_correct_B = 1 if q["target_tool"] in selected_B else 0
            results_a.append({"N": N, "method": "RAG Top-5", "query_id": q["query_id"], "is_correct": is_correct_B, "recall_5": recall_5, "tokens": tokens_B})

df_a = pd.DataFrame(results_a)

def bootstrap_ci(data, n_iterations=1000):
    queries_ids = data['query_id'].unique()
    scores = []
    for _ in range(n_iterations):
        sample_q = np.random.choice(queries_ids, size=len(queries_ids), replace=True)
        sample_df = pd.concat([data[data['query_id'] == q] for q in sample_q])
        scores.append(sample_df["is_correct"].mean())
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)

print("\n--- РЕЗУЛЬТАТЫ ЧАСТИ A ---")
for N in N_values:
    for method in ["Full Context", "RAG Top-5"]:
        subset = df_a[(df_a["N"] == N) & (df_a["method"] == method)]
        acc = subset["is_correct"].mean()
        tokens = subset["tokens"].mean()
        ci_low, ci_high = bootstrap_ci(subset)
        
        if method == "RAG Top-5":
            recall = subset["recall_5"].mean()
            print(f"N={N:3} | {method:12} | Точность: {acc:.2f} ({ci_low:.2f}-{ci_high:.2f}) | Токены: {tokens:.0f} | Recall@5: {recall:.2f}")
        else:
            print(f"N={N:3} | {method:12} | Точность: {acc:.2f} ({ci_low:.2f}-{ci_high:.2f}) | Токены: {tokens:.0f}")

print("\n=========================================")
print("ЧАСТЬ B: Эксперимент с петлями (LoopGuard)")
print("=========================================")

class LoopGuard:
    def __init__(self, max_same=3, window=6):
        self.calls, self.obs = [], []
        self.n, self.w = max_same, window

    def check(self, name, args, observation):
        self.calls.append(name + json.dumps(args, sort_keys=True))
        self.obs.append(str(observation))
        last = self.calls[-self.w:]
        
        if Counter(last)[last[-1]] >= self.n:
            return "повтор: смените инструмент или аргументы"
        if len(last) >= 4 and last[-4:-2] == last[-2:]:
            return "колебание между двумя действиями"
        return None

def unstable_ocr_extract(mode="unstable_no_hint"):
    if random.random() < 0.5:
        if mode == "unstable_no_hint":
            return {"isError": False, "content": ""}
        elif mode == "fixed_hint":
            return {"isError": True, "content": "Ошибка: ничего не найдено; попробуйте другое значение."}
    return {"isError": False, "content": "Текст успешно распознан"}

def simulate_episode(use_guard=False, mode="unstable_no_hint", max_steps=10):
    guard = LoopGuard(max_same=3, window=6)
    
    for step in range(max_steps):
        args = {"handle": "doc_42", "response_format": "concise"}
        obs = unstable_ocr_extract(mode)
        
        if use_guard:
            guard_signal = guard.check("ocr_extract", args, obs)
            if guard_signal:
                return {"success": False, "loop": True, "steps": step + 1}
        
        if obs.get("content") == "Текст успешно распознан":
            return {"success": True, "loop": False, "steps": step + 1}
            
    return {"success": False, "loop": True, "steps": max_steps}

conditions = [
    {"name": "1. Без LoopGuard, API возвращает пустоту", "guard": False, "mode": "unstable_no_hint"},
    {"name": "2. С LoopGuard, API возвращает пустоту", "guard": True, "mode": "unstable_no_hint"},
    {"name": "3. Без LoopGuard, API дает подсказку-ошибку", "guard": False, "mode": "fixed_hint"}
]

print("--- РЕЗУЛЬТАТЫ ЧАСТИ B ---")
for cond in conditions:
    results_b = [simulate_episode(cond["guard"], cond["mode"]) for _ in range(20)]
    success_rate = sum(1 for r in results_b if r["success"]) / 20
    loop_rate = sum(1 for r in results_b if r["loop"]) / 20
    avg_steps = sum(r["steps"] for r in results_b) / 20
    
    print(f"{cond['name']}")
    print(f"  Успех: {success_rate*100:3.0f}% | Петли: {loop_rate*100:3.0f}% | Среднее число шагов: {avg_steps:.1f}")