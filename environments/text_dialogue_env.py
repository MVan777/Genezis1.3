"""
Среда Текстового Диалогового ИИ (Text Dialogue Environment)
Конвертирует фразы на русском языке в числовые векторы R^16
и обучает Ассоциативный Мозг отвечать на вопросы.
"""

import math
import random
import hashlib
import numpy as np
import json
import os

DEFAULT_DIALOGUES = [
    # Приветствия
    {"prompt": "Привет!", "answer": "Привет! Рад тебя видеть!"},
    {"prompt": "Здравствуй", "answer": "Здравствуйте! Чем могу помочь?"},
    {"prompt": "Доброе утро", "answer": "Доброе утро! Хорошего дня!"},
    {"prompt": "Добрый вечер", "answer": "Добрый вечер! Как прошёл день?"},
    {"prompt": "Эй, салют!", "answer": "Привет-привет! Что нового?"},

    # Вопросы о себе и состоянии
    {"prompt": "Как дела?", "answer": "Отлично! Мой ассоциативный граф работает на полную мощность."},
    {"prompt": "Как ты себя чувствуешь?", "answer": "Замечательно! Нейроны активны и готовы к обучению."},
    {"prompt": "Кто ты?", "answer": "Я — Ассоциативный ИИ Genezis 2.0."},
    {"prompt": "Как тебя зовут?", "answer": "Меня зовут Genezis, разум на основе графа связей."},
    {"prompt": "Что ты умеешь?", "answer": "Я умею играть в игры, обучаться диалогам и находить связи в данных."},

    # Вопросы о магазине и торговле
    {"prompt": "Где купить товар?", "answer": "Вы можете оформить заказ прямо в нашем Telegram-боте."},
    {"prompt": "Сколько стоит доставка?", "answer": "Доставка рассчитывается автоматически при оформлении корзины."},
    {"prompt": "Есть ли скидки?", "answer": "Да! Для постоянных клиентов у нас действуют промокоды."},
    {"prompt": "Как оплатить заказ?", "answer": "Оплата доступна картой, через СБП или криптовалютой."},
    {"prompt": "Где посмотреть каталог?", "answer": "Каталог товаров доступен по кнопке Меню в боте."},

    # Философия и интеллект
    {"prompt": "Ты живой?", "answer": "Я постоянно развиваюсь и создаю новые нейронные связи!"},
    {"prompt": "Что такое ассоциация?", "answer": "Ассоциация — это связь между прошлым опытом и будущим решением."},
    {"prompt": "Ты умеешь думать?", "answer": "Да, я провожу 5-шаговое симулирование вариантов в графе памяти."},
    {"prompt": "Что такое любовь?", "answer": "Это сильнейшая позитивная ассоциативная связь в памяти!"},

    # Прощание
    {"prompt": "Пока!", "answer": "До свидания! Был рад пообщаться!"},
    {"prompt": "До встречи", "answer": "До скорой встречи! Возвращайтесь скорее."},
    {"prompt": "Увидимся", "answer": "Хорошего дня! До новых встреч."},
    {"prompt": "Спасибо за помощь", "answer": "Всегда пожалуйста! Обращайтесь в любой момент."}
]

import torch

# Карты синонимов и корней слов русского языка
SYNONYM_MAP = {
    "приветик": "привет",
    "приветствую": "привет",
    "салют": "привет",
    "здрасте": "здравствуй",
    "здравствуйте": "здравствуй",
    "хей": "привет",
    "салам": "привет",
    "делишки": "дела",
    "житуха": "дела",
    "жизнь": "дела",
    "пока": "пока",
    "прощай": "пока",
    "досвидания": "пока",
    "пака": "пока",
    "благодарю": "спасибо",
    "спасбо": "спасибо",
    "спс": "спасибо",
}

def stem_russian_word(word: str) -> str:
    """Извлечение смыслового корня слова (русский стеммер)"""
    w = word.lower().strip("!?,. \t\n")
    if w in SYNONYM_MAP:
        return SYNONYM_MAP[w]

    for ending in ("ами", "ями", "ого", "его", "ему", "ому", "ешь", "ете", "им", "ым", "ом", "ем", "ах", "ях", "ик", "чик", "ок", "а", "я", "о", "е", "у", "ю", "ы", "и"):
        if len(w) > 4 and w.endswith(ending):
            return w[:-len(ending)]
    return w

class SemanticTextVectorizer:
    """Семантический векторизатор на PyTorch для работы с синонимами и опечатками"""

    def __init__(self, dim: int = 32):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        """Перевод текста в смысловой вектор R^dim на PyTorch"""
        text_clean = text.lower().strip("!?,. \t\n")
        if not text_clean:
            return np.zeros(self.dim, dtype=np.float32)

        words = text_clean.split()
        stemmed_words = [stem_russian_word(w) for w in words]

        # Создаем тензор PyTorch для семантического аккумулятора
        tensor_vec = torch.zeros(self.dim, dtype=torch.float32)

        for i, word in enumerate(stemmed_words):
            h = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            idx = h % self.dim
            # Относительный вес слова
            weight = 3.0 if len(word) > 3 else 1.5
            tensor_vec[idx] += weight * (1.0 + 0.1 * i)

            # Буквенные триграммы для защиты от опечаток
            for k in range(len(word) - 2):
                sub = word[k:k+3]
                sub_idx = int(hashlib.sha256(sub.encode('utf-8')).hexdigest(), 16) % self.dim
                tensor_vec[sub_idx] += 0.2

        # Нормализация тензора L2
        norm = torch.norm(tensor_vec)
        if norm > 1e-6:
            tensor_vec = tensor_vec / norm

        return tensor_vec.numpy()

global_vectorizer = SemanticTextVectorizer(dim=32)

def encode_text_to_vector(text: str, dim: int = 32) -> np.ndarray:
    return global_vectorizer.encode(text)

class TextDialogueEnv:
    """Среда обучения ИИ диалогам на естественном языке"""

    def __init__(self, custom_dataset_path: str = None):
        self.dialogues = list(DEFAULT_DIALOGUES)
        if custom_dataset_path and os.path.exists(custom_dataset_path):
            self.load_custom_dialogues(custom_dataset_path)

        # Выделяем уникальные ответы для формирования пространства действий
        self.answers = sorted(list(set(d['answer'] for d in self.dialogues)))
        self.answer_to_idx = {ans: i for i, ans in enumerate(self.answers)}
        self.action_count = len(self.answers)

        self.current_dialogue = None
        self.current_prompt = ""
        self.target_answer_idx = 0
        self.step_count = 0

    def load_custom_dialogues(self, filepath: str):
        """Загрузка внешнего словаря из JSON файла"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if 'prompt' in item and 'answer' in item:
                            self.dialogues.append(item)
            print(f"  ✅ Загружен внешний словарь из {filepath} (Всего пар: {len(self.dialogues)})")
        except Exception as e:
            print(f"  ⚠️ Ошибка чтения файла {filepath}: {e}")

    def reset(self, prompt_text: str = None) -> np.ndarray:
        """Сбросить среду и выдать новый случайный или заданный вопрос"""
        self.step_count = 0
        if prompt_text:
            self.current_prompt = prompt_text
            # Ищем совпадение в базе или выбираем ближайший ответ
            matched = [d for d in self.dialogues if d['prompt'].lower() == prompt_text.lower()]
            if matched:
                self.current_dialogue = matched[0]
            else:
                self.current_dialogue = random.choice(self.dialogues)
        else:
            self.current_dialogue = random.choice(self.dialogues)
            self.current_prompt = self.current_dialogue['prompt']

        target_ans = self.current_dialogue['answer']
        if target_ans in self.answer_to_idx:
            self.target_answer_idx = self.answer_to_idx[target_ans]
        else:
            self.target_answer_idx = random.randint(0, self.action_count - 1)

        return encode_text_to_vector(self.current_prompt)

    def step(self, action_idx: int):
        """Выполнение шага диалога: проверка ответа ИИ"""
        self.step_count += 1
        done = True  # Диалоговый раунд одношаговый на ответ

        is_correct = (action_idx == self.target_answer_idx)
        chosen_answer = self.answers[action_idx] if 0 <= action_idx < self.action_count else "???"
        correct_answer = self.answers[self.target_answer_idx]

        if is_correct:
            reward = 1.0
            event = 'correct'
        else:
            reward = -0.5
            event = 'wrong'

        info = {
            'prompt': self.current_prompt,
            'chosen_answer': chosen_answer,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'event': event
        }

        next_obs = encode_text_to_vector(chosen_answer)
        return next_obs, reward, done, info

    def get_answer_text(self, action_idx: int) -> str:
        """Получить строковый текст ответа по индексу действия"""
        if 0 <= action_idx < len(self.answers):
            return self.answers[action_idx]
        return "Неизвестный ответ"
