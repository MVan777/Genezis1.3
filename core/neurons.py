"""
Разные типы нейронов для двухуровневой памяти
Наследуют базовый Neuron из neuron.py
"""

import numpy as np
import time
from core.neuron import Neuron
from config import NEUTRAL_THRESHOLD

class ShortTermNeuron(Neuron):
    """
    Краткосрочный нейрон - живёт ограниченное время
    Используется для недавнего опыта
    """

    def __init__(self, situation, action, result_flag, lifetime=1000, timestamp=None):
        super().__init__(situation, action, result_flag, timestamp)
        self.lifetime = lifetime  # сколько шагов живет
        self.importance = 0.0      # важность (0-1)
        self.last_accessed = time.time()
        self.access_count = 0

    def access(self):
        """Отметить, что нейрон использовался"""
        self.access_count += 1
        self.last_accessed = time.time()
        # Важность растёт с использованием
        self.importance = min(1.0, self.access_count / 10)

    def should_remove(self):
        """Проверка, нужно ли удалить нейрон"""
        # По флагу
        if abs(self.flag) < NEUTRAL_THRESHOLD:
            return True

        # По времени жизни
        if time.time() - self.created_at > self.lifetime:
            return True

        # По силе
        if self.strength < 0.1:
            return True

        return False

    def get_consolidation_score(self):
        """Оценка для переноса в долговременную память"""
        score = (
            self.importance * 0.4 +
            abs(self.flag) * 0.3 +
            (self.strength / 10) * 0.2 +
            (self.usage_count / 20) * 0.1
        )
        return min(1.0, score)


class LongTermNeuron(Neuron):
    """
    Долгосрочный нейрон - живёт практически вечно
    Хранит важный, проверенный опыт
    """

    def similarity(self, other_neuron):
        """Насколько похожи два краткосрочных нейрона"""
        if self.action != other_neuron.action:
            return 0

        # Косинусная близость ситуаций
        a = self.situation
        b = other_neuron.situation

        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]

        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)

        if a_norm == 0 or b_norm == 0:
            return 0

        sim = np.dot(a, b) / (a_norm * b_norm)
        return sim

    def __init__(self, situation, action, result_flag, source_id=None, timestamp=None):
        super().__init__(situation, action, result_flag, timestamp)
        self.lifetime = float('inf')  # живёт вечно
        self.source_id = source_id    # ID исходного краткосрочного нейрона
        self.consolidated_at = time.time()
        self.confidence = 0.0          # уверенность (растёт со временем)

    def should_remove(self):
        """Долгосрочные нейроны удаляются только при флаге близком к 0"""
        if abs(self.flag) < NEUTRAL_THRESHOLD:
            return True
        # Никогда не удаляются по времени
        return False

    def strengthen(self, new_flag=None, timestamp=None):
        """Усиление с защитой от переобучения"""
        super().strengthen(new_flag, timestamp)
        # Уверенность растёт медленно
        self.confidence = min(1.0, self.confidence + 0.05)