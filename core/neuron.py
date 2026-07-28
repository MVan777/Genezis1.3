"""
ЯДРО: Класс Neuron
Добавлены: timestamp, lifetime, методы similarity и merge для сжатия
"""

import numpy as np
import random
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import NEURON_VIS_WIDTH, NEURON_VIS_HEIGHT, NEUTRAL_THRESHOLD, NEURON_LIFETIME

class Neuron:
    """Один нейрон = один паттерн (ситуация + действие) с флагом результата"""

    _next_id = 0
    _positions = {}  # id -> (x, y)

    def __init__(self, situation, action, result_flag, timestamp=None):
        self.id = Neuron._next_id
        Neuron._next_id += 1
        self.situation = np.array(situation, dtype=np.float32)
        self.action = action
        self.flag = result_flag
        self.strength = 1.0
        self.usage_count = 1
        self.last_used = timestamp if timestamp else time.time()
        self.created_at = timestamp if timestamp else time.time()
        self.age = 0
        self.lifetime = NEURON_LIFETIME  # сколько живет
        self.next_associations = {}  # id_следующего_нейрона -> вес связи (последовательные ассоциации)

        # Позиция для визуализации
        Neuron._positions[self.id] = (
            random.randint(50, NEURON_VIS_WIDTH - 50),
            random.randint(50, NEURON_VIS_HEIGHT - 50)
        )

    def add_next_association(self, next_neuron_id, amount=0.1):
        """Добавить/усилить последовательную ассоциативную связь с нейроном следующего шага"""
        if next_neuron_id == self.id:
            return
        curr = self.next_associations.get(next_neuron_id, 0.0)
        self.next_associations[next_neuron_id] = min(1.0, curr + amount)

    def strengthen(self, new_flag=None, timestamp=None):
        """Усиление при повторении"""
        self.strength += 0.2
        self.usage_count += 1
        self.last_used = timestamp if timestamp else time.time()
        if new_flag is not None:
            self.flag = 0.8 * self.flag + 0.2 * new_flag
            self.flag = max(-1.0, min(1.0, self.flag))

    def weaken(self):
        """Ослабление при редком использовании"""
        self.strength *= 0.95

    def should_remove(self):
        """Проверка, нужно ли удалить нейрон"""
        # По флагу
        if abs(self.flag) < NEUTRAL_THRESHOLD:
            return True
        # По силе
        if self.strength < 0.1:
            return True
        # По времени жизни
        if time.time() - self.created_at > self.lifetime:
            return True
        return False

    # ===== НОВЫЕ МЕТОДЫ ДЛЯ СЖАТИЯ =====
    def similarity(self, other_neuron):
        """Насколько похожи два нейрона (для сжатия)"""
        if self.action != other_neuron.action:
            return 0

        # Косинусная близость ситуаций
        a = self.situation
        b = other_neuron.situation

        # Если разная длина - обрезаем
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]

        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)

        if a_norm == 0 or b_norm == 0:
            return 0

        sim = np.dot(a, b) / (a_norm * b_norm)
        return sim

    def merge(self, other_neuron):
        """Объединить два нейрона (для сжатия)"""
        # Усредняем флаг с учетом силы
        total_strength = self.strength + other_neuron.strength
        self.flag = (self.flag * self.strength + other_neuron.flag * other_neuron.strength) / total_strength
        self.strength = total_strength / 2  # средняя сила
        self.usage_count += other_neuron.usage_count
        self.last_used = max(self.last_used, other_neuron.last_used)
    # ====================================

    def get_color(self):
        """Цвет нейрона для визуализации"""
        try:
            if self.flag > NEUTRAL_THRESHOLD:
                intensity = min(105, int(100 * self.flag))
                r, g, b = 50, 150 + intensity, 50
            elif self.flag < -NEUTRAL_THRESHOLD:
                intensity = min(105, int(100 * abs(self.flag)))
                r, g, b = 150 + intensity, 50, 50
            else:
                r, g, b = 100, 100, 100

            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))

            return (r, g, b)
        except:
            return (100, 100, 100)