"""
Стратегические Цепочки Связей (Strategic Sequence Chaining)
Объединяет последовательности временных нейронов в 3-5 шаговые макро-комбо манёвры
"""

import numpy as np
from collections import defaultdict

class StrategicSequenceChainer:
    """Модуль выявления и исполнения 3-5 шаговых последовательностей действий"""

    def __init__(self, max_chain_length=5):
        self.max_chain_length = max_chain_length
        self.active_sequences = defaultdict(float)  # chain_tuple -> success_weight
        self.current_history = []

    def record_step(self, neuron_id, action, result_flag):
        """Записать шаг последовательности"""
        self.current_history.append((neuron_id, action, result_flag))
        if len(self.current_history) > self.max_chain_length:
            self.current_history.pop(0)

        # Если последовательность завершилась высоким успехом (flag > 0.5), запоминаем цепочку
        if result_flag > 0.5 and len(self.current_history) >= 3:
            actions_chain = tuple(item[1] for item in self.current_history)
            self.active_sequences[actions_chain] += 0.2

    def get_best_combo(self, current_action):
        """Найти наилучшую продолженную цепочку действий"""
        best_chain = None
        best_score = 0.0

        for chain, score in self.active_sequences.items():
            if chain and chain[0] == current_action and score > best_score:
                best_score = score
                best_chain = chain

        return best_chain, best_score
