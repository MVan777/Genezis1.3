"""
Консолидация памяти - перенос важного из краткосрочной в долгосрочную
"""

import time
import numpy as np
from collections import defaultdict
from core.neurons import ShortTermNeuron, LongTermNeuron

class MemoryConsolidator:
    """
    Управляет переносом нейронов между краткосрочной и долгосрочной памятью
    """

    def __init__(self, short_term_cluster, long_term_cluster):
        self.short_term = short_term_cluster
        self.long_term = long_term_cluster
        self.last_consolidation = time.time()
        self.consolidation_interval = 300  # секунд между консолидациями

        # Статистика
        self.stats = {
            'total_consolidated': 0,
            'total_removed': 0,
            'avg_importance': 0.0
        }

    def consolidate(self, force=False):
        """
        Запустить консолидацию памяти
        Переносит важные краткосрочные нейроны в долгосрочные
        """
        now = time.time()
        if not force and now - self.last_consolidation < self.consolidation_interval:
            return

        print(f"\n🧹 КОНСОЛИДАЦИЯ ПАМЯТИ")
        print(f"  Краткосрочных: {len(self.short_term.neurons)}")
        print(f"  Долгосрочных: {len(self.long_term.neurons)}")

        consolidated = 0
        removed = 0
        importance_sum = 0

        # Анализируем каждый краткосрочный нейрон
        for neuron in list(self.short_term.neurons):
            if not isinstance(neuron, ShortTermNeuron):
                continue

            score = neuron.get_consolidation_score()
            importance_sum += score

            # Если нейрон важен - переносим в долгосрочный
            if score > 0.7:
                self._consolidate_neuron(neuron)
                consolidated += 1

            # Если нейрон устарел - удаляем
            elif neuron.should_remove():
                self.short_term.remove_neuron(neuron.id)
                removed += 1

        # Обновляем статистику
        self.stats['total_consolidated'] += consolidated
        self.stats['total_removed'] += removed
        if len(self.short_term.neurons) > 0:
            self.stats['avg_importance'] = importance_sum / len(self.short_term.neurons)

        print(f"  ✅ Перенесено в долгосрочную: {consolidated}")
        print(f"  🗑️ Удалено устаревших: {removed}")
        print(f"  📊 Осталось краткосрочных: {len(self.short_term.neurons)}")

        self.last_consolidation = now

        # Возвращаем количество перенесённых нейронов
        return consolidated

    def _consolidate_neuron(self, short_term_neuron):
        """Перенести один нейрон в долгосрочную память (или объединить в концептуальный нейрон-прототип)"""
        similar = self.long_term.find_similar(short_term_neuron.situation, threshold=0.7, max_results=5)
        matched_lt = None
        for idx, sim, lt_neuron in similar:
            if lt_neuron.action == short_term_neuron.action:
                matched_lt = lt_neuron
                break

        if matched_lt:
            # Объединяем в обобщенный концептуальный нейрон-прототип
            if hasattr(matched_lt, 'merge'):
                matched_lt.merge(short_term_neuron)
            else:
                matched_lt.strength += 0.2
                matched_lt.usage_count += short_term_neuron.usage_count
                matched_lt.flag = 0.8 * matched_lt.flag + 0.2 * short_term_neuron.flag
            if hasattr(matched_lt, 'confidence'):
                matched_lt.confidence = min(1.0, matched_lt.confidence + 0.1)

            if hasattr(short_term_neuron, 'next_associations'):
                for next_id, w in short_term_neuron.next_associations.items():
                    matched_lt.add_next_association(next_id, w)
        else:
            # Создаём долгосрочный нейрон-копию
            long_term = LongTermNeuron(
                short_term_neuron.situation,
                short_term_neuron.action,
                short_term_neuron.flag,
                source_id=short_term_neuron.id
            )

            # Копируем важные атрибуты
            long_term.strength = short_term_neuron.strength
            long_term.usage_count = short_term_neuron.usage_count
            long_term.confidence = min(1.0, getattr(short_term_neuron, 'importance', 0.5))

            if hasattr(short_term_neuron, 'next_associations'):
                long_term.next_associations = dict(short_term_neuron.next_associations)

            # Добавляем в долгосрочный кластер
            self.long_term.add_neuron(long_term)

        # Удаляем из краткосрочного
        self.short_term.remove_neuron(short_term_neuron.id)

    def auto_cleanup(self, max_short_term=1000):
        """
        Автоматическая очистка при переполнении
        """
        if len(self.short_term.neurons) > max_short_term:
            print(f"  ⚠️ Краткосрочная память переполнена ({len(self.short_term.neurons)})")
            # Сортируем по важности
            self.short_term.neurons.sort(
                key=lambda n: n.get_consolidation_score() if hasattr(n, 'get_consolidation_score') else 0,
                reverse=True
            )
            # Оставляем только самые важные
            self.short_term.neurons = self.short_term.neurons[:max_short_term]
            print(f"  ✂️ Сокращено до {max_short_term}")