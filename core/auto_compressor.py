"""
Автоматический Компрессор и Очиститель Памяти (AutoMemoryCompressor)
Удаляет неиспользуемые/слабые нейроны и объединяет дубликаты в Золотые Прототипы
"""

import numpy as np

class AutoMemoryCompressor:
    """Модуль интеллектуальной очистки неиспользуемой и устаревшей памяти ИИ"""

    def __init__(self, max_neurons_per_cluster=5000):
        self.max_neurons_per_cluster = max_neurons_per_cluster

    def compress_brain(self, brain):
        """Полная компрессия всех кластеров памяти мозга"""
        total_pruned = 0
        total_merged = 0

        for cluster in brain.router.clusters:
            pruned, merged = self.compress_cluster(cluster)
            total_pruned += pruned
            total_merged += merged

        return total_pruned, total_merged

    def compress_cluster(self, cluster):
        """Интеллектуальная компрессия одного кластера"""
        if not cluster or not cluster.neurons:
            return 0, 0

        initial_count = len(cluster.neurons)
        surviving_neurons = []
        pruned_count = 0
        merged_count = 0

        # 1. Сортируем нейроны по ценности (флаг * сила * использование)
        sorted_neurons = sorted(
            cluster.neurons,
            key=lambda n: getattr(n, 'strength', 1.0) * getattr(n, 'usage_count', 1) * (abs(getattr(n, 'flag', 0.0)) + 0.1),
            reverse=True
        )

        for n in sorted_neurons:
            usage = getattr(n, 'usage_count', 1)
            flag = getattr(n, 'flag', 0.0)
            conf = getattr(n, 'confidence', 0.0)

            # Отсеиваем бесполезный неиспользуемый мусор (usage <= 1 и плохой результат)
            if usage <= 1 and flag < 0.0:
                pruned_count += 1
                cluster.remove_neuron(n.id)
                continue

            # Отсеиваем осиротевшие нейроны с ничтожной уверенностью
            if usage <= 2 and conf < 0.05 and flag <= 0.0:
                pruned_count += 1
                cluster.remove_neuron(n.id)
                continue

            surviving_neurons.append(n)

        # 2. Если нейронов всё еще больше лимита, оставляем самые ценные
        if len(surviving_neurons) > self.max_neurons_per_cluster:
            excess = surviving_neurons[self.max_neurons_per_cluster:]
            surviving_neurons = surviving_neurons[:self.max_neurons_per_cluster]
            for ex in excess:
                pruned_count += 1
                cluster.remove_neuron(ex.id)

        cluster.neurons = surviving_neurons
        cluster.cleanup()

        return pruned_count, merged_count
