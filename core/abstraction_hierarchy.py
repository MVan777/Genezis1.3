"""
3-Уровневая Иерархия Абстракции Памяти (Abstraction Hierarchy)
Группирует нейроны: Сырые ситуации -> Тактические маневры -> Стратегические Концепты
"""

import time
import numpy as np

class AbstractionHierarchy:
    """Управление 3-уровневой абстракцией памяти в ассоциативном графе"""

    def __init__(self):
        self.raw_situations = []      # Уровень 1: Сырой опыт
        self.tactical_maneuvers = []  # Уровень 2: Тактические шаблоны
        self.strategic_concepts = [] # Уровень 3: Долгосрочные Золотые Прототипы

    def process_memory(self, cluster):
        """Обработать кластер нейронов и сгруппировать по уровням абстракции"""
        if not cluster or not cluster.neurons:
            return

        level1 = 0
        level2 = 0
        level3 = 0

        for neuron in cluster.neurons:
            usage = getattr(neuron, 'usage_count', 1)
            conf = getattr(neuron, 'confidence', 0.0)

            if conf > 0.6 or usage > 15:
                if neuron not in self.strategic_concepts:
                    self.strategic_concepts.append(neuron)
                level3 += 1
            elif usage > 5:
                if neuron not in self.tactical_maneuvers:
                    self.tactical_maneuvers.append(neuron)
                level2 += 1
            else:
                if neuron not in self.raw_situations:
                    self.raw_situations.append(neuron)
                level1 += 1

        return {'raw': level1, 'tactical': level2, 'strategic': level3}
