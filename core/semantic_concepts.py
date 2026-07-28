"""
Семантическая Сеть Концептов (Semantic Concept Network)
Категоризирует нейроны и опыт в смысловые категории высшего порядка
"""

import numpy as np
from collections import defaultdict

class SemanticConceptNetwork:
    """Семантическая группировка опыта по смысловым категориям (Защита, Атака, Ресурс)"""

    def __init__(self):
        self.categories = {
            'DEFENSE': [],     # Защитные ситуации
            'OFFENSE': [],     # Агрессивно-атакующие
            'RESOURCE': [],    # Поиск ресурсов
            'EVASION': []      # Уклонения и маневры
        }

    def categorize_neuron(self, neuron):
        """Отнести нейрон к семантической категории"""
        flag = getattr(neuron, 'flag', 0.0)
        action = getattr(neuron, 'action', 0)

        if flag < -0.5:
            self.categories['EVASION'].append(neuron.id)
        elif flag > 0.5 and action in [1, 2]:
            self.categories['OFFENSE'].append(neuron.id)
        elif flag > 0.5:
            self.categories['RESOURCE'].append(neuron.id)
        else:
            self.categories['DEFENSE'].append(neuron.id)

    def get_category_strength(self, category_name):
        """Получить вес семантической категории"""
        return len(self.categories.get(category_name, []))
