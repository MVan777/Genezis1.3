"""
Связи между кластерами памяти
"""

from collections import defaultdict
import time
import numpy as np


class ClusterConnections:
    """Матрица связей между кластерами"""

    def __init__(self):
        self.matrix = defaultdict(float)  # (id1, id2) -> strength
        self.last_updated = defaultdict(float)

    def strengthen(self, cluster_id1, cluster_id2, amount=0.1):
        """Усилить связь между двумя кластерами"""
        if cluster_id1 == cluster_id2:
            return

        key = self._key(cluster_id1, cluster_id2)
        self.matrix[key] += amount
        self.last_updated[key] = time.time()

        if self.matrix[key] > 1.0:
            self.matrix[key] = 1.0

    def weaken(self, cluster_id1, cluster_id2, amount=0.1):
        """Ослабить связь"""
        key = self._key(cluster_id1, cluster_id2)
        if key in self.matrix:
            self.matrix[key] -= amount
            if self.matrix[key] <= 0.01:
                del self.matrix[key]
                del self.last_updated[key]

    def get_strength(self, cluster_id1, cluster_id2):
        """Получить силу связи"""
        key = self._key(cluster_id1, cluster_id2)
        return self.matrix.get(key, 0.0)

    def get_related(self, cluster_id, threshold=0.1):
        """Получить все связанные кластеры"""
        result = []
        for (id1, id2), strength in self.matrix.items():
            if id1 == cluster_id and strength > threshold:
                result.append((id2, strength))
            elif id2 == cluster_id and strength > threshold:
                result.append((id1, strength))

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def decay_all(self, factor=0.95, threshold=0.05):
        """Ослабить все связи"""
        keys = list(self.matrix.keys())
        for key in keys:
            self.matrix[key] *= factor
            if self.matrix[key] < threshold:
                del self.matrix[key]
                if key in self.last_updated:
                    del self.last_updated[key]

    def _key(self, id1, id2):
        """Симметричный ключ"""
        return (min(id1, id2), max(id1, id2))

    def get_stats(self):
        """Статистика связей"""
        if not self.matrix:
            return {'total': 0, 'avg_strength': 0, 'max_strength': 0}

        strengths = list(self.matrix.values())
        return {
            'total': len(self.matrix),
            'avg_strength': np.mean(strengths),
            'max_strength': max(strengths)
        }

    def to_dict(self):
        """Для сохранения"""
        return {
            'matrix': dict(self.matrix),
            'last_updated': dict(self.last_updated)
        }

    def from_dict(self, data):
        """Загрузка из словаря"""
        self.matrix = defaultdict(float, data.get('matrix', {}))
        self.last_updated = defaultdict(float, data.get('last_updated', {}))