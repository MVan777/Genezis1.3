"""
Ядро агента — общее для всех экземпляров
"""

import numpy as np
import time
from core.router import Router
from core.consolidator import MemoryConsolidator

class Brain:
    """
    Мозг — общая память и логика для всех агентов
    """

    def __init__(self, brain_id=0):
        self.id = brain_id
        self.router = Router()
        self.consolidator = None
        self.created = time.time()
        self.last_consolidation = time.time()

        # Статистика мозга
        self.stats = {
            'total_neurons': 0,
            'clusters': 0,
            'simulations_run': 0,
            'lessons_learned': 0
        }

        # Создаём кластеры для разных типов памяти
        self.short_term = self.router.create_cluster(domain="short_term")
        self.long_term = self.router.create_cluster(domain="long_term")
        self.tactical = self.router.create_cluster(domain="tactical")  # для стратегий

    def get_stats(self):
        """Статистика мозга"""
        self.stats['total_neurons'] = sum(len(c.neurons) for c in self.router.clusters)
        self.stats['clusters'] = len(self.router.clusters)
        return self.stats


# Глобальный реестр мозгов (для эволюции)
brains_registry = {}

def create_brain():
    """Создать новый мозг"""
    brain_id = len(brains_registry)
    brain = Brain(brain_id)
    brains_registry[brain_id] = brain
    return brain

def get_brain(brain_id):
    """Получить мозг по ID"""
    return brains_registry.get(brain_id)