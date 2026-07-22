"""
Банк элиты - хранит лучших агентов и их нейроны между запусками
"""

import pickle
import os
from copy import deepcopy
import time

class EliteBank:
    """Хранилище лучших агентов за всё время"""

    def __init__(self, filename="elite_bank.pkl", max_size=5):
        self.filename = filename
        self.max_size = max_size
        self.agents = []  # список (агент, счёт, эпоха, время)
        self.load()

    def add_agent(self, agent, score, epoch):
        """Добавить агента в банк"""
        # Создаём копию, чтобы не изменился потом
        agent_copy = deepcopy(agent)

        # Добавляем с метаданными
        self.agents.append({
            'agent': agent_copy,
            'score': score,
            'epoch': epoch,
            'time': time.time(),
            'neurons': len(agent_copy.memory.neurons) if hasattr(agent_copy, 'memory') else 0
        })

        # Сортируем по счёту
        self.agents.sort(key=lambda x: x['score'], reverse=True)

        # Оставляем только лучших (например, 5)
        self.agents = self.agents[:self.max_size]

        # Сохраняем
        self.save()

        print(f"  💾 Агент со счётом {score:.2f} добавлен в банк элиты")

    def get_top_agents(self, n=3):
        """Вернуть n лучших агентов"""
        return [a['agent'] for a in self.agents[:n]]

    def get_top_scores(self):
        """Вернуть список лучших счетов"""
        return [a['score'] for a in self.agents]

    def get_all_neurons_count(self):
        """Суммарное количество нейронов в банке"""
        return sum(a['neurons'] for a in self.agents)

    def save(self):
        """Сохранить банк в файл"""
        with open(self.filename, 'wb') as f:
            pickle.dump(self.agents, f)
        print(f"  💿 Банк элиты сохранён ({len(self.agents)} агентов)")

    def load(self):
        """Загрузить банк из файла"""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'rb') as f:
                    self.agents = pickle.load(f)
                print(f"  📀 Банк элиты загружен ({len(self.agents)} агентов, рекорд: {self.get_top_scores()[0] if self.agents else 0:.2f})")
            except:
                self.agents = []
        else:
            self.agents = []

    def merge_memories(self, agent):
        """Объединить память агента с банком (сложная операция)"""
        # TODO: объединение нейронов из разных агентов
        # Это будущее - скрещивание стратегий
        pass