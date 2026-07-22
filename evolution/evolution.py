"""
Эволюция с сохранением элиты (топ-2 лучших агентов)
"""

import numpy as np
import pickle
import os
from copy import deepcopy
import matplotlib.pyplot as plt
from config import ELITE_SIZE

class Evolution:
    """Управление эпохами с элитарностью"""

    def __init__(self):
        self.generation = 0
        self.best_agent = None
        self.best_score = -float('inf')
        self.history = []
        self.elite_agents = []  # список (агент, счёт)

    def new_epoch(self, agents, scores):
        """
        Начать новую эпоху, выбрать лучших и сохранить элиту
        """
        # Сортируем агентов по счёту
        sorted_pairs = sorted(zip(agents, scores), key=lambda x: x[1], reverse=True)

        # Обновляем элиту
        self.elite_agents = []
        for i in range(min(ELITE_SIZE, len(sorted_pairs))):
            agent_copy = deepcopy(sorted_pairs[i][0])
            agent_score = sorted_pairs[i][1]
            self.elite_agents.append((agent_copy, agent_score))

            # Сохраняем в файл
            self.save_agent(agent_copy, f"best_agent_v{i}.pkl")
            print(f"  💾 Элита #{i+1} сохранена (счёт: {agent_score:.2f})")

        # Лучший агент этой эпохи
        best_agent = sorted_pairs[0][0]
        best_score = sorted_pairs[0][1]

        # Сохраняем в историю
        self.history.append({
            'generation': self.generation,
            'best_score': best_score,
            'avg_score': np.mean(scores),
            'memory_size': len(best_agent.memory.neurons),
            'elite_scores': [s for _, s in self.elite_agents]
        })

        # Проверяем, не побит ли абсолютный рекорд
        if best_score > self.best_score:
            self.best_score = best_score
            self.best_agent = deepcopy(best_agent)
            print(f"\n*** НОВЫЙ АБСОЛЮТНЫЙ РЕКОРД! {self.best_score:.2f} ***")
            self.save_agent(self.best_agent, "best_agent_absolute.pkl")

        self.generation += 1
        return best_agent

    def get_elite(self):
        """Вернуть случайного агента из элиты (для разнообразия)"""
        if not self.elite_agents:
            return None
        # Случайный выбор из элиты
        import random
        return random.choice(self.elite_agents)[0]

    def save_agent(self, agent, filename="best_agent.pkl"):
        """Сохранить агента"""
        with open(filename, 'wb') as f:
            pickle.dump(agent, f)

    def load_best(self, filename="best_agent_v0.pkl"):
        """Загрузить лучшего агента (пробует разные версии)"""
        # Пробуем загрузить абсолютного чемпиона
        if os.path.exists("best_agent_absolute.pkl"):
            with open("best_agent_absolute.pkl", 'rb') as f:
                self.best_agent = pickle.load(f)
            print(f"Загружен абсолютный чемпион со счётом {self.best_score}")
            return self.best_agent

        # Пробуем загрузить элиту
        for i in range(ELITE_SIZE):
            filename = f"best_agent_v{i}.pkl"
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    agent = pickle.load(f)
                self.elite_agents.append((agent, 0))  # счёт пока неизвестен
                print(f"Загружен элитный агент v{i}")

        if self.elite_agents:
            return self.elite_agents[0][0]
        return None

    def plot_history(self):
        """Построить график обучения"""
        if not self.history:
            return

        gens = [h['generation'] for h in self.history]
        best = [h['best_score'] for h in self.history]
        avg = [h['avg_score'] for h in self.history]

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(gens, best, 'b-', label='Лучший в эпохе')
        plt.plot(gens, avg, 'r--', label='Средний')
        plt.axhline(y=self.best_score, color='g', linestyle=':', label=f'Рекорд: {self.best_score:.2f}')
        plt.xlabel('Поколение')
        plt.ylabel('Счёт')
        plt.title('Прогресс обучения')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        mem = [h['memory_size'] for h in self.history]
        plt.plot(gens, mem, 'g-')
        plt.xlabel('Поколение')
        plt.ylabel('Нейронов в памяти')
        plt.title('Рост памяти')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        if 'elite_scores' in self.history[0]:
            elite_data = [h['elite_scores'] for h in self.history]
            for i in range(len(elite_data[0])):
                elite_vals = [e[i] if i < len(e) else None for e in elite_data]
                plt.plot(gens, elite_vals, '--', label=f'Элита #{i+1}')
        plt.xlabel('Поколение')
        plt.ylabel('Счёт элиты')
        plt.title('Качество элиты')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('learning_curve.png')
        plt.show()