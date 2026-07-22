"""
Эволюция через дуэли двух агентов
"""

import random
import numpy as np
from copy import deepcopy
from agent.agent_instance import AgentInstance
from core.brain import create_brain, get_brain
from elite_bank import EliteBank

class DuelEvolution:
    """
    Эволюция, где агенты сражаются друг с другом
    """

    def __init__(self, population_size=10):
        self.population_size = population_size
        self.population = []
        self.bank = EliteBank(max_size=10)
        self.generation = 0

        # Статистика
        self.stats = {
            'generations': 0,
            'total_duels': 0,
            'avg_fitness': 0,
            'best_fitness': 0
        }

    def create_population(self):
        """Создать начальную популяцию"""
        self.population = []

        for i in range(self.population_size):
            # Каждый агент получает свой мозг
            brain = create_brain()
            agent = AgentInstance(brain, instance_id=i)
            self.population.append(agent)

        print(f"Создана популяция из {self.population_size} агентов")

    def run_tournament(self, games_per_pair=3):
        """
        Турнир — каждый с каждым
        """
        scores = {agent.name: 0 for agent in self.population}

        # Каждый с каждым
        for i in range(len(self.population)):
            for j in range(i+1, len(self.population)):
                agent1 = self.population[i]
                agent2 = self.population[j]

                print(f"\n⚔️ Дуэль: {agent1.name} vs {agent2.name}")

                for game in range(games_per_pair):
                    # Создаём арену
                    arena = DuelArena(agent1, agent2)
                    arena.reset()

                    # Играем до конца
                    while not arena.finished:
                        winner = arena.step()

                    # Начисляем очки
                    if winner == agent1:
                        scores[agent1.name] += 3
                        print(f"  Игра {game+1}: победил {agent1.name}")
                    elif winner == agent2:
                        scores[agent2.name] += 3
                        print(f"  Игра {game+1}: победил {agent2.name}")
                    else:
                        scores[agent1.name] += 1
                        scores[agent2.name] += 1
                        print(f"  Игра {game+1}: ничья")

                    self.stats['total_duels'] += 1

        return scores

    def evolve(self, generations=10):
        """
        Основной цикл эволюции
        """
        if not self.population:
            self.create_population()

        for gen in range(generations):
            print(f"\n{'='*50}")
            print(f"ПОКОЛЕНИЕ {self.generation + 1}")
            print(f"{'='*50}")

            # Турнир
            scores = self.run_tournament()

            # Сортируем по очкам
            sorted_agents = sorted(
                self.population,
                key=lambda a: scores[a.name],
                reverse=True
            )

            # Лучшие идут в банк
            for agent in sorted_agents[:3]:
                self.bank.add_agent(agent, scores[agent.name], self.generation)

            # Статистика
            best_score = scores[sorted_agents[0].name]
            avg_score = np.mean(list(scores.values()))

            print(f"\n📊 Итоги поколения:")
            print(f"  Лучший: {sorted_agents[0].name} - {best_score} очков")
            print(f"  Средний: {avg_score:.2f}")

            self.stats['generations'] += 1
            self.stats['best_fitness'] = max(self.stats['best_fitness'], best_score)
            self.stats['avg_fitness'] = avg_score

            # Создаём следующее поколение
            self._next_generation(sorted_agents)

            self.generation += 1

    def _next_generation(self, sorted_agents):
        """
        Создать следующее поколение
        """
        new_population = []

        # Элита (лучшие 20%)
        elite_count = max(2, self.population_size // 5)
        for i in range(elite_count):
            # Копируем агента с тем же мозгом
            agent = sorted_agents[i]
            new_agent = AgentInstance(agent.brain, instance_id=i)
            new_population.append(new_agent)

        # Остальные — новые с мутациями
        while len(new_population) < self.population_size:
            # Берём случайного родителя из лучших
            parent = random.choice(sorted_agents[:elite_count*2])

            # Создаём новый мозг (мутация)
            from core.brain import create_brain
            new_brain = create_brain()

            # Копируем стратегию с мутацией
            new_agent = AgentInstance(new_brain, instance_id=len(new_population))
            new_agent.strategy_weights = parent.strategy_weights.copy()

            # Мутация
            for key in new_agent.strategy_weights:
                if random.random() < 0.3:
                    new_agent.strategy_weights[key] += random.uniform(-0.2, 0.2)
                    new_agent.strategy_weights[key] = max(0, min(1, new_agent.strategy_weights[key]))

            new_population.append(new_agent)

        self.population = new_population