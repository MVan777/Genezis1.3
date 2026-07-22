"""
Тонкий агент — только интерфейс, мозг общий
"""

import random
import numpy as np
from collections import defaultdict
from config import ACTION_COUNT


class AgentInstance:
    """
    Экземпляр агента — использует общий мозг
    """

    def __init__(self, brain, instance_id=0):
        self.brain = brain
        self.id = instance_id
        self.name = f"Agent_{brain.id}_{instance_id}"

        # Временное состояние (только для этого экземпляра)
        self.last_state = None
        self.last_action = None
        self.last_similar = []
        self.game_history = []
        self.fitness = 0.0

        # Стратегия (может отличаться у разных экземпляров)
        self.strategy_weights = {
            'aggressive': random.uniform(0, 1),
            'defensive': random.uniform(0, 1),
            'explorative': random.uniform(0, 1),
            'greedy': random.uniform(0, 1)
        }

    def act(self, state, explore=True):
        """Выбрать действие на основе состояния"""
        # Выбираем кластер через роутер мозга
        cluster = self.brain.router.select_cluster(state)

        # Ищем похожие ситуации во всех кластерах мозга
        similar = self._find_similar(state)
        self.last_similar = similar

        # Голосование с учётом стратегии
        action, confidence = self._vote_with_strategy(similar)

        if action is None or confidence < 0.2:
            action = random.randint(0, ACTION_COUNT - 1) if explore else 4

        self.last_state = state
        self.last_action = action

        # Запоминаем для анализа
        self.game_history.append({
            'state': state,
            'action': action,
            'confidence': confidence,
            'strategy': self.strategy_weights.copy()
        })

        return action

    def _find_similar(self, state, threshold=0.4):
        """Искать похожее во всех кластерах мозга"""
        results = []
        for cluster in self.brain.router.clusters:
            similar = cluster.find_similar(state, threshold=threshold)
            for idx, sim, neuron in similar:
                results.append((neuron, sim, 1.0))
        return results

    def _vote_with_strategy(self, similar):
        """Голосование с учётом стратегии"""
        if not similar:
            return None, 0

        votes = defaultdict(float)

        for neuron, sim, weight in similar:
            # Базовая оценка
            base = sim * neuron.flag * neuron.strength

            # Модификаторы стратегии
            if neuron.action in [0, 1, 2, 3]:  # движение
                base *= self.strategy_weights['explorative']
            elif neuron.action == 5:  # атака
                base *= self.strategy_weights['aggressive']

            votes[neuron.action] += base

        best = max(votes.items(), key=lambda x: x[1])
        return best[0], abs(best[1])

    def learn(self, reward, new_state, done):
        """Обучение (запись в мозг)"""
        if self.last_state is None or self.last_action is None:
            return

        # Создаём нейрон в краткосрочной памяти мозга
        from core.neurons import ShortTermNeuron
        neuron = ShortTermNeuron(self.last_state, self.last_action, reward)
        self.brain.short_term.add_neuron(neuron)

        # Обновляем fitness
        self.fitness += reward

        if done:
            # Анализ смерти
            self._analyze_death()

    def _analyze_death(self):
        """Анализ смерти с учётом стратегии"""
        if len(self.game_history) < 5:
            return

        # Смотрим последние шаги
        last_steps = self.game_history[-5:]

        # Оцениваем, была ли стратегия удачной
        total_reward = sum(s.get('reward', 0) for s in last_steps)

        if total_reward < -10:
            # Плохая стратегия — меняем веса
            for key in self.strategy_weights:
                self.strategy_weights[key] *= random.uniform(0.8, 1.2)
            print(f"  {self.name} меняет стратегию: {self.strategy_weights}")

    def get_state(self):
        """Состояние экземпляра для сохранения"""
        return {
            'id': self.id,
            'name': self.name,
            'fitness': self.fitness,
            'strategy': self.strategy_weights,
            'brain_id': self.brain.id
        }