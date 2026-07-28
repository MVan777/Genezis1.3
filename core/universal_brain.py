"""
Универсальный Движок Ассоциативного ИИ (Universal Associative Brain)
Полностью абстрагирован от конкретной среды (Environment Agnostic)
Подходит для Игр, Робототехники, Трейдинга, Telegram-ботов
"""

import random
import time
import numpy as np
from collections import defaultdict

from core.router import Router
from core.neurons import ShortTermNeuron, LongTermNeuron
from core.meta_sensor import MetaSensorNormalizer
from core.goal_system import MacroGoalSystem
from core.spatial_memory import SpatialMentalMap

class UniversalAssociativeBrain:
    """Универсальный Ассоциативный Мозг ИИ с единым стандартом act() / learn()"""

    def __init__(self, action_count=5, router=None):
        self.action_count = action_count
        self.router = router if router else Router()
        self.normalizer = MetaSensorNormalizer()
        self.goal_system = MacroGoalSystem()
        self.spatial_map = SpatialMentalMap()

        self.active_cluster = None
        self.last_obs = None
        self.last_action = None
        self.last_neuron = None

        self.emotions = {'fear': 0.0, 'aggression': 0.0, 'curiosity': 0.0, 'calm': 0.0}
        self.current_goal = 0
        self.history = []
        self.stats = {'steps': 0, 'lessons': 0, 'curiosity_rewards': 0.0}

    def _find_neuron_by_id(self, neuron_id):
        """Быстрый поиск нейрона по ID для предвосхищения"""
        if self.active_cluster:
            for n in self.active_cluster.neurons:
                if n.id == neuron_id:
                    return n
        for cluster in self.router.clusters:
            for n in cluster.neurons:
                if n.id == neuron_id:
                    return n
        return None

    def act(self, raw_observation, explore=True):
        """
        Универсальный метод выбора действия на основе любого вектора R^D
        """
        obs = self.normalizer.normalize(raw_observation)
        self.stats['steps'] += 1

        # Обновляем макро-цель и эмоциональное состояние
        self.current_goal = self.goal_system.select_goal(obs)

        # Выбор контекстного кластера памяти
        self.active_cluster = self.router.select_cluster(obs)

        # Поиск похожих нейронов во всех кластерах
        similar = self._find_similar(obs)
        best_action, confidence = self._vote(similar)

        if best_action is not None and confidence > 0.1:
            action = best_action
        elif explore:
            action = random.randint(0, self.action_count - 1)
        else:
            action = 0

        self.history.append({
            'obs': obs,
            'action': action,
            'confidence': confidence,
            'timestamp': time.time()
        })

        self.last_obs = obs
        self.last_action = action
        return action

    def _find_similar(self, obs, threshold=0.4):
        """Найти похожие состояния во всех кластерах"""
        results = []
        if self.active_cluster:
            sims = self.active_cluster.find_similar(obs, threshold=threshold)
            for idx, sim, n in sims:
                results.append((n, sim, 1.0))
        for cluster in self.router.clusters:
            if cluster != self.active_cluster:
                sims = cluster.find_similar(obs, threshold=threshold * 0.8)
                for idx, sim, n in sims:
                    results.append((n, sim, 0.5))
        return results

    def _vote(self, similar_neurons):
        """Взвешенное голосование с многошаговым предвосхищением (2-step Lookahead)"""
        if not similar_neurons:
            return None, 0.0

        votes = defaultdict(float)

        for neuron, sim, weight in similar_neurons:
            if hasattr(neuron, 'access'):
                neuron.access()

            type_weight = 1.5 if hasattr(neuron, 'confidence') else 1.0

            # 2-step Spreading Activation Lookahead
            lookahead_bonus = 0.0
            if hasattr(neuron, 'next_associations') and neuron.next_associations:
                for next_id, w_next in neuron.next_associations.items():
                    next_n = self._find_neuron_by_id(next_id)
                    if next_n:
                        lookahead_bonus += 0.3 * w_next * next_n.flag * next_n.strength

            eff_flag = neuron.flag + lookahead_bonus
            votes[neuron.action] += sim * weight * eff_flag * neuron.strength * type_weight

        if not votes:
            return None, 0.0

        best = max(votes.items(), key=lambda x: x[1])
        if len(votes) > 1:
            second = sorted(votes.values(), reverse=True)[1]
            conf = best[1] - second
        else:
            conf = abs(best[1])

        return best[0], conf

    def learn(self, reward, next_raw_observation, done):
        """
        Универсальное обучение по результату шага
        """
        if self.last_obs is None or self.last_action is None or self.active_cluster is None:
            return

        next_obs = self.normalizer.normalize(next_raw_observation)

        # Внутреннее любопытство (Intrinsic Surprise Reward)
        surprise = float(np.linalg.norm(next_obs - self.last_obs))
        curiosity_reward = min(0.5, surprise * 0.1)
        self.stats['curiosity_rewards'] += curiosity_reward

        total_reward = reward + curiosity_reward
        flag = self._reward_to_flag(total_reward)

        # Создаем новый краткосрочный нейрон
        new_neuron = ShortTermNeuron(self.last_obs, self.last_action, flag)
        self.active_cluster.add_neuron(new_neuron)

        # Последовательная временная связь N_{t-1} -> N_t
        if self.last_neuron is not None and hasattr(self.last_neuron, 'add_next_association'):
            self.last_neuron.add_next_association(new_neuron.id, amount=0.2)
        self.last_neuron = new_neuron

        if done:
            self._decay_on_reset()

    def _decay_on_reset(self):
        """Ослабление и 5-шаговое распределение ошибок при завершении эпизода"""
        if self.history and len(self.history) > 1:
            steps_back = min(5, len(self.history))
            discount = 1.0
            for k in range(1, steps_back + 1):
                moment = self.history[-k]
                obs_k = moment.get('obs')
                if obs_k is not None and self.active_cluster:
                    similar = self.active_cluster.find_similar(obs_k, threshold=0.6, max_results=3)
                    for idx, sim, n in similar:
                        if n.action == moment.get('action'):
                            n.flag = max(-1.0, n.flag - 0.2 * discount)
                            n.strength *= (1.0 - 0.1 * discount)
                discount *= 0.7

    def _reward_to_flag(self, reward):
        if reward > 5.0:
            return 1.0
        elif reward > 1.0:
            return 0.5
        elif reward > 0.1:
            return 0.2
        elif reward < -1.0:
            return -1.0
        elif reward < -0.2:
            return -0.5
        elif abs(reward) < 0.1:
            return 0.0
        else:
            return float(reward / 5.0)

    def reset_episode(self):
        """Сбросить локальные состояния для нового эпизода"""
        self.last_obs = None
        self.last_action = None
        self.last_neuron = None
        self.history = []
