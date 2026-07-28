"""
Главный Движок Genezis 3.0 Ultimate (Genezis 3.0 Ultimate Engine)
Флагманский Универсальный Ассоциативный ИИ Высшего Порядка
"""

import time
import numpy as np
from collections import defaultdict

from core.router import Router
from core.neurons import ShortTermNeuron, LongTermNeuron
from core.meta_sensor import MetaSensorNormalizer
from core.goal_system import MacroGoalSystem
from core.spatial_memory import SpatialMentalMap
from core.sequence_chaining import StrategicSequenceChainer
from core.mental_simulator import DeepMentalGraphSimulator
from core.meta_adapter import MetaLearningAdapter
from core.cross_transfer import CrossDomainTransfer
from core.abstraction_hierarchy import AbstractionHierarchy

class Genezis3UltimateEngine:
    """Максимальный Универсальный Ассоциативный Движок ИИ Genezis 3.0"""

    def __init__(self, action_count=5, router=None):
        self.action_count = action_count
        self.router = router if router else Router()
        self.normalizer = MetaSensorNormalizer()
        self.goal_system = MacroGoalSystem()
        self.spatial_map = SpatialMentalMap()

        # Технологии Фазы 4 и Genezis 3.0
        self.chainer = StrategicSequenceChainer()
        self.deep_simulator = DeepMentalGraphSimulator()
        self.meta_adapter = MetaLearningAdapter()
        self.cross_transfer = CrossDomainTransfer()
        self.hierarchy = AbstractionHierarchy()

        self.reward_history = []
        self.active_cluster = None
        self.last_obs = None
        self.last_action = None
        self.last_neuron = None
        self.last_similar_neurons = []

        self.emotions = {'fear': 0.0, 'aggression': 0.0, 'curiosity': 0.0, 'calm': 0.0}
        self.current_goal = 0
        self.history = []
        self.stats = {'steps': 0, 'lessons': 0, 'curiosity_rewards': 0.0}

    def _find_neuron_by_id(self, neuron_id):
        """Поиск нейрона во всех кластерах по ID"""
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
        """Универсальное принятие решений для вектора любой размерности R^D"""
        obs = self.normalizer.normalize(raw_observation)
        self.stats['steps'] += 1

        # Мета-Адаптация параметров на лету
        params = self.meta_adapter.adapt(obs, self.reward_history)
        conf_thresh = params['confidence_threshold']

        # Извлечение лучшего комбо-маневра из стратегических цепочек
        combo_chain, combo_score = self.chainer.get_best_combo(self.last_action)

        # Определение целей и контекста
        self.current_goal = self.goal_system.select_goal(obs)
        self.active_cluster = self.router.select_cluster(obs)

        similar = self._find_similar(obs)
        self.last_similar_neurons = similar

        best_action, confidence = self._vote(similar)

        if combo_chain and combo_score > 0.8:
            action = combo_chain[1] if len(combo_chain) > 1 else combo_chain[0]
        elif best_action is not None and confidence > conf_thresh:
            action = best_action
        elif explore:
            action = int(np.random.randint(0, self.action_count))
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
        """Поиск сходства по всей оперативной памяти"""
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
        """Взвешенное голосование с 5-шаговым глубоким моделированием будущего"""
        if not similar_neurons:
            return None, 0.0

        votes = defaultdict(float)
        future_scores = self.deep_simulator.evaluate_future_branches(similar_neurons, self)

        for neuron, sim, weight in similar_neurons:
            if hasattr(neuron, 'access'):
                neuron.access()

            type_weight = 1.5 if hasattr(neuron, 'confidence') else 1.0
            future_bonus = future_scores.get(neuron.action, 0.0)

            eff_flag = neuron.flag + 0.4 * future_bonus
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
        """Шаг обучения с мета-любопытством и снами"""
        if self.last_obs is None or self.last_action is None or self.active_cluster is None:
            return

        next_obs = self.normalizer.normalize(next_raw_observation)
        self.reward_history.append(reward)

        # Внутреннее любопытство
        surprise = float(np.linalg.norm(next_obs - self.last_obs))
        curiosity_reward = min(0.5, surprise * self.meta_adapter.curiosity_weight)
        self.stats['curiosity_rewards'] += curiosity_reward

        total_reward = reward + curiosity_reward
        flag = self._reward_to_flag(total_reward)

        new_neuron = ShortTermNeuron(self.last_obs, self.last_action, result_flag=flag)
        self.active_cluster.add_neuron(new_neuron)

        # Обучение стратегических цепочек шагов
        self.chainer.record_step(new_neuron.id, self.last_action, flag)

        if self.last_neuron is not None and hasattr(self.last_neuron, 'add_next_association'):
            self.last_neuron.add_next_association(new_neuron.id, amount=0.2)
        self.last_neuron = new_neuron

        # Периодическая организация 3-уровневой иерархии абстракций
        if self.stats['steps'] % 50 == 0:
            self.hierarchy.process_memory(self.active_cluster)

        if done:
            self._decay_on_reset()
            self.analyze_and_simulate()

    def analyze_and_simulate(self):
        """Ретроспективные сны во время ошибок"""
        if not self.history or len(self.history) < 5:
            return 0

        key_moments = [h for h in self.history if h.get('confidence', 1.0) < 0.3]
        lessons = 0

        for moment in key_moments[:5]:
            obs_m = moment['obs']
            action_m = moment['action']
            for alt_action in range(self.action_count):
                if alt_action == action_m:
                    continue
                sim_neuron = ShortTermNeuron(obs_m, alt_action, result_flag=0.3)
                if self.active_cluster:
                    self.active_cluster.add_neuron(sim_neuron)
                    lessons += 1

        self.stats['lessons'] += lessons
        return lessons

    def _decay_on_reset(self):
        """Ослабление при сбросе"""
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
        """Сброс состояния"""
        self.last_obs = None
        self.last_action = None
        self.last_neuron = None
        self.history = []
