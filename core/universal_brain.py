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
from core.sequence_chaining import StrategicSequenceChainer
from core.mental_simulator import DeepMentalGraphSimulator
from core.meta_adapter import MetaLearningAdapter
from core.cross_transfer import CrossDomainTransfer
from core.system_dual_process import SystemDualProcessEngine
from core.intuitive_salience import IntuitiveSalienceEngine
from core.semantic_concepts import SemanticConceptNetwork

class UniversalAssociativeBrain:
    """Универсальный Ассоциативный Мозг ИИ (Genezis 2.0) с единым стандартом act() / learn()"""

    def __init__(self, action_count=5, router=None):
        self.action_count = action_count
        self.router = router if router else Router()
        self.normalizer = MetaSensorNormalizer()
        self.goal_system = MacroGoalSystem()
        self.spatial_map = SpatialMentalMap()

        # Фаза 4: Мета-Ассоциативные Модули Высшего Порядка
        self.chainer = StrategicSequenceChainer()
        self.deep_simulator = DeepMentalGraphSimulator()
        self.meta_adapter = MetaLearningAdapter()
        self.cross_transfer = CrossDomainTransfer()

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

    def _rebuild_neuron_cache(self):
        """Быстрая пересборка индекса ID -> Нейрон для O(1) поиска"""
        self._neuron_cache = {}
        for cluster in self.router.clusters:
            for n in cluster.neurons:
                self._neuron_cache[n.id] = n

    def _find_neuron_by_id(self, neuron_id):
        """Быстрый O(1) поиск нейрона по ID для предвосхищения"""
        if not hasattr(self, '_neuron_cache') or self._neuron_cache is None:
            self._rebuild_neuron_cache()
            
        neuron = self._neuron_cache.get(neuron_id)
        if neuron is not None:
            return neuron
            
        # Резервный поиск, если нейрон был добавлен в кластер в обход кэша
        if self.active_cluster:
            for n in self.active_cluster.neurons:
                if n.id == neuron_id:
                    self._neuron_cache[n.id] = n
                    return n
        for cluster in self.router.clusters:
            for n in cluster.neurons:
                if n.id == neuron_id:
                    self._neuron_cache[n.id] = n
                    return n
        return None

    def _ensure_attributes(self):
        """Гарантирует инициализацию всех атрибутов для ранее сохраненных из pickle объектов"""
        if not hasattr(self, 'chainer'):
            self.chainer = StrategicSequenceChainer()
        if not hasattr(self, 'deep_simulator'):
            self.deep_simulator = DeepMentalGraphSimulator()
        if not hasattr(self, 'meta_adapter'):
            self.meta_adapter = MetaLearningAdapter()
        if not hasattr(self, 'cross_transfer'):
            self.cross_transfer = CrossDomainTransfer()
        if not hasattr(self, 'dual_process'):
            self.dual_process = SystemDualProcessEngine()
        if not hasattr(self, 'salience_engine'):
            self.salience_engine = IntuitiveSalienceEngine()
        if not hasattr(self, 'semantic_concepts'):
            self.semantic_concepts = SemanticConceptNetwork()
        if not hasattr(self, 'reward_history'):
            self.reward_history = []
        if not hasattr(self, 'last_similar_neurons'):
            self.last_similar_neurons = []

    def act(self, raw_observation, explore=True):
        """
        Универсальный метод выбора действия на основе любого вектора R^D
        """
        self._ensure_attributes()
        obs = self.normalizer.normalize(raw_observation)
        self.stats['steps'] += 1

        # Обновляем макро-цель и эмоциональное состояние
        self.current_goal = self.goal_system.select_goal(obs)

        # Выбор контекстного кластера памяти
        self.active_cluster = self.router.select_cluster(obs)

        # Поиск похожих нейронов во всех кластерах
        similar = self._find_similar(obs)
        self.last_similar_neurons = similar
        best_action, confidence = self._vote(similar)

        if best_action is not None and (confidence > 0.05 or not explore):
            action = best_action
        elif explore:
            action = random.randint(0, self.action_count - 1)
        elif best_action is not None:
            action = best_action
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

    def predict_action_probabilities(self, raw_observation):
        """
        Рассчитать процентные вероятности ИИ для каждого действия (HOLD, BUY, SELL)
        """
        self._ensure_attributes()
        obs = self.normalizer.normalize(raw_observation)
        self.active_cluster = self.router.select_cluster(obs)
        similar = self._find_similar(obs)

        if not similar:
            return {0: 50.0, 1: 25.0, 2: 25.0}

        votes = defaultdict(float)
        for neuron, sim, weight in similar:
            type_weight = 1.5 if hasattr(neuron, 'confidence') else 1.0
            eff_flag = max(0.1, neuron.flag + 1.0)
            votes[neuron.action] += sim * weight * eff_flag * neuron.strength * type_weight

        total_v = sum(votes.values())
        if total_v <= 0:
            return {0: 50.0, 1: 25.0, 2: 25.0}

        p_hold = (votes.get(0, 0.0) / total_v) * 100.0
        p_buy = (votes.get(1, 0.0) / total_v) * 100.0
        p_sell = (votes.get(2, 0.0) / total_v) * 100.0

        tot_p = p_hold + p_buy + p_sell
        if tot_p > 0:
            p_hold = (p_hold / tot_p) * 100.0
            p_buy = (p_buy / tot_p) * 100.0
            p_sell = (p_sell / tot_p) * 100.0

        return {0: round(p_hold, 1), 1: round(p_buy, 1), 2: round(p_sell, 1)}

    def _vote(self, similar_neurons):
        """Взвешенное голосование с глубоким 5-шаговым моделированием по графу памяти"""
        if not similar_neurons:
            return None, 0.0

        votes = defaultdict(float)

        # Фаза 4: 5-шаговая глубокая волновое моделирование будущего по графу
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
        """
        Универсальное обучение по результату шага
        """
        self._ensure_attributes()
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
        if hasattr(self, '_neuron_cache') and self._neuron_cache is not None:
            self._neuron_cache[new_neuron.id] = new_neuron

        # Последовательная временная связь N_{t-1} -> N_t (только внутри одного эпизода)
        if self.last_neuron is not None and hasattr(self.last_neuron, 'add_next_association'):
            self.last_neuron.add_next_association(new_neuron.id, amount=0.2)
            
        # При сильном отбивании или пропуске проводим временное распределение кредита подкрепления по траектории (25-30 шагов)
        if reward < -2.0:
            steps_back = min(30, len(self.history))
            discount = 1.0
            for k in range(1, steps_back + 1):
                moment = self.history[-k]
                obs_k = moment.get('obs')
                if obs_k is not None and self.active_cluster:
                    similar = self.active_cluster.find_similar(obs_k, threshold=0.5, max_results=3)
                    for idx, sim, n in similar:
                        if n.action == moment.get('action'):
                            n.flag = max(-1.0, n.flag - 0.3 * discount)
                            n.strength *= (1.0 - 0.05 * discount)
                discount *= 0.92
        elif reward > 2.0:
            steps_back = min(25, len(self.history))
            discount = 1.0
            for k in range(1, steps_back + 1):
                moment = self.history[-k]
                obs_k = moment.get('obs')
                if obs_k is not None and self.active_cluster:
                    similar = self.active_cluster.find_similar(obs_k, threshold=0.5, max_results=3)
                    for idx, sim, n in similar:
                        if n.action == moment.get('action'):
                            n.flag = min(1.0, n.flag + 0.3 * discount)
                            n.strength = min(2.0, n.strength + 0.1 * discount)
                discount *= 0.92

        if done:
            self.last_neuron = None
            self._decay_on_reset()
            self.analyze_and_simulate()
        else:
            self.last_neuron = new_neuron

    def analyze_and_simulate(self):
        """Ретроспективный контрфактический самоанализ ('Сны' и виртуальное моделирование альтернатив)"""
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
        """Ослабление и 25-шаговое распределение ошибок при завершении эпизода"""
        if self.history and len(self.history) > 1:
            steps_back = min(25, len(self.history))
            discount = 1.0
            for k in range(1, steps_back + 1):
                moment = self.history[-k]
                obs_k = moment.get('obs')
                if obs_k is not None and self.active_cluster:
                    similar = self.active_cluster.find_similar(obs_k, threshold=0.5, max_results=3)
                    for idx, sim, n in similar:
                        if n.action == moment.get('action'):
                            n.flag = max(-1.0, n.flag - 0.2 * discount)
                            n.strength *= (1.0 - 0.05 * discount)
                discount *= 0.92

    def _reward_to_flag(self, reward):
        if reward >= 3.0:
            return 1.0
        elif reward > 1.0:
            return 0.6
        elif reward > 0.05:
            return 0.2
        elif reward < -3.0:
            return -1.0
        elif reward < -1.0:
            return -0.6
        elif reward < -0.05:
            return -0.2
        else:
            return 0.0

    def reset_episode(self):
        """Сбросить локальные состояния для нового эпизода"""
        self.last_obs = None
        self.last_action = None
        self.last_neuron = None
        self.history = []
