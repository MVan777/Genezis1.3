"""
Агент с ассоциативной памятью и самоанализом
"""
import random
import sys
import os
import numpy as np
from collections import defaultdict
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory import AssociativeMemory
from core.router import Router
from core.cluster import MemoryCluster
from core.neuron import Neuron
from core.neurons import ShortTermNeuron, LongTermNeuron
from config import ACTION_COUNT, CONFIDENCE_THRESHOLD, ENEMY_KILL_REWARD  # ← ДОБАВЛЕНО
from game.game import Game

class Agent:
    """Агент с ассоциативной памятью и самоанализом"""

    def __init__(self, router=None):
        self.router = router if router else Router()
        self.active_cluster = None
        self.last_state = None
        self.last_action = None
        self.last_similar_neurons = []
        self.used_clusters = set()

        self.stats = {
            'games': 0,
            'total_reward': 0,
            'avg_health': 0,
            'deaths': 0,
            'clusters_used': 0,
            'simulations_run': 0,
            'lessons_learned': 0
        }

        # Для анализа
        self.game_history = []
        self.death_situations = []
        self.analysis_depth = 20
        self.simulation_threshold = 0.3

        # Консолидация
        self.consolidator = None
        self.last_consolidation_check = time.time()
        self.consolidation_frequency = 10
        self.games_since_consolidation = 0

    def act(self, state, explore=True):
        """Выбрать действие"""
        self.active_cluster = self.router.select_cluster(state)
        self.used_clusters.add(self.active_cluster.id)

        similar = self._find_similar_across_clusters(state)
        self.last_similar_neurons = similar

        best_action, confidence = self._vote(similar)

        if best_action is not None and confidence > CONFIDENCE_THRESHOLD:
            action = best_action
        elif explore:
            action = random.randint(0, ACTION_COUNT - 1)
        else:
            action = 4

        # Записываем в историю
        self.game_history.append({
            'state': state.copy() if hasattr(state, 'copy') else state,
            'action_taken': action,
            'confidence': confidence,
            'similar_count': len(similar),
            'timestamp': time.time()
        })

        self.last_state = state
        self.last_action = action

        return action

    def _find_similar_across_clusters(self, state, threshold=0.4):
        """Искать похожие ситуации во всех кластерах"""
        results = []

        # Сначала в активном кластере
        if self.active_cluster:
            similar = self.active_cluster.find_similar(state, threshold=threshold)
            for idx, sim, neuron in similar:
                results.append((neuron, sim, 1.0))

        # Потом в остальных (с меньшим весом)
        if len(results) < 5:
            for cluster in self.router.clusters:
                if cluster != self.active_cluster:
                    similar = cluster.find_similar(state, threshold=threshold*0.8)
                    for idx, sim, neuron in similar:
                        results.append((neuron, sim, 0.5))

        return results

    def _vote(self, similar_neurons):
        """Взвешенное голосование с учётом типа памяти"""
        if not similar_neurons:
            return None, 0

        action_votes = defaultdict(float)

        for neuron, sim, cluster_weight in similar_neurons:
            # Отмечаем использование (для краткосрочных)
            if hasattr(neuron, 'access'):
                neuron.access()

            # Долгосрочные нейроны имеют больший вес
            type_weight = 1.5 if hasattr(neuron, 'confidence') else 1.0

            vote = sim * cluster_weight * neuron.flag * neuron.strength * type_weight
            action_votes[neuron.action] += vote

        best_action = max(action_votes.items(), key=lambda x: x[1])

        if len(action_votes) > 1:
            second_best = sorted(action_votes.values(), reverse=True)[1]
            confidence = best_action[1] - second_best
        else:
            confidence = abs(best_action[1])

        return best_action[0], confidence

    def learn(self, reward, new_state, done):
        """Обучение на реальном опыте"""
        if self.last_state is None or self.last_action is None or self.active_cluster is None:
            return

        flag = self._reward_to_flag(reward)

        # Создаём краткосрочный нейрон
        from core.neurons import ShortTermNeuron
        new_neuron = ShortTermNeuron(self.last_state, self.last_action, flag)
        self.active_cluster.add_neuron(new_neuron)

        # ===== УСИЛЕНИЕ СВЯЗЕЙ С ПОХОЖИМИ НЕЙРОНАМИ =====
        similar = self.active_cluster.find_similar(self.last_state, threshold=0.5, max_results=5)

        if hasattr(self.active_cluster, 'strengthen_connection'):
            for idx, sim, neuron in similar:
                if neuron.id != new_neuron.id:
                    self.active_cluster.strengthen_connection(
                        new_neuron.id, neuron.id, amount=sim * 0.3
                    )
        # =================================================

        # Обновляем историю
        if self.game_history:
            self.game_history[-1]['reward'] = reward
            self.game_history[-1]['flag'] = flag

        # ===== КОМПРЕССИЯ ПОСЛЕ УБИЙСТВА ВРАГА =====
        if reward == ENEMY_KILL_REWARD:
            self._auto_compress()
            print(f"  🗜️ Компрессия после убийства")
        # ===========================================

        if done:
            self.stats['deaths'] += 1
            self._record_death()
            self._decay_after_death()

            # Запускаем самоанализ
            print(f"\n🧠 Агент умер. Запускаю самоанализ...")
            lessons = self.analyze_and_simulate()
            self.stats['lessons_learned'] += lessons

            # ===== СТАТИСТИКА ФЛАГОВ =====
            pos, neg, neu = 0, 0, 0
            for cluster in self.router.clusters:
                for neuron in cluster.neurons:
                    if neuron.flag > 0.2:
                        pos += 1
                    elif neuron.flag < -0.2:
                        neg += 1
                    else:
                        neu += 1
            print(f"  📊 Флаги: +{pos} / -{neg} / 0{neu}")
            # ==============================

            self.games_since_consolidation += 1
            if self.games_since_consolidation >= self.consolidation_frequency:
                self._run_consolidation()

            # Компрессия каждые 5 смертей (как было)
            if self.stats['deaths'] % 5 == 0:
                self._auto_compress()


    def _auto_compress(self):
        """Автоматическая компрессия всех кластеров"""
        if not hasattr(self, 'router') or not self.router:
            return 0

        total = 0
        for cluster in self.router.clusters:
            if len(cluster.neurons) > 50:  # если много нейронов
                compressed = cluster.compress()
                total += compressed

        if total > 0:
            print(f"  🗜️ Автокомпрессия: сжато {total} нейронов")
        return total

    def _compress_clusters(self):
        """Запустить сжатие во всех кластерах"""
        compressed = 0
        for cluster in self.router.clusters:
            compressed += cluster.compress()
        if compressed > 0:
            print(f"  🗜️ Сжато {compressed} нейронов")
        return compressed


    def _reward_to_flag(self, reward):
        """Преобразовать награду во флаг"""
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
            return reward / 5.0

    def _record_death(self):
        """Записать ситуацию смерти"""
        if self.game_history:
            death_moment = self.game_history[-1].copy()
            death_moment['death'] = True
            self.death_situations.append(death_moment)

    def _decay_after_death(self):
        """Ослабление после смерти"""
        for cluster_id in self.used_clusters:
            cluster = self._get_cluster_by_id(cluster_id)
            if cluster:
                cluster.decay(factor=0.95)

        # Усиливаем связи между использованными кластерами
        used_list = list(self.used_clusters)
        for i in range(len(used_list)):
            for j in range(i+1, len(used_list)):
                self.router.connections.strengthen(used_list[i], used_list[j])

    def _get_cluster_by_id(self, cluster_id):
        """Найти кластер по ID"""
        for cluster in self.router.clusters:
            if cluster.id == cluster_id:
                return cluster
        return None

    # ===== НОВЫЕ МЕТОДЫ: САМОАНАЛИЗ =====

    def analyze_and_simulate(self):
        """Полный анализ последней игры с симуляцией альтернатив"""
        if not self.game_history or len(self.game_history) < 5:
            return 0

        print(f"\n🔍 АНАЛИЗ ИГРЫ (шагов: {len(self.game_history)})")

        key_moments = self._find_key_moments()
        print(f"   Найдено ключевых моментов: {len(key_moments)}")

        lessons = 0

        for i, moment in enumerate(key_moments):
            print(f"\n   ⏱️  Момент {i+1}: шаг {moment['step']}, уверенность {moment['confidence']:.2f}")

            for alt_action in range(ACTION_COUNT):
                if alt_action == moment['action_taken']:
                    continue

                result = self._simulate_from(moment, alt_action)
                self.stats['simulations_run'] += 1

                if result['better_than_real']:
                    print(f"     ✅ Альтернатива {alt_action} лучше!")
                    self._learn_from_simulation(moment['state'], alt_action, +0.7)
                    lessons += 1
                elif result['much_worse']:
                    print(f"     ❌ Альтернатива {alt_action} хуже")
                    self._learn_from_simulation(moment['state'], alt_action, -0.5)
                    lessons += 1

        print(f"\n   📚 Итого уроков: {lessons}")
        return lessons

    def _find_key_moments(self, threshold=0.3):
        """Найти ключевые моменты для анализа"""
        key_moments = []

        for i, step in enumerate(self.game_history):
            # Моменты с низкой уверенностью
            if step.get('confidence', 1.0) < threshold:
                step_copy = step.copy()
                step_copy['step'] = i
                step_copy['reason'] = 'low_confidence'
                key_moments.append(step_copy)
                continue

            # Моменты перед уроном
            if i < len(self.game_history) - 1:
                next_step = self.game_history[i+1]
                if next_step.get('reward', 0) < -0.5:
                    step_copy = step.copy()
                    step_copy['step'] = i
                    step_copy['reason'] = 'damage_taken'
                    key_moments.append(step_copy)

        return key_moments[:10]

    def _simulate_from(self, moment, alt_action, depth=20):
        """Симулировать альтернативный сценарий"""
        # Создаём копию игры
        sim_game = Game()

        # Пытаемся загрузить состояние
        if hasattr(sim_game, 'load_from_state'):
            sim_game.load_from_state(moment.get('state', {}))

        # Делаем альтернативное действие
        new_state, reward, health, alive, event = sim_game.step(alt_action)

        steps_survived = 1
        total_reward = reward

        # Дальше действуем по текущей политике
        for _ in range(depth - 1):
            if not alive:
                break

            action = self.act(new_state, explore=False)
            new_state, reward, health, alive, event = sim_game.step(action)

            steps_survived += 1
            total_reward += reward

        # Сравниваем с реальным результатом
        real_outcome = self._get_real_outcome(moment['step'])

        return {
            'survived': alive,
            'steps': steps_survived,
            'total_reward': total_reward,
            'better_than_real': total_reward > real_outcome['reward'],
            'much_worse': total_reward < real_outcome['reward'] - 10
        }

    def _get_real_outcome(self, start_step):
        """Реальный исход от данного момента"""
        total_reward = 0

        for i in range(start_step, len(self.game_history)):
            total_reward += self.game_history[i].get('reward', 0)

        return {'reward': total_reward}

    def _learn_from_simulation(self, state, action, flag):
        """Создать новый нейрон на основе симуляции"""
        from core.neurons import ShortTermNeuron
        new_neuron = ShortTermNeuron(state, action, flag)
        if self.active_cluster:
            self.active_cluster.add_neuron(new_neuron)

    def _run_consolidation(self):
        """Запустить консолидацию памяти"""
        if not self.router or not self.router.clusters:
            return

        if not self.consolidator:
            from core.consolidator import MemoryConsolidator

            # Ищем кластеры
            short_term = None
            long_term = None

            for cluster in self.router.clusters:
                if cluster.domain == "short_term":
                    short_term = cluster
                elif cluster.domain == "long_term":
                    long_term = cluster

            if not short_term:
                short_term = self.router.create_cluster(domain="short_term")
            if not long_term:
                long_term = self.router.create_cluster(domain="long_term")

            self.consolidator = MemoryConsolidator(short_term, long_term)

        consolidated = self.consolidator.consolidate()
        if consolidated:
            print(f"  🧠 Консолидация: {consolidated} нейронов перенесено")
            self.stats['lessons_learned'] += consolidated

        self.games_since_consolidation = 0

    def new_game(self):
        """Новая игра"""
        self.last_state = None
        self.last_action = None
        self.game_history = []
        self.used_clusters = set()
        if self.stats['games'] % 5 == 0:  # каждые 5 игр
            self._auto_compress()