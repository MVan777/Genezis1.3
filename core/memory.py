"""
ЯДРО: Класс AssociativeMemory
Добавлено: сжатие похожих нейронов
"""

import numpy as np
from collections import defaultdict
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.neuron import Neuron
from config import (SIMILARITY_THRESHOLD, NEUTRAL_THRESHOLD,
                   COMPRESSION_SIMILARITY, COMPRESSION_INTERVAL,
                   MAX_NEURONS_BEFORE_COMPRESSION)


class AssociativeMemory:
    """Ассоциативная память на нейронах"""

    def __init__(self):
        self.neurons = []
        self.epoch = 0
        self.step_counter = 0
        self.connection_matrix = defaultdict(float)
        self.last_compression = 0

    def add_neuron(self, situation, action, result_flag):
        """Добавить новый нейрон"""
        neuron = Neuron(situation, action, result_flag, timestamp=time.time())
        self.neurons.append(neuron)
        return neuron.id

    def find_similar(self, situation, threshold=SIMILARITY_THRESHOLD):
        """Найти похожие нейроны"""
        if not self.neurons:
            return []

        situation = np.array(situation, dtype=np.float32)

        similarities = []
        for i, neuron in enumerate(self.neurons):
            neuron_sit = neuron.situation
            min_len = min(len(situation), len(neuron_sit))
            sit_trim = situation[:min_len]
            neu_trim = neuron_sit[:min_len]

            sit_norm = np.linalg.norm(sit_trim)
            pat_norm = np.linalg.norm(neu_trim)

            if sit_norm == 0 or pat_norm == 0:
                sim = 0
            else:
                sim = np.dot(sit_trim, neu_trim) / (sit_norm * pat_norm)

            # Учитываем свежесть (чем свежее, тем важнее)
            age_factor = max(0.5, 1.0 - (time.time() - neuron.last_used) / 10000)
            sim = sim * neuron.strength * age_factor
            similarities.append((i, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [(i, sim) for i, sim in similarities if sim > threshold]

    def get_best_action(self, situation):
        """Получить лучшее действие"""
        similar = self.find_similar(situation, threshold=SIMILARITY_THRESHOLD)

        if not similar:
            return None, 0, []

        action_votes = defaultdict(float)
        voting_neurons = []

        for idx, sim in similar:
            neuron = self.neurons[idx]
            vote = sim * abs(neuron.flag) * (1 if neuron.flag > 0 else -1)
            action_votes[neuron.action] += vote
            voting_neurons.append((idx, sim, neuron.flag, neuron.action))

        if not action_votes:
            return None, 0, voting_neurons

        best_action = max(action_votes.items(), key=lambda x: x[1])

        if len(action_votes) > 1:
            second_best = sorted(action_votes.values(), reverse=True)[1]
            confidence = best_action[1] - second_best
        else:
            confidence = abs(best_action[1])

        return best_action[0], confidence, voting_neurons

    def update_with_result(self, situation, action, result_flag):
        """Обновить память"""
        exact_match = None
        for i, neuron in enumerate(self.neurons):
            if np.array_equal(neuron.situation, situation) and neuron.action == action:
                exact_match = i
                break

        if exact_match is not None:
            self.neurons[exact_match].strengthen(result_flag, timestamp=time.time())
            self._strengthen_connections(exact_match)
        else:
            new_id = self.add_neuron(situation, action, result_flag)
            self._create_initial_connections(new_id)

        self.step_counter += 1

        # Периодическое сжатие
        if self.step_counter - self.last_compression > COMPRESSION_INTERVAL:
            self.compress()
            self.last_compression = self.step_counter

        # Очистка старых нейронов
        if self.step_counter % 50 == 0:
            self.cleanup()

    def _strengthen_connections(self, neuron_idx):
        """Усилить связи нейрона с похожими"""
        neuron = self.neurons[neuron_idx]
        for i, other in enumerate(self.neurons):
            if i != neuron_idx:
                sim = self._similarity(neuron.situation, other.situation)
                if sim > 0.3:
                    # ИСПРАВЛЕНО: используем get() для безопасного увеличения
                    key1 = (neuron.id, other.id)
                    key2 = (other.id, neuron.id)

                    self.connection_matrix[key1] = self.connection_matrix.get(key1, 0) + 0.1
                    self.connection_matrix[key2] = self.connection_matrix.get(key2, 0) + 0.1

    def _create_initial_connections(self, neuron_id):
        """Создать начальные связи для нового нейрона"""
        neuron = self._get_neuron_by_id(neuron_id)
        if not neuron:
            return

        for other in self.neurons:
            if other.id != neuron_id:
                sim = self._similarity(neuron.situation, other.situation)
                if sim > 0.5:
                    # ИСПРАВЛЕНО: используем прямую установку
                    self.connection_matrix[(neuron_id, other.id)] = sim * 0.5
                    self.connection_matrix[(other.id, neuron_id)] = sim * 0.5

    def _get_neuron_by_id(self, neuron_id):
        """Найти нейрон по ID"""
        for neuron in self.neurons:
            if neuron.id == neuron_id:
                return neuron
        return None

    def _similarity(self, a, b):
        """Косинусная близость двух векторов"""
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]

        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0
        return np.dot(a, b) / (a_norm * b_norm)

    # ========= НОВЫЙ МЕТОД: СЖАТИЕ =========
    def compress(self):
        """Сжать похожие нейроны"""
        old_count = len(self.neurons)
        if old_count < 50:
            return

        print(f"  Сжатие памяти: {old_count} нейронов...")

        # Группируем по действиям
        neurons_by_action = defaultdict(list)
        for neuron in self.neurons:
            neurons_by_action[neuron.action].append(neuron)

        new_neurons = []
        compressed_count = 0

        for action, group in neurons_by_action.items():
            if len(group) < 2:
                new_neurons.extend(group)
                continue

            # Сортируем по силе
            group.sort(key=lambda n: n.strength, reverse=True)
            used = set()

            for i, n1 in enumerate(group):
                if n1.id in used:
                    continue

                current_cluster = [n1]
                used.add(n1.id)

                # Ищем похожих
                for j, n2 in enumerate(group[i + 1:], i + 1):
                    if n2.id in used:
                        continue

                    sim = n1.similarity(n2)
                    if sim > COMPRESSION_SIMILARITY:
                        current_cluster.append(n2)
                        used.add(n2.id)

                if len(current_cluster) == 1:
                    new_neurons.append(n1)
                else:
                    compressed_count += len(current_cluster) - 1
                    main_neuron = current_cluster[0]

                    for other in current_cluster[1:]:
                        main_neuron.merge(other)

                        # ИСПРАВЛЕНО: безопасный перенос связей
                        connections_to_transfer = []
                        for (id1, id2), strength in self.connection_matrix.items():
                            if id2 == other.id and id1 != main_neuron.id:
                                connections_to_transfer.append((id1, main_neuron.id, strength))
                            if id1 == other.id and id2 != main_neuron.id:
                                connections_to_transfer.append((main_neuron.id, id2, strength))

                        # Добавляем перенесенные связи
                        for id1, id2, strength in connections_to_transfer:
                            self.connection_matrix[(id1, id2)] = self.connection_matrix.get((id1, id2), 0) + strength

                        # Удаляем старые связи с other
                        keys_to_delete = [key for key in self.connection_matrix.keys()
                                          if key[0] == other.id or key[1] == other.id]
                        for key in keys_to_delete:
                            del self.connection_matrix[key]

                    new_neurons.append(main_neuron)

        self.neurons = new_neurons

        # Финальная очистка
        valid_ids = {n.id for n in self.neurons}
        self.connection_matrix = {
            (id1, id2): s for (id1, id2), s in self.connection_matrix.items()
            if id1 in valid_ids and id2 in valid_ids and id1 != id2
        }

        print(f"  Сжато: {compressed_count} нейронов, осталось: {len(self.neurons)}")
    # ========================================


    def cleanup(self):
        """Удалить старые и слабые нейроны"""
        old_count = len(self.neurons)

        to_remove = []
        for neuron in self.neurons:
            if neuron.should_remove():
                to_remove.append(neuron.id)

        self.neurons = [n for n in self.neurons if n.id not in to_remove]

        # Удаляем связи
        keys_to_remove = []
        for (id1, id2) in list(self.connection_matrix.keys()):
            if id1 in to_remove or id2 in to_remove:
                keys_to_remove.append((id1, id2))

        for key in keys_to_remove:
            del self.connection_matrix[key]

        if len(self.neurons) != old_count:
            print(f"  Очистка памяти: удалено {old_count - len(self.neurons)} нейронов")

    def decay_all(self):
        """Ослабить все нейроны"""
        for neuron in self.neurons:
            neuron.weaken()

        for key in list(self.connection_matrix.keys()):
            self.connection_matrix[key] *= 0.95
            if self.connection_matrix[key] < 0.1:
                del self.connection_matrix[key]

    def get_stats(self):
        """Статистика памяти"""
        if not self.neurons:
            return {
                'total': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 0,
                'avg_strength': 0,
                'connections': 0,
                'avg_age': 0
            }

        flags = [n.flag for n in self.neurons]
        strengths = [n.strength for n in self.neurons]
        ages = [time.time() - n.created_at for n in self.neurons]

        return {
            'total': len(self.neurons),
            'positive': sum(1 for f in flags if f > NEUTRAL_THRESHOLD),
            'negative': sum(1 for f in flags if f < -NEUTRAL_THRESHOLD),
            'neutral': sum(1 for f in flags if abs(f) <= NEUTRAL_THRESHOLD),
            'avg_strength': np.mean(strengths),
            'connections': len(self.connection_matrix),
            'avg_age': np.mean(ages)
        }

    def get_neurons_for_viz(self):
        """Данные для визуализации"""
        neurons_data = []
        for neuron in self.neurons:
            neurons_data.append({
                'id': neuron.id,
                'pos': Neuron._positions.get(neuron.id, (100, 100)),
                'color': neuron.get_color(),
                'strength': neuron.strength,
                'flag': neuron.flag,
                'action': neuron.action,
                'age': time.time() - neuron.created_at
            })
        return neurons_data, self.connection_matrix