"""
Кластер памяти - хранилище нейронов для одного типа миров/опыта
"""

import numpy as np
import time
from collections import defaultdict
from core.neuron import Neuron
from config import NEUTRAL_THRESHOLD, COMPRESSION_SIMILARITY


class MemoryCluster:
    """Кластер памяти для определённого домена (типа мира)"""

    _next_id = 0

    def __init__(self, domain="unknown", parent_id=None):
        self.id = MemoryCluster._next_id
        MemoryCluster._next_id += 1

        self.domain = domain
        self.parent_id = parent_id
        self.created = time.time()
        self.last_used = time.time()
        self.usage_count = 0

        self.neurons = []
        self.neuron_ids = set()

        self.cluster_connections = defaultdict(float)

        self.total_neurons_created = 0
        self.total_neurons_removed = 0

        self.connection_matrix = defaultdict(float)  # связи между нейронами ВНУТРИ кластера

    def strengthen_connection(self, neuron_id1, neuron_id2, amount=0.1):
        """Усилить связь между двумя нейронами в кластере"""
        if neuron_id1 == neuron_id2:
            return
        key = (min(neuron_id1, neuron_id2), max(neuron_id1, neuron_id2))
        self.connection_matrix[key] += amount
        if self.connection_matrix[key] > 1.0:
            self.connection_matrix[key] = 1.0


    def add_neuron(self, neuron):
        """Добавить нейрон в кластер"""
        if neuron.id in self.neuron_ids:
            return False

        neuron.cluster_id = self.id
        self.neurons.append(neuron)
        self.neuron_ids.add(neuron.id)
        self.last_used = time.time()
        self.usage_count += 1
        self.total_neurons_created += 1
        return True

    def remove_neuron(self, neuron_id):
        """Удалить нейрон из кластера"""
        if neuron_id not in self.neuron_ids:
            return False

        self.neurons = [n for n in self.neurons if n.id != neuron_id]
        self.neuron_ids.remove(neuron_id)
        self.total_neurons_removed += 1
        return True

    def find_similar(self, situation, threshold=0.5, max_results=20):
        """Найти похожие нейроны в кластере с векторизацией NumPy"""
        if not self.neurons:
            return []

        situation = np.array(situation, dtype=np.float32)
        sit_norm = np.linalg.norm(situation)
        if sit_norm == 0:
            return []

        sit_len = len(situation)
        curr_time = time.time()

        # Быстрая матричная векторизация для совпадения длин ситуаций
        if all(len(n.situation) == sit_len for n in self.neurons):
            matrix = np.array([n.situation for n in self.neurons], dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1)
            norms[norms == 0] = 1e-8

            dots = np.dot(matrix, situation)
            sims = dots / (norms * sit_norm)

            results = []
            for i, (sim, neuron) in enumerate(zip(sims, self.neurons)):
                age_factor = max(0.5, 1.0 - (curr_time - neuron.last_used) / 10000)
                final_sim = float(sim * neuron.strength * age_factor)
                if final_sim > threshold:
                    results.append((i, final_sim, neuron))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:max_results]

        # Поэлементный fallback для разной длины векторов
        similarities = []
        for i, neuron in enumerate(self.neurons):
            neuron_sit = neuron.situation
            min_len = min(sit_len, len(neuron_sit))
            sit_trim = situation[:min_len]
            neu_trim = neuron_sit[:min_len]

            sn = np.linalg.norm(sit_trim)
            nn = np.linalg.norm(neu_trim)

            if sn == 0 or nn == 0:
                sim = 0.0
            else:
                sim = float(np.dot(sit_trim, neu_trim) / (sn * nn))

            age_factor = max(0.5, 1.0 - (curr_time - neuron.last_used) / 10000)
            final_sim = sim * neuron.strength * age_factor
            if final_sim > threshold:
                similarities.append((i, final_sim, neuron))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]

    def cleanup(self):
        """Очистка нейтральных и слабых нейронов"""
        old_count = len(self.neurons)

        to_remove = []
        for neuron in self.neurons:
            if abs(neuron.flag) < NEUTRAL_THRESHOLD:
                to_remove.append(neuron.id)
            elif neuron.strength < 0.1:
                to_remove.append(neuron.id)
            elif hasattr(neuron, 'should_remove') and neuron.should_remove():
                to_remove.append(neuron.id)

        for nid in to_remove:
            self.remove_neuron(nid)

        return len(self.neurons) != old_count

    def compress(self, threshold=COMPRESSION_SIMILARITY):
        """Сжать похожие нейроны внутри кластера"""
        if len(self.neurons) < 3:  # Уменьшил порог
            return 0

        old_count = len(self.neurons)
        by_action = defaultdict(list)

        for n in self.neurons:
            by_action[n.action].append(n)

        new_neurons = []
        compressed = 0

        for action, group in by_action.items():
            if len(group) < 2:
                new_neurons.extend(group)
                continue

            group.sort(key=lambda n: n.strength, reverse=True)
            used = set()

            for i, n1 in enumerate(group):
                if n1.id in used:
                    continue

                cluster = [n1]
                used.add(n1.id)

                for n2 in group[i + 1:]:
                    if n2.id in used:
                        continue

                    if hasattr(n1, 'similarity'):
                        sim = n1.similarity(n2)
                    else:
                        sim = self._similarity(n1.situation, n2.situation)

                    if sim > threshold:
                        cluster.append(n2)
                        used.add(n2.id)

                if len(cluster) == 1:
                    new_neurons.append(n1)
                else:
                    compressed += len(cluster) - 1
                    main = cluster[0]
                    for other in cluster[1:]:
                        if hasattr(main, 'merge'):
                            main.merge(other)
                        else:
                            main.strength = (main.strength + other.strength) / 2
                            main.flag = (main.flag + other.flag) / 2
                            main.usage_count += other.usage_count
                    new_neurons.append(main)

        self.neurons = new_neurons
        self.neuron_ids = {n.id for n in self.neurons}

        # ВОЗВРАЩАЕМ количество сжатых
        return compressed

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

    def decay(self, factor=0.95):
        """Ослабить все нейроны"""
        for neuron in self.neurons:
            neuron.strength *= factor

    def get_stats(self):
        """Статистика кластера"""
        if not self.neurons:
            return {
                'total': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 0,
                'avg_strength': 0,
                'avg_age': 0,
                'usage': self.usage_count,
                'domain': self.domain
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
            'avg_age': np.mean(ages),
            'usage': self.usage_count,
            'domain': self.domain
        }

    def to_dict(self):
        """Для сохранения"""
        return {
            'id': self.id,
            'domain': self.domain,
            'parent_id': self.parent_id,
            'created': self.created,
            'last_used': self.last_used,
            'usage_count': self.usage_count,
            'neurons': [n.id for n in self.neurons],
            'total_neurons_created': self.total_neurons_created,
            'total_neurons_removed': self.total_neurons_removed
        }

    # ===== ДОПОЛНИТЕЛЬНЫЕ МЕТОДЫ =====
    def get_neurons_by_type(self, neuron_type):
        """Получить нейроны определённого типа"""
        if neuron_type == "short_term":
            from core.neurons import ShortTermNeuron
            return [n for n in self.neurons if isinstance(n, ShortTermNeuron)]
        elif neuron_type == "long_term":
            from core.neurons import LongTermNeuron
            return [n for n in self.neurons if isinstance(n, LongTermNeuron)]
        else:
            return self.neurons

    def cleanup_short_term(self):
        """Очистить устаревшие краткосрочные нейроны"""
        from core.neurons import ShortTermNeuron

        removed = 0
        for neuron in list(self.neurons):
            if isinstance(neuron, ShortTermNeuron) and neuron.should_remove():
                self.remove_neuron(neuron.id)
                removed += 1

        if removed > 0:
            print(f"  Кластер {self.id}: удалено {removed} устаревших нейронов")
        return removed