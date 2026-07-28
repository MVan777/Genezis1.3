"""
Маршрутизатор - выбирает активный кластер на основе состояния мира
"""

import time
import numpy as np
from core.cluster import MemoryCluster
from core.cluster_connections import ClusterConnections


class Router:
    """Выбирает, какой кластер памяти использовать для текущей ситуации"""

    def __init__(self, clusters=None, connections=None):
        self.clusters = clusters if clusters else []
        self.connections = connections if connections else ClusterConnections()
        self.active_cluster = None
        self.last_cluster = None
        self.history = []

        # Кэш для быстрого поиска по domain
        self.domain_index = {}
        self._rebuild_index()

    def _rebuild_index(self):
        """Перестроить индекс по domain"""
        self.domain_index = {}
        for cluster in self.clusters:
            if cluster.domain not in self.domain_index:
                self.domain_index[cluster.domain] = []
            self.domain_index[cluster.domain].append(cluster)

    def add_cluster(self, cluster):
        """Добавить новый кластер"""
        self.clusters.append(cluster)
        if cluster.domain not in self.domain_index:
            self.domain_index[cluster.domain] = []
        self.domain_index[cluster.domain].append(cluster)
        return cluster

    def create_cluster(self, domain="unknown", parent_id=None):
        """Создать и добавить новый кластер"""
        cluster = MemoryCluster(domain=domain, parent_id=parent_id)
        self.add_cluster(cluster)
        return cluster

    def classify_world(self, state, domain=None):
        """
        Динамически определить контекстный домен (dialogue, tennis, emergency, combat, exploration) по состоянию
        """
        if domain:
            return domain

        if isinstance(state, dict):
            if 'domain' in state:
                return state['domain']
            if 'world_type' in state:
                return state['world_type']

        if isinstance(state, (list, np.ndarray, tuple)):
            # 16-мерный вектор текстовой среды диалога
            if len(state) == 16:
                return "dialogue"

            # 5-мерное состояние игры Выживание
            if len(state) == 5:
                health_norm = float(state[0])
                enemy_near = float(state[3]) if len(state) > 3 else 0.0
                weapon_status = float(state[4]) if len(state) > 4 else 0.0

                if health_norm < 0.35:
                    return "emergency"
                elif enemy_near > 0.5 or weapon_status > 0:
                    return "combat"
                else:
                    return "exploration"

        return "exploration"

    def select_cluster(self, state, domain=None, create_if_missing=True):
        """
        Выбрать подходящий кластер для текущего состояния
        """
        world_type = self.classify_world(state, domain=domain)

        candidates = self.domain_index.get(world_type, [])

        if candidates:
            cluster = max(candidates, key=lambda c: c.usage_count)
        elif create_if_missing:
            cluster = self.create_cluster(domain=world_type)
            print(f"  Создан новый кластер: {cluster.id} для домена '{world_type}'")
        else:
            return None

        self.last_cluster = self.active_cluster
        self.active_cluster = cluster
        cluster.last_used = time.time()
        cluster.usage_count += 1

        self.history.append({
            'time': time.time(),
            'from': self.last_cluster.id if self.last_cluster else None,
            'to': cluster.id,
            'domain': world_type
        })

        return cluster

    def find_similar_across_clusters(self, situation, threshold=0.5, max_per_cluster=10):
        """Искать похожие ситуации во всех кластерах"""
        results = []

        for cluster in self.clusters:
            similar = cluster.find_similar(situation, threshold=threshold, max_results=max_per_cluster)
            for idx, sim, neuron in similar:
                results.append((cluster.id, neuron, sim))

        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def get_stats(self):
        """Статистика роутера"""
        return {
            'total_clusters': len(self.clusters),
            'active_cluster': self.active_cluster.id if self.active_cluster else None,
            'connections': self.connections.get_stats(),
            'clusters': [c.get_stats() for c in self.clusters]
        }

    def save_all(self):
        """Сохранить все кластеры и связи"""
        data = {
            'clusters': [c.to_dict() for c in self.clusters],
            'connections': self.connections.to_dict(),
            'history': self.history[-100:]
        }
        return data

    def load_all(self, data, neurons_by_id):
        """Загрузить кластеры и связи"""
        self.clusters = []
        self.domain_index = {}

        for c_data in data.get('clusters', []):
            cluster = MemoryCluster(domain=c_data.get('domain', 'unknown'))
            cluster.id = c_data.get('id', 0)
            cluster.parent_id = c_data.get('parent_id')
            cluster.created = c_data.get('created', time.time())
            cluster.last_used = c_data.get('last_used', time.time())
            cluster.usage_count = c_data.get('usage_count', 0)

            for nid in c_data.get('neurons', []):
                if nid in neurons_by_id:
                    neuron = neurons_by_id[nid]
                    neuron.cluster_id = cluster.id
                    cluster.neurons.append(neuron)
                    cluster.neuron_ids.add(nid)

            self.add_cluster(cluster)

        if 'connections' in data:
            self.connections.from_dict(data['connections'])

        self.history = data.get('history', [])