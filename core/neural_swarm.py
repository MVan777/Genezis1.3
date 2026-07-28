"""
Координатор Мульти-Агентного Роя (NeuralSwarmCoordinator)
Запускает рой из 3-5 специализированных мозгов ИИ и синхронизирует Золотые Прототипы по Шине Памяти
"""

import numpy as np
from core.genezis_3_ultimate import Genezis3UltimateEngine

class SwarmMemoryBus:
    """Шина мгновенного обмена выученным опытом между членами роя"""

    def __init__(self):
        self.shared_golden_prototypes = []

    def broadcast_prototypes(self, sender_id, prototypes):
        """Отправить прототипы в общий банк роя"""
        added = 0
        for proto in prototypes:
            if proto not in self.shared_golden_prototypes:
                self.shared_golden_prototypes.append(proto)
                added += 1
        return added

    def sync_brain(self, brain):
        """Синхронизировать память одного мозга с общим банком роя"""
        if not self.shared_golden_prototypes:
            return 0

        synced = 0
        target_cluster = brain.router.clusters[0] if brain.router.clusters else None
        if target_cluster:
            for proto in self.shared_golden_prototypes:
                if proto.id not in [n.id for n in target_cluster.neurons]:
                    target_cluster.add_neuron(proto)
                    synced += 1
        return synced

class NeuralSwarmCoordinator:
    """Координатор взаимодействия специализированного роя мозгов ИИ"""

    def __init__(self, agent_count=3, action_count=3):
        self.bus = SwarmMemoryBus()
        self.agents = []
        self.roles = ['ATTACKER', 'DEFENDER', 'ANALYST']

        for i in range(agent_count):
            role = self.roles[i % len(self.roles)]
            brain = Genezis3UltimateEngine(action_count=action_count)
            self.agents.append({
                'id': f"SwarmAgent-{i+1}",
                'role': role,
                'brain': brain,
                'wins': 0
            })

    def run_swarm_step(self, obs_list=None):
        """Выполнить симуляционный шаг взаимодействия роя"""
        if obs_list is None:
            obs_list = [np.random.uniform(-1, 1, 10).astype(np.float32) for _ in self.agents]

        actions = []
        for i, agent in enumerate(self.agents):
            brain = agent['brain']
            obs = obs_list[i]
            action = brain.act(obs)
            actions.append(action)

            # Выполняем синхронизацию памяти с шиной роя
            self.bus.sync_brain(brain)

        return actions

    def share_swarm_experience(self):
        """Обменяться лучшими нейронами между всеми членами роя"""
        total_broadcasted = 0
        for agent in self.agents:
            brain = agent['brain']
            prototypes = [n for c in brain.router.clusters for n in c.neurons if getattr(n, 'confidence', 0) > 0.5]
            broadcasted = self.bus.broadcast_prototypes(agent['id'], prototypes)
            total_broadcasted += broadcasted

        # Рассылаем собранные золотые прототипы всем агентам
        total_synced = 0
        for agent in self.agents:
            total_synced += self.bus.sync_brain(agent['brain'])

        return total_broadcasted, total_synced
