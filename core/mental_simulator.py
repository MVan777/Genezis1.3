"""
Глубокий Симулятор Графа Памяти (Deep Mental Graph Simulator)
Проводит 5-шаговое волновое распространение активации (Spreading Activation) в нейрографе
"""

import numpy as np

class DeepMentalGraphSimulator:
    """Глубокое мысленное моделирование будущего на 5 шагов вперед по графу памяти"""

    def __init__(self, simulation_depth=5):
        self.simulation_depth = simulation_depth

    def evaluate_future_branches(self, start_neurons, brain_ref):
        """
        Волновое распространение активации по связям next_associations
        Возвращает оценку перспективности каждого доступного действия (Action -> Score)
        """
        if not start_neurons:
            return {}

        action_future_scores = {}

        for neuron, sim, weight in start_neurons:
            action = neuron.action
            cumulative_score = neuron.flag * neuron.strength * sim

            # Запускаем волновой поход в глубину до simulation_depth
            curr_nodes = [(neuron, 1.0)]  # (neuron, decay_factor)
            
            for depth in range(1, self.simulation_depth + 1):
                next_nodes = []
                decay = 0.7 ** depth

                for curr_n, curr_w in curr_nodes:
                    if hasattr(curr_n, 'next_associations') and curr_n.next_associations:
                        for next_id, edge_w in curr_n.next_associations.items():
                            next_n = brain_ref._find_neuron_by_id(next_id)
                            if next_n:
                                node_contrib = next_n.flag * next_n.strength * edge_w * curr_w * decay
                                cumulative_score += node_contrib
                                next_nodes.append((next_n, curr_w * edge_w))

                curr_nodes = next_nodes[:5]  # ограничиваем ширину ветвления
                if not curr_nodes:
                    break

            if action not in action_future_scores:
                action_future_scores[action] = 0.0
            action_future_scores[action] += cumulative_score

        return action_future_scores
