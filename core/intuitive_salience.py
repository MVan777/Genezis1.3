"""
Движок Эвристической Интуиции (Intuitive Salience Engine)
Приоритезирует интуитивные прыжки к воспоминаниям с высокой эмоциональной яркостью (|Delta R| > 1.0)
"""

import numpy as np

class IntuitiveSalienceEngine:
    """Управление яркими эмоциональными мысленными прыжками ИИ"""

    def __init__(self, salience_threshold=1.0):
        self.salience_threshold = salience_threshold
        self.salient_neurons = []

    def register_experience(self, neuron, delta_reward):
        """Зарегистрировать эмоциональную яркость события"""
        salience_score = abs(float(delta_reward))
        neuron.salience = salience_score

        if salience_score >= self.salience_threshold:
            if neuron not in self.salient_neurons:
                self.salient_neurons.append(neuron)

    def find_salient_leaps(self, current_obs, threshold=0.3):
        """Найти интуитивные прыжки к наиболее ярким впечатлениям"""
        if not self.salient_neurons:
            return []

        leaps = []
        obs_array = np.array(current_obs, dtype=np.float32)
        obs_norm = np.linalg.norm(obs_array)
        if obs_norm == 0:
            return []

        for neuron in self.salient_neurons:
            sit_len = min(len(obs_array), len(neuron.situation))
            neu_trim = neuron.situation[:sit_len]
            obs_trim = obs_array[:sit_len]
            nn = np.linalg.norm(neu_trim)

            if nn > 0:
                sim = float(np.dot(obs_trim, neu_trim) / (obs_norm * nn))
                if sim > threshold:
                    leaps.append((neuron, sim * neuron.salience))

        leaps.sort(key=lambda x: x[1], reverse=True)
        return leaps[:3]
