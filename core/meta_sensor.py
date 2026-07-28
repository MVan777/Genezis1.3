"""
Мета-Сенсор Нормализации (Meta-Sensor Normalizer)
Динамически отслеживает распределение входящих векторов любой размерности
"""

import numpy as np

class MetaSensorNormalizer:
    """Автоматически нормализует произвольные векторы входных наблюдений R^D"""

    def __init__(self, momentum=0.01):
        self.momentum = momentum
        self.running_min = None
        self.running_max = None
        self.running_mean = None
        self.count = 0

    def normalize(self, observation):
        """Нормализовать входящий вектор в диапазон [-1, 1]"""
        obs = np.array(observation, dtype=np.float32).flatten()
        if len(obs) == 0:
            return obs

        if self.running_min is None or len(self.running_min) != len(obs):
            self.running_min = obs.copy()
            self.running_max = obs.copy()
            self.running_mean = obs.copy()

        self.count += 1
        self.running_min = np.minimum(self.running_min, obs)
        self.running_max = np.maximum(self.running_max, obs)
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * obs

        diff = self.running_max - self.running_min
        diff[diff == 0] = 1e-5

        # Нормализация в [-1, 1]
        norm_obs = 2.0 * ((obs - self.running_min) / diff) - 1.0
        return norm_obs
