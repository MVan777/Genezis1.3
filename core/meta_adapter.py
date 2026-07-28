"""
Мета-Адаптер Обучения (Meta-Learning Adapter)
Динамически регулирует пороги уверенности, любопытства и параметры поиска на основе изменчивости среды
"""

import numpy as np

class MetaLearningAdapter:
    """Адаптивный регулятор параметров управления ИИ на основе изменчивости входящих данных"""

    def __init__(self):
        self.volatility_history = []
        self.confidence_threshold = 0.15
        self.curiosity_weight = 0.1
        self.simulation_depth = 5

    def adapt(self, current_obs, reward_history):
        """Адаптировать параметры ИИ в реальном времени"""
        if len(reward_history) > 10:
            reward_std = float(np.std(reward_history[-10:]))
            
            # Если среда очень изменчивая/непредсказуемая - повышаем любопытство и глубину планирования
            if reward_std > 1.0:
                self.curiosity_weight = min(0.3, self.curiosity_weight * 1.05)
                self.simulation_depth = 6
                self.confidence_threshold = 0.10
            else:
                self.curiosity_weight = max(0.05, self.curiosity_weight * 0.98)
                self.simulation_depth = 5
                self.confidence_threshold = 0.15

        return {
            'confidence_threshold': self.confidence_threshold,
            'curiosity_weight': self.curiosity_weight,
            'simulation_depth': self.simulation_depth
        }
