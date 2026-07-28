"""
Синтетическое Окружение (Benchmark Environment)
Демонстрирует обучение UniversalAssociativeBrain на произвольной не-игровой задаче
"""

import numpy as np
import random

class BenchmarkEnv:
    """Стандартный интерфейс OpenAI Gym API для синтетической задачи классификации/управления"""

    def __init__(self, obs_dim=20, action_count=4):
        self.obs_dim = obs_dim
        self.action_count = action_count
        self.target_pattern = np.sin(np.linspace(0, 2 * np.pi, obs_dim))
        self.step_count = 0
        self.max_steps = 100

    def reset(self):
        """Сброс среды"""
        self.step_count = 0
        noise = np.random.normal(0, 0.1, self.obs_dim)
        return (self.target_pattern + noise).astype(np.float32)

    def step(self, action):
        """Выполнить шаг в среде"""
        self.step_count += 1
        noise = np.random.normal(0, 0.1, self.obs_dim)
        next_obs = (self.target_pattern + noise).astype(np.float32)

        # Оптимальное действие определяется по сумме признаков
        optimal_action = int(np.argmax(next_obs)) % self.action_count

        if action == optimal_action:
            reward = 2.0
        else:
            reward = -0.5

        done = (self.step_count >= self.max_steps)
        info = {'optimal_action': optimal_action}

        return next_obs, reward, done, info
