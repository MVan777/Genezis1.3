"""
Адаптер Сеточной Игры (Grid Survival Environment Adapter)
Приводит существующий класс Game к стандарту OpenAI Gym (reset, step)
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game.game import Game

class GridSurvivalEnv:
    """Адаптер игры выживания Genezis к стандарту OpenAI Gym API"""

    def __init__(self, screen=None, font=None):
        self.game = Game(screen=screen, font=font)

    def reset(self):
        """Сбросить среду и вернуть начальное наблюдение"""
        return self.game.reset()

    def step(self, action):
        """Сделать шаг и вернуть (next_obs, reward, done, info)"""
        next_obs, reward, health, alive, event = self.game.step(action)
        done = not alive
        info = {
            'health': health,
            'event': event,
            'step_count': self.game.step_count
        }
        return next_obs, reward, done, info

    def draw(self, screen, font):
        """Отрисовка в Pygame"""
        self.game.draw(screen, font)
