"""
Арена для дуэлей двух агентов
"""

import pygame
import numpy as np
import time
from game.game import Game


class DuelArena:
    """
    Два агента в одной игре
    """

    def __init__(self, agent1, agent2, screen=None, font=None):
        self.agent1 = agent1
        self.agent2 = agent2
        self.game = Game(screen, font)
        self.screen = screen
        self.font = font

        self.step_count = 0
        self.max_steps = 500
        self.finished = False

        # История дуэли
        self.history = []

    def reset(self):
        """Новая дуэль"""
        self.game.reset()
        self.agent1.game_history = []
        self.agent2.game_history = []
        self.step_count = 0
        self.finished = False

        # Размещаем агентов в разных местах
        self.game.player1_pos = [2, 2]
        self.game.player2_pos = [self.game.grid_size - 3, self.game.grid_size - 3]

        return self._get_state()

    def _get_state(self):
        """Состояние для обоих агентов"""
        # Базовая информация о мире
        world_state = {
            'grid': self.game.grid.copy(),
            'resources': self.game.get_resources(),
            'time': self.step_count
        }

        # Информация для первого агента
        state1 = {
            **world_state,
            'my_pos': self.game.player1_pos,
            'enemy_pos': self.game.player2_pos,
            'my_health': self.game.player1_health,
            'enemy_health': self.game.player2_health,
            'my_weapon': self.game.player1_weapon,
            'enemy_weapon': self.game.player2_weapon
        }

        # Информация для второго
        state2 = {
            **world_state,
            'my_pos': self.game.player2_pos,
            'enemy_pos': self.game.player1_pos,
            'my_health': self.game.player2_health,
            'enemy_health': self.game.player1_health,
            'my_weapon': self.game.player2_weapon,
            'enemy_weapon': self.game.player1_weapon
        }

        return state1, state2

    def step(self):
        """Один шаг дуэли"""
        if self.finished:
            return

        # Получаем состояния
        state1, state2 = self._get_state()

        # Агенты выбирают действия
        action1 = self.agent1.act(state1)
        action2 = self.agent2.act(state2)

        # Игровой шаг с двумя игроками
        results = self.game.step_duel(action1, action2)

        # Агенты получают награды
        self.agent1.learn(results['reward1'], results['state1'], results['done'])
        self.agent2.learn(results['reward2'], results['state2'], results['done'])

        # Запоминаем
        self.history.append({
            'step': self.step_count,
            'action1': action1,
            'action2': action2,
            'health1': results['health1'],
            'health2': results['health2']
        })

        self.step_count += 1

        # Проверка окончания
        if results['done'] or self.step_count >= self.max_steps:
            self.finished = True
            winner = self._get_winner()
            return winner

        return None

    def _get_winner(self):
        """Определить победителя"""
        if self.game.player1_health <= 0:
            return self.agent2
        elif self.game.player2_health <= 0:
            return self.agent1
        else:
            # По очкам
            if self.agent1.fitness > self.agent2.fitness:
                return self.agent1
            else:
                return self.agent2

    def draw(self, screen, font):
        """Отрисовка дуэли"""
        if not screen:
            return

        # Отрисовываем игру
        self.game.draw_duel(screen, font)

        # Информация об агентах
        y_offset = 10
        texts = [
            f"Дуэль: {self.agent1.name} vs {self.agent2.name}",
            f"Шаг: {self.step_count}/{self.max_steps}",
            f"Здоровье: {self.game.player1_health} : {self.game.player2_health}",
            f"Стратегия 1: {self.agent1.strategy_weights}",
            f"Стратегия 2: {self.agent2.strategy_weights}"
        ]

        for text in texts:
            surf = font.render(str(text), True, (255, 255, 255))
            screen.blit(surf, (10, y_offset))
            y_offset += 25