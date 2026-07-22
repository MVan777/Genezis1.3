"""
Враг - движущийся противник
"""

import numpy as np
import random
import pygame
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GRID_SIZE, ENEMY_DAMAGE, COLORS, CELL_SIZE

class Enemy:
    """Враг, который движется к игроку"""

    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        """Разместить врага в случайном месте"""
        self.x = random.randint(0, self.grid_size - 1)
        self.y = random.randint(0, self.grid_size - 1)
        self.health = 100  # здоровье врага
        self.alive = True
        self.last_move = 0

    def move_towards(self, player_x, player_y, game_grid):
        """Двигаться к игроку (не наступает на яд)"""
        if not self.alive:
            return

        # Вычисляем направление к игроку
        dx = player_x - self.x
        dy = player_y - self.y

        # Нормализуем
        if abs(dx) > abs(dy):
            # Двигаемся по x
            new_x = self.x + (1 if dx > 0 else -1)
            new_y = self.y
        else:
            # Двигаемся по y
            new_x = self.x
            new_y = self.y + (1 if dy > 0 else -1)

        # Проверяем, можно ли туда пойти (не яд и не игрок)
        if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
            if game_grid[new_x, new_y] != 1:  # 1 = яд
                self.x = new_x
                self.y = new_y

    def attack_player(self, player_health):
        """Атаковать игрока, если рядом"""
        return player_health - ENEMY_DAMAGE

    def take_damage(self, damage):
        """Получить урон от оружия"""
        self.health -= damage
        if self.health <= 0:
            self.alive = False
            return True  # враг умер
        return False

    def is_near(self, player_x, player_y, distance=1):
        """Проверить, рядом ли с игроком"""
        return abs(self.x - player_x) <= distance and abs(self.y - player_y) <= distance

    def draw(self, screen, cell_size):
        """Отрисовка врага"""
        if not self.alive:
            return

        rect = pygame.Rect(self.y * cell_size, self.x * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, COLORS['enemy'], rect)

        # Полоска здоровья врага
        health_percent = self.health / 100
        bar_width = cell_size - 4
        bar_height = 3
        bar_x = self.y * cell_size + 2
        bar_y = self.x * cell_size + cell_size - 5

        pygame.draw.rect(screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))
        if health_percent > 0:
            health_color = (0, 255, 0) if health_percent > 0.5 else (255, 255, 0)
            pygame.draw.rect(screen, health_color, (bar_x, bar_y, bar_width * health_percent, bar_height))