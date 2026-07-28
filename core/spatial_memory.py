"""
Ментальная Пространственная Карта (Spatial Mental Map)
Отслеживает тепловую карту полезности и опасности на игровой сетке
"""

import numpy as np

class SpatialMentalMap:
    """Хранит географическую память о полезных и опасных районах сетки"""

    def __init__(self, grid_size=12):
        self.grid_size = grid_size
        self.utility_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.danger_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.visit_map = np.zeros((grid_size, grid_size), dtype=np.float32)

    def record_visit(self, pos):
        """Зафиксировать посещение координаты"""
        x, y = pos[0], pos[1]
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.visit_map[x, y] += 1.0

    def record_utility(self, pos, reward):
        """Запомнить ценную клетку (спавн эликсира/оружия)"""
        x, y = pos[0], pos[1]
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.utility_map[x, y] = 0.8 * self.utility_map[x, y] + 0.2 * reward

    def record_danger(self, pos, damage):
        """Запомнить опасную клетку (урон от яда/врага)"""
        x, y = pos[0], pos[1]
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.danger_map[x, y] = 0.8 * self.danger_map[x, y] + 0.2 * damage

    def get_spatial_vector(self, player_pos):
        """
        Рассчитать вектор направления к району с наивысшим ожидаемым профитом
        (высокая полезность, низкий риск, малый уровень затаптывания)
        """
        px, py = player_pos[0], player_pos[1]
        score_map = self.utility_map - self.danger_map - 0.05 * self.visit_map

        best_score = -999.0
        best_x, best_y = px, py

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if score_map[x, y] > best_score:
                    best_score = score_map[x, y]
                    best_x, best_y = x, y

        dx = (best_x - px) / float(self.grid_size)
        dy = (best_y - py) / float(self.grid_size)
        dist = (abs(best_x - px) + abs(best_y - py)) / float(self.grid_size * 2)

        return float(dx), float(dy), float(dist)
