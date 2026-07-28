"""
Игровая логика с регенерацией элементов и безопасной зоной
"""

import numpy as np
import random
import pygame
import sys
import os

# Добавляем путь к корневой папке проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from game.enemy import Enemy

# Типы клеток (должны совпадать с config.py)
EMPTY = 0
POISON = 1
ANTIDOTE = 2
ELIXIR = 3
WEAPON = 4
FOOD = 5
SAFE_ZONE = 6

class Game:
    """Игровая среда с безопасной зоной"""

    def __init__(self, screen=None, font=None):
        self.screen = screen
        self.font = font
        self.reset()

    def reset(self):
        """Сброс игры"""
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.player_pos = [GRID_SIZE // 2, GRID_SIZE // 2]
        self.health = BASE_MAX_HEALTH
        self.antidote_effect = 0
        self.time_on_cell = 0
        self.last_pos = self.player_pos.copy()
        self.step_count = 0
        self.alive = True
        self.last_event = "none"
        self.last_reward = 0
        self.episode_reward = 0

        self.reached_milestones = set()

        self.enemy = Enemy(GRID_SIZE)
        self.enemy.reset()
        self.weapon_uses = 0
        self.steps_in_safe_zone = 0

        # ===== БЕЗОПАСНАЯ ЗОНА =====
        self.safe_zone_pos = [GRID_SIZE - 2, GRID_SIZE - 2]  # правый нижний угол
        self.safe_zone_active = False
        self.last_safe_zone_exit = 0
        self.safe_zone_heal = SAFE_ZONE_HEAL
        # ============================

        # Карта посещений
        self.visited = {}
        self.visited[tuple(self.player_pos)] = 0

        # Регенерация
        self.last_regen_check = 0
        self.last_elixir_step = 0
        self.last_drought_cycles = 0
        self.last_actions = [0.0, 0.0]

        # Размещаем начальные элементы
        self._place_elements()
        self.poison_timer = 0

        return self._get_state()

    def _place_elements(self):
        """Разместить элементы с учётом безопасной зоны"""
        self.grid.fill(EMPTY)

        # Размещаем безопасную зону
        self.grid[self.safe_zone_pos[0], self.safe_zone_pos[1]] = SAFE_ZONE

        for _ in range(8):
            self._place_random(ELIXIR)
        for _ in range(5):
            self._place_random(POISON)
            self._place_random(ANTIDOTE)
        for _ in range(3):  # оружие
            self._place_random(WEAPON)

        self.grid[self.player_pos[0], self.player_pos[1]] = EMPTY

    def _place_random(self, element):
        """Разместить элемент в случайной пустой клетке"""
        attempts = 0
        while attempts < 100:
            x = random.randint(0, GRID_SIZE - 1)
            y = random.randint(0, GRID_SIZE - 1)
            # Не размещаем на безопасной зоне и на игроке
            if (self.grid[x, y] == EMPTY and
                [x, y] != self.player_pos and
                [x, y] != self.safe_zone_pos):
                self.grid[x, y] = element
                return
            attempts += 1

    def _regen_elements(self):
        """Регенерировать элементы, если их мало"""
        elixir_count = np.sum(self.grid == ELIXIR)
        poison_count = np.sum(self.grid == POISON)
        antidote_count = np.sum(self.grid == ANTIDOTE)

        if elixir_count < REGEN_THRESHOLD:
            for _ in range(REGEN_ELIXIR):
                self._place_random(ELIXIR)

        if poison_count < 3:
            for _ in range(REGEN_POISON):
                self._place_random(POISON)

        if antidote_count < 3:
            for _ in range(REGEN_ANTIDOTE):
                self._place_random(ANTIDOTE)

    def _find_nearest_item(self, item_type):
        """Найти направление и расстояние до ближайшего предмета определенного типа"""
        positions = np.argwhere(self.grid == item_type)
        if len(positions) == 0:
            return 0.0, 0.0, 1.0
        px, py = self.player_pos
        dists = np.abs(positions[:, 0] - px) + np.abs(positions[:, 1] - py)
        min_idx = np.argmin(dists)
        target_x, target_y = positions[min_idx]
        dx = (target_x - px) / float(GRID_SIZE)
        dy = (target_y - py) / float(GRID_SIZE)
        dist_norm = float(dists[min_idx]) / float(GRID_SIZE * 2)
        return float(dx), float(dy), float(dist_norm)

    def _get_state(self):
        """Состояние для ИИ с информацией о веторах направления и сенсорах"""
        health_norm = self.health / BASE_MAX_HEALTH
        antidote_norm = self.antidote_effect / ANTIDOTE_DURATION

        # Расстояние до безопасной зоны
        dist_x = abs(self.player_pos[0] - self.safe_zone_pos[0])
        dist_y = abs(self.player_pos[1] - self.safe_zone_pos[1])
        max_dist = GRID_SIZE * 2
        safe_zone_dist = (dist_x + dist_y) / max_dist
        safe_zone_norm = 1.0 - safe_zone_dist  # чем ближе, тем выше значение

        # Сенсоры
        sensors = self._get_sensor_data()

        # Информация о враге
        enemy_dx = (self.enemy.x - self.player_pos[0]) / float(GRID_SIZE) if self.enemy.alive else 0.0
        enemy_dy = (self.enemy.y - self.player_pos[1]) / float(GRID_SIZE) if self.enemy.alive else 0.0
        enemy_dist = (abs(self.enemy.x - self.player_pos[0]) + abs(self.enemy.y - self.player_pos[1])) / float(GRID_SIZE * 2) if self.enemy.alive else 1.0
        enemy_near = 1.0 if (self.enemy.alive and abs(enemy_dx * GRID_SIZE) <= 1 and abs(enemy_dy * GRID_SIZE) <= 1) else 0.0

        # Относительные направления до объектов
        elixir_vec = self._find_nearest_item(ELIXIR)
        poison_vec = self._find_nearest_item(POISON)
        weapon_vec = self._find_nearest_item(WEAPON)
        enemy_vec = (enemy_dx, enemy_dy, enemy_dist)

        # Буфер инерции действий
        action_history = self.last_actions if hasattr(self, 'last_actions') else [0.0, 0.0]

        # Оружие
        weapon_status = self.weapon_uses / WEAPON_USES

        steps_since_elixir = self.step_count - self.last_elixir_step
        drought_norm = min(1.0, steps_since_elixir / 50)

        dir_features = list(elixir_vec) + list(poison_vec) + list(weapon_vec) + list(enemy_vec) + list(action_history)

        state = [health_norm, antidote_norm, drought_norm, enemy_near, weapon_status, safe_zone_norm] + dir_features + sensors
        return np.array(state, dtype=np.float32)

    def step(self, action):
        """Сделать шаг в игре"""
        if not self.alive:
            return self._get_state(), 0, self.health, False, "dead"

        old_pos = self.player_pos.copy()
        if not hasattr(self, 'last_actions'):
            self.last_actions = [0.0, 0.0]
        self.last_actions = [self.last_actions[1], action / float(ACTION_COUNT)]
        event = "none"
        reward = 0

        # ===== 1. ДВИЖЕНИЕ =====
        if action == 0:  # up
            self.player_pos[0] = max(0, self.player_pos[0] - 1)
        elif action == 1:  # down
            self.player_pos[0] = min(GRID_SIZE - 1, self.player_pos[0] + 1)
        elif action == 2:  # left
            self.player_pos[1] = max(0, self.player_pos[1] - 1)
        elif action == 3:  # right
            self.player_pos[1] = min(GRID_SIZE - 1, self.player_pos[1] + 1)

        self.step_count += 1

        # Проверка движения
        if self.player_pos == old_pos:
            self.time_on_cell += 1
        else:
            self.time_on_cell = 0

        # ===== 2. ПРОВЕРКА БЕЗОПАСНОЙ ЗОНЫ =====
        if (self.player_pos[0] == self.safe_zone_pos[0] and
                self.player_pos[1] == self.safe_zone_pos[1]):

            # Проверка на слишком частые входы/выходы
            if hasattr(self, 'last_safe_zone_exit') and self.step_count - self.last_safe_zone_exit < 5:
                reward -= 0.5
                print(f"  ⚠️ Слишком часто входишь/выходишь!")

            # Вход в безопасную зону
            if not self.safe_zone_active:
                self.safe_zone_active = True
                self.steps_in_safe_zone = 0
                print(f"  🏠 Вход в безопасную зону")

            # Находимся в зоне
            self.steps_in_safe_zone += 1
            self.health = min(self.health + self.safe_zone_heal, ABSOLUTE_MAX_HEALTH)
            event = "safe_zone_heal"
            reward += 0.5

            # ===== ШТРАФЫ ВНУТРИ ЗОНЫ =====
            # Голод растёт медленнее
            if hasattr(self, 'hunger'):
                self.hunger += self.hunger_rate / 2

            # Штраф за слишком долгое сидение
            if self.steps_in_safe_zone > 10:
                reward -= 0.2
                if self.steps_in_safe_zone % 5 == 0:
                    print(f"  ⚠️ Слишком долго в безопасной зоне ({self.steps_in_safe_zone} шагов)")
            # ================================

        else:
            # Выход из зоны
            if self.safe_zone_active:
                self.safe_zone_active = False
                self.last_safe_zone_exit = self.step_count
                print(f"  🚪 Выход из безопасной зоны")
        # =====================================

        # ===== 3. АТАКА =====
        if action == 5:
            if self.weapon_uses > 0 and self.enemy.alive and self.enemy.is_near(self.player_pos[0], self.player_pos[1]):
                killed = self.enemy.take_damage(WEAPON_DAMAGE)
                self.weapon_uses -= 1
                if killed:
                    event = "enemy_killed"
                    reward += ENEMY_KILL_REWARD
                    print(f"  ВРАГ УБИТ! +{ENEMY_KILL_REWARD}")
                else:
                    event = "enemy_hit"
                    reward += 0.5
                    print(f"  ПОПАЛ! Осталось HP врага: {self.enemy.health}")
            else:
                event = "attack_failed"
                reward -= 0.2

        # ===== 4. ДВИЖЕНИЕ ВРАГА =====
        if self.step_count % 5 == 0 and self.enemy.alive:
            self.enemy.move_towards(self.player_pos[0], self.player_pos[1], self.grid)

            if self.enemy.is_near(self.player_pos[0], self.player_pos[1]):
                self.health = self.enemy.attack_player(self.health)
                event = "enemy_attack"
                reward += ENEMY_DAMAGE_REWARD
                print(f"  ВРАГ АТАКОВАЛ! -{ENEMY_DAMAGE} здоровья")

        # ===== 5. ШТРАФ ЗА БЛУЖДАНИЯ =====
        steps_since_elixir = self.step_count - self.last_elixir_step
        if steps_since_elixir >= EXPLORATION_DROUGHT and self.alive:
            drought_cycles = steps_since_elixir // EXPLORATION_DROUGHT
            if drought_cycles > self.last_drought_cycles:
                self.health -= EXPLORATION_PENALTY
                self.last_drought_cycles = drought_cycles
                event = "exploration_drought"
                reward += EXPLORATION_REWARD_PENALTY
                print(f"  ШТРАФ: {steps_since_elixir} шагов без эликсира!")

        reward += SURVIVAL_BONUS

        # ===== 6. ПОСЕЩЕНИЕ КЛЕТОК =====
        current_pos_tuple = tuple(self.player_pos)
        if current_pos_tuple in self.visited:
            steps_ago = self.step_count - self.visited[current_pos_tuple]
            if steps_ago < REVISIT_MEMORY :
                reward -= REVISIT_PENALTY
                event = "revisit_penalty"
        else:
            reward += NEW_CELL_REWARD
            event = "new_cell"

        self.visited[current_pos_tuple] = self.step_count

        # ===== 7. ШТРАФ ЗА СТОЯНИЕ =====
        if self.time_on_cell >= STAND_STILL_TIME :
            self.health -= STAND_STILL_PENALTY
            event = "stand_penalty"
            reward -= 0.5

        # ===== 8. ЧТО НА ТЕКУЩЕЙ КЛЕТКЕ? =====
        cell_type = self.grid[self.player_pos[0], self.player_pos[1]]

        if cell_type == WEAPON:
            self.weapon_uses = WEAPON_USES
            self.grid[self.player_pos[0], self.player_pos[1]] = EMPTY
            event = "weapon_pickup"
            reward += WEAPON_PICKUP_REWARD
            print(f"  ОРУЖИЕ! +{WEAPON_USES} выстрелов")

        elif cell_type == POISON:
            event = "poison"

        elif cell_type == ANTIDOTE:
            self.antidote_effect = ANTIDOTE_DURATION
            self.grid[self.player_pos[0], self.player_pos[1]] = EMPTY
            event = "antidote"
            reward += ANTIDOTE_REWARD

        elif cell_type == ELIXIR:
            self.health = self.health + ELIXIR_HEAL
            self.grid[self.player_pos[0], self.player_pos[1]] = EMPTY
            event = "elixir"
            reward += ELIXIR_REWARD
            self.last_elixir_step = self.step_count
            self.last_drought_cycles = 0
            print(f"  ЭЛИКСИР! +{ELIXIR_HEAL} здоровья (теперь {self.health})")
            self.check_health_milestones()

        # ===== 9. АНТИДОТ =====
        if self.antidote_effect > 0:
            self.antidote_effect -= 1

        # ===== 10. ТАЙМЕР ЯДА =====
        self.poison_timer += 1
        if self.poison_timer >= POISON_TICK:
            self.poison_timer = 0
            if self.grid[self.player_pos[0], self.player_pos[1]] == POISON and self.antidote_effect <= 0:
                self.health -= POISON_DAMAGE
                event = "poison_damage"
                reward += POISON_REWARD

        # ===== 11. ПРОВЕРКА СМЕРТИ =====
        if self.health <= 0:
            self.alive = False
            self.health = 0
            event = "death"
            reward += DEATH_REWARD

        # ===== 12. ШТРАФ ЗА ШАГ =====
        if self.alive:
            reward -= STEP_PENALTY

        # ===== 13. РЕГЕНЕРАЦИЯ ЭЛЕМЕНТОВ =====
        if self.step_count - self.last_regen_check > REGEN_CHECK_INTERVAL:
            self._regen_elements()
            self.last_regen_check = self.step_count

        self.last_event = event
        self.last_reward = reward
        self.episode_reward += reward

        return self._get_state(), reward, self.health, self.alive, event

    def draw(self, screen, font):
        """Отрисовка игры с безопасной зоной"""
        if not screen or not font:
            return

        # Игровое поле
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)

                if [x, y] == self.player_pos:
                    color = COLORS['player']
                elif self.grid[x, y] == POISON:
                    color = COLORS['poison']
                elif self.grid[x, y] == ANTIDOTE:
                    color = COLORS['antidote']
                elif self.grid[x, y] == ELIXIR:
                    color = COLORS['elixir']
                elif self.grid[x, y] == WEAPON:
                    color = COLORS['weapon']
                elif self.grid[x, y] == SAFE_ZONE:
                    # Безопасная зона - мерцающий зелёный
                    t = pygame.time.get_ticks() // 500 % 2
                    if t:
                        color = (50, 255, 50)  # ярко-зелёный
                    else:
                        color = (30, 150, 30)  # тёмно-зелёный
                else:
                    if (x, y) in self.visited:
                        time_since = self.step_count - self.visited[(x, y)]
                        if time_since < REVISIT_MEMORY:
                            dark = max(50, 200 - time_since * 15)
                            color = (dark, dark, dark)
                        else:
                            color = COLORS['empty']
                    else:
                        color = COLORS['empty']

                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, COLORS['grid'], rect, 1)

        # Отрисовка врага
        self.enemy.draw(screen, CELL_SIZE)

        # Панель информации
        panel_rect = pygame.Rect(GAME_WIDTH, 0, INFO_PANEL_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(screen, COLORS['panel_bg'], panel_rect)

        steps_since_elixir = self.step_count - self.last_elixir_step

        y_offset = 20
        texts = [
            f"Здоровье: {self.health}",
            f"Антидот: {self.antidote_effect}",
            f"Событие: {self.last_event}",
            f"Награда: {self.last_reward:.2f}",
            f"Шаг: {self.step_count}",
            f"",
            f"Шагов без эликсира: {steps_since_elixir}",
            f"Штрафов: {self.last_drought_cycles}",
            f"Оружие: {self.weapon_uses}/{WEAPON_USES}",
            f"Враг: {self.enemy.health if self.enemy.alive else 0}HP",
            f"Безопасная зона: {'✅' if self.safe_zone_active else '❌'}",
            f"",
            f"Элементы:",
            f"Яд: {np.sum(self.grid == POISON)}",
            f"Антидот: {np.sum(self.grid == ANTIDOTE)}",
            f"Эликсир: {np.sum(self.grid == ELIXIR)}",
            f"Оружие: {np.sum(self.grid == WEAPON)}",
            f"",
            f"Посещено: {len(self.visited)}"
        ]

        for i, text in enumerate(texts):
            if text:
                text_surface = font.render(text, True, COLORS['text'])
                screen.blit(text_surface, (GAME_WIDTH + 10, y_offset + i * 20))

        # Полоска здоровья
        health_bar_width = 200
        health_bar_height = 20
        health_percent = self.health / BASE_MAX_HEALTH

        bar_bg_rect = pygame.Rect(GAME_WIDTH + 10, y_offset + 380, health_bar_width, health_bar_height)
        bar_fill_rect = pygame.Rect(GAME_WIDTH + 10, y_offset + 380, health_bar_width * health_percent, health_bar_height)

        pygame.draw.rect(screen, (100, 100, 100), bar_bg_rect)

        if health_percent > 0.5:
            color = (0, 255, 0)
        elif health_percent > 0.2:
            color = (255, 255, 0)
        else:
            color = (255, 0, 0)

        pygame.draw.rect(screen, color, bar_fill_rect)
        pygame.draw.rect(screen, COLORS['text'], bar_bg_rect, 2)

        # Индикатор голода/штрафов
        if steps_since_elixir > 0:
            hunger_width = int((steps_since_elixir / 50) * health_bar_width)
            hunger_width = min(health_bar_width, hunger_width)
            hunger_rect = pygame.Rect(GAME_WIDTH + 10, y_offset + 405, hunger_width, 5)
            pygame.draw.rect(screen, (255, 100, 0), hunger_rect)

    def check_health_milestones(self):
        """Проверка достижения порогов здоровья"""
        if not hasattr(self, 'reached_milestones'):
            self.reached_milestones = set()

        for milestone, bonus in HEALTH_MILESTONES.items():
            if self.health >= milestone and milestone not in self.reached_milestones:
                self.reached_milestones.add(milestone)
                if bonus > 0:
                    self.health += bonus
                    self.health = min(self.health, ABSOLUTE_MAX_HEALTH)
                    print(f"  🏆 ДОСТИГ {milestone} HP! +{bonus} бонус! (теперь {self.health})")
                else:
                    print(f"  ⭐ ДОСТИГ {milestone} HP! (порог пройден)")
                return milestone, bonus
        return None, 0

    def _get_sensor_data(self):
        """Получить данные с сенсоров (обзор на 2 клетки)"""
        sensors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dx, dy in directions:
            for dist in range(1, SENSOR_RANGE + 1):
                x = self.player_pos[0] + dx * dist
                y = self.player_pos[1] + dy * dist

                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    cell = self.grid[x, y]
                    enemy_here = (self.enemy.alive and self.enemy.x == x and self.enemy.y == y)

                    if enemy_here:
                        sensors.append(5)
                    elif cell == POISON:
                        sensors.append(1)
                    elif cell == ANTIDOTE:
                        sensors.append(2)
                    elif cell == ELIXIR:
                        sensors.append(3)
                    elif cell == WEAPON:
                        sensors.append(4)
                    elif cell == SAFE_ZONE:
                        sensors.append(6)  # безопасная зона
                    else:
                        sensors.append(0)
                else:
                    sensors.append(-1)

        return sensors

    def get_full_state(self):
        """Получить полное состояние для сохранения"""
        return {
            'player_pos': self.player_pos.copy(),
            'health': self.health,
            'antidote_effect': self.antidote_effect,
            'time_on_cell': self.time_on_cell,
            'step_count': self.step_count,
            'alive': self.alive,
            'grid': self.grid.copy(),
            'enemy': {
                'x': self.enemy.x,
                'y': self.enemy.y,
                'health': self.enemy.health,
                'alive': self.enemy.alive
            } if hasattr(self, 'enemy') else None,
            'weapon_uses': self.weapon_uses,
            'visited': dict(self.visited),
            'last_elixir_step': self.last_elixir_step,
            'last_drought_cycles': self.last_drought_cycles,
            'safe_zone_pos': self.safe_zone_pos,
            'safe_zone_active': self.safe_zone_active
        }

    def load_from_state(self, state_dict):
        """Загрузить состояние из словаря"""
        if isinstance(state_dict, dict):
            self.player_pos = state_dict.get('player_pos', self.player_pos).copy()
            self.health = state_dict.get('health', self.health)
            self.antidote_effect = state_dict.get('antidote_effect', self.antidote_effect)
            self.step_count = state_dict.get('step_count', self.step_count)
            self.alive = state_dict.get('alive', True)
            self.safe_zone_pos = state_dict.get('safe_zone_pos', self.safe_zone_pos)
            self.safe_zone_active = state_dict.get('safe_zone_active', False)

    def add_resource(self, resource_type):
        """Ручное добавление ресурса (для панели)"""
        self._place_random(resource_type)
        print(f"  ➕ Добавлен ресурс: {resource_type}")

    def add_enemy(self):
        """Ручное добавление врага"""
        if not self.enemy.alive:
            self.enemy.reset()
            print(f"  👹 Добавлен новый враг")