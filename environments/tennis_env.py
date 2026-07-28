"""
Окружение Теннис / Pong Duel (Tennis Environment)
Левая ракетка: Алгоритм-соперник (Heuristic Rule Bot)
Правая ракетка: Ассоциативный ИИ (UniversalAssociativeBrain)
"""

import numpy as np
import random
import pygame

class TennisEnv:
    """Дуэльное окружение тенниса в стандарте OpenAI Gym API"""

    def __init__(self, width=600, height=400):
        self.width = width
        self.height = height
        self.paddle_width = 12
        self.paddle_height = 70
        self.paddle_speed = 6
        self.ball_radius = 8

        self.reset()

    def reset(self):
        """Сброс состояния матча"""
        self.ball_x = self.width / 2.0
        self.ball_y = self.height / 2.0

        # Начальная скорость мяча
        angle_dir = 1 if random.random() > 0.5 else -1
        self.ball_vx = 5.0 * angle_dir
        self.ball_vy = random.choice([-3.0, -2.0, 2.0, 3.0])

        self.opp_paddle_y = self.height / 2.0 - self.paddle_height / 2.0
        self.ai_paddle_y = self.height / 2.0 - self.paddle_height / 2.0

        self.score_opp = 0
        self.score_ai = 0
        self.step_count = 0
        self.max_steps = 1000

        return self._get_observation()

    def _get_observation(self):
        """Вектор наблюдения R^7 для Ассоциативного ИИ"""
        norm_ball_x = self.ball_x / float(self.width)
        norm_ball_y = self.ball_y / float(self.height)
        norm_vx = self.ball_vx / 10.0
        norm_vy = self.ball_vy / 10.0
        norm_ai_y = (self.ai_paddle_y + self.paddle_height / 2.0) / float(self.height)
        norm_opp_y = (self.opp_paddle_y + self.paddle_height / 2.0) / float(self.height)
        dy = (self.ball_y - (self.ai_paddle_y + self.paddle_height / 2.0)) / float(self.height)

        obs = [norm_ball_x, norm_ball_y, norm_vx, norm_vy, norm_ai_y, norm_opp_y, dy]
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """
        Выполнить шаг в теннисе
        action: 0 - STAY, 1 - UP, 2 - DOWN
        """
        self.step_count += 1
        reward = 0.0
        event = "none"

        # ===== 1. ДВИЖЕНИЕ РАКЕТКИ ИИ (Правая сторона) =====
        if action == 1:  # UP
            self.ai_paddle_y = max(0, self.ai_paddle_y - self.paddle_speed)
        elif action == 2:  # DOWN
            self.ai_paddle_y = min(self.height - self.paddle_height, self.ai_paddle_y + self.paddle_speed)

        # ===== 2. ДВИЖЕНИЕ АЛГОРИТМА-СОПЕРНИКА (Левая сторона) =====
        opp_center = self.opp_paddle_y + self.paddle_height / 2.0
        if self.ball_x < self.width * 0.7:  # реагирует, когда мяч на его стороне
            if opp_center < self.ball_y - 5:
                self.opp_paddle_y = min(self.height - self.paddle_height, self.opp_paddle_y + self.paddle_speed * 0.8)
            elif opp_center > self.ball_y + 5:
                self.opp_paddle_y = max(0, self.opp_paddle_y - self.paddle_speed * 0.8)

        # ===== 3. ДВИЖЕНИЕ МЯЧА =====
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Отскок от верхнего и нижнего краев
        if self.ball_y - self.ball_radius <= 0 or self.ball_y + self.ball_radius >= self.height:
            self.ball_vy *= -1.0

        # ===== 4. ПРОВЕРКА ОТБИВАНИЯ И ГОЛОВ =====
        # Отбивание левой ракеткой (Алгоритм)
        if self.ball_x - self.ball_radius <= 20:
            if self.opp_paddle_y <= self.ball_y <= self.opp_paddle_y + self.paddle_height:
                self.ball_vx = abs(self.ball_vx) * 1.05  # небольшое ускорение
                event = "opp_hit"
            else:
                # Гол от ИИ!
                self.score_ai += 1
                reward += 5.0
                event = "ai_goal"
                self._reset_ball(direction=1)

        # Отбивание правой ракеткой (Наш Ассоциативный ИИ)
        elif self.ball_x + self.ball_radius >= self.width - 20:
            if self.ai_paddle_y <= self.ball_y <= self.ai_paddle_y + self.paddle_height:
                self.ball_vx = -abs(self.ball_vx) * 1.05
                reward += 2.0  # Отличная награда за отбивание мяча
                event = "ai_hit"
            else:
                # Пропуск мяча ИИ
                self.score_opp += 1
                reward -= 3.0
                event = "ai_miss"
                self._reset_ball(direction=-1)

        # Бонус за точное следование за мячом
        ai_center = self.ai_paddle_y + self.paddle_height / 2.0
        if abs(ai_center - self.ball_y) < 20:
            reward += 0.05

        done = (self.step_count >= self.max_steps or self.score_ai >= 10 or self.score_opp >= 10)
        info = {
            'score_ai': self.score_ai,
            'score_opp': self.score_opp,
            'event': event
        }

        return self._get_observation(), reward, done, info

    def _reset_ball(self, direction=1):
        """Сброс мяча в центр"""
        self.ball_x = self.width / 2.0
        self.ball_y = self.height / 2.0
        self.ball_vx = 5.0 * direction
        self.ball_vy = random.choice([-3.0, -2.0, 2.0, 3.0])

    def draw(self, screen, offset_x=0, offset_y=0):
        """Отрисовка корта тенниса в Pygame"""
        # Задний фон корта
        court_rect = pygame.Rect(offset_x, offset_y, self.width, self.height)
        pygame.draw.rect(screen, (20, 35, 25), court_rect)
        pygame.draw.rect(screen, (100, 200, 120), court_rect, 2)

        # Центральная сетка
        for y in range(0, self.height, 20):
            pygame.draw.line(screen, (100, 200, 120), (offset_x + self.width // 2, offset_y + y), (offset_x + self.width // 2, offset_y + y + 10), 2)

        # Левая ракетка (Алгоритм) - Красный цвет
        opp_rect = pygame.Rect(offset_x + 10, offset_y + self.opp_paddle_y, self.paddle_width, self.paddle_height)
        pygame.draw.rect(screen, (255, 80, 80), opp_rect)

        # Правая ракетка (Наш ИИ) - Голубой цвет
        ai_rect = pygame.Rect(offset_x + self.width - 10 - self.paddle_width, offset_y + self.ai_paddle_y, self.paddle_width, self.paddle_height)
        pygame.draw.rect(screen, (0, 200, 255), ai_rect)

        # Мяч - Ярко-желтый
        pygame.draw.circle(screen, (255, 255, 50), (int(offset_x + self.ball_x), int(offset_y + self.ball_y)), self.ball_radius)
