"""
Панель управления для визуализации
"""

import pygame

class Button:
    """Кнопка для панели управления"""
    def __init__(self, x, y, width, height, text, color=(100,100,100), hover_color=(150,150,150)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.is_hovered = False
        self.is_visible = True

    def draw(self, screen, font):
        if not self.is_visible:
            return

        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (200,200,200), self.rect, 2)

        text_surface = font.render(self.text, True, (255,255,255))
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        if not self.is_visible:
            return False

        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)

        if event.type == pygame.MOUSEBUTTONDOWN and self.is_hovered:
            return True
        return False


class Slider:
    """Слайдер для регулировки параметров"""
    def __init__(self, x, y, width, min_val, max_val, default, text):
        self.rect = pygame.Rect(x, y, width, 10)
        self.handle_rect = pygame.Rect(x + (default-min_val)/(max_val-min_val)*width - 5, y-5, 10, 20)
        self.min_val = min_val
        self.max_val = max_val
        self.value = default
        self.text = text
        self.dragging = False

    def draw(self, screen, font):
        # Линия
        pygame.draw.rect(screen, (100,100,100), self.rect)
        pygame.draw.rect(screen, (200,200,200), self.rect, 1)

        # Ползунок
        color = (150,150,255) if self.dragging else (200,200,200)
        pygame.draw.rect(screen, color, self.handle_rect)

        # Текст
        text_surface = font.render(f"{self.text}: {self.value:.2f}", True, (255,255,255))
        screen.blit(text_surface, (self.rect.x, self.rect.y - 20))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.handle_rect.collidepoint(event.pos):
                self.dragging = True
                return True

        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False

        elif event.type == pygame.MOUSEMOTION and self.dragging:
            new_x = max(self.rect.x, min(self.rect.x + self.rect.width, event.pos[0]))
            self.handle_rect.x = new_x - 5
            self.value = self.min_val + (new_x - self.rect.x) / self.rect.width * (self.max_val - self.min_val)
            return True
        return False


class ControlPanel:
    """Панель управления"""

    def __init__(self, screen, font, x, y, width, height):
        self.screen = screen
        self.font = font
        self.rect = pygame.Rect(x, y, width, height)
        self.visible = True

        # Кнопки управления
        button_y = y + 10
        self.buttons = {
            'start': Button(x + 10, button_y, 50, 25, "▶", (50,150,50)),
            'pause': Button(x + 70, button_y, 50, 25, "⏸️", (150,150,50)),
            'save': Button(x + 130, button_y, 50, 25, "💾", (50,50,150)),
            'analyze': Button(x + 190, button_y, 50, 25, "🔍", (150,50,150)),
            'compress': Button(x + 250, button_y, 50, 25, "🗜️", (100,100,100)),
        }

        # Кнопки ресурсов
        resource_y = button_y + 35
        self.resource_buttons = {
            'elixir': Button(x + 10, resource_y, 50, 25, "💊", (50,150,50)),
            'antidote': Button(x + 70, resource_y, 50, 25, "💉", (50,50,150)),
            'poison': Button(x + 130, resource_y, 50, 25, "☠️", (150,50,50)),
            'enemy': Button(x + 190, resource_y, 50, 25, "👹", (150,50,150)),
        }

        # Слайдеры
        slider_y = resource_y + 40
        self.sliders = {
            'explore': Slider(x + 10, slider_y, 200, 0.0, 1.0, 1.0, "Исследование"),
            'speed': Slider(x + 10, slider_y + 50, 200, 1, 60, 30, "Скорость"),
        }

        self.stats_y = slider_y + 120

    def draw(self, agent, game, bank, total_neurons=0, total_connections=0):
        if not self.visible:
            return

        # Фон панели
        s = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        s.fill((30, 30, 30, 200))
        self.screen.blit(s, (self.rect.x, self.rect.y))
        pygame.draw.rect(self.screen, (100,100,100), self.rect, 2)

        # Заголовок
        title = self.font.render("ПАНЕЛЬ", True, (255,255,255))
        self.screen.blit(title, (self.rect.x + 10, self.rect.y + 5))

        # Кнопки управления
        for btn in self.buttons.values():
            btn.draw(self.screen, self.font)

        # Кнопки ресурсов
        for btn in self.resource_buttons.values():
            btn.draw(self.screen, self.font)

        # Слайдеры
        for slider in self.sliders.values():
            slider.draw(self.screen, self.font)

        # Статистика
        self._draw_stats(agent, game, bank, total_neurons, total_connections)

    def _draw_stats(self, agent, game, bank, total_neurons, total_connections):
        y = self.rect.y + self.stats_y

        # Статистика игры и агента
        stats = [
            f"Счёт: {game.episode_reward if hasattr(game, 'episode_reward') else 0:.1f}",
            f"Здоровье: {game.health if hasattr(game, 'health') else 0}",
            f"Нейронов: {total_neurons}",
            f"Связей: {total_connections}",
            f"Кластеров: {len(agent.router.clusters) if hasattr(agent, 'router') else 0}",
            f"Симуляций: {agent.stats.get('simulations_run', 0)}",
            f"Уроков: {agent.stats.get('lessons_learned', 0)}",
        ]

        for i, stat in enumerate(stats):
            text = self.font.render(stat, True, (200,200,200))
            self.screen.blit(text, (self.rect.x + 10, y + i * 20))

        # Банк элиты
        if bank and bank.agents:
            y += len(stats) * 20 + 10
            title = self.font.render("БАНК ЭЛИТЫ:", True, (255,255,0))
            self.screen.blit(title, (self.rect.x + 10, y))

            for i, a in enumerate(bank.agents[:5]):
                score_text = self.font.render(f"#{i+1}: {a['score']:.2f}", True, (200,200,200))
                self.screen.blit(score_text, (self.rect.x + 20, y + 20 + i * 20))

    def handle_event(self, event):
        if not self.visible:
            return {}

        result = {}

        for key, btn in self.buttons.items():
            if btn.handle_event(event):
                result[key] = True

        for key, btn in self.resource_buttons.items():
            if btn.handle_event(event):
                result[f'add_{key}'] = True

        for key, slider in self.sliders.items():
            if slider.handle_event(event):
                result[key] = slider.value

        return result