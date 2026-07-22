"""
Визуализация нейронов - показывает нейроны и их связи
"""

import pygame
import random
from config import *

class NeuronVisualizer:
    def __init__(self, screen, font):
        self.screen = screen
        self.font = font
        self.last_update = 0
        self.frame_counter = 0
        self.neuron_positions = {}  # id -> (x, y)

        self.cluster_colors = [
            (100, 100, 255),  # синий
            (100, 255, 100),  # зелёный
            (255, 100, 100),  # красный
            (255, 255, 100),  # жёлтый
            (255, 100, 255),  # розовый
            (100, 255, 255),  # голубой
        ]

    def draw(self, agent, current_state=None, current_action=None, similar_neurons=None):
        """Отрисовка нейронов и связей между ними"""
        self.frame_counter += 1

        router = agent.router if hasattr(agent, 'router') else None
        active_cluster = agent.active_cluster if hasattr(agent, 'active_cluster') else None

        if not router or not router.clusters:
            text = self.font.render("Нет кластеров памяти", True, COLORS['text'])
            self.screen.blit(text, (NEURON_VIS_X + 50, 50))
            return

        # Фон
        bg_rect = pygame.Rect(NEURON_VIS_X - 10, 0, NEURON_VIS_WIDTH, NEURON_VIS_HEIGHT)
        pygame.draw.rect(self.screen, COLORS['neuron_bg'], bg_rect)

        # Собираем все нейроны из всех кластеров
        all_neurons = []
        neuron_map = {}  # id -> neuron_data

        for cluster in router.clusters:
            if not cluster.neurons:
                continue

            color_idx = cluster.id % len(self.cluster_colors)
            cluster_color = self.cluster_colors[color_idx]

            for neuron in cluster.neurons[:100]:  # показываем до 100 нейронов на кластер
                # Генерируем или получаем позицию
                if neuron.id not in self.neuron_positions:
                    # Разбрасываем нейроны по всему пространству
                    self.neuron_positions[neuron.id] = (
                        random.randint(50, NEURON_VIS_WIDTH - 50),
                        random.randint(50, NEURON_VIS_HEIGHT - 50)
                    )

                x, y = self.neuron_positions[neuron.id]
                neuron_data = {
                    'id': neuron.id,
                    'neuron': neuron,
                    'cluster_id': cluster.id,
                    'color': cluster_color,
                    'pos': (x, y),
                    'screen_x': NEURON_VIS_X + x,
                    'screen_y': y
                }
                all_neurons.append(neuron_data)
                neuron_map[neuron.id] = neuron_data

        # ===== РИСУЕМ СВЯЗИ МЕЖДУ НЕЙРОНАМИ =====
        # Собираем все связи из всех кластеров
        connections = []

        for cluster in router.clusters:
            if hasattr(cluster, 'connection_matrix'):
                for (id1, id2), strength in cluster.connection_matrix.items():
                    if strength > 0.1 and id1 in neuron_map and id2 in neuron_map:
                        connections.append((id1, id2, strength))

        # Сортируем по силе и рисуем
        connections.sort(key=lambda x: x[2], reverse=True)

        for id1, id2, strength in connections[:500]:  # максимум 500 связей
            n1 = neuron_map[id1]
            n2 = neuron_map[id2]

            # Цвет от серого до белого в зависимости от силы
            color_val = min(200, int(150 + strength * 100))
            color = (color_val, color_val, color_val)
            width = max(1, int(strength * 2))

            pygame.draw.line(
                self.screen, color,
                (n1['screen_x'], n1['screen_y']),
                (n2['screen_x'], n2['screen_y']),
                width
            )

        # ===== РИСУЕМ НЕЙРОНЫ =====
        # Определяем, какие нейроны похожи на текущую ситуацию
        similar_ids = []
        if similar_neurons:
            similar_ids = [n[0].id if len(n) > 0 else None for n in similar_neurons if n]

        for neuron_data in all_neurons:
            neuron = neuron_data['neuron']
            screen_x = neuron_data['screen_x']
            screen_y = neuron_data['screen_y']

            # Размер зависит от силы нейрона
            size = 3 + int(neuron.strength * 8)

            # Обводка белым для активных нейронов
            if neuron.id in similar_ids:
                pygame.draw.circle(self.screen, (255, 255, 255), (screen_x, screen_y), size + 3)

            # Сам нейрон
            pygame.draw.circle(self.screen, neuron_data['color'], (screen_x, screen_y), size)

            # Обводка цветом действия
            pygame.draw.circle(self.screen, ACTION_COLORS[neuron.action], (screen_x, screen_y), size, 2)

        # Статистика
        self._draw_stats(router, active_cluster, current_action, len(connections))

    def _draw_stats(self, router, active_cluster, current_action, connection_count):
        """Статистика на панели"""
        y_offset = 10
        texts = [
            f"Кластеров: {len(router.clusters)}",
            f"Связей: {connection_count}",
            f"Активный: {active_cluster.domain if active_cluster else 'нет'}",
            f"Нейронов: {sum(len(c.neurons) for c in router.clusters)}",
        ]

        for i, text in enumerate(texts):
            text_surface = self.font.render(text, True, COLORS['text'])
            self.screen.blit(text_surface, (NEURON_VIS_X + 10, y_offset + i * 25))

        if current_action is not None:
            action_text = f"Действие: {ACTIONS[current_action]}"
            text_surface = self.font.render(action_text, True, ACTION_COLORS[current_action])
            self.screen.blit(text_surface, (NEURON_VIS_X + 10, y_offset + 150))