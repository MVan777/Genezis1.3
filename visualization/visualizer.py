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

    def draw(self, agent, current_state=None, current_action=None, similar_neurons=None, speed_label="1x"):
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

        # ===== РИСУЕМ ВРЕМЕННЫЕ ПОСЛЕДОВАТЕЛЬНЫЕ СВЯЗИ (N_{t-1} -> N_t) =====
        for n1_id, n1_data in neuron_map.items():
            n1 = n1_data['neuron']
            if hasattr(n1, 'next_associations') and n1.next_associations:
                for n2_id, weight in n1.next_associations.items():
                    if n2_id in neuron_map and weight > 0.1:
                        n2_data = neuron_map[n2_id]
                        # Бирюзовые линии для временных связей
                        line_color = (0, 200, 220)
                        width = max(1, int(weight * 3))
                        pygame.draw.line(
                            self.screen, line_color,
                            (n1_data['screen_x'], n1_data['screen_y']),
                            (n2_data['screen_x'], n2_data['screen_y']),
                            width
                        )
                        # Точка в направлении назначения
                        mid_x = int(0.7 * n2_data['screen_x'] + 0.3 * n1_data['screen_x'])
                        mid_y = int(0.7 * n2_data['screen_y'] + 0.3 * n1_data['screen_y'])
                        pygame.draw.circle(self.screen, (0, 255, 255), (mid_x, mid_y), 2)

        # ===== РИСУЕМ СВЯЗИ СХОДСТВА МЕЖДУ НЕЙРОНАМИ =====
        connections = []
        for cluster in router.clusters:
            if hasattr(cluster, 'connection_matrix'):
                for (id1, id2), strength in cluster.connection_matrix.items():
                    if strength > 0.1 and id1 in neuron_map and id2 in neuron_map:
                        connections.append((id1, id2, strength))

        connections.sort(key=lambda x: x[2], reverse=True)

        for id1, id2, strength in connections[:500]:  # максимум 500 связей
            n1 = neuron_map[id1]
            n2 = neuron_map[id2]

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
        similar_ids = []
        if similar_neurons:
            similar_ids = [n[0].id if len(n) > 0 else None for n in similar_neurons if n]

        for neuron_data in all_neurons:
            neuron = neuron_data['neuron']
            screen_x = neuron_data['screen_x']
            screen_y = neuron_data['screen_y']

            size = 3 + int(neuron.strength * 8)

            # Обводка золотом для прототипов / долгосрочных нейронов
            if getattr(neuron, 'confidence', 0) > 0.5 or neuron.usage_count > 5:
                pygame.draw.circle(self.screen, (255, 215, 0), (screen_x, screen_y), size + 4, 1)

            # Обводка белым для активных нейронов
            if neuron.id in similar_ids:
                pygame.draw.circle(self.screen, (255, 255, 255), (screen_x, screen_y), size + 3)

            # Сам нейрон
            pygame.draw.circle(self.screen, neuron_data['color'], (screen_x, screen_y), size)

            # Обводка цветом действия
            pygame.draw.circle(self.screen, ACTION_COLORS[neuron.action], (screen_x, screen_y), size, 2)

        # Считаем все типы связей (пространственные + временные)
        total_connections = len(connections)
        for n_data in all_neurons:
            n_obj = n_data['neuron']
            if hasattr(n_obj, 'next_associations'):
                total_connections += len(n_obj.next_associations)

        # Статистика
        self._draw_stats(agent, router, active_cluster, current_action, total_connections, speed_label)

    def _draw_stats(self, agent, router, active_cluster, current_action, connection_count, speed_label="1x"):
        """Статистика на панели с макро-целями, эмоциями и скоростью"""
        y_offset = 10
        goal_name = agent.goal_system.get_goal_name() if hasattr(agent, 'goal_system') else "исследование"
        emotions = agent.emotions if hasattr(agent, 'emotions') else {}
        top_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else "normal"

        texts = [
            f"Скорость: {speed_label} [Keys: 1,2,3,4]",
            f"Кластеров: {len(router.clusters)}",
            f"Связей: {connection_count}",
            f"Домен: {active_cluster.domain if active_cluster else 'нет'}",
            f"Нейронов: {sum(len(c.neurons) for c in router.clusters)}",
            f"Цель: {goal_name}",
            f"Эмоция: {top_emotion}"
        ]

        for i, text in enumerate(texts):
            text_surface = self.font.render(text, True, COLORS['text'])
            self.screen.blit(text_surface, (NEURON_VIS_X + 10, y_offset + i * 22))

        if current_action is not None:
            action_text = f"Действие: {ACTIONS[current_action]}"
            text_surface = self.font.render(action_text, True, ACTION_COLORS[current_action])
            self.screen.blit(text_surface, (NEURON_VIS_X + 10, y_offset + 160))