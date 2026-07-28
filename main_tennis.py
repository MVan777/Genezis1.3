"""
Скрипт Запуска Теннисного Матча (Tennis Dual Launcher)
Левая сторона: Алгоритмический правило-бот
Правая сторона: Наш Универсальный Ассоциативный ИИ (UniversalAssociativeBrain)
"""

import sys
import os
import pygame
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.tennis_env import TennisEnv
from core.universal_brain import UniversalAssociativeBrain
from visualization.visualizer import NeuronVisualizer
from config import COLORS

def run_tennis_gui():
    """Запуск матча Теннис с графикой Pygame"""
    pygame.init()

    court_width = 600
    court_height = 400
    neuron_vis_width = 500
    window_width = court_width + neuron_vis_width
    window_height = max(court_height + 80, 600)

    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("🎾 Ассоциативный ИИ vs Алгоритм - Дуэль в Теннис")
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()

    env = TennisEnv(width=court_width, height=court_height)
    brain = UniversalAssociativeBrain(action_count=3)  # 0: STAY, 1: UP, 2: DOWN
    visualizer = NeuronVisualizer(screen, font)

    import pickle
    tennis_save_path = "tennis_brain.pkl"
    if os.path.exists(tennis_save_path):
        try:
            with open(tennis_save_path, 'rb') as f:
                saved_brain = pickle.load(f)
                if hasattr(saved_brain, 'router'):
                    brain = saved_brain
                    print(f"  ✅ Загружен сохраненный Ассоциативный Мозг Тенниса из {tennis_save_path}")
        except Exception as e:
            print(f"  ⚠️ Ошибка загрузки тенниса: {e}")

    obs = env.reset()
    running = True
    episodes = 0
    total_ai_score = 0
    total_opp_score = 0

    speed_mode = 1  # 1: 1x (60 FPS), 2: 3x (180 FPS), 3: 6x (360 FPS), 4: TURBO MAX
    speed_labels = {1: "1x", 2: "3x", 3: "6x", 4: "⚡ TURBO MAX"}
    frame_count = 0

    print("=" * 60)
    print("🎾 ТЕННИСНЫЙ МАТЧ ЗАПУЩЕН: Алгоритм (Слева) vs Ассоциативный ИИ (Справа)")
    print("⌨️  Клавиши управления скоростью: [1] = 1x, [2] = 3x, [3] = 6x, [4 / Пробел] = TURBO MAX")
    print("=" * 60)

    while running:
        frame_count += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    speed_mode = 1
                elif event.key == pygame.K_2:
                    speed_mode = 2
                elif event.key == pygame.K_3:
                    speed_mode = 3
                elif event.key == pygame.K_4 or event.key == pygame.K_SPACE:
                    speed_mode = 4

        # Ассоциативный ИИ выбирает действие (0: stay, 1: up, 2: down)
        action = brain.act(obs, explore=True)
        next_obs, reward, done, info = env.step(action)

        # Мозг обучается на лету
        brain.learn(reward, next_obs, done)
        obs = next_obs

        # Отрисовка с пропуском кадров в Турбо-режиме для гипер-скорости
        should_render = (speed_mode != 4) or (frame_count % 5 == 0)

        if should_render:
            screen.fill(COLORS['background'])
            env.draw(screen, offset_x=10, offset_y=50)

            # Вывод текущего счета
            score_text = f"🤖 Алгоритм: {info['score_opp']}   vs   🧠 Ассоциативный ИИ: {info['score_ai']}"
            score_surface = font.render(score_text, True, (255, 255, 255))
            screen.blit(score_surface, (20, 15))

            # Отрисовка активной нейронной памяти на правой панели
            similar_neurons = getattr(brain, 'last_similar_neurons', None)
            visualizer.draw(brain, obs, action, similar_neurons, speed_label=speed_labels[speed_mode])

            pygame.display.flip()

        # Регулировка FPS в зависимости от выбранного режима
        if speed_mode == 1:
            clock.tick(60)
        elif speed_mode == 2:
            clock.tick(180)
        elif speed_mode == 3:
            clock.tick(360)
        elif speed_mode == 4:
            clock.tick(0)  # Без ограничений!

        if done:
            episodes += 1
            total_ai_score += info['score_ai']
            total_opp_score += info['score_opp']
            
            # Сохраняем состояние тенниса
            try:
                with open(tennis_save_path, 'wb') as f:
                    pickle.dump(brain, f)
            except:
                pass

            total_n = sum(len(c.neurons) for c in brain.router.clusters)
            total_c = sum(sum(len(n.next_associations) for n in c.neurons) for c in brain.router.clusters)
            print(f"🏆 Матч {episodes} завершен | ИИ: {info['score_ai']} vs Алгоритм: {info['score_opp']} | Нейронов: {total_n} | Связей: {total_c}")
            obs = env.reset()

    pygame.quit()
    print("Матч окончен.")

if __name__ == "__main__":
    run_tennis_gui()
