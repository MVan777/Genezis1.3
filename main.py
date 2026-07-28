"""
Главный файл - бесконечное обучение с автосохранением
"""

import pygame
import time
import numpy as np
from copy import deepcopy
import random
import os
import pickle
import signal
import sys

from config import *
from game.game import Game, FOOD, SAFE_ZONE
from agent.agent import Agent
from evolution.evolution import Evolution
from visualization.visualizer import NeuronVisualizer
from visualization.control_panel import ControlPanel
from elite_bank import EliteBank
from core.router import Router
from core.cluster import MemoryCluster
from core.cluster_connections import ClusterConnections

# ============================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ДЛЯ СОХРАНЕНИЯ
# ============================================
SAVE_FILES = {
    'clusters': 'clusters.pkl',
    'connections': 'connections.pkl',
    'agent': 'agent_state.pkl',
    'bank': 'elite_bank.pkl',
    'stats': 'training_stats.pkl'
}

AUTOSAVE_INTERVAL = 300  # секунд между автосохранениями (5 минут)
GAMES_PER_SESSION = 10    # игр перед паузой/сохранением

# ============================================
# ФУНКЦИИ СОХРАНЕНИЯ/ЗАГРУЗКИ
# ============================================

def save_all(agent, bank, stats):
    """Сохранить всё состояние"""
    print(f"\n💾 АВТОСОХРАНЕНИЕ...")

    # Сохраняем кластеры
    if hasattr(agent, 'router'):
        clusters_data = agent.router.save_all()
        with open(SAVE_FILES['clusters'], 'wb') as f:
            pickle.dump(clusters_data, f)

    # Сохраняем связи
    if hasattr(agent, 'router') and hasattr(agent.router, 'connections'):
        with open(SAVE_FILES['connections'], 'wb') as f:
            pickle.dump(agent.router.connections.to_dict(), f)

    # Сохраняем состояние агента
    agent_data = {
        'stats': agent.stats,
        'death_situations': agent.death_situations[-100:] if hasattr(agent, 'death_situations') else [],
        'game_history': agent.game_history[-1000:] if hasattr(agent, 'game_history') else [],
        'simulations_run': agent.stats.get('simulations_run', 0),
        'lessons_learned': agent.stats.get('lessons_learned', 0)
    }
    with open(SAVE_FILES['agent'], 'wb') as f:
        pickle.dump(agent_data, f)

    # Сохраняем банк
    if bank:
        bank.save()

    # Сохраняем статистику
    with open(SAVE_FILES['stats'], 'wb') as f:
        pickle.dump(stats, f)

    print(f"✅ Всё сохранено")


def load_all():
    """Загрузить всё состояние"""
    print(f"\n📀 ЗАГРУЗКА СОСТОЯНИЯ...")

    # Создаём агента
    agent = Agent()

    # Загружаем банк
    bank = EliteBank()
    bank.load()

    # Собираем все нейроны из банка
    neurons_by_id = {}
    if bank and bank.agents:
        print(f"  📦 Сбор нейронов из банка...")
        for agent_data in bank.agents:
            agent_obj = agent_data['agent']
            if hasattr(agent_obj, 'router') and agent_obj.router:
                for cluster in agent_obj.router.clusters:
                    for neuron in cluster.neurons:
                        neurons_by_id[neuron.id] = neuron
        print(f"  ✅ Загружено {len(neurons_by_id)} нейронов")

    # Загружаем кластеры и связи
    if os.path.exists(SAVE_FILES['clusters']) and os.path.exists(SAVE_FILES['connections']):
        try:
            with open(SAVE_FILES['clusters'], 'rb') as f:
                clusters_data = pickle.load(f)
            with open(SAVE_FILES['connections'], 'rb') as f:
                connections_data = pickle.load(f)

            if hasattr(agent, 'router'):
                agent.router.load_all(clusters_data, neurons_by_id)
                agent.router.connections.from_dict(connections_data)
            print(f"  ✅ Загружены кластеры ({len(agent.router.clusters)} кластеров)")
        except Exception as e:
            print(f"  ⚠️ Ошибка загрузки кластеров: {e}")


    # Загружаем статистику
    stats = {
        'total_games': 0,
        'total_simulations': 0,
        'total_lessons': 0,
        'start_time': time.time(),
        'best_score': 0
    }

    if os.path.exists(SAVE_FILES['stats']):
        try:
            with open(SAVE_FILES['stats'], 'rb') as f:
                stats = pickle.load(f)
            print(f"  ✅ Загружена статистика")
        except:
            pass

    # Загружаем состояние агента
    if os.path.exists(SAVE_FILES['agent']):
        try:
            with open(SAVE_FILES['agent'], 'rb') as f:
                agent_data = pickle.load(f)
                agent.stats.update(agent_data.get('stats', {}))
                if hasattr(agent, 'death_situations'):
                    agent.death_situations = agent_data.get('death_situations', [])
                if hasattr(agent, 'game_history'):
                    agent.game_history = agent_data.get('game_history', [])
            print(f"  ✅ Загружен агент")
        except:
            pass

    print(f"✅ Загрузка завершена")
    return agent, bank, stats


# ============================================
# ФУНКЦИЯ ПОДСЧЁТА СТАТИСТИКИ НЕЙРОНОВ
# ============================================
def get_neuron_stats(agent):
    """Получить полную статистику по нейронам и всем типам связей (временным и пространственным)"""
    if not hasattr(agent, 'router') or not agent.router:
        return 0, 0

    total_neurons = 0
    total_connections = 0

    for cluster in agent.router.clusters:
        total_neurons += len(cluster.neurons)
        # Пространственные связи элементов
        if hasattr(cluster, 'connection_matrix'):
            total_connections += len(cluster.connection_matrix)
        # Последовательные временные связи (N_t-1 -> N_t)
        for neuron in cluster.neurons:
            if hasattr(neuron, 'next_associations'):
                total_connections += len(neuron.next_associations)

    # Межкластерные связи
    if hasattr(agent.router, 'connections') and hasattr(agent.router.connections, 'matrix'):
        total_connections += len(agent.router.connections.matrix)

    return total_neurons, total_connections


# ============================================
# ОБРАБОТКА ЗАКРЫТИЯ
# ============================================
def signal_handler(sig, frame, agent, bank, stats):
    """Обработка Ctrl+C и закрытия"""
    print("\n\n⚠️  Получен сигнал завершения. Сохраняю...")
    save_all(agent, bank, stats)
    sys.exit(0)


# ============================================
# ОСНОВНАЯ ФУНКЦИЯ - БЕСКОНЕЧНОЕ ОБУЧЕНИЕ
# ============================================
def main():
    print("="*60)
    print("АССОЦИАТИВНЫЙ ИИ - БЕСКОНЕЧНОЕ ОБУЧЕНИЕ")
    print("="*60)
    print("\n🟢 Агент живёт постоянно. Закрытие окна = пауза + сохранение")
    print(f"🔄 Автосохранение каждые {AUTOSAVE_INTERVAL} секунд")
    print(f"🎮 Сессии по {GAMES_PER_SESSION} игр\n")

    # Загружаем состояние
    agent, bank, stats = load_all()

    # ===== ИНИЦИАЛИЗАЦИЯ РЕКОРДОВ =====
    if bank and bank.agents:
        best_absolute_score = bank.agents[0]['score']
        print(f"🏆 Текущий рекорд: {best_absolute_score:.2f}")
    else:
        best_absolute_score = 0
        print("🏆 Рекордов пока нет")

    session_best = 0
    # ===================================

    # Получаем статистику нейронов
    total_neurons, total_connections = get_neuron_stats(agent)

    # Регистрируем обработчик закрытия
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, agent, bank, stats))

    # Инициализация pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Ассоциативный ИИ - Бесконечное обучение")
    font = pygame.font.Font(None, 20)
    clock = pygame.time.Clock()

    visualizer = NeuronVisualizer(screen, font)

    # ===== ПАНЕЛЬ УПРАВЛЕНИЯ =====
    # Рассчитываем позицию: после информационной панели
    control_panel_x = GAME_WIDTH + INFO_PANEL_WIDTH + 20
    control_panel = ControlPanel(screen, font, control_panel_x, 10, 280, 400)
    # ==============================

    # Счётчики
    games_played = stats.get('total_games', 0)
    last_save_time = time.time()
    paused = False
    current_speed = 30
    explore_rate = 1.0

    print(f"\n📊 Статистика на старте:")
    print(f"  Всего игр сыграно: {games_played}")
    print(f"  Кластеров: {len(agent.router.clusters) if hasattr(agent, 'router') else 0}")
    print(f"  Нейронов: {total_neurons}")
    print(f"  Связей: {total_connections}")
    print(f"  Симуляций проведено: {agent.stats.get('simulations_run', 0)}")
    print(f"  Уроков извлечено: {agent.stats.get('lessons_learned', 0)}")
    print(f"  Лучший рекорд: {best_absolute_score:.2f}")

    # ===== БЕСКОНЕЧНЫЙ ЦИКЛ =====
    while True:
        # Сброс лучшего счёта сессии
        session_best = 0

        # Играем сессию игр
        for game_num in range(GAMES_PER_SESSION):
            print(f"\n{'='*50}")
            print(f"🎮 ИГРА {games_played + 1} (сессия {game_num+1}/{GAMES_PER_SESSION})")
            print(f"{'='*50}")

            # Создаём игру
            game = Game(screen, font)
            state = game.reset()
            agent.new_game()

            game_reward = 0
            steps = 0
            running = True

            # Одна игра до смерти
            while game.alive and running:
                # Обработка событий (закрытие окна)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\n👋 Закрытие окна. Сохраняю и выключаю...")
                        save_all(agent, bank, stats)
                        pygame.quit()
                        return

                    # Обработка панели управления
                    panel_result = control_panel.handle_event(event)

                    if 'pause' in panel_result:
                        paused = not paused
                    if 'save' in panel_result:
                        save_all(agent, bank, stats)
                    if 'speed' in panel_result:
                        current_speed = int(panel_result['speed'])
                    if 'explore' in panel_result:
                        explore_rate = panel_result['explore']
                    if 'analyze' in panel_result:
                        agent.analyze_and_simulate()
                    if 'compress' in panel_result:
                        total = 0
                        for cluster in agent.router.clusters:
                            total += cluster.compress()
                        print(f"  🗜️ Сжато {total} нейронов")

                    # Кнопки добавления ресурсов
                    if 'add_elixir' in panel_result:
                        if hasattr(game, 'add_resource'):
                            game.add_resource(ELIXIR)
                    if 'add_antidote' in panel_result:
                        if hasattr(game, 'add_resource'):
                            game.add_resource(ANTIDOTE)
                    if 'add_poison' in panel_result:
                        if hasattr(game, 'add_resource'):
                            game.add_resource(POISON)
                    if 'add_enemy' in panel_result:
                        if hasattr(game, 'add_enemy'):
                            game.add_enemy()

                if paused:
                    clock.tick(10)
                    continue

                # Шаг игры
                explore = random.random() < explore_rate
                action = agent.act(state, explore=explore)
                new_state, reward, health, alive, event_text = game.step(action)
                agent.learn(reward, new_state, not alive)

                state = new_state
                game_reward += reward
                steps += 1
                game.step_count = steps

                # Получаем актуальную статистику нейронов
                total_neurons, total_connections = get_neuron_stats(agent)

                # Отрисовка
                screen.fill(COLORS['background'])
                game.draw(screen, font)  # информационная панель рисуется здесь
                control_panel.draw(agent, game, bank, total_neurons, total_connections)
                visualizer.draw(agent, state, action, agent.last_similar_neurons)

                # Статус
                status_text = font.render(
                    f"Игр: {games_played+1} | Шаг: {steps} | Награда: {game_reward:.1f} | Скорость: {current_speed}",
                    True, (255,255,0)
                )
                screen.blit(status_text, (10, 10))

                pygame.display.flip()
                clock.tick(current_speed)

            # Игра закончена
            games_played += 1
            stats['total_games'] = games_played

            print(f"  📊 Итог: {game_reward:.2f}, шагов: {steps}")
            print(f"  🧠 Симуляций всего: {agent.stats.get('simulations_run', 0)}")
            print(f"  📚 Уроков всего: {agent.stats.get('lessons_learned', 0)}")

            # ===== СОХРАНЕНИЕ В БАНК ЭЛИТЫ =====
            if game_reward > 5:  # сохраняем только приличные результаты
                bank.add_agent(agent, game_reward, games_played)
                print(f"  🏆 Агент добавлен в банк со счётом {game_reward:.2f}")

            # Проверка рекорда сессии
            if game_reward > session_best:
                session_best = game_reward

            # Проверка абсолютного рекорда
            if game_reward > best_absolute_score:
                best_absolute_score = game_reward
                stats['best_score'] = best_absolute_score
                print(f"  👑 НОВЫЙ АБСОЛЮТНЫЙ РЕКОРД! {game_reward:.2f}")
                # Сохраняем чемпиона отдельно
                with open("best_agent_ever.pkl", 'wb') as f:
                    pickle.dump(agent, f)
            # ===================================

            # Автосохранение по времени
            if time.time() - last_save_time > AUTOSAVE_INTERVAL:
                save_all(agent, bank, stats)
                last_save_time = time.time()

        # Конец сессии
        print(f"\n📦 Сессия из {GAMES_PER_SESSION} игр завершена")
        print(f"🏆 Лучший счёт сессии: {session_best:.2f}")
        print(f"🏆 Абсолютный рекорд: {best_absolute_score:.2f}")

        # Показываем банк элиты
        if bank and bank.agents:
            print(f"\n🏆 БАНК ЭЛИТЫ (топ-5):")
            for i, a in enumerate(bank.agents[:5]):
                print(f"  #{i+1}: {a['score']:.2f}")

        save_all(agent, bank, stats)
        last_save_time = time.time()

        # Небольшая пауза между сессиями
        print("⏸️  Пауза 5 секунд перед следующей сессией...")
        time.sleep(5)


if __name__ == "__main__":
    main()