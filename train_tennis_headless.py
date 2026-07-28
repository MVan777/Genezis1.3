"""
Скрипт Фонового Быстрого Обучения Теннису (Headless Tennis Trainer)
Работает без графики Pygame на скорости 50x-100x
Выводит телеметрию матчей в терминал и сохраняет мозг в tennis_brain.pkl
"""

import sys
import os
import time
import pickle
import signal

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.tennis_env import TennisEnv
from core.universal_brain import UniversalAssociativeBrain
from core.auto_compressor import AutoMemoryCompressor

auto_compressor = AutoMemoryCompressor()

SAVE_PATH = "tennis_brain.pkl"

def run_headless_training(max_matches=100):
    """Фоновый прогон теннисных матчей на максимальной скорости CPU"""
    env = TennisEnv()
    brain = UniversalAssociativeBrain(action_count=3)

    # Загружаем сохраненное состояние тенниса (если есть)
    if os.path.exists(SAVE_PATH):
        try:
            with open(SAVE_PATH, 'rb') as f:
                saved_brain = pickle.load(f)
                if hasattr(saved_brain, 'router'):
                    brain = saved_brain
                    print(f"  ✅ Загружен ранее сохранённый Ассоциативный Мозг Тенниса из {SAVE_PATH}")
        except Exception as e:
            print(f"  ⚠️ Ошибка загрузки {SAVE_PATH}: {e}")

    # Обработчик сигнала Ctrl+C для безопасного сохранения
    def signal_handler(sig, frame):
        print("\n\n⚠️  Получен сигнал останова. Сохраняю состояние тенниса...")
        save_brain(brain)
        print("✅ Сохранение завершено. Выход.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 70)
    print("🚀 СУПЕР-БЫСТРОЕ ФОНОВОЕ ОБУЧЕНИЕ ТЕННИСУ (50x - 100x CPU Speed)")
    print("🎾 Алгоритмический бот (Слева)  vs  Ассоциативный ИИ (Справа)")
    print(f"🔄 Результаты сохраняются в '{SAVE_PATH}'")
    print("⌨️  Нажмите Ctrl+C в любой момент для остановки и сохранения")
    print("=" * 70)

    start_time = time.time()
    match_count = 0

    try:
        while match_count < max_matches:
            match_count += 1
            obs = env.reset()
            done = False
            hits = 0
            goals = 0
            misses = 0

            while not done:
                # Ассоциативный ИИ выбирает действие (0: stay, 1: up, 2: down)
                action = brain.act(obs, explore=True)
                next_obs, reward, done, info = env.step(action)

                # Мозг обучается на лету
                brain.learn(reward, next_obs, done)
                obs = next_obs

                ev = info.get('event', 'none')
                if ev == 'ai_hit':
                    hits += 1
                elif ev == 'ai_goal':
                    goals += 1
                elif ev == 'ai_miss':
                    misses += 1

            # Сбрасываем локальную историю эпизода
            brain.reset_episode()

            # Авто-компрессия и очистка каждые 3 матча
            pruned = 0
            if match_count % 3 == 0:
                pruned, merged = auto_compressor.compress_brain(brain)
                if hasattr(brain, '_rebuild_neuron_cache'):
                    brain._rebuild_neuron_cache()

            # Статистика нейронов и связей
            total_n = sum(len(c.neurons) for c in brain.router.clusters)
            total_c = sum(sum(len(n.next_associations) for n in c.neurons) for c in brain.router.clusters)
            lessons = brain.stats.get('lessons', 0)

            # Форматированный вывод матча в терминал
            win_symbol = "🏆" if info['score_ai'] > info['score_opp'] else "📊"
            pruned_str = f" (🧹-{pruned})" if pruned > 0 else ""
            tot_shots = hits + misses
            hit_pct = (hits / tot_shots * 100.0) if tot_shots > 0 else 0.0
            print(f"{win_symbol} Матч {match_count:03d} | ИИ {info['score_ai']:02d} : {info['score_opp']:02d} Алгоритм | "
                  f"Отбито: {hits:02d} | Пропущено: {misses:02d} | Точность: {hit_pct:.1f}% | "
                  f"Нейронов: {total_n:04d}{pruned_str} | Связей: {total_c:04d} | Уроков: {lessons:03d}")

            # Сохраняем мозг каждые 5 матчей
            if match_count % 5 == 0:
                save_brain(brain)

    except KeyboardInterrupt:
        pass

    elapsed = time.time() - start_time
    save_brain(brain)
    print("=" * 70)
    print(f"✅ ФОНОВОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО ЗА {elapsed:.1f} сек ({match_count} матчей)")
    print(f"💡 Для просмотра игры в реальном времени запустите: python main_tennis.py")
    print("=" * 70)

def save_brain(brain):
    """Сохранить состояние мозга в pickle файл"""
    try:
        with open(SAVE_PATH, 'wb') as f:
            pickle.dump(brain, f)
    except Exception as e:
        print(f"⚠️ Ошибка сохранения: {e}")

if __name__ == "__main__":
    matches = 200
    if len(sys.argv) > 1:
        try:
            matches = int(sys.argv[1])
        except:
            pass
    run_headless_training(max_matches=matches)
