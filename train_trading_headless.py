"""
Скрипт Фонового Обучения и Бэктестинга Трейдинга (Headless Trading Trainer)
Запускает Ассоциативный ИИ Genezis 2.0 на исторических/синтетических свечных данных биржи
Выводит финансовую телеметрию: Баланс, Win Rate %, PnL, Количество сделок и сохраняет мозг в trading_brain.pkl
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

from environments.trading_env import TradingEnv
from core.universal_brain import UniversalAssociativeBrain
from core.auto_compressor import AutoMemoryCompressor

auto_compressor = AutoMemoryCompressor()
SAVE_PATH = "trading_brain.pkl"

def run_trading_backtest(max_episodes=20):
    """Фоновый бэктест и обучение Ассоциативного ИИ на биржевых данных"""
    env = TradingEnv(initial_balance=1000.0)
    brain = UniversalAssociativeBrain(action_count=3)

    if os.path.exists(SAVE_PATH):
        try:
            with open(SAVE_PATH, 'rb') as f:
                saved_brain = pickle.load(f)
                if hasattr(saved_brain, 'router'):
                    brain = saved_brain
                    print(f"  ✅ Загружен ранее сохранённый Ассоциативный Мозг Трейдинга из {SAVE_PATH}")
        except Exception as e:
            print(f"  ⚠️ Ошибка загрузки {SAVE_PATH}: {e}")

    def signal_handler(sig, frame):
        print("\n\n⚠️  Получен сигнал останова. Сохраняю состояние трейдинга...")
        save_brain(brain)
        print("✅ Сохранение завершено. Выход.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 75)
    print("📈 ФОНОВЫЙ БЭКТЕСТ И ОБУЧЕНИЕ АССОЦИАТИВНОГО ИИ НА БИРЖЕВЫХ ГРАФИКАХ")
    print("💵 Стартовый капитал: $1000.00  |  Комиссия сделки: 0.05%")
    print(f"🔄 Результаты сохраняются в '{SAVE_PATH}'")
    print("=" * 75)

    start_time = time.time()

    for ep in range(1, max_episodes + 1):
        obs = env.reset()
        done = False
        step = 0

        while not done:
            step += 1
            # Выбор торгового действия (0: HOLD, 1: BUY/LONG, 2: SELL/SHORT)
            action = brain.act(obs, explore=True)
            next_obs, reward, done, info = env.step(action)

            # Мозг обучается на PnL и сделках
            brain.learn(reward, next_obs, done)
            obs = next_obs

        brain.reset_episode()

        # Авто-компрессия каждые 3 торговые сессии
        pruned = 0
        if ep % 3 == 0:
            pruned, _ = auto_compressor.compress_brain(brain)
            if hasattr(brain, '_rebuild_neuron_cache'):
                brain._rebuild_neuron_cache()

        total_n = sum(len(c.neurons) for c in brain.router.clusters)
        bal = info['balance']
        win_rate = info['win_rate']
        trades = info['total_trades']
        wins = info['winning_trades']

        pnl_pct = ((bal - env.initial_balance) / env.initial_balance) * 100.0
        status_icon = "📈" if pnl_pct >= 0 else "📉"
        pruned_str = f" (🧹-{pruned})" if pruned > 0 else ""

        print(f"{status_icon} Эпизод {ep:02d} | Баланс: ${bal:.2f} ({pnl_pct:+.1f}%) | "
              f"Сделок: {trades:02d} | Винрейт: {win_rate:.1f}% ({wins}/{trades}) | "
              f"Нейронов: {total_n:04d}{pruned_str}")

        if ep % 5 == 0:
            save_brain(brain)

    elapsed = time.time() - start_time
    save_brain(brain)
    print("=" * 75)
    print(f"✅ БЭКТЕСТ И ОБУЧЕНИЕ ЗАВЕРШЕНЫ ЗА {elapsed:.1f} сек ({max_episodes} торговых сессий)")
    print("=" * 75)

def save_brain(brain):
    try:
        with open(SAVE_PATH, 'wb') as f:
            pickle.dump(brain, f)
    except Exception as e:
        print(f"⚠️ Ошибка сохранения: {e}")

if __name__ == "__main__":
    episodes = 15
    if len(sys.argv) > 1:
        try:
            episodes = int(sys.argv[1])
        except:
            pass
    run_trading_backtest(max_episodes=episodes)
