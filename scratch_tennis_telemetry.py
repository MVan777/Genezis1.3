"""
Скрипт Телеметрии и Анализа работы Ассоциативного Мозга в Теннисе
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.tennis_env import TennisEnv
from core.universal_brain import UniversalAssociativeBrain

def analyze_tennis_matches(num_matches=5):
    env = TennisEnv()
    brain = UniversalAssociativeBrain(action_count=3)

    print("============================================================")
    print("📊 ТЕЛЕМЕТРИЯ РАБОТЫ АССОЦИАТИВНОГО МОЗГА ИИ В ТЕННИСЕ")
    print("============================================================")

    for m in range(1, num_matches + 1):
        obs = env.reset()
        done = False
        hits = 0
        goals = 0
        misses = 0
        rallies = 0

        while not done:
            action = brain.act(obs, explore=True)
            next_obs, reward, done, info = env.step(action)
            brain.learn(reward, next_obs, done)
            obs = next_obs

            ev = info.get('event', 'none')
            if ev == 'ai_hit':
                hits += 1
                rallies += 1
            elif ev == 'ai_goal':
                goals += 1
            elif ev == 'ai_miss':
                misses += 1

        total_neurons = sum(len(c.neurons) for c in brain.router.clusters)
        temporal_links = sum(sum(len(n.next_associations) for n in c.neurons) for c in brain.router.clusters)
        curiosity_rew = brain.stats.get('curiosity_rewards', 0.0)
        lessons = brain.stats.get('lessons', 0)

        print(f"\n🎮 МАТЧ {m}:")
        print(f"   🏆 Счет: ИИ {info['score_ai']} : {info['score_opp']} Алгоритм")
        print(f"   🎾 Успешных отбиваний ракеткой: {hits} | Пропусков: {misses} | Забито голов: {goals}")
        print(f"   🧠 Состояние Мозга: Нейронов={total_neurons}, Временных связей={temporal_links}")
        print(f"   💡 Извлечено уроков из снов: {lessons} | Любопытство: {curiosity_rew:.2f}")
        print(f"   🌐 Активных кластеров памяти: {len(brain.router.clusters)}")

    print("\n============================================================")
    print("✅ АНАЛИЗ ЗАВЕРШЕН")
    print("============================================================")

if __name__ == "__main__":
    analyze_tennis_matches(3)
