"""
Скрипт Автоматического Фонового Обучения Диалогового ИИ (Headless Dialogue Trainer)
Обучает Ассоциативный Мозг на словаре вопросов и ответов с сохранением в dialogue_brain.pkl
"""

import sys
import os
import time
import pickle
import signal

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.text_dialogue_env import TextDialogueEnv
from core.universal_brain import UniversalAssociativeBrain
from core.auto_compressor import AutoMemoryCompressor

auto_compressor = AutoMemoryCompressor()
SAVE_PATH = "dialogue_brain.pkl"

def run_headless_dialogue_training(rounds=100):
    """Автоматический прогон обучения на словаре диалогов"""
    env = TextDialogueEnv()
    brain = UniversalAssociativeBrain(action_count=env.action_count)

    if os.path.exists(SAVE_PATH):
        try:
            with open(SAVE_PATH, 'rb') as f:
                saved_brain = pickle.load(f)
                if hasattr(saved_brain, 'router'):
                    brain = saved_brain
                    print(f"  ✅ Загружен ранее сохранённый Разговорный Мозг из '{SAVE_PATH}'")
        except Exception as e:
            print(f"  ⚠️ Ошибка загрузки {SAVE_PATH}: {e}")

    def signal_handler(sig, frame):
        print("\n\n⚠️ Получен сигнал останова. Сохраняю состояние разговорного мозга...")
        save_brain(brain)
        print("✅ Сохранение завершено. Выход.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 75)
    print("🗣️ СУПЕР-БЫСТРОЕ ФОНОВОЕ ОБУЧЕНИЕ РАЗГОВОРНОГО ИИ (Genezis Dialogue 2.0)")
    print(f"📚 Словарь содержит {len(env.dialogues)} пар вопросов и ответов ({env.action_count} уникальных ответов)")
    print(f"💾 Результаты сохраняются в '{SAVE_PATH}'")
    print("=" * 75)

    start_time = time.time()
    correct_count = 0

    for r in range(1, rounds + 1):
        obs = env.reset()
        action = brain.act(obs, explore=True)
        next_obs, reward, done, info = env.step(action)
        brain.learn(reward, next_obs, done)

        # Если ответ верный — сразу укрепляем созданный нейрон как Золотой Прототип
        if info['is_correct'] and brain.last_neuron:
            brain.last_neuron.strength = 5.0
            brain.last_neuron.usage_count = 10
            brain.last_neuron.flag = 1.0
            if hasattr(brain.last_neuron, 'confidence'):
                brain.last_neuron.confidence = 1.0

        brain.reset_episode()

        if info['is_correct']:
            correct_count += 1

        pruned = 0
        if r % 10 == 0:
            pruned, merged = auto_compressor.compress_brain(brain)
            if hasattr(brain, '_rebuild_neuron_cache'):
                brain._rebuild_neuron_cache()

        total_n = sum(len(c.neurons) for c in brain.router.clusters)
        lessons = brain.stats.get('lessons', 0)
        accuracy = (correct_count / r) * 100.0

        win_sym = "✅" if info['is_correct'] else "❌"
        pruned_str = f" (🧹-{pruned})" if pruned > 0 else ""

        print(f"{win_sym} Раунд {r:03d}/{rounds:03d} | В: \"{info['prompt']}\" -> "
              f"О: \"{info['chosen_answer'][:30]}...\" | Точность: {accuracy:.1f}% | "
              f"Нейронов: {total_n:04d}{pruned_str} | Уроков: {lessons:03d}")

        if r % 50 == 0:
            # Оценка качества ответов обученного мозга без случайного исследования
            eval_correct = 0
            for d in env.dialogues:
                test_obs = env.reset(prompt_text=d['prompt'])
                pred_act = brain.act(test_obs, explore=False)
                if pred_act == env.target_answer_idx:
                    eval_correct += 1
            eval_acc = (eval_correct / len(env.dialogues)) * 100.0
            print(f"  🎯 ТЕСТ ТОЧНОСТИ МОЗГА на словаре: {eval_acc:.1f}% ({eval_correct}/{len(env.dialogues)})")
            save_brain(brain)

    elapsed = time.time() - start_time
    save_brain(brain)
    print("=" * 75)
    print(f"🎉 ОБУЧЕНИЕ РАЗГОВОРНОГО ИИ ЗАВЕРШЕНО ЗА {elapsed:.2f} сек!")
    print(f"📊 Итоговая точность на словаре: {accuracy:.1f}% ({correct_count}/{rounds})")
    print(f"💡 Для интерактивного чата с ИИ запустите: python main_dialogue.py")
    print("=" * 75)

def save_brain(brain):
    try:
        with open(SAVE_PATH, 'wb') as f:
            pickle.dump(brain, f)
    except Exception as e:
        print(f"⚠️ Ошибка сохранения: {e}")

if __name__ == "__main__":
    r_count = 100
    if len(sys.argv) > 1:
        try:
            r_count = int(sys.argv[1])
        except:
            pass
    run_headless_dialogue_training(rounds=r_count)
