"""
Интерактивный Лаунчер 3-й Игры: Разговорный Ассоциативный ИИ (Genezis Dialogue Terminal)
Позволяет общаться с ИИ в терминале в реальном времени и обучать его отвечать на любые вопросы.
"""

import sys
import os
import pickle

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.text_dialogue_env import TextDialogueEnv, encode_text_to_vector
from core.universal_brain import UniversalAssociativeBrain
from train_dialogue_headless import run_headless_dialogue_training

SAVE_PATH = "dialogue_brain.pkl"

def load_or_create_brain(action_count):
    if os.path.exists(SAVE_PATH):
        try:
            with open(SAVE_PATH, 'rb') as f:
                saved_brain = pickle.load(f)
                if hasattr(saved_brain, 'router'):
                    print(f"  ✅ Загружен разговорный мозг из '{SAVE_PATH}'")
                    return saved_brain
        except Exception as e:
            print(f"  ⚠️ Ошибка загрузки {SAVE_PATH}: {e}")
    return UniversalAssociativeBrain(action_count=action_count)

def interactive_chat():
    env = TextDialogueEnv()
    brain = load_or_create_brain(env.action_count)

    print("\n" + "=" * 70)
    print("💬 ИНТЕРАКТИВНЫЙ ЧАТ С АССОЦИАТИВНЫМ ИИ (Genezis 2.0 Dialogue)")
    print("=" * 70)
    print("💡 Инструкция:")
    print("  - Задайте любой вопрос или напишите фразу.")
    print("  - ИИ ответит на основе сохранённых ассоциативных связей.")
    print("  - Вы можете подкрепить ответ (1 = Хорошо, 0 = Плохо) для обучения на лету.")
    print("  - Введите 'выход' или 'exit' для завершения чата.\n")

    while True:
        try:
            user_input = input("\n👤 Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() in ['выход', 'exit', 'quit']:
            print("\n👋 Завершение чата. Сохранение знаний...")
            with open(SAVE_PATH, 'wb') as f:
                pickle.dump(brain, f)
            print("✅ Мозг сохранён в 'dialogue_brain.pkl'. До свидания!")
            break

        obs = env.reset(prompt_text=user_input)
        action_idx = brain.act(obs, explore=False)
        answer_text = env.get_answer_text(action_idx)

        print(f"🤖 Genezis: {answer_text}")

        # Быстрый шаг обучения
        next_obs, reward, done, info = env.step(action_idx)
        brain.learn(reward, next_obs, done)
        brain.reset_episode()

def main():
    print("============================================================")
    print("🗣️  GENEZIS 2.0 — ИГРА №3: РАЗГОВОРНЫЙ АССОЦИАТИВНЫЙ ИИ")
    print("============================================================")
    print("1. 💬 Интерактивный Чат с ИИ в реальном времени")
    print("2. 🚀 Быстрое Авто-Обучение на словаре диалогов (50 раундов)")
    print("3. ❌ Выход")
    print("============================================================")

    try:
        choice = input("Выберите режим (1-3): ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    if choice == '1':
        interactive_chat()
    elif choice == '2':
        run_headless_dialogue_training(rounds=50)
    else:
        print("Выход.")

if __name__ == "__main__":
    main()
