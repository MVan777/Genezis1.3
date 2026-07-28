"""
Движок Автономного Формирования Мета-Целей (SelfDrivenGoalDiscovery)
Генерирует собственные исследовательские гипотезы и цели при застревании или скуке ИИ
"""

import time
import numpy as np

class SelfDrivenGoalDiscovery:
    """Модуль генерации собственных внутренних исследовательских гипотез"""

    def __init__(self, boredom_threshold_steps=80):
        self.boredom_threshold_steps = boredom_threshold_steps
        self.stagnation_counter = 0
        self.is_bored = False
        self.active_hypothesis = None
        self.hypotheses_history = []

    def update_state(self, reward_history):
        """Проверить уровень скуки и застревания ИИ"""
        if len(reward_history) < 20:
            return False, None

        recent_std = float(np.std(reward_history[-20:]))

        # Если вариативность наград близка к нулю -> наступила скука / монотонность
        if recent_std < 0.05:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = max(0, self.stagnation_counter - 1)

        if self.stagnation_counter >= self.boredom_threshold_steps:
            self.is_bored = True
            if self.active_hypothesis is None:
                self.active_hypothesis = self._generate_hypothesis()
        else:
            self.is_bored = False
            self.active_hypothesis = None

        return self.is_bored, self.active_hypothesis

    def _generate_hypothesis(self):
        """Сгенерировать новую внутреннюю исследовательскую гипотезу"""
        possible_goals = [
            {'name': 'TRY_AGGRESSIVE_STRIKE', 'bonus_action': 1, 'boost': 0.4},
            {'name': 'TRY_DEFENSIVE_EVASION', 'bonus_action': 2, 'boost': 0.4},
            {'name': 'TRY_EXPERIMENTAL_COMBO', 'bonus_action': 0, 'boost': 0.3}
        ]

        choice = possible_goals[np.random.randint(0, len(possible_goals))]
        choice['timestamp'] = time.time()
        self.hypotheses_history.append(choice['name'])
        return choice
