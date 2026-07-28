"""
Иерархическая система макро-целей (Macro Goal System)
Выбирает стратегическую цель для ИИ на основе контекста
"""

import numpy as np

# Определения Макро-Целей
GOAL_EXPLORE = 0
GOAL_GATHER_ELIXIR = 1
GOAL_HUNT_ENEMY = 2
GOAL_RETREAT_SAFEZONE = 3
GOAL_FIND_WEAPON = 4

GOAL_NAMES = {
    GOAL_EXPLORE: "Разведка",
    GOAL_GATHER_ELIXIR: "Сбор Эликсиров",
    GOAL_HUNT_ENEMY: "Охота на Врага",
    GOAL_RETREAT_SAFEZONE: "Отступление",
    GOAL_FIND_WEAPON: "Поиск Оружия"
}

class MacroGoalSystem:
    """Управляет стратегическими намерениями и макро-целями ИИ"""

    def __init__(self):
        self.current_goal = GOAL_EXPLORE
        self.goal_history = []

    def select_goal(self, state_dict_or_vec):
        """
        Выбрать доминирующую макро-цель по вектору/словарю состояния
        """
        if isinstance(state_dict_or_vec, (list, np.ndarray, tuple)):
            health_norm = float(state_dict_or_vec[0]) if len(state_dict_or_vec) > 0 else 1.0
            drought_norm = float(state_dict_or_vec[2]) if len(state_dict_or_vec) > 2 else 0.0
            enemy_near = float(state_dict_or_vec[3]) if len(state_dict_or_vec) > 3 else 0.0
            weapon_status = float(state_dict_or_vec[4]) if len(state_dict_or_vec) > 4 else 0.0
            safe_zone_norm = float(state_dict_or_vec[5]) if len(state_dict_or_vec) > 5 else 0.0

            # 1. Отступление в безопасную зону при критическом HP
            if health_norm < 0.35:
                self.current_goal = GOAL_RETREAT_SAFEZONE

            # 2. Сбор эликсира при голоде или среднем HP
            elif drought_norm > 0.4 or health_norm < 0.7:
                self.current_goal = GOAL_GATHER_ELIXIR

            # 3. Охота на врага, если есть оружие и враг близко
            elif weapon_status > 0 and enemy_near > 0.3:
                self.current_goal = GOAL_HUNT_ENEMY

            # 4. Поиск оружия, если оружия нет
            elif weapon_status == 0:
                self.current_goal = GOAL_FIND_WEAPON

            # 5. Иначе свободное исследование
            else:
                self.current_goal = GOAL_EXPLORE

        self.goal_history.append(self.current_goal)
        return self.current_goal

    def get_goal_one_hot(self):
        """Вектор one-hot выбранной макро-цели"""
        vec = [0.0] * 5
        vec[self.current_goal] = 1.0
        return vec

    def get_goal_name(self):
        """Понятное название текущей цели"""
        return GOAL_NAMES.get(self.current_goal, "Неизвестно")
