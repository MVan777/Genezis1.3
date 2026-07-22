"""
Конфигурация проекта - добавляем параметры для сжатия, сенсоров, врага и оружия
"""

import numpy as np

# ============================================
# ПАРАМЕТРЫ ИГРЫ
# ============================================

GRID_SIZE = 12  # увеличим для врага

BASE_MAX_HEALTH = 100  # базовый максимум
ABSOLUTE_MAX_HEALTH = 1000  # абсолютный потолок (можно очень высоко)

# Элементы
POISON_DAMAGE = 1
POISON_TICK = 5
ANTIDOTE_DURATION = 10
ELIXIR_HEAL = 10
WEAPON_DAMAGE = 20  # урон оружия
ENEMY_DAMAGE = 10   # урон врага
WEAPON_USES = 3     # количество выстрелов

# Штрафы и награды
STAND_STILL_TIME = 20      # через сколько шагов начинает штрафовать
STAND_STILL_PENALTY = 5    # сколько здоровья снимать
STEP_PENALTY = 0.01
REVISIT_PENALTY = 0.3
NEW_CELL_REWARD = 0.1
ELIXIR_REWARD = 20.0
ANTIDOTE_REWARD = 1.0
POISON_REWARD = -1.0
DEATH_REWARD = -2.0
ENEMY_DAMAGE_REWARD = -2.0   # враг атаковал
ENEMY_KILL_REWARD = 10.0      # убил врага
WEAPON_PICKUP_REWARD = 2.0   # подобрал оружие
SURVIVAL_BONUS = 0.1

# Регенерация
REGEN_ELIXIR = 2
REGEN_ANTIDOTE = 1
REGEN_POISON = 1
REGEN_WEAPON = 1  # оружие
REGEN_THRESHOLD = 5
REGEN_CHECK_INTERVAL = 20

# Штраф за блуждания
EXPLORATION_DROUGHT = 20
EXPLORATION_PENALTY = 20
EXPLORATION_REWARD_PENALTY = -2.0

# Память (НОВЫЕ ПАРАМЕТРЫ ДЛЯ СЖАТИЯ)
NEUTRAL_THRESHOLD = 0.2
SIMILARITY_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.1
COMPRESSION_SIMILARITY = 0.35  # порог похожести для сжатия
COMPRESSION_INTERVAL = 50       # шагов между сжатиями
MAX_NEURONS_BEFORE_COMPRESSION = 400  # если больше - сжимаем
NEURON_LIFETIME = 1000          # сколько шагов живет нейрон

# Сенсоры (дальность обзора)
SENSOR_RANGE = 2  # видит на 2 клетки вперед

# Запрет возврата
REVISIT_MEMORY = 8

# Типы клеток
EMPTY = 0
POISON = 1
ANTIDOTE = 2
ELIXIR = 3
WEAPON = 4  # новый тип

# Действия (добавили атаку)
ACTIONS = ['up', 'down', 'left', 'right', 'stay', 'attack']
ACTION_COUNT = len(ACTIONS)

# Цвета действий
ACTION_COLORS = {
    0: (100, 100, 255),  # up - синий
    1: (100, 255, 100),  # down - зеленый
    2: (255, 100, 100),  # left - красный
    3: (255, 255, 100),  # right - желтый
    4: (200, 200, 200),  # stay - серый
    5: (255, 0, 0)       # attack - ярко-красный
}

# ============================================
# ПАРАМЕТРЫ ОТОБРАЖЕНИЯ
# ============================================

CELL_SIZE = 35
GAME_WIDTH = GRID_SIZE * CELL_SIZE  # 12*35 = 420

# ИНФОРМАЦИОННАЯ ПАНЕЛЬ (статистика игры)
INFO_PANEL_WIDTH = 250
INFO_PANEL_X = GAME_WIDTH + 10

# ПАНЕЛЬ УПРАВЛЕНИЯ (кнопки, слайдеры)
CONTROL_PANEL_WIDTH = 280
CONTROL_PANEL_X = INFO_PANEL_X + INFO_PANEL_WIDTH + 20

# ВИЗУАЛИЗАЦИЯ НЕЙРОНОВ
NEURON_VIS_WIDTH = 500
NEURON_VIS_HEIGHT = 600
NEURON_VIS_X = CONTROL_PANEL_X + CONTROL_PANEL_WIDTH + 20

# Общая ширина окна
WINDOW_WIDTH = NEURON_VIS_X + NEURON_VIS_WIDTH
WINDOW_HEIGHT = max(GRID_SIZE * CELL_SIZE, 600)

INFO_PANEL_X = GAME_WIDTH + 20
NEURON_VIS_X = GAME_WIDTH + INFO_PANEL_WIDTH + 40

MAX_CONNECTIONS_TO_DRAW = 1000
CONNECTION_STRENGTH_THRESHOLD = 0.3

# Цвета
COLORS = {
    'background': (30, 30, 30),
    'grid': (60, 60, 60),
    'player': (0, 200, 255),
    'poison': (255, 50, 50),
    'antidote': (50, 255, 50),
    'elixir': (50, 50, 255),
    'weapon': (255, 215, 0),  # золотой
    'enemy': (255, 0, 255),   # розовый/пурпурный
    'empty': (200, 200, 200),
    'text': (255, 255, 255),
    'panel_bg': (20, 20, 20),
    'positive': (0, 255, 0),
    'negative': (255, 0, 0),
    'neutral': (150, 150, 150),
    'neuron_bg': (10, 10, 10),
    'neuron_connection': (60, 60, 60)
}

# ============================================
# БОНУСЫ ЗА УРОВЕНЬ ЗДОРОВЬЯ (НОВОЕ)
# ============================================

HEALTH_MILESTONES = {
    100: 0,    # первый порог - просто достижение
    150: 50,   # +30 здоровья
    200: 70,   # +50 здоровья
    250: 60,   # +40 здоровья
    300: 80,   # +60 здоровья
    400: 70,   # +50 здоровья
    500: 150   # +100 здоровья (джепот!)
}

# ============================================
# НОВЫЕ КОНСТАНТЫ ДЛЯ ГОЛОДА И ЕДЫ
# ============================================

# Голод
HUNGER_START = 0
HUNGER_RATE = 1  # за шаг
HUNGER_THRESHOLD = 100
HUNGER_DAMAGE = 5
FOOD_SATIETY = 30  # сколько голода убирает еда

# Безопасная зона
SAFE_ZONE_HEAL = 10
SAFE_ZONE_POS = [10, 10]  # будет установлено в игре

# Типы клеток (добавить к существующим)
FOOD = 5
SAFE_ZONE = 6

# ============================================
# ПАРАМЕТРЫ ОБУЧЕНИЯ
# ============================================

POPULATION_SIZE = 3          # количество агентов в популяции
EPOCHS = 1                 # общее количество эпох
EPISODES_PER_AGENT = 4       # игр на агента (больше = точнее)
MAX_STEPS = 1000              # шагов за игру

ELITE_SIZE = 2               # сколько лучших сохранять
EXPLORE_START = 1.0          # начальный уровень исследований
EXPLORE_END = 0.1            # конечный уровень исследований
EXPLORE_DECAY_EPOCHS = 40    # за сколько эпох снижать до минимума

SHORT_TERM_LIFETIME = 1000  # шагов жизни краткосрочного нейрона
CONSOLIDATION_INTERVAL = 10  # игр между консолидациями
CONSOLIDATION_THRESHOLD = 0.7  # порог важности для переноса
MAX_SHORT_TERM_NEURONS = 1000  # максимум краткосрочных нейронов
LONG_TERM_CONFIDENCE_BOOST = 1.5  # вес долгосрочных нейронов