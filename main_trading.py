"""
Графический Интерфейс Трейдинга (Trading AI Dashboard - Pygame GUI)
Визуализация свечного графика цен, сигналов ИИ (BUY/SELL/HOLD), RSI и финансового дашборда PnL
Запуск: python main_trading.py
"""

import sys
import os
import time
import pickle
import numpy as np
import pygame

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.trading_env import TradingEnv
from core.universal_brain import UniversalAssociativeBrain

SAVE_PATH = "trading_brain.pkl"

class TradingVisualizer:
    """Визуализатор торгового терминала на базе Pygame"""

    def __init__(self, width=1050, height=720):
        pygame.init()
        pygame.display.set_caption("📈 Genezis 2.0 - Trading AI Terminal & Analytics")

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        # Цветовая палитра профессионального торгового терминала
        self.BG_COLOR = (17, 24, 39)        # #111827
        self.PANEL_BG = (31, 41, 55)        # #1F2937
        self.PANEL_BORDER = (55, 65, 81)    # #374151
        self.TEXT_COLOR = (243, 244, 246)   # #F3F4F6
        self.MUTED_TEXT = (156, 163, 175)   # #9CA3AF
        
        self.GREEN_BULL = (16, 185, 129)    # #10B981 (Покупка / Прибыль)
        self.RED_BEAR = (239, 68, 68)       # #EF4444 (Продажа / Убыток)
        self.BLUE_ACCENT = (59, 130, 246)   # #3B82F6 (Акцент / ИИ)
        self.YELLOW_RSI = (245, 158, 11)    # #F59E0B (Индикатор RSI)

        # Шрифты
        try:
            self.font_title = pygame.font.SysFont("segoe ui", 20, bold=True)
            self.font_main = pygame.font.SysFont("segoe ui", 15)
            self.font_bold = pygame.font.SysFont("segoe ui", 15, bold=True)
            self.font_small = pygame.font.SysFont("segoe ui", 12)
        except Exception:
            self.font_title = pygame.font.Font(None, 24)
            self.font_main = pygame.font.Font(None, 18)
            self.font_bold = pygame.font.Font(None, 18)
            self.font_small = pygame.font.Font(None, 14)

        # Инициализация среды и Ассоциативного ИИ
        self.env = TradingEnv()
        self.brain = UniversalAssociativeBrain(action_count=3)

        if os.path.exists(SAVE_PATH):
            try:
                with open(SAVE_PATH, 'rb') as f:
                    saved_brain = pickle.load(f)
                    if hasattr(saved_brain, 'router'):
                        self.brain = saved_brain
                        print(f"  ✅ Загружен сохраненный Ассоциативный Мозг Трейдинга из {SAVE_PATH}")
            except Exception as e:
                print(f"  ⚠️ Ошибка загрузки {SAVE_PATH}: {e}")

        self.obs = self.env.reset()
        self.running = True
        self.paused = False
        self.speed = 10  # FPS (шагов в секунду)
        self.last_action = 0
        self.last_info = {'balance': 1000.0, 'position': 0, 'total_trades': 0, 'winning_trades': 0, 'win_rate': 0.0}
        self.action_names = {0: "HOLD", 1: "BUY (LONG)", 2: "SELL (SHORT)"}
        self.signals_history = []  # История точек входа [(step, price, action)]

    def run(self):
        """Главный цикл визуализации"""
        while self.running:
            self._handle_events()

            if not self.paused:
                self._update_step()

            self._draw()
            self.clock.tick(self.speed)

        pygame.quit()

    def _handle_events(self):
        """Обработка пользовательских команд с клавиатуры"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_UP:
                    self.speed = min(120, self.speed + 10)
                elif event.key == pygame.K_DOWN:
                    self.speed = max(1, self.speed - 5)
                elif event.key == pygame.K_r:
                    self.obs = self.env.reset()
                    self.signals_history = []
                elif event.key == pygame.K_1:
                    self.env.set_expiry_steps(5)
                elif event.key == pygame.K_2:
                    self.env.set_expiry_steps(15)
                elif event.key == pygame.K_3:
                    self.env.set_expiry_steps(30)
                elif event.key == pygame.K_4:
                    self.env.set_expiry_steps(60)
                elif event.key == pygame.K_m:
                    new_m = "live" if self.env.mode == "sim" else "sim"
                    self.env.set_mode(new_m)
                    self.signals_history = []
                    self.obs = self.env._get_observation()
                elif event.key == pygame.K_b:
                    self.env.change_symbol("BTC/USDT")
                    self.signals_history = []
                    self.obs = self.env._get_observation()
                elif event.key == pygame.K_e:
                    self.env.change_symbol("ETH/USDT")
                    self.signals_history = []
                    self.obs = self.env._get_observation()
                elif event.key == pygame.K_s:
                    self.env.change_symbol("SOL/USDT")
                    self.signals_history = []
                    self.obs = self.env._get_observation()
                elif event.key == pygame.K_F1:
                    self.speed = 1
                elif event.key == pygame.K_F2:
                    self.speed = 2
                elif event.key == pygame.K_F3:
                    self.speed = 10
                elif event.key == pygame.K_F4:
                    self.speed = 100

    def _update_step(self):
        """Выполнение одного шага торговли"""
        if self.env.mode == "live":
            self.env.update_live_tick()

        probs = self.brain.predict_action_probabilities(self.obs)
        action = self.brain.act(self.obs, explore=True)
        next_obs, reward, done, info = self.env.step(action)
        self.brain.learn(reward, next_obs, done)

        if action != self.last_action and action != 0:
            self.signals_history.append((self.env.current_step, self.env.prices[self.env.current_step - 1], action))

        info['probs'] = probs
        self.last_action = action
        self.last_info = info
        self.obs = next_obs

        if done:
            self.brain.reset_episode()
            self.obs = self.env.reset(preserve_step=True)
            self.signals_history = []

    def _draw(self):
        """Полная отрисовка дашборда"""
        self.screen.fill(self.BG_COLOR)

        # 1. Шапка (Header Panel)
        self._draw_header()

        # 2. График цены свечей (Price Chart)
        self._draw_price_chart(x=20, y=60, w=680, h=380)

        # 3. Под-график RSI (RSI Sub-chart)
        self._draw_rsi_chart(x=20, y=460, w=680, h=230)

        # 4. Правая аналитическая панель (Analytics Dashboard)
        self._draw_analytics_panel(x=720, y=60, w=310, h=630)

        pygame.display.flip()

    def _draw_header(self):
        """Верхняя информационная плашка"""
        header_rect = pygame.Rect(0, 0, self.width, 45)
        pygame.draw.rect(self.screen, self.PANEL_BG, header_rect)
        pygame.draw.line(self.screen, self.PANEL_BORDER, (0, 45), (self.width, 45), 2)

        sym = self.last_info.get('active_symbol', 'BTC/USDT')
        mode = self.last_info.get('mode', 'sim')
        mode_str = "[LIVE BINANCE]" if mode == "live" else "[SIMULATION]"
        mode_color = self.GREEN_BULL if mode == "live" else self.BLUE_ACCENT

        speed_lbl = f"{self.speed}x" if self.speed > 1 else "1x REAL-TIME"
        if self.speed >= 100:
            speed_lbl = "100x TURBO"

        title = self.font_title.render(f"📈 GENEZIS 2.0 {mode_str} [{sym}] | ⚡ {speed_lbl}", True, mode_color)
        self.screen.blit(title, (20, 10))

        controls = self.font_small.render("[F1-F4: Скорость 1x-100x | M: SIM/LIVE | B,E,S,X: Пары | 1-4: Экспирация]", True, self.MUTED_TEXT)
        self.screen.blit(controls, (self.width - 450, 15))

    def _draw_price_chart(self, x, y, w, h):
        """Отрисовка главного свечного графика цен с детальной информацией о сделках"""
        panel_rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, self.PANEL_BG, panel_rect, border_radius=6)
        pygame.draw.rect(self.screen, self.PANEL_BORDER, panel_rect, width=1, border_radius=6)

        title = self.font_bold.render("ГРАФИК СВЕЧЕЙ И ОРДЕРОВ ИИ (CANDLES & ORDERS)", True, self.TEXT_COLOR)
        self.screen.blit(title, (x + 15, y + 10))

        # Берем окно последних 60 баров
        curr_step = self.env.current_step
        window_size = 60
        start_idx = max(0, curr_step - window_size)
        window_prices = self.env.prices[start_idx:curr_step + 1]

        window_ohlc = self.env.ohlcv[start_idx:curr_step + 1] if hasattr(self.env, 'ohlcv') else []

        if window_ohlc:
            min_p = min([c['low'] for c in window_ohlc]) * 0.995
            max_p = max([c['high'] for c in window_ohlc]) * 1.005
        else:
            min_p = np.min(window_prices) * 0.995
            max_p = np.max(window_prices) * 1.005

        p_range = max(0.001, max_p - min_p)

        # Отрисовка фоновой сетки
        for i in range(1, 4):
            grid_y = y + 40 + i * (h - 60) // 4
            pygame.draw.line(self.screen, (40, 50, 65), (x + 10, grid_y), (x + w - 10, grid_y), 1)

        # Отрисовка Японских Свечей (High-Low тени и Open-Close тела)
        step_w = (w - 30) / float(window_size)
        candle_w = max(3.0, step_w * 0.65)

        for i, c in enumerate(window_ohlc):
            cx = float(x + 15 + i * step_w)
            o_p, h_p, l_p, cl_p = c['open'], c['high'], c['low'], c['close']

            hy = float(y + h - 20 - ((h_p - min_p) / p_range) * (h - 60))
            ly = float(y + h - 20 - ((l_p - min_p) / p_range) * (h - 60))
            oy = float(y + h - 20 - ((o_p - min_p) / p_range) * (h - 60))
            cy = float(y + h - 20 - ((cl_p - min_p) / p_range) * (h - 60))

            is_bull = (cl_p >= o_p)
            c_color = self.GREEN_BULL if is_bull else self.RED_BEAR

            # 1. Тень свечи
            pygame.draw.line(self.screen, c_color, (cx, hy), (cx, ly), 1)

            # 2. Тело свечи
            top_y = min(oy, cy)
            body_h = max(2.0, abs(cy - oy))
            body_rect = pygame.Rect(cx - candle_w / 2.0, top_y, candle_w, body_h)
            pygame.draw.rect(self.screen, c_color, body_rect)

        # Отрисовка завершенных опционов (Где открыл, где закрылось, плашка WIN / LOSS)
        comp_opts = getattr(self.env, 'completed_options', [])
        for opt in comp_opts[-10:]:
            ex_step = opt.get('exit_step', 0)
            st_step = opt.get('entry_step', 0)
            if start_idx <= ex_step <= curr_step or start_idx <= st_step <= curr_step:
                ex_offset = min(window_size, max(0, ex_step - start_idx))
                ex_x = float(x + 15 + ex_offset * step_w)
                ex_price = opt.get('exit_price', opt['strike_price'])
                ex_y = float(y + h - 20 - ((ex_price - min_p) / p_range) * (h - 60))
                
                res = opt.get('result', 'WIN')
                res_bg = self.GREEN_BULL if res == 'WIN' else self.RED_BEAR
                res_text = f"🏆 WIN +${opt.get('profit', 8.5):.2f}" if res == 'WIN' else f"❌ LOSS -${opt['amount']:.2f}"

                # Рисуем плашку исхода у свечи закрытия
                badge_lbl = self.font_bold.render(res_text, True, (255, 255, 255))
                badge_rect = pygame.Rect(ex_x - 45, ex_y - 18, 90, 16)
                pygame.draw.rect(self.screen, res_bg, badge_rect, border_radius=4)
                self.screen.blit(badge_lbl, (ex_x - 40, ex_y - 18))

        # Отрисовка активных опционов (Страйк цена и остаток времени)
        active_opts = self.last_info.get('active_options', [])
        for opt in active_opts:
            strike_p = opt['strike_price']
            rem_steps = opt['expiry_step'] - curr_step
            strike_y = float(y + h - 20 - ((strike_p - min_p) / p_range) * (h - 60))
            line_color = self.GREEN_BULL if opt['direction'] == 1 else self.RED_BEAR
            
            pygame.draw.line(self.screen, line_color, (x + 15, strike_y), (x + w - 15, strike_y), 1)
            lbl = self.font_small.render(f"Strike: ${strike_p:.2f} ({'UP' if opt['direction'] == 1 else 'DOWN'}) | Экспирация через: {rem_steps}м", True, line_color)
            self.screen.blit(lbl, (x + 25, strike_y - 14))

        # Отрисовка новостных маркеров на шкале времени (NEWS EVENTS)
        if hasattr(self.env, 'news_calendar'):
            for ev_step, ev_data in self.env.news_calendar.events.items():
                if start_idx <= ev_step <= curr_step:
                    idx_offset = ev_step - start_idx
                    nx = float(x + 15 + idx_offset * step_w)
                    pygame.draw.line(self.screen, self.YELLOW_RSI, (nx, y + 35), (nx, y + h - 20), 1)
                    news_lbl = self.font_small.render("📰 NEWS", True, self.YELLOW_RSI)
                    self.screen.blit(news_lbl, (nx - 18, y + 42))

        # Отрисовка маркеров сигналов входа ИИ (BUY / SELL)
        for sig_step, sig_price, sig_action in self.signals_history:
            if start_idx <= sig_step <= curr_step:
                idx_offset = sig_step - start_idx
                sx = float(x + 15 + idx_offset * step_w)
                sy = float(y + h - 20 - ((sig_price - min_p) / p_range) * (h - 60))

                if sig_action == 1:  # BUY
                    pygame.draw.polygon(self.screen, self.GREEN_BULL, [(sx, sy - 12.0), (sx - 6.0, sy), (sx + 6.0, sy)])
                elif sig_action == 2:  # SELL
                    pygame.draw.polygon(self.screen, self.RED_BEAR, [(sx, sy + 12.0), (sx - 6.0, sy), (sx + 6.0, sy)])

        # Текущая цена и дата
        curr_sym = self.last_info.get('active_symbol', 'BTC/USDT')
        curr_price = float(self.env.prices[curr_step])
        curr_date = self.last_info.get('current_date', '')
        curr_txt = self.font_bold.render(f"{curr_sym}: ${curr_price:.2f}", True, self.TEXT_COLOR)
        date_txt = self.font_small.render(curr_date, True, self.MUTED_TEXT)
        self.screen.blit(curr_txt, (x + w - 170, y + 8))
        self.screen.blit(date_txt, (x + w - 170, y + 25))

    def _draw_rsi_chart(self, x, y, w, h):
        """Отрисовка индикатора RSI (Осциллятор)"""
        panel_rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, self.PANEL_BG, panel_rect, border_radius=6)
        pygame.draw.rect(self.screen, self.PANEL_BORDER, panel_rect, width=1, border_radius=6)

        title = self.font_bold.render("ИНДИКАТОР RSI (14) - ИНДЕКС ОТНОСИТЕЛЬНОЙ СИЛЫ", True, self.TEXT_COLOR)
        self.screen.blit(title, (x + 15, y + 10))

        # Уровни 70% (Перекупленность) и 30% (Перепроданность)
        y_70 = y + 40 + int((1.0 - 0.70) * (h - 60))
        y_30 = y + 40 + int((1.0 - 0.30) * (h - 60))

        pygame.draw.line(self.screen, self.RED_BEAR, (x + 10, y_70), (x + w - 10, y_70), 1)
        pygame.draw.line(self.screen, self.GREEN_BULL, (x + 10, y_30), (x + w - 10, y_30), 1)

        t_70 = self.font_small.render("70% Overbought", True, self.RED_BEAR)
        t_30 = self.font_small.render("30% Oversold", True, self.GREEN_BULL)
        self.screen.blit(t_70, (x + w - 110, y_70 - 14))
        self.screen.blit(t_30, (x + w - 100, y_30 + 2))

        # Рисуем кривую RSI
        curr_step = self.env.current_step
        window_size = 60
        start_idx = max(30, curr_step - window_size)
        step_w = (w - 30) / float(window_size)
        rsi_points = []

        for i, st in enumerate(range(start_idx, curr_step + 1)):
            sub = self.env.prices[st - 14:st + 1]
            diffs = np.diff(sub)
            gains = np.where(diffs > 0, diffs, 0)
            losses = np.where(diffs < 0, -diffs, 0)
            ag = float(np.mean(gains)) if len(gains) > 0 else 1e-8
            al = float(np.mean(losses)) if len(losses) > 0 else 1e-8
            rsi = 100.0 if al == 0 else 100.0 - (100.0 / (1.0 + (ag / al)))
            
            rx = float(x + 15 + i * step_w)
            ry = float(y + 40 + (1.0 - rsi / 100.0) * (h - 60))
            rsi_points.append((rx, ry))

        if len(rsi_points) > 1:
            pygame.draw.lines(self.screen, self.YELLOW_RSI, False, rsi_points, 2)

    def _draw_analytics_panel(self, x, y, w, h):
        """Правая панель с аналитикой баланса, позиций и состояния ИИ"""
        panel_rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, self.PANEL_BG, panel_rect, border_radius=6)
        pygame.draw.rect(self.screen, self.PANEL_BORDER, panel_rect, width=1, border_radius=6)

        title = self.font_bold.render("ФИНАНСОВЫЙ ДАШБОРД ИИ", True, self.TEXT_COLOR)
        self.screen.blit(title, (x + 15, y + 15))

        curr_y = y + 45

        # 1. Карточка Баланса и PnL
        bal = self.last_info['balance']
        pnl = bal - self.env.initial_balance
        pnl_pct = (pnl / self.env.initial_balance) * 100.0
        pnl_color = self.GREEN_BULL if pnl >= 0 else self.RED_BEAR

        pygame.draw.rect(self.screen, (24, 32, 47), (x + 15, curr_y, w - 30, 80), border_radius=6)
        lbl_bal = self.font_small.render("ТЕКУЩИЙ БАЛАНС:", True, self.MUTED_TEXT)
        val_bal = self.font_title.render(f"${bal:.2f}", True, self.TEXT_COLOR)
        val_pnl = self.font_bold.render(f"{pnl:+.2f}$ ({pnl_pct:+.1f}%)", True, pnl_color)

        self.screen.blit(lbl_bal, (x + 25, curr_y + 10))
        self.screen.blit(val_bal, (x + 25, curr_y + 28))
        self.screen.blit(val_pnl, (x + 25, curr_y + 53))

        curr_y += 95

        # 2. Карточка Позиции и Экспирации
        pos = self.last_info['position']
        pos_str = "FLAT (ВНЕ РЫНКА)" if pos == 0 else ("LONG (ПОКУПКА)" if pos == 1 else "SHORT (ПРОДАЖА)")
        pos_color = self.MUTED_TEXT if pos == 0 else (self.GREEN_BULL if pos == 1 else self.RED_BEAR)
        exp_setting = self.last_info.get('selected_expiry', 15)

        pygame.draw.rect(self.screen, (24, 32, 47), (x + 15, curr_y, w - 30, 95), border_radius=6)
        lbl_pos = self.font_small.render("АКТИВНАЯ ПОЗИЦИЯ & ТАЙМФРЕЙМ:", True, self.MUTED_TEXT)
        val_pos = self.font_bold.render(pos_str, True, pos_color)
        act_txt = self.font_small.render(f"Решение ИИ: {self.action_names[self.last_action]}", True, self.BLUE_ACCENT)
        exp_txt = self.font_bold.render(f"Таймер прогноза: {exp_setting} мин (Клавиши 1-4)", True, self.YELLOW_RSI)

        self.screen.blit(lbl_pos, (x + 25, curr_y + 8))
        self.screen.blit(val_pos, (x + 25, curr_y + 26))
        self.screen.blit(act_txt, (x + 25, curr_y + 48))
        self.screen.blit(exp_txt, (x + 25, curr_y + 68))

        curr_y += 110

        # 3. Карточка Статистики Сделок и Исходов
        trades = self.last_info['total_trades']
        wins = self.last_info['winning_trades']
        win_rate = self.last_info['win_rate']
        last_exp = self.last_info.get('last_expired')

        pygame.draw.rect(self.screen, (24, 32, 47), (x + 15, curr_y, w - 30, 120), border_radius=6)
        lbl_st = self.font_small.render("СТАТИСТИКА ПРОГНОЗОВ С ЭКСПИРАЦИЕЙ:", True, self.MUTED_TEXT)
        t_tr = self.font_main.render(f"Всего закрытых опционов: {trades}", True, self.TEXT_COLOR)
        t_win = self.font_main.render(f"Точных: {wins} / Неточных: {trades - wins}", True, self.TEXT_COLOR)
        t_wr = self.font_bold.render(f"Точность (Win Rate): {win_rate:.1f}%", True, self.GREEN_BULL if win_rate >= 50 else self.YELLOW_RSI)

        if last_exp:
            res_str = f"Последний прогноз: {last_exp['result']} ({last_exp['profit']:+.2f}$)"
            res_color = self.GREEN_BULL if last_exp['result'] == 'WIN' else self.RED_BEAR
            t_last = self.font_small.render(res_str, True, res_color)
            self.screen.blit(t_last, (x + 25, curr_y + 96))

        self.screen.blit(lbl_st, (x + 25, curr_y + 8))
        self.screen.blit(t_tr, (x + 25, curr_y + 28))
        self.screen.blit(t_win, (x + 25, curr_y + 48))
        self.screen.blit(t_wr, (x + 25, curr_y + 70))

        curr_y += 135

        # 4. Состояние Новостного Фона & Шкала Вероятностей ИИ (Live Neural Probabilities Gauge)
        probs = self.last_info.get('probs', {0: 50.0, 1: 25.0, 2: 25.0})
        p_buy = probs.get(1, 0.0)
        p_sell = probs.get(2, 0.0)
        p_hold = probs.get(0, 0.0)

        regime_lbl = self.last_info.get('market_regime_label', '↔️ КОНСОЛИДАЦИЯ / ФЛЭТ')
        ai_strat = self.last_info.get('ai_strategy', 'Mean Reversion (Отбой от границ)')

        pygame.draw.rect(self.screen, (24, 32, 47), (x + 15, curr_y, w - 30, 135), border_radius=6)
        lbl_st_title = self.font_small.render("ЖИВЫЕ ВЕРОЯТНОСТИ & СТРАТЕГИЯ ИИ:", True, self.YELLOW_RSI)
        t_reg = self.font_bold.render(f"Фаза рынка: {regime_lbl}", True, self.TEXT_COLOR)
        t_str = self.font_small.render(f"Логика: {ai_strat}", True, self.BLUE_ACCENT)
        t_probs = self.font_small.render(f"Мнение: CALL {p_buy:.1f}% | PUT {p_sell:.1f}% | HOLD {p_hold:.1f}%", True, self.TEXT_COLOR)

        # Визуальная шкала (Gauge Bar)
        bar_w = float(w - 50)
        bar_h = 10.0
        bar_x = float(x + 25)
        bar_y = float(curr_y + 112)

        pygame.draw.rect(self.screen, (40, 50, 65), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
        w_buy = float(bar_w * (p_buy / 100.0))
        w_sell = float(bar_w * (p_sell / 100.0))

        if w_buy > 0:
            pygame.draw.rect(self.screen, self.GREEN_BULL, (bar_x, bar_y, w_buy, bar_h), border_top_left_radius=3, border_bottom_left_radius=3)
        if w_sell > 0:
            pygame.draw.rect(self.screen, self.RED_BEAR, (bar_x + bar_w - w_sell, bar_y, w_sell, bar_h), border_top_right_radius=3, border_bottom_right_radius=3)

        self.screen.blit(lbl_st_title, (x + 25, curr_y + 8))
        self.screen.blit(t_reg, (x + 25, curr_y + 28))
        self.screen.blit(t_str, (x + 25, curr_y + 50))
        self.screen.blit(t_probs, (x + 25, curr_y + 72))

        curr_y += 150

        # 5. Карточка Состояния Мозга ИИ
        total_n = sum(len(c.neurons) for c in self.brain.router.clusters)
        active_c = self.brain.active_cluster.domain if self.brain.active_cluster else "trading"
        lessons = self.brain.stats.get('lessons', 0)

        pygame.draw.rect(self.screen, (24, 32, 47), (x + 15, curr_y, w - 30, 110), border_radius=6)
        lbl_br = self.font_small.render("СОСТОЯНИЕ АССОЦИАТИВНОГО ИИ:", True, self.MUTED_TEXT)
        t_cl = self.font_main.render(f"Доменный кластер: '{active_c}'", True, self.BLUE_ACCENT)
        t_nr = self.font_main.render(f"Активных нейронов: {total_n}", True, self.TEXT_COLOR)
        t_ls = self.font_main.render(f"Уроков / Контрфактов: {lessons}", True, self.TEXT_COLOR)

        self.screen.blit(lbl_br, (x + 25, curr_y + 8))
        self.screen.blit(t_cl, (x + 25, curr_y + 28))
        self.screen.blit(t_nr, (x + 25, curr_y + 48))
        self.screen.blit(t_ls, (x + 25, curr_y + 70))

if __name__ == "__main__":
    vis = TradingVisualizer()
    vis.run()
