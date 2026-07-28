"""
Окружение Биржевого Трейдинга (Trading Environment)
Совместимо со стандартом OpenAI Gym API
Поддерживает работу с крипто/фондовыми свечами (OHLCV), расчет RSI, волатильности и PnL с учетом комиссий.
"""

import numpy as np
import random

class TradingEnv:
    """Биржевое окружение для обучения и бэктестинга Ассоциативного ИИ"""

    def __init__(self, df_prices=None, initial_balance=1000.0, fee_pct=0.0005):
        self.initial_balance = initial_balance
        self.fee_pct = fee_pct  # 0.05% комиссия за сделку (стандарт Binance/Bybit)

        if df_prices is not None and len(df_prices) > 50:
            self.prices = np.array(df_prices, dtype=np.float32)
        else:
            # Генерируем реалистичный синтетический график биржа/крипта (Geometric Brownian Motion + Тренд)
            self.prices = self._generate_synthetic_market_data(steps=500)

        self.reset()

    def _generate_synthetic_market_data(self, steps=500):
        """Генерация синтетических свечей биржи с волнами Эллиотта, трендами и шумом"""
        np.random.seed(42)
        price = 100.0
        prices = [price]
        trend = 0.0005
        
        for t in range(steps):
            cycle = 0.002 * np.sin(t / 15.0)
            noise = np.random.normal(0, 0.012)
            ret = trend + cycle + noise
            price *= (1.0 + ret)
            prices.append(price)

        return np.array(prices, dtype=np.float32)

    def reset(self):
        """Сброс состояния торговли на начало графика"""
        self.current_step = 30  # Начинаем после накопительного окна индикаторов
        self.balance = self.initial_balance
        self.position = 0  # 0: FLAT, 1: LONG, -1: SHORT
        self.entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.trades_history = []
        self.pnl_history = [self.balance]

        return self._get_observation()

    def _calculate_rsi(self, window=14):
        """Расчёт индекса относительной силы (RSI)"""
        sub_prices = self.prices[self.current_step - window:self.current_step + 1]
        diffs = np.diff(sub_prices)
        gains = np.where(diffs > 0, diffs, 0)
        losses = np.where(diffs < 0, -diffs, 0)

        avg_gain = np.mean(gains) if len(gains) > 0 else 1e-8
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-8

        if avg_loss == 0:
            return 1.0
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi / 100.0  # Нормализовано от 0.0 до 1.0

    def _get_observation(self):
        """
        Вектор состояния биржи R^7 для Ассоциативного ИИ:
        [ret_1, ret_5, ret_15, rsi_norm, vola_norm, position, unrealized_pnl]
        """
        curr_price = self.prices[self.current_step]

        # Доходность за 1, 5, 15 баров
        ret_1 = (curr_price - self.prices[self.current_step - 1]) / self.prices[self.current_step - 1]
        ret_5 = (curr_price - self.prices[self.current_step - 5]) / self.prices[self.current_step - 5]
        ret_15 = (curr_price - self.prices[self.current_step - 15]) / self.prices[self.current_step - 15]

        # RSI и Волатильность (стандартное отклонение за 10 баров)
        rsi_norm = self._calculate_rsi(window=14)
        sub = self.prices[self.current_step - 10:self.current_step + 1]
        vola = np.std(np.diff(sub) / sub[:-1]) if len(sub) > 1 else 0.01
        vola_norm = min(1.0, vola * 20.0)

        # Статус позиции и нереализованный PnL
        pos_norm = float(self.position)
        unrealized_pnl = 0.0
        if self.position == 1:
            unrealized_pnl = (curr_price - self.entry_price) / self.entry_price
        elif self.position == -1:
            unrealized_pnl = (self.entry_price - curr_price) / self.entry_price

        obs = [ret_1 * 10.0, ret_5 * 5.0, ret_15 * 3.0, rsi_norm, vola_norm, pos_norm, unrealized_pnl * 5.0, 0.5]
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """
        Выполнить шаг в биржевой торговле
        action: 0 - HOLD / FLAT, 1 - BUY (LONG), 2 - SELL (SHORT)
        """
        self.current_step += 1
        curr_price = self.prices[self.current_step - 1]
        next_price = self.prices[self.current_step]

        reward = 0.0
        trade_event = "none"

        # Целевая позиция по выбору ИИ
        target_pos = 0
        if action == 1:
            target_pos = 1  # LONG
        elif action == 2:
            target_pos = -1 # SHORT

        # Изменение позиции (совершение сделки)
        if target_pos != self.position:
            # Если была открыта позиция — закрываем и фиксируем PnL
            if self.position != 0:
                if self.position == 1:
                    realized_pnl = (curr_price - self.entry_price) / self.entry_price
                else:
                    realized_pnl = (self.entry_price - curr_price) / self.entry_price

                # Учитываем комиссию за закрытие
                realized_pnl -= self.fee_pct
                pnl_dollars = self.balance * realized_pnl
                self.balance += pnl_dollars

                self.total_trades += 1
                if realized_pnl > 0:
                    self.winning_trades += 1
                    reward += 3.0 + realized_pnl * 20.0
                    trade_event = "win_trade"
                else:
                    reward -= 4.0 + abs(realized_pnl) * 20.0
                    trade_event = "loss_trade"

                self.position = 0

            # Открываем новую позицию (если target_pos != 0)
            if target_pos != 0:
                self.position = target_pos
                self.entry_price = curr_price
                # Комиссия за открытие сделки
                self.balance *= (1.0 - self.fee_pct)
                reward -= 0.05  # Небольшой штраф за избыточные комиссии при частой торговле

        # Награда за удержание прибыльной позиции (Holding Gain)
        if self.position != 0:
            step_return = (next_price - curr_price) / curr_price
            if self.position == 1:
                step_pnl = step_return
            else:
                step_pnl = -step_return

            reward += step_pnl * 10.0

        self.pnl_history.append(self.balance)
        done = (self.current_step >= len(self.prices) - 2 or self.balance <= self.initial_balance * 0.3)

        info = {
            'balance': self.balance,
            'position': self.position,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': (self.winning_trades / self.total_trades * 100.0) if self.total_trades > 0 else 0.0,
            'event': trade_event
        }

        return self._get_observation(), reward, done, info
