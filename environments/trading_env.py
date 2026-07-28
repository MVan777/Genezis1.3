"""
Окружение Биржевого Трейдинга (Trading Environment)
Совместимо со стандартом OpenAI Gym API
Поддерживает работу с крипто/фондовыми свечами (OHLCV), расчет RSI, волатильности и PnL с учетом комиссий.
"""

import numpy as np
import random
import datetime
import time
from environments.data_loader import HistoricalDataLoader, NewsCalendar
from environments.live_market_stream import BinanceLiveStream

class TradingEnv:
    """Биржевое окружение для обучения и бэктестинга Ассоциативного ИИ с поддержкой живого рынка"""

    def __init__(self, df_prices=None, initial_balance=1000.0, fee_pct=0.0005):
        self.initial_balance = initial_balance
        self.fee_pct = fee_pct  # 0.05% комиссия за сделку (стандарт Binance/Bybit)

        self.loader = HistoricalDataLoader()
        self.live_stream = BinanceLiveStream()
        self.mode = "sim"  # "sim": Обучение/Бэктест, "live": Живой Рынок Binance в реальном времени
        self.active_symbol = "BTC/USDT"

        if df_prices is not None and len(df_prices) > 50:
            self.prices = np.array(df_prices, dtype=np.float32)
            self.dates = [datetime.datetime.now() for _ in range(len(self.prices))]
            self.ohlcv = [{'open': float(p), 'high': float(p*1.005), 'low': float(p*0.995), 'close': float(p), 'volume': 1000.0} for p in self.prices]
        else:
            # Загружаем свечи по выбранному тикеру
            self.dates, self.prices, self.ohlcv = self.loader.load_symbol_csv(symbol=self.active_symbol, total_candles=1000)

        # Подключаем макроэкономический новостной календарь с датами и временем
        self.news_calendar = NewsCalendar(total_steps=len(self.prices))
        self.reset()

    def set_mode(self, mode_name):
        """Переключить режим работы: 'sim' (Обучение/Бэктест с 01.01.2021) или 'live' (Живой рынок Binance)"""
        if mode_name in ("sim", "live"):
            self.mode = mode_name
            if self.mode == "live":
                ldates, lprices, lohlcv = self.live_stream.fetch_live_klines(symbol=self.active_symbol, interval="1m", limit=60)
                if lprices and len(lprices) > 30:
                    self.dates, self.prices, self.ohlcv = ldates, np.array(lprices, dtype=np.float32), lohlcv
            else:
                self.dates, self.prices, self.ohlcv = self.loader.load_symbol_csv(symbol=self.active_symbol)

            self.current_step = max(30, len(self.prices) - 1) if self.mode == "live" else 30
            self.reset(preserve_step=(self.mode == "sim"))
            return True
        return False

    def update_live_tick(self):
        """Обновить свечи с живого рынка Binance при работе в режиме 'live' (без прокрутки истории)"""
        if self.mode == "live":
            ldates, lprices, lohlcv = self.live_stream.fetch_live_klines(symbol=self.active_symbol, interval="1m", limit=60)
            if lprices and len(lprices) > 30:
                self.dates, self.prices, self.ohlcv = ldates, np.array(lprices, dtype=np.float32), lohlcv
                self.current_step = max(30, len(self.prices) - 1)

    def change_symbol(self, symbol_name):
        """Смена активной торговой пары (Crypto & Forex)"""
        valid_symbols = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF", "EUR/GBP")
        if symbol_name in valid_symbols:
            self.active_symbol = symbol_name
            if self.mode == "live":
                ldates, lprices, lohlcv = self.live_stream.fetch_live_klines(symbol=self.active_symbol, interval="1m", limit=60)
                if lprices and len(lprices) > 30:
                    self.dates, self.prices, self.ohlcv = ldates, np.array(lprices, dtype=np.float32), lohlcv
                else:
                    self.dates, self.prices, self.ohlcv = self.loader.load_symbol_csv(symbol=symbol_name, total_candles=1000)
            else:
                self.dates, self.prices, self.ohlcv = self.loader.load_symbol_csv(symbol=symbol_name, total_candles=1000)

            self.news_calendar = NewsCalendar(total_steps=len(self.prices))
            self.current_step = max(30, len(self.prices) - 1) if self.mode == "live" else 30
            self.reset(preserve_step=True)
            return True
        return False

    def reset(self, preserve_step=False):
        """Сброс состояния торговли на начало графика (или сохранение текущего шага при preserve_step=True)"""
        if not preserve_step and self.mode != "live":
            self.current_step = 30  # Начинаем после накопительного окна индикаторов
        elif self.mode == "live":
            self.current_step = max(30, len(self.prices) - 1)

        self.balance = self.initial_balance
        self.position = 0  # 0: FLAT, 1: LONG, -1: SHORT
        self.entry_price = 0.0
        self.selected_expiry_steps = 15  # По умолчанию 15 мин / шагов (опции: 5, 15, 30, 60)
        self.active_options = []
        self.completed_options = []

        self.total_trades = 0
        self.winning_trades = 0
        self.trades_history = []
        self.pnl_history = [self.balance]

        return self._get_observation()

    def set_expiry_steps(self, steps):
        """Изменить выбранное время экспирации (5, 15, 30, 60)"""
        if steps in (5, 15, 30, 60):
            self.selected_expiry_steps = steps
            return True
        return False

    def _calculate_rsi(self, window=14):
        """Расчёт индекса относительной силы (RSI) с защитой границ"""
        start_i = max(0, self.current_step - window)
        sub_prices = self.prices[start_i:self.current_step + 1]
        if len(sub_prices) < 2:
            return 0.5

        diffs = np.diff(sub_prices)
        gains = np.where(diffs > 0, diffs, 0)
        losses = np.where(diffs < 0, -diffs, 0)

        avg_gain = np.mean(gains) if len(gains) > 0 else 1e-8
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-8

        if avg_loss == 0:
            return 1.0
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi / 100.0

    def _get_observation(self):
        """
        Вектор состояния биржи R^10 Двойного Анализа для Ассоциативного ИИ:
        [ret_1, ret_5, ret_15, rsi_norm, vola_norm, pos_norm, unrealized_pnl, expiry_norm, news_sentiment, news_vola]
        """
        curr_price = self.prices[self.current_step]

        idx_1 = max(0, self.current_step - 1)
        idx_5 = max(0, self.current_step - 5)
        idx_15 = max(0, self.current_step - 15)

        ret_1 = (curr_price - self.prices[idx_1]) / self.prices[idx_1] if self.prices[idx_1] > 0 else 0.0
        ret_5 = (curr_price - self.prices[idx_5]) / self.prices[idx_5] if self.prices[idx_5] > 0 else 0.0
        ret_15 = (curr_price - self.prices[idx_15]) / self.prices[idx_15] if self.prices[idx_15] > 0 else 0.0

        # RSI и Волатильность (стандартное отклонение за 10 баров)
        rsi_norm = self._calculate_rsi(window=14)
        sub = self.prices[max(0, self.current_step - 10):self.current_step + 1]
        vola = np.std(np.diff(sub) / sub[:-1]) if len(sub) > 1 else 0.01
        vola_norm = min(1.0, vola * 20.0)

        # Статус позиции и нереализованный PnL
        pos_norm = float(self.position)
        unrealized_pnl = 0.0
        if self.position == 1:
            unrealized_pnl = (curr_price - self.entry_price) / self.entry_price
        elif self.position == -1:
            unrealized_pnl = (self.entry_price - curr_price) / self.entry_price

        expiry_norm = float(self.selected_expiry_steps) / 60.0

        # Данные новостного календаря по точной дате и времени
        news_info = self.news_calendar.get_news_at_step(self.current_step)
        news_sentiment = float(news_info['sentiment'])
        news_vola_norm = float(news_info['volatility_impact']) / 3.5

        obs = [
            ret_1 * 10.0, ret_5 * 5.0, ret_15 * 3.0, rsi_norm, vola_norm,
            pos_norm, unrealized_pnl * 5.0, expiry_norm,
            news_sentiment, news_vola_norm
        ]
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """
        Выполнить шаг в биржевой торговле с экспирацией
        action: 0 - HOLD / FLAT, 1 - BUY (CALL / UP), 2 - SELL (PUT / DOWN)
        """
        if self.mode == "live":
            self.current_step = max(0, len(self.prices) - 1)
        else:
            self.current_step = min(len(self.prices) - 1, self.current_step + 1)

        curr_price = self.prices[max(0, self.current_step - 1)]
        next_price = self.prices[self.current_step]

        reward = 0.0
        trade_event = "none"
        last_expired_info = None

        # ===== 1. ОТКРЫТИЕ НОВОГО ОПЦИОНА ПО СИГНАЛУ ИИ =====
        if action in (1, 2) and len(self.active_options) < 5:  # Ограничение до 5 одновременных опционов
            opt_direction = 1 if action == 1 else -1
            option_contract = {
                'id': len(self.completed_options) + len(self.active_options) + 1,
                'direction': opt_direction,
                'entry_step': self.current_step,
                'expiry_step': self.current_step + self.selected_expiry_steps,
                'expiry_timestamp': time.time() + (self.selected_expiry_steps * 60) if self.mode == "live" else 0,
                'strike_price': curr_price,
                'amount': 10.0,
                'expiry_setting': self.selected_expiry_steps
            }
            self.active_options.append(option_contract)
            self.position = opt_direction
            self.entry_price = curr_price
            trade_event = "open_option"

        # ===== 2. ПРОВЕРКА ИСТЕЧЕНИЯ СРОКА ЭКСПИРАЦИИ (EXPIRATION CHECK) =====
        remaining_active = []
        now_ts = time.time()

        for opt in self.active_options:
            is_expired = False
            if self.mode == "live" and opt.get('expiry_timestamp', 0) > 0:
                is_expired = (now_ts >= opt['expiry_timestamp'])
                opt['remaining_wall_seconds'] = max(0, int(opt['expiry_timestamp'] - now_ts))
            else:
                is_expired = (self.current_step >= opt['expiry_step'])
                opt['remaining_wall_seconds'] = max(0, (opt['expiry_step'] - self.current_step) * 60)

            if is_expired:
                strike = opt['strike_price']
                exit_price = curr_price
                direction = opt['direction']

                # Проверка успешности прогноза
                if direction == 1:
                    is_win = (exit_price > strike)
                else:
                    is_win = (exit_price < strike)

                self.total_trades += 1

                if is_win:
                    payout = opt['amount'] * 0.85  # Прибыль +85%
                    self.balance += payout
                    reward += 5.0
                    opt['result'] = 'WIN'
                    opt['profit'] = payout
                    self.winning_trades += 1
                    trade_event = "option_win"
                else:
                    loss = opt['amount']
                    self.balance -= loss
                    reward -= 6.0
                    opt['result'] = 'LOSS'
                    opt['profit'] = -loss
                    trade_event = "option_loss"

                opt['exit_price'] = exit_price
                opt['exit_step'] = self.current_step
                self.completed_options.append(opt)
                last_expired_info = opt
            else:
                remaining_active.append(opt)

        self.active_options = remaining_active

        # Если нет активных опционов — статус FLAT
        if not self.active_options:
            self.position = 0

        # Если баланс упал ниже $300 (ликвидация): пополняем депозит и наказываем ИИ без сброса назад!
        if self.balance <= self.initial_balance * 0.3:
            self.liquidations = getattr(self, 'liquidations', 0) + 1
            reward -= 20.0
            self.balance = self.initial_balance
            trade_event = "liquidation_penalty_refill"

        self.pnl_history.append(self.balance)

        # Непрерывность движения: закольцовываем массив при достижении конца без прыжков в ноль
        if self.current_step >= len(self.prices) - 2:
            self.current_step = 30
            done = True
        else:
            done = False

        news_info = self.news_calendar.get_news_at_step(self.current_step)

        # Определение фазового состояния рынка (Market Regime & Strategy Reasoning)
        curr_p = self.prices[self.current_step]
        prev_15_p = self.prices[max(0, self.current_step - 15)]
        ret_15 = (curr_p - prev_15_p) / prev_15_p if prev_15_p > 0 else 0.0

        if news_info.get('active', False):
            regime = "NEWS_SPIKE"
            regime_label = "⚡ ИМПУЛЬС НОВОСТЕЙ"
            strategy = "News Momentum (Реакция на новости)"
        elif ret_15 > 0.025:
            regime = "TREND_UP"
            regime_label = "📈 БЫЧИЙ ТРЕНД"
            strategy = "Trend Following (Следование тренду)"
        elif ret_15 < -0.025:
            regime = "TREND_DOWN"
            regime_label = "📉 МЕДВЕЖИЙ ТРЕНД"
            strategy = "Trend Following (Следование тренду)"
        else:
            regime = "FLAT_CONSOLIDATION"
            regime_label = "↔️ КОНСОЛИДАЦИЯ / ФЛЭТ"
            strategy = "Mean Reversion (Отбой от границ)"

        info = {
            'balance': self.balance,
            'position': self.position,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': (self.winning_trades / self.total_trades * 100.0) if self.total_trades > 0 else 0.0,
            'liquidations': getattr(self, 'liquidations', 0),
            'selected_expiry': self.selected_expiry_steps,
            'active_options': self.active_options,
            'last_expired': last_expired_info,
            'news_info': news_info,
            'current_date': self.dates[self.current_step].strftime("%Y-%m-%d %H:%M") if hasattr(self, 'dates') and self.current_step < len(self.dates) else "",
            'market_regime': regime,
            'market_regime_label': regime_label,
            'ai_strategy': strategy,
            'active_symbol': self.active_symbol,
            'mode': self.mode,
            'event': trade_event
        }

        return self._get_observation(), reward, done, info
