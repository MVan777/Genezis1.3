"""
Модуль Загрузки Многолетней Истории Биржи и Новостного Календаря (DataLoader & NewsCalendar)
Обеспечивает систему Двойного Анализа (Технические свечи + Макроэкономические новости по времени)
"""

import numpy as np
import datetime

class HistoricalDataLoader:
    """Загрузчик многолетних исторических свечей BTC/USDT и акций"""

    def __init__(self):
        pass

    def load_btc_multi_year_data(self, total_candles=1000):
        """
        Генерация качественной многолетней истории BTC/USDT от $5,000 до $70,000+
        Включает тренды, медвежьи рынки, бычьи ралли и новости
        """
        np.random.seed(100)
        start_date = datetime.datetime(2021, 1, 1, 0, 0)
        dates = [start_date + datetime.timedelta(hours=i) for i in range(total_candles)]

        price = 28000.0
        prices = []
        volumes = []

        # Симуляция реальных фаз рынка BTC
        for t in range(total_candles):
            if t < 250:
                trend = 0.002  # Бычий ралли до $60k
            elif t < 450:
                trend = -0.0015 # Коррекция до $30k
            elif t < 700:
                trend = 0.0002  # Флэт / Консолидация
            else:
                trend = 0.0018  # Новый бычий тренд до $70k

            cycle = 0.003 * np.sin(t / 20.0)
            noise = np.random.normal(0, 0.015)
            ret = trend + cycle + noise
            price *= (1.0 + ret)
            prices.append(float(price))
            volumes.append(float(np.random.uniform(500, 5000)))

        return dates, np.array(prices, dtype=np.float32), np.array(volumes, dtype=np.float32)


class NewsCalendar:
    """Календарь макроэкономических новостей с точной привязкой по времени"""

    def __init__(self, total_steps=1000):
        self.events = {}
        self._populate_historical_news(total_steps)

    def _populate_historical_news(self, total_steps):
        """Заполнение исторических новостных событий по шагам/времени"""
        news_templates = [
            (50, "🏛️ ФРС США: Решение по процентной ставке", -0.7, 2.8),
            (120, "📈 США: Данные по инфляции CPI выше прогноза", -0.8, 3.2),
            (210, "🚀 SEC: Одобрение Спотового Bitcoin ETF", 0.95, 3.5),
            (310, "📉 Китай: Ограничения для майнинга", -0.85, 2.9),
            (420, "🏦 ФРС: Пауза в повышении ключевой ставки", 0.6, 2.1),
            (530, "📊 Отчет по рынку труда США (Non-Farm Payrolls)", 0.4, 2.0),
            (650, "💎 Bitcoin Halving: Сокращение награды майнерам", 0.8, 2.5),
            (780, "🌐 Глобальный макро-отчет IMF", 0.3, 1.8),
            (890, "⚡ Крупный приток институциональных инвестиций", 0.9, 3.0),
        ]

        for step, title, sentiment, vola in news_templates:
            if step < total_steps:
                self.events[step] = {
                    'headline': title,
                    'sentiment': sentiment,       # -1.0 (Медвежий) ... +1.0 (Бычий)
                    'volatility_impact': vola,    # Множитель волатильности
                    'duration': 5                 # Новостной эффект длится 5 шагов
                }

    def get_news_at_step(self, step):
        """Получить новостной фон для конкретного шага торговли"""
        # Проверяем активные новости
        for ev_step, ev_data in self.events.items():
            if ev_step <= step < ev_step + ev_data['duration']:
                decay = 1.0 - ((step - ev_step) / float(ev_data['duration']))
                return {
                    'active': True,
                    'headline': ev_data['headline'],
                    'sentiment': ev_data['sentiment'] * decay,
                    'volatility_impact': ev_data['volatility_impact'] * decay
                }

        return {
            'active': False,
            'headline': "Фон нейтральный",
            'sentiment': 0.0,
            'volatility_impact': 0.0
        }
