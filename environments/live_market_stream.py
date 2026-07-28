"""
Модуль Потоковых Живых Котировок Binance (Binance Live Market Stream)
Получает настоящие живые тики и свечи секунда в секунду с биржи Binance через открытое API
"""

import urllib.request
import json
import datetime
import time

class BinanceLiveStream:
    """Клиент реального времени котировок Binance"""

    def __init__(self):
        self.symbol_map = {
            "BTC/USDT": "BTCUSDT",
            "ETH/USDT": "ETHUSDT",
            "SOL/USDT": "SOLUSDT",
            "EUR/USD": "EURUSDT",
            "GBP/USD": "GBPUSDT",
            "USD/JPY": "USDJPY",
            "AUD/USD": "AUDUSDT",
            "USD/CHF": "USDCHF",
            "EUR/GBP": "EURGBP"
        }
        self.last_fetch_time = 0
        self.cached_klines = []

    def fetch_live_price(self, symbol="BTC/USDT"):
        """Получить текущую цену тика реального времени с биржи Binance"""
        raw_symbol = self.symbol_map.get(symbol, "BTCUSDT")
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={raw_symbol}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return float(data['price'])
        except Exception:
            return None

    def fetch_live_klines(self, symbol="BTC/USDT", interval="1m", limit=60):
        """Получить массив последних свечей реального времени с биржи Binance"""
        raw_symbol = self.symbol_map.get(symbol, "BTCUSDT")
        url = f"https://api.binance.com/api/v3/klines?symbol={raw_symbol}&interval={interval}&limit={limit}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                dates, prices, ohlcv = [], [], []

                for k in data:
                    open_time = datetime.datetime.fromtimestamp(k[0] / 1000.0)
                    o_p, h_p, l_p, cl_p = float(k[1]), float(k[2]), float(k[3]), float(k[4])
                    vol = float(k[5])

                    candle = {
                        'open': o_p,
                        'high': h_p,
                        'low': l_p,
                        'close': cl_p,
                        'volume': vol
                    }
                    dates.append(open_time)
                    prices.append(cl_p)
                    ohlcv.append(candle)

                return dates, prices, ohlcv
        except Exception as e:
            return None, None, None
