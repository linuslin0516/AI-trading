import hashlib
import hmac
import logging
import time
from datetime import datetime, timezone

import numpy as np
import requests

logger = logging.getLogger(__name__)

# 可用的端點
MARKET_DATA_URL = "https://data-api.binance.vision"
FUTURES_URL = "https://testnet.binancefuture.com"


class MarketData:
    def __init__(self, config: dict):
        self.config = config
        binance_cfg = config["binance"]
        self.symbols = binance_cfg.get("symbols", ["BTCUSDT", "ETHUSDT"])
        self.api_key = binance_cfg.get("api_key", "")
        self.api_secret = binance_cfg.get("api_secret", "")
        self.session = requests.Session()
        logger.info("MarketData initialized (data-api + futures testnet)")

    def _market_get(self, path: str, params: dict | None = None) -> dict | list:
        url = f"{MARKET_DATA_URL}{path}"
        r = self.session.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def _futures_get(self, path: str, params: dict | None = None) -> dict | list:
        url = f"{FUTURES_URL}{path}"
        r = self.session.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_symbol_data(self, symbol: str) -> dict:
        """獲取單一幣種的完整市場數據"""
        try:
            data = {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # 即時價格
            ticker = self._market_get("/api/v3/ticker/price", {"symbol": symbol})
            data["price"] = float(ticker["price"])

            # 24h 統計
            stats = self._market_get("/api/v3/ticker/24hr", {"symbol": symbol})
            data["price_change_24h"] = float(stats["priceChangePercent"])
            data["volume_24h"] = float(stats["quoteVolume"])
            data["high_24h"] = float(stats["highPrice"])
            data["low_24h"] = float(stats["lowPrice"])

            # K 線數據
            data["klines"] = self._get_klines(symbol)

            # 技術指標（基於 1h 收盤價）
            data["technical_indicators"] = self._calc_indicators(data["klines"])

            # 15m 技術指標（基於 15m 收盤價，用於入場時機判斷）
            data["technical_indicators_15m"] = self._calc_15m_indicators(data["klines"])

            # 收盤價趨勢摘要（讓 AI 快速理解趨勢方向，不受影子線干擾）
            data["close_trend"] = self._close_trend_summary(data["klines"])

            # 資金費率（期貨）
            data["funding_rate"] = self._get_funding_rate(symbol)

            # 持倉比例
            data["long_short_ratio"] = self._get_long_short_ratio(symbol)

            return data

        except Exception as e:
            logger.error("Error fetching data for %s: %s", symbol, e)
            return {"symbol": symbol, "error": str(e)}

    def get_all_symbols_data(self) -> dict[str, dict]:
        result = {}
        for symbol in self.symbols:
            result[symbol] = self.get_symbol_data(symbol)
        return result

    def _get_klines(self, symbol: str) -> dict:
        """多時間週期 K 線（各週期取適當數量，避免過多噪音）"""
        # 各時間週期的 K 線數量（不需要都取 100 根）
        interval_limits = {
            "5m": 30,    # 2.5 小時
            "15m": 40,   # 10 小時
            "1h": 48,    # 2 天
            "4h": 30,    # 5 天
            "1d": 14,    # 2 週
        }
        result = {}
        for interval, limit in interval_limits.items():
            try:
                klines = self._market_get("/api/v3/klines", {
                    "symbol": symbol, "interval": interval, "limit": limit
                })
                result[interval] = [
                    {
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                        "timestamp": k[0],
                    }
                    for k in klines
                ]
            except Exception as e:
                logger.warning("Failed to get %s klines for %s: %s", interval, symbol, e)
                result[interval] = []
        return result

    def _calc_indicators(self, klines: dict) -> dict:
        """計算技術指標（基於 1h K 線）"""
        hourly = klines.get("1h", [])
        if len(hourly) < 30:
            return {}

        closes = np.array([k["close"] for k in hourly])
        volumes = np.array([k["volume"] for k in hourly])

        indicators = {}

        # RSI (14)
        indicators["RSI_14"] = self._calc_rsi(closes, 14)

        # MACD
        macd, signal, hist = self._calc_macd(closes)
        if hist[-1] > 0 and hist[-2] <= 0:
            indicators["MACD"] = "bullish_cross"
        elif hist[-1] < 0 and hist[-2] >= 0:
            indicators["MACD"] = "bearish_cross"
        elif hist[-1] > 0:
            indicators["MACD"] = "bullish"
        else:
            indicators["MACD"] = "bearish"
        indicators["MACD_histogram"] = round(float(hist[-1]), 4)

        # 布林通道
        bb_upper, bb_mid, bb_lower = self._calc_bollinger(closes, 20)
        current = closes[-1]
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            bb_position = (current - bb_lower) / bb_range
        else:
            bb_position = 0.5
        indicators["BB_upper"] = round(float(bb_upper), 2)
        indicators["BB_middle"] = round(float(bb_mid), 2)
        indicators["BB_lower"] = round(float(bb_lower), 2)
        if bb_position > 0.8:
            indicators["BB_position"] = "upper"
        elif bb_position < 0.2:
            indicators["BB_position"] = "lower"
        elif bb_position > 0.5:
            indicators["BB_position"] = "mid_to_upper"
        else:
            indicators["BB_position"] = "mid_to_lower"

        # 成交量變化
        avg_vol = np.mean(volumes[-20:])
        recent_vol = np.mean(volumes[-3:])
        if avg_vol > 0:
            indicators["volume_surge"] = round(
                float((recent_vol - avg_vol) / avg_vol), 4
            )
        else:
            indicators["volume_surge"] = 0

        # EMA
        indicators["EMA_7"] = round(float(self._calc_ema(closes, 7)), 2)
        indicators["EMA_25"] = round(float(self._calc_ema(closes, 25)), 2)
        indicators["EMA_99"] = round(float(self._calc_ema(closes, min(99, len(closes)))), 2)

        # 趨勢判斷
        if indicators["EMA_7"] > indicators["EMA_25"] > indicators["EMA_99"]:
            indicators["trend"] = "strong_bullish"
        elif indicators["EMA_7"] > indicators["EMA_25"]:
            indicators["trend"] = "bullish"
        elif indicators["EMA_7"] < indicators["EMA_25"] < indicators["EMA_99"]:
            indicators["trend"] = "strong_bearish"
        elif indicators["EMA_7"] < indicators["EMA_25"]:
            indicators["trend"] = "bearish"
        else:
            indicators["trend"] = "neutral"

        return indicators

    def _close_trend_summary(self, klines: dict) -> dict:
        """產出各時間週期的收盤價趨勢摘要（去除影子線噪音）"""
        summary = {}

        for interval in ["15m", "1h", "4h"]:
            candles = klines.get(interval, [])
            if len(candles) < 10:
                continue

            closes = [c["close"] for c in candles]
            recent_closes = closes[-10:]  # 最近 10 根

            # 收盤價方向：最近 10 根中漲 vs 跌的比例
            ups = sum(1 for i in range(1, len(recent_closes)) if recent_closes[i] > recent_closes[i - 1])
            downs = len(recent_closes) - 1 - ups
            if ups > downs + 2:
                direction = "bullish"
            elif downs > ups + 2:
                direction = "bearish"
            else:
                direction = "sideways"

            # 收盤價變化幅度
            pct_change = (recent_closes[-1] - recent_closes[0]) / recent_closes[0] * 100

            # 最近 3 根收盤價（最重要的即時趨勢）
            last_3 = recent_closes[-3:]
            if last_3[-1] > last_3[-2] > last_3[-3]:
                momentum = "accelerating_up"
            elif last_3[-1] < last_3[-2] < last_3[-3]:
                momentum = "accelerating_down"
            elif last_3[-1] > last_3[-2]:
                momentum = "turning_up"
            elif last_3[-1] < last_3[-2]:
                momentum = "turning_down"
            else:
                momentum = "flat"

            # 收盤價的支撐/壓力位（基於收盤價，非影子線）
            sorted_closes = sorted(closes[-30:])
            support = round(sorted_closes[2], 2)   # 低位第 3 根收盤價
            resistance = round(sorted_closes[-3], 2)  # 高位第 3 根收盤價

            summary[interval] = {
                "direction": direction,
                "momentum": momentum,
                "change_pct": round(pct_change, 3),
                "current_close": round(recent_closes[-1], 2),
                "close_support": support,
                "close_resistance": resistance,
                "recent_3_closes": [round(c, 2) for c in last_3],
            }

        return summary

    def _calc_15m_indicators(self, klines: dict) -> dict:
        """計算 15m 技術指標（基於收盤價）"""
        candles = klines.get("15m", [])
        if len(candles) < 30:
            return {}

        closes = np.array([c["close"] for c in candles])
        indicators = {}

        indicators["RSI_14"] = self._calc_rsi(closes, 14)

        macd, signal, hist = self._calc_macd(closes)
        if hist[-1] > 0 and hist[-2] <= 0:
            indicators["MACD"] = "bullish_cross"
        elif hist[-1] < 0 and hist[-2] >= 0:
            indicators["MACD"] = "bearish_cross"
        elif hist[-1] > 0:
            indicators["MACD"] = "bullish"
        else:
            indicators["MACD"] = "bearish"

        indicators["EMA_7"] = round(float(self._calc_ema(closes, 7)), 2)
        indicators["EMA_25"] = round(float(self._calc_ema(closes, 25)), 2)

        if indicators["EMA_7"] > indicators["EMA_25"]:
            indicators["trend"] = "bullish"
        elif indicators["EMA_7"] < indicators["EMA_25"]:
            indicators["trend"] = "bearish"
        else:
            indicators["trend"] = "neutral"

        return indicators

    @staticmethod
    def _calc_rsi(closes: np.ndarray, period: int = 14) -> float:
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 2)

    @staticmethod
    def _calc_macd(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
        def ema(data, span):
            alpha = 2 / (span + 1)
            result = np.zeros_like(data, dtype=float)
            result[0] = data[0]
            for i in range(1, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
            return result

        ema_fast = ema(closes, fast)
        ema_slow = ema(closes, slow)
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def _calc_bollinger(closes: np.ndarray, period: int = 20, std_dev: float = 2.0):
        mid = np.mean(closes[-period:])
        std = np.std(closes[-period:])
        return mid + std_dev * std, mid, mid - std_dev * std

    @staticmethod
    def _calc_ema(closes: np.ndarray, span: int) -> float:
        alpha = 2 / (span + 1)
        result = closes[0]
        for i in range(1, len(closes)):
            result = alpha * closes[i] + (1 - alpha) * result
        return result

    def _get_funding_rate(self, symbol: str) -> float | None:
        try:
            data = self._futures_get("/fapi/v1/fundingRate", {
                "symbol": symbol, "limit": 1
            })
            if data:
                return float(data[-1]["fundingRate"])
        except Exception:
            pass
        return None

    def _get_long_short_ratio(self, symbol: str) -> float | None:
        try:
            data = self._futures_get("/futures/data/globalLongShortAccountRatio", {
                "symbol": symbol, "period": "5m", "limit": 1
            })
            if data:
                return float(data[-1]["longShortRatio"])
        except Exception:
            pass
        return None

    def get_current_price(self, symbol: str) -> float | None:
        try:
            ticker = self._market_get("/api/v3/ticker/price", {"symbol": symbol})
            return float(ticker["price"])
        except Exception as e:
            logger.error("Failed to get price for %s: %s", symbol, e)
            return None
