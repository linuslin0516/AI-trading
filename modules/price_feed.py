"""
Binance WebSocket 即時價格串流

訂閱 BTCUSDT / ETHUSDT 的 miniTicker 串流，
將最新價格存在記憶體中供 PaperTrader 使用。
"""

import asyncio
import json
import logging
import time

import aiohttp

logger = logging.getLogger(__name__)

BINANCE_WS_URL = "wss://fstream.binance.com/ws"


class PriceFeed:
    """透過 Binance WebSocket 取得即時價格"""

    def __init__(self, symbols: list[str] | None = None):
        self.symbols = [s.lower() for s in (symbols or ["BTCUSDT", "ETHUSDT"])]
        # 最新價格快取: {"BTCUSDT": 97000.50, "ETHUSDT": 2700.30}
        self._prices: dict[str, float] = {}
        # 上次更新時間戳: {"BTCUSDT": 1700000000.0}
        self._updated_at: dict[str, float] = {}
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None
        self._running = False
        self._reconnect_delay = 1  # 初始重連延遲（秒）
        self._max_reconnect_delay = 60

    def get_price(self, symbol: str) -> float | None:
        """取得最新價格（無阻塞）。若 WebSocket 尚未收到資料則回傳 None。"""
        return self._prices.get(symbol.upper())

    def get_price_age(self, symbol: str) -> float:
        """取得價格的年齡（秒）。若無資料回傳 inf。"""
        ts = self._updated_at.get(symbol.upper())
        if ts is None:
            return float("inf")
        return time.time() - ts

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and not self._ws.closed

    async def start(self):
        """啟動 WebSocket 連線（在背景持續運行）"""
        self._running = True
        logger.info("PriceFeed starting for %s", [s.upper() for s in self.symbols])
        asyncio.create_task(self._run_forever())

    async def stop(self):
        """停止 WebSocket 連線"""
        self._running = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("PriceFeed stopped")

    async def _run_forever(self):
        """持續連線，斷線自動重連"""
        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._running:
                    break
                logger.warning("PriceFeed connection error: %s, reconnecting in %ds...",
                               e, self._reconnect_delay)
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self._max_reconnect_delay
                )

    async def _connect_and_listen(self):
        """建立 WebSocket 連線並監聽價格更新"""
        # 組合串流名稱: btcusdt@miniTicker/ethusdt@miniTicker
        streams = "/".join(f"{s}@miniTicker" for s in self.symbols)
        url = f"{BINANCE_WS_URL}/{streams}"

        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()

        logger.info("PriceFeed connecting to %s", url)
        async with self._session.ws_connect(url, heartbeat=20, timeout=30) as ws:
            self._ws = ws
            self._reconnect_delay = 1  # 連線成功，重置重連延遲
            logger.info("PriceFeed connected")

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        # miniTicker 格式: {"e":"24hrMiniTicker","s":"BTCUSDT","c":"97000.50",...}
                        symbol = data.get("s", "").upper()
                        close_price = data.get("c")
                        if symbol and close_price:
                            price = float(close_price)
                            self._prices[symbol] = price
                            self._updated_at[symbol] = time.time()
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        logger.debug("PriceFeed parse error: %s", e)

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logger.warning("PriceFeed WebSocket closed/error: %s", msg.type)
                    break

        self._ws = None
