import asyncio
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timezone

import requests

from modules.database import Database

logger = logging.getLogger(__name__)

FUTURES_URL = "https://testnet.binancefuture.com"
MARKET_DATA_URL = "https://data-api.binance.vision"


class BinanceTrader:
    def __init__(self, config: dict, db: Database):
        self.config = config
        self.db = db

        binance_cfg = config["binance"]
        trading_cfg = config.get("trading", {})
        self.leverage_map = trading_cfg.get("leverage_map", {})
        self.default_leverage = trading_cfg.get("default_leverage", 25)
        self.api_key = binance_cfg.get("api_key", "")
        self.api_secret = binance_cfg.get("api_secret", "")

        # 手續費設定
        fees_cfg = trading_cfg.get("fees", {})
        self.taker_rate = fees_cfg.get("taker_rate", 0.0004)
        self.maker_rate = fees_cfg.get("maker_rate", 0.0002)
        self.slippage = fees_cfg.get("slippage", 0.0001)

        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})

        logger.info("BinanceTrader initialized (futures testnet, leverage=%s, taker=%.4f%%)",
                     self.leverage_map or self.default_leverage, self.taker_rate * 100)

    def _sign(self, params: dict) -> dict:
        """為請求添加簽名"""
        params["timestamp"] = int(time.time() * 1000)
        query = "&".join(f"{k}={v}" for k, v in params.items())
        signature = hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params

    def _futures_get(self, path: str, params: dict | None = None, signed: bool = False):
        params = params or {}
        if signed:
            params = self._sign(params)
        r = self.session.get(f"{FUTURES_URL}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def _futures_post(self, path: str, params: dict | None = None):
        params = self._sign(params or {})
        r = self.session.post(f"{FUTURES_URL}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def _futures_delete(self, path: str, params: dict | None = None):
        params = self._sign(params or {})
        r = self.session.delete(f"{FUTURES_URL}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def execute_trade(self, decision: dict) -> dict:
        """根據 AI 決策下單"""
        symbol = decision["symbol"]
        action = decision["action"]  # LONG / SHORT
        entry_price = decision["entry"]["price"]
        strategy = decision["entry"].get("strategy", "LIMIT")
        stop_loss = decision["stop_loss"]
        take_profit = decision["take_profit"]  # list
        position_size_pct = decision["position_size"]

        leverage = self.leverage_map.get(symbol, self.default_leverage)

        try:
            # 1. 設定槓桿
            try:
                self._futures_post("/fapi/v1/leverage", {
                    "symbol": symbol, "leverage": leverage
                })
                logger.info("Leverage set to %dx for %s", leverage, symbol)
            except Exception as e:
                logger.warning("Failed to set leverage: %s", e)

            # 2. 計算數量
            quantity = self._calc_quantity(symbol, entry_price, position_size_pct)
            if quantity <= 0:
                return {"success": False, "error": "Invalid quantity"}

            # 3. 確定方向
            side = "BUY" if action == "LONG" else "SELL"

            # 4. 下單
            if strategy == "MARKET":
                order = self._place_market_order(symbol, side, quantity)
            else:
                order = self._place_limit_order(symbol, side, quantity, entry_price)

            if not order:
                return {"success": False, "error": "Order placement failed"}

            order_id = str(order.get("orderId", ""))

            # 5. 建立 trade 記錄
            trade = self.db.create_trade(
                symbol=symbol,
                direction=action,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size_pct,
                leverage=leverage,
                confidence=decision.get("confidence"),
                ai_reasoning=decision.get("reasoning", {}).get("analyst_consensus", ""),
                analyst_opinions=decision.get("_analyst_messages", []),
                technical_signals=decision.get("reasoning", {}),
                entry_order_id=order_id,
                status="OPEN",
                market_condition=decision.get("_market_condition"),
            )

            # 6. 設定停損停利
            self._set_sl_tp(symbol, action, quantity, stop_loss, take_profit)

            logger.info(
                "Trade executed: #%d %s %s @ %s qty=%s",
                trade.id, action, symbol, entry_price, quantity,
            )

            return {
                "success": True,
                "trade_id": trade.id,
                "order_id": order_id,
                "symbol": symbol,
                "direction": action,
                "entry_price": entry_price,
                "quantity": quantity,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }

        except Exception as e:
            logger.error("Trade execution error: %s", e)
            return {"success": False, "error": str(e)}

    def _calc_quantity(self, symbol: str, price: float, position_pct: float) -> float:
        """計算下單數量"""
        try:
            account = self._futures_get("/fapi/v2/account", signed=True)
            balance = float(account["totalWalletBalance"])

            leverage = self.leverage_map.get(symbol, self.default_leverage)
            amount_usdt = balance * (position_pct / 100) * leverage
            quantity = amount_usdt / price

            # 取得交易對精度
            exchange_info = self._futures_get("/fapi/v1/exchangeInfo")
            for s in exchange_info.get("symbols", []):
                if s["symbol"] == symbol:
                    for f in s["filters"]:
                        if f["filterType"] == "LOT_SIZE":
                            step = float(f["stepSize"])
                            precision = len(str(step).rstrip("0").split(".")[-1])
                            quantity = round(quantity, precision)
                            break
                    break

            return quantity

        except Exception as e:
            logger.error("Failed to calculate quantity: %s", e)
            return 0

    def _place_market_order(self, symbol: str, side: str, quantity: float) -> dict | None:
        try:
            order = self._futures_post("/fapi/v1/order", {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": quantity,
            })
            logger.info("Market order placed: %s", order.get("orderId"))
            return order
        except Exception as e:
            logger.error("Market order failed: %s", e)
            return None

    def _place_limit_order(self, symbol: str, side: str,
                           quantity: float, price: float) -> dict | None:
        try:
            # 取得價格精度
            exchange_info = self._futures_get("/fapi/v1/exchangeInfo")
            for s in exchange_info.get("symbols", []):
                if s["symbol"] == symbol:
                    for f in s["filters"]:
                        if f["filterType"] == "PRICE_FILTER":
                            tick_size = float(f["tickSize"])
                            precision = len(str(tick_size).rstrip("0").split(".")[-1])
                            price = round(price, precision)
                            break
                    break

            order = self._futures_post("/fapi/v1/order", {
                "symbol": symbol,
                "side": side,
                "type": "LIMIT",
                "timeInForce": "GTC",
                "quantity": quantity,
                "price": price,
            })
            logger.info("Limit order placed: %s @ %s", order.get("orderId"), price)
            return order
        except Exception as e:
            logger.error("Limit order failed: %s", e)
            return None

    def _set_sl_tp(self, symbol: str, direction: str,
                   quantity: float, stop_loss: float,
                   take_profit: list[float]):
        """設定停損和停利單"""
        close_side = "SELL" if direction == "LONG" else "BUY"

        # 取得數量精度
        qty_precision = 8
        try:
            exchange_info = self._futures_get("/fapi/v1/exchangeInfo")
            for s in exchange_info.get("symbols", []):
                if s["symbol"] == symbol:
                    for f in s["filters"]:
                        if f["filterType"] == "LOT_SIZE":
                            step = float(f["stepSize"])
                            qty_precision = len(str(step).rstrip("0").split(".")[-1])
                            break
                    break
        except Exception as e:
            logger.warning("Failed to get quantity precision: %s", e)

        # 停損：不掛交易所 STOP_MARKET（testnet 插針會誤觸發）
        # 改由 monitor_positions 本地監控，連續 2 次確認才平倉
        logger.info("Stop loss at %s (local monitor, no exchange order)", stop_loss)

        if not take_profit:
            return

        if len(take_profit) >= 2:
            # 有兩個目標：TP1 平 50%，TP2 平剩餘全部
            # TP1（平倉 50%）
            try:
                tp_qty = round(quantity * 0.5, qty_precision)
                self._futures_post("/fapi/v1/order", {
                    "symbol": symbol,
                    "side": close_side,
                    "type": "TAKE_PROFIT_MARKET",
                    "stopPrice": take_profit[0],
                    "quantity": tp_qty,
                })
                logger.info("Take profit 1 set at %s (qty=%s)", take_profit[0], tp_qty)
            except Exception as e:
                logger.warning("Failed to set TP1: %s", e)

            # TP2（closePosition 平剩餘全部）
            try:
                self._futures_post("/fapi/v1/order", {
                    "symbol": symbol,
                    "side": close_side,
                    "type": "TAKE_PROFIT_MARKET",
                    "stopPrice": take_profit[1],
                    "closePosition": "true",
                })
                logger.info("Take profit 2 set at %s (close all)", take_profit[1])
            except Exception as e:
                logger.warning("Failed to set TP2: %s", e)
        else:
            # 只有一個目標：全部平倉
            try:
                self._futures_post("/fapi/v1/order", {
                    "symbol": symbol,
                    "side": close_side,
                    "type": "TAKE_PROFIT_MARKET",
                    "stopPrice": take_profit[0],
                    "closePosition": "true",
                })
                logger.info("Take profit set at %s (close all)", take_profit[0])
            except Exception as e:
                logger.warning("Failed to set TP: %s", e)

    def resync_sl_tp(self, trade_id: int) -> dict:
        """重新設定指定交易的 SL/TP 掛單（修復遺失的掛單）"""
        trade = self.db.get_trade(trade_id)
        if not trade:
            return {"success": False, "error": "Trade not found"}
        if trade.status == "CLOSED":
            return {"success": False, "error": "Trade already closed"}

        symbol = trade.symbol
        direction = trade.direction

        try:
            # 1. 取消該交易對所有掛單
            try:
                self._futures_delete("/fapi/v1/allOpenOrders", {"symbol": symbol})
                logger.info("Cancelled existing orders for %s", symbol)
            except Exception as e:
                logger.warning("Failed to cancel existing orders: %s", e)

            # 2. 查詢實際持倉數量
            positions = self._futures_get("/fapi/v2/positionRisk", signed=True)
            pos_qty = 0
            for pos in positions:
                if pos["symbol"] == symbol:
                    pos_qty = abs(float(pos["positionAmt"]))
                    break

            if pos_qty <= 0:
                return {"success": False, "error": f"No position found on Binance for {symbol}"}

            # 3. 解析 TP
            tp = trade.take_profit
            if isinstance(tp, str):
                tp = json.loads(tp)

            # 4. 重新掛 SL/TP
            self._set_sl_tp(symbol, direction, pos_qty, trade.stop_loss, tp or [])

            logger.info("Resynced SL/TP for trade #%d: SL=%s TP=%s qty=%s",
                        trade_id, trade.stop_loss, tp, pos_qty)

            return {
                "success": True,
                "trade_id": trade_id,
                "stop_loss": trade.stop_loss,
                "take_profit": tp,
                "quantity": pos_qty,
            }

        except Exception as e:
            logger.error("Failed to resync SL/TP for trade #%d: %s", trade_id, e)
            return {"success": False, "error": str(e)}

    def adjust_trade(self, trade_id: int, new_stop_loss: float | None = None,
                     new_take_profit: list[float] | None = None) -> dict:
        """根據 AI 決策調整現有持倉的止盈止損"""
        trade = self.db.get_trade(trade_id)
        if not trade:
            return {"success": False, "error": "Trade not found"}
        if trade.status == "CLOSED":
            return {"success": False, "error": "Trade already closed"}

        symbol = trade.symbol
        direction = trade.direction
        changes = []

        try:
            # 先取消該交易對所有 SL/TP 掛單
            try:
                self._futures_delete("/fapi/v1/allOpenOrders", {"symbol": symbol})
                logger.info("Cancelled existing orders for %s", symbol)
            except Exception as e:
                logger.warning("Failed to cancel existing orders: %s", e)

            # 重新設定 SL/TP
            close_side = "SELL" if direction == "LONG" else "BUY"
            sl = new_stop_loss if new_stop_loss else trade.stop_loss
            tp = new_take_profit if new_take_profit else (
                json.loads(trade.take_profit) if isinstance(trade.take_profit, str) else trade.take_profit
            )

            # 停損：不掛交易所（由本地 monitor 監控）
            logger.info("Stop loss at %s (local monitor)", sl)

            # 停利
            if tp:
                try:
                    self._futures_post("/fapi/v1/order", {
                        "symbol": symbol,
                        "side": close_side,
                        "type": "TAKE_PROFIT_MARKET",
                        "stopPrice": tp[0],
                        "closePosition": "true",
                    })
                    logger.info("New take profit set at %s", tp[0])
                except Exception as e:
                    logger.warning("Failed to set new TP: %s", e)

            # 更新資料庫
            update_fields = {}
            if new_stop_loss:
                update_fields["stop_loss"] = new_stop_loss
                changes.append(f"SL: {trade.stop_loss} -> {new_stop_loss}")
            if new_take_profit:
                update_fields["take_profit"] = new_take_profit
                changes.append(f"TP: {trade.take_profit} -> {new_take_profit}")

            if update_fields:
                self.db.update_trade(trade_id, **update_fields)

            logger.info("Trade #%d adjusted: %s", trade_id, ", ".join(changes))

            return {
                "success": True,
                "trade_id": trade_id,
                "changes": changes,
                "new_stop_loss": sl,
                "new_take_profit": tp,
            }

        except Exception as e:
            logger.error("Failed to adjust trade #%d: %s", trade_id, e)
            return {"success": False, "error": str(e)}

    def calc_fee_pct(self, leverage: int, entry_type: str = "TAKER", exit_type: str = "TAKER") -> float:
        """計算往返手續費 (佔保證金的百分比)

        fee_pct = (entry_fee + exit_fee + slippage*2) * leverage * 100
        """
        entry_rate = self.taker_rate if entry_type == "TAKER" else self.maker_rate
        exit_rate = self.taker_rate if exit_type == "TAKER" else self.maker_rate
        return (entry_rate + exit_rate + self.slippage * 2) * leverage * 100

    def cancel_all_orders(self, symbol: str) -> dict:
        """取消指定交易對的所有掛單"""
        try:
            result = self._futures_delete("/fapi/v1/allOpenOrders", {"symbol": symbol})
            logger.info("Cancelled all open orders for %s", symbol)
            return {"success": True, "result": result}
        except Exception as e:
            logger.error("Failed to cancel orders for %s: %s", symbol, e)
            return {"success": False, "error": str(e)}

    def get_recent_orders(self, symbol: str, limit: int = 20) -> list[dict]:
        """查詢 Binance 最近的訂單歷史"""
        try:
            orders = self._futures_get("/fapi/v1/allOrders", {
                "symbol": symbol,
                "limit": limit,
            }, signed=True)
            results = []
            for o in orders[-limit:]:
                results.append({
                    "orderId": o.get("orderId"),
                    "type": o.get("type"),
                    "side": o.get("side"),
                    "status": o.get("status"),
                    "price": o.get("avgPrice") or o.get("price"),
                    "stopPrice": o.get("stopPrice"),
                    "qty": o.get("executedQty"),
                    "time": datetime.fromtimestamp(
                        o.get("updateTime", 0) / 1000, tz=timezone.utc
                    ).strftime("%m-%d %H:%M:%S"),
                })
            return results
        except Exception as e:
            logger.error("Failed to get orders: %s", e)
            return []

    def close_trade(self, trade_id: int, exit_price: float | None = None) -> dict:
        """手動或自動平倉"""
        trade = self.db.get_trade(trade_id)
        if not trade:
            return {"success": False, "error": "Trade not found"}

        if trade.status == "CLOSED":
            return {"success": False, "error": "Trade already closed"}

        symbol = trade.symbol
        direction = trade.direction

        try:
            # 1. 取消該交易對所有掛單（SL/TP）
            try:
                self._futures_delete("/fapi/v1/allOpenOrders", {"symbol": symbol})
                logger.info("Cancelled all open orders for %s", symbol)
            except Exception as e:
                logger.warning("Failed to cancel open orders: %s", e)

            # 2. 市價平倉
            close_side = "SELL" if direction == "LONG" else "BUY"
            try:
                # 查詢實際持倉數量
                positions = self._futures_get("/fapi/v2/positionRisk", signed=True)
                pos_qty = 0
                for pos in positions:
                    if pos["symbol"] == symbol:
                        pos_qty = abs(float(pos["positionAmt"]))
                        break

                if pos_qty > 0:
                    self._futures_post("/fapi/v1/order", {
                        "symbol": symbol,
                        "side": close_side,
                        "type": "MARKET",
                        "quantity": pos_qty,
                        "reduceOnly": "true",
                    })
                    logger.info("Closed position: %s %s qty=%s", close_side, symbol, pos_qty)
                else:
                    logger.info("No Binance position found for %s, updating DB only", symbol)
            except Exception as e:
                logger.warning("Market close order failed: %s (updating DB anyway)", e)

            # 3. 取得成交後的價格
            if not exit_price:
                r = self.session.get(
                    f"{MARKET_DATA_URL}/api/v3/ticker/price",
                    params={"symbol": symbol}, timeout=10,
                )
                exit_price = float(r.json()["price"])

            # 計算盈虧（扣除手續費 + 滑點）
            leverage = self.leverage_map.get(symbol, self.default_leverage)
            if direction == "LONG":
                profit_pct = (exit_price - trade.entry_price) / trade.entry_price * 100
            else:
                profit_pct = (trade.entry_price - exit_price) / trade.entry_price * 100

            profit_pct *= leverage

            # 扣除往返手續費
            fee_pct = self.calc_fee_pct(leverage)
            profit_pct -= fee_pct

            # 計算持倉時間
            now = datetime.now(timezone.utc)
            trade_ts = trade.timestamp
            if trade_ts.tzinfo is None:
                trade_ts = trade_ts.replace(tzinfo=timezone.utc)
            hold_duration = int((now - trade_ts).total_seconds())

            # 更新記錄
            outcome = "WIN" if profit_pct > 0 else ("LOSS" if profit_pct < 0 else "BREAKEVEN")
            self.db.update_trade(
                trade_id,
                exit_price=exit_price,
                profit_pct=round(profit_pct, 4),
                profit_usd=round(profit_pct * trade.position_size / 100, 4),
                hold_duration=hold_duration,
                outcome=outcome,
                status="CLOSED",
                closed_at=now,
            )

            logger.info(
                "Trade #%d closed: %s %.2f%% (fee: %.2f%%) @ %s",
                trade_id, outcome, profit_pct, fee_pct, exit_price,
            )

            return {
                "success": True,
                "trade_id": trade_id,
                "exit_price": exit_price,
                "profit_pct": profit_pct,
                "fee_pct": fee_pct,
                "outcome": outcome,
                "hold_duration": hold_duration,
            }

        except Exception as e:
            logger.error("Failed to close trade #%d: %s", trade_id, e)
            return {"success": False, "error": str(e)}

    async def monitor_positions(self, callback=None):
        """持續監控持倉（每 30 秒更新）

        透過查詢 Binance 實際持倉偵測：
        - 持倉歸零 → SL 或 TP2 已在交易所觸發
        - 持倉數量減少 ~50% → TP1 部分止盈已觸發
        - Binance 查詢失敗時 → 回退到本地價格檢查
        """
        logger.info("Position monitor started")

        # 追蹤每筆交易的初始數量和 TP1 通知狀態
        _pos_state: dict[int, dict] = {}

        while True:
            try:
                open_trades = self.db.get_open_trades()
                if not open_trades:
                    _pos_state.clear()
                    await asyncio.sleep(30)
                    continue

                # 批次查詢 Binance 所有持倉（一次 API 呼叫）
                binance_positions: dict[str, dict] = {}
                binance_ok = False
                try:
                    raw_positions = self._futures_get(
                        "/fapi/v2/positionRisk", signed=True
                    )
                    for pos in raw_positions:
                        sym = pos["symbol"]
                        binance_positions[sym] = {
                            "qty": abs(float(pos["positionAmt"])),
                            "mark_price": float(pos.get("markPrice", 0)),
                        }
                    binance_ok = True
                except Exception as e:
                    logger.warning("Failed to query Binance positions: %s", e)

                for trade in open_trades:
                    try:
                        symbol = trade.symbol

                        # 取得當前價格（優先用 Binance mark price）
                        bp = binance_positions.get(symbol, {})
                        current_price = bp.get("mark_price", 0)
                        if not current_price:
                            r = self.session.get(
                                f"{MARKET_DATA_URL}/api/v3/ticker/price",
                                params={"symbol": symbol}, timeout=10,
                            )
                            current_price = float(r.json()["price"])

                        actual_qty = bp.get("qty", -1)  # -1 = 查詢失敗

                        # 初始化追蹤狀態
                        if trade.id not in _pos_state:
                            _pos_state[trade.id] = {
                                "initial_qty": actual_qty if actual_qty > 0 else None,
                                "tp1_notified": False,
                                "sl_breach_count": 0,  # 連續觸及 SL 次數
                            }
                        state = _pos_state[trade.id]

                        # 補記初始數量（首次查到持倉時）
                        if state["initial_qty"] is None and actual_qty > 0:
                            state["initial_qty"] = actual_qty

                        # ── Case A: Binance 持倉已完全平倉 ──
                        if (binance_ok and actual_qty == 0
                                and state["initial_qty"] is not None
                                and state["initial_qty"] > 0):
                            tp_list = (
                                json.loads(trade.take_profit)
                                if isinstance(trade.take_profit, str)
                                else trade.take_profit
                            ) or []

                            # 判斷觸發類型：SL / TP / 強平(liquidation)
                            sl = trade.stop_loss or 0
                            last_tp = tp_list[-1] if tp_list else 0

                            if trade.direction == "LONG":
                                near_sl = sl and current_price <= sl * 1.005
                                near_tp = last_tp and current_price >= last_tp * 0.995
                                # 強平：價格低於止損很多（或介於入場與止損之間的極端位置）
                                liquidated = sl and current_price < sl * 0.99
                            else:
                                near_sl = sl and current_price >= sl * 0.995
                                near_tp = last_tp and current_price <= last_tp * 1.005
                                liquidated = sl and current_price > sl * 1.01

                            if liquidated:
                                event = "liquidation"
                            elif near_sl:
                                event = "stop_loss"
                            elif near_tp:
                                event = "take_profit"
                            else:
                                event = "closed_unknown"

                            logger.warning(
                                "Exchange closed trade #%d (%s %s), event=%s, "
                                "price=%.4f, entry=%.4f, sl=%.4f, tp=%s",
                                trade.id, trade.direction, symbol, event,
                                current_price, trade.entry_price, sl, tp_list,
                            )

                            # 先取消殘留掛單（TP 單等），避免影響未來倉位
                            try:
                                self._futures_delete(
                                    "/fapi/v1/allOpenOrders", {"symbol": symbol}
                                )
                                logger.info("Cleaned up remaining orders for %s", symbol)
                            except Exception as e:
                                logger.warning("Failed to clean up orders for %s: %s", symbol, e)

                            result = self.close_trade(trade.id, current_price)
                            if callback:
                                await callback(event, trade, result)

                            _pos_state.pop(trade.id, None)
                            continue

                        # ── Case B: TP1 部分止盈偵測（數量減少 >30%）──
                        if (actual_qty > 0
                                and state["initial_qty"] is not None
                                and not state["tp1_notified"]
                                and actual_qty < state["initial_qty"] * 0.7):

                            state["tp1_notified"] = True
                            tp_list = (
                                json.loads(trade.take_profit)
                                if isinstance(trade.take_profit, str)
                                else trade.take_profit
                            ) or []

                            logger.info(
                                "TP1 partial close for trade #%d: qty %.6f -> %.6f",
                                trade.id, state["initial_qty"], actual_qty,
                            )

                            # TP1 命中 → 止損移至保本價（成本 + 手續費）
                            old_sl = trade.stop_loss
                            fee_rate = (self.taker_rate + self.taker_rate
                                        + self.slippage * 2)
                            if trade.direction == "LONG":
                                breakeven_price = trade.entry_price * (1 + fee_rate)
                            else:
                                breakeven_price = trade.entry_price * (1 - fee_rate)
                            breakeven_price = round(breakeven_price, 2)
                            self.db.update_trade(trade.id, stop_loss=breakeven_price)
                            logger.info(
                                "Breakeven SL for trade #%d: %.2f -> %.2f "
                                "(entry=%.2f + fee=%.4f%%)",
                                trade.id, old_sl or 0, breakeven_price,
                                trade.entry_price, fee_rate * 100,
                            )

                            if callback:
                                await callback("tp1_hit", trade, {
                                    "current_price": current_price,
                                    "tp1_price": tp_list[0] if tp_list else 0,
                                    "closed_qty": state["initial_qty"] - actual_qty,
                                    "remaining_qty": actual_qty,
                                    "breakeven_sl": breakeven_price,
                                    "old_sl": old_sl,
                                })

                        # ── Case C: 正常監控 + 計算未實現盈虧 ──
                        leverage = self.leverage_map.get(symbol, self.default_leverage)
                        if trade.direction == "LONG":
                            unrealized = (current_price - trade.entry_price) / trade.entry_price * 100
                        else:
                            unrealized = (trade.entry_price - current_price) / trade.entry_price * 100
                        unrealized *= leverage
                        fee_pct = self.calc_fee_pct(leverage)
                        unrealized -= fee_pct

                        # ── Case D: 本地 SL 監控（連續 2 次確認，避免插針） ──
                        sl = trade.stop_loss or 0
                        if sl and actual_qty != 0:
                            sl_breached = (
                                (trade.direction == "LONG" and current_price <= sl) or
                                (trade.direction == "SHORT" and current_price >= sl)
                            )
                            if sl_breached:
                                state["sl_breach_count"] += 1
                                logger.warning(
                                    "SL breach #%d for trade #%d (%s %s): "
                                    "price=%.2f, sl=%.2f",
                                    state["sl_breach_count"], trade.id,
                                    trade.direction, symbol, current_price, sl,
                                )
                                if state["sl_breach_count"] >= 4:
                                    # 連續 4 次確認（~2 分鐘）→ 執行止損
                                    logger.warning(
                                        "SL confirmed for trade #%d after %d checks",
                                        trade.id, state["sl_breach_count"],
                                    )
                                    result = self.close_trade(trade.id, current_price)
                                    if callback:
                                        await callback("stop_loss", trade, result)
                                    _pos_state.pop(trade.id, None)
                                    continue
                            else:
                                # 價格回到 SL 之上，重置計數
                                if state["sl_breach_count"] > 0:
                                    logger.info(
                                        "SL breach reset for trade #%d "
                                        "(price recovered to %.2f)",
                                        trade.id, current_price,
                                    )
                                    state["sl_breach_count"] = 0

                        if callback:
                            await callback("update", trade, {
                                "current_price": current_price,
                                "unrealized_pct": unrealized,
                            })

                    except Exception as e:
                        logger.warning("Monitor error for trade #%d: %s", trade.id, e)

                # 清理已結束交易的追蹤記錄
                active_ids = {t.id for t in open_trades}
                for tid in list(_pos_state.keys()):
                    if tid not in active_ids:
                        _pos_state.pop(tid)

            except Exception as e:
                logger.error("Position monitor error: %s", e)

            await asyncio.sleep(30)
