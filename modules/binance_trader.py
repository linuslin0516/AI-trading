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

        # 停損
        try:
            self._futures_post("/fapi/v1/order", {
                "symbol": symbol,
                "side": close_side,
                "type": "STOP_MARKET",
                "stopPrice": stop_loss,
                "closePosition": "true",
            })
            logger.info("Stop loss set at %s", stop_loss)
        except Exception as e:
            logger.warning("Failed to set SL: %s", e)

        # 停利（第一目標，平倉 50%）
        if take_profit:
            try:
                tp_qty = round(quantity * 0.5, 8)
                self._futures_post("/fapi/v1/order", {
                    "symbol": symbol,
                    "side": close_side,
                    "type": "TAKE_PROFIT_MARKET",
                    "stopPrice": take_profit[0],
                    "quantity": tp_qty,
                })
                logger.info("Take profit 1 set at %s (qty=%s)", take_profit[0], tp_qty)
            except Exception as e:
                logger.warning("Failed to set TP: %s", e)

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

            # 停損
            try:
                self._futures_post("/fapi/v1/order", {
                    "symbol": symbol,
                    "side": close_side,
                    "type": "STOP_MARKET",
                    "stopPrice": sl,
                    "closePosition": "true",
                })
                logger.info("New stop loss set at %s", sl)
            except Exception as e:
                logger.warning("Failed to set new SL: %s", e)

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
        """持續監控持倉（每 30 秒更新）"""
        logger.info("Position monitor started")
        while True:
            try:
                open_trades = self.db.get_open_trades()
                for trade in open_trades:
                    try:
                        r = self.session.get(
                            f"{MARKET_DATA_URL}/api/v3/ticker/price",
                            params={"symbol": trade.symbol}, timeout=10,
                        )
                        current_price = float(r.json()["price"])

                        # 計算未實現盈虧（含預估手續費）
                        leverage = self.leverage_map.get(trade.symbol, self.default_leverage)
                        if trade.direction == "LONG":
                            unrealized = (current_price - trade.entry_price) / trade.entry_price * 100
                        else:
                            unrealized = (trade.entry_price - current_price) / trade.entry_price * 100

                        unrealized *= leverage

                        # 扣除預估往返手續費
                        fee_pct = self.calc_fee_pct(leverage)
                        unrealized -= fee_pct

                        # 檢查是否觸及停損
                        if trade.direction == "LONG" and current_price <= trade.stop_loss:
                            logger.warning("Stop loss hit for trade #%d", trade.id)
                            result = self.close_trade(trade.id, current_price)
                            if callback:
                                await callback("stop_loss", trade, result)

                        elif trade.direction == "SHORT" and current_price >= trade.stop_loss:
                            logger.warning("Stop loss hit for trade #%d", trade.id)
                            result = self.close_trade(trade.id, current_price)
                            if callback:
                                await callback("stop_loss", trade, result)

                        # 檢查是否觸及停利
                        elif trade.take_profit:
                            tp_list = json.loads(trade.take_profit) if isinstance(trade.take_profit, str) else trade.take_profit
                            if tp_list and (
                                (trade.direction == "LONG" and current_price >= tp_list[-1]) or
                                (trade.direction == "SHORT" and current_price <= tp_list[-1])
                            ):
                                logger.info("Take profit hit for trade #%d", trade.id)
                                result = self.close_trade(trade.id, current_price)
                                if callback:
                                    await callback("take_profit", trade, result)

                        elif callback:
                            await callback("update", trade, {
                                "current_price": current_price,
                                "unrealized_pct": unrealized,
                            })

                    except Exception as e:
                        logger.warning("Monitor error for trade #%d: %s", trade.id, e)

            except Exception as e:
                logger.error("Position monitor error: %s", e)

            await asyncio.sleep(30)
