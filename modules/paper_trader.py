import asyncio
import json
import logging
from datetime import datetime, timezone

import requests

from modules.database import Database

logger = logging.getLogger(__name__)

MARKET_DATA_URL = "https://data-api.binance.vision"
FUTURES_INFO_URL = "https://fapi.binance.com"


class PaperTrader:
    """紙上交易模組 — 使用 Binance 主網真實價格，不實際下單"""

    def __init__(self, config: dict, db: Database):
        self.config = config
        self.db = db

        trading_cfg = config.get("trading", {})
        self.leverage_map = trading_cfg.get("leverage_map", {})
        self.default_leverage = trading_cfg.get("default_leverage", 25)
        self.paper_balance = trading_cfg.get("paper_balance", 10000)

        # 手續費設定（模擬真實手續費）
        fees_cfg = trading_cfg.get("fees", {})
        self.taker_rate = fees_cfg.get("taker_rate", 0.0004)
        self.maker_rate = fees_cfg.get("maker_rate", 0.0002)
        self.slippage = fees_cfg.get("slippage", 0.0001)

        self.session = requests.Session()

        # 虛擬持倉追蹤：{symbol: {qty, direction, trade_id, initial_qty, tp1_hit}}
        self._positions: dict[str, dict] = {}

        # 快取交易對精度（避免重複查詢）
        self._precision_cache: dict[str, int] = {}

        # 啟動時修正 profit_usd 欄位 & 恢復未平倉持倉
        self._migrate_profit_usd()
        self._restore_positions()

        logger.info(
            "PaperTrader initialized (mainnet prices, virtual balance=%.2f, leverage=%s)",
            self.paper_balance, self.leverage_map or self.default_leverage,
        )

    def _migrate_profit_usd(self):
        """一次性修正 profit_usd 和 PARTIAL_CLOSE 的 TP1 利潤"""
        # 修正已平倉交易的 profit_usd
        closed = [t for t in self.db.get_closed_trades(500)
                  if t.status == "CLOSED" and t.profit_pct is not None]
        for t in closed:
            margin = self.paper_balance * (t.position_size or 0) / 100
            correct_usd = round(margin * t.profit_pct / 100, 4)
            if abs((t.profit_usd or 0) - correct_usd) > 0.01:
                self.db.update_trade(t.id, profit_usd=correct_usd)
                logger.info("Fixed profit_usd for trade #%d: %.4f -> %.4f",
                            t.id, t.profit_usd or 0, correct_usd)

        # 修正 PARTIAL_CLOSE 交易：補算 TP1 利潤 + 倉位減半
        partial = [t for t in self.db.get_open_trades()
                   if t.status == "PARTIAL_CLOSE"]
        for t in partial:
            if t.profit_usd and t.profit_usd != 0:
                continue  # 已有 TP1 利潤，跳過
            tp_list = (
                json.loads(t.take_profit)
                if isinstance(t.take_profit, str)
                else t.take_profit
            ) or []
            if not tp_list:
                continue
            tp1_price = tp_list[0]
            leverage = self.leverage_map.get(t.symbol, self.default_leverage)
            if t.direction == "LONG":
                tp1_pct = (tp1_price - t.entry_price) / t.entry_price * 100
            else:
                tp1_pct = (t.entry_price - tp1_price) / t.entry_price * 100
            tp1_pct *= leverage
            tp1_pct -= self.calc_fee_pct(leverage)
            # 原始倉位% = 當前 position_size × 2（還沒減半的舊資料）
            orig_size = t.position_size * 2 if t.position_size else 0
            half_margin = self.paper_balance * orig_size / 100 / 2
            tp1_profit = round(half_margin * tp1_pct / 100, 4)
            # 更新 DB
            updates = {"profit_usd": tp1_profit}
            # 若倉位尚未減半，補減半
            if orig_size > 0 and t.position_size == orig_size:
                updates["position_size"] = orig_size / 2
            self.db.update_trade(t.id, **updates)
            logger.info("Fixed TP1 profit for trade #%d: $%.4f (pos_size=%.2f%%)",
                        t.id, tp1_profit, updates.get("position_size", t.position_size))

    def _restore_positions(self):
        """從 DB 恢復未平倉的虛擬持倉"""
        open_trades = self.db.get_open_trades()
        for trade in open_trades:
            tp_list = (
                json.loads(trade.take_profit)
                if isinstance(trade.take_profit, str)
                else trade.take_profit
            ) or []

            # 用入場價和倉位大小反推數量
            leverage = self.leverage_map.get(trade.symbol, self.default_leverage)
            balance = self._get_virtual_balance()
            amount_usdt = balance * (trade.position_size / 100) * leverage
            qty = amount_usdt / trade.entry_price if trade.entry_price else 0

            # 判斷 TP1 是否已觸發（PARTIAL_CLOSE 狀態）
            tp1_hit = trade.status == "PARTIAL_CLOSE"
            if tp1_hit:
                qty *= 0.5

            precision = self._get_qty_precision(trade.symbol)
            qty = round(qty, precision)

            self._positions[trade.symbol] = {
                "qty": qty,
                "initial_qty": qty * 2 if tp1_hit else qty,
                "direction": trade.direction,
                "trade_id": trade.id,
                "tp1_hit": tp1_hit,
            }
            logger.info(
                "Restored paper position: #%d %s %s qty=%.6f (tp1_hit=%s)",
                trade.id, trade.direction, trade.symbol, qty, tp1_hit,
            )

    # ── 價格與精度 ──

    def _get_price(self, symbol: str) -> float:
        """取得 Binance 主網即時價格"""
        r = self.session.get(
            f"{MARKET_DATA_URL}/api/v3/ticker/price",
            params={"symbol": symbol}, timeout=10,
        )
        r.raise_for_status()
        return float(r.json()["price"])

    def _get_qty_precision(self, symbol: str) -> int:
        """取得交易對的數量精度（快取）"""
        if symbol in self._precision_cache:
            return self._precision_cache[symbol]

        try:
            r = self.session.get(
                f"{FUTURES_INFO_URL}/fapi/v1/exchangeInfo", timeout=10,
            )
            r.raise_for_status()
            for s in r.json().get("symbols", []):
                if s["symbol"] == symbol:
                    for f in s["filters"]:
                        if f["filterType"] == "LOT_SIZE":
                            step = float(f["stepSize"])
                            precision = len(str(step).rstrip("0").split(".")[-1])
                            self._precision_cache[symbol] = precision
                            return precision
        except Exception as e:
            logger.warning("Failed to get precision for %s: %s", symbol, e)

        self._precision_cache[symbol] = 3
        return 3

    # ── 虛擬餘額 ──

    def _get_virtual_balance(self) -> float:
        """計算虛擬帳戶餘額 = 初始金額 + 已實現盈虧"""
        all_trades = self.db.get_closed_trades(500)

        # 已平倉交易：用加權公式
        closed = [t for t in all_trades if t.status == "CLOSED"]
        closed_pnl = sum(
            (t.profit_pct or 0) * (t.position_size or 0) / 100
            for t in closed
        )
        closed_usd = self.paper_balance * closed_pnl / 100

        # PARTIAL_CLOSE 交易的 TP1 已實現利潤（存在 profit_usd）
        partial = [t for t in self.db.get_open_trades()
                   if t.status == "PARTIAL_CLOSE"]
        tp1_usd = sum(t.profit_usd or 0 for t in partial)

        return self.paper_balance + closed_usd + tp1_usd

    def _get_used_margin(self) -> float:
        """計算已使用保證金"""
        open_trades = self.db.get_open_trades()
        balance = self._get_virtual_balance()
        return sum(
            balance * (t.position_size / 100) for t in open_trades
        )

    # ── Binance API 模擬（供 Telegram 指令相容） ──

    def _futures_get(self, path: str, params: dict | None = None, signed: bool = False):
        """模擬 Binance API 回應，供 Telegram 指令使用"""
        if path == "/fapi/v2/account":
            balance = self._get_virtual_balance()
            used = self._get_used_margin()
            unrealized = self._calc_total_unrealized()
            return {
                "totalWalletBalance": str(balance),
                "totalUnrealizedProfit": str(unrealized),
                "totalMarginBalance": str(balance + unrealized),
                "availableBalance": str(balance - used),
            }
        if path == "/fapi/v2/positionRisk":
            return self._get_virtual_position_risk()
        # 其他路徑回傳空
        return {}

    def _calc_total_unrealized(self) -> float:
        """計算所有持倉的未實現盈虧（USD）"""
        total = 0
        for symbol, pos in self._positions.items():
            try:
                current_price = self._get_price(symbol)
                trade = self.db.get_trade(pos["trade_id"])
                if not trade:
                    continue
                leverage = self.leverage_map.get(symbol, self.default_leverage)
                if pos["direction"] == "LONG":
                    pnl_pct = (current_price - trade.entry_price) / trade.entry_price * 100
                else:
                    pnl_pct = (trade.entry_price - current_price) / trade.entry_price * 100
                pnl_pct *= leverage
                fee_pct = self.calc_fee_pct(leverage)
                pnl_pct -= fee_pct
                total += pnl_pct * trade.position_size / 100
            except Exception:
                pass
        return total

    def _get_virtual_position_risk(self) -> list[dict]:
        """模擬 positionRisk API 回應"""
        result = []
        for symbol, pos in self._positions.items():
            try:
                price = self._get_price(symbol)
            except Exception:
                price = 0
            amt = pos["qty"] if pos["direction"] == "LONG" else -pos["qty"]
            result.append({
                "symbol": symbol,
                "positionAmt": str(amt),
                "markPrice": str(price),
            })
        return result

    # ── 核心交易方法 ──

    def execute_trade(self, decision: dict) -> dict:
        """模擬下單（使用主網真實價格）"""
        symbol = decision["symbol"]
        action = decision["action"]  # LONG / SHORT
        entry_price = decision["entry"]["price"]
        stop_loss = decision["stop_loss"]
        take_profit = decision["take_profit"]  # list
        position_size_pct = decision["position_size"]

        leverage = self.leverage_map.get(symbol, self.default_leverage)

        try:
            # 1. 取得真實價格（MARKET 單用當前價，LIMIT 單用指定價）
            strategy = decision["entry"].get("strategy", "LIMIT")
            if strategy == "MARKET":
                entry_price = self._get_price(symbol)

            # 2. 計算數量
            quantity = self._calc_quantity(symbol, entry_price, position_size_pct)
            if quantity <= 0:
                return {"success": False, "error": "Invalid quantity"}

            # 3. 建立 trade 記錄
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
                entry_order_id=f"PAPER-{int(datetime.now(timezone.utc).timestamp())}",
                status="OPEN",
                market_condition=decision.get("_market_condition"),
            )

            # 4. 記錄虛擬持倉
            self._positions[symbol] = {
                "qty": quantity,
                "initial_qty": quantity,
                "direction": action,
                "trade_id": trade.id,
                "tp1_hit": False,
            }

            logger.info(
                "Paper trade executed: #%d %s %s @ %s qty=%s (virtual)",
                trade.id, action, symbol, entry_price, quantity,
            )

            return {
                "success": True,
                "trade_id": trade.id,
                "order_id": f"PAPER-{trade.id}",
                "symbol": symbol,
                "direction": action,
                "entry_price": entry_price,
                "quantity": quantity,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }

        except Exception as e:
            logger.error("Paper trade execution error: %s", e)
            return {"success": False, "error": str(e)}

    def _calc_quantity(self, symbol: str, price: float, position_pct: float) -> float:
        """計算下單數量（基於虛擬餘額）"""
        try:
            balance = self._get_virtual_balance()
            leverage = self.leverage_map.get(symbol, self.default_leverage)
            amount_usdt = balance * (position_pct / 100) * leverage
            quantity = amount_usdt / price

            precision = self._get_qty_precision(symbol)
            quantity = round(quantity, precision)

            return quantity

        except Exception as e:
            logger.error("Failed to calculate quantity: %s", e)
            return 0

    def close_trade(self, trade_id: int, exit_price: float | None = None) -> dict:
        """平倉（使用主網真實價格計算盈虧）"""
        trade = self.db.get_trade(trade_id)
        if not trade:
            return {"success": False, "error": "Trade not found"}

        if trade.status == "CLOSED":
            return {"success": False, "error": "Trade already closed"}

        symbol = trade.symbol
        direction = trade.direction

        try:
            # 取得真實出場價格
            if not exit_price:
                exit_price = self._get_price(symbol)

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

            # 更新記錄：profit_usd = 剩餘倉位盈虧 + TP1 已實現利潤
            tp1_profit = trade.profit_usd or 0  # TP1 時存入的已實現利潤
            margin_usd = self._get_virtual_balance() * (trade.position_size or 0) / 100
            remaining_profit_usd = margin_usd * profit_pct / 100
            actual_profit_usd = remaining_profit_usd + tp1_profit

            # 如果 TP1 已實現，用總利潤重算整體報酬率
            if tp1_profit > 0:
                full_margin = margin_usd * 2  # TP1 時 position_size 已減半
                if full_margin > 0:
                    profit_pct = actual_profit_usd / full_margin * 100

            # outcome 基於總利潤（含 TP1 已實現）決定
            outcome = "WIN" if actual_profit_usd > 0 else ("LOSS" if actual_profit_usd < 0 else "BREAKEVEN")
            self.db.update_trade(
                trade_id,
                exit_price=exit_price,
                profit_pct=round(profit_pct, 4),
                profit_usd=round(actual_profit_usd, 4),
                hold_duration=hold_duration,
                outcome=outcome,
                status="CLOSED",
                closed_at=now,
            )

            # 移除虛擬持倉
            self._positions.pop(symbol, None)

            logger.info(
                "Paper trade #%d closed: %s %.2f%% (fee: %.2f%%) @ %s",
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
            logger.error("Failed to close paper trade #%d: %s", trade_id, e)
            return {"success": False, "error": str(e)}

    def adjust_trade(self, trade_id: int, new_stop_loss: float | None = None,
                     new_take_profit: list[float] | None = None) -> dict:
        """調整止盈止損（純 DB 更新，無交易所掛單）"""
        trade = self.db.get_trade(trade_id)
        if not trade:
            return {"success": False, "error": "Trade not found"}
        if trade.status == "CLOSED":
            return {"success": False, "error": "Trade already closed"}

        changes = []
        update_fields = {}

        if new_stop_loss:
            update_fields["stop_loss"] = new_stop_loss
            changes.append(f"SL: {trade.stop_loss} -> {new_stop_loss}")
        if new_take_profit:
            update_fields["take_profit"] = new_take_profit
            changes.append(f"TP: {trade.take_profit} -> {new_take_profit}")

        if update_fields:
            self.db.update_trade(trade_id, **update_fields)

        sl = new_stop_loss if new_stop_loss else trade.stop_loss
        tp = new_take_profit if new_take_profit else (
            json.loads(trade.take_profit) if isinstance(trade.take_profit, str) else trade.take_profit
        )

        logger.info("Paper trade #%d adjusted: %s", trade_id, ", ".join(changes))

        return {
            "success": True,
            "trade_id": trade_id,
            "changes": changes,
            "new_stop_loss": sl,
            "new_take_profit": tp,
        }

    def resync_sl_tp(self, trade_id: int) -> dict:
        """紙上交易無交易所掛單，直接回傳成功"""
        trade = self.db.get_trade(trade_id)
        if not trade:
            return {"success": False, "error": "Trade not found"}
        if trade.status == "CLOSED":
            return {"success": False, "error": "Trade already closed"}

        tp = trade.take_profit
        if isinstance(tp, str):
            tp = json.loads(tp)

        # 確保虛擬持倉存在
        if trade.symbol not in self._positions:
            qty = self._calc_quantity(trade.symbol, trade.entry_price, trade.position_size)
            self._positions[trade.symbol] = {
                "qty": qty,
                "initial_qty": qty,
                "direction": trade.direction,
                "trade_id": trade.id,
                "tp1_hit": trade.status == "PARTIAL_CLOSE",
            }

        pos = self._positions.get(trade.symbol, {})
        logger.info(
            "Paper resync for trade #%d: SL=%s TP=%s qty=%s (no exchange orders)",
            trade_id, trade.stop_loss, tp, pos.get("qty", 0),
        )

        return {
            "success": True,
            "trade_id": trade_id,
            "stop_loss": trade.stop_loss,
            "take_profit": tp,
            "quantity": pos.get("qty", 0),
        }

    def cancel_all_orders(self, symbol: str) -> dict:
        """紙上交易無交易所掛單"""
        logger.info("Paper cancel_all_orders for %s (no-op)", symbol)
        return {"success": True, "result": "paper_mode"}

    def get_recent_orders(self, symbol: str, limit: int = 20) -> list[dict]:
        """紙上交易無交易所訂單紀錄"""
        return []

    def calc_fee_pct(self, leverage: int, entry_type: str = "TAKER", exit_type: str = "TAKER") -> float:
        """計算往返手續費（與 BinanceTrader 相同公式）"""
        entry_rate = self.taker_rate if entry_type == "TAKER" else self.maker_rate
        exit_rate = self.taker_rate if exit_type == "TAKER" else self.maker_rate
        return (entry_rate + exit_rate + self.slippage * 2) * leverage * 100

    # ── 持倉監控 ──

    async def monitor_positions(self, callback=None):
        """持續監控虛擬持倉（每 30 秒，使用主網真實價格）

        與 BinanceTrader 的差異：
        - 不查詢交易所持倉，使用記憶體中的虛擬持倉
        - TP1/TP2 由本地價格比對觸發（BinanceTrader 是交易所掛單觸發）
        - SL 同樣使用連續 4 次確認機制
        """
        logger.info("Paper position monitor started (mainnet prices)")

        _pos_state: dict[int, dict] = {}

        while True:
            try:
                open_trades = self.db.get_open_trades()
                if not open_trades:
                    self._positions.clear()
                    _pos_state.clear()
                    await asyncio.sleep(30)
                    continue

                for trade in open_trades:
                    try:
                        symbol = trade.symbol

                        # 取得主網真實價格
                        current_price = self._get_price(symbol)

                        # 確保虛擬持倉存在
                        pos = self._positions.get(symbol)
                        if not pos or pos["trade_id"] != trade.id:
                            continue

                        actual_qty = pos["qty"]

                        # 初始化追蹤狀態
                        if trade.id not in _pos_state:
                            _pos_state[trade.id] = {
                                "sl_breach_count": 0,
                            }
                        state = _pos_state[trade.id]

                        # 解析 TP 列表
                        tp_list = (
                            json.loads(trade.take_profit)
                            if isinstance(trade.take_profit, str)
                            else trade.take_profit
                        ) or []

                        # ── Case A: TP1 檢查（價格觸及第一目標） ──
                        if (not pos["tp1_hit"]
                                and len(tp_list) >= 2
                                and actual_qty > 0):

                            tp1_hit = (
                                (trade.direction == "LONG" and current_price >= tp_list[0]) or
                                (trade.direction == "SHORT" and current_price <= tp_list[0])
                            )

                            if tp1_hit:
                                pos["tp1_hit"] = True
                                old_qty = actual_qty
                                precision = self._get_qty_precision(symbol)
                                new_qty = round(actual_qty * 0.5, precision)
                                pos["qty"] = new_qty

                                logger.info(
                                    "TP1 partial close for paper trade #%d: qty %.6f -> %.6f",
                                    trade.id, old_qty, new_qty,
                                )

                                # 計算 TP1 已實現利潤
                                tp1_price = tp_list[0]
                                leverage = self.leverage_map.get(symbol, self.default_leverage)
                                if trade.direction == "LONG":
                                    tp1_pct = (tp1_price - trade.entry_price) / trade.entry_price * 100
                                else:
                                    tp1_pct = (trade.entry_price - tp1_price) / trade.entry_price * 100
                                tp1_pct *= leverage
                                tp1_fee = self.calc_fee_pct(leverage)
                                tp1_pct -= tp1_fee
                                half_margin = self._get_virtual_balance() * (trade.position_size or 0) / 100 / 2
                                tp1_profit_usd = round(half_margin * tp1_pct / 100, 4)

                                logger.info(
                                    "TP1 realized profit for trade #%d: %.2f%% on margin $%.2f = $%.4f",
                                    trade.id, tp1_pct, half_margin, tp1_profit_usd,
                                )

                                # 更新 DB：記錄 TP1 利潤、倉位減半
                                self.db.update_trade(
                                    trade.id,
                                    status="PARTIAL_CLOSE",
                                    profit_usd=tp1_profit_usd,
                                    position_size=trade.position_size / 2,
                                )

                                # 移動 SL 到保本價
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
                                    "Breakeven SL for paper trade #%d: %.2f -> %.2f "
                                    "(entry=%.2f + fee=%.4f%%)",
                                    trade.id, old_sl or 0, breakeven_price,
                                    trade.entry_price, fee_rate * 100,
                                )

                                if callback:
                                    await callback("tp1_hit", trade, {
                                        "current_price": current_price,
                                        "tp1_price": tp_list[0],
                                        "closed_qty": old_qty - new_qty,
                                        "remaining_qty": new_qty,
                                        "breakeven_sl": breakeven_price,
                                        "old_sl": old_sl,
                                    })

                                # 重新讀取 trade（SL 已更新）
                                trade = self.db.get_trade(trade.id)

                        # ── Case B: TP2 檢查（價格觸及第二目標） ──
                        if pos["tp1_hit"] and len(tp_list) >= 2 and actual_qty > 0:
                            tp2_hit = (
                                (trade.direction == "LONG" and current_price >= tp_list[1]) or
                                (trade.direction == "SHORT" and current_price <= tp_list[1])
                            )

                            if tp2_hit:
                                logger.info(
                                    "TP2 hit for paper trade #%d, closing position",
                                    trade.id,
                                )
                                result = self.close_trade(trade.id, current_price)
                                if callback:
                                    await callback("take_profit", trade, result)
                                _pos_state.pop(trade.id, None)
                                continue

                        # 單目標 TP 檢查
                        if (not pos["tp1_hit"]
                                and len(tp_list) == 1
                                and actual_qty > 0):
                            tp_hit = (
                                (trade.direction == "LONG" and current_price >= tp_list[0]) or
                                (trade.direction == "SHORT" and current_price <= tp_list[0])
                            )
                            if tp_hit:
                                logger.info(
                                    "TP hit for paper trade #%d, closing position",
                                    trade.id,
                                )
                                result = self.close_trade(trade.id, current_price)
                                if callback:
                                    await callback("take_profit", trade, result)
                                _pos_state.pop(trade.id, None)
                                continue

                        # ── Case C: 計算未實現盈虧 ──
                        leverage = self.leverage_map.get(symbol, self.default_leverage)
                        if trade.direction == "LONG":
                            unrealized = (current_price - trade.entry_price) / trade.entry_price * 100
                        else:
                            unrealized = (trade.entry_price - current_price) / trade.entry_price * 100
                        unrealized *= leverage
                        fee_pct = self.calc_fee_pct(leverage)
                        unrealized -= fee_pct

                        # ── Case D: 本地 SL 監控（連續 4 次確認） ──
                        sl = trade.stop_loss or 0
                        if sl and actual_qty > 0:
                            sl_breached = (
                                (trade.direction == "LONG" and current_price <= sl) or
                                (trade.direction == "SHORT" and current_price >= sl)
                            )
                            if sl_breached:
                                state["sl_breach_count"] += 1
                                logger.warning(
                                    "SL breach #%d for paper trade #%d (%s %s): "
                                    "price=%.2f, sl=%.2f",
                                    state["sl_breach_count"], trade.id,
                                    trade.direction, symbol, current_price, sl,
                                )
                                if state["sl_breach_count"] >= 4:
                                    logger.warning(
                                        "SL confirmed for paper trade #%d after %d checks",
                                        trade.id, state["sl_breach_count"],
                                    )
                                    result = self.close_trade(trade.id, current_price)
                                    if callback:
                                        await callback("stop_loss", trade, result)
                                    _pos_state.pop(trade.id, None)
                                    continue
                            else:
                                if state["sl_breach_count"] > 0:
                                    logger.info(
                                        "SL breach reset for paper trade #%d "
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
                        logger.warning("Paper monitor error for trade #%d: %s", trade.id, e)

                # 清理已結束交易的追蹤記錄
                active_ids = {t.id for t in open_trades}
                for tid in list(_pos_state.keys()):
                    if tid not in active_ids:
                        _pos_state.pop(tid)

            except Exception as e:
                logger.error("Paper position monitor error: %s", e)

            await asyncio.sleep(30)
