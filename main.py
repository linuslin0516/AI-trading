"""
AI 自動交易系統 - 主程式入口

工作流程：
1. Discord 監聽分析師頻道
2. 累積訊息後觸發分析
3. Claude AI 深度分析 + 市場數據
4. 風控檢查
5. Telegram 通知 + 30 秒確認
6. Binance Testnet 下單
7. 持倉監控
8. 平倉後 AI 覆盤學習
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, time as dtime, timedelta, timezone

from modules.ai_analyzer import AIAnalyzer
from modules.binance_trader import BinanceTrader
from modules.database import Database
from modules.decision_engine import DecisionEngine
from modules.discord_listener import DiscordListener
from modules.economic_calendar import EconomicCalendar
from modules.learning_engine import LearningEngine
from modules.market_data import MarketData
from modules.message_scorer import MessageScorer
from modules.telegram_notifier import TelegramNotifier
from utils.helpers import load_config, setup_logging
from utils.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class TradingBot:
    def __init__(self):
        # 載入配置
        self.config = load_config()
        setup_logging(self.config)

        logger.info("=" * 50)
        logger.info("AI Trading Bot starting...")
        logger.info("=" * 50)

        # 初始化所有模組
        tz_name = self.config.get("schedule", {}).get("timezone", "UTC")
        self.db = Database(self.config["database"]["path"], tz_name=tz_name)
        self.market = MarketData(self.config)
        self.ai = AIAnalyzer(self.config)
        self.risk = RiskManager(self.config, self.db)
        self.calendar = EconomicCalendar(self.config)
        self.scorer = MessageScorer(self.config)
        self.decision = DecisionEngine(
            self.config, self.db, self.market, self.ai, self.risk,
            self.calendar, self.scorer,
        )
        trading_mode = self.config.get("trading", {}).get("mode", "testnet")
        if trading_mode == "paper":
            from modules.paper_trader import PaperTrader
            from modules.price_feed import PriceFeed
            symbols = self.config.get("binance", {}).get("symbols", ["BTCUSDT", "ETHUSDT"])
            self.price_feed = PriceFeed(symbols=symbols)
            self.trader = PaperTrader(self.config, self.db, price_feed=self.price_feed)
            logger.info("Trading mode: PAPER (mainnet prices + WebSocket, no real orders)")
        else:
            self.price_feed = None
            self.trader = BinanceTrader(self.config, self.db)
            logger.info("Trading mode: TESTNET (Binance testnet)")
        self.telegram = TelegramNotifier(self.config, db=self.db, trader=self.trader)
        self.learning = LearningEngine(self.config, self.db, self.ai, self.risk)
        self.discord = DiscordListener(self.config)

        # 初始化分析師到數據庫
        for ch in self.config["discord"]["monitored_channels"]:
            self.db.get_or_create_analyst(ch["analyst"], ch.get("initial_weight", 1.0))

        # 設定 Discord 回調
        self.discord.set_analysis_callback(self._on_signals_received)

        self._running = True
        self._last_ai_call_time: datetime | None = None

    async def start(self):
        """啟動所有服務"""
        logger.info("Starting all services...")

        # 啟動時印出 data 目錄內容（方便確認 volume 掛載）
        import os
        data_dir = os.path.dirname(self.config.get("database", {}).get("path", "./data/trades.db"))
        abs_data = os.path.abspath(data_dir)
        if os.path.isdir(abs_data):
            files = os.listdir(abs_data)
            sizes = {f: os.path.getsize(os.path.join(abs_data, f)) for f in files}
            logger.info("Data directory [%s]: %s", abs_data, sizes)
        else:
            logger.warning("Data directory [%s] does NOT exist!", abs_data)

        # 啟動時重新覆盤失敗的交易
        self._retry_failed_reviews()

        # 載入分析師最新權重
        self._sync_analyst_weights()

        # 啟動 Telegram
        try:
            await self.telegram.start()
            self.telegram._briefing_callback = self.generate_and_send_briefing
            self.telegram._review_callback = self._manual_review
            logger.info("Telegram bot started")
        except Exception as e:
            logger.error("Telegram start failed: %s", e)

        # 啟動 WebSocket 價格串流
        if self.price_feed:
            await self.price_feed.start()
            logger.info("WebSocket price feed started")

        # 啟動持倉監控
        monitor_task = asyncio.create_task(
            self.trader.monitor_positions(callback=self._on_position_event)
        )

        # 啟動每日早報（8:00 AM）和晚報（10:00 PM）
        morning_task = asyncio.create_task(self._morning_briefing_loop())
        evening_task = asyncio.create_task(self._evening_summary_loop())

        # 啟動快速回饋學習
        feedback_task = asyncio.create_task(self._quick_feedback_loop())

        # 啟動延遲覆盤（平倉 4 小時後才覆盤，附帶後續價格走勢）
        review_task = asyncio.create_task(self._delayed_review_loop())

        # 啟動市場掃描器
        scanner_cfg = self.config.get("market_scanner", {})
        if scanner_cfg.get("enabled", False):
            scanner_task = asyncio.create_task(self._market_scanner_loop())
            logger.info("Market scanner enabled (interval=%ds, lookback=%dh)",
                        scanner_cfg.get("interval_seconds", 300),
                        scanner_cfg.get("lookback_hours", 4))

        # 啟動 Discord（這會阻塞）
        try:
            logger.info("Starting Discord listener (this blocks)...")
            await self.discord.start()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error("Discord error: %s", e)
        finally:
            await self.shutdown()

    async def shutdown(self):
        """優雅關閉"""
        logger.info("Shutting down...")
        self._running = False
        try:
            await self.discord.stop()
        except Exception:
            pass
        try:
            await self.telegram.stop()
        except Exception:
            pass
        logger.info("Shutdown complete")

    # ── 核心回調 ──

    async def _on_signals_received(self, messages: list):
        """
        Discord 訊息累積後觸發的分析流程

        messages: list of AnalystMessage
        """
        logger.info("=" * 40)
        logger.info("Analysis triggered with %d messages", len(messages))
        logger.info("=" * 40)

        try:
            # 記錄 AI 呼叫時間（供掃描器冷卻判斷）
            self._last_ai_call_time = datetime.now(timezone.utc)

            # 0. 儲存所有分析師訊息到資料庫（供早報/晚報/掃描器使用）
            for m in messages:
                # 提取圖片 URL（不存 base64，只存 URL 供掃描器重新下載）
                img_urls = [
                    {"url": img["url"], "media_type": img["media_type"]}
                    for img in getattr(m, "images", [])
                    if img.get("url")
                ] or None
                self.db.save_analyst_message(
                    analyst_name=m.analyst,
                    channel=m.channel_name,
                    content=m.content,
                    images=img_urls,
                )

            analyst_names = [m.analyst for m in messages]

            # 1. 決策引擎處理（跟單模式 vs AI 自主模式）
            follow_mode = self.config.get("discord", {}).get("follow_mode", False)
            if follow_mode:
                decisions = self.decision.process_follow_signals(messages)
                if not decisions:
                    logger.info("No actionable signal — skipping")
                    return
                for decision in decisions:
                    await self._execute_decision(decision, messages, analyst_names)
                return

            # AI 自主模式：單一決策
            decision = self.decision.process_signals(messages)
            if decision is None:
                logger.info("No actionable signal — skipping")
                return
            await self._execute_decision(decision, messages, analyst_names)

        except Exception as e:
            logger.exception("Error in signal processing pipeline")
            await self.telegram.send_error(f"分析流程錯誤: {e}")

    async def _execute_decision(self, decision: dict, messages: list, analyst_names: list):
        """執行單一交易決策（SKIP / CLOSE / LONG / SHORT）"""
        action = decision.get("action", "")

        # SKIP
        if action == "SKIP":
            logger.info("No actionable signal — skipping")
            self.db.save_ai_decision(
                decision, outcome="SKIP", analyst_names=analyst_names,
            )
            return

        # CLOSE — 跟單模式平倉指令
        if action == "CLOSE":
            await self._handle_follow_close(decision)
            return

        # 風控拒絕
        if decision.get("_rejected"):
            logger.warning("Signal rejected by risk manager")
            self.db.save_ai_decision(
                decision, outcome="REJECTED", analyst_names=analyst_names,
            )
            return

        # 交易停用
        trading_cfg = self.config.get("trading", {})
        if not trading_cfg.get("enabled", True):
            logger.info("Trading disabled — signal only mode")
            await self.telegram.send_signal(decision, countdown=0)
            return

        # Telegram 通知 + 確認倒數
        countdown = trading_cfg.get("confirmation_delay", 30)
        result = await self.telegram.send_signal(decision, countdown=countdown)

        if result.get("cancelled"):
            logger.info("Trade cancelled by user")
            self.db.save_ai_decision(
                decision, outcome="CANCELLED",
                analyst_names=analyst_names,
                cancel_reason=result.get("cancel_reason", ""),
            )
            return

        # 執行交易
        if trading_cfg.get("auto_execute", True):
            trade_result = self.trader.execute_trade(decision)

            if trade_result.get("success"):
                self.db.save_ai_decision(
                    decision, outcome="EXECUTED",
                    analyst_names=analyst_names,
                    trade_id=trade_result["trade_id"],
                )

                for m in messages:
                    self.db.record_analyst_call(
                        trade_id=trade_result["trade_id"],
                        analyst_name=m.analyst,
                        direction=decision["action"],
                        message=m.content,
                    )

                if trade_result.get("pending"):
                    await self.telegram.send_pending_order(trade_result)
                    logger.info("Trade #%d LIMIT order pending", trade_result["trade_id"])
                else:
                    await self.telegram.send_entry_confirmation(trade_result)
                    self.risk.record_trade_time()
                    logger.info("Trade #%d executed successfully", trade_result["trade_id"])

                # 跟單加倉點：執行第二個入場點
                if decision.get("_addon_entry"):
                    await self._handle_addon_entry(decision["_addon_entry"])
            else:
                error = trade_result.get("error", "Unknown error")
                logger.error("Trade execution failed: %s", error)
                await self.telegram.send_error(f"交易執行失敗: {error}")

    async def _on_position_event(self, event_type: str, trade, data: dict):
        """持倉監控回調"""
        if event_type == "limit_filled":
            # LIMIT 掛單成交通知
            logger.info("LIMIT order filled: #%d %s %s @ %s",
                        data["trade_id"], data["direction"],
                        data["symbol"], data["entry_price"])
            await self.telegram.send_entry_confirmation(data)
            self.risk.record_trade_time()

            # 若有同方向其他持倉，顯示平均成本
            try:
                symbol = data["symbol"]
                direction = data["direction"]
                open_trades = self.db.get_open_trades()
                same_dir = [t for t in open_trades
                            if t.symbol == symbol and t.direction == direction]
                if len(same_dir) >= 2:
                    avg_cost = round(sum(t.entry_price for t in same_dir) / len(same_dir), 2)
                    await self.telegram._safe_send(
                        f"📊 加倉成交 {symbol} {direction}\n"
                        f"持倉數: {len(same_dir)} 倉 | 平均成本: {avg_cost}"
                    )
            except Exception:
                pass
            return

        if event_type == "tp1_hit":
            # 部分止盈通知（TP1 / TP2 通用）
            tp_index = data.get("tp_index", 0)   # 0=TP1, 1=TP2
            tp_label = f"TP{tp_index + 1}"
            tp_price = data.get("tp1_price", 0)
            closed_qty = data.get("closed_qty", 0)
            remaining = data.get("remaining_qty", 0)
            current = data.get("current_price", 0)
            breakeven_sl = data.get("breakeven_sl")
            old_sl = data.get("old_sl", 0)

            logger.info("%s hit for trade #%d %s", tp_label, trade.id, trade.symbol)

            # TP1：止損移到保本
            if tp_index == 0 and breakeven_sl:
                sl_line = f"\n\n🛡️ 保本機制啟動\n止損移至: {old_sl} → {breakeven_sl}"
                next_hint = "\n繼續持有，等待下一個目標..."
            else:
                sl_line = ""
                next_hint = "\n繼續持有，等待下一個目標..."

            text = (
                f"🎯 {tp_label} 止盈到達！\n\n"
                f"#{trade.id} {trade.direction} {trade.symbol}\n"
                f"目標價格: {tp_price}\n"
                f"當前價格: {current}\n"
                f"已平倉數量: {closed_qty}\n"
                f"剩餘倉位: {remaining}"
                f"{sl_line}"
                f"{next_hint}"
            )
            try:
                await self.telegram._safe_send(text)
            except Exception as e:
                logger.warning("Failed to send %s notification: %s", tp_label, e)
            return

        if event_type in ("stop_loss", "take_profit", "liquidation", "closed_unknown"):
            logger.info(
                "Position closed by %s: trade #%d %s",
                event_type, trade.id, trade.symbol,
            )

            # 強平特別警告
            if event_type == "liquidation":
                liq_text = (
                    f"💀 強制平倉 (Liquidation)！\n\n"
                    f"#{trade.id} {trade.direction} {trade.symbol}\n"
                    f"入場價: {trade.entry_price}\n"
                    f"止損價: {trade.stop_loss}\n"
                    f"平倉價: {data.get('exit_price', 'N/A')}\n\n"
                    f"⚠️ 倉位被交易所強平，價格已超過止損位。\n"
                    f"請檢查槓桿倍數和保證金是否足夠。"
                )
                try:
                    await self.telegram._safe_send(liq_text)
                except Exception as e:
                    logger.error("Failed to send liquidation alert: %s", e)

            if event_type == "closed_unknown":
                unk_text = (
                    f"❓ 倉位異常關閉\n\n"
                    f"#{trade.id} {trade.direction} {trade.symbol}\n"
                    f"入場價: {trade.entry_price}\n"
                    f"止損價: {trade.stop_loss}\n"
                    f"當前價: {data.get('exit_price', 'N/A')}\n\n"
                    f"倉位在交易所端消失，原因不明。\n"
                    f"請至 Binance 確認。"
                )
                try:
                    await self.telegram._safe_send(unk_text)
                except Exception as e:
                    logger.error("Failed to send unknown close alert: %s", e)

            # 記錄平倉（覆盤延遲 4 小時後執行）
            learn_result = await self.learning.on_trade_closed(trade.id)
            events = learn_result.get("events", [])

            # 發送平倉通知（不含覆盤，覆盤稍後才會到）
            await self.telegram.send_exit_notification(trade, data, review=None)

            # 發送學習事件通知
            for event in events:
                await self.telegram.send_learning_event(event)

        elif event_type == "update":
            # 可選：重要價格變動時通知
            pass

    async def _handle_follow_close(self, decision: dict):
        """跟單模式：分析師說平倉，自動平掉該幣種所有持倉"""
        symbol = decision.get("symbol", "")
        open_trades = self.db.get_open_trades()
        to_close = [t for t in open_trades if t.symbol == symbol]

        if not to_close:
            logger.info("Follow CLOSE: no open position for %s", symbol)
            try:
                await self.telegram._safe_send(f"⚠️ 分析師指示平倉 {symbol}，但目前沒有持倉")
            except Exception:
                pass
            return

        for trade in to_close:
            result = self.trader.close_trade(trade.id)
            if result.get("success"):
                logger.info("Follow CLOSE: closed #%d %s %s", trade.id, trade.direction, symbol)
                await self.telegram.send_exit_notification(trade, result, review=None)
                await self.learning.on_trade_closed(trade.id)
            else:
                logger.error("Follow CLOSE failed for #%d: %s", trade.id, result.get("error"))

    async def _handle_addon_entry(self, addon: dict):
        """跟單加倉：執行第二個入場點，並計算兩倉平均成本"""
        try:
            addon_result = self.trader.execute_trade(addon)
            if not addon_result.get("success"):
                logger.error("Addon entry failed: %s", addon_result.get("error"))
                return

            symbol = addon["symbol"]
            action = addon["action"]
            entry_2_price = addon["entry"]["price"]
            logger.info("Addon trade #%d placed @ %.2f", addon_result["trade_id"], entry_2_price)

            # 計算兩倉平均成本
            open_trades = self.db.get_open_trades()
            same_dir = [t for t in open_trades if t.symbol == symbol and t.direction == action]
            if len(same_dir) >= 2:
                prices = [t.entry_price for t in same_dir]
                avg_cost = round(sum(prices) / len(prices), 2)
                avg_text = f"\n平均成本: {avg_cost}"
            else:
                avg_text = ""

            # 發送加倉通知
            text = (
                f"➕ 加倉掛單\n\n"
                f"#{addon_result['trade_id']} {action} {symbol}\n"
                f"加倉點: {entry_2_price}{avg_text}\n"
                f"止損: {addon['stop_loss']} | 止盈: {addon['take_profit']}"
            )
            await self.telegram._safe_send(text)
        except Exception as e:
            logger.error("Addon entry error: %s", e)

    # ── 快速回饋學習 ──

    async def _quick_feedback_loop(self):
        """每 60 秒檢查持倉的快速回饋點（5min/30min/1hr）"""
        logger.info("Quick feedback loop started")
        await asyncio.sleep(30)  # 初始延遲

        while self._running:
            try:
                open_trades = self.db.get_open_trades()
                for trade in open_trades:
                    price = self.market.get_current_price(trade.symbol)
                    if price:
                        self.learning.check_quick_feedback(trade, price)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Quick feedback error: %s", e)
            await asyncio.sleep(60)

    # ── 延遲覆盤 ──

    async def _delayed_review_loop(self):
        """每 30 分鐘檢查是否有需要覆盤的交易（平倉超過 4 小時）"""
        logger.info("Delayed review loop started (check every 30min, review after 4h)")
        await asyncio.sleep(60)  # 初始延遲

        while self._running:
            try:
                await self.learning.run_pending_reviews(
                    notify_callback=self._on_delayed_review
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Delayed review loop error: %s", e)
            await asyncio.sleep(1800)  # 每 30 分鐘檢查一次

    async def _on_delayed_review(self, trade, review, events):
        """延遲覆盤完成後的通知回調"""
        try:
            # 發送覆盤結果（作為獨立通知）
            if review:
                result_data = {
                    "exit_price": trade.exit_price,
                    "profit_pct": trade.profit_pct,
                    "outcome": trade.outcome,
                    "hold_duration": trade.hold_duration or 0,
                }
                await self.telegram.send_exit_notification(trade, result_data, review)

            # 同步分析師權重
            self._sync_analyst_weights()

            # 發送學習事件
            for event in events:
                await self.telegram.send_learning_event(event)
        except Exception as e:
            logger.error("Failed to send delayed review notification: %s", e)

    # ── 市場掃描器 ──

    async def _market_scanner_loop(self):
        """每 N 分鐘主動掃描市場，結合近期分析師觀點尋找入場機會"""
        scanner_cfg = self.config.get("market_scanner", {})
        interval = scanner_cfg.get("interval_seconds", 300)
        lookback_hours = scanner_cfg.get("lookback_hours", 4)
        min_cooldown = scanner_cfg.get("min_cooldown_seconds", 600)
        min_messages = scanner_cfg.get("min_analyst_messages", 1)

        # 幣種關鍵字對應（用於過濾分析師訊息，簡繁體都要）
        symbol_keywords = {
            "BTCUSDT": ["BTC", "btc", "比特幣", "比特币", "大餅", "大饼"],
            "ETHUSDT": ["ETH", "eth", "乙太", "以太", "以太坊", "姨太"],
        }
        allowed_symbols = self.config.get("trading", {}).get(
            "allowed_symbols", ["BTCUSDT", "ETHUSDT"]
        )
        all_keywords = []
        for sym in allowed_symbols:
            all_keywords.extend(symbol_keywords.get(sym, [sym.replace("USDT", "")]))

        logger.info("Market scanner started (interval=%ds, lookback=%dh, cooldown=%ds)",
                     interval, lookback_hours, min_cooldown)

        # 初始延遲 60 秒，等其他模組啟動完成
        await asyncio.sleep(60)

        while self._running:
            try:
                # 1. 冷卻檢查
                if self._last_ai_call_time:
                    elapsed = (datetime.now(timezone.utc) - self._last_ai_call_time).total_seconds()
                    if elapsed < min_cooldown:
                        logger.debug("Scanner: cooldown active (%.0fs / %ds), skipping",
                                     elapsed, min_cooldown)
                        await asyncio.sleep(interval)
                        continue

                # 2. 查詢近期相關分析師訊息
                recent_msgs = self.db.get_recent_analyst_messages_for_symbols(
                    hours=lookback_hours,
                    keywords=all_keywords,
                )

                if len(recent_msgs) < min_messages:
                    logger.debug("Scanner: only %d relevant messages (need %d), skipping",
                                 len(recent_msgs), min_messages)
                    await asyncio.sleep(interval)
                    continue

                # 3. 交易是否啟用
                if not self.config.get("trading", {}).get("enabled", True):
                    await asyncio.sleep(interval)
                    continue

                # 4. 掃描所有允許的幣種（同方向可同時持倉）
                open_trades = self.db.get_open_trades()
                held_symbols = {t.symbol for t in open_trades}
                scan_symbols = allowed_symbols  # 不再跳過已持倉幣種

                # 5. 過濾關鍵字（只保留需要掃描的幣種）
                scan_keywords = []
                for sym in scan_symbols:
                    scan_keywords.extend(symbol_keywords.get(sym, [sym.replace("USDT", "")]))

                relevant_msgs = [
                    m for m in recent_msgs
                    if any(kw.upper() in m.content.upper() for kw in scan_keywords)
                ] if scan_symbols != allowed_symbols else recent_msgs

                if len(relevant_msgs) < min_messages:
                    logger.debug("Scanner: only %d relevant messages for %s, skipping",
                                 len(relevant_msgs), scan_symbols)
                    await asyncio.sleep(interval)
                    continue

                logger.info("Scanner: found %d relevant messages, scanning %s (held: %s)",
                            len(relevant_msgs), scan_symbols,
                            ", ".join(held_symbols) if held_symbols else "none")

                # 6. 執行掃描分析（只掃沒有持倉的幣種）
                await self._on_scanner_triggered(relevant_msgs, scan_symbols)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Market scanner error: %s", e)
                try:
                    await self.telegram.send_error(f"市場掃描器錯誤: {e}")
                except Exception:
                    pass

            await asyncio.sleep(interval)

    async def _on_scanner_triggered(self, db_messages: list, symbols: list[str]):
        """處理掃描器觸發的分析流程"""
        logger.info("=" * 40)
        logger.info("Scanner analysis triggered with %d messages for %s",
                     len(db_messages), symbols)
        logger.info("=" * 40)

        try:
            # 記錄 AI 呼叫時間
            self._last_ai_call_time = datetime.now(timezone.utc)

            # 1. 決策引擎處理掃描信號
            decision = self.decision.process_scanner_signals(db_messages, symbols)

            if decision is None:
                logger.info("Scanner: no actionable signal")
                return

            action = decision.get("action", "")
            analyst_names = ["scanner"]

            # 2. SKIP
            if action == "SKIP":
                logger.info("Scanner: AI recommends SKIP")
                self.db.save_ai_decision(
                    decision, outcome="SKIP", analyst_names=analyst_names,
                )
                return

            # 3. 被風控拒絕
            if decision.get("_rejected"):
                logger.warning("Scanner signal rejected by risk manager")
                self.db.save_ai_decision(
                    decision, outcome="REJECTED", analyst_names=analyst_names,
                )
                return

            # 5. 交易關閉檢查
            trading_cfg = self.config.get("trading", {})
            if not trading_cfg.get("enabled", True):
                logger.info("Scanner: trading disabled")
                return

            # 6. Telegram 通知 + 確認倒數（標記為掃描器觸發）
            countdown = trading_cfg.get("confirmation_delay", 30)
            decision["_scanner_triggered"] = True

            result = await self.telegram.send_signal(decision, countdown=countdown)

            if result.get("cancelled"):
                logger.info("Scanner trade cancelled by user")
                self.db.save_ai_decision(
                    decision, outcome="CANCELLED",
                    analyst_names=analyst_names,
                    cancel_reason=result.get("cancel_reason", ""),
                )
                return

            # 7. 執行交易
            if trading_cfg.get("auto_execute", True):
                trade_result = self.trader.execute_trade(decision)

                if trade_result.get("success"):
                    self.db.save_ai_decision(
                        decision, outcome="EXECUTED",
                        analyst_names=analyst_names,
                        trade_id=trade_result["trade_id"],
                    )

                    # 記錄原始分析師的貢獻
                    for m in db_messages:
                        self.db.record_analyst_call(
                            trade_id=trade_result["trade_id"],
                            analyst_name=m.analyst_name,
                            direction=decision["action"],
                            message=m.content,
                        )

                    # 翻倉通知
                    if trade_result.get("flipped"):
                        await self._handle_flip_notification(trade_result["flipped"])

                    if trade_result.get("pending"):
                        await self.telegram.send_pending_order(trade_result)
                        logger.info("Scanner trade #%d LIMIT order pending",
                                    trade_result["trade_id"])
                    else:
                        await self.telegram.send_entry_confirmation(trade_result)
                        self.risk.record_trade_time()
                        logger.info("Scanner trade #%d executed successfully",
                                    trade_result["trade_id"])
                else:
                    error = trade_result.get("error", "Unknown error")
                    logger.error("Scanner trade execution failed: %s", error)
                    await self.telegram.send_error(f"掃描器交易執行失敗: {error}")

        except Exception as e:
            logger.exception("Error in scanner analysis pipeline")
            try:
                await self.telegram.send_error(f"掃描器分析錯誤: {e}")
            except Exception:
                pass

    # ── 工具方法 ──

    def _format_decisions(self, decisions) -> list[dict]:
        """將 DB 的 AIDecision 記錄轉成 dict list 供 AI 報告使用"""
        result = []
        for d in decisions:
            reasoning = d.reasoning or ""
            if reasoning.startswith("{"):
                try:
                    r = json.loads(reasoning)
                    # 提取關鍵推理摘要
                    reasoning = r.get("skip_reason", "") or r.get("summary", "") or str(r)
                except (json.JSONDecodeError, TypeError):
                    pass

            result.append({
                "timestamp": d.timestamp.strftime("%H:%M") if d.timestamp else "",
                "symbol": d.symbol or "",
                "action": d.action or "",
                "confidence": d.confidence or 0,
                "outcome": d.outcome or "",
                "reasoning": reasoning,
                "risk_summary": d.risk_summary or "",
                "cancel_reason": d.cancel_reason or "",
            })
        return result

    def _sync_analyst_weights(self):
        """同步數據庫中的分析師權重到 Discord listener"""
        analysts = self.db.get_all_analysts()
        for a in analysts:
            self.discord.update_analyst_weight(a.name, a.current_weight)
        logger.info("Synced %d analyst weights", len(analysts))

    def _retry_failed_reviews(self):
        """啟動時檢查並重新覆盤失敗的交易"""
        import time as _time
        closed = self.db.get_closed_trades(limit=500)
        need_review = []
        for t in closed:
            review = t.review
            if not review:
                need_review.append(t)
                continue
            try:
                r = json.loads(review) if isinstance(review, str) else review
                if not isinstance(r, dict) or "error" in r or "overall_score" not in r:
                    need_review.append(t)
            except (json.JSONDecodeError, TypeError):
                need_review.append(t)

        if not need_review:
            logger.info("All closed trades have valid reviews")
            return

        logger.info("Found %d trades with failed reviews, re-reviewing...", len(need_review))
        success = 0
        for trade in need_review:
            try:
                analyst_opinions = trade.analyst_opinions or "N/A"
                technical_signals = trade.technical_signals or "{}"
                ai_reasoning = trade.ai_reasoning or "N/A"
                if len(analyst_opinions) > 3000:
                    analyst_opinions = analyst_opinions[:3000] + "\n...(截斷)"
                if len(ai_reasoning) > 2000:
                    ai_reasoning = ai_reasoning[:2000] + "\n...(截斷)"

                trade_data = {
                    "symbol": trade.symbol,
                    "direction": trade.direction,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "stop_loss": trade.stop_loss,
                    "take_profit": trade.take_profit or "[]",
                    "position_size": trade.position_size,
                    "confidence": trade.confidence,
                    "hold_duration": f"{(trade.hold_duration or 0) // 60}m",
                    "outcome": trade.outcome,
                    "profit_pct": trade.profit_pct,
                    "analyst_opinions": analyst_opinions,
                    "technical_signals": (
                        json.loads(technical_signals)
                        if isinstance(technical_signals, str)
                        else technical_signals
                    ),
                    "ai_reasoning": ai_reasoning,
                    "quick_feedback": "N/A",
                }

                review = self.ai.review_trade(trade_data)
                if review and "overall_score" in review and "error" not in review:
                    self.db.update_trade(trade.id, review=review)
                    logger.info("Re-review Trade #%d OK: score=%s/10",
                                trade.id, review.get("overall_score"))
                    success += 1
                    _time.sleep(2)  # 避免 API rate limit
                else:
                    logger.warning("Re-review Trade #%d failed: %s",
                                   trade.id, review)
            except Exception as e:
                logger.error("Re-review Trade #%d error: %s", trade.id, e)

        logger.info("Re-review complete: %d/%d succeeded", success, len(need_review))

    def _get_local_tz(self):
        """取得設定的時區"""
        tz_name = self.config.get("schedule", {}).get("timezone", "Asia/Taipei")
        try:
            from zoneinfo import ZoneInfo
            return ZoneInfo(tz_name)
        except Exception:
            # fallback: UTC+8
            return timezone(timedelta(hours=8))

    def _seconds_until(self, target_hour: int, target_minute: int = 0) -> float:
        """計算距離下一個目標時間的秒數（本地時區）"""
        local_tz = self._get_local_tz()
        now = datetime.now(local_tz)
        target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return (target - now).total_seconds()

    async def _manual_review(self, trade_id: int) -> dict:
        """手動觸發覆盤"""
        trade = self.db.get_trade(trade_id)
        if not trade:
            return {"review": None, "error": "Trade not found"}
        if trade.status not in ("CLOSED",):
            return {"review": None, "error": "Trade not closed yet"}
        result = await self.learning.on_trade_closed(trade_id)
        return result

    async def generate_and_send_briefing(self):
        """產生並發送早報（供定時任務和手動指令共用）"""
        logger.info("Generating morning briefing...")

        # 1. 取得過去 24 小時分析師訊息
        recent_msgs = self.db.get_recent_analyst_messages(hours=24)
        analyst_msgs = [
            {
                "analyst": m.analyst_name,
                "content": m.content,
                "timestamp": m.timestamp.strftime("%m-%d %H:%M"),
            }
            for m in recent_msgs
        ]

        # 2. 取得市場數據
        market_data = {}
        for symbol in self.config["binance"].get("symbols", ["BTCUSDT"]):
            data = self.market.get_symbol_data(symbol)
            if "error" not in data:
                market_data[symbol] = data

        # 3. 取得持倉
        open_trades = self.db.get_open_trades()
        open_trades_info = [
            {
                "trade_id": t.id,
                "symbol": t.symbol,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "stop_loss": t.stop_loss,
                "take_profit": json.loads(t.take_profit) if isinstance(t.take_profit, str) else t.take_profit,
            }
            for t in open_trades
        ] if open_trades else None

        # 4. 績效統計
        performance = self.db.get_performance_stats()

        # 5. 過去 24 小時 AI 決策記錄
        recent_decisions = self._format_decisions(
            self.db.get_recent_decisions(hours=24)
        )

        # 5.5 今日經濟日曆
        econ_events = self.calendar.get_events(days_ahead=2)
        econ_text = self.calendar.format_for_ai(econ_events)

        # 6. AI 產出早報
        briefing = self.ai.generate_morning_briefing(
            analyst_messages=analyst_msgs,
            market_data=market_data,
            open_trades=open_trades_info,
            performance_stats=performance,
            recent_decisions=recent_decisions,
            economic_events=econ_text,
        )

        # 7. 發送 Telegram
        await self.telegram.send_morning_briefing(briefing)
        logger.info("Morning briefing sent")

    async def _morning_briefing_loop(self):
        """每日 8:00 AM 發送早報"""
        morning_hour = self.config.get("schedule", {}).get("morning_hour", 8)
        logger.info("Morning briefing scheduled at %d:00", morning_hour)

        while self._running:
            try:
                wait = self._seconds_until(morning_hour)
                logger.info("Next morning briefing in %.0f seconds", wait)
                await asyncio.sleep(wait)

                if not self._running:
                    break

                await self.generate_and_send_briefing()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Morning briefing error: %s", e)
                try:
                    await self.telegram.send_error(f"早報產生失敗: {e}")
                except Exception:
                    pass
                await asyncio.sleep(60)

    async def _evening_summary_loop(self):
        """每日 10:00 PM 發送晚報"""
        evening_hour = self.config.get("schedule", {}).get("evening_hour", 22)
        logger.info("Evening summary scheduled at %d:00", evening_hour)

        while self._running:
            try:
                wait = self._seconds_until(evening_hour)
                logger.info("Next evening summary in %.0f seconds", wait)
                await asyncio.sleep(wait)

                if not self._running:
                    break

                logger.info("Generating evening summary...")

                # 1. 取得今日分析師訊息
                today_msgs = self.db.get_today_analyst_messages()
                analyst_msgs = [
                    {
                        "analyst": m.analyst_name,
                        "content": m.content,
                        "timestamp": m.timestamp.strftime("%H:%M"),
                    }
                    for m in today_msgs
                ]

                # 2. 取得今日交易
                today_trades = self.db.get_today_trades()
                today_trades_info = [t.to_dict() for t in today_trades]

                # 3. 取得持倉
                open_trades = self.db.get_open_trades()
                open_trades_info = [
                    {
                        "trade_id": t.id,
                        "symbol": t.symbol,
                        "direction": t.direction,
                        "entry_price": t.entry_price,
                        "stop_loss": t.stop_loss,
                    }
                    for t in open_trades
                ] if open_trades else None

                # 4. 今日績效
                day_stats = self.db.get_performance_stats(days=1)
                day_stats["today_pnl"] = self.db.get_today_pnl()

                # 5. 總績效
                overall_stats = self.db.get_performance_stats()

                # 6. 今日 AI 決策記錄
                today_decisions = self._format_decisions(
                    self.db.get_today_decisions()
                )

                # 6.5 今日經濟數據公布結果
                today_econ = self.calendar.get_today_events()
                econ_text = self.calendar.format_for_ai(today_econ)

                # 7. AI 產出晚報
                summary = self.ai.generate_evening_summary(
                    today_trades=today_trades_info,
                    analyst_messages=analyst_msgs,
                    open_trades=open_trades_info,
                    performance_stats=day_stats,
                    overall_stats=overall_stats,
                    today_decisions=today_decisions,
                    economic_events=econ_text,
                )

                # 7. 發送 Telegram
                stats_for_tg = {
                    "total": len(today_trades),
                    "win_rate": day_stats.get("win_rate", 0),
                    "today_pnl": day_stats.get("today_pnl", 0),
                    "total_profit_pct": overall_stats.get("total_profit_pct", 0),
                }
                await self.telegram.send_evening_summary(summary, stats_for_tg)
                logger.info("Evening summary sent")

                # 8. 檢查緊急停止
                if self.risk.is_emergency_stop():
                    await self.telegram.send_error(
                        "🛑 緊急停止：總虧損已達上限！系統已暫停交易。"
                    )
                    self.config["trading"]["enabled"] = False

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Evening summary error: %s", e)
                try:
                    await self.telegram.send_error(f"晚報產生失敗: {e}")
                except Exception:
                    pass
                await asyncio.sleep(60)


def main():
    bot = TradingBot()

    # 處理 Ctrl+C
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(bot.start())
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt")
        loop.run_until_complete(bot.shutdown())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
