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
        self.db = Database(self.config["database"]["path"])
        self.market = MarketData(self.config)
        self.ai = AIAnalyzer(self.config)
        self.risk = RiskManager(self.config, self.db)
        self.calendar = EconomicCalendar(self.config)
        self.decision = DecisionEngine(
            self.config, self.db, self.market, self.ai, self.risk, self.calendar
        )
        self.trader = BinanceTrader(self.config, self.db)
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

        # 載入分析師最新權重
        self._sync_analyst_weights()

        # 啟動 Telegram
        try:
            await self.telegram.start()
            logger.info("Telegram bot started")
        except Exception as e:
            logger.error("Telegram start failed: %s", e)

        # 啟動持倉監控
        monitor_task = asyncio.create_task(
            self.trader.monitor_positions(callback=self._on_position_event)
        )

        # 啟動每日早報（8:00 AM）和晚報（10:00 PM）
        morning_task = asyncio.create_task(self._morning_briefing_loop())
        evening_task = asyncio.create_task(self._evening_summary_loop())

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

            # 0. 儲存所有分析師訊息到資料庫（供早報/晚報使用）
            for m in messages:
                self.db.save_analyst_message(
                    analyst_name=m.analyst,
                    channel=m.channel_name,
                    content=m.content,
                )

            analyst_names = [m.analyst for m in messages]

            # 1. 決策引擎處理
            decision = self.decision.process_signals(messages)

            if decision is None:
                logger.info("No actionable signal — skipping")
                return

            action = decision.get("action", "")

            # 2. SKIP — AI 決定不操作
            if action == "SKIP":
                logger.info("No actionable signal — skipping")
                self.db.save_ai_decision(
                    decision, outcome="SKIP", analyst_names=analyst_names,
                )
                return

            # 3. 處理調整持倉
            if action == "ADJUST":
                await self._handle_adjust(decision)
                return

            # 4. 檢查是否被風控拒絕
            if decision.get("_rejected"):
                logger.warning("Signal rejected by risk manager")
                self.db.save_ai_decision(
                    decision, outcome="REJECTED", analyst_names=analyst_names,
                )
                await self.telegram.send_rejected_signal(decision)
                return

            # 5. 交易設定檢查
            trading_cfg = self.config.get("trading", {})
            if not trading_cfg.get("enabled", True):
                logger.info("Trading disabled — signal only mode")
                await self.telegram.send_signal(decision, countdown=0)
                return

            # 6. Telegram 通知 + 30 秒確認
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

            # 7. 執行交易
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

                    await self.telegram.send_entry_confirmation(trade_result)
                    self.risk.record_trade_time()
                    logger.info("Trade #%d executed successfully", trade_result["trade_id"])
                else:
                    error = trade_result.get("error", "Unknown error")
                    logger.error("Trade execution failed: %s", error)
                    await self.telegram.send_error(f"交易執行失敗: {error}")

        except Exception as e:
            logger.exception("Error in signal processing pipeline")
            await self.telegram.send_error(f"分析流程錯誤: {e}")

    async def _handle_adjust(self, decision: dict):
        """處理 AI 的 ADJUST 決策 — 調整現有持倉的止盈止損"""
        trade_id = decision.get("trade_id")
        new_sl = decision.get("new_stop_loss")
        new_tp = decision.get("new_take_profit")
        reasoning = decision.get("reasoning", {})

        logger.info("Adjusting trade #%s: SL=%s, TP=%s", trade_id, new_sl, new_tp)

        # 發送 Telegram 通知
        text = (
            f"🔄 AI 建議調整持倉\n\n"
            f"交易 #{trade_id} | {decision.get('symbol', '?')}\n"
            f"信心: {decision.get('confidence', 0)}%\n\n"
            f"調整內容:\n"
        )
        if new_sl:
            text += f"  停損 → {new_sl}\n"
        if new_tp:
            text += f"  目標 → {new_tp}\n"
        text += (
            f"\n原因: {reasoning.get('adjustment_reason', 'N/A')}\n"
            f"分析師: {reasoning.get('analyst_consensus', 'N/A')}\n"
        )

        await self.telegram.bot.send_message(
            chat_id=self.telegram.chat_id, text=text
        )

        # 執行調整
        result = self.trader.adjust_trade(trade_id, new_sl, new_tp)

        if result.get("success"):
            changes = "\n".join(f"  • {c}" for c in result.get("changes", []))
            await self.telegram.bot.send_message(
                chat_id=self.telegram.chat_id,
                text=f"✅ 交易 #{trade_id} 已調整\n{changes}",
            )
        else:
            await self.telegram.send_error(
                f"調整失敗: {result.get('error', 'Unknown')}"
            )

    async def _on_position_event(self, event_type: str, trade, data: dict):
        """持倉監控回調"""
        if event_type in ("stop_loss", "take_profit"):
            logger.info(
                "Position closed by %s: trade #%d",
                event_type, trade.id,
            )

            # AI 覆盤 + 學習流程
            learn_result = await self.learning.on_trade_closed(trade.id)
            review = learn_result.get("review")
            events = learn_result.get("events", [])

            # 發送平倉通知
            await self.telegram.send_exit_notification(trade, data, review)

            # 同步分析師權重
            self._sync_analyst_weights()

            # 發送所有學習事件通知
            for event in events:
                await self.telegram.send_learning_event(event)

        elif event_type == "update":
            # 可選：重要價格變動時通知
            pass

    # ── 市場掃描器 ──

    async def _market_scanner_loop(self):
        """每 N 分鐘主動掃描市場，結合近期分析師觀點尋找入場機會"""
        scanner_cfg = self.config.get("market_scanner", {})
        interval = scanner_cfg.get("interval_seconds", 300)
        lookback_hours = scanner_cfg.get("lookback_hours", 4)
        min_cooldown = scanner_cfg.get("min_cooldown_seconds", 600)
        min_messages = scanner_cfg.get("min_analyst_messages", 1)

        # 幣種關鍵字對應（用於過濾分析師訊息）
        symbol_keywords = {
            "BTCUSDT": ["BTC", "比特幣", "大餅"],
            "ETHUSDT": ["ETH", "乙太", "以太", "姨太"],
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

                logger.info("Scanner: found %d relevant analyst messages, triggering analysis",
                            len(recent_msgs))

                # 4. 執行掃描分析
                await self._on_scanner_triggered(recent_msgs, allowed_symbols)

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

            # 3. ADJUST
            if action == "ADJUST":
                await self._handle_adjust(decision)
                return

            # 4. 被風控拒絕
            if decision.get("_rejected"):
                logger.warning("Scanner signal rejected by risk manager")
                self.db.save_ai_decision(
                    decision, outcome="REJECTED", analyst_names=analyst_names,
                )
                await self.telegram.send_rejected_signal(decision)
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

                # 6. 發送 Telegram
                await self.telegram.send_morning_briefing(briefing)
                logger.info("Morning briefing sent")

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
