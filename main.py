"""
DC Trading Bot — 跟單模式 (Follow Mode Only)

流程：Discord 收到分析師訊息 → AI 解析訊號 → Telegram 倒數確認 → 執行 Paper Trade
"""

import asyncio
import logging

from modules.ai_analyzer import AIAnalyzer
from modules.database import Database
from modules.discord_listener import DiscordListener
from modules.paper_trader import PaperTrader
from modules.price_feed import PriceFeed
from modules.telegram_notifier import TelegramNotifier
from utils.helpers import load_config, setup_logging

logger = logging.getLogger(__name__)


class TradingBot:
    def __init__(self):
        self.config = load_config()
        setup_logging(self.config)

        logger.info("=" * 50)
        logger.info("DC Trading Bot (Follow Mode) starting...")
        logger.info("=" * 50)

        tz_name = self.config.get("schedule", {}).get("timezone", "UTC")
        self.db = Database(self.config["database"]["path"], tz_name=tz_name)

        symbols = self.config.get("binance", {}).get("symbols", ["BTCUSDT", "ETHUSDT"])
        self.price_feed = PriceFeed(symbols=symbols)
        self.trader = PaperTrader(self.config, self.db, price_feed=self.price_feed)

        self.ai = AIAnalyzer(self.config)
        self.telegram = TelegramNotifier(self.config, db=self.db, trader=self.trader)
        self.discord = DiscordListener(self.config)

        self.discord.set_analysis_callback(self._on_signals_received)

    async def start(self):
        logger.info("Starting services...")

        await self.telegram.start()
        logger.info("Telegram started")

        await self.price_feed.start()
        logger.info("WebSocket price feed started")

        asyncio.create_task(
            self.trader.monitor_positions(callback=self._on_position_event)
        )

        try:
            logger.info("Starting Discord listener...")
            await self.discord.start()
        except Exception as e:
            logger.error("Discord error: %s", e)
        finally:
            await self.shutdown()

    async def shutdown(self):
        logger.info("Shutting down...")
        try:
            await self.discord.stop()
        except Exception:
            pass
        try:
            await self.telegram.stop()
        except Exception:
            pass
        logger.info("Shutdown complete")

    # ── 核心：Discord 訊息 → AI 解析 → 執行 ──

    async def _on_signals_received(self, messages: list):
        logger.info("=" * 40)
        logger.info("Analysis triggered: %d message(s)", len(messages))
        logger.info("=" * 40)

        try:
            # 儲存訊息到 DB
            for m in messages:
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

            # AI 解析訊號
            lines = [
                f"[{m.analyst} {m.timestamp.strftime('%H:%M')}]: {m.content}"
                for m in messages
            ]
            all_images = [img for m in messages for img in getattr(m, "images", [])]
            combined_text = "\n".join(lines)

            parsed_list = self.ai.parse_signal(combined_text, images=all_images[:4] or None)
            logger.info("AI parsed %d signal(s)", len(parsed_list))

            analyst_names = [m.analyst for m in messages]
            for parsed in parsed_list:
                await self._handle_parsed_signal(parsed, messages, analyst_names)

        except Exception as e:
            logger.exception("Signal processing error")
            await self.telegram.send_error(f"訊號處理錯誤: {e}")

    async def _handle_parsed_signal(self, parsed: dict, messages: list, analyst_names: list):
        action = parsed.get("action", "SKIP")
        symbol = parsed.get("symbol", "BTCUSDT")

        if action == "SKIP":
            logger.info("SKIP [%s]: %s", symbol, parsed.get("skip_reason", ""))
            return

        if action == "CLOSE":
            await self._handle_close(symbol)
            return

        if action not in ("LONG", "SHORT"):
            return

        entry_1 = parsed.get("entry_1")
        entry_2 = parsed.get("entry_2")
        stop_loss = parsed.get("stop_loss")
        take_profit = parsed.get("take_profit") or []

        # 必要欄位驗證
        if not entry_1 or entry_1.get("price") is None:
            logger.warning("[%s] Missing entry_1, skipping", symbol)
            return
        if stop_loss is None:
            logger.warning("[%s] Missing stop_loss, skipping", symbol)
            return
        if not take_profit:
            logger.warning("[%s] Missing take_profit, skipping", symbol)
            return

        # 倉位（雙入場點各半）
        follow_pos = self.config.get("trading", {}).get("follow_position_size", 5.0)
        has_entry_2 = entry_2 and entry_2.get("price") is not None
        pos_size = round(follow_pos / 2, 1) if has_entry_2 else follow_pos

        # 從 config 取槓桿
        leverage = self._get_leverage(analyst_names)

        analyst_tag = ", ".join(set(analyst_names))
        decision = {
            "action": action,
            "symbol": symbol,
            "confidence": 90,
            "leverage": leverage,
            "entry": entry_1,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": pos_size,
            "risk_reward": 2.0,
            "reasoning": {
                "analyst_consensus": f"跟單: {analyst_tag}",
                "technical": "100% 按照分析師指定點位",
                "sentiment": "N/A",
            },
            "risk_assessment": {},
            "_follow_mode": True,
            "_analyst_messages": [
                {"analyst": m.analyst, "content": m.content} for m in messages
            ],
        }

        logger.info(
            "Signal: %s %s @ %s (SL=%s TP=%s lev=%dx pos=%.1f%%)",
            action, symbol, entry_1["price"], stop_loss, take_profit, leverage, pos_size,
        )

        # Telegram 倒數確認
        countdown = self.config.get("trading", {}).get("confirmation_delay", 15)
        result = await self.telegram.send_signal(decision, countdown=countdown)

        if result.get("cancelled"):
            logger.info("Trade cancelled by user")
            return

        # 執行主交易
        await self._execute_trade(decision, analyst_names, messages)

        # 執行加倉點
        if has_entry_2:
            addon = {**decision, "entry": entry_2, "position_size": round(follow_pos / 2, 1)}
            await self._execute_addon(addon)

    async def _execute_trade(self, decision: dict, analyst_names: list, messages: list):
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
                logger.info("Trade #%d LIMIT pending @ %s",
                            trade_result["trade_id"], decision["entry"]["price"])
            else:
                await self.telegram.send_entry_confirmation(trade_result)
                logger.info("Trade #%d MARKET executed", trade_result["trade_id"])
        else:
            error = trade_result.get("error", "Unknown")
            logger.error("Trade failed: %s", error)
            await self.telegram.send_error(f"交易執行失敗: {error}")

    async def _execute_addon(self, decision: dict):
        result = self.trader.execute_trade(decision)
        if result.get("success"):
            if result.get("pending"):
                await self.telegram.send_pending_order(result)
            else:
                await self.telegram.send_entry_confirmation(result)
            logger.info("Addon trade #%d placed @ %s",
                        result["trade_id"], decision["entry"]["price"])
        else:
            logger.error("Addon trade failed: %s", result.get("error"))

    async def _handle_close(self, symbol: str):
        open_trades = self.db.get_open_trades()
        to_close = [t for t in open_trades if t.symbol == symbol]

        if not to_close:
            await self.telegram._safe_send(f"⚠️ 分析師指示平倉 {symbol}，但目前沒有持倉")
            return

        for trade in to_close:
            result = self.trader.close_trade(trade.id)
            if result.get("success"):
                await self.telegram.send_exit_notification(trade, result, review=None)
                logger.info("Closed #%d %s %s", trade.id, trade.direction, symbol)
            else:
                logger.error("Close failed #%d: %s", trade.id, result.get("error"))

    async def _on_position_event(self, event_type: str, trade, data: dict):
        if event_type == "limit_filled":
            logger.info("LIMIT filled: #%d %s %s @ %s",
                        data["trade_id"], data["direction"],
                        data["symbol"], data["entry_price"])
            await self.telegram.send_entry_confirmation(data)
            return

        if event_type == "tp1_hit":
            tp_index = data.get("tp_index", 0)
            tp_label = f"TP{tp_index + 1}"
            text = (
                f"🎯 {tp_label} 止盈到達！\n\n"
                f"#{trade.id} {trade.direction} {trade.symbol}\n"
                f"目標價格: {data.get('tp1_price', 0)}\n"
                f"已平倉: {data.get('closed_qty', 0)}\n"
                f"剩餘倉位: {data.get('remaining_qty', 0)}"
            )
            if tp_index == 0 and data.get("breakeven_sl"):
                text += f"\n\n🛡️ 止損移至保本: {data['breakeven_sl']}"
            await self.telegram._safe_send(text)
            return

        if event_type in ("stop_loss", "take_profit", "liquidation"):
            await self.telegram.send_exit_notification(trade, data, review=None)

    # ── 工具 ──

    def _get_leverage(self, analyst_names: list) -> int:
        channels = self.config.get("discord", {}).get("monitored_channels", [])
        for ch in channels:
            if ch["analyst"] in analyst_names:
                return ch.get("leverage", 100)
        return self.config.get("trading", {}).get("default_leverage", 100)


def main():
    bot = TradingBot()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(bot.start())
    except KeyboardInterrupt:
        loop.run_until_complete(bot.shutdown())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
