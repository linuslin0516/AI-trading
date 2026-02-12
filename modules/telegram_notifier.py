import asyncio
import json
import logging
from datetime import datetime, timezone

import requests
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler, ContextTypes,
    MessageHandler, filters,
)

from utils.helpers import format_price, format_pct

logger = logging.getLogger(__name__)

MARKET_DATA_URL = "https://data-api.binance.vision"


class TelegramNotifier:
    def __init__(self, config: dict, db=None, trader=None):
        self.config = config
        self._db = db
        self._trader = trader
        tg_cfg = config.get("telegram", {})
        self.bot_token = tg_cfg.get("bot_token", "")
        self.chat_id = tg_cfg.get("chat_id", "")
        self.notify_cfg = config.get("notifications", {})

        self.bot = Bot(token=self.bot_token)
        self._app: Application | None = None
        self._pending_decisions: dict[str, dict] = {}  # msg_id -> decision
        self._cancel_callbacks: dict[str, asyncio.Event] = {}
        self._cancel_reasons: dict[str, dict] = {}  # msg_id -> {event, reason, waiting_text}

        logger.info("TelegramNotifier initialized")

    async def start(self):
        """啟動 Telegram Bot（持續輪詢，隨時接收指令）"""
        self._app = (
            Application.builder()
            .token(self.bot_token)
            .build()
        )
        self._app.add_handler(CallbackQueryHandler(self._button_callback))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("stop", self._cmd_stop))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(CommandHandler("test_trade", self._cmd_test_trade))
        self._app.add_handler(CommandHandler("positions", self._cmd_positions))
        self._app.add_handler(CommandHandler("pnl", self._cmd_pnl))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._text_handler)
        )

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot started with persistent polling")

    async def stop(self):
        if self._app:
            if self._app.updater and self._app.updater.running:
                await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

    # ── 通知方法 ──

    async def send_signal(self, decision: dict, countdown: int = 30) -> dict:
        """
        發送交易訊號通知，附帶倒數計時和取消按鈕

        Returns:
            {"executed": True/False, "cancelled": bool}
        """
        if not self.notify_cfg.get("notify_on_signal", True):
            return {"executed": True, "cancelled": False}

        action = decision.get("action", "?")
        symbol = decision.get("symbol", "?")
        confidence = decision.get("confidence", 0)
        entry = decision.get("entry", {})
        sl = decision.get("stop_loss", 0)
        tp = decision.get("take_profit", [])
        rr = decision.get("risk_reward", 0)
        reasoning = decision.get("reasoning", {})
        risk = decision.get("risk_assessment", {})
        pos_size = decision.get("position_size", 0)

        direction_icon = "🟢 LONG (做多)" if action == "LONG" else "🔴 SHORT (做空)"

        text = (
            f"{'=' * 30}\n"
            f"🔔 交易訊號\n"
            f"{'=' * 30}\n\n"
            f"{direction_icon}\n"
            f"交易對: {symbol}\n"
            f"信心分數: {confidence}%\n\n"
            f"📊 交易計畫\n"
            f"━━━━━━━━━━━━━━━\n"
            f"進場: {format_price(entry.get('price', 0))} ({entry.get('strategy', 'LIMIT')})\n"
            f"停損: {format_price(sl)}\n"
            f"目標 1: {format_price(tp[0]) if tp else 'N/A'}\n"
            f"目標 2: {format_price(tp[1]) if len(tp) > 1 else 'N/A'}\n"
            f"倉位: {pos_size}%\n"
            f"風報比: {rr:.2f}\n\n"
            f"🤖 AI 分析\n"
            f"━━━━━━━━━━━━━━━\n"
            f"共識: {reasoning.get('analyst_consensus', 'N/A')}\n"
            f"技術: {reasoning.get('technical', 'N/A')}\n"
            f"情緒: {reasoning.get('sentiment', 'N/A')}\n\n"
            f"📈 風險評估\n"
            f"━━━━━━━━━━━━━━━\n"
            f"最大虧損: {risk.get('max_loss_pct', 0):.2f}%\n"
            f"預期獲利: {risk.get('expected_profit_pct', [0])[0]:.2f}%\n"
            f"勝率: {risk.get('win_probability', 0) * 100:.0f}%\n\n"
            f"⏱️ {countdown} 秒後自動執行...\n"
        )

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("❌ 取消", callback_data="cancel"),
                InlineKeyboardButton("⚡ 立即執行", callback_data="execute_now"),
            ]
        ])

        logger.info("Sending trade signal to Telegram (chat_id=%s)...", self.chat_id)
        try:
            msg = await asyncio.wait_for(
                self.bot.send_message(
                    chat_id=self.chat_id,
                    text=text,
                    reply_markup=keyboard,
                ),
                timeout=30,
            )
        except asyncio.TimeoutError:
            logger.error("Telegram send_message timed out after 30s")
            return {"executed": True, "cancelled": False}
        except Exception as e:
            logger.error("Telegram send_message failed: %s", e)
            return {"executed": True, "cancelled": False}

        logger.info("Signal sent to Telegram (msg_id=%s)", msg.message_id)

        msg_id = str(msg.message_id)
        self._pending_decisions[msg_id] = decision
        cancel_event = asyncio.Event()
        self._cancel_callbacks[msg_id] = cancel_event

        logger.info("Waiting for trade confirmation (msg_id=%s, countdown=%ds)", msg_id, countdown)

        # 倒數計時（polling 已持續運行，不需要額外啟動）
        execute_now = False
        cancelled = False
        cancel_reason = ""

        for remaining in range(countdown, 0, -5):
            if cancel_event.is_set():
                # 檢查是取消還是立即執行
                if self._pending_decisions.get(msg_id, {}).get("_execute_now"):
                    execute_now = True
                else:
                    cancelled = True
                break
            await asyncio.sleep(min(5, remaining))

        if cancelled:
            cancel_reason = await self._ask_cancel_reason()

        # 清理
        self._pending_decisions.pop(msg_id, None)
        self._cancel_callbacks.pop(msg_id, None)

        if cancelled:
            await self.bot.edit_message_text(
                chat_id=self.chat_id,
                message_id=msg.message_id,
                text=text.replace(
                    f"⏱️ {countdown} 秒後自動執行...",
                    f"❌ 已取消\n原因：{cancel_reason}"
                ),
            )
            return {"executed": False, "cancelled": True, "cancel_reason": cancel_reason}

        status_text = "⚡ 立即執行中..." if execute_now else "✅ 倒數結束，執行中..."
        await self.bot.edit_message_text(
            chat_id=self.chat_id,
            message_id=msg.message_id,
            text=text.replace(
                f"⏱️ {countdown} 秒後自動執行...",
                status_text,
            ),
        )

        return {"executed": True, "cancelled": False}

    async def send_entry_confirmation(self, trade_result: dict):
        """進場確認通知"""
        if not self.notify_cfg.get("notify_on_entry", True):
            return

        text = (
            f"✅ 已進場\n\n"
            f"交易 #{trade_result['trade_id']}\n"
            f"{trade_result['direction']} {trade_result['symbol']}\n"
            f"進場價: {format_price(trade_result['entry_price'])}\n"
            f"數量: {trade_result['quantity']}\n"
            f"停損: {format_price(trade_result['stop_loss'])}\n"
            f"目標: {', '.join(format_price(t) for t in trade_result['take_profit'])}\n\n"
            f"📊 持倉監控中..."
        )
        await self.bot.send_message(chat_id=self.chat_id, text=text)

    async def send_position_update(self, trade, current_price: float, unrealized_pct: float):
        """持倉更新（可選，避免太頻繁）"""
        pass  # 只在重要變化時發送

    async def send_exit_notification(self, trade, result: dict, review: dict | None = None):
        """平倉通知 + AI 覆盤"""
        if not self.notify_cfg.get("notify_on_exit", True):
            return

        outcome_icon = "✅" if result.get("outcome") == "WIN" else "❌"
        profit = result.get("profit_pct", 0)
        hold_sec = result.get("hold_duration", 0)
        hold_str = self._format_duration(hold_sec)

        text = (
            f"{'=' * 30}\n"
            f"{outcome_icon} 交易完成 | {trade.symbol} {trade.direction}\n"
            f"{'=' * 30}\n\n"
            f"📊 交易摘要\n"
            f"━━━━━━━━━━━━━━━\n"
            f"進場: {format_price(trade.entry_price)}\n"
            f"出場: {format_price(result.get('exit_price', 0))}\n"
            f"獲利: {format_pct(profit)}\n"
            f"持倉: {hold_str}\n"
        )

        if review:
            text += (
                f"\n🤖 AI 覆盤\n"
                f"━━━━━━━━━━━━━━━\n"
                f"時機評估: {review.get('timing_assessment', 'N/A')}\n"
                f"出場評估: {review.get('exit_assessment', 'N/A')}\n\n"
            )

            # 分析師表現
            analysts = review.get("analyst_performance", [])
            if analysts:
                text += "分析師表現:\n"
                for a in analysts:
                    icon = "✅" if a.get("was_correct") else "❌"
                    adj = a.get("weight_adjustment", 0)
                    adj_str = f"+{adj:.2f}" if adj >= 0 else f"{adj:.2f}"
                    text += f"  {icon} {a['name']}: {a.get('comment', '')} ({adj_str})\n"
                text += "\n"

            # 經驗教訓
            lessons = review.get("lessons_learned", [])
            if lessons:
                text += "💡 經驗教訓:\n"
                for l in lessons:
                    text += f"  • {l}\n"
                text += "\n"

            text += f"整體評分: {review.get('overall_score', 'N/A')}/10\n"

        await self.bot.send_message(chat_id=self.chat_id, text=text)

    async def send_daily_summary(self, stats: dict):
        """每日總結"""
        if not self.notify_cfg.get("daily_summary", True):
            return

        text = (
            f"{'=' * 30}\n"
            f"📈 每日總結\n"
            f"{'=' * 30}\n\n"
            f"總交易: {stats.get('total', 0)} 筆\n"
            f"勝率: {stats.get('win_rate', 0):.1f}%\n"
            f"今日盈虧: {format_pct(stats.get('today_pnl', 0))}\n"
            f"總盈虧: {format_pct(stats.get('total_profit_pct', 0))}\n"
            f"最大回撤: {format_pct(stats.get('max_drawdown', 0))}\n"
        )
        await self.bot.send_message(chat_id=self.chat_id, text=text)

    async def send_morning_briefing(self, briefing: dict):
        """每日早報（8:00 AM）"""
        date = datetime.now().strftime("%Y-%m-%d")
        strategy = briefing.get("today_strategy", "N/A")
        overview = briefing.get("market_overview", "N/A")
        analyst_summary = briefing.get("analyst_summary", "N/A")
        risk_notes = briefing.get("risk_notes", "N/A")
        confidence = briefing.get("confidence_level", "N/A")

        text = (
            f"{'=' * 30}\n"
            f"🌅 每日早報 | {date}\n"
            f"{'=' * 30}\n\n"
            f"📊 市場概況\n"
            f"━━━━━━━━━━━━━━━\n"
            f"{overview}\n\n"
            f"🗣️ 分析師觀點整理\n"
            f"━━━━━━━━━━━━━━━\n"
            f"{analyst_summary}\n\n"
            f"🎯 今日交易思路\n"
            f"━━━━━━━━━━━━━━━\n"
            f"{strategy}\n\n"
        )

        # 關鍵價位
        key_levels = briefing.get("key_levels", {})
        if key_levels:
            text += "📍 關鍵價位\n━━━━━━━━━━━━━━━\n"
            for symbol, levels in key_levels.items():
                support = levels.get("support", [])
                resistance = levels.get("resistance", [])
                support_str = ", ".join(format_price(s) for s in support) if support else "N/A"
                resist_str = ", ".join(format_price(r) for r in resistance) if resistance else "N/A"
                text += f"  {symbol}:\n    支撐: {support_str}\n    壓力: {resist_str}\n"
            text += "\n"

        # 觀察清單
        watchlist = briefing.get("watchlist", [])
        if watchlist:
            text += "👀 今日觀察\n━━━━━━━━━━━━━━━\n"
            for w in watchlist:
                bias_icon = {"偏多": "🟢", "偏空": "🔴"}.get(w.get("bias", ""), "⚪")
                text += f"  {bias_icon} {w['symbol']}: {w.get('bias', '?')} — {w.get('reason', '')}\n"
            text += "\n"

        text += (
            f"⚠️ 風險提醒\n"
            f"━━━━━━━━━━━━━━━\n"
            f"{risk_notes}\n\n"
            f"信心水平: {confidence}\n"
        )

        await self.bot.send_message(chat_id=self.chat_id, text=text)

    async def send_evening_summary(self, summary: dict, stats: dict):
        """每日晚報（10:00 PM）"""
        date = datetime.now().strftime("%Y-%m-%d")
        day_summary = summary.get("day_summary", "N/A")
        analyst_review = summary.get("analyst_review", "N/A")
        tomorrow = summary.get("tomorrow_outlook", "N/A")
        score = summary.get("overall_score", "N/A")

        text = (
            f"{'=' * 30}\n"
            f"🌙 每日晚報 | {date}\n"
            f"{'=' * 30}\n\n"
            f"📋 今日摘要\n"
            f"━━━━━━━━━━━━━━━\n"
            f"{day_summary}\n\n"
        )

        # 交易回顧
        trades_review = summary.get("trades_review", [])
        if trades_review:
            text += "📊 交易回顧\n━━━━━━━━━━━━━━━\n"
            for t in trades_review:
                text += (
                    f"  #{t.get('trade_id', '?')} {t.get('symbol', '?')} "
                    f"{t.get('direction', '?')}: {t.get('result', 'N/A')}\n"
                    f"    {t.get('comment', '')}\n"
                )
            text += "\n"
        else:
            text += "📊 今日無交易\n\n"

        # 績效數據
        text += (
            f"📈 今日績效\n"
            f"━━━━━━━━━━━━━━━\n"
            f"  交易筆數: {stats.get('total', 0)}\n"
            f"  勝率: {stats.get('win_rate', 0):.1f}%\n"
            f"  今日盈虧: {format_pct(stats.get('today_pnl', 0))}\n"
            f"  總盈虧: {format_pct(stats.get('total_profit_pct', 0))}\n\n"
        )

        text += (
            f"🗣️ 分析師表現\n"
            f"━━━━━━━━━━━━━━━\n"
            f"{analyst_review}\n\n"
        )

        # 經驗教訓
        lessons = summary.get("lessons", [])
        if lessons:
            text += "💡 今日心得\n━━━━━━━━━━━━━━━\n"
            for l in lessons:
                text += f"  • {l}\n"
            text += "\n"

        text += (
            f"🔮 明日展望\n"
            f"━━━━━━━━━━━━━━━\n"
            f"{tomorrow}\n\n"
            f"今日評分: {score}/10\n"
        )

        await self.bot.send_message(chat_id=self.chat_id, text=text)

    async def send_learning_event(self, event: dict):
        """AI 學習事件通知"""
        if not self.notify_cfg.get("notify_on_learning", True):
            return

        text = (
            f"🤖 AI 學習事件\n\n"
            f"類型: {event.get('type', 'N/A')}\n"
            f"內容: {event.get('description', 'N/A')}\n"
        )
        await self.bot.send_message(chat_id=self.chat_id, text=text)

    async def send_rejected_signal(self, decision: dict):
        """被風控拒絕的訊號"""
        text = (
            f"⚠️ 訊號被風控拒絕\n\n"
            f"{decision.get('action', '?')} {decision.get('symbol', '?')}\n"
            f"信心: {decision.get('confidence', 0)}%\n\n"
            f"風控結果:\n{decision.get('_risk_summary', 'N/A')}\n"
        )
        await self.bot.send_message(chat_id=self.chat_id, text=text)

    async def send_error(self, error_msg: str):
        """錯誤通知"""
        text = f"🚨 系統錯誤\n\n{error_msg}"
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=text)
        except Exception:
            logger.error("Failed to send error notification")

    # ── 取消原因 ──

    async def _ask_cancel_reason(self) -> str:
        """取消交易後，詢問用戶原因（60 秒等待）"""
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("方向不對", callback_data="cr_direction"),
                InlineKeyboardButton("信心不足", callback_data="cr_confidence"),
            ],
            [
                InlineKeyboardButton("等待更好時機", callback_data="cr_timing"),
                InlineKeyboardButton("✏️ 自行輸入", callback_data="cr_custom"),
            ],
        ])

        msg = await self.bot.send_message(
            chat_id=self.chat_id,
            text="❌ 交易已取消\n\n請問取消原因：",
            reply_markup=keyboard,
        )

        msg_id = str(msg.message_id)
        reason_event = asyncio.Event()
        self._cancel_reasons[msg_id] = {
            "event": reason_event,
            "reason": "",
            "waiting_text": False,
        }

        try:
            await asyncio.wait_for(reason_event.wait(), timeout=60)
            reason = self._cancel_reasons[msg_id]["reason"]
        except asyncio.TimeoutError:
            reason = "未說明"

        self._cancel_reasons.pop(msg_id, None)

        await self.bot.edit_message_text(
            chat_id=self.chat_id,
            message_id=msg.message_id,
            text=f"❌ 交易已取消\n原因：{reason}",
        )

        logger.info("Cancel reason: %s", reason)
        return reason

    # ── 回調處理 ──

    async def _button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()

        msg_id = str(query.message.message_id)

        # 交易確認按鈕
        if msg_id in self._cancel_callbacks:
            if query.data == "cancel":
                self._cancel_callbacks[msg_id].set()
            elif query.data == "execute_now":
                if msg_id in self._pending_decisions:
                    self._pending_decisions[msg_id]["_execute_now"] = True
                self._cancel_callbacks[msg_id].set()
            return

        # 取消原因按鈕
        if msg_id in self._cancel_reasons:
            preset_reasons = {
                "cr_direction": "方向不對",
                "cr_confidence": "信心不足",
                "cr_timing": "等待更好時機",
            }
            if query.data in preset_reasons:
                self._cancel_reasons[msg_id]["reason"] = preset_reasons[query.data]
                self._cancel_reasons[msg_id]["event"].set()
            elif query.data == "cr_custom":
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text="請輸入您的取消原因：",
                )
                self._cancel_reasons[msg_id]["waiting_text"] = True
            return

    async def _text_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """接收用戶輸入的文字（用於自行輸入取消原因）"""
        for msg_id, data in self._cancel_reasons.items():
            if data.get("waiting_text"):
                data["reason"] = update.message.text
                data["waiting_text"] = False
                data["event"].set()
                return

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        open_count = len(self._db.get_open_trades()) if self._db else 0
        text = (
            "🤖 系統運行中\n\n"
            f"持倉: {open_count} 筆\n"
            "使用 /help 查看所有指令"
        )
        await update.message.reply_text(text)

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🛑 緊急停止指令已接收")
        # 主程式會偵測到這個事件

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = (
            "🤖 AI 交易系統指令\n"
            "━━━━━━━━━━━━━━━\n\n"
            "/status - 系統狀態\n"
            "/positions - 查看當前持倉\n"
            "/pnl - 查看績效總覽\n"
            "/test_trade - 執行測試交易\n"
            "/stop - 緊急停止\n"
            "/help - 顯示此說明\n"
        )
        await update.message.reply_text(text)

    async def _cmd_test_trade(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """測試交易 - 在 Binance Testnet 下一筆小額測試單"""
        if str(update.effective_chat.id) != str(self.chat_id):
            return

        if not self._trader:
            await update.message.reply_text("❌ 交易模組未初始化")
            return

        await update.message.reply_text("🧪 正在執行測試交易...\nLONG BTCUSDT (1% 倉位)")

        try:
            # 取得當前 BTC 價格
            r = requests.get(
                f"{MARKET_DATA_URL}/api/v3/ticker/price",
                params={"symbol": "BTCUSDT"}, timeout=10,
            )
            price = float(r.json()["price"])

            # 建立測試決策（MARKET 單，1% 倉位）
            decision = {
                "action": "LONG",
                "symbol": "BTCUSDT",
                "confidence": 85,
                "entry": {"price": price, "strategy": "MARKET"},
                "stop_loss": round(price * 0.98, 2),
                "take_profit": [round(price * 1.02, 2), round(price * 1.04, 2)],
                "risk_reward": 2.0,
                "position_size": 1.0,
                "reasoning": {
                    "analyst_consensus": "系統測試交易",
                    "technical": "測試流程驗證",
                    "sentiment": "N/A",
                },
                "risk_assessment": {
                    "max_loss_pct": 2.0,
                    "expected_profit_pct": [2.0, 4.0],
                    "win_probability": 0.5,
                },
                "_analyst_messages": [],
            }

            # 執行交易
            trade_result = self._trader.execute_trade(decision)

            if trade_result.get("success"):
                # 記錄到資料庫
                if self._db:
                    self._db.save_ai_decision(
                        decision, outcome="EXECUTED",
                        analyst_names=["TEST"],
                        trade_id=trade_result["trade_id"],
                    )

                # 發送進場通知
                await self.send_entry_confirmation(trade_result)

                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=(
                        f"✅ 測試交易成功！\n\n"
                        f"交易 #{trade_result['trade_id']}\n"
                        f"LONG BTCUSDT @ {format_price(price)}\n"
                        f"數量: {trade_result['quantity']}\n\n"
                        f"使用 /positions 查看持倉\n"
                        f"使用 /pnl 查看績效"
                    ),
                )
            else:
                await update.message.reply_text(
                    f"❌ 測試交易失敗:\n{trade_result.get('error', 'Unknown')}"
                )

        except Exception as e:
            logger.exception("Test trade error")
            await update.message.reply_text(f"❌ 測試交易錯誤: {e}")

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """查看當前持倉"""
        if str(update.effective_chat.id) != str(self.chat_id):
            return

        if not self._db:
            await update.message.reply_text("❌ 資料庫未初始化")
            return

        open_trades = self._db.get_open_trades()

        if not open_trades:
            await update.message.reply_text("📊 目前沒有持倉")
            return

        text = f"📊 當前持倉 ({len(open_trades)} 筆)\n{'=' * 25}\n\n"

        for t in open_trades:
            # 取得當前價格計算未實現盈虧
            try:
                r = requests.get(
                    f"{MARKET_DATA_URL}/api/v3/ticker/price",
                    params={"symbol": t.symbol}, timeout=10,
                )
                current_price = float(r.json()["price"])
                leverage = t.leverage or 1

                if t.direction == "LONG":
                    pnl_pct = (current_price - t.entry_price) / t.entry_price * 100 * leverage
                else:
                    pnl_pct = (t.entry_price - current_price) / t.entry_price * 100 * leverage

                pnl_icon = "🟢" if pnl_pct >= 0 else "🔴"
            except Exception:
                current_price = 0
                pnl_pct = 0
                pnl_icon = "⚪"

            direction_icon = "🟢" if t.direction == "LONG" else "🔴"
            tp_list = json.loads(t.take_profit) if isinstance(t.take_profit, str) and t.take_profit else []

            text += (
                f"{direction_icon} #{t.id} | {t.direction} {t.symbol}\n"
                f"  槓桿: {t.leverage}x\n"
                f"  進場: {format_price(t.entry_price)}\n"
                f"  現價: {format_price(current_price)}\n"
                f"  {pnl_icon} 未實現: {pnl_pct:+.2f}%\n"
                f"  停損: {format_price(t.stop_loss)}\n"
                f"  目標: {', '.join(format_price(p) for p in tp_list) if tp_list else 'N/A'}\n"
                f"  倉位: {t.position_size}%\n"
                f"━━━━━━━━━━━━━━━\n"
            )

        await update.message.reply_text(text)

    async def _cmd_pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """查看績效總覽"""
        if str(update.effective_chat.id) != str(self.chat_id):
            return

        if not self._db:
            await update.message.reply_text("❌ 資料庫未初始化")
            return

        stats = self._db.get_performance_stats()
        today_pnl = self._db.get_today_pnl()
        today_trades = self._db.get_today_trades()
        open_trades = self._db.get_open_trades()

        text = (
            f"{'=' * 25}\n"
            f"📈 績效總覽\n"
            f"{'=' * 25}\n\n"
            f"📊 總績效\n"
            f"━━━━━━━━━━━━━━━\n"
            f"總交易: {stats['total']} 筆\n"
            f"勝: {stats['wins']} | 負: {stats['losses']}\n"
            f"勝率: {stats['win_rate']:.1f}%\n"
            f"總盈虧: {format_pct(stats['total_profit_pct'])}\n"
            f"平均盈虧: {format_pct(stats['avg_profit_pct'])}\n"
            f"最大回撤: {stats['max_drawdown']:.2f}%\n\n"
            f"📅 今日\n"
            f"━━━━━━━━━━━━━━━\n"
            f"今日交易: {len(today_trades)} 筆\n"
            f"今日盈虧: {format_pct(today_pnl)}\n\n"
            f"📦 持倉: {len(open_trades)} 筆\n"
        )

        if open_trades:
            text += "\n使用 /positions 查看持倉詳情\n"

        await update.message.reply_text(text)

    # ── 工具方法 ──

    @staticmethod
    def _format_duration(seconds: int) -> str:
        if seconds < 60:
            return f"{seconds}秒"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes}分鐘"
        hours = minutes // 60
        mins = minutes % 60
        if hours < 24:
            return f"{hours}小時 {mins}分鐘"
        days = hours // 24
        hrs = hours % 24
        return f"{days}天 {hrs}小時"
