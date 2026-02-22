import asyncio
import json
import logging
from datetime import datetime, timezone

import httpx
import requests
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler, ContextTypes,
    MessageHandler, filters,
)
from telegram.request import HTTPXRequest

from utils.helpers import format_price, format_pct

logger = logging.getLogger(__name__)

FUTURES_URL = "https://fapi.binance.com"


class TelegramNotifier:
    def __init__(self, config: dict, db=None, trader=None):
        self.config = config
        self._db = db
        self._trader = trader
        tg_cfg = config.get("telegram", {})
        self.bot_token = tg_cfg.get("bot_token", "")
        self.chat_id = tg_cfg.get("chat_id", "")
        self.notify_cfg = config.get("notifications", {})

        # 獨立的 sender bot（有明確的 HTTP timeout，不受 polling 連線影響）
        self.bot = Bot(
            token=self.bot_token,
            request=HTTPXRequest(
                connection_pool_size=8,
                connect_timeout=10.0,
                read_timeout=15.0,
                write_timeout=10.0,
                pool_timeout=5.0,
            ),
        )
        self._app: Application | None = None
        self._pending_decisions: dict[str, dict] = {}  # msg_id -> decision
        self._cancel_callbacks: dict[str, asyncio.Event] = {}
        self._cancel_reasons: dict[str, dict] = {}  # msg_id -> {event, reason, waiting_text}
        self._briefing_callback = None  # main.py 設定
        self._review_callback = None    # main.py 設定
        self._signal_callback = None    # main.py 設定（手動跟單）
        self._replay_callback = None    # main.py 設定（重播歷史訊號）

        logger.info("TelegramNotifier initialized")

    async def start(self):
        """啟動 Telegram Bot（持續輪詢，隨時接收指令）"""
        self._app = (
            Application.builder()
            .token(self.bot_token)
            .request(HTTPXRequest(connection_pool_size=20, pool_timeout=10.0))
            .build()
        )
        self._app.add_handler(CallbackQueryHandler(self._button_callback))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("stop", self._cmd_stop))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(CommandHandler("test_trade", self._cmd_test_trade))
        self._app.add_handler(CommandHandler("test_signal", self._cmd_test_signal))
        self._app.add_handler(CommandHandler("follow", self._cmd_follow))
        self._app.add_handler(CommandHandler("replay", self._cmd_replay))
        self._app.add_handler(CommandHandler("positions", self._cmd_positions))
        self._app.add_handler(CommandHandler("pnl", self._cmd_pnl))
        self._app.add_handler(CommandHandler("close", self._cmd_close))
        self._app.add_handler(CommandHandler("close_all", self._cmd_close_all))
        self._app.add_handler(CommandHandler("fix_tp", self._cmd_fix_tp))
        self._app.add_handler(CommandHandler("orders", self._cmd_orders))
        self._app.add_handler(CommandHandler("cancel_orders", self._cmd_cancel_orders))
        self._app.add_handler(CommandHandler("briefing", self._cmd_briefing))
        self._app.add_handler(CommandHandler("reset_trades", self._cmd_reset_trades))
        self._app.add_handler(CommandHandler("review", self._cmd_review))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._text_handler)
        )

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

        # self.bot 保持獨立（有明確 timeout），不替換為 Application 的 bot
        # Application 的 bot 只用於 polling / callback 內部

        logger.info("Telegram bot started with persistent polling")

    async def stop(self):
        if self._app:
            if self._app.updater and self._app.updater.running:
                await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

    # ── 工具方法 ──

    async def _safe_send(self, text: str) -> None:
        """直接用 httpx 非同步發送訊息（DNS timeout 也有效）"""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                await client.post(
                    f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                    json={"chat_id": self.chat_id, "text": text},
                )
        except Exception as e:
            logger.warning("_safe_send failed: %s", e)

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
        is_scanner = decision.get("_scanner_triggered", False)
        is_paper = self.config.get("trading", {}).get("mode") == "paper"
        paper_tag = " [模擬]" if is_paper else ""
        source_label = ("🔍 掃描器主動發現" if is_scanner else "🔔 交易訊號") + paper_tag

        # 計算預估手續費（使用 AI 決定的槓桿）
        fee_cost = 0
        if self._trader:
            trading_cfg = self.config.get("trading", {})
            leverage_map = trading_cfg.get("leverage_map", {})
            default_lev = trading_cfg.get("default_leverage", 25)
            max_lev = leverage_map.get(symbol, default_lev)
            lev = min(int(decision.get("leverage", max_lev)), max_lev)
            fee_cost = self._trader.calc_fee_pct(lev)

        # 掃描器觸發原因
        scanner_trigger = reasoning.get("scanner_trigger", "")
        scanner_line = f"觸發: {scanner_trigger}\n" if scanner_trigger else ""

        text = (
            f"{'=' * 30}\n"
            f"{source_label}\n"
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
            f"槓桿: {lev}x\n"
            f"風報比: {rr:.2f}\n"
            f"預估手續費: -{fee_cost:.2f}%\n\n"
            f"🤖 AI 分析\n"
            f"━━━━━━━━━━━━━━━\n"
            f"共識: {reasoning.get('analyst_consensus', 'N/A')}\n"
            f"技術: {reasoning.get('technical', 'N/A')}\n"
            f"情緒: {reasoning.get('sentiment', 'N/A')}\n"
            f"{scanner_line}\n"
            f"📈 風險評估\n"
            f"━━━━━━━━━━━━━━━\n"
            f"最大虧損: {risk.get('max_loss_pct', 0):.2f}%\n"
            f"預期獲利: {risk.get('expected_profit_pct', [0])[0]:.2f}%\n"
            f"手續費成本: {risk.get('fee_cost_pct', fee_cost):.2f}%\n"
            f"勝率: {risk.get('win_probability', 0) * 100:.0f}%\n\n"
            f"⏱️ {countdown} 秒後自動執行...\n"
        )

        logger.info("Sending trade signal to Telegram (chat_id=%s)...", self.chat_id)

        keyboard_dict = {
            "inline_keyboard": [[
                {"text": "❌ 取消", "callback_data": "cancel"},
                {"text": "⚡ 立即執行", "callback_data": "execute_now"},
            ]]
        }

        # httpx.AsyncClient: timeout 覆蓋 DNS + connect + read，根本解決 DNS 卡住問題
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                    json={"chat_id": self.chat_id, "text": text, "reply_markup": keyboard_dict},
                )
                resp.raise_for_status()
                msg_message_id = resp.json()["result"]["message_id"]
        except Exception as e:
            logger.error("Telegram sendMessage failed: %s — executing trade directly", e)
            return {"executed": True, "cancelled": False}

        logger.info("Signal sent to Telegram (msg_id=%s)", msg_message_id)

        msg_id = str(msg_message_id)
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

        logger.info("Countdown done (msg_id=%s, execute_now=%s, cancelled=%s)",
                    msg_id, execute_now, cancelled)

        if cancelled:
            cancel_reason = await self._ask_cancel_reason()

        # 清理
        self._pending_decisions.pop(msg_id, None)
        self._cancel_callbacks.pop(msg_id, None)

        async def _edit_msg(new_text: str) -> None:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.post(
                        f"https://api.telegram.org/bot{self.bot_token}/editMessageText",
                        json={
                            "chat_id": self.chat_id,
                            "message_id": msg_message_id,
                            "text": new_text,
                        },
                    )
            except Exception:
                pass

        if cancelled:
            cancelled_text = text.replace(
                f"⏱️ {countdown} 秒後自動執行...",
                f"❌ 已取消\n原因：{cancel_reason}"
            )
            await _edit_msg(cancelled_text)
            return {"executed": False, "cancelled": True, "cancel_reason": cancel_reason}

        status_text = "⚡ 立即執行中..." if execute_now else "✅ 倒數結束，執行中..."
        executing_text = text.replace(f"⏱️ {countdown} 秒後自動執行...", status_text)
        logger.info("Editing message to execution status (msg_id=%s)...", msg_id)
        await _edit_msg(executing_text)
        logger.info("Message edited OK, send_signal returning (msg_id=%s)", msg_id)

        return {"executed": True, "cancelled": False}

    async def send_entry_confirmation(self, trade_result: dict):
        """進場確認通知"""
        if not self.notify_cfg.get("notify_on_entry", True):
            return

        is_paper = self.config.get("trading", {}).get("mode") == "paper"
        paper_tag = " [模擬]" if is_paper else ""
        text = (
            f"✅ 已進場{paper_tag}\n\n"
            f"交易 #{trade_result['trade_id']}\n"
            f"{trade_result['direction']} {trade_result['symbol']}\n"
            f"進場價: {format_price(trade_result['entry_price'])}\n"
            f"數量: {trade_result['quantity']}\n"
            f"停損: {format_price(trade_result['stop_loss'])}\n"
            f"目標: {', '.join(format_price(t) for t in trade_result['take_profit'])}\n\n"
            f"📊 持倉監控中..."
        )
        await self._safe_send(text)

    async def send_pending_order(self, trade_result: dict):
        """LIMIT 掛單通知（等待成交）"""
        is_paper = self.config.get("trading", {}).get("mode") == "paper"
        paper_tag = " [模擬]" if is_paper else ""
        text = (
            f"📋 掛單已送出{paper_tag}\n\n"
            f"交易 #{trade_result['trade_id']}\n"
            f"{trade_result['direction']} {trade_result['symbol']}\n"
            f"掛單價: {format_price(trade_result['entry_price'])}\n"
            f"數量: {trade_result['quantity']}\n"
            f"停損: {format_price(trade_result['stop_loss'])}\n"
            f"目標: {', '.join(format_price(t) for t in trade_result['take_profit'])}\n\n"
            f"⏳ LIMIT 限價單，等待市場價觸及掛單價..."
        )
        await self._safe_send(text)

    async def send_position_update(self, trade, current_price: float, unrealized_pct: float):
        """持倉更新（可選，避免太頻繁）"""
        pass  # 只在重要變化時發送

    async def send_exit_notification(self, trade, result: dict, review: dict | None = None):
        """平倉通知 + AI 覆盤"""
        if not self.notify_cfg.get("notify_on_exit", True):
            return

        outcome_icon = "✅" if result.get("outcome") == "WIN" else "❌"
        profit = result.get("profit_pct", 0)
        fee = result.get("fee_pct", 0)
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
            f"獲利: {format_pct(profit)} (手續費: -{fee:.2f}%)\n"
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

        await self._safe_send(text)

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
        await self._safe_send(text)

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

        await self._safe_send(text)

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

        await self._safe_send(text)

    async def send_learning_event(self, event: dict):
        """AI 學習事件通知"""
        if not self.notify_cfg.get("notify_on_learning", True):
            return

        text = (
            f"🤖 AI 學習事件\n\n"
            f"類型: {event.get('type', 'N/A')}\n"
            f"內容: {event.get('description', 'N/A')}\n"
        )
        await self._safe_send(text)

    async def send_rejected_signal(self, decision: dict):
        """被風控拒絕的訊號"""
        text = (
            f"⚠️ 訊號被風控拒絕\n\n"
            f"{decision.get('action', '?')} {decision.get('symbol', '?')}\n"
            f"信心: {decision.get('confidence', 0)}%\n\n"
            f"風控結果:\n{decision.get('_risk_summary', 'N/A')}\n"
        )
        await self._safe_send(text)

    async def send_error(self, error_msg: str):
        """錯誤通知"""
        text = f"🚨 系統錯誤\n\n{error_msg}"
        await self._safe_send(text)

    # ── 取消原因 ──

    async def _ask_cancel_reason(self) -> str:
        """取消交易後，詢問用戶原因（60 秒等待）"""
        keyboard_dict = {
            "inline_keyboard": [
                [
                    {"text": "方向不對", "callback_data": "cr_direction"},
                    {"text": "信心不足", "callback_data": "cr_confidence"},
                ],
                [
                    {"text": "等待更好時機", "callback_data": "cr_timing"},
                    {"text": "✏️ 自行輸入", "callback_data": "cr_custom"},
                ],
            ]
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": "❌ 交易已取消\n\n請問取消原因：",
                        "reply_markup": keyboard_dict,
                    },
                )
                cr_msg_id = str(resp.json()["result"]["message_id"])
        except Exception as e:
            logger.warning("_ask_cancel_reason send failed: %s", e)
            return "未說明"

        reason_event = asyncio.Event()
        self._cancel_reasons[cr_msg_id] = {
            "event": reason_event,
            "reason": "",
            "waiting_text": False,
        }

        try:
            await asyncio.wait_for(reason_event.wait(), timeout=60)
            reason = self._cancel_reasons[cr_msg_id]["reason"]
        except asyncio.TimeoutError:
            reason = "未說明"

        self._cancel_reasons.pop(cr_msg_id, None)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"https://api.telegram.org/bot{self.bot_token}/editMessageText",
                    json={
                        "chat_id": self.chat_id,
                        "message_id": int(cr_msg_id),
                        "text": f"❌ 交易已取消\n原因：{reason}",
                    },
                )
        except Exception:
            pass

        logger.info("Cancel reason: %s", reason)
        return reason

    # ── 回調處理 ──

    async def _button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        msg_id = str(query.message.message_id)

        # 持倉刷新按鈕（不在交易流程中，保持原邏輯）
        if query.data == "refresh_positions":
            try:
                await query.answer("刷新中...")
            except Exception:
                pass  # query 可能已過期，忽略
            try:
                text = self._build_positions_text()
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔄 刷新", callback_data="refresh_positions")]
                ])
                await query.edit_message_text(text=text, reply_markup=keyboard)
            except Exception as e:
                logger.warning("Failed to refresh positions: %s", e)
            return

        # ── STEP 1: 立即設定 event（純 Python，無 I/O，不能被 httpx 阻塞）──
        # 必須在 query.answer() 之前，否則 httpx 卡住會讓 event 永遠不被設定
        if msg_id in self._cancel_callbacks:
            if query.data == "cancel":
                self._cancel_callbacks[msg_id].set()
            elif query.data == "execute_now":
                if msg_id in self._pending_decisions:
                    self._pending_decisions[msg_id]["_execute_now"] = True
                self._cancel_callbacks[msg_id].set()

        # 取消原因按鈕（event 設定也在 I/O 之前）
        if msg_id in self._cancel_reasons:
            preset_reasons = {
                "cr_direction": "方向不對",
                "cr_confidence": "信心不足",
                "cr_timing": "等待更好時機",
            }
            if query.data in preset_reasons:
                self._cancel_reasons[msg_id]["reason"] = preset_reasons[query.data]
                self._cancel_reasons[msg_id]["event"].set()

        # ── STEP 2: 回應 TG callback query（fire-and-forget，不 await）──
        async def _answer() -> None:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.post(
                        f"https://api.telegram.org/bot{self.bot_token}/answerCallbackQuery",
                        json={"callback_query_id": query.id},
                    )
            except Exception:
                pass
        asyncio.create_task(_answer())

        # cr_custom 需要等用戶輸入，在 event 設定後才處理
        if msg_id in self._cancel_reasons and query.data == "cr_custom":
            await self._safe_send("請輸入您的取消原因：")
            self._cancel_reasons[msg_id]["waiting_text"] = True

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
            "/close <id> - 平倉指定交易\n"
            "/close_all - 平掉所有持倉\n"
            "/fix_tp [id] - 重設止盈止損掛單\n"
            "/orders [symbol] - 查看 Binance 訂單歷史\n"
            "/cancel_orders <symbol> - 取消殘留掛單\n"
            "/briefing - 手動觸發早報\n"
            "/reset_trades confirm - 清除所有交易資料\n"
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
                f"{FUTURES_URL}/fapi/v1/ticker/price",
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

                await self._safe_send(
                    f"✅ 測試交易成功！\n\n"
                    f"交易 #{trade_result['trade_id']}\n"
                    f"LONG BTCUSDT @ {format_price(price)}\n"
                    f"數量: {trade_result['quantity']}\n\n"
                    f"使用 /positions 查看持倉\n"
                    f"使用 /pnl 查看績效"
                )
            else:
                await update.message.reply_text(
                    f"❌ 測試交易失敗:\n{trade_result.get('error', 'Unknown')}"
                )

        except Exception as e:
            logger.exception("Test trade error")
            await update.message.reply_text(f"❌ 測試交易錯誤: {e}")

    async def _cmd_test_signal(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """測試完整流程：send_signal 倒數 → execute_trade → 進場確認"""
        if str(update.effective_chat.id) != str(self.chat_id):
            return

        if not self._trader:
            await update.message.reply_text("❌ 交易模組未初始化")
            return

        await update.message.reply_text("🧪 開始完整流程測試（10 秒倒數）...")

        try:
            r = requests.get(
                f"{FUTURES_URL}/fapi/v1/ticker/price",
                params={"symbol": "BTCUSDT"}, timeout=10,
            )
            price = float(r.json()["price"])

            decision = {
                "action": "LONG",
                "symbol": "BTCUSDT",
                "confidence": 85,
                "entry": {"price": price, "strategy": "MARKET"},
                "stop_loss": round(price * 0.98, 2),
                "take_profit": [round(price * 1.02, 2), round(price * 1.04, 2)],
                "leverage": 100,
                "risk_reward": 2.0,
                "position_size": 1.0,
                "reasoning": {
                    "analyst_consensus": "【測試訊號】完整流程驗證",
                    "technical": "N/A",
                    "sentiment": "N/A",
                },
                "risk_assessment": {
                    "max_loss_pct": 2.0,
                    "expected_profit_pct": [2.0, 4.0],
                    "fee_cost_pct": 0.1,
                    "win_probability": 0.5,
                },
                "_analyst_messages": [],
            }

            # 走完整 send_signal 流程（倒數 10 秒，可取消）
            result = await self.send_signal(decision, countdown=10)

            if result.get("cancelled"):
                await update.message.reply_text("❌ 測試訊號已取消")
                return

            # 執行交易
            trade_result = self._trader.execute_trade(decision)
            if trade_result.get("success"):
                if self._db:
                    self._db.save_ai_decision(
                        decision, outcome="EXECUTED",
                        analyst_names=["TEST"],
                        trade_id=trade_result["trade_id"],
                    )
                await self.send_entry_confirmation(trade_result)
                await update.message.reply_text(
                    f"✅ 完整流程測試通過！\n"
                    f"交易 #{trade_result['trade_id']} 已建立\n"
                    f"使用 /positions 查看持倉"
                )
            else:
                await update.message.reply_text(
                    f"❌ 執行失敗: {trade_result.get('error', 'Unknown')}"
                )

        except Exception as e:
            logger.exception("Test signal error")
            await update.message.reply_text(f"❌ 測試流程錯誤: {e}")

    async def _cmd_follow(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """手動跟單指令：直接輸入分析師點位執行交易

        格式：/follow {LONG|SHORT} {BTC|ETH} {entry1[,entry2]} {sl} {tp1[,tp2,tp3]}

        範例（空單，兩個進場點，三個止盈）：
          /follow SHORT BTC 67800,68600 69300 67000,66300,65600

        範例（多單，單一進場點，兩個止盈）：
          /follow LONG BTC 95000 93000 97000,99000
        """
        if str(update.effective_chat.id) != str(self.chat_id):
            return

        if not self._signal_callback:
            await update.message.reply_text("❌ 訊號回調未初始化")
            return

        args = context.args
        if len(args) < 5:
            await update.message.reply_text(
                "❌ 格式錯誤\n\n"
                "正確格式：\n"
                "/follow {LONG|SHORT} {BTC|ETH} {進場點} {止損} {止盈點}\n\n"
                "範例：\n"
                "/follow SHORT BTC 67800,68600 69300 67000,66300,65600\n"
                "/follow LONG BTC 95000 93000 97000,99000"
            )
            return

        try:
            action = args[0].upper()
            if action not in ("LONG", "SHORT"):
                await update.message.reply_text("❌ 方向必須是 LONG 或 SHORT")
                return

            # 幣種解析
            sym_raw = args[1].upper()
            symbol_map = {
                "BTC": "BTCUSDT", "BTCUSDT": "BTCUSDT",
                "ETH": "ETHUSDT", "ETHUSDT": "ETHUSDT",
            }
            symbol = symbol_map.get(sym_raw)
            if not symbol:
                await update.message.reply_text(f"❌ 不支援幣種：{sym_raw}（只支援 BTC / ETH）")
                return

            # 進場點（逗號分隔，可多個）
            entries_raw = [float(x) for x in args[2].split(",") if x]
            entry_1_price = entries_raw[0]
            entry_2_price = entries_raw[1] if len(entries_raw) > 1 else None

            # 止損
            stop_loss = float(args[3])

            # 止盈點（逗號分隔，可多個）
            tp_raw = args[4].split(",") if "," in args[4] else args[4:]
            take_profit = [float(x) for x in tp_raw if x]

            if not take_profit:
                await update.message.reply_text("❌ 至少需要一個止盈點")
                return

        except (ValueError, IndexError) as e:
            await update.message.reply_text(f"❌ 解析失敗：{e}\n請確認所有數字格式正確")
            return

        # 從 config 取倉位和槓桿
        trading_cfg = self.config.get("trading", {})
        follow_pos = trading_cfg.get("follow_position_size", 5.0)
        channels = self.config.get("discord", {}).get("monitored_channels", [])
        leverage = channels[0].get("leverage", 100) if channels else 100

        has_entry_2 = entry_2_price is not None
        pos_size = round(follow_pos / 2, 1) if has_entry_2 else follow_pos

        decision = {
            "action": action,
            "symbol": symbol,
            "confidence": 90,
            "leverage": leverage,
            "entry": {"price": entry_1_price, "strategy": "LIMIT"},
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": pos_size,
            "risk_reward": 2.0,
            "reasoning": {
                "analyst_consensus": "手動跟單",
                "technical": "用戶手動輸入點位",
                "sentiment": "N/A",
            },
            "risk_assessment": {},
            "_follow_mode": True,
            "_analyst_messages": [],
            "_entry_2": {"price": entry_2_price, "strategy": "LIMIT"} if has_entry_2 else None,
        }

        preview = (
            f"📋 手動跟單確認\n\n"
            f"{'🟢 LONG (做多)' if action == 'LONG' else '🔴 SHORT (做空)'}\n"
            f"交易對: {symbol}\n"
            f"進場 1: {entry_1_price} (LIMIT)\n"
        )
        if has_entry_2:
            preview += f"進場 2: {entry_2_price} (LIMIT)\n"
        preview += (
            f"止損: {stop_loss}\n"
            f"止盈: {', '.join(str(t) for t in take_profit)}\n"
            f"倉位: {pos_size}% × {leverage}x\n"
        )
        await update.message.reply_text(preview)

        await self._signal_callback(decision)

    async def _cmd_replay(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """重播過去 N 小時的分析師訊號，重新跑 AI 解析 + 執行交易

        格式：/replay [小時數]
        範例：/replay 2   （預設 2 小時）
        """
        if str(update.effective_chat.id) != str(self.chat_id):
            return

        if not self._replay_callback:
            await update.message.reply_text("❌ replay 回調未初始化")
            return

        hours = 2.0
        if context.args:
            try:
                hours = float(context.args[0])
            except ValueError:
                await update.message.reply_text("❌ 格式：/replay [小時數]，例如 /replay 2")
                return

        await update.message.reply_text(
            f"🔄 掃描過去 {hours} 小時的分析師訊息，重新解析並執行...\n"
            f"（跳過 60 秒收集窗口，直接送 AI 分析）"
        )
        await self._replay_callback(hours)

    def _build_positions_text(self) -> str:
        """產生持倉資訊文字（供 /positions 和刷新按鈕共用）"""
        is_paper = self.config.get("trading", {}).get("mode") == "paper"
        balance_label = "💰 虛擬帳戶 [模擬]" if is_paper else "💰 帳戶資訊"
        balance_text = ""
        if self._trader:
            try:
                account = self._trader._futures_get("/fapi/v2/account", signed=True)
                wallet = float(account.get("totalWalletBalance", 0))
                unrealized = float(account.get("totalUnrealizedProfit", 0))
                margin = float(account.get("totalMarginBalance", 0))
                available = float(account.get("availableBalance", 0))
                balance_text = (
                    f"{balance_label}\n"
                    f"━━━━━━━━━━━━━━━\n"
                    f"錢包餘額: {wallet:,.2f} USDT\n"
                    f"未實現盈虧: {unrealized:+,.2f} USDT\n"
                    f"保證金餘額: {margin:,.2f} USDT\n"
                    f"可用餘額: {available:,.2f} USDT\n\n"
                )
            except Exception as e:
                balance_text = f"{balance_label}: 查詢失敗 ({e})\n\n"

        open_trades = self._db.get_open_trades() if self._db else []

        if not open_trades:
            return f"{balance_text}📊 目前沒有持倉"

        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        text = f"{balance_text}📊 當前持倉 ({len(open_trades)} 筆)\n{'=' * 25}\n\n"

        for t in open_trades:
            try:
                r = requests.get(
                    f"{FUTURES_URL}/fapi/v1/ticker/price",
                    params={"symbol": t.symbol}, timeout=10,
                )
                current_price = float(r.json()["price"])
                leverage = t.leverage or 1

                if t.direction == "LONG":
                    pnl_pct = (current_price - t.entry_price) / t.entry_price * 100 * leverage
                else:
                    pnl_pct = (t.entry_price - current_price) / t.entry_price * 100 * leverage

                fee_pct = 0
                if self._trader:
                    fee_pct = self._trader.calc_fee_pct(leverage)
                    pnl_pct -= fee_pct

                pnl_icon = "🟢" if pnl_pct >= 0 else "🔴"
            except Exception:
                current_price = 0
                pnl_pct = 0
                fee_pct = 0
                pnl_icon = "⚪"

            direction_icon = "🟢" if t.direction == "LONG" else "🔴"
            tp_list = json.loads(t.take_profit) if isinstance(t.take_profit, str) and t.take_profit else []

            text += (
                f"{direction_icon} #{t.id} | {t.direction} {t.symbol}\n"
                f"  槓桿: {t.leverage}x\n"
                f"  進場: {format_price(t.entry_price)}\n"
                f"  現價: {format_price(current_price)}\n"
                f"  {pnl_icon} 未實現: {pnl_pct:+.2f}% (手續費: -{fee_pct:.2f}%)\n"
                f"  停損: {format_price(t.stop_loss)}\n"
                f"  目標: {', '.join(format_price(p) for p in tp_list) if tp_list else 'N/A'}\n"
                f"  倉位: {t.position_size}%\n"
                f"━━━━━━━━━━━━━━━\n"
            )

        text += f"\n更新時間: {now_str}"
        return text

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """查看當前持倉"""
        if str(update.effective_chat.id) != str(self.chat_id):
            return

        if not self._db:
            await update.message.reply_text("❌ 資料庫未初始化")
            return

        text = self._build_positions_text()
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 刷新", callback_data="refresh_positions")]
        ])
        await update.message.reply_text(text, reply_markup=keyboard)

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

    async def _cmd_close(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """平倉指定交易: /close <trade_id>"""
        if str(update.effective_chat.id) != str(self.chat_id):
            return

        if not self._trader or not self._db:
            await update.message.reply_text("❌ 模組未初始化")
            return

        if not context.args:
            await update.message.reply_text("用法: /close <交易ID>\n例如: /close 1")
            return

        try:
            trade_id = int(context.args[0])
        except ValueError:
            await update.message.reply_text("❌ 交易 ID 必須是數字")
            return

        trade = self._db.get_trade(trade_id)
        if not trade:
            await update.message.reply_text(f"❌ 找不到交易 #{trade_id}")
            return
        if trade.status == "CLOSED":
            await update.message.reply_text(f"❌ 交易 #{trade_id} 已經平倉了")
            return

        await update.message.reply_text(f"⏳ 正在平倉 #{trade_id} {trade.direction} {trade.symbol}...")

        result = self._trader.close_trade(trade_id)

        if result.get("success"):
            pnl = result.get("profit_pct", 0)
            fee = result.get("fee_pct", 0)
            pnl_icon = "🟢" if pnl >= 0 else "🔴"
            await update.message.reply_text(
                f"✅ 交易 #{trade_id} 已平倉\n\n"
                f"{trade.direction} {trade.symbol}\n"
                f"進場: {format_price(trade.entry_price)}\n"
                f"出場: {format_price(result.get('exit_price', 0))}\n"
                f"{pnl_icon} 盈虧: {pnl:+.2f}% (手續費: -{fee:.2f}%)\n"
                f"結果: {result.get('outcome', 'N/A')}"
            )
        else:
            await update.message.reply_text(
                f"❌ 平倉失敗: {result.get('error', 'Unknown')}"
            )

    async def _cmd_close_all(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """平掉所有持倉"""
        if str(update.effective_chat.id) != str(self.chat_id):
            return

        if not self._trader or not self._db:
            await update.message.reply_text("❌ 模組未初始化")
            return

        open_trades = self._db.get_open_trades()

        if not open_trades:
            await update.message.reply_text("📊 目前沒有持倉可平")
            return

        await update.message.reply_text(
            f"⏳ 正在平倉所有持倉 ({len(open_trades)} 筆)..."
        )

        results = []
        for t in open_trades:
            result = self._trader.close_trade(t.id)
            if result.get("success"):
                pnl = result.get("profit_pct", 0)
                fee = result.get("fee_pct", 0)
                pnl_icon = "🟢" if pnl >= 0 else "🔴"
                results.append(
                    f"{pnl_icon} #{t.id} {t.direction} {t.symbol}: {pnl:+.2f}% (費: -{fee:.2f}%)"
                )
            else:
                results.append(
                    f"❌ #{t.id} {t.symbol}: {result.get('error', 'Failed')}"
                )

        text = "✅ 全部平倉完成\n\n" + "\n".join(results)
        await update.message.reply_text(text)

    async def _cmd_fix_tp(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """重設止盈止損掛單: /fix_tp [trade_id] (不填=全部持倉)"""
        if str(update.effective_chat.id) != str(self.chat_id):
            return

        if not self._trader or not self._db:
            await update.message.reply_text("❌ 模組未初始化")
            return

        # 決定要修復哪些交易
        if context.args:
            try:
                trade_id = int(context.args[0])
            except ValueError:
                await update.message.reply_text("用法: /fix_tp [交易ID]\n不填 ID 則修復所有持倉")
                return

            trade = self._db.get_trade(trade_id)
            if not trade:
                await update.message.reply_text(f"❌ 找不到交易 #{trade_id}")
                return
            if trade.status == "CLOSED":
                await update.message.reply_text(f"❌ 交易 #{trade_id} 已平倉")
                return
            trades_to_fix = [trade]
        else:
            trades_to_fix = self._db.get_open_trades()

        if not trades_to_fix:
            await update.message.reply_text("📊 目前沒有持倉需要修復")
            return

        await update.message.reply_text(
            f"🔧 正在重設 {len(trades_to_fix)} 筆交易的止盈止損..."
        )

        results = []
        for t in trades_to_fix:
            tp_list = json.loads(t.take_profit) if isinstance(t.take_profit, str) and t.take_profit else []
            result = self._trader.resync_sl_tp(t.id)
            if result.get("success"):
                tp_str = ", ".join(format_price(p) for p in tp_list) if tp_list else "N/A"
                results.append(
                    f"✅ #{t.id} {t.direction} {t.symbol}\n"
                    f"   SL: {format_price(t.stop_loss)}\n"
                    f"   TP: {tp_str}\n"
                    f"   數量: {result.get('quantity', '?')}"
                )
            else:
                results.append(
                    f"❌ #{t.id} {t.symbol}: {result.get('error', 'Failed')}"
                )

        text = "🔧 止盈止損重設完成\n\n" + "\n\n".join(results)
        await update.message.reply_text(text)

    async def _cmd_orders(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """查看 Binance 訂單歷史: /orders [symbol]"""
        if str(update.effective_chat.id) != str(self.chat_id):
            return

        if not self._trader:
            await update.message.reply_text("❌ 交易模組未初始化")
            return

        symbol = "BTCUSDT"
        if context.args:
            symbol = context.args[0].upper()
            if not symbol.endswith("USDT"):
                symbol += "USDT"

        await update.message.reply_text(f"🔍 查詢 {symbol} 訂單歷史...")

        orders = self._trader.get_recent_orders(symbol, limit=15)
        if not orders:
            await update.message.reply_text(f"❌ 沒有找到 {symbol} 的訂單紀錄")
            return

        lines = [f"📋 {symbol} 最近訂單\n"]
        for o in orders:
            status_icon = {"FILLED": "✅", "CANCELED": "🚫", "NEW": "⏳", "EXPIRED": "⏰"}.get(
                o["status"], "❓"
            )
            stop_info = f" @{o['stopPrice']}" if o.get("stopPrice") and o["stopPrice"] != "0" else ""
            lines.append(
                f"{status_icon} {o['type']} {o['side']}\n"
                f"   價格: {o['price']}{stop_info}\n"
                f"   數量: {o['qty']} | {o['time']}"
            )

        await update.message.reply_text("\n".join(lines))

    async def _cmd_cancel_orders(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """取消指定交易對的所有掛單: /cancel_orders <symbol>"""
        if str(update.effective_chat.id) != str(self.chat_id):
            return

        if not self._trader:
            await update.message.reply_text("❌ 交易模組未初始化")
            return

        if not context.args:
            await update.message.reply_text("用法: /cancel_orders BTC\n取消該幣種所有殘留掛單")
            return

        symbol = context.args[0].upper()
        if not symbol.endswith("USDT"):
            symbol += "USDT"

        result = self._trader.cancel_all_orders(symbol)
        if result.get("success"):
            await update.message.reply_text(f"✅ 已取消 {symbol} 所有掛單")
        else:
            await update.message.reply_text(f"❌ 取消失敗: {result.get('error')}")

    async def _cmd_briefing(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """手動觸發早報: /briefing"""
        if str(update.effective_chat.id) != str(self.chat_id):
            return

        if not self._briefing_callback:
            await update.message.reply_text("❌ 早報功能未初始化")
            return

        await update.message.reply_text("📝 正在產生早報，請稍候...")
        try:
            await self._briefing_callback()
        except Exception as e:
            await update.message.reply_text(f"❌ 早報產生失敗: {e}")

    async def _cmd_reset_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """清除所有交易資料，保留分析師訊息: /reset_trades"""
        if str(update.effective_chat.id) != str(self.chat_id):
            return

        if not self._db:
            await update.message.reply_text("❌ 資料庫未初始化")
            return

        # 需要確認
        if not context.args or context.args[0] != "confirm":
            await update.message.reply_text(
                "⚠️ 此操作將清除所有交易資料：\n\n"
                "• 所有交易記錄 (trades)\n"
                "• AI 決策記錄 (ai_decisions)\n"
                "• 分析師判斷記錄 (analyst_calls)\n"
                "• 學習日誌 (learning_log)\n"
                "• 訊號模式 (signal_patterns)\n"
                "• 分析師權重重設為 1.0\n\n"
                "✅ 保留：分析師訊息 (analyst_messages)\n\n"
                "確認請輸入：/reset_trades confirm"
            )
            return

        try:
            result = self._db.reset_trade_data()

            # 清除 PaperTrader 的虛擬持倉
            if self._trader and hasattr(self._trader, "_positions"):
                self._trader._positions.clear()

            text = (
                "✅ 交易資料已清除\n\n"
                f"• 交易: {result['trades']} 筆\n"
                f"• 分析師判斷: {result['analyst_calls']} 筆\n"
                f"• AI 決策: {result['ai_decisions']} 筆\n"
                f"• 學習日誌: {result['learning_logs']} 筆\n"
                f"• 訊號模式: {result['signal_patterns']} 筆\n"
                f"• 分析師權重: {result['analysts_reset']} 人已重設\n\n"
                "系統已準備好進行全新的 Paper Trading"
            )
            await update.message.reply_text(text)
        except Exception as e:
            await update.message.reply_text(f"❌ 清除失敗: {e}")

    async def _cmd_review(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """手動觸發覆盤: /review <trade_id>"""
        if str(update.effective_chat.id) != str(self.chat_id):
            return

        if not context.args:
            await update.message.reply_text("用法: /review <trade_id>\n例如: /review 3")
            return

        try:
            trade_id = int(context.args[0])
        except ValueError:
            await update.message.reply_text("❌ trade_id 必須是數字")
            return

        if not self._review_callback:
            await update.message.reply_text("❌ 覆盤功能未初始化")
            return

        await update.message.reply_text(f"🔄 正在覆盤 trade #{trade_id}...")

        try:
            result = await self._review_callback(trade_id)
            if result and result.get("review"):
                review = result["review"]
                score = review.get("overall_score", "N/A")
                timing = review.get("timing_assessment", "N/A")
                exit_a = review.get("exit_assessment", "N/A")
                lessons = review.get("lessons_learned", [])
                lessons_text = "\n".join(f"  • {l}" for l in lessons[:3]) if lessons else "N/A"
                text = (
                    f"✅ Trade #{trade_id} 覆盤完成\n\n"
                    f"評分: {score}/10\n"
                    f"進場: {timing}\n"
                    f"出場: {exit_a}\n\n"
                    f"教訓:\n{lessons_text}"
                )
                await update.message.reply_text(text[:4000])
            else:
                await update.message.reply_text(f"❌ Trade #{trade_id} 覆盤失敗")
        except Exception as e:
            await update.message.reply_text(f"❌ 覆盤錯誤: {e}")

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
