import base64
import json
import logging
from datetime import datetime, timezone

import requests

from modules.ai_analyzer import AIAnalyzer
from modules.database import Database
from modules.economic_calendar import EconomicCalendar
from modules.market_data import MarketData
from utils.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    協調分析流程：
    1. 收到分析師訊息 batch
    2. 拉取市場數據 + 目前持倉
    3. 調用 AI 分析（AI 可以選擇開倉、調整持倉、或不操作）
    4. 風控檢查
    5. 回傳最終決策
    """

    def __init__(
        self,
        config: dict,
        db: Database,
        market_data: MarketData,
        ai_analyzer: AIAnalyzer,
        risk_manager: RiskManager,
        economic_calendar: EconomicCalendar | None = None,
    ):
        self.config = config
        self.db = db
        self.market = market_data
        self.ai = ai_analyzer
        self.risk = risk_manager
        self.calendar = economic_calendar
        logger.info("DecisionEngine initialized")

    def process_signals(self, messages: list) -> dict | None:
        """
        處理來自 Discord 的一批分析師訊息

        回傳:
          - 開倉決策 (action=LONG/SHORT) 或
          - 調整決策 (action=ADJUST) 或
          - None (SKIP / 風控拒絕)
        """
        logger.info("Processing %d analyst messages", len(messages))

        # 1. 準備分析師訊息（附帶最新權重）
        analyst_msgs = []
        for m in messages:
            weight = self.db.get_analyst_weight(m.analyst)
            analyst_msgs.append({
                "analyst": m.analyst,
                "content": m.content,
                "weight": weight,
                "channel": m.channel_name,
                "timestamp": m.timestamp.isoformat(),
                "images": getattr(m, "images", []),
            })

        # 2. 偵測提及的幣種
        symbols = self._detect_symbols(messages)
        if not symbols:
            symbols = self.config["binance"].get("symbols", ["BTCUSDT"])
        logger.info("Detected symbols: %s", symbols)

        # 3. 獲取市場數據
        all_market_data = {}
        for symbol in symbols:
            all_market_data[symbol] = self.market.get_symbol_data(symbol)

        # 4. 取得目前持倉
        open_trades = self.db.get_open_trades()
        open_trades_info = [
            {
                "trade_id": t.id,
                "symbol": t.symbol,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "stop_loss": t.stop_loss,
                "take_profit": json.loads(t.take_profit) if isinstance(t.take_profit, str) else t.take_profit,
                "position_size": t.position_size,
                "confidence": t.confidence,
            }
            for t in open_trades
        ]

        # 5. 取得歷史績效和已知模式
        performance_stats = self.db.get_performance_stats()
        patterns = self.db.get_high_winrate_patterns()
        pattern_dicts = [
            {
                "name": p.pattern_name,
                "win_rate": p.win_rate,
                "occurrences": p.occurrences,
                "avg_profit": p.avg_profit,
            }
            for p in patterns
        ]

        # 6. AI 分析（把所有市場數據一起給 AI）
        # 合併所有幣種數據
        combined_market = {}
        for symbol, mdata in all_market_data.items():
            if "error" not in mdata:
                combined_market[symbol] = mdata

        if not combined_market:
            logger.warning("No valid market data available")
            return None

        # 7. 取得經濟日曆（接下來 24 小時 + 最近公布的數據）
        econ_text = ""
        if self.calendar:
            upcoming = self.calendar.get_upcoming_events(hours=24)
            recent = self.calendar.get_recent_releases(hours=4)
            all_econ = recent + upcoming
            econ_text = self.calendar.format_for_ai(all_econ)

        decision = self.ai.analyze(
            analyst_messages=analyst_msgs,
            market_data=combined_market,
            open_trades=open_trades_info if open_trades_info else None,
            performance_stats=performance_stats,
            known_patterns=pattern_dicts,
            economic_events=econ_text,
        )

        action = decision.get("action", "SKIP")

        # 7. 處理不同決策類型
        if action == "SKIP":
            reason = decision.get("reasoning", {}).get("skip_reason", "N/A")
            logger.info("AI recommends SKIP: %s", reason)
            return decision  # 回傳完整決策供記錄學習

        if action == "ADJUST":
            # 調整現有持倉，不需要風控檢查
            decision["_analyst_messages"] = analyst_msgs
            logger.info(
                "AI recommends ADJUST trade #%s: SL=%s TP=%s",
                decision.get("trade_id"),
                decision.get("new_stop_loss"),
                decision.get("new_take_profit"),
            )
            return decision

        if action in ("LONG", "SHORT"):
            # 開新倉 → 風控檢查
            risk_result = self.risk.check(decision)

            if not risk_result.passed:
                logger.warning(
                    "Decision rejected by risk manager:\n%s",
                    risk_result.summary(),
                )
                return {
                    **decision,
                    "_rejected": True,
                    "_risk_summary": risk_result.summary(),
                    "_risk_checks": risk_result.checks,
                }

            decision["_rejected"] = False
            decision["_risk_summary"] = risk_result.summary()
            decision["_risk_checks"] = risk_result.checks
            decision["_analyst_messages"] = analyst_msgs

            logger.info(
                "Decision approved: %s %s confidence=%d rr=%.2f",
                action,
                decision.get("symbol", "?"),
                decision.get("confidence", 0),
                decision.get("risk_reward", 0),
            )
            return decision

        logger.warning("Unknown action: %s", action)
        return None

    def process_scanner_signals(self, db_messages: list, symbols: list[str]) -> dict | None:
        """
        處理市場掃描器觸發的分析

        db_messages: 資料庫中的 AnalystMessage 記錄
        symbols: 要掃描的交易對清單
        """
        logger.info("Scanner processing %d recent analyst messages for %s",
                     len(db_messages), symbols)

        # 1. 把 DB 訊息轉成 AI 需要的格式（附帶權重 + 圖片）
        analyst_msgs = []
        for m in db_messages:
            weight = self.db.get_analyst_weight(m.analyst_name)
            # 從 DB 載入圖片 URL 並重新下載
            images = self._load_images_from_db(m)
            analyst_msgs.append({
                "analyst": m.analyst_name,
                "content": m.content,
                "weight": weight,
                "channel": m.channel or "",
                "timestamp": m.timestamp.strftime("%m-%d %H:%M") if m.timestamp else "",
                "images": images,
            })

        # 2. 取得市場數據（含所有 K 線和技術指標）
        combined_market = {}
        for symbol in symbols:
            data = self.market.get_symbol_data(symbol)
            if "error" not in data:
                combined_market[symbol] = data

        if not combined_market:
            logger.warning("Scanner: no valid market data available")
            return None

        # 3. 取得持倉、績效、模式
        open_trades = self.db.get_open_trades()
        open_trades_info = [
            {
                "trade_id": t.id,
                "symbol": t.symbol,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "stop_loss": t.stop_loss,
                "take_profit": json.loads(t.take_profit)
                    if isinstance(t.take_profit, str) else t.take_profit,
                "position_size": t.position_size,
                "confidence": t.confidence,
            }
            for t in open_trades
        ]

        performance_stats = self.db.get_performance_stats()
        patterns = self.db.get_high_winrate_patterns()
        pattern_dicts = [
            {
                "name": p.pattern_name,
                "win_rate": p.win_rate,
                "occurrences": p.occurrences,
                "avg_profit": p.avg_profit,
            }
            for p in patterns
        ]

        # 4. 經濟日曆
        econ_text = ""
        if self.calendar:
            try:
                upcoming = self.calendar.get_upcoming_events(hours=24)
                recent = self.calendar.get_recent_releases(hours=4)
                all_econ = recent + upcoming
                econ_text = self.calendar.format_for_ai(all_econ)
            except Exception as e:
                logger.warning("Scanner: economic calendar error: %s", e)

        # 5. 呼叫 AI 掃描器分析
        decision = self.ai.analyze_scanner(
            analyst_messages=analyst_msgs,
            market_data=combined_market,
            open_trades=open_trades_info if open_trades_info else None,
            performance_stats=performance_stats,
            known_patterns=pattern_dicts,
            economic_events=econ_text,
        )

        action = decision.get("action", "SKIP")

        # 6. 處理決策
        if action == "SKIP":
            reason = decision.get("reasoning", {}).get("skip_reason", "N/A")
            logger.info("Scanner AI recommends SKIP: %s", reason)
            return decision

        if action == "ADJUST":
            decision["_analyst_messages"] = analyst_msgs
            return decision

        if action in ("LONG", "SHORT"):
            risk_result = self.risk.check(decision)

            if not risk_result.passed:
                logger.warning("Scanner decision rejected by risk manager:\n%s",
                               risk_result.summary())
                return {
                    **decision,
                    "_rejected": True,
                    "_risk_summary": risk_result.summary(),
                    "_risk_checks": risk_result.checks,
                }

            decision["_rejected"] = False
            decision["_risk_summary"] = risk_result.summary()
            decision["_risk_checks"] = risk_result.checks
            decision["_analyst_messages"] = analyst_msgs

            logger.info(
                "Scanner decision approved: %s %s confidence=%d rr=%.2f",
                action,
                decision.get("symbol", "?"),
                decision.get("confidence", 0),
                decision.get("risk_reward", 0),
            )
            return decision

        logger.warning("Scanner: unknown action: %s", action)
        return None

    def _detect_symbols(self, messages: list) -> list[str]:
        """從分析師訊息中偵測提及的幣種"""
        known = {
            "BTC": "BTCUSDT", "比特幣": "BTCUSDT", "比特币": "BTCUSDT",
            "大餅": "BTCUSDT", "大饼": "BTCUSDT",
            "ETH": "ETHUSDT", "乙太": "ETHUSDT", "以太": "ETHUSDT",
            "以太坊": "ETHUSDT", "姨太": "ETHUSDT",
            "SOL": "SOLUSDT",
            "BNB": "BNBUSDT",
            "XRP": "XRPUSDT",
            "DOGE": "DOGEUSDT",
            "ADA": "ADAUSDT",
            "AVAX": "AVAXUSDT",
            "DOT": "DOTUSDT",
            "MATIC": "MATICUSDT",
            "LINK": "LINKUSDT",
        }
        found = set()
        for m in messages:
            text = m.content.upper()
            for keyword, symbol in known.items():
                if keyword.upper() in text:
                    found.add(symbol)
        # 如果沒偵測到，用配置的
        return list(found) if found else list(self.config["binance"].get("symbols", []))

    @staticmethod
    def _load_images_from_db(msg) -> list[dict]:
        """從 DB 的 AnalystMessage 載入圖片 URL 並重新下載為 base64"""
        if not getattr(msg, "images", None):
            return []
        try:
            url_list = json.loads(msg.images) if isinstance(msg.images, str) else msg.images
        except (json.JSONDecodeError, TypeError):
            return []
        if not url_list:
            return []

        result = []
        for img_info in url_list[:4]:  # 最多 4 張
            url = img_info.get("url", "")
            if not url:
                continue
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200 and len(resp.content) <= 5 * 1024 * 1024:
                    result.append({
                        "base64": base64.b64encode(resp.content).decode("utf-8"),
                        "media_type": img_info.get("media_type", "image/png"),
                    })
                else:
                    logger.warning("Image download failed: status=%d size=%d url=%s",
                                   resp.status_code, len(resp.content), url[:80])
            except Exception as e:
                logger.warning("Failed to download image from DB URL: %s", e)
        return result
