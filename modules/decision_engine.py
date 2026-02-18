import base64
import json
import logging
from datetime import datetime, timedelta, timezone

import requests

from modules.ai_analyzer import AIAnalyzer
from modules.database import Database
from modules.economic_calendar import EconomicCalendar
from modules.market_data import MarketData
from modules.message_scorer import MessageScorer
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
        message_scorer: MessageScorer | None = None,
    ):
        self.config = config
        self.db = db
        self.market = market_data
        self.ai = ai_analyzer
        self.risk = risk_manager
        self.calendar = economic_calendar
        self.scorer = message_scorer

        # 從 config 建立分析師風格對照表
        self._analyst_styles: dict[str, str] = {}
        for ch in config.get("discord", {}).get("monitored_channels", []):
            style = ch.get("style", "")
            if style:
                self._analyst_styles[ch["analyst"]] = style

        logger.info("DecisionEngine initialized")

    def process_signals(self, messages: list) -> dict | None:
        """
        處理來自 Discord 的一批分析師訊息

        回傳:
          - 開倉決策 (action=LONG/SHORT) 或
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

        # 2. 套用時間衰減 + 試用期折扣
        analyst_msgs = self._apply_time_decay(analyst_msgs)
        analyst_msgs = self._apply_trial_period(analyst_msgs)

        # 2.5 品質評分過濾
        if self.scorer:
            analyst_msgs = self.scorer.score_messages(analyst_msgs)
            if not analyst_msgs:
                logger.info("All messages filtered by quality scoring")
                return None

        # 3. 偵測提及的幣種
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

        # 6.5 根據市場狀態調整分析師權重
        analyst_msgs = self._apply_market_specialization(analyst_msgs, combined_market)

        # 7. 取得經濟日曆（接下來 24 小時 + 最近公布的數據）
        econ_text = ""
        if self.calendar:
            upcoming = self.calendar.get_upcoming_events(hours=24)
            recent = self.calendar.get_recent_releases(hours=4)
            all_econ = recent + upcoming
            econ_text = self.calendar.format_for_ai(all_econ)

        # 取得分析師績效檔案（含風格描述）
        analyst_names_in_batch = list(set(m["analyst"] for m in analyst_msgs))
        analyst_profiles = self.db.get_analyst_profiles(names=analyst_names_in_batch)
        for p in analyst_profiles:
            p["style"] = self._analyst_styles.get(p["name"], "")

        # 取得近期覆盤教訓
        review_lessons = self.db.get_recent_review_lessons(limit=10)

        # 市場狀態策略指引
        market_strategy_hint = self._build_market_strategy_hint(combined_market)

        # 計算分析師共識（在所有權重調整之後）
        consensus = self._calc_consensus(analyst_msgs)
        logger.info("Consensus: %s (strength=%.1f%%)", consensus["dominant"], consensus["strength"])

        decision = self.ai.analyze(
            analyst_messages=analyst_msgs,
            market_data=combined_market,
            open_trades=open_trades_info if open_trades_info else None,
            performance_stats=performance_stats,
            known_patterns=pattern_dicts,
            economic_events=econ_text,
            consensus=consensus,
            analyst_profiles=analyst_profiles,
            review_lessons=review_lessons,
            market_strategy_hint=market_strategy_hint,
        )

        action = decision.get("action", "SKIP")

        # 7. 處理不同決策類型
        if action == "SKIP":
            reason = decision.get("reasoning", {}).get("skip_reason", "N/A")
            logger.info("AI recommends SKIP: %s", reason)
            return decision  # 回傳完整決策供記錄學習

        if action in ("LONG", "SHORT"):
            # 附加市場狀態
            symbol = decision.get("symbol", "")
            sym_data = combined_market.get(symbol, {})
            market_cond = sym_data.get("technical_indicators", {}).get(
                "market_condition", "UNKNOWN")
            decision["_market_condition"] = market_cond

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
                "Decision approved: %s %s confidence=%d rr=%.2f [%s]",
                action,
                decision.get("symbol", "?"),
                decision.get("confidence", 0),
                decision.get("risk_reward", 0),
                market_cond,
            )
            return decision

        logger.warning("Unknown action: %s", action)
        return None

    def process_follow_signals(self, messages: list) -> list[dict] | None:
        """
        跟單模式：AI 只解析分析師訊息，提取入場/止損/止盈點位，100% 按照執行
        支援：單入場點、雙入場點（加倉）、平倉指令、多位分析師同時發訊號
        回傳 list[dict]，每個元素代表一個獨立訊號（不同幣種或不同方向）
        """
        logger.info("Follow mode: parsing %d messages", len(messages))

        # 1. 合併所有訊息 + 收集圖片
        lines = []
        all_images = []
        analyst_names = []
        for m in messages:
            lines.append(f"[{m.analyst} {m.timestamp.strftime('%H:%M')}]: {m.content}")
            analyst_names.append(m.analyst)
            all_images.extend(getattr(m, "images", []))
        combined_text = "\n".join(lines)
        analyst_tag = ", ".join(set(analyst_names))

        # 2. AI 解析訊號（可能回傳多個）
        parsed_list = self.ai.parse_signal(combined_text, images=all_images[:4] or None)
        logger.info("Follow mode: AI returned %d signal(s)", len(parsed_list))

        results = []
        follow_pos = self.config.get("trading", {}).get("follow_position_size", 5.0)
        leverage = self._get_follow_leverage(analyst_names)

        for parsed in parsed_list:
            action = parsed.get("action", "SKIP")
            symbol = parsed.get("symbol", "")

            # SKIP
            if action == "SKIP":
                logger.info("Follow mode SKIP [%s]: %s", symbol, parsed.get("skip_reason", ""))
                results.append(parsed)
                continue

            # CLOSE
            if action == "CLOSE":
                logger.info("Follow mode CLOSE detected for %s", symbol)
                results.append({
                    "action": "CLOSE",
                    "symbol": symbol,
                    "_analyst_messages": [
                        {"analyst": m.analyst, "content": m.content} for m in messages
                    ],
                })
                continue

            if action not in ("LONG", "SHORT"):
                continue

            entry_1 = parsed.get("entry_1")
            entry_2 = parsed.get("entry_2")
            stop_loss = parsed.get("stop_loss")
            take_profit = parsed.get("take_profit") or []

            if not entry_1 or entry_1.get("price") is None or stop_loss is None:
                logger.warning("Follow mode: missing entry_1 or stop_loss for %s, skipping", symbol)
                results.append({"action": "SKIP", "skip_reason": f"{symbol} 缺少入場價或止損"})
                continue

            if not take_profit:
                logger.warning("Follow mode: missing take_profit for %s, skipping", symbol)
                results.append({"action": "SKIP", "skip_reason": f"{symbol} 缺少止盈目標"})
                continue

            # 倉位大小：有兩個點位則各佔一半
            pos_size_1 = round(follow_pos / 2, 1) if entry_2 else follow_pos

            # 建立決策
            rr = self._calc_rr(action, entry_1["price"], stop_loss, take_profit)
            decision = {
                "action": action,
                "symbol": symbol,
                "confidence": 90,
                "leverage": leverage,
                "entry": entry_1,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "position_size": pos_size_1,
                "risk_reward": rr,
                "reasoning": {
                    "analyst_consensus": f"跟單: {analyst_tag}",
                    "technical": "100% 按照分析師指定點位",
                    "sentiment": "N/A",
                    "historical": "N/A",
                },
                "_follow_mode": True,
                "_analyst_messages": [
                    {"analyst": m.analyst, "content": m.content} for m in messages
                ],
            }

            # 風控（跟單模式：跳過 confidence/RR，只檢查允許幣種）
            risk_result = self.risk.check(decision, follow_mode=True)
            if not risk_result.passed:
                logger.warning("Follow mode risk check failed for %s:\n%s",
                               symbol, risk_result.summary())
                results.append({
                    **decision,
                    "_rejected": True,
                    "_risk_summary": risk_result.summary(),
                    "_risk_checks": risk_result.checks,
                })
                continue

            decision["_rejected"] = False
            decision["_risk_summary"] = risk_result.summary()
            decision["_risk_checks"] = risk_result.checks

            # 若有第二個入場點，附加為加倉決策
            if entry_2 and entry_2.get("price") is not None:
                decision["_addon_entry"] = {
                    "action": action,
                    "symbol": symbol,
                    "entry": entry_2,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "leverage": leverage,
                    "position_size": round(follow_pos / 2, 1),
                    "_follow_mode": True,
                    "_is_addon": True,
                    "confidence": 90,
                    "risk_reward": rr,
                    "reasoning": decision["reasoning"],
                }
                logger.info("Follow mode: dual entry for %s (%.2f + %.2f)",
                            symbol, entry_1["price"], entry_2["price"])

            logger.info("Follow mode decision: %s %s @ %.2f (SL=%.2f TP=%s lev=%dx pos=%.1f%%)",
                        action, symbol, entry_1["price"], stop_loss, take_profit, leverage, pos_size_1)
            results.append(decision)

        return results if results else None

    def _get_follow_leverage(self, analyst_names: list) -> int:
        """從 config 取分析師指定的槓桿倍數"""
        channels = self.config.get("discord", {}).get("monitored_channels", [])
        for name in analyst_names:
            for ch in channels:
                if ch.get("analyst") == name:
                    lev = ch.get("leverage")
                    if lev:
                        return int(lev)
        # fallback: leverage_map 最大值 或 default_leverage
        lev_map = self.config.get("trading", {}).get("leverage_map", {})
        if lev_map:
            return max(lev_map.values())
        return self.config.get("trading", {}).get("default_leverage", 100)

    @staticmethod
    def _calc_rr(action: str, entry: float, sl: float, tp_list: list) -> float:
        """計算風報比（用最後一個 TP）"""
        if not tp_list or not sl or not entry:
            return 0.0
        tp = tp_list[-1]
        if action == "LONG":
            reward = abs(tp - entry)
            risk = abs(entry - sl)
        else:
            reward = abs(entry - tp)
            risk = abs(sl - entry)
        return round(reward / risk, 2) if risk > 0 else 0.0

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

        # 2. 套用時間衰減 + 試用期折扣
        analyst_msgs = self._apply_time_decay(analyst_msgs)
        analyst_msgs = self._apply_trial_period(analyst_msgs)

        # 2.5 品質評分過濾
        if self.scorer:
            analyst_msgs = self.scorer.score_messages(analyst_msgs)
            if not analyst_msgs:
                logger.info("Scanner: all messages filtered by quality scoring")
                return None

        # 3. 取得市場數據（含所有 K 線和技術指標）
        combined_market = {}
        for symbol in symbols:
            data = self.market.get_symbol_data(symbol)
            if "error" not in data:
                combined_market[symbol] = data

        if not combined_market:
            logger.warning("Scanner: no valid market data available")
            return None

        # 3.5 根據市場狀態調整分析師權重
        analyst_msgs = self._apply_market_specialization(analyst_msgs, combined_market)

        # 4. 取得持倉、績效、模式
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

        # 取得分析師績效檔案（含風格描述）
        analyst_names_in_batch = list(set(m["analyst"] for m in analyst_msgs))
        analyst_profiles = self.db.get_analyst_profiles(names=analyst_names_in_batch)
        for p in analyst_profiles:
            p["style"] = self._analyst_styles.get(p["name"], "")

        # 取得近期覆盤教訓
        review_lessons = self.db.get_recent_review_lessons(limit=10)

        # 市場狀態策略指引
        market_strategy_hint = self._build_market_strategy_hint(combined_market)

        # 計算分析師共識
        consensus = self._calc_consensus(analyst_msgs)
        logger.info("Scanner consensus: %s (strength=%.1f%%)",
                     consensus["dominant"], consensus["strength"])

        # 5. 呼叫 AI 掃描器分析
        decision = self.ai.analyze_scanner(
            analyst_messages=analyst_msgs,
            market_data=combined_market,
            open_trades=open_trades_info if open_trades_info else None,
            performance_stats=performance_stats,
            known_patterns=pattern_dicts,
            economic_events=econ_text,
            consensus=consensus,
            analyst_profiles=analyst_profiles,
            review_lessons=review_lessons,
            market_strategy_hint=market_strategy_hint,
        )

        action = decision.get("action", "SKIP")

        # 6. 處理決策
        if action == "SKIP":
            reason = decision.get("reasoning", {}).get("skip_reason", "N/A")
            logger.info("Scanner AI recommends SKIP: %s", reason)
            return decision

        if action in ("LONG", "SHORT"):
            # 附加市場狀態
            symbol = decision.get("symbol", "")
            sym_data = combined_market.get(symbol, {})
            market_cond = sym_data.get("technical_indicators", {}).get(
                "market_condition", "UNKNOWN")
            decision["_market_condition"] = market_cond

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
                "Scanner decision approved: %s %s confidence=%d rr=%.2f [%s]",
                action,
                decision.get("symbol", "?"),
                decision.get("confidence", 0),
                decision.get("risk_reward", 0),
                market_cond,
            )
            return decision

        logger.warning("Scanner: unknown action: %s", action)
        return None

    def _build_market_strategy_hint(self, market_data: dict) -> str:
        """根據當前市場狀態（TRENDING/RANGING）產生策略指引"""
        hints = []
        for symbol, data in market_data.items():
            indicators = data.get("technical_indicators", {})
            condition = indicators.get("market_condition", "UNKNOWN")
            adx = indicators.get("ADX", 0)
            trend = indicators.get("trend", "neutral")

            if condition == "TRENDING":
                hints.append(
                    f"📈 {symbol} 趨勢行情 (ADX={adx}, 趨勢={trend})\n"
                    f"  策略: 順勢交易，不要逆勢。"
                    f"止盈設寬（BTC 2-4%, ETH 3-5%），讓利潤奔跑。"
                    f"止損設在關鍵支撐/阻力位外側。"
                    f"回調到 EMA 附近是好的入場時機。"
                )
            elif condition == "RANGING":
                hints.append(
                    f"📊 {symbol} 盤整行情 (ADX={adx})\n"
                    f"  策略: 均值回歸，高拋低吸。"
                    f"止盈適中（BTC 1-2%, ETH 1.5-2.5%），不貪不急。"
                    f"在布林帶上下軌附近反向操作勝率較高。"
                )
            else:
                hints.append(f"⚠️ {symbol} 市場狀態不明 — 建議降低倉位或觀望")

        return "\n".join(hints) if hints else "無市場狀態數據"

    def _apply_time_decay(self, analyst_msgs: list) -> list:
        """對分析師訊息套用時間衰減，近期訊息權重較高"""
        now = datetime.now(timezone.utc)
        for msg in analyst_msgs:
            ts_str = msg.get("timestamp", "")
            try:
                if "T" in ts_str:
                    ts = datetime.fromisoformat(ts_str)
                else:
                    # scanner format: "MM-DD HH:MM"
                    ts = datetime.strptime(ts_str, "%m-%d %H:%M").replace(
                        year=now.year, tzinfo=timezone.utc
                    )
            except (ValueError, TypeError):
                msg["time_decay"] = 1.0
                continue

            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            age = now - ts
            if age < timedelta(minutes=30):
                factor = 1.0
            elif age < timedelta(hours=2):
                factor = 0.8
            elif age < timedelta(hours=6):
                factor = 0.5
            elif age < timedelta(hours=24):
                factor = 0.2
            else:
                factor = 0.1

            msg["weight"] *= factor
            msg["time_decay"] = factor

        decayed = [m for m in analyst_msgs if m.get("time_decay", 1.0) < 1.0]
        if decayed:
            logger.info("Time decay applied: %d/%d messages decayed",
                        len(decayed), len(analyst_msgs))
        return analyst_msgs

    def _apply_trial_period(self, analyst_msgs: list) -> list:
        """新分析師（<N 筆交易記錄）套用試用期權重折扣"""
        learn_cfg = self.config.get("learning", {})
        trial_calls = learn_cfg.get("trial_period_calls", 10)
        trial_discount = learn_cfg.get("trial_period_discount", 0.5)

        for msg in analyst_msgs:
            analyst = self.db.get_or_create_analyst(msg["analyst"])
            if analyst.total_calls < trial_calls:
                msg["weight"] *= trial_discount
                msg["trial_period"] = True
                logger.debug("Trial period discount for %s (calls=%d)",
                             msg["analyst"], analyst.total_calls)
        return analyst_msgs

    def _apply_market_specialization(self, analyst_msgs: list, market_data: dict) -> list:
        """根據市場狀態（TRENDING/RANGING）調整分析師權重"""
        # 從任一幣種取得 market_condition
        market_condition = None
        for symbol, data in market_data.items():
            indicators = data.get("technical_indicators", {})
            if "market_condition" in indicators:
                market_condition = indicators["market_condition"]
                break

        if not market_condition:
            return analyst_msgs

        for msg in analyst_msgs:
            analyst = self.db.get_or_create_analyst(msg["analyst"])
            # 只有有足夠資料的分析師才調整
            if analyst.total_calls < 10:
                continue

            overall = analyst.accuracy or 0.5
            if overall == 0:
                overall = 0.5

            if market_condition == "TRENDING":
                spec_accuracy = analyst.trend_accuracy or overall
            else:
                spec_accuracy = analyst.range_accuracy or overall

            # ±30% adjustment, clamped
            if overall > 0:
                adj = 0.3 * (spec_accuracy - overall) / overall
                adj = max(-0.3, min(0.3, adj))
                msg["weight"] *= (1 + adj)
                if abs(adj) > 0.05:
                    logger.debug("Specialization: %s weight adj %.0f%% (%s market)",
                                 msg["analyst"], adj * 100, market_condition)

        return analyst_msgs

    def _calc_consensus(self, analyst_msgs: list) -> dict:
        """計算加權多空共識強度"""
        bullish_weight = 0.0
        bearish_weight = 0.0
        neutral_weight = 0.0

        bullish_kw = ["多", "long", "買", "buy", "看漲", "bullish", "做多",
                       "上漲", "反彈", "突破", "支撐"]
        bearish_kw = ["空", "short", "賣", "sell", "看跌", "bearish", "做空",
                       "下跌", "回調", "跌破", "壓力"]

        for msg in analyst_msgs:
            text = msg["content"].lower()
            weight = msg["weight"]

            is_bull = any(k in text for k in bullish_kw)
            is_bear = any(k in text for k in bearish_kw)

            if is_bull and not is_bear:
                bullish_weight += weight
            elif is_bear and not is_bull:
                bearish_weight += weight
            else:
                neutral_weight += weight

        total = bullish_weight + bearish_weight + neutral_weight
        if total == 0:
            return {
                "bullish_pct": 0, "bearish_pct": 0, "neutral_pct": 100,
                "dominant": "NEUTRAL", "strength": 0,
            }

        dominant = "BULLISH" if bullish_weight > bearish_weight else "BEARISH"
        if bullish_weight == bearish_weight:
            dominant = "NEUTRAL"

        return {
            "bullish_pct": round(bullish_weight / total * 100, 1),
            "bearish_pct": round(bearish_weight / total * 100, 1),
            "neutral_pct": round(neutral_weight / total * 100, 1),
            "dominant": dominant,
            "strength": round(abs(bullish_weight - bearish_weight) / total * 100, 1),
        }

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
