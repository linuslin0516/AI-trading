import json
import logging
from datetime import datetime, timedelta, timezone

import requests

from modules.ai_analyzer import AIAnalyzer
from modules.database import Database
from utils.risk_manager import RiskManager

FUTURES_URL = "https://fapi.binance.com"

logger = logging.getLogger(__name__)


class LearningEngine:
    """
    AI 學習引擎：
    A. 每筆交易後 → 更新分析師權重
    B. 每 20 筆 → 分析訊號模式
    C. 每 50 筆 → 優化策略參數（並即時套用）
    D. 平倉後 → 生成覆盤報告
    """

    def __init__(self, config: dict, db: Database, ai_analyzer: AIAnalyzer,
                 risk_manager: RiskManager | None = None):
        self.config = config
        self.db = db
        self.ai = ai_analyzer
        self.risk = risk_manager

        learn_cfg = config.get("learning", {})
        self.enabled = learn_cfg.get("enabled", True)
        self.min_trades = learn_cfg.get("min_trades_before_learning", 10)
        self.weight_update_freq = learn_cfg.get("analyst_weight_update_frequency", 1)
        self.param_opt_freq = learn_cfg.get("parameter_optimization_frequency", 50)
        self.pattern_freq = learn_cfg.get("pattern_analysis_frequency", 20)
        self.weight_decay = learn_cfg.get("weight_decay", 0.95)
        self.perf_weight = learn_cfg.get("performance_weight", 0.7)
        self.recency_weight = learn_cfg.get("recency_weight", 0.3)

        logger.info("LearningEngine initialized (enabled=%s)", self.enabled)

    async def on_trade_closed(self, trade_id: int) -> dict:
        """
        交易關閉時的輕量處理（不做覆盤，等待延遲覆盤）

        只記錄基本資訊和模式，覆盤由 run_pending_reviews 延遲 4 小時執行。
        """
        result = {"review": None, "events": []}

        if not self.enabled:
            return result

        trade = self.db.get_trade(trade_id)
        if not trade or trade.status != "CLOSED":
            return result

        logger.info("Trade #%d closed, review scheduled in ~4 hours", trade_id)

        # 記錄訊號模式（不需要延遲）
        self._record_pattern(trade)

        result["events"].append({
            "type": "TRADE_CLOSED",
            "description": (
                f"Trade #{trade.id} 已平倉: "
                f"{trade.outcome} {trade.profit_pct:+.2f}%\n"
                f"覆盤將在 4 小時後執行（等待後續價格數據）"
            ),
        })

        return result

    async def run_pending_reviews(self, notify_callback=None):
        """
        檢查並執行延遲覆盤（平倉超過 4 小時且尚未覆盤的交易）

        Args:
            notify_callback: async func(trade, review, events) — 發送 TG 通知
        """
        if not self.enabled:
            return

        review_delay_hours = 4
        cutoff = datetime.now(timezone.utc) - timedelta(hours=review_delay_hours)

        trades = self.db.get_unreviewed_closed_trades(before=cutoff)
        if not trades:
            return

        for trade in trades:
            try:
                logger.info("Running delayed review for trade #%d (closed %s)",
                            trade.id, trade.closed_at)

                # 取得平倉後 4 小時的 1h K 線
                post_close_data = self._fetch_post_close_klines(trade)

                # AI 覆盤（含後續價格走勢）
                review = self._run_review(trade, post_close_data=post_close_data)
                events = []

                if review:
                    events.append({
                        "type": "TRADE_REVIEW",
                        "description": (
                            f"Trade #{trade.id} 覆盤完成: "
                            f"{trade.outcome} {trade.profit_pct:+.2f}%, "
                            f"評分={review.get('overall_score', 'N/A')}/10"
                        ),
                    })

                    # 更新分析師權重
                    weight_changes = self._update_analyst_weights(trade, review)
                    if weight_changes:
                        events.append({
                            "type": "WEIGHT_UPDATE",
                            "description": "分析師權重更新:\n" + "\n".join(weight_changes),
                        })

                # 大規模學習檢查
                stats = self.db.get_performance_stats()
                total_trades = stats.get("total", 0)

                if total_trades >= self.min_trades:
                    if total_trades % self.pattern_freq == 0:
                        patterns = self._analyze_patterns()
                        if patterns:
                            events.append({
                                "type": "PATTERN_FOUND",
                                "description": f"發現 {len(patterns)} 個高勝率模式",
                            })
                    if total_trades % self.param_opt_freq == 0:
                        changes = self._optimize_parameters()
                        if changes:
                            events.append({
                                "type": "PARAM_OPTIMIZED",
                                "description": "策略參數已自動優化:\n" + "\n".join(
                                    f"  {k}: {v['old']} → {v['new']}"
                                    for k, v in changes.items()
                                ),
                            })

                # 發送通知
                if notify_callback and (review or events):
                    await notify_callback(trade, review, events)

            except Exception as e:
                logger.error("Delayed review failed for trade #%d: %s", trade.id, e)

    def _fetch_post_close_klines(self, trade) -> str:
        """取得平倉後 4 小時的 1h K 線走勢"""
        try:
            close_time = trade.closed_at
            if close_time and close_time.tzinfo is None:
                close_time = close_time.replace(tzinfo=timezone.utc)
            if not close_time:
                return "無法取得平倉時間"

            start_ms = int(close_time.timestamp() * 1000)
            r = requests.get(
                f"{FUTURES_URL}/fapi/v1/klines",
                params={
                    "symbol": trade.symbol,
                    "interval": "1h",
                    "startTime": start_ms,
                    "limit": 4,
                },
                timeout=10,
            )
            r.raise_for_status()
            klines = r.json()

            if not klines:
                return "尚無足夠的後續K線數據"

            lines = [f"出場價: {trade.exit_price}"]
            for i, k in enumerate(klines):
                o, h, l, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
                lines.append(
                    f"+{i+1}h: 開 {o:.1f} → 收 {c:.1f} "
                    f"(最高 {h:.1f} / 最低 {l:.1f})"
                )

            # 計算整體走勢摘要
            if klines:
                first_open = float(klines[0][1])
                last_close = float(klines[-1][4])
                total_move = last_close - trade.exit_price
                direction = "上漲" if total_move > 0 else "下跌"
                lines.append(
                    f"\n→ 平倉後 {len(klines)} 小時價格{direction} "
                    f"{abs(total_move):.1f} 點 "
                    f"({total_move / trade.exit_price * 100:+.2f}%)"
                )

            return "\n".join(lines)

        except Exception as e:
            logger.warning("Failed to fetch post-close klines for trade #%d: %s",
                           trade.id, e)
            return "無法取得後續價格數據"

    def _check_price_deviation(self, trade) -> float | None:
        """比對 testnet exit price 與主網當前價格的偏差百分比

        Returns:
            偏差百分比（正=testnet 偏高, 負=testnet 偏低）, None=無法比對
        """
        try:
            r = requests.get(
                f"{FUTURES_URL}/fapi/v1/ticker/price",
                params={"symbol": trade.symbol}, timeout=5,
            )
            mainnet_price = float(r.json()["price"])
            if mainnet_price <= 0 or not trade.exit_price:
                return None

            deviation = (trade.exit_price - mainnet_price) / mainnet_price * 100
            logger.info(
                "Price deviation for %s: testnet_exit=%.2f, mainnet=%.2f, dev=%.1f%%",
                trade.symbol, trade.exit_price, mainnet_price, deviation,
            )
            return deviation
        except Exception as e:
            logger.warning("Failed to check price deviation: %s", e)
            return None

    def _run_review(self, trade, post_close_data: str = "") -> dict | None:
        """調用 AI 進行覆盤"""
        try:
            analyst_opinions = trade.analyst_opinions or "N/A"
            technical_signals = trade.technical_signals or "{}"
            take_profit = trade.take_profit or "[]"

            # 快速回饋資訊
            quick_fb = {}
            if trade.quick_feedback:
                try:
                    quick_fb = json.loads(trade.quick_feedback) if isinstance(
                        trade.quick_feedback, str) else trade.quick_feedback
                except (json.JSONDecodeError, TypeError):
                    quick_fb = {}

            # 截斷過長欄位，避免超過 Claude 200K token 限制
            if len(analyst_opinions) > 3000:
                analyst_opinions = analyst_opinions[:3000] + "\n...(截斷)"
            ai_reasoning = trade.ai_reasoning or "N/A"
            if len(ai_reasoning) > 2000:
                ai_reasoning = ai_reasoning[:2000] + "\n...(截斷)"

            trade_data = {
                "symbol": trade.symbol,
                "direction": trade.direction,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "stop_loss": trade.stop_loss,
                "take_profit": take_profit,
                "position_size": trade.position_size,
                "confidence": trade.confidence,
                "hold_duration": self._format_duration(trade.hold_duration or 0),
                "outcome": trade.outcome,
                "profit_pct": trade.profit_pct,
                "analyst_opinions": analyst_opinions,
                "technical_signals": json.loads(technical_signals) if isinstance(technical_signals, str) else technical_signals,
                "ai_reasoning": ai_reasoning,
                "quick_feedback": quick_fb if quick_fb else "N/A",
                "post_close_price_action": post_close_data or "N/A",
            }

            review = self.ai.review_trade(trade_data)

            # 檢查覆盤結果是否有效（超時/API 錯誤會返回無效格式）
            if not review or "error" in review or "overall_score" not in review:
                logger.warning(
                    "Review invalid for trade #%d (error=%s), retrying once...",
                    trade.id, review.get("error") if review else "None",
                )
                review = self.ai.review_trade(trade_data)
                if not review or "error" in review or "overall_score" not in review:
                    logger.error("Review retry also failed for trade #%d", trade.id)
                    return None

            # 保存覆盤結果
            self.db.update_trade(trade.id, review=review)

            self.db.add_learning_log(
                event_type="REVIEW",
                description=f"Trade #{trade.id} reviewed: {trade.outcome} {trade.profit_pct:+.2f}%",
                details={"trade_id": trade.id, "score": review.get("overall_score")},
            )

            logger.info("Review complete for trade #%d, score=%s",
                         trade.id, review.get("overall_score"))
            return review

        except Exception as e:
            logger.error("Review failed for trade #%d: %s", trade.id, e)
            return None

    def _update_analyst_weights(self, trade, review: dict) -> list[str]:
        """根據覆盤結果更新分析師權重，回傳變更描述"""
        analyst_performance = review.get("analyst_performance", [])
        changes = []

        for ap in analyst_performance:
            name = ap.get("name", "")
            was_correct = ap.get("was_correct", False)
            weight_adj = ap.get("weight_adjustment", 0)

            if not name:
                continue

            # 記錄分析師判斷結果
            self.db.mark_analyst_call_result(trade.id, name, was_correct)

            # 取得當前分析師數據
            analyst = self.db.get_or_create_analyst(name)

            # 更新統計
            new_total = analyst.total_calls + 1
            new_correct = analyst.correct_calls + (1 if was_correct else 0)
            new_accuracy = new_correct / new_total if new_total > 0 else 0

            # 計算近期準確率
            recent_7d = self._calc_recent_accuracy(name, days=7)
            recent_30d = self._calc_recent_accuracy(name, days=30)

            # 計算新權重
            # weight = (overall_accuracy * perf_w + recent_accuracy * recency_w) * decay_factor
            blended_accuracy = (
                new_accuracy * self.perf_weight +
                recent_7d * self.recency_weight
            )

            # 權重在 0.5 ~ 2.0 之間
            new_weight = max(0.5, min(2.0, blended_accuracy * 2))

            # 加上 AI 建議的微調
            new_weight = max(0.5, min(2.0, new_weight + weight_adj))

            # 更新趨勢/盤整專項準確率
            update_kwargs = dict(
                total_calls=new_total,
                correct_calls=new_correct,
                accuracy=round(new_accuracy, 4),
                current_weight=round(new_weight, 4),
                recent_7d_accuracy=round(recent_7d, 4),
                recent_30d_accuracy=round(recent_30d, 4),
            )

            market_cond = getattr(trade, "market_condition", None)
            if market_cond == "TRENDING":
                # 簡單移動平均更新
                old_ta = analyst.trend_accuracy or 0.5
                n_trend = max(analyst.total_calls // 2, 1)  # 估算趨勢次數
                update_kwargs["trend_accuracy"] = round(
                    old_ta + (1.0 if was_correct else 0.0 - old_ta) / (n_trend + 1), 4
                )
            elif market_cond == "RANGING":
                old_ra = analyst.range_accuracy or 0.5
                n_range = max(analyst.total_calls // 2, 1)
                update_kwargs["range_accuracy"] = round(
                    old_ra + (1.0 if was_correct else 0.0 - old_ra) / (n_range + 1), 4
                )

            self.db.update_analyst(name, **update_kwargs)

            self.db.add_learning_log(
                event_type="WEIGHT_UPDATE",
                description=f"{name}: weight {analyst.current_weight:.3f} -> {new_weight:.3f}",
                details={
                    "analyst": name,
                    "old_weight": analyst.current_weight,
                    "new_weight": new_weight,
                    "accuracy": new_accuracy,
                    "recent_7d": recent_7d,
                    "was_correct": was_correct,
                },
            )

            icon = "✅" if was_correct else "❌"
            changes.append(
                f"  {icon} {name}: {analyst.current_weight:.3f} → {new_weight:.3f} "
                f"(準確率 {new_accuracy * 100:.0f}%)"
            )

            logger.info(
                "Analyst %s: weight %.3f -> %.3f (accuracy=%.1f%%, 7d=%.1f%%)",
                name, analyst.current_weight, new_weight,
                new_accuracy * 100, recent_7d * 100,
            )

        return changes

    def _calc_recent_accuracy(self, analyst_name: str, days: int) -> float:
        """計算近 N 天的準確率"""
        with self.db.get_session() as s:
            from modules.database import AnalystCall
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            calls = (s.query(AnalystCall)
                     .filter(
                         AnalystCall.analyst_name == analyst_name,
                         AnalystCall.timestamp >= cutoff,
                         AnalystCall.was_correct.isnot(None),
                     ).all())

            if not calls:
                return 0.5  # 沒有數據時返回中性值

            correct = sum(1 for c in calls if c.was_correct == 1)
            return correct / len(calls)

    def _record_pattern(self, trade):
        """記錄本次交易的訊號模式"""
        try:
            # 解析技術指標
            tech = {}
            if trade.technical_signals:
                raw = trade.technical_signals
                if isinstance(raw, str):
                    raw = json.loads(raw)
                tech = raw

            # 解析分析師意見
            analysts = []
            if trade.analyst_opinions:
                raw = trade.analyst_opinions
                if isinstance(raw, str):
                    raw = json.loads(raw)
                analysts = raw if isinstance(raw, list) else []

            # 生成模式名稱
            analyst_names = sorted(set(a.get("analyst", "") for a in analysts if a.get("analyst")))
            consensus_key = "+".join(analyst_names) if analyst_names else "unknown"

            # 技術特徵
            tech_features = []
            technical = tech.get("technical", "")
            if "bullish" in str(technical).lower():
                tech_features.append("bullish_tech")
            if "bearish" in str(technical).lower():
                tech_features.append("bearish_tech")

            tech_key = "+".join(tech_features) if tech_features else "mixed"

            pattern_name = f"{consensus_key}|{tech_key}"

            win = trade.outcome == "WIN"
            profit = trade.profit_pct or 0

            self.db.upsert_pattern(
                pattern_name=pattern_name,
                conditions={
                    "analysts": analyst_names,
                    "technical": tech_key,
                    "direction": trade.direction,
                },
                win=win,
                profit=profit,
            )

        except Exception as e:
            logger.warning("Failed to record pattern: %s", e)

    def _analyze_patterns(self) -> list:
        """分析累積的訊號模式，回傳高勝率模式"""
        patterns = self.db.get_high_winrate_patterns(min_occurrences=3, min_winrate=0.5)

        if patterns:
            for p in patterns:
                self.db.add_learning_log(
                    event_type="PATTERN_FOUND",
                    description=f"High win-rate pattern: {p.pattern_name} ({p.win_rate:.0%}, n={p.occurrences})",
                    details={
                        "pattern": p.pattern_name,
                        "win_rate": p.win_rate,
                        "occurrences": p.occurrences,
                        "avg_profit": p.avg_profit,
                    },
                )

            logger.info("Pattern analysis: found %d high-winrate patterns", len(patterns))

        return patterns or []

    def _optimize_parameters(self) -> dict:
        """基於歷史數據優化策略參數，並即時套用到系統"""
        trades = self.db.get_closed_trades(limit=200)
        if len(trades) < self.param_opt_freq:
            return {}

        changes = {}
        trading_cfg = self.config.get("trading", {})
        limits = self.config.get("risk_limits", {})

        # ── 1. 優化最低信心門檻 ──
        confidence_buckets = {}
        for t in trades:
            if t.confidence is None:
                continue
            bucket = int(t.confidence // 10) * 10  # 50, 60, 70, 80, 90
            if bucket not in confidence_buckets:
                confidence_buckets[bucket] = {"wins": 0, "total": 0}
            confidence_buckets[bucket]["total"] += 1
            if t.outcome == "WIN":
                confidence_buckets[bucket]["wins"] += 1

        best_threshold = 75
        best_edge = 0
        for threshold in sorted(confidence_buckets.keys()):
            wins = sum(b["wins"] for c, b in confidence_buckets.items() if c >= threshold)
            total = sum(b["total"] for c, b in confidence_buckets.items() if c >= threshold)
            if total >= 5:
                win_rate = wins / total
                avg_profit = sum(
                    t.profit_pct or 0 for t in trades
                    if t.confidence and t.confidence >= threshold
                ) / total
                edge = win_rate * avg_profit
                if edge > best_edge:
                    best_edge = edge
                    best_threshold = threshold

        old_confidence = trading_cfg.get("min_confidence", 75)
        if best_threshold != old_confidence:
            changes["min_confidence"] = {"old": old_confidence, "new": best_threshold}

        # ── 2. 優化最低風報比 ──
        rr_buckets = {}
        for t in trades:
            if not hasattr(t, "risk_reward") or t.risk_reward is None:
                continue
            bucket = round(t.risk_reward, 1)
            if bucket not in rr_buckets:
                rr_buckets[bucket] = {"wins": 0, "total": 0, "profit": 0}
            rr_buckets[bucket]["total"] += 1
            rr_buckets[bucket]["profit"] += t.profit_pct or 0
            if t.outcome == "WIN":
                rr_buckets[bucket]["wins"] += 1

        if rr_buckets:
            best_rr = 2.0
            best_rr_edge = 0
            for rr_threshold in [1.0, 1.5, 2.0, 2.5, 3.0]:
                wins = sum(b["wins"] for rr, b in rr_buckets.items() if rr >= rr_threshold)
                total = sum(b["total"] for rr, b in rr_buckets.items() if rr >= rr_threshold)
                if total >= 5:
                    profit = sum(b["profit"] for rr, b in rr_buckets.items() if rr >= rr_threshold)
                    edge = profit / total
                    if edge > best_rr_edge:
                        best_rr_edge = edge
                        best_rr = rr_threshold

            old_rr = trading_cfg.get("min_risk_reward", 2.0)
            if best_rr != old_rr:
                changes["min_risk_reward"] = {"old": old_rr, "new": best_rr}

        # ── 套用變更 ──
        if changes:
            for param, vals in changes.items():
                # 更新運行中的 config
                trading_cfg[param] = vals["new"]

                # 更新 RiskManager 的對應屬性
                if self.risk and hasattr(self.risk, param):
                    # 確保不超過硬限制
                    if param == "max_position_size":
                        vals["new"] = min(vals["new"], limits.get("absolute_max_position", 5.0))
                    setattr(self.risk, param, vals["new"])

                logger.info("Parameter applied: %s = %s → %s", param, vals["old"], vals["new"])

        self.db.add_learning_log(
            event_type="PARAM_OPTIMIZED",
            description=f"Parameter optimization: {changes}" if changes else "No changes needed",
            details={
                "confidence_buckets": {str(k): v for k, v in confidence_buckets.items()},
                "changes": {k: v for k, v in changes.items()},
            },
        )

        if changes:
            logger.info("Parameter optimization applied %d changes", len(changes))
        else:
            logger.info("Parameter optimization: no changes needed")

        return changes

    def check_quick_feedback(self, trade, current_price: float) -> dict | None:
        """檢查交易的快速回饋點（5min/30min/1hr）

        Returns: {label, direction_correct, pnl_pct} 或 None（無新 checkpoint）
        """
        if not trade.timestamp or not trade.entry_price or not current_price:
            return None

        elapsed = (datetime.now(timezone.utc) - trade.timestamp).total_seconds()

        # 載入已檢查的 checkpoints
        existing = {}
        if trade.quick_feedback:
            try:
                existing = json.loads(trade.quick_feedback) if isinstance(
                    trade.quick_feedback, str) else trade.quick_feedback
            except (json.JSONDecodeError, TypeError):
                existing = {}

        checkpoints = [
            (300, "5min"),
            (1800, "30min"),
            (3600, "1hr"),
        ]

        for secs, label in checkpoints:
            if elapsed >= secs and label not in existing:
                # 判斷方向是否正確
                if trade.direction == "LONG":
                    direction_correct = current_price > trade.entry_price
                else:
                    direction_correct = current_price < trade.entry_price

                pnl_pct = abs(current_price - trade.entry_price) / trade.entry_price * 100
                if not direction_correct:
                    pnl_pct = -pnl_pct

                # 記錄到 DB
                existing[label] = {
                    "correct": direction_correct,
                    "pnl_pct": round(pnl_pct, 3),
                    "price": current_price,
                }
                self.db.update_trade(trade.id, quick_feedback=existing)

                logger.info(
                    "Quick feedback [%s] trade #%d %s: %s (pnl=%.3f%%)",
                    label, trade.id, trade.symbol,
                    "correct" if direction_correct else "wrong", pnl_pct,
                )
                return {"label": label, "direction_correct": direction_correct,
                        "pnl_pct": pnl_pct}

        return None

    def get_analyst_report(self) -> str:
        """生成分析師績效報告"""
        analysts = self.db.get_all_analysts()
        if not analysts:
            return "尚無分析師數據"

        lines = ["📊 分析師績效報告", "=" * 30]
        for a in sorted(analysts, key=lambda x: x.current_weight, reverse=True):
            accuracy_pct = a.accuracy * 100
            lines.append(
                f"\n{a.name}:\n"
                f"  權重: {a.current_weight:.3f}\n"
                f"  總體準確率: {accuracy_pct:.1f}% ({a.correct_calls}/{a.total_calls})\n"
                f"  近 7 日: {a.recent_7d_accuracy * 100:.1f}%\n"
                f"  近 30 日: {a.recent_30d_accuracy * 100:.1f}%"
            )

        return "\n".join(lines)

    @staticmethod
    def _format_duration(seconds: int) -> str:
        if seconds < 60:
            return f"{seconds}秒"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes}分鐘"
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours}小時 {mins}分鐘"
