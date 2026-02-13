import json
import logging
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text, create_engine
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    symbol = Column(String(20), nullable=False)
    direction = Column(String(10), nullable=False)  # LONG / SHORT

    # 決策記錄
    analyst_opinions = Column(Text)   # JSON
    technical_signals = Column(Text)  # JSON
    market_data = Column(Text)        # JSON
    ai_reasoning = Column(Text)
    confidence = Column(Float)

    # 交易執行
    entry_price = Column(Float)
    exit_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Text)  # JSON list
    position_size = Column(Float)
    leverage = Column(Integer, default=1)

    # Binance 訂單
    entry_order_id = Column(String(50))
    exit_order_id = Column(String(50))

    # 結果
    profit_pct = Column(Float)
    profit_usd = Column(Float)
    hold_duration = Column(Integer)  # seconds
    outcome = Column(String(20))  # WIN / LOSS / BREAKEVEN

    # AI 覆盤
    review = Column(Text)  # JSON
    status = Column(String(20), default="PENDING")
    # PENDING -> OPEN -> PARTIAL_CLOSE -> CLOSED

    closed_at = Column(DateTime)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "symbol": self.symbol,
            "direction": self.direction,
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stop_loss": self.stop_loss,
            "take_profit": json.loads(self.take_profit) if self.take_profit else [],
            "position_size": self.position_size,
            "profit_pct": self.profit_pct,
            "profit_usd": self.profit_usd,
            "outcome": self.outcome,
            "status": self.status,
        }


class Analyst(Base):
    __tablename__ = "analysts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, nullable=False)
    total_calls = Column(Integer, default=0)
    correct_calls = Column(Integer, default=0)
    accuracy = Column(Float, default=0.0)
    current_weight = Column(Float, default=1.0)

    trend_accuracy = Column(Float, default=0.0)
    range_accuracy = Column(Float, default=0.0)

    recent_7d_accuracy = Column(Float, default=0.0)
    recent_30d_accuracy = Column(Float, default=0.0)

    last_updated = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class AnalystCall(Base):
    """每筆分析師判斷的詳細記錄"""
    __tablename__ = "analyst_calls"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(Integer, nullable=False)
    analyst_name = Column(String(50), nullable=False)
    direction = Column(String(10))  # LONG / SHORT / NEUTRAL
    message_content = Column(Text)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    was_correct = Column(Integer)  # 1=correct, 0=wrong, NULL=pending


class AnalystMessage(Base):
    """所有接收到的分析師訊息（不僅是觸發交易的）"""
    __tablename__ = "analyst_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analyst_name = Column(String(50), nullable=False)
    channel = Column(String(100))
    content = Column(Text, nullable=False)
    images = Column(Text)  # JSON: [{"url": "...", "media_type": "image/png"}, ...]
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class AIDecision(Base):
    """記錄每一次 AI 決策（包含 SKIP、被風控拒絕、用戶取消）"""
    __tablename__ = "ai_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    symbol = Column(String(20))
    action = Column(String(10))       # LONG / SHORT / SKIP / ADJUST
    confidence = Column(Float)
    reasoning = Column(Text)          # JSON - AI 的完整推理
    outcome = Column(String(20))      # EXECUTED / SKIP / REJECTED / CANCELLED
    risk_summary = Column(Text)       # 風控拒絕原因
    cancel_reason = Column(Text)      # 用戶取消原因
    analyst_sources = Column(Text)    # JSON - 觸發的分析師列表
    trade_id = Column(Integer)        # 關聯到 trades 表（僅 EXECUTED）


class LearningLog(Base):
    __tablename__ = "learning_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    event_type = Column(String(50))
    # WEIGHT_UPDATE, PATTERN_FOUND, PARAM_OPTIMIZED, REVIEW
    description = Column(Text)
    details = Column(Text)  # JSON


class SignalPattern(Base):
    """已發現的高勝率訊號組合"""
    __tablename__ = "signal_patterns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pattern_name = Column(String(100))
    conditions = Column(Text)  # JSON
    occurrences = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    avg_profit = Column(Float, default=0.0)
    last_seen = Column(DateTime)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Database:
    def __init__(self, db_path: str = "./data/trades.db", tz_name: str = "UTC"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self._migrate(self.engine)
        self._Session = sessionmaker(bind=self.engine)
        self._tz = ZoneInfo(tz_name)
        logger.info("Database initialized: %s", db_path)

    @staticmethod
    def _migrate(engine):
        """自動遷移：為已存在的資料表新增缺少的欄位"""
        from sqlalchemy import inspect, text
        inspector = inspect(engine)
        # analyst_messages: 新增 images 欄位
        if "analyst_messages" in inspector.get_table_names():
            cols = [c["name"] for c in inspector.get_columns("analyst_messages")]
            if "images" not in cols:
                with engine.begin() as conn:
                    conn.execute(text(
                        "ALTER TABLE analyst_messages ADD COLUMN images TEXT"
                    ))
                logger.info("Migrated: added 'images' column to analyst_messages")

    def get_session(self) -> Session:
        return self._Session()

    # ── Trade operations ──

    def create_trade(self, **kwargs) -> Trade:
        with self.get_session() as s:
            for field in ("analyst_opinions", "technical_signals",
                          "market_data", "take_profit", "review"):
                if field in kwargs and not isinstance(kwargs[field], str):
                    kwargs[field] = json.dumps(kwargs[field], ensure_ascii=False)
            trade = Trade(**kwargs)
            s.add(trade)
            s.commit()
            s.refresh(trade)
            logger.info("Trade created: #%d %s %s", trade.id, trade.symbol, trade.direction)
            return trade

    def update_trade(self, trade_id: int, **kwargs):
        with self.get_session() as s:
            for field in ("analyst_opinions", "technical_signals",
                          "market_data", "take_profit", "review"):
                if field in kwargs and not isinstance(kwargs[field], str):
                    kwargs[field] = json.dumps(kwargs[field], ensure_ascii=False)
            s.query(Trade).filter(Trade.id == trade_id).update(kwargs)
            s.commit()

    def get_trade(self, trade_id: int) -> Trade | None:
        with self.get_session() as s:
            return s.query(Trade).filter(Trade.id == trade_id).first()

    def get_open_trades(self) -> list[Trade]:
        with self.get_session() as s:
            return (s.query(Trade)
                    .filter(Trade.status.in_(["OPEN", "PARTIAL_CLOSE"]))
                    .all())

    def get_recent_trades(self, days: int = 7) -> list[Trade]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with self.get_session() as s:
            return (s.query(Trade)
                    .filter(Trade.timestamp >= cutoff)
                    .order_by(Trade.timestamp.desc())
                    .all())

    def get_closed_trades(self, limit: int = 100) -> list[Trade]:
        with self.get_session() as s:
            return (s.query(Trade)
                    .filter(Trade.status == "CLOSED")
                    .order_by(Trade.timestamp.desc())
                    .limit(limit)
                    .all())

    def get_today_trades(self) -> list[Trade]:
        # 用設定的時區計算「今天零點」，再轉回 UTC 查詢
        local_now = datetime.now(self._tz)
        local_midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_start_utc = local_midnight.astimezone(timezone.utc)
        with self.get_session() as s:
            return (s.query(Trade)
                    .filter(Trade.timestamp >= today_start_utc)
                    .all())

    # ── Analyst operations ──

    def get_or_create_analyst(self, name: str, initial_weight: float = 1.0) -> Analyst:
        with self.get_session() as s:
            analyst = s.query(Analyst).filter(Analyst.name == name).first()
            if not analyst:
                analyst = Analyst(name=name, current_weight=initial_weight)
                s.add(analyst)
                s.commit()
                s.refresh(analyst)
            return analyst

    def update_analyst(self, name: str, **kwargs):
        with self.get_session() as s:
            kwargs["last_updated"] = datetime.now(timezone.utc)
            s.query(Analyst).filter(Analyst.name == name).update(kwargs)
            s.commit()

    def get_all_analysts(self) -> list[Analyst]:
        with self.get_session() as s:
            return s.query(Analyst).all()

    def get_analyst_weight(self, name: str) -> float:
        with self.get_session() as s:
            analyst = s.query(Analyst).filter(Analyst.name == name).first()
            return analyst.current_weight if analyst else 1.0

    # ── Analyst call operations ──

    def record_analyst_call(self, trade_id: int, analyst_name: str,
                            direction: str, message: str):
        with self.get_session() as s:
            call = AnalystCall(
                trade_id=trade_id,
                analyst_name=analyst_name,
                direction=direction,
                message_content=message,
            )
            s.add(call)
            s.commit()

    def get_analyst_calls_for_trade(self, trade_id: int) -> list[AnalystCall]:
        with self.get_session() as s:
            return (s.query(AnalystCall)
                    .filter(AnalystCall.trade_id == trade_id)
                    .all())

    def mark_analyst_call_result(self, trade_id: int, analyst_name: str,
                                 was_correct: bool):
        with self.get_session() as s:
            (s.query(AnalystCall)
             .filter(AnalystCall.trade_id == trade_id,
                     AnalystCall.analyst_name == analyst_name)
             .update({"was_correct": 1 if was_correct else 0}))
            s.commit()

    # ── Analyst message storage ──

    def save_analyst_message(
        self, analyst_name: str, channel: str, content: str,
        images: list[dict] | None = None,
    ):
        with self.get_session() as s:
            msg = AnalystMessage(
                analyst_name=analyst_name,
                channel=channel,
                content=content,
                images=json.dumps(images, ensure_ascii=False) if images else None,
            )
            s.add(msg)
            s.commit()

    def get_recent_analyst_messages(self, hours: int = 24) -> list[AnalystMessage]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        with self.get_session() as s:
            return (s.query(AnalystMessage)
                    .filter(AnalystMessage.timestamp >= cutoff)
                    .order_by(AnalystMessage.timestamp.desc())
                    .all())

    def get_recent_analyst_messages_for_symbols(
        self, hours: int = 4, keywords: list[str] | None = None
    ) -> list[AnalystMessage]:
        """取得最近 N 小時內提及特定幣種關鍵字的分析師訊息"""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        with self.get_session() as s:
            messages = (
                s.query(AnalystMessage)
                .filter(AnalystMessage.timestamp >= cutoff)
                .order_by(AnalystMessage.timestamp.desc())
                .all()
            )
            if keywords is None:
                return messages
            result = []
            for m in messages:
                text_upper = m.content.upper()
                for kw in keywords:
                    if kw.upper() in text_upper:
                        result.append(m)
                        break
            return result

    def get_today_analyst_messages(self) -> list[AnalystMessage]:
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        with self.get_session() as s:
            return (s.query(AnalystMessage)
                    .filter(AnalystMessage.timestamp >= today_start)
                    .order_by(AnalystMessage.timestamp.asc())
                    .all())

    # ── AI Decision log ──

    def save_ai_decision(self, decision: dict, outcome: str,
                         analyst_names: list[str] | None = None,
                         cancel_reason: str = "",
                         trade_id: int | None = None):
        """儲存 AI 的每一次決策"""
        reasoning = decision.get("reasoning", {})
        if not isinstance(reasoning, str):
            reasoning = json.dumps(reasoning, ensure_ascii=False)

        with self.get_session() as s:
            record = AIDecision(
                symbol=decision.get("symbol", ""),
                action=decision.get("action", "SKIP"),
                confidence=decision.get("confidence", 0),
                reasoning=reasoning,
                outcome=outcome,
                risk_summary=decision.get("_risk_summary", ""),
                cancel_reason=cancel_reason,
                analyst_sources=json.dumps(analyst_names or [], ensure_ascii=False),
                trade_id=trade_id,
            )
            s.add(record)
            s.commit()
            logger.info("AI decision saved: %s %s → %s",
                        record.action, record.symbol, outcome)

    def get_recent_decisions(self, hours: int = 24) -> list[AIDecision]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        with self.get_session() as s:
            return (s.query(AIDecision)
                    .filter(AIDecision.timestamp >= cutoff)
                    .order_by(AIDecision.timestamp.desc())
                    .all())

    def get_today_decisions(self) -> list[AIDecision]:
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        with self.get_session() as s:
            return (s.query(AIDecision)
                    .filter(AIDecision.timestamp >= today_start)
                    .order_by(AIDecision.timestamp.asc())
                    .all())

    # ── Learning log ──

    def add_learning_log(self, event_type: str, description: str,
                         details: dict | None = None):
        with self.get_session() as s:
            log = LearningLog(
                event_type=event_type,
                description=description,
                details=json.dumps(details, ensure_ascii=False) if details else None,
            )
            s.add(log)
            s.commit()

    # ── Signal patterns ──

    def upsert_pattern(self, pattern_name: str, conditions: dict,
                       win: bool, profit: float):
        with self.get_session() as s:
            pat = (s.query(SignalPattern)
                   .filter(SignalPattern.pattern_name == pattern_name)
                   .first())
            if pat:
                pat.occurrences += 1
                if win:
                    pat.wins += 1
                pat.win_rate = pat.wins / pat.occurrences
                pat.avg_profit = (
                    (pat.avg_profit * (pat.occurrences - 1) + profit)
                    / pat.occurrences
                )
                pat.last_seen = datetime.now(timezone.utc)
            else:
                pat = SignalPattern(
                    pattern_name=pattern_name,
                    conditions=json.dumps(conditions, ensure_ascii=False),
                    occurrences=1,
                    wins=1 if win else 0,
                    win_rate=1.0 if win else 0.0,
                    avg_profit=profit,
                    last_seen=datetime.now(timezone.utc),
                )
                s.add(pat)
            s.commit()

    def get_high_winrate_patterns(self, min_occurrences: int = 5,
                                  min_winrate: float = 0.6) -> list[SignalPattern]:
        with self.get_session() as s:
            return (s.query(SignalPattern)
                    .filter(SignalPattern.occurrences >= min_occurrences,
                            SignalPattern.win_rate >= min_winrate)
                    .order_by(SignalPattern.win_rate.desc())
                    .all())

    # ── Statistics ──

    def get_performance_stats(self, days: int | None = None) -> dict:
        trades = (self.get_recent_trades(days) if days
                  else self.get_closed_trades(500))
        closed = [t for t in trades if t.status == "CLOSED"]
        if not closed:
            return {
                "total": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_profit_pct": 0, "total_profit_usd": 0,
                "avg_profit_pct": 0, "max_drawdown": 0,
            }

        wins = [t for t in closed if t.outcome == "WIN"]
        losses = [t for t in closed if t.outcome == "LOSS"]
        profits = [t.profit_pct or 0 for t in closed]

        # simple max drawdown
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in profits:
            cumulative += p
            peak = max(peak, cumulative)
            max_dd = min(max_dd, cumulative - peak)

        return {
            "total": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(closed) * 100 if closed else 0,
            "total_profit_pct": sum(profits),
            "total_profit_usd": sum(t.profit_usd or 0 for t in closed),
            "avg_profit_pct": sum(profits) / len(closed) if closed else 0,
            "max_drawdown": max_dd,
        }

    def get_today_pnl(self) -> float:
        trades = self.get_today_trades()
        return sum(t.profit_pct or 0 for t in trades if t.status == "CLOSED")

    def get_today_consecutive_losses(self) -> int:
        """計算今日從最近一筆往回數的連續虧損次數"""
        local_now = datetime.now(self._tz)
        local_midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_start = local_midnight.astimezone(timezone.utc)
        with self.get_session() as s:
            trades = (s.query(Trade)
                      .filter(Trade.timestamp >= today_start,
                              Trade.status == "CLOSED")
                      .order_by(Trade.timestamp.desc())
                      .all())

        consecutive = 0
        for t in trades:
            if t.outcome == "LOSS":
                consecutive += 1
            else:
                break  # 遇到非虧損就停止計算
        return consecutive
