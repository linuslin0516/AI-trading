import logging
from datetime import datetime, timedelta, timezone

from modules.database import Database

logger = logging.getLogger(__name__)


class RiskCheckResult:
    def __init__(self):
        self.passed = True
        self.checks: list[dict] = []

    def add_check(self, name: str, passed: bool, detail: str):
        self.checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            self.passed = False

    def summary(self) -> str:
        lines = []
        for c in self.checks:
            icon = "pass" if c["passed"] else "FAIL"
            lines.append(f"[{icon}] {c['name']}: {c['detail']}")
        return "\n".join(lines)


class RiskManager:
    def __init__(self, config: dict, db: Database):
        self.config = config
        self.db = db

        trading = config.get("trading", {})
        limits = config.get("risk_limits", {})

        # AI 可調整的軟限制
        self.min_confidence = trading.get("min_confidence", 75)
        self.min_risk_reward = trading.get("min_risk_reward", 2.0)
        self.max_position_size = trading.get("max_position_size", 5.0)
        self.max_positions = trading.get("max_positions", 2)
        self.max_daily_trades = trading.get("max_daily_trades", 5)
        self.max_daily_loss = trading.get("max_daily_loss", 15.0)
        self.max_consecutive_losses = trading.get("max_consecutive_losses", 3)
        self.allowed_symbols = trading.get("allowed_symbols", [])

        # 硬限制（不可修改）
        self.absolute_max_position = limits.get("absolute_max_position", 5.0)
        self.absolute_max_daily_loss = limits.get("absolute_max_daily_loss", 10.0)
        self.emergency_stop_loss = limits.get("emergency_stop_loss", 20.0)
        self.cooldown_minutes = limits.get("cooldown_minutes", 30)

        self._last_trade_time: datetime | None = None
        logger.info("RiskManager initialized")

    def check(self, decision: dict) -> RiskCheckResult:
        """執行所有風控檢查"""
        result = RiskCheckResult()

        confidence = decision.get("confidence", 0)
        risk_reward = decision.get("risk_reward", 0)
        position_size = decision.get("position_size", 0)

        # 1. 信心分數
        result.add_check(
            "信心分數",
            confidence >= self.min_confidence,
            f"{confidence}% (最低 {self.min_confidence}%)",
        )

        # 2. 風報比
        result.add_check(
            "風報比",
            risk_reward >= self.min_risk_reward,
            f"{risk_reward:.2f} (最低 {self.min_risk_reward})",
        )

        # 3. 倉位大小（軟限制 + 硬限制）
        effective_max = min(self.max_position_size, self.absolute_max_position)
        result.add_check(
            "倉位大小",
            position_size <= effective_max,
            f"{position_size}% (上限 {effective_max}%)",
        )

        # 4. 持倉數量
        open_trades = self.db.get_open_trades()
        result.add_check(
            "持倉數量",
            len(open_trades) < self.max_positions,
            f"{len(open_trades)}/{self.max_positions}",
        )

        # 5. 今日下單次數
        today_trades = self.db.get_today_trades()
        today_trade_count = len(today_trades)
        result.add_check(
            "今日單數",
            today_trade_count < self.max_daily_trades,
            f"{today_trade_count}/{self.max_daily_trades}",
        )

        # 6. 今日虧損
        today_pnl = self.db.get_today_pnl()
        effective_max_loss = min(self.max_daily_loss, self.absolute_max_daily_loss)
        result.add_check(
            "今日虧損",
            abs(today_pnl) < effective_max_loss if today_pnl < 0 else True,
            f"{today_pnl:+.2f}% (上限 -{effective_max_loss}%)",
        )

        # 6.5 連續虧損檢查（連輸 N 次當日停止）
        consecutive_losses = self.db.get_today_consecutive_losses()
        result.add_check(
            "連續虧損",
            consecutive_losses < self.max_consecutive_losses,
            f"連輸 {consecutive_losses} 次 (上限 {self.max_consecutive_losses})"
            if consecutive_losses >= self.max_consecutive_losses
            else f"連輸 {consecutive_losses} 次，尚可交易",
        )

        # 7. 總虧損緊急停止
        stats = self.db.get_performance_stats()
        total_pnl = stats.get("total_profit_pct", 0)
        result.add_check(
            "總虧損",
            total_pnl > -self.emergency_stop_loss,
            f"{total_pnl:+.2f}% (緊急停止 -{self.emergency_stop_loss}%)",
        )

        # 7. 冷卻時間
        cooldown_ok = True
        if self._last_trade_time:
            elapsed = (datetime.now(timezone.utc) - self._last_trade_time).total_seconds()
            cooldown_ok = elapsed >= self.cooldown_minutes * 60
        result.add_check(
            "冷卻時間",
            cooldown_ok,
            f"已等待 {int(elapsed // 60)}m" if self._last_trade_time and not cooldown_ok
            else "OK",
        )

        # 8. 允許的交易對
        symbol = decision.get("symbol", "")
        direction = decision.get("action", "")
        if self.allowed_symbols:
            result.add_check(
                "允許幣種",
                symbol in self.allowed_symbols,
                f"{symbol}" if symbol in self.allowed_symbols
                else f"{symbol} 不在允許列表 {self.allowed_symbols}",
            )

        # 9. 是否重複方向
        duplicate = any(
            t.symbol == symbol and t.direction == direction
            for t in open_trades
        )
        result.add_check(
            "重複持倉",
            not duplicate,
            f"已持有 {symbol} {direction}" if duplicate else "OK",
        )

        if result.passed:
            logger.info("Risk check PASSED for %s %s", symbol, direction)
        else:
            failed = [c["name"] for c in result.checks if not c["passed"]]
            logger.warning("Risk check FAILED: %s", ", ".join(failed))

        return result

    def record_trade_time(self):
        self._last_trade_time = datetime.now(timezone.utc)

    def update_soft_limits(self, **kwargs):
        """AI 學習引擎可以更新軟限制"""
        for key, value in kwargs.items():
            if hasattr(self, key) and key not in (
                "absolute_max_position",
                "absolute_max_daily_loss",
                "emergency_stop_loss",
            ):
                old = getattr(self, key)
                setattr(self, key, value)
                logger.info("Risk param updated: %s %s -> %s", key, old, value)

    def is_emergency_stop(self) -> bool:
        stats = self.db.get_performance_stats()
        total_pnl = stats.get("total_profit_pct", 0)
        if total_pnl <= -self.emergency_stop_loss:
            logger.critical(
                "EMERGENCY STOP: total PnL %.2f%% <= -%.2f%%",
                total_pnl, self.emergency_stop_loss,
            )
            return True
        return False
