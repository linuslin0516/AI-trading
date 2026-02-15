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
        self.cooldown_minutes = limits.get("cooldown_minutes", 30)

        self._last_trade_time: datetime | None = None
        logger.info("RiskManager initialized")

    def check(self, decision: dict) -> RiskCheckResult:
        """執行所有風控檢查（模擬模式：全部只顯示不阻擋，讓 AI 多學習）"""
        result = RiskCheckResult()

        confidence = decision.get("confidence", 0)
        risk_reward = decision.get("risk_reward", 0)
        position_size = decision.get("position_size", 0)
        symbol = decision.get("symbol", "")
        direction = decision.get("action", "")
        open_trades = self.db.get_open_trades()
        today_trades = self.db.get_today_trades()
        today_pnl = self.db.get_today_pnl()
        consecutive_losses = self.db.get_today_consecutive_losses()

        # ── 硬性檢查（即使模擬也阻擋）──
        # 1. 信心分數（太低的信號沒有學習價值）
        result.add_check(
            "信心分數",
            confidence >= self.min_confidence,
            f"{confidence}% (最低 {self.min_confidence}%)",
        )

        # 2. 允許的交易對
        if self.allowed_symbols:
            result.add_check(
                "允許幣種",
                symbol in self.allowed_symbols,
                f"{symbol}" if symbol in self.allowed_symbols
                else f"{symbol} 不在允許列表",
            )

        # 3. 重複持倉（同幣種同方向不重複開）
        duplicate = any(
            t.symbol == symbol and t.direction == direction
            for t in open_trades
        )
        result.add_check(
            "重複持倉",
            not duplicate,
            f"已持有 {symbol} {direction}" if duplicate else "OK",
        )

        # ── 資訊顯示（不阻擋，供覆盤參考）──
        effective_max = min(self.max_position_size, self.absolute_max_position)
        info_checks = [
            ("風報比", f"{risk_reward:.2f} (參考 {self.min_risk_reward})"),
            ("倉位大小", f"{position_size}% (上限 {effective_max}%)"),
            ("持倉數量", f"{len(open_trades)}/{self.max_positions}"),
            ("今日單數", f"{len(today_trades)}/{self.max_daily_trades}"),
            ("今日盈虧", f"{today_pnl:+.2f}%"),
            ("連續虧損", f"連輸 {consecutive_losses} 次"),
        ]
        for name, detail in info_checks:
            result.add_check(name, True, detail)

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
            ):
                old = getattr(self, key)
                setattr(self, key, value)
                logger.info("Risk param updated: %s %s -> %s", key, old, value)

    def is_emergency_stop(self) -> bool:
        """模擬階段不觸發緊急停止"""
        return False
