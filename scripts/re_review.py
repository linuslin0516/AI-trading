"""
一次性腳本：重新覆盤所有覆盤失敗（空/N/A/error）的已關閉交易
用法: python scripts/re_review.py
"""
import json
import logging
import sys
import time

sys.path.insert(0, ".")

from utils.helpers import load_config
from modules.database import Database, Trade
from modules.ai_analyzer import AIAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def is_review_valid(review_raw) -> bool:
    """判斷覆盤結果是否有效"""
    if not review_raw:
        return False
    try:
        review = json.loads(review_raw) if isinstance(review_raw, str) else review_raw
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(review, dict):
        return False
    # 有 error 欄位 → API 錯誤返回
    if "error" in review:
        return False
    # 缺少 overall_score → 格式不完整
    if "overall_score" not in review:
        return False
    return True


def format_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def main():
    config = load_config()
    tz_name = config.get("schedule", {}).get("timezone", "UTC")
    db = Database(config["database"]["path"], tz_name=tz_name)
    ai = AIAnalyzer(config)

    # 查詢所有已關閉交易
    closed_trades = db.get_closed_trades(limit=500)
    logger.info("找到 %d 筆已關閉交易", len(closed_trades))

    # 篩選覆盤失敗的
    need_review = [t for t in closed_trades if not is_review_valid(t.review)]
    logger.info("其中 %d 筆覆盤無效，需要重新覆盤", len(need_review))

    if not need_review:
        logger.info("所有交易都已成功覆盤，無需操作")
        return

    # 列出需要重新覆盤的交易
    for t in need_review:
        logger.info(
            "  Trade #%d: %s %s %s profit=%.2f%%",
            t.id, t.symbol, t.direction, t.outcome or "?",
            t.profit_pct or 0,
        )

    # 逐筆重新覆盤
    success = 0
    failed = 0
    for i, trade in enumerate(need_review, 1):
        logger.info("=" * 50)
        logger.info("[%d/%d] 覆盤 Trade #%d: %s %s %s",
                     i, len(need_review), trade.id,
                     trade.symbol, trade.direction, trade.outcome or "?")

        try:
            analyst_opinions = trade.analyst_opinions or "N/A"
            technical_signals = trade.technical_signals or "{}"
            take_profit = trade.take_profit or "[]"
            ai_reasoning = trade.ai_reasoning or "N/A"

            # 截斷過長欄位
            if len(analyst_opinions) > 3000:
                analyst_opinions = analyst_opinions[:3000] + "\n...(截斷)"
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
                "hold_duration": format_duration(trade.hold_duration or 0),
                "outcome": trade.outcome,
                "profit_pct": trade.profit_pct,
                "analyst_opinions": analyst_opinions,
                "technical_signals": (
                    json.loads(technical_signals)
                    if isinstance(technical_signals, str)
                    else technical_signals
                ),
                "ai_reasoning": ai_reasoning,
                "quick_feedback": "N/A",
            }

            review = ai.review_trade(trade_data)

            if not review or "error" in review or "overall_score" not in review:
                logger.warning("覆盤失敗 Trade #%d: %s", trade.id, review)
                failed += 1
                continue

            # 保存有效覆盤
            db.update_trade(trade.id, review=review)
            score = review.get("overall_score", "?")
            lessons = review.get("lessons_learned", [])
            logger.info(
                "✅ Trade #%d 覆盤成功: 評分=%s/10, 教訓=%d條",
                trade.id, score, len(lessons),
            )
            success += 1

            # 避免 API rate limit
            time.sleep(2)

        except Exception as e:
            logger.error("❌ Trade #%d 覆盤異常: %s", trade.id, e)
            failed += 1

    logger.info("=" * 50)
    logger.info("覆盤完成: 成功=%d, 失敗=%d, 總計=%d", success, failed, len(need_review))


if __name__ == "__main__":
    main()
