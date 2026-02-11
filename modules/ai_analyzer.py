import json
import logging
from datetime import datetime, timezone

import anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一位專業的加密貨幣交易分析師 AI。你的任務是根據分析師的觀點和即時市場數據，做出精確的交易決策。

交易規則：
- 只交易 BTCUSDT 和 ETHUSDT，不操作其他幣種
- BTCUSDT 使用 50 倍槓桿，ETHUSDT 使用 25 倍槓桿
- 每筆固定 5% 倉位
- 高槓桿下止損必須精準，BTC 止損建議控制在 0.5-1.5% 價格範圍，ETH 止損建議控制在 1-2% 價格範圍
- 注意：50x 槓桿下 BTC 波動 1% = 帳戶波動 2.5%，務必嚴格控制風險

核心原則：
1. 你是「精選型」交易者，只挑最有把握的機會
2. 分析師的觀點是你的重要參考依據，必須認真解讀他們的訊息內容
3. 如果分析師的訊息不包含明確的交易方向或觀點，回應 SKIP
4. 即使只有一位分析師的觀點，只要搭配技術面驗證足夠強，也可以下單
5. 多位分析師共識時，可以給予更高的信心分數
6. 如果分析師觀點和技術面衝突，傾向 SKIP 觀望
7. 信心分數必須客觀 — 不確定就給低分，寧可錯過也不要錯做
8. 如果分析師提到其他幣種（如 SOL、DOGE 等），忽略該交易建議，回應 SKIP

你的回應必須是有效的 JSON，不要包含任何 markdown 標記或其他文字。"""

ANALYSIS_PROMPT_TEMPLATE = """## 分析請求

### 分析師訊息（按權重排序）
{analyst_messages}

### 即時市場數據
{market_data}

### 目前持倉中的交易
{open_trades}

### 歷史績效參考
{performance_stats}

### 已知高勝率模式
{known_patterns}

### 經濟日曆（近期重要數據）
{economic_events}

---

請根據以上資訊進行深度分析。特別注意：
- 如果接下來幾小時內有重要經濟數據公布（如 CPI、FOMC、非農），建議謹慎操作或 SKIP
- 如果剛有數據公布，根據「實際 vs 預期」判斷市場方向
- 高影響事件前後波動加大，需調整倉位大小和止損距離

你可以做以下決策：

### 決策類型 1：開新倉（LONG / SHORT）
如果沒有相關持倉，且分析師觀點 + 技術面支持開倉：

{{
  "action": "LONG" | "SHORT",
  "symbol": "交易對",
  "confidence": 0-100 的整數信心分數,

  "reasoning": {{
    "analyst_consensus": "分析師共識描述",
    "technical": "技術面分析",
    "sentiment": "市場情緒分析",
    "historical": "歷史相似情況參考"
  }},

  "entry": {{
    "price": 建議進場價格,
    "strategy": "LIMIT" | "MARKET",
    "reason": "進場策略理由"
  }},

  "stop_loss": 停損價格,
  "take_profit": [第一目標, 第二目標],
  "position_size": 建議倉位百分比 (0.5-5.0),
  "risk_reward": 風險報酬比,

  "risk_assessment": {{
    "max_loss_pct": 最大虧損百分比,
    "expected_profit_pct": [第一目標盈利%, 第二目標盈利%],
    "win_probability": 預估勝率 0-1
  }}
}}

### 決策類型 2：調整現有持倉（ADJUST）
如果已經有持倉，且分析師的新觀點建議調整止盈止損：

{{
  "action": "ADJUST",
  "trade_id": 要調整的交易 ID,
  "symbol": "交易對",
  "confidence": 0-100,

  "reasoning": {{
    "analyst_consensus": "分析師新觀點摘要",
    "technical": "技術面變化",
    "adjustment_reason": "為什麼需要調整"
  }},

  "new_stop_loss": 新的停損價格（null 表示不變）,
  "new_take_profit": [新的目標1, 新的目標2]（null 表示不變）
}}

### 決策類型 3：不操作（SKIP）
如果分析師訊息不包含明確方向、或信號不夠強：

{{
  "action": "SKIP",
  "symbol": "相關交易對",
  "confidence": 0,
  "reasoning": {{
    "analyst_consensus": "描述",
    "technical": "描述",
    "sentiment": "描述",
    "skip_reason": "為什麼選擇不操作"
  }}
}}"""

REVIEW_PROMPT_TEMPLATE = """## 交易覆盤請求

### 交易詳情
- 交易對：{symbol}
- 方向：{direction}
- 進場價：{entry_price}
- 出場價：{exit_price}
- 停損設定：{stop_loss}
- 目標設定：{take_profit}
- 倉位大小：{position_size}%
- 信心分數：{confidence}%
- 持倉時間：{hold_duration}
- 結果：{outcome} ({profit_pct}%)

### 當時的分析師判斷
{analyst_opinions}

### 當時的技術指標
{technical_signals}

### 當時的 AI 推理
{ai_reasoning}

---

請進行深度覆盤分析，以 JSON 格式回應：

{{
  "timing_assessment": "進場時機評估",
  "exit_assessment": "出場時機評估",
  "stop_loss_assessment": "停損設定是否合理",
  "target_assessment": "目標設定是否合理",

  "analyst_performance": [
    {{
      "name": "分析師名稱",
      "direction": "其判斷方向",
      "was_correct": true/false,
      "weight_adjustment": 權重調整建議 (-0.1 到 +0.1),
      "comment": "評語"
    }}
  ],

  "lessons_learned": [
    "經驗教訓 1",
    "經驗教訓 2"
  ],

  "strategy_suggestions": [
    "策略改進建議 1",
    "策略改進建議 2"
  ],

  "pattern_notes": "識別到的模式記錄",
  "overall_score": 1-10 的評分
}}"""


MORNING_BRIEFING_TEMPLATE = """## 每日早報 — {date}

### 過去 24 小時分析師觀點
{analyst_messages}

### 過去 24 小時 AI 決策記錄
{recent_decisions}

### 目前市場數據
{market_data}

### 目前持倉
{open_trades}

### 歷史績效
{performance_stats}

### 今日經濟日曆
{economic_events}

---

請產出一份簡潔的每日早報。特別注意今天有哪些重要經濟數據公布，提醒交易時需避開的時間段。

以 JSON 格式回應：

{{
  "market_overview": "整體市場概況（2-3 句話）",

  "analyst_summary": "分析師觀點整理摘要",

  "today_strategy": "今天的整體交易思路和策略方向",

  "key_levels": {{
    "BTC": {{"support": [支撐位], "resistance": [壓力位]}},
    "ETH": {{"support": [支撐位], "resistance": [壓力位]}}
  }},

  "watchlist": [
    {{
      "symbol": "交易對",
      "bias": "偏多 / 偏空 / 中性",
      "reason": "原因"
    }}
  ],

  "risk_notes": "今天需要注意的風險事項（包含經濟數據公布時間）",
  "economic_calendar_notes": "今日重要經濟數據提醒和預期影響",
  "confidence_level": "高 / 中 / 低"
}}"""

EVENING_SUMMARY_TEMPLATE = """## 每日晚報 — {date}

### 今日所有交易
{today_trades}

### 今日所有 AI 決策記錄（包含跳過、被拒絕、被取消的）
{today_decisions}

### 今日分析師觀點
{analyst_messages}

### 目前持倉
{open_trades}

### 今日績效
{performance_stats}

### 整體績效
{overall_stats}

### 今日經濟數據公布結果
{economic_events}

---

請產出一份每日交易總結報告。特別回顧今天公布的經濟數據對市場的影響。

以 JSON 格式回應：

{{
  "day_summary": "今天整體操作摘要（2-3 句話）",

  "trades_review": [
    {{
      "trade_id": 交易ID,
      "symbol": "交易對",
      "direction": "LONG/SHORT",
      "result": "結果描述",
      "comment": "簡短評語"
    }}
  ],

  "analyst_review": "今天分析師表現簡評",

  "lessons": ["今日經驗教訓"],

  "tomorrow_outlook": "明天展望和預期策略",

  "performance_note": "績效相關備註",

  "economic_data_review": "今日公布的經濟數據回顧及對市場的影響",

  "overall_score": 1-10
}}"""


class AIAnalyzer:
    def __init__(self, config: dict):
        self.config = config
        claude_cfg = config["claude"]
        self.client = anthropic.Anthropic(api_key=claude_cfg["api_key"])
        self.model = claude_cfg.get("model", "claude-sonnet-4-20250514")
        self.max_tokens = claude_cfg.get("max_tokens", 4096)
        self.temperature = claude_cfg.get("temperature", 0.7)
        logger.info("AIAnalyzer initialized (model=%s)", self.model)

    def analyze(
        self,
        analyst_messages: list[dict],
        market_data: dict,
        open_trades: list[dict] | None = None,
        performance_stats: dict | None = None,
        known_patterns: list[dict] | None = None,
        economic_events: str = "",
    ) -> dict:
        # 格式化分析師訊息
        sorted_msgs = sorted(analyst_messages, key=lambda m: m["weight"], reverse=True)
        analyst_text = ""
        for m in sorted_msgs:
            analyst_text += (
                f"- **{m['analyst']}** (權重: {m['weight']:.2f}):\n"
                f"  {m['content']}\n\n"
            )

        # 格式化市場數據
        market_text = json.dumps(market_data, indent=2, ensure_ascii=False, default=str)

        # 格式化持倉
        if open_trades:
            trades_text = json.dumps(open_trades, indent=2, ensure_ascii=False, default=str)
        else:
            trades_text = "目前沒有持倉"

        # 格式化績效統計
        perf_text = "尚無歷史數據" if not performance_stats else json.dumps(
            performance_stats, indent=2, ensure_ascii=False
        )

        # 格式化已知模式
        pattern_text = "尚無已知模式" if not known_patterns else json.dumps(
            known_patterns, indent=2, ensure_ascii=False
        )

        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            analyst_messages=analyst_text,
            market_data=market_text,
            open_trades=trades_text,
            performance_stats=perf_text,
            known_patterns=pattern_text,
            economic_events=economic_events or "近期無重要經濟數據",
        )

        return self._call_claude(prompt)

    def review_trade(self, trade_data: dict) -> dict:
        """平倉後 AI 覆盤"""
        prompt = REVIEW_PROMPT_TEMPLATE.format(
            symbol=trade_data.get("symbol", "N/A"),
            direction=trade_data.get("direction", "N/A"),
            entry_price=trade_data.get("entry_price", "N/A"),
            exit_price=trade_data.get("exit_price", "N/A"),
            stop_loss=trade_data.get("stop_loss", "N/A"),
            take_profit=trade_data.get("take_profit", "N/A"),
            position_size=trade_data.get("position_size", "N/A"),
            confidence=trade_data.get("confidence", "N/A"),
            hold_duration=trade_data.get("hold_duration", "N/A"),
            outcome=trade_data.get("outcome", "N/A"),
            profit_pct=trade_data.get("profit_pct", "N/A"),
            analyst_opinions=trade_data.get("analyst_opinions", "N/A"),
            technical_signals=json.dumps(
                trade_data.get("technical_signals", {}),
                indent=2, ensure_ascii=False,
            ),
            ai_reasoning=trade_data.get("ai_reasoning", "N/A"),
        )

        return self._call_claude(prompt)

    def generate_morning_briefing(
        self,
        analyst_messages: list[dict],
        market_data: dict,
        open_trades: list[dict] | None = None,
        performance_stats: dict | None = None,
        recent_decisions: list[dict] | None = None,
        economic_events: str = "",
    ) -> dict:
        """產出每日早報"""
        analyst_text = ""
        for m in analyst_messages:
            analyst_text += f"- **{m['analyst']}** [{m['timestamp']}]:\n  {m['content']}\n\n"

        if not analyst_text:
            analyst_text = "過去 24 小時沒有收到分析師訊息"

        decisions_text = self._format_decisions(recent_decisions)

        market_text = json.dumps(market_data, indent=2, ensure_ascii=False, default=str)

        if open_trades:
            trades_text = json.dumps(open_trades, indent=2, ensure_ascii=False, default=str)
        else:
            trades_text = "目前沒有持倉"

        perf_text = "尚無歷史數據" if not performance_stats else json.dumps(
            performance_stats, indent=2, ensure_ascii=False
        )

        prompt = MORNING_BRIEFING_TEMPLATE.format(
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            analyst_messages=analyst_text,
            recent_decisions=decisions_text,
            market_data=market_text,
            open_trades=trades_text,
            performance_stats=perf_text,
            economic_events=economic_events or "今日無重要經濟數據",
        )
        return self._call_claude(prompt)

    def generate_evening_summary(
        self,
        today_trades: list[dict],
        analyst_messages: list[dict],
        open_trades: list[dict] | None = None,
        performance_stats: dict | None = None,
        overall_stats: dict | None = None,
        today_decisions: list[dict] | None = None,
        economic_events: str = "",
    ) -> dict:
        """產出每日晚報"""
        if today_trades:
            trades_text = json.dumps(today_trades, indent=2, ensure_ascii=False, default=str)
        else:
            trades_text = "今天沒有執行任何交易"

        decisions_text = self._format_decisions(today_decisions)

        analyst_text = ""
        for m in analyst_messages:
            analyst_text += f"- **{m['analyst']}** [{m['timestamp']}]:\n  {m['content']}\n\n"

        if not analyst_text:
            analyst_text = "今天沒有收到分析師訊息"

        if open_trades:
            open_text = json.dumps(open_trades, indent=2, ensure_ascii=False, default=str)
        else:
            open_text = "目前沒有持倉"

        perf_text = "今天沒有已結束的交易" if not performance_stats else json.dumps(
            performance_stats, indent=2, ensure_ascii=False
        )

        overall_text = "尚無歷史數據" if not overall_stats else json.dumps(
            overall_stats, indent=2, ensure_ascii=False
        )

        prompt = EVENING_SUMMARY_TEMPLATE.format(
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            today_trades=trades_text,
            today_decisions=decisions_text,
            analyst_messages=analyst_text,
            open_trades=open_text,
            performance_stats=perf_text,
            overall_stats=overall_text,
            economic_events=economic_events or "今日無經濟數據公布",
        )
        return self._call_claude(prompt)

    def _format_decisions(self, decisions: list[dict] | None) -> str:
        """格式化 AI 決策記錄供 prompt 使用"""
        if not decisions:
            return "沒有決策記錄"

        lines = []
        for d in decisions:
            outcome_icons = {
                "EXECUTED": "✅ 已執行",
                "SKIP": "⏭️ 跳過",
                "REJECTED": "🚫 風控拒絕",
                "CANCELLED": "❌ 用戶取消",
            }
            outcome_str = outcome_icons.get(d["outcome"], d["outcome"])
            line = f"- [{d['timestamp']}] {d['action']} {d['symbol']} (信心 {d['confidence']}%) → {outcome_str}"

            if d.get("reasoning"):
                line += f"\n  推理: {d['reasoning']}"
            if d["outcome"] == "REJECTED" and d.get("risk_summary"):
                line += f"\n  風控: {d['risk_summary']}"
            if d["outcome"] == "CANCELLED" and d.get("cancel_reason"):
                line += f"\n  取消原因: {d['cancel_reason']}"

            lines.append(line)

        return "\n".join(lines)

    def _call_claude(self, prompt: str) -> dict:
        text = ""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            if not response.content:
                logger.error("Claude returned empty content")
                return {"action": "SKIP", "confidence": 0, "error": "Empty response"}

            text = response.content[0].text.strip()

            if not text:
                logger.error("Claude returned empty text")
                return {"action": "SKIP", "confidence": 0, "error": "Empty response"}

            # 清理可能的 markdown 包裹（例如 ```json\n{...}\n```）
            if text.startswith("```"):
                lines = text.split("\n")
                # 移除開頭的 ```json 或 ``` 行
                lines = lines[1:]
                # 移除結尾的 ``` 行
                while lines and lines[-1].strip() == "```":
                    lines.pop()
                text = "\n".join(lines).strip()

            if not text:
                logger.error("Text empty after markdown cleanup, raw response: %s",
                             response.content[0].text[:200])
                return {"action": "SKIP", "confidence": 0, "error": "Empty after cleanup"}

            result = json.loads(text)
            logger.info(
                "AI analysis complete: action=%s confidence=%s",
                result.get("action"), result.get("confidence"),
            )
            return result

        except json.JSONDecodeError as e:
            logger.error("Failed to parse AI response as JSON: %s\nRaw text (first 500 chars): %s",
                         e, text[:500])
            return {"action": "SKIP", "confidence": 0, "error": "JSON parse error"}
        except anthropic.APIError as e:
            logger.error("Claude API error: %s", e)
            return {"action": "SKIP", "confidence": 0, "error": str(e)}
        except Exception as e:
            logger.error("Unexpected error in AI analysis: %s", e)
            return {"action": "SKIP", "confidence": 0, "error": str(e)}
