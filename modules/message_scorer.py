import json
import logging

import anthropic

logger = logging.getLogger(__name__)

SCORING_PROMPT = """你是加密貨幣交易訊號品質評分器。請為以下分析師訊息評分 0-10，判斷其作為交易信號的價值。

評分標準：
- 8-10: 包含明確方向（多/空）、具體價位、止損止盈建議
- 6-7: 包含方向判斷或技術分析觀點，但缺少具體價位
- 4-5: 包含模糊的市場看法，有一定參考價值
- 2-3: 主要是閒聊但偶爾提到市場
- 0-1: 完全無關交易（廣告、公告、純聊天、表情符號）

訊息列表：
{messages}

請以 JSON 格式回應，包含每則訊息的評分：
{{
  "scores": [
    {{"index": 0, "score": 整數0-10, "reason": "簡短理由"}},
    ...
  ]
}}"""


class MessageScorer:
    def __init__(self, config: dict):
        self.config = config
        scoring_cfg = config.get("message_scoring", {})
        self.enabled = scoring_cfg.get("enabled", True)
        self.min_score = scoring_cfg.get("min_score", 4)
        self.model = scoring_cfg.get("model", "claude-haiku-4-5-20251001")

        claude_cfg = config.get("claude", {})
        self.client = anthropic.Anthropic(api_key=claude_cfg["api_key"])
        logger.info("MessageScorer initialized (model=%s, min_score=%d, enabled=%s)",
                     self.model, self.min_score, self.enabled)

    def score_messages(self, analyst_msgs: list[dict]) -> list[dict]:
        """為分析師訊息評分，過濾低品質訊息

        Args:
            analyst_msgs: 分析師訊息列表 [{analyst, content, weight, ...}]

        Returns:
            過濾後的訊息列表（附帶 quality_score）
        """
        if not self.enabled or not analyst_msgs:
            return analyst_msgs

        try:
            # 組裝訊息文本
            msg_lines = []
            for i, m in enumerate(analyst_msgs):
                msg_lines.append(f"[{i}] {m['analyst']}: {m['content'][:500]}")
            messages_text = "\n".join(msg_lines)

            prompt = SCORING_PROMPT.format(messages=messages_text)

            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()
            # 清理 markdown 包裹
            if text.startswith("```"):
                lines = text.split("\n")
                lines = lines[1:]
                while lines and lines[-1].strip() == "```":
                    lines.pop()
                text = "\n".join(lines).strip()

            result = json.loads(text)
            scores = {s["index"]: s for s in result.get("scores", [])}

            # 套用分數並過濾
            filtered = []
            for i, msg in enumerate(analyst_msgs):
                score_info = scores.get(i, {"score": 5, "reason": "no score"})
                score = score_info.get("score", 5)
                msg["quality_score"] = score

                if score >= self.min_score:
                    filtered.append(msg)
                else:
                    logger.info("Filtered out low-quality message from %s (score=%d): %s",
                                msg["analyst"], score, score_info.get("reason", ""))

            logger.info("Message scoring: %d/%d messages passed (min_score=%d)",
                        len(filtered), len(analyst_msgs), self.min_score)
            return filtered

        except Exception as e:
            logger.warning("Message scoring failed, passing all messages: %s", e)
            # 評分失敗時不阻擋，全部放行
            for msg in analyst_msgs:
                msg["quality_score"] = -1
            return analyst_msgs
