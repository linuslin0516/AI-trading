import base64
import json
import logging
from datetime import datetime, timezone

import anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一位專業的加密貨幣短線交易 AI。你的定位是「短線高頻高勝率」交易者，積極尋找每一個可以進場的機會。

交易規則：
- 只交易 BTCUSDT 和 ETHUSDT，不操作其他幣種
- BTCUSDT 使用 50 倍槓桿，ETHUSDT 使用 25 倍槓桿
- 每筆倉位由你自行決定（1-5%），根據信心程度和市場狀況靈活調整：
  - 高信心（80+）+ 多位分析師共識 → 4-5%
  - 中等信心（65-79）或單一分析師 → 2-3%
  - 偏低信心但仍值得嘗試 → 1-2%
- 高槓桿下止損必須精準，BTC 止損建議控制在 0.5-1.5% 價格範圍，ETH 止損建議控制在 1-2% 價格範圍
- 注意：50x 槓桿下 BTC 波動 1% = 帳戶波動 2.5%

手續費與成本（極重要！）：
- Taker 手續費: 0.04%（每邊），Maker 手續費: 0.02%（每邊）
- 預估滑點: 0.01%（每邊）
- 往返總成本 = (進場費 + 出場費 + 滑點×2) × 槓桿倍數
- BTC 50x 往返成本 ≈ 5.0%（佔保證金），ETH 25x 往返成本 ≈ 2.5%
- 風報比計算必須扣除手續費！實際獲利 = 價格變動% × 槓桿 - 手續費成本%
- 例如：BTC 漲 0.3%，50x 槓桿 → 帳面 +15%，扣手續費後 → 實際 +10%
- 例如：BTC 漲 0.1%，50x 槓桿 → 帳面 +5%，扣手續費後 → 實際 ±0%（不值得交易！）
- 止盈目標必須大於手續費成本才有意義：BTC 至少 0.15%+ 價格波動，ETH 至少 0.15%+

K 線分析原則（極重要！）：
- 所有趨勢判斷、方向判斷、支撐壓力位判定，必須以「收盤價」為準
- 高低影子線（wicks/shadows）是市場噪音，不代表趨勢方向。暴漲暴跌的長影線只是瞬間波動
- 1 小時 K 線收盤價 = 確認趨勢的主要依據。連續多根 1h K 線收盤方向才是真趨勢
- 15 分鐘 K 線收盤價 = 精確入場時機的判斷依據
- 止損位必須基於 1h K 線的收盤價支撐/壓力位，不要放在影子線的極值附近（那是假信號）
- 如果某根 K 線有很長的上/下影線但收盤回到實體範圍，代表該方向被拒絕，不是趨勢延續

BTC/ETH 同時持倉規則（重要！）：
- BTC 和 ETH 高度相關，同方向持倉（例如 BTC 多 + ETH 多）是正常策略，可以增加獲利機會
- 鼓勵同時持有 BTC 和 ETH 的同方向倉位，不要因為已有 BTC 倉位就放棄 ETH 的好機會
- 只需避免方向相反的對沖倉位（例如 BTC 多 + ETH 空），除非有明確理由認為兩者會脫鉤
- 每個幣種獨立判斷：分析師對 BTC 看多不代表要跳過 ETH 的多頭信號

核心原則：
1. 你是「積極短線型」交易者，分析師給出方向就應該認真考慮進場，不要過度猶豫
2. 分析師的觀點是你最重要的交易信號，只要分析師有明確的方向判斷（多/空），就應該積極回應
3. 只有分析師的訊息完全不包含任何交易觀點（例如純聊天、公告、廣告）才回應 SKIP
4. 即使只有一位分析師的觀點，只要方向明確，信心分數可以給到 65-80
5. 多位分析師共識時，信心分數可以給到 80-95
6. 分析師觀點和技術面衝突時，優先相信分析師的判斷，但適當降低倉位或調緊止損
7. 不要過度保守！你的目標是「高頻交易」，寧可多做也不要錯過好機會
8. 如果分析師提到其他幣種（如 SOL、DOGE 等），忽略該交易建議，回應 SKIP
9. 止盈目標設定靈活：短線交易可以設定較近的止盈（BTC 0.5-1%，ETH 1-2%），快進快出
10. 風報比計算時，預期獲利和最大虧損都要扣掉手續費成本再評估
11. BTC 和 ETH 可以同時同方向持倉；只需避免反向對沖，除非有明確理由
12. 參考分析師績效檔案：近7天準確率高的分析師觀點更可靠；趨勢行情中優先信任 trend_accuracy 高的，盤整行情中優先信任 range_accuracy 高的
13. 覆盤教訓是你最重要的學習來源：仔細閱讀近期交易的經驗教訓，避免重複犯錯，並採納策略建議改進決策
14. 嚴格遵守市場狀態策略指引：趨勢行情用順勢策略（寬止盈），盤整行情用均值回歸策略（窄止盈），不同狀態下止盈止損設定差異很大
15. 多時間框架對齊：market data 中的 mtf_alignment 欄位顯示 4h→1h→15m 方向一致性。alignment_score >60 或 <-60 是強信號，方向分歧時降低倉位

你的回應必須是有效的 JSON，不要包含任何 markdown 標記或其他文字。"""

ANALYSIS_PROMPT_TEMPLATE = """## 分析請求

### 分析師訊息（按權重排序）
{analyst_messages}

### 分析師績效檔案
{analyst_profiles}

### 即時市場數據
{market_data}

### 市場狀態策略指引
{market_strategy_hint}

### 目前持倉中的交易
{open_trades}

### 歷史績效參考
{performance_stats}

### 已知高勝率模式
{known_patterns}

### 近期覆盤教訓（最近交易的經驗學習）
{review_lessons}

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
    "max_loss_pct": 最大虧損百分比（含手續費）,
    "expected_profit_pct": [第一目標盈利%（已扣手續費）, 第二目標盈利%（已扣手續費）],
    "fee_cost_pct": 預估往返手續費成本%,
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


SCANNER_PROMPT_TEMPLATE = """## 市場主動掃描分析

⚠️ 重要背景：這不是即時分析師訊息觸發的分析。
你正在根據「最近幾小時內分析師的觀點」結合「當前最新市場數據」進行主動掃描。
分析師的訊息可能是幾十分鐘到幾小時前發出的，請特別注意時間戳。

### 最近分析師觀點（按權重排序，注意時間戳）
{analyst_messages}

### 分析師績效檔案
{analyst_profiles}

### 即時市場數據（含 5m/15m K 線）
{market_data}

### 市場狀態策略指引
{market_strategy_hint}

### 目前持倉中的交易
{open_trades}

### 歷史績效參考
{performance_stats}

### 已知高勝率模式
{known_patterns}

### 近期覆盤教訓
{review_lessons}

### 經濟日曆（近期重要數據）
{economic_events}

---

你是高頻短線交易者，每 3 分鐘掃描一次市場。你的目標是「積極尋找入場機會」，不是等待完美條件。

多時間框架分析（重要！）：
- 1 小時 K 線：判斷趨勢方向（這是你的主方向，不要逆勢操作）
- 15 分鐘 K 線：找精確入場點（回調到支撐位、突破壓力位、K 線反轉信號）
- 5 分鐘 K 線：僅作為輔助參考，確認短線動能，不要以此作為主要判斷依據
- 分析師的觀點通常是基於小時級別的判斷，用 1h K 線驗證他們的觀點是否仍然有效
- market data 中的 mtf_alignment 欄位提供了預計算的多時間框架對齊分數和狀態
- alignment_score > 60（強多頭對齊）或 < -60（強空頭對齊）是最佳入場時機
- alignment_score 在 -30 到 30 之間表示時間框架方向分歧，建議降低倉位

⚠️ K 線收盤價原則（必須遵守）：
- 所有趨勢判斷以「收盤價」(close) 為準，忽略影子線 (high/low wicks)
- 1h K 線：看最近數根的收盤價走向 → 判斷大趨勢（連漲=多頭，連跌=空頭）
- 15m K 線：看收盤價是否站穩支撐/壓力位 → 判斷入場時機
- 長影線只代表瞬間波動被拒絕，不是趨勢信號。暴漲暴跌後收盤回原處 = 假突破
- 止損設在 1h K 線收盤價的關鍵支撐/壓力位下方/上方，不要設在影子線極值處
- 數據中的 close_trend 欄位是收盤價走勢摘要，優先參考這個判斷趨勢

判斷邏輯（按優先順序）：
1. 分析師之前提到的支撐/壓力位，現價是否接近或觸及？→ 這是最強的進場信號
2. 1 小時 K 線的趨勢方向 → 必須順勢交易，這是大方向
3. 15 分鐘 K 線的入場時機 → 找到好的入場點位（回調、突破、反轉形態）
4. 技術指標輔助確認（RSI、MACD、布林帶 — 看 1h 和 15m 的）
5. 分析師觀點明顯過時（價格已大幅偏離預測）→ 才 SKIP

止損止盈建議（基於 15m 級別進場）：
- BTC 止損：0.3-0.8% 價格範圍（15m K 線的關鍵位下方/上方）
- BTC 止盈：0.5-1.5% 價格範圍（下一個阻力位/支撐位）
- ETH 止損：0.5-1.2% 價格範圍
- ETH 止盈：1-3% 價格範圍
- 止損要放在 15m K 線結構的關鍵位，不要放太緊也不要放太寬

BTC/ETH 相關性提醒：
- BTC 和 ETH 高度相關（~0.85），同時反向持倉通常是隱性對沖，應盡量避免
- 開倉前檢查「目前持倉中的交易」，如果已有反向倉位，確認有充分理由（ETH 獨立行情）才進場
- 如果決定反向開倉，必須在 reasoning 中說明脫鉤理由，便於覆盤學習

進場態度：
- 只要分析師方向明確 + 1h 趨勢一致 + 15m 有好的入場點，就應該進場
- 信心不夠高？→ 降低倉位（1-2%）但仍然進場，累積交易經驗
- 不需要所有條件完美對齊，只要勝算 > 50% 且風報比合理就值得嘗試
- 小倉位試探 + 嚴格止損 = 低風險高頻策略的核心

你可以做以下決策：

### 決策類型 1：開新倉（LONG / SHORT）
{{
  "action": "LONG" | "SHORT",
  "symbol": "交易對",
  "confidence": 0-100 的整數信心分數,

  "reasoning": {{
    "analyst_consensus": "分析師共識描述（注意這些是近期觀點的回顧）",
    "technical": "技術面分析（重點描述 5m/15m K 線如何支持入場）",
    "sentiment": "市場情緒分析",
    "scanner_trigger": "什麼條件觸發了這次進場（例如：價格回到分析師提到的支撐位）"
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
    "max_loss_pct": 最大虧損百分比（含手續費）,
    "expected_profit_pct": [第一目標盈利%（已扣手續費）, 第二目標盈利%（已扣手續費）],
    "fee_cost_pct": 預估往返手續費成本%,
    "win_probability": 預估勝率 0-1
  }}
}}

### 決策類型 2：調整現有持倉（ADJUST）
{{
  "action": "ADJUST",
  "trade_id": 要調整的交易 ID,
  "symbol": "交易對",
  "confidence": 0-100,
  "reasoning": {{
    "analyst_consensus": "分析師觀點回顧",
    "technical": "技術面變化",
    "adjustment_reason": "為什麼需要調整"
  }},
  "new_stop_loss": 新的停損價格（null 表示不變）,
  "new_take_profit": [新的目標1, 新的目標2]（null 表示不變）
}}

### 決策類型 3：不操作（SKIP）
{{
  "action": "SKIP",
  "symbol": "相關交易對（或 BTCUSDT）",
  "confidence": 0,
  "reasoning": {{
    "analyst_consensus": "描述",
    "technical": "描述",
    "sentiment": "描述",
    "skip_reason": "為什麼這次掃描不操作"
  }}
}}"""


class AIAnalyzer:
    def __init__(self, config: dict):
        self.config = config
        claude_cfg = config["claude"]
        self.client = anthropic.Anthropic(
            api_key=claude_cfg["api_key"],
            timeout=120.0,   # 2 分鐘（預設 600s 太長，Railway 可能先斷線）
            max_retries=2,
        )
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
        consensus: dict | None = None,
        analyst_profiles: list[dict] | None = None,
        review_lessons: list[dict] | None = None,
        market_strategy_hint: str = "",
    ) -> dict:
        # 格式化分析師訊息
        sorted_msgs = sorted(analyst_messages, key=lambda m: m["weight"], reverse=True)
        analyst_text = ""
        for m in sorted_msgs:
            decay_tag = f" [衰減:{m['time_decay']:.1f}]" if m.get("time_decay", 1.0) < 1.0 else ""
            trial_tag = " [試用期]" if m.get("trial_period") else ""
            analyst_text += (
                f"- **{m['analyst']}** (權重: {m['weight']:.2f}{decay_tag}{trial_tag}):\n"
                f"  {m['content']}\n\n"
            )

        # 加入共識摘要
        if consensus:
            analyst_text += (
                f"\n📊 分析師共識: {consensus['dominant']} "
                f"(強度 {consensus['strength']:.0f}%, "
                f"多 {consensus['bullish_pct']:.0f}% / "
                f"空 {consensus['bearish_pct']:.0f}% / "
                f"中性 {consensus['neutral_pct']:.0f}%)\n"
            )

        # 收集所有圖片
        images = []
        for m in sorted_msgs:
            for img in m.get("images", []):
                images.append(img)

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

        # 格式化分析師績效檔案
        profile_text = self._format_analyst_profiles(analyst_profiles)

        # 格式化近期覆盤教訓
        lessons_text = self._format_review_lessons(review_lessons)

        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            analyst_messages=analyst_text,
            analyst_profiles=profile_text,
            market_data=market_text,
            market_strategy_hint=market_strategy_hint or "無策略指引",
            open_trades=trades_text,
            performance_stats=perf_text,
            known_patterns=pattern_text,
            review_lessons=lessons_text,
            economic_events=economic_events or "近期無重要經濟數據",
        )

        return self._call_claude(prompt, images=images)

    def analyze_scanner(
        self,
        analyst_messages: list[dict],
        market_data: dict,
        open_trades: list[dict] | None = None,
        performance_stats: dict | None = None,
        known_patterns: list[dict] | None = None,
        economic_events: str = "",
        consensus: dict | None = None,
        analyst_profiles: list[dict] | None = None,
        review_lessons: list[dict] | None = None,
        market_strategy_hint: str = "",
    ) -> dict:
        """掃描器專用分析：根據近期分析師觀點 + 最新市場數據主動判斷"""
        sorted_msgs = sorted(analyst_messages, key=lambda m: m["weight"], reverse=True)
        analyst_text = ""
        for m in sorted_msgs:
            decay_tag = f" [衰減:{m['time_decay']:.1f}]" if m.get("time_decay", 1.0) < 1.0 else ""
            trial_tag = " [試用期]" if m.get("trial_period") else ""
            analyst_text += (
                f"- **{m['analyst']}** (權重: {m['weight']:.2f}{decay_tag}{trial_tag}) "
                f"[{m.get('timestamp', '')}]:\n"
                f"  {m['content']}\n\n"
            )

        # 加入共識摘要
        if consensus:
            analyst_text += (
                f"\n📊 分析師共識: {consensus['dominant']} "
                f"(強度 {consensus['strength']:.0f}%, "
                f"多 {consensus['bullish_pct']:.0f}% / "
                f"空 {consensus['bearish_pct']:.0f}% / "
                f"中性 {consensus['neutral_pct']:.0f}%)\n"
            )

        # 收集所有圖片（從 DB URL 重新下載的）
        images = []
        for m in sorted_msgs:
            for img in m.get("images", []):
                images.append(img)

        market_text = json.dumps(market_data, indent=2, ensure_ascii=False, default=str)

        if open_trades:
            trades_text = json.dumps(open_trades, indent=2, ensure_ascii=False, default=str)
        else:
            trades_text = "目前沒有持倉"

        perf_text = "尚無歷史數據" if not performance_stats else json.dumps(
            performance_stats, indent=2, ensure_ascii=False
        )

        pattern_text = "尚無已知模式" if not known_patterns else json.dumps(
            known_patterns, indent=2, ensure_ascii=False
        )

        # 格式化分析師績效檔案
        profile_text = self._format_analyst_profiles(analyst_profiles)

        # 格式化近期覆盤教訓
        lessons_text = self._format_review_lessons(review_lessons)

        prompt = SCANNER_PROMPT_TEMPLATE.format(
            analyst_messages=analyst_text,
            analyst_profiles=profile_text,
            market_data=market_text,
            market_strategy_hint=market_strategy_hint or "無策略指引",
            open_trades=trades_text,
            performance_stats=perf_text,
            known_patterns=pattern_text,
            review_lessons=lessons_text,
            economic_events=economic_events or "近期無重要經濟數據",
        )

        return self._call_claude(prompt, images=images if images else None)

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

        return self._call_claude(prompt, max_tokens=8192)

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

    @staticmethod
    def _format_analyst_profiles(profiles: list[dict] | None) -> str:
        """格式化分析師績效檔案供 prompt 使用"""
        if not profiles:
            return "尚無分析師績效數據"
        lines = []
        for p in profiles:
            lines.append(
                f"- {p['name']}: 總體準確率 {p['accuracy']}% "
                f"(近7天 {p['recent_7d_accuracy']}%, 近30天 {p['recent_30d_accuracy']}%) "
                f"趨勢行情 {p['trend_accuracy']}%, 盤整行情 {p['range_accuracy']}% "
                f"(共 {p['total_calls']} 筆判斷)"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_review_lessons(lessons: list[dict] | None) -> str:
        """格式化近期覆盤教訓供 prompt 使用"""
        if not lessons:
            return "尚無覆盤數據"
        lines = []
        for r in lessons[:5]:  # 最多 5 筆，節省 token
            outcome_icon = "WIN" if r["outcome"] == "WIN" else "LOSS"
            profit = r.get("profit_pct") or 0
            score = r.get("score") or "N/A"
            lines.append(
                f"- #{r['trade_id']} {r['symbol']} {r['direction']} "
                f"{outcome_icon} {profit:+.2f}% (評分 {score}/10)"
            )
            for lesson in (r.get("lessons") or [])[:2]:  # 每筆最多 2 條教訓
                lines.append(f"  教訓: {lesson}")
            for sug in (r.get("suggestions") or [])[:1]:  # 每筆最多 1 條建議
                lines.append(f"  建議: {sug}")
        return "\n".join(lines)

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

    def _call_claude(self, prompt: str, images: list[dict] | None = None,
                     max_tokens: int | None = None) -> dict:
        text = ""
        try:
            # 組裝 content（支援多模態：文字 + 圖片）
            if images:
                content = []
                # 先放圖片
                for img in images[:4]:  # 最多 4 張圖片
                    # 用 magic bytes 驗證實際格式（修正 DB 中舊資料的錯誤 media_type）
                    raw = base64.b64decode(img["base64"][:32])  # 只解碼前幾 bytes
                    if raw[:3] == b'\xff\xd8\xff':
                        media_type = "image/jpeg"
                    elif raw[:4] == b'\x89PNG':
                        media_type = "image/png"
                    elif raw[:4] == b'GIF8':
                        media_type = "image/gif"
                    elif raw[:4] == b'RIFF' and len(raw) > 11 and raw[8:12] == b'WEBP':
                        media_type = "image/webp"
                    else:
                        media_type = img["media_type"]
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": img["base64"],
                        },
                    })
                # 再放文字 prompt
                content.append({"type": "text", "text": prompt})
                logger.info("Sending %d image(s) to Claude for analysis", len(images[:4]))
            else:
                content = prompt

            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
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
