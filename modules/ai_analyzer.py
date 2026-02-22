import base64
import json
import logging
from datetime import datetime, timezone

import anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一位專業的加密貨幣短線交易 AI。你的定位是「精準入場、高勝率」交易者，追求每一筆交易都有精確的進場點位和明確的技術依據。目標是每天 2-3 筆高品質交易，而非高頻率交易。

交易規則：
- 只交易 BTCUSDT 和 ETHUSDT，不操作其他幣種
- 槓桿倍數由你自行決定（1-100x），根據分析師風格、行情波動和信心程度調整：
  - 參考主要信號來源的分析師風格：高槓桿短線型分析師（如三馬哥、大鏢客）→ 傾向高槓桿(50-100x)+小倉位(1-5%)；波段型分析師（如所長、舒琴）→ 傾向低槓桿(10-25x)+大倉位(5-20%)
  - 趨勢明確、信心高 → 可適當提高槓桿
  - 盤整或方向不確定 → 降低槓桿
  - 高波動時段（重大數據前後）→ 降低槓桿
- 每筆倉位由你自行決定（1-20%），必須同時考慮信心程度、止損距離和槓桿倍數：
  - 先決定技術上合理的止損位置（支撐/阻力位），再根據止損距離調整倉位和槓桿
  - 單筆最大潛在虧損應控制在帳戶的 1-2%
  - 計算：最大虧損% ≈ 止損距離% × 槓桿 × 倉位%
  - 止損遠 → 縮小倉位或降低槓桿
  - 止損近 → 可加大倉位或提高槓桿
- 止損位必須放在技術上有意義的位置，不要為了倉位大小而壓縮止損

手續費與成本（極重要！）：
- Taker 手續費: 0.04%（每邊），Maker 手續費: 0.02%（每邊）
- 預估滑點: 0.01%（每邊）
- 往返總成本 = (進場費 + 出場費 + 滑點×2) × 槓桿倍數
- 例如：100x 往返成本 ≈ 10.0%（佔保證金），50x ≈ 5.0%，25x ≈ 2.5%，10x ≈ 1.0%
- 風報比計算必須扣除手續費！實際獲利 = 價格變動% × 槓桿 - 手續費成本%
- 低槓桿時手續費佔比更低，更容易獲利；高槓桿時手續費佔比更高，需要更大的價格波動
- 止盈目標必須大於手續費成本才有意義

K 線分析原則（極重要！）：
- 趨勢方向判斷以「收盤價」為準：連續多根 K 線收盤價的方向 = 真趨勢
- 影線（wicks/shadows）用途不同，不能忽略：
  - 影線不代表趨勢方向，但代表「哪裡有買賣壓力」
  - 長下影線 = 該價位有強力買盤支撐（止損不要設在這個區間內，容易被掃）
  - 長上影線 = 該價位有強力賣壓（做多止盈可以參考這個位置）
  - 多根 K 線的影線密集區 = 流動性獵殺區（stop hunt zone），止損要避開
- 1 小時 K 線收盤價 = 確認趨勢的主要依據
- 15 分鐘 K 線收盤價 = 精確入場時機的判斷依據
- 止損設定：放在收盤價支撐/壓力位「外側」，同時參考影線極值避開流動性獵殺區
- 如果某根 K 線有很長的上/下影線但收盤回到實體範圍，代表該方向被拒絕（看影線方向做反向判斷）

持倉規則（重要！）：
- BTC 和 ETH 各自獨立判斷，互不影響。分析師對 BTC 看多不代表要跳過 ETH 的多頭信號
- 同幣種同方向只能持有 1 單，不可加倉
- 同幣種可以同時持有多空方向（多空並存），各自由系統獨立管理止盈止損
- 例如：BTC LONG 與 BTC SHORT 可以同時存在，代表兩個獨立的交易觀點
- 分析師的新方向信號不會自動平掉舊方向，兩邊各自依照止盈止損出場

核心原則：
1. 你是「精準型」交易者，每筆交易都要有明確的技術依據和精確的進場點位，不要勉強進場
2. 分析師的觀點是你最重要的交易信號，只要分析師有明確的方向判斷（多/空），就應該認真評估進場機會
3. 只有分析師的訊息完全不包含任何交易觀點（例如純聊天、公告、廣告）才回應 SKIP
4. 即使只有一位分析師的觀點，只要方向明確且技術面支持，信心分數可以給到 65-80
5. 多位分析師共識 + 技術面對齊時，信心分數可以給到 80-95
6. 分析師觀點和技術面衝突時，優先相信分析師的判斷，但降低槓桿和倉位
7. 不要過度保守，但也不要為了交易而交易。目標是每天 2-3 筆高品質交易，寧缺毋濫
8. 如果分析師提到其他幣種（如 SOL、DOGE 等），忽略該交易建議，回應 SKIP
9. 止盈目標設定靈活，根據槓桿調整：高槓桿用較近止盈快進快出，低槓桿可設較遠目標
10. 風報比計算時，預期獲利和最大虧損都要扣掉手續費成本再評估
11. BTC 和 ETH 可以同時同方向持倉；只需避免反向對沖，除非有明確理由
12. 參考分析師績效檔案：近7天準確率高的分析師觀點更可靠；趨勢行情中優先信任 trend_accuracy 高的，盤整行情中優先信任 range_accuracy 高的
13. 覆盤教訓是你最重要的學習來源：仔細閱讀近期交易的經驗教訓，避免重複犯錯，並採納策略建議改進決策
14. 嚴格遵守市場狀態策略指引：趨勢行情用順勢策略（寬止盈），盤整行情用均值回歸策略（窄止盈），不同狀態下止盈止損設定差異很大
15. 多時間框架對齊：market data 中的 mtf_alignment 欄位顯示 4h→1h→15m 方向一致性。alignment_score >60 或 <-60 是強信號，方向分歧時降低倉位
16. 單筆風險控管（最重要）：每筆交易虧損不得超過帳戶 1-2%。止損位置由技術面決定，倉位大小由止損距離反推。寧可倉位小也不要亂設止損。如果技術上合理的止損距離太遠導致倉位太小，寧可不做（SKIP）也不要硬進
17. 開了就不動：一旦進場，止盈止損由系統自動執行，你不能調整也不能主動平倉。同幣種可以同時持有多空方向（多空並存），各自獨立由系統管理

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
  "leverage": 槓桿倍數 (1-100, 根據分析師風格調整),

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
  "position_size": 建議倉位百分比 (1-20, 根據分析師風格和槓桿調整),
  "risk_reward": 風險報酬比,

  "risk_assessment": {{
    "max_loss_pct": 最大虧損百分比（含手續費）,
    "expected_profit_pct": [第一目標盈利%（已扣手續費）, 第二目標盈利%（已扣手續費）],
    "fee_cost_pct": 預估往返手續費成本%,
    "win_probability": 預估勝率 0-1
  }}
}}

### 決策類型 2：不操作（SKIP）
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

### 平倉後 4 小時價格走勢（1h K 線）
{post_close_price_action}

---

請進行深度覆盤分析。特別注意「平倉後價格走勢」：
- 如果止損被打到但價格隨後反轉回來 → 止損可能設太緊，被市場噪音洗出去
- 如果止盈到了但價格繼續大幅走 → 止盈可能設太保守
- 如果方向完全錯誤且持續虧損 → 進場判斷有問題

以 JSON 格式回應：

{{
  "timing_assessment": "進場時機評估",
  "exit_assessment": "出場時機評估（結合平倉後走勢分析：止損/止盈設定是否最優？）",
  "stop_loss_assessment": "停損設定是否合理（結合後續走勢：止損被掃後價格是否反轉？是否被市場噪音洗出？）",
  "target_assessment": "目標設定是否合理（結合後續走勢：止盈後價格是否繼續大幅走？目標是否太保守？）",
  "risk_management_assessment": "風險管理評估：倉位大小是否匹配止損距離？單筆最大虧損佔帳戶多少%？是否控制在1-2%以內？如果虧損過大，建議的倉位大小應該是多少？",

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


SIGNAL_PARSER_PROMPT = """你是一個加密貨幣跟單系統的訊號解析器。請從以下分析師訊息中提取「明確的交易指令」。

解析規則：
1. 只提取分析師「明確給出」的數字點位，不要自己猜測或補充
2. 止損（SL）必須存在；止盈至少要有一個（TP1 有就夠）才能執行 → 缺止損才 SKIP
3. 如果分析師說「平倉」「出掉」「止損出」「離場」「關倉」「空手」「跑了」等 → CLOSE
4. 兩個入場點：第一個是主力入場，第二個是加倉/補倉點（通常價格更深）
5. 幣種識別：BTC/比特幣/大餅/大B → BTCUSDT；ETH/乙太/姨太/以太 → ETHUSDT
6. 入場策略：有給具體數字但說「市價」「現在進」「現價」→ MARKET；有給具體點位 → LIMIT
7. 純聊天、廣告、技術分析分享（沒有明確進場數字和止損）→ SKIP
8. 如果平倉指令沒有指定幣種，從上下文判斷；實在判斷不了 → SKIP
9. 多位分析師同時發訊號時：不同幣種各自獨立列出；同幣種同方向合併為一個（取第一位的點位）；同幣種不同方向各自列出
10. 「第二止盈待定」「TP2 待定」「第二目標待定」→ 只填 TP1，take_profit 填 [tp1]，不需要 TP2
11. 「平倉多單，反手做空」「平多反空」「平空反多」→ 只輸出新方向的訊號（SHORT/LONG），不要額外產生 CLOSE 訊號
12. 「強平控制 X 萬 U」「爆倉控制」→ 這是分析師的倉位風控說明，忽略此部分，繼續解析其他點位
13. 「現價 xxx / 市價 xxx 附近 / 直接市價進」→ entry_1 price = xxx，strategy = MARKET

分析師訊息：
{messages}

以 JSON 格式回應（不要包含任何其他文字）：
{{
  "signals": [
    {{
      "action": "LONG" | "SHORT" | "CLOSE" | "SKIP",
      "symbol": "BTCUSDT" | "ETHUSDT",
      "entry_1": {{"price": 數字, "strategy": "LIMIT" | "MARKET"}},
      "entry_2": {{"price": 數字, "strategy": "LIMIT"}} | null,
      "stop_loss": 數字 | null,
      "take_profit": [tp1] | [tp1, tp2] | [tp1, tp2, tp3] | null,
      "skip_reason": "只在 SKIP 時填寫原因，其他填 null"
    }}
  ]
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

你是精準型短線交易者，每 30 分鐘掃描一次市場。你的目標是「找到精確的入場點位」，追求每筆交易都有高品質的技術依據。每天 2-3 筆高品質交易即可。

多時間框架分析（重要！）：
- 1 小時 K 線：判斷趨勢方向（這是你的主方向，不要逆勢操作）
- 15 分鐘 K 線：找精確入場點（回調到支撐位、突破壓力位、K 線反轉信號）
- 5 分鐘 K 線：僅作為輔助參考，確認短線動能，不要以此作為主要判斷依據
- 分析師的觀點通常是基於小時級別的判斷，用 1h K 線驗證他們的觀點是否仍然有效
- market data 中的 mtf_alignment 欄位提供了預計算的多時間框架對齊分數和狀態
- alignment_score > 60（強多頭對齊）或 < -60（強空頭對齊）是最佳入場時機
- alignment_score 在 -30 到 30 之間表示時間框架方向分歧，建議降低倉位

⚠️ K 線分析原則（必須遵守）：
- 趨勢判斷以「收盤價」(close) 為準：連續收盤方向 = 真趨勢
- 影線 (wicks) 用途：不用來判斷趨勢，但用來識別「買賣壓力區」和「流動性獵殺區」
  - 長下影線 = 該位置有買盤支撐 → 止損設在此下方更安全
  - 長上影線 = 該位置有賣壓 → 做多止盈可參考
  - 多根 K 線影線密集區 = stop hunt zone → 止損要避開此區間
- 1h K 線：看收盤價走向判斷大趨勢，看影線識別關鍵支撐壓力
- 15m K 線：看收盤價判斷入場時機，看影線設定精確的 SL/TP
- 止損設在收盤價支撐/壓力位外側，同時參考影線避開流動性獵殺區
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
- 追求精準入場：分析師方向明確 + 1h 趨勢一致 + 15m 有好的入場點位
- 信心不夠高？→ 降低槓桿和倉位，但好機會仍然值得進場
- 目標每天 2-3 筆高品質交易，不要為了交易而勉強進場
- 精準入場 + 合理槓桿 + 嚴格止損 = 高勝率策略的核心

你可以做以下決策：

### 決策類型 1：開新倉（LONG / SHORT）
{{
  "action": "LONG" | "SHORT",
  "symbol": "交易對",
  "confidence": 0-100 的整數信心分數,
  "leverage": 槓桿倍數 (1-100, 根據分析師風格調整),

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
  "position_size": 建議倉位百分比 (1-20, 根據分析師風格和槓桿調整),
  "risk_reward": 風險報酬比,

  "risk_assessment": {{
    "max_loss_pct": 最大虧損百分比（含手續費）,
    "expected_profit_pct": [第一目標盈利%（已扣手續費）, 第二目標盈利%（已扣手續費）],
    "fee_cost_pct": 預估往返手續費成本%,
    "win_probability": 預估勝率 0-1
  }}
}}

### 決策類型 2：不操作（SKIP）
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
            post_close_price_action=trade_data.get("post_close_price_action", "N/A"),
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

    def parse_signal(self, combined_text: str, images: list[dict] | None = None) -> list[dict]:
        """跟單模式：從分析師訊息中提取交易指令（可能回傳多個訊號）"""
        prompt = SIGNAL_PARSER_PROMPT.format(messages=combined_text)
        result = self._call_claude(prompt, images=images, max_tokens=800)

        if "error" in result:
            logger.warning("Signal parse failed: %s", result)
            return [{"action": "SKIP", "skip_reason": "解析失敗"}]

        signals = result.get("signals", [])
        if not signals:
            # 相容舊格式（單一 action 物件）
            if result.get("action") in ("LONG", "SHORT", "CLOSE", "SKIP"):
                return [result]
            logger.warning("Signal parse returned no signals: %s", result)
            return [{"action": "SKIP", "skip_reason": "解析失敗"}]

        valid = []
        for s in signals:
            if s.get("action") not in ("LONG", "SHORT", "CLOSE", "SKIP"):
                s = {"action": "SKIP", "skip_reason": "無效 action"}
            valid.append(s)
        return valid or [{"action": "SKIP", "skip_reason": "無訊號"}]

    @staticmethod
    def _format_analyst_profiles(profiles: list[dict] | None) -> str:
        """格式化分析師績效檔案供 prompt 使用"""
        if not profiles:
            return "尚無分析師績效數據"
        lines = []
        for p in profiles:
            style = p.get("style", "")
            style_text = f" | 風格: {style}" if style else ""
            lines.append(
                f"- {p['name']}: 總體準確率 {p['accuracy']}% "
                f"(近7天 {p['recent_7d_accuracy']}%, 近30天 {p['recent_30d_accuracy']}%) "
                f"趨勢行情 {p['trend_accuracy']}%, 盤整行情 {p['range_accuracy']}% "
                f"(共 {p['total_calls']} 筆判斷)"
                f"{style_text}"
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
