# DC Trading - AI 自動加密貨幣交易系統

## 專案概述

DC Trading 是一套基於 AI 的全自動加密貨幣合約交易系統。系統監聽 Discord 分析師頻道的交易訊號，透過 Claude AI 深度分析後結合即時市場數據進行決策，在 Binance Futures 上自動執行交易，並透過 Telegram 即時通知用戶。系統具備完整的學習引擎，能從每筆交易中學習並持續優化策略。

---

## 系統架構

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Discord     │────▶│  Decision     │────▶│  Binance Futures │
│  Listener    │     │  Engine       │     │  Testnet         │
│  (7 頻道)    │     │  + AI 分析    │     │  (下單/監控)     │
└─────────────┘     └──────┬───────┘     └────────┬────────┘
                           │                       │
┌─────────────┐     ┌──────▼───────┐     ┌────────▼────────┐
│  Market      │────▶│  Risk        │     │  Position        │
│  Scanner     │     │  Manager     │     │  Monitor         │
│  (每 3 分鐘) │     │  (風控閘門)  │     │  (每 30 秒)      │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
┌─────────────┐     ┌──────────────┐     ┌────────▼────────┐
│  Economic    │     │  Learning     │◀───│  Telegram        │
│  Calendar    │     │  Engine       │     │  Notifier        │
│  (ForexFactory)│   │  (AI 覆盤)   │     │  (通知+指令)     │
└─────────────┘     └──────────────┘     └─────────────────┘
```

---

## 技術棧

### 核心語言與運行環境

| 技術 | 版本 | 用途 |
|------|------|------|
| Python | 3.11.9 | 主要開發語言 |
| asyncio | 內建 | 非同步事件驅動架構 |
| SQLite | 內建 | 輕量級持久化資料庫 |

### Python 依賴套件

| 套件 | 版本 | 用途 |
|------|------|------|
| `anthropic` | 0.42.0 | Claude AI API 客戶端 |
| `discord.py-self` | 2.1.0 | Discord Self-Bot（用戶模式監聽） |
| `python-telegram-bot` | 20.7 | Telegram Bot 通知與指令處理 |
| `python-binance` | 1.0.19 | Binance API（實際使用自訂 HTTP wrapper） |
| `sqlalchemy` | 2.0.25 | ORM 資料庫操作 |
| `pandas` | 2.1.4 | 數據處理與分析 |
| `numpy` | 1.26.2 | 數值運算 |
| `requests` | 2.31.0 | HTTP 請求（市場數據、Binance API） |
| `aiohttp` | 3.9.1 | 非同步 HTTP（Discord/Telegram 輪詢） |
| `pyyaml` | 6.0.1 | YAML 設定檔解析 |
| `python-dotenv` | 1.0.0 | 環境變數載入 |

### 外部 API 與服務

| API | 端點 | 用途 |
|-----|------|------|
| Claude AI (Anthropic) | `api.anthropic.com/v1/messages` | AI 交易分析、覆盤、晨報/晚報 |
| Binance Futures Testnet | `testnet.binancefuture.com` | 合約交易執行（下單/持倉/掛單） |
| Binance Market Data | `data-api.binance.vision` | 即時行情、K 線、技術指標（主網數據） |
| Discord Gateway | WebSocket | 監聽分析師頻道訊息 |
| Telegram Bot API | Bot API | 用戶通知、交易確認、指令控制 |
| ForexFactory | `nfs.faireconomy.media` | 經濟日曆（USD 高影響事件） |

### 部署平台

| 技術 | 用途 |
|------|------|
| Railway.com | 雲端部署平台 |
| NIXPACKS | 自動建置（Railway 預設） |
| Volume Mount | `/app/data` 持久化存儲（5 GB） |
| GitHub | 原始碼管理，push 自動觸發部署 |

---

## 專案結構

```
dc-trading/
├── main.py                          # 主程式入口 - 非同步事件循環
├── config.yaml                      # 完整設定檔
├── requirements.txt                 # Python 依賴
├── runtime.txt                      # Python 版本指定 (3.11.9)
├── Procfile                         # Railway 啟動指令
├── railway.json                     # Railway 部署設定
├── .env.example                     # 環境變數範本
│
├── modules/                         # 核心模組
│   ├── ai_analyzer.py              # Claude AI 整合（分析/覆盤/報告）
│   ├── binance_trader.py           # Binance Futures API 封裝
│   ├── database.py                 # SQLAlchemy ORM 模型與操作
│   ├── decision_engine.py          # 訊號處理與決策管線
│   ├── discord_listener.py         # Discord 自走式監聽器
│   ├── economic_calendar.py        # 經濟日曆整合
│   ├── learning_engine.py          # AI 學習引擎
│   ├── market_data.py              # 市場數據獲取與處理
│   └── telegram_notifier.py        # Telegram 通知與指令
│
├── utils/                           # 工具模組
│   ├── helpers.py                  # 設定載入、日誌設定、格式化
│   └── risk_manager.py             # 風控系統
│
├── data/                            # 持久化數據（Railway Volume）
│   └── trades.db                   # SQLite 資料庫
│
└── logs/                            # 系統日誌
    └── system.log                  # 滾動日誌（100MB × 5）
```

---

## 模組詳細說明

### 1. Discord Listener (`discord_listener.py`)

- 使用 `discord.py-self`（Self-Bot 用戶模式）監聽 7 個分析師頻道
- 支援多語言關鍵字偵測（BTC/比特幣/大餅、ETH/以太/姨太）
- 訊息緩衝 60 秒後批次觸發分析
- 記錄所有分析師訊息（含圖片 URL）到資料庫

### 2. AI Analyzer (`ai_analyzer.py`)

- 模型：`claude-sonnet-4-20250514`
- 三種分析模式：
  - **交易分析**：完整市場數據 + 分析師觀點 → LONG/SHORT/SKIP/ADJUST
  - **交易覆盤**：平倉後分析入場/出場時機、分析師判斷正確性
  - **晨報/晚報**：每日市場摘要與策略建議
- 核心原則：
  - 所有趨勢判斷以**收盤價**為準（忽略影子線噪音）
  - 1 小時 K 線收盤價 = 確認趨勢
  - 15 分鐘 K 線收盤價 = 精確入場
  - 手續費必須納入風報比計算（BTC 50x ≈ 5% 往返成本）

### 3. Market Data (`market_data.py`)

- 多時間週期 K 線：5m(30根)、15m(40根)、1h(48根)、4h(30根)、1d(14根)
- 技術指標計算：RSI、MACD、EMA、SMA
- 收盤價趨勢摘要：方向、動量、支撐/壓力位（純收盤價）
- 市場統計：24h 成交量、資金費率、多空比
- 數據來源：Binance 主網公開 API

### 4. Decision Engine (`decision_engine.py`)

- 協調整個決策管線：
  1. 偵測提及的幣種
  2. 獲取市場數據（K 線 + 指標 + 統計）
  3. 獲取經濟日曆事件
  4. 查詢當前持倉
  5. 取得歷史勝率模式
  6. 呼叫 AI 分析
  7. 風控驗證
  8. 返回最終決策

### 5. Risk Manager (`risk_manager.py`)

**軟限制**（AI 可調整）：
- 最低信心分數：60%
- 最低風報比：1.5x
- 最大倉位：5%
- 每日最多交易：20 筆
- 每日最大虧損：15%
- 最大連續虧損：3 筆

**硬限制**（不可覆蓋）：
- 絕對最大倉位：5%
- 絕對最大日虧：20%
- 緊急停止：總虧損 40%

**其他檢查**：
- 冷卻時間（同幣種 5 分鐘）
- 重複持倉（同幣種同方向）
- 允許幣種白名單

### 6. Binance Trader (`binance_trader.py`)

- 自訂 HTTP wrapper（HMAC-SHA256 簽名）
- 支援 MARKET / LIMIT 訂單
- 槓桿設定：BTC 50x、ETH 25x
- SL/TP 訂單管理：
  - SL：STOP_MARKET（closePosition 全平）
  - TP1：TAKE_PROFIT_MARKET（平 50% 數量）
  - TP2：TAKE_PROFIT_MARKET（closePosition 全平）
- 持倉監控（每 30 秒）：
  - 批次查詢 Binance 持倉（單次 API 呼叫）
  - 偵測完全平倉 → 區分 SL/TP/強平/異常
  - 偵測部分平倉 → TP1 通知
  - Binance 不可用時回退到本地價格檢查
- 訂單管理：查詢歷史、取消殘留掛單

### 7. Telegram Notifier (`telegram_notifier.py`)

**連線池**：20 個並行連線（HTTPXRequest），避免 Pool Timeout

**可用指令**：
| 指令 | 功能 |
|------|------|
| `/status` | 系統狀態 |
| `/positions` | 當前持倉詳情 |
| `/pnl` | 績效總覽 |
| `/test_trade` | 測試交易 |
| `/close <id>` | 平倉指定交易 |
| `/close_all` | 平掉所有持倉 |
| `/fix_tp [id]` | 重設止盈止損掛單 |
| `/orders [symbol]` | Binance 訂單歷史 |
| `/cancel_orders <symbol>` | 取消殘留掛單 |
| `/stop` | 緊急停止 |
| `/help` | 指令說明 |

**通知類型**：
- 交易訊號（30 秒倒數 + 執行/取消按鈕）
- 入場確認
- TP1 部分止盈通知
- 平倉通知 + AI 覆盤
- 強平警告 (💀)
- 異常關閉警告 (❓)
- 風控拒絕通知
- 學習事件通知
- 每日晨報 / 晚報

### 8. Learning Engine (`learning_engine.py`)

**學習流程**（每次平倉觸發）：
1. **Testnet 價格偏差檢查**：比對主網價格，偏差 >5% 跳過學習
2. **AI 覆盤**：分析入場/出場時機、分析師判斷正確性
3. **分析師權重更新**：
   - 正確判斷 → 提高權重
   - 錯誤判斷 → 降低權重
   - 加權：整體準確率 70% + 近 7 日準確率 30%
   - 權重範圍：0.5 ~ 2.0
4. **模式記錄**：分析師組合 + 技術特徵 → 勝率統計
5. **批量學習**（自動觸發）：
   - 每 20 筆 → 分析高勝率模式
   - 每 50 筆 → 優化策略參數（信心門檻、風報比等）

### 9. Economic Calendar (`economic_calendar.py`)

- 來源：ForexFactory（免費 API）
- 過濾 USD 經濟事件
- 快取 2 小時（避免頻繁請求）
- 提供給 AI 做交易決策參考
- 高影響事件期間可能影響交易策略

---

## 資料庫結構（SQLite + SQLAlchemy ORM）

### 核心資料表

| 資料表 | 用途 | 主要欄位 |
|--------|------|----------|
| `trades` | 交易記錄 | symbol, direction, entry/exit_price, SL, TP, profit_pct, outcome, review |
| `analysts` | 分析師資料 | name, weight, accuracy, 7d/30d 準確率 |
| `analyst_calls` | 分析師歸因 | trade_id, analyst_name, was_correct |
| `analyst_messages` | Discord 訊息 | analyst_name, channel, content, images |
| `ai_decisions` | AI 決策記錄 | symbol, action, confidence, reasoning, outcome |
| `learning_logs` | 學習事件 | event_type, description, details |
| `signal_patterns` | 訊號模式 | pattern_name, occurrences, win_rate, avg_profit |

### 交易狀態流

```
PENDING → OPEN → PARTIAL_CLOSE → CLOSED
                                    │
                              outcome: WIN / LOSS / BREAKEVEN
```

---

## 交易工作流程

### A. 訊號進場流程

```
分析師 Discord 訊息
    ↓
60 秒緩衝收集
    ↓
偵測幣種 (BTC/ETH)
    ↓
獲取市場數據 (K 線 + 指標 + 統計)
    ↓
獲取經濟日曆
    ↓
Claude AI 分析 → LONG / SHORT / SKIP / ADJUST
    ↓
風控檢查 (信心/風報比/倉位/日虧損...)
    ↓
Telegram 通知 (30 秒倒數 + 確認按鈕)
    ↓
用戶確認 → Binance 下單 + 設定 SL/TP
```

### B. 持倉監控流程

```
每 30 秒查詢 Binance 持倉
    ↓
偵測倉位變化：
  ├── 完全平倉 (qty=0) → 判斷 SL/TP/強平/異常
  ├── 部分平倉 (qty↓50%) → TP1 通知
  └── 正常 → 計算未實現盈虧
    ↓
平倉 → 清理殘留掛單 → 更新 DB → AI 覆盤 → 學習
```

### C. 市場掃描器

```
每 3 分鐘主動掃描
    ↓
查詢近 4 小時分析師訊息
    ↓
過濾未持倉幣種
    ↓
觸發同一分析管線
    ↓
標記為 "Scanner 觸發"
```

---

## 部署設定

### Railway 部署

- **建置器**：NIXPACKS（自動偵測 Python）
- **啟動指令**：`python main.py`（worker process）
- **Volume**：掛載 `/app/data`（5 GB，存放 SQLite DB）
- **環境變數**：
  - `DISCORD_TOKEN` - Discord 用戶 Token
  - `CLAUDE_API_KEY` - Anthropic API Key
  - `TELEGRAM_BOT_TOKEN` - Telegram Bot Token
  - `TELEGRAM_CHAT_ID` - 通知目標 Chat ID
  - `BINANCE_API_KEY` - Binance Testnet API Key
  - `BINANCE_API_SECRET` - Binance Testnet API Secret

### 日誌系統

- 滾動檔案：`./logs/system.log`（100 MB × 5 備份）
- 同時輸出到 console（Railway Logs 可見）
- 啟動時印出 data 目錄內容（確認 Volume 掛載）

---

## 風險控制機制

| 層級 | 機制 | 閾值 |
|------|------|------|
| 單筆 | 最低信心分數 | ≥ 60% |
| 單筆 | 最低風報比 | ≥ 1.5x |
| 單筆 | 最大倉位 | ≤ 5% |
| 日級 | 每日最多交易 | ≤ 20 筆 |
| 日級 | 每日最大虧損 | ≤ 15%（軟）/ 20%（硬） |
| 日級 | 連續虧損限制 | ≤ 3 筆 |
| 帳戶 | 緊急停止 | 總虧損 > 40% |
| 操作 | 冷卻時間 | 同幣種 5 分鐘 |
| 操作 | 重複持倉 | 同幣種同方向禁止 |
| 學習 | Testnet 偏差保護 | 主網偏差 > 5% 跳過學習 |

---

## 交易參數

| 參數 | BTC | ETH |
|------|-----|-----|
| 槓桿 | 50x | 25x |
| Taker 手續費 | 0.04% | 0.04% |
| Maker 手續費 | 0.02% | 0.02% |
| 滑點預估 | 0.01% | 0.01% |
| 往返成本（Taker） | ~5% | ~2.5% |
| 最小價格波動 | ≥ 0.15% | ≥ 0.15% |

---

## 定時任務

| 任務 | 時間 | 內容 |
|------|------|------|
| 晨報 | 每日 08:00 | 市場概覽、分析師摘要、今日策略、關鍵價位 |
| 晚報 | 每日 22:00 | 交易回顧、分析師績效、教訓、明日展望 |
| 市場掃描 | 每 3 分鐘 | 主動掃描分析師觀點 + 市場機會 |
| 持倉監控 | 每 30 秒 | 偵測 SL/TP/強平，計算未實現盈虧 |

---

## 監控的 Discord 頻道

| 頻道名稱 | 初始權重 | 說明 |
|----------|---------|------|
| 大鏢客 | 1.0 | 分析師頻道 |
| 三馬哥合約 | 1.0 | 合約分析 |
| 幣圈所長課堂 | 1.0 | 教學 + 分析 |
| 舒琴 | 1.0 | 分析師 |
| VIVI-BTC | 1.0 | BTC 專屬分析 |
| VIVI-ETH | 1.0 | ETH 專屬分析 |
| VIVI-短線 | 1.0 | 短線交易 |

*權重由學習引擎根據歷史準確率動態調整（範圍 0.5 ~ 2.0）*

---

## 關鍵設計決策

1. **非同步事件驅動**：所有模組使用 asyncio，支援並行處理 Discord 監聽、Telegram 指令、持倉監控、市場掃描
2. **分析師優先策略**：分析師觀點是主要交易訊號，技術分析為輔助確認
3. **收盤價原則**：所有趨勢判斷以 K 線收盤價為準，忽略影子線噪音
4. **手續費意識**：所有盈虧和風報比計算都扣除完整往返手續費
5. **學習保護**：Testnet 價格偏差 >5% 時跳過學習，避免錯誤數據污染
6. **多層風控**：軟限制（AI 可調）+ 硬限制（不可覆蓋），確保資金安全
7. **優雅降級**：任何外部 API 失敗時有 fallback 機制，系統不會崩潰
