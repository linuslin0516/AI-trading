# DC Trading Bot - Railway 部署指南

## Step 1: 推送程式碼到 GitHub

```bash
cd dc-trading

# 初始化 git（如果還沒有的話）
git init
git add .
git commit -m "Initial commit"

# 在 GitHub 建立一個 private repo，然後：
git remote add origin https://github.com/你的帳號/dc-trading.git
git branch -M main
git push -u origin main
```

> 確認 `.env` 不在 repo 裡面（已在 `.gitignore`）

## Step 2: 建立 Railway 專案

1. 前往 https://railway.com 用 GitHub 帳號登入
2. 點 **New Project** → **Deploy from GitHub repo**
3. 選擇你的 `dc-trading` repo
4. Railway 會自動偵測 Python 專案並開始 build

## Step 3: 設定環境變數

在 Railway 專案頁面：

1. 點進你的 service → **Variables** 分頁
2. 點 **Raw Editor**，貼上：

```
DISCORD_TOKEN=你的Discord_Token
CLAUDE_API_KEY=你的Claude_API_Key
BINANCE_API_KEY=你的Binance_API_Key
BINANCE_API_SECRET=你的Binance_API_Secret
TELEGRAM_BOT_TOKEN=你的Telegram_Bot_Token
TELEGRAM_CHAT_ID=你的Telegram_Chat_ID
FINNHUB_API_KEY=你的Finnhub_API_Key
```

3. 點 **Apply** 儲存

## Step 4: 掛載 Volume（保存資料庫）

Railway 每次部署會重置檔案系統，需要掛 Volume 保存 SQLite 資料庫：

1. 在 service 頁面 → **Settings** 分頁
2. 往下找到 **Volumes** → **+ Mount Volume**
3. 設定：
   - **Mount Path**: `/app/data`
4. 點 **Apply**

這樣 `data/trades.db` 就會保存在持久化儲存中，重新部署不會遺失。

## Step 5: 確認部署

1. 回到 **Deployments** 分頁，確認 build 成功
2. 點進最新的 deployment → 查看 **Logs**
3. 應該會看到：

```
AI Trading Bot starting...
Database initialized: ./data/trades.db
EconomicCalendar initialized (source=ForexFactory)
Starting Discord listener...
```

## 日常管理

### 查看日誌
Railway 頁面 → Deployments → 點最新部署 → **View Logs**

### 更新程式碼
本地修改後 push 到 GitHub，Railway 會自動重新部署：

```bash
git add .
git commit -m "Update trading logic"
git push
```

### 手動重啟
Railway 頁面 → Deployments → **Redeploy**

### 暫停服務
Railway 頁面 → Settings → **Remove Service**（或刪除部署）

## 費用說明

- Railway Hobby Plan $5/月（需綁卡）
- 這個 bot 資源消耗很低（~256MB RAM），通常在額度內
- 超過的話約 $5-7/月

## 注意事項

- **Volume 很重要**：不掛 Volume 的話每次部署 DB 都會清空
- **環境變數**：所有 API Key 都在 Railway Variables 設定，不要推到 GitHub
- **自動部署**：push 到 main 分支就會自動部署，小心不要推壞的程式碼
- **日誌**：bot 的日誌可以直接在 Railway 網頁看到，也會透過 Telegram 通知你
