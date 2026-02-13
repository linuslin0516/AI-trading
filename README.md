# DC Trading - AI Automated Crypto Trading System

An AI-powered automated cryptocurrency futures trading system that monitors Discord analyst channels, analyzes signals with Claude AI, executes trades on Binance Futures, and continuously learns from outcomes.

## Overview

DC Trading bridges the gap between human trading analysts and automated execution. It listens to multiple Discord analyst channels in real-time, synthesizes their opinions with live market data through Claude AI, applies strict risk management, and executes trades on Binance Futures — all while learning from every trade to improve over time.

### How It Works

```
Discord Analysts ──▶ Message Buffer (60s) ──▶ Claude AI Analysis
                                                      │
                                              Risk Manager Gate
                                                      │
                                          Telegram Confirmation
                                           (30s countdown)
                                                      │
                                         Binance Futures Execution
                                           (SL + TP1/TP2)
                                                      │
                                      Position Monitor (every 30s)
                                                      │
                                       AI Post-Trade Review + Learning
```

## Features

### Signal Processing
- Monitors 7 Discord analyst channels simultaneously
- 60-second message buffering for batch analysis
- Multi-language keyword detection (English/Chinese)
- Analyst weight system — better analysts have more influence

### AI Decision Engine
- Claude AI (Sonnet 4) analyzes analyst opinions + market data
- Multi-timeframe K-line analysis (5m / 15m / 1h / 4h / 1d)
- Technical indicators: RSI, MACD, EMA, SMA
- Economic calendar integration (ForexFactory)
- Close-price-based analysis (ignores wick noise)
- Fee-aware RR calculations (accounts for leverage costs)

### Trade Execution
- Binance Futures with configurable leverage (BTC 50x, ETH 25x)
- Automatic SL/TP order placement
- Two-target TP system: TP1 closes 50%, TP2 closes remaining
- 30-second Telegram confirmation with Execute/Cancel buttons

### Position Monitoring
- Real-time position tracking via Binance API (every 30s)
- Detects: stop loss, take profit, TP1 partial close, liquidation
- Automatic cleanup of orphaned orders
- Fallback to price-based monitoring if API unavailable

### Risk Management
- Confidence threshold (min 60%)
- Risk-reward ratio check (min 1.5x)
- Position sizing limits (max 5% per trade)
- Daily loss limits (soft 15% / hard 20%)
- Consecutive loss protection (3 losses = stop for the day)
- Emergency stop (40% total account loss)
- Cooldown between trades (5 min)
- Duplicate position prevention

### Learning Engine
- AI post-trade review on every closed trade
- Dynamic analyst weight adjustment based on accuracy
- Signal pattern discovery (every 20 trades)
- Strategy parameter optimization (every 50 trades)
- Testnet price deviation protection — skips learning if exit price deviates >5% from mainnet

### Market Scanner
- Proactive market scanning every 3 minutes
- Combines recent analyst opinions with live market data
- Identifies trading opportunities without waiting for new signals

### Telegram Bot Commands
| Command | Description |
|---------|-------------|
| `/status` | System status |
| `/positions` | Current open positions |
| `/pnl` | Performance overview |
| `/orders [symbol]` | Binance order history |
| `/close <id>` | Close a specific trade |
| `/close_all` | Close all positions |
| `/fix_tp [id]` | Resync SL/TP orders |
| `/cancel_orders <symbol>` | Cancel orphaned orders |
| `/test_trade` | Execute a test trade |
| `/help` | Command list |

### Scheduled Reports
- **Morning Briefing (8:00 AM)** — Market overview, analyst summary, daily strategy
- **Evening Summary (10:00 PM)** — Trade review, analyst performance, lessons learned

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.11 |
| AI | Claude API (Anthropic) — Sonnet 4 |
| Exchange | Binance Futures Testnet |
| Market Data | Binance Public API |
| Signals | Discord (self-bot listener) |
| Notifications | Telegram Bot API |
| Database | SQLite + SQLAlchemy ORM |
| Economic Data | ForexFactory |
| Deployment | Railway.com (NIXPACKS) |
| Async | asyncio |

## Project Structure

```
dc-trading/
├── main.py                     # Entry point — async event loop
├── config.yaml                 # Full configuration
├── requirements.txt            # Python dependencies
├── Procfile                    # Railway start command
├── railway.json                # Railway deployment config
│
├── modules/
│   ├── ai_analyzer.py          # Claude AI integration
│   ├── binance_trader.py       # Binance Futures API wrapper
│   ├── database.py             # SQLAlchemy models & operations
│   ├── decision_engine.py      # Signal processing pipeline
│   ├── discord_listener.py     # Discord channel monitor
│   ├── economic_calendar.py    # ForexFactory integration
│   ├── learning_engine.py      # Post-trade learning
│   ├── market_data.py          # Market data & indicators
│   └── telegram_notifier.py    # Telegram notifications & commands
│
├── utils/
│   ├── helpers.py              # Config loading, logging, formatting
│   └── risk_manager.py         # Risk controls & position sizing
│
└── data/
    └── trades.db               # SQLite database (persistent volume)
```

## Setup

### Prerequisites

- Python 3.11+
- Discord user account token
- Anthropic API key (Claude)
- Binance Futures Testnet API key
- Telegram Bot token + Chat ID

### Installation

```bash
git clone https://github.com/linuslin0516/AI-trading.git
cd AI-trading
pip install -r requirements.txt
```

### Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Fill in your API keys in `.env`:
```env
DISCORD_TOKEN=your_discord_token
CLAUDE_API_KEY=your_claude_api_key
BINANCE_API_KEY=your_binance_testnet_api_key
BINANCE_API_SECRET=your_binance_testnet_secret
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

3. Edit `config.yaml` to configure:
   - Discord channels to monitor
   - Trading parameters (leverage, position size, risk limits)
   - Market scanner settings
   - Learning engine parameters

### Running

```bash
python main.py
```

### Deploy to Railway

1. Push to GitHub
2. Create a new project on [Railway](https://railway.com)
3. Connect your GitHub repo
4. Add environment variables in Railway dashboard
5. Create a volume mounted at `/app/data` for database persistence
6. Deploy — Railway auto-detects Python and builds with NIXPACKS

## Database Schema

| Table | Purpose |
|-------|---------|
| `trades` | Complete trade lifecycle (entry → exit → review) |
| `analysts` | Analyst profiles with dynamic weights & accuracy |
| `analyst_calls` | Per-trade analyst attribution (correct/incorrect) |
| `analyst_messages` | All Discord messages from monitored channels |
| `ai_decisions` | Every AI decision (LONG/SHORT/SKIP/ADJUST) |
| `learning_logs` | Learning events (weight updates, pattern discovery) |
| `signal_patterns` | Discovered high-win-rate signal patterns |

## Architecture

### Design Principles

1. **Event-Driven Async** — All modules run concurrently via asyncio
2. **Analyst-First Strategy** — Analyst opinions are the primary signal; technicals confirm
3. **Close-Price Analysis** — All trend analysis uses candle closes, not wicks
4. **Fee-Conscious** — Every RR calculation deducts full round-trip fees
5. **Graceful Degradation** — External API failures trigger fallbacks, not crashes
6. **Multi-Layer Risk** — Soft limits (AI-tunable) + hard limits (non-negotiable)
7. **Learning Protection** — Testnet price anomalies don't pollute learning data

### Trade Lifecycle

```
PENDING → OPEN → PARTIAL_CLOSE (TP1) → CLOSED
                                          │
                                   WIN / LOSS / BREAKEVEN
                                          │
                                   AI Review + Weight Update
```

## Disclaimer

This project is for **educational and research purposes only**. It currently runs on Binance Futures Testnet. Cryptocurrency trading involves significant risk. Use at your own discretion.

## License

MIT
