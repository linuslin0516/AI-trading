"""Test all API connections"""
import os
import hmac
import hashlib
import time

import requests
from dotenv import load_dotenv

load_dotenv()

print("=== Connection Tests ===\n")

# 1. Telegram
print("[1/4] Telegram...")
tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
r = requests.get(f"https://api.telegram.org/bot{tg_token}/getMe")
data = r.json()
print(f"  OK - @{data['result']['username']}")

# 2. Claude
print("[2/4] Claude AI...")
import anthropic
client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
msg = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=20,
    messages=[{"role": "user", "content": "say OK"}],
)
print(f"  OK - {msg.content[0].text}")

# 3. Binance Market Data
print("[3/4] Binance Market Data...")
r = requests.get("https://data-api.binance.vision/api/v3/ticker/price?symbol=BTCUSDT")
btc = r.json()
print(f"  OK - BTC: ${float(btc['price']):,.2f}")

r = requests.get("https://data-api.binance.vision/api/v3/ticker/price?symbol=ETHUSDT")
eth = r.json()
print(f"  OK - ETH: ${float(eth['price']):,.2f}")

# 4. Binance Futures Testnet
print("[4/4] Binance Futures Testnet...")
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
timestamp = int(time.time() * 1000)
query = f"timestamp={timestamp}"
sig = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
headers = {"X-MBX-APIKEY": api_key}
r = requests.get(
    f"https://testnet.binancefuture.com/fapi/v2/account?{query}&signature={sig}",
    headers=headers,
)
acct = r.json()
balance = float(acct["totalWalletBalance"])
print(f"  OK - Balance: ${balance:,.2f} USDT")

print("\nAll 4 connections successful!")
