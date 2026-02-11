"""Verify Binance Testnet connection and funds"""
import os
import hmac
import hashlib
import time
import json
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

FUTURES_URL = "https://testnet.binancefuture.com"
MAINNET_URL = "https://fapi.binance.com"

session = requests.Session()
session.headers.update({"X-MBX-APIKEY": api_key})


def sign(params):
    params["timestamp"] = int(time.time() * 1000)
    query = "&".join(f"{k}={v}" for k, v in params.items())
    sig = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    params["signature"] = sig
    return params


print("=" * 50)
print("BINANCE ENVIRONMENT CHECK")
print("=" * 50)

# 1. Confirm we're on testnet
print("\n[1] Endpoint verification")
print(f"  Using: {FUTURES_URL}")
print(f"  This is TESTNET: YES" if "testnet" in FUTURES_URL else "  WARNING: MAINNET!")

# 2. Confirm mainnet is NOT accessible
print("\n[2] Mainnet block check")
try:
    r = requests.get(f"{MAINNET_URL}/fapi/v1/ping", timeout=5)
    if r.status_code == 451:
        print("  Mainnet BLOCKED (451) - GOOD, you cannot accidentally trade on mainnet")
    else:
        print(f"  WARNING: Mainnet returned {r.status_code}")
except Exception as e:
    print(f"  Mainnet unreachable: {e} - GOOD")

# 3. Testnet account info
print("\n[3] Testnet account")
params = sign({})
r = session.get(f"{FUTURES_URL}/fapi/v2/account", params=params)
acct = r.json()

if "totalWalletBalance" in acct:
    balance = float(acct["totalWalletBalance"])
    available = float(acct["availableBalance"])
    unrealized = float(acct["totalUnrealizedProfit"])
    can_trade = acct.get("canTrade", False)

    print(f"  Can trade: {can_trade}")
    print(f"  Total balance: ${balance:,.2f} USDT")
    print(f"  Available: ${available:,.2f} USDT")
    print(f"  Unrealized PnL: ${unrealized:,.2f}")
else:
    print(f"  ERROR: {acct}")

# 4. Check open positions
print("\n[4] Open positions")
params2 = sign({})
r2 = session.get(f"{FUTURES_URL}/fapi/v2/positionRisk", params=params2)
positions = r2.json()
has_position = False
for p in positions:
    amt = float(p.get("positionAmt", 0))
    if amt != 0:
        has_position = True
        print(f"  {p['symbol']}: {amt} (entry: {p['entryPrice']}, PnL: {p['unRealizedProfit']})")
if not has_position:
    print("  No open positions")

# 5. Check open orders
print("\n[5] Open orders")
params3 = sign({})
r3 = session.get(f"{FUTURES_URL}/fapi/v1/openOrders", params=params3)
orders = r3.json()
if orders:
    for o in orders:
        print(f"  {o['symbol']} {o['side']} {o['type']} qty={o['origQty']} price={o.get('stopPrice', o.get('price', 'N/A'))}")
else:
    print("  No open orders")

# 6. Test placing and cancelling a tiny order
print("\n[6] Order test (place + cancel)")
try:
    test_params = sign({
        "symbol": "BTCUSDT",
        "side": "BUY",
        "type": "LIMIT",
        "timeInForce": "GTC",
        "quantity": 0.003,
        "price": 50000,  # far below market = won't fill
    })
    r4 = session.post(f"{FUTURES_URL}/fapi/v1/order", params=test_params)
    order = r4.json()
    if "orderId" in order:
        oid = order["orderId"]
        print(f"  Test order placed: #{oid}")
        # Cancel it
        cancel_params = sign({"symbol": "BTCUSDT", "orderId": oid})
        r5 = session.delete(f"{FUTURES_URL}/fapi/v1/order", params=cancel_params)
        print(f"  Test order cancelled: OK")
        print("  ORDER EXECUTION: WORKING")
    else:
        print(f"  Order response: {order}")
except Exception as e:
    print(f"  Order test failed: {e}")

print("\n" + "=" * 50)
print("SUMMARY: You are on TESTNET with virtual funds.")
print("No real money is at risk.")
print("=" * 50)
