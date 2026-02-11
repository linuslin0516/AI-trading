import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

import yaml
from dotenv import load_dotenv


def load_config(path: str = "config.yaml") -> dict:
    load_dotenv()
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 從環境變數注入 secrets
    config["discord"]["token"] = os.getenv("DISCORD_TOKEN", "")
    config["claude"]["api_key"] = os.getenv("CLAUDE_API_KEY", "")
    config["binance"]["api_key"] = os.getenv("BINANCE_API_KEY", "")
    config["binance"]["api_secret"] = os.getenv("BINANCE_API_SECRET", "")
    config["telegram"]["bot_token"] = os.getenv("TELEGRAM_BOT_TOKEN", "")
    config["telegram"]["chat_id"] = os.getenv("TELEGRAM_CHAT_ID", "")
    config.setdefault("finnhub", {})["api_key"] = os.getenv("FINNHUB_API_KEY", "")

    return config


def setup_logging(config: dict) -> logging.Logger:
    log_cfg = config.get("logging", {})
    log_file = log_cfg.get("file", "./logs/system.log")
    log_level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    max_bytes = log_cfg.get("max_size_mb", 100) * 1024 * 1024
    backup_count = log_cfg.get("backup_count", 5)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    root = logging.getLogger()
    root.setLevel(log_level)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)-25s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    fh.setLevel(log_level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # 隱藏 httpx 的 polling log（Telegram getUpdates 每 10 秒一次太吵）
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return root


def safe_json_loads(text: str, default=None):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def format_price(price: float) -> str:
    if price >= 1000:
        return f"{price:,.2f}"
    if price >= 1:
        return f"{price:.4f}"
    return f"{price:.8f}"


def format_pct(pct: float) -> str:
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def pct_change(old: float, new: float) -> float:
    if old == 0:
        return 0
    return (new - old) / old * 100
