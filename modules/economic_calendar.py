"""
經濟日曆模組 — 透過 ForexFactory (faireconomy) 免費 API 取得重要經濟數據公布時間

功能：
1. 取得本週重要經濟事件（High / Medium impact）
2. 判斷即將公布的數據（幾小時內）
3. 數據公布後比對「預期 vs 實際」判斷利多/利空
4. 快取機制（每小時更新一次，API 有頻率限制）
"""

import logging
import re
from datetime import datetime, timedelta, timezone

import requests

logger = logging.getLogger(__name__)

# ForexFactory 免費 JSON 端點（每週更新，限 2 次 / 5 分鐘）
FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# 只關注美元（USD）相關事件，因為對加密貨幣影響最大
CRYPTO_RELEVANT_CURRENCIES = {"USD", "ALL"}


class EconomicCalendar:
    def __init__(self, config: dict):
        self._cache: list[dict] = []
        self._cache_time: datetime | None = None
        self._cache_ttl = timedelta(hours=2)  # 2 小時快取（API 有頻率限制）
        logger.info("EconomicCalendar initialized (source=ForexFactory)")

    def _fetch_week(self) -> list[dict]:
        """從 ForexFactory 取得本週經濟日曆"""
        try:
            r = requests.get(FF_CALENDAR_URL, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 (trading-bot)"
            })
            r.raise_for_status()
            raw = r.json()
            logger.info("Fetched %d raw events from ForexFactory", len(raw))
            return raw
        except Exception as e:
            logger.error("ForexFactory calendar error: %s", e)
            return []

    def get_events(self, days_ahead: int = 7) -> list[dict]:
        """
        取得本週重要經濟事件（僅 High / Medium）

        回傳格式:
        [{
            "event": "CPI m/m",
            "country": "USD",
            "time": "2026-02-12T13:30:00+00:00",
            "impact": "high" | "medium",
            "forecast": "0.3%",
            "previous": "0.2%",
        }, ...]
        """
        now = datetime.now(timezone.utc)

        # 檢查快取
        if self._cache_time and now - self._cache_time < self._cache_ttl:
            return self._cache

        raw_events = self._fetch_week()
        if not raw_events:
            return self._cache  # 回傳舊快取

        events = []
        for e in raw_events:
            impact = (e.get("impact") or "").lower()
            country = e.get("country", "")

            # 只保留 High/Medium 且與美元相關的事件
            if impact not in ("high", "medium"):
                continue
            if country not in CRYPTO_RELEVANT_CURRENCIES:
                continue

            event_time = self._parse_time(e.get("date", ""))

            event = {
                "event": e.get("title", ""),
                "country": country,
                "time": event_time.isoformat() if event_time else e.get("date", ""),
                "impact": impact,
                "forecast": e.get("forecast", ""),
                "previous": e.get("previous", ""),
            }
            events.append(event)

        # 按時間排序
        events.sort(key=lambda x: x.get("time", ""))

        # 更新快取
        self._cache = events
        self._cache_time = now

        high_count = sum(1 for ev in events if ev["impact"] == "high")
        logger.info("Processed %d USD economic events (%d high-impact)",
                     len(events), high_count)

        return events

    def get_upcoming_events(self, hours: int = 24) -> list[dict]:
        """取得接下來 N 小時內的事件"""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours)

        events = self.get_events()
        upcoming = []

        for e in events:
            event_time = self._parse_time(e.get("time", ""))
            if event_time and now <= event_time <= cutoff:
                e_copy = dict(e)
                e_copy["hours_until"] = round(
                    (event_time - now).total_seconds() / 3600, 1
                )
                upcoming.append(e_copy)

        return upcoming

    def get_today_events(self) -> list[dict]:
        """取得今日所有事件"""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        events = self.get_events()
        return [e for e in events if today in e.get("time", "")]

    def get_recent_releases(self, hours: int = 4) -> list[dict]:
        """取得最近 N 小時的事件（不論是否已公布）"""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=hours)

        events = self.get_events()
        recent = []

        for e in events:
            event_time = self._parse_time(e.get("time", ""))
            if event_time and cutoff <= event_time <= now:
                recent.append(e)

        return recent

    def format_for_ai(self, events: list[dict]) -> str:
        """將事件格式化成 AI 可讀的文字"""
        if not events:
            return "近期無重要經濟數據公布"

        lines = []
        for e in events:
            impact_icon = {"high": "!!!", "medium": "!!"}.get(e["impact"], "!")
            line = f"[{impact_icon}] {e['country']} {e['event']}"

            if e.get("time"):
                line += f" @ {e['time']}"

            parts = []
            if e.get("forecast"):
                parts.append(f"預期: {e['forecast']}")
            if e.get("previous"):
                parts.append(f"前值: {e['previous']}")
            if parts:
                line += f" ({', '.join(parts)})"

            # 嘗試判斷利多/利空（如果已公布）
            surprise = self._try_analyze(e)
            if surprise:
                line += f" -> {surprise}"

            if e.get("hours_until") is not None:
                line += f"  [距今 {e['hours_until']} 小時]"

            lines.append(line)

        return "\n".join(lines)

    def _try_analyze(self, event: dict) -> str:
        """嘗試根據 forecast vs previous 推斷趨勢方向"""
        forecast = self._parse_number(event.get("forecast", ""))
        previous = self._parse_number(event.get("previous", ""))

        if forecast is None or previous is None:
            return ""

        event_name = event.get("event", "").upper()

        # 通膨相關：預期上升 = 利空
        inflation_keywords = ["CPI", "PPI", "PCE", "INFLATION", "PRICE INDEX"]
        if any(kw in event_name for kw in inflation_keywords):
            if forecast > previous:
                return "預期利空(通膨升)"
            elif forecast < previous:
                return "預期利多(通膨降)"

        # 就業相關：預期上升 = 利多
        job_keywords = ["PAYROLL", "NFP", "EMPLOYMENT", "JOB"]
        if any(kw in event_name for kw in job_keywords):
            if forecast > previous:
                return "預期利多(就業強)"
            elif forecast < previous:
                return "預期利空(就業弱)"

        return ""

    @staticmethod
    def _parse_number(text: str) -> float | None:
        """從字串中提取數字（例如 '0.3%' → 0.3, '-12.5K' → -12.5）"""
        if not text:
            return None
        cleaned = re.sub(r'[%KMBTkmbts]', '', text.strip())
        try:
            return float(cleaned)
        except ValueError:
            return None

    @staticmethod
    def _parse_time(time_str: str) -> datetime | None:
        """解析時間字串"""
        if not time_str:
            return None
        # ForexFactory 格式: "2026-02-08T18:30:00-05:00"
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(time_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError:
                continue
        return None
