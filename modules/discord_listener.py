import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Coroutine

import discord

logger = logging.getLogger(__name__)


@dataclass
class AnalystMessage:
    analyst: str
    channel_id: str
    channel_name: str
    content: str
    timestamp: datetime
    weight: float = 1.0


class MessageBuffer:
    """
    收到訊息後等待一小段時間（collect_window），
    讓同時段的其他分析師訊息也能一起收進來，
    然後整批送給 AI 分析。
    即使只有一則訊息也會觸發。
    """

    def __init__(self, config: dict):
        trigger_cfg = config.get("trigger", {})
        # 收集窗口：收到第一則訊息後等待 N 秒，收集同時段的其他訊息
        self.collect_window = trigger_cfg.get("collect_window_seconds", 60)

        self.messages: list[AnalystMessage] = []
        self._timer_task: asyncio.Task | None = None
        self._callback: Callable[[list[AnalystMessage]], Coroutine] | None = None

    def set_callback(self, callback: Callable[[list[AnalystMessage]], Coroutine]):
        self._callback = callback

    async def add_message(self, msg: AnalystMessage):
        self.messages.append(msg)
        logger.info(
            "Buffer +1 [%s] from %s (total: %d)",
            msg.channel_name, msg.analyst, len(self.messages),
        )

        # 第一則訊息：啟動收集計時器
        # 後續訊息：重置計時器（再多等一下看有沒有更多）
        self._reset_timer()

    def _reset_timer(self):
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
        self._timer_task = asyncio.create_task(self._wait_and_trigger())

    async def _wait_and_trigger(self):
        try:
            await asyncio.sleep(self.collect_window)
            if self.messages:
                count = len(self.messages)
                analysts = set(m.analyst for m in self.messages)
                logger.info(
                    "Collect window ended — triggering analysis "
                    "(%d messages from %d analysts: %s)",
                    count, len(analysts), ", ".join(analysts),
                )
                await self._trigger()
        except asyncio.CancelledError:
            pass

    async def _trigger(self):
        if not self.messages:
            return
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()

        batch = list(self.messages)
        self.messages.clear()

        if self._callback:
            try:
                await self._callback(batch)
            except Exception:
                logger.exception("Error in analysis callback")


class DiscordListener:
    def __init__(self, config: dict):
        self.config = config
        self.token = config["discord"]["token"]

        # 建立頻道 ID → 分析師資訊 映射
        self.channel_map: dict[str, dict] = {}
        for ch in config["discord"]["monitored_channels"]:
            self.channel_map[ch["id"]] = {
                "name": ch["name"],
                "analyst": ch["analyst"],
                "weight": ch.get("initial_weight", 1.0),
            }

        self.buffer = MessageBuffer(config)
        self._client: discord.Client | None = None
        self._analyst_weights: dict[str, float] = {}

    def update_analyst_weight(self, analyst: str, weight: float):
        self._analyst_weights[analyst] = weight

    def set_analysis_callback(self, callback):
        self.buffer.set_callback(callback)

    async def start(self):
        client = discord.Client()
        self._client = client

        @client.event
        async def on_ready():
            logger.info("Discord connected as %s", client.user)
            channels = [client.get_channel(int(cid)) for cid in self.channel_map]
            available = [c for c in channels if c is not None]
            logger.info(
                "Monitoring %d/%d channels",
                len(available), len(self.channel_map),
            )

        @client.event
        async def on_message(message: discord.Message):
            cid = str(message.channel.id)
            if cid not in self.channel_map:
                return

            # 忽略自己的訊息
            if message.author == client.user:
                return

            info = self.channel_map[cid]
            analyst = info["analyst"]
            weight = self._analyst_weights.get(analyst, info["weight"])

            content = message.content
            # 處理嵌入消息
            for embed in message.embeds:
                if embed.description:
                    content += "\n" + embed.description
                if embed.title:
                    content = embed.title + "\n" + content

            if not content.strip():
                return

            msg = AnalystMessage(
                analyst=analyst,
                channel_id=cid,
                channel_name=info["name"],
                content=content.strip(),
                timestamp=datetime.now(timezone.utc),
                weight=weight,
            )
            await self.buffer.add_message(msg)

        logger.info("Starting Discord listener...")
        await client.start(self.token)

    async def stop(self):
        if self._client:
            await self._client.close()
            logger.info("Discord listener stopped")
