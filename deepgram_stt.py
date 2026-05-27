"""deepgram_stt.py — Deepgram Flux streaming STT (Option 1, trust Flux turn detection)."""

import asyncio
import logging

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType

from config import DEEPGRAM_API_KEY, DEEPGRAM_MODEL, SAMPLE_RATE

log = logging.getLogger(__name__)


class DeepgramStream:
    def __init__(self):
        self._client = AsyncDeepgramClient(api_key=DEEPGRAM_API_KEY)
        self._conn_cm = None
        self._conn = None
        self._final_queue: asyncio.Queue = asyncio.Queue()
        self._listen_task = None

    async def start(self):
        self._conn_cm = self._client.listen.v2.connect(
            model=DEEPGRAM_MODEL,
            encoding="linear16",
            sample_rate=SAMPLE_RATE,
            eot_threshold=0.85,
            eot_timeout_ms=7000,
        )
        self._conn = await self._conn_cm.__aenter__()
        self._conn.on(EventType.MESSAGE, self._on_message)
        self._conn.on(EventType.ERROR, lambda e: log.error(f"[DG] error: {e}"))
        self._conn.on(EventType.CLOSE, lambda _: log.info("[DG] connection closed"))
        self._listen_task = asyncio.create_task(self._conn.start_listening())
        log.info(f"[DG] Flux streaming open (model={DEEPGRAM_MODEL})")

    def _on_message(self, message):
        try:
            if isinstance(message, dict):
                mtype      = message.get("type", "")
                event      = message.get("event", "")
                transcript = (message.get("transcript", "") or "").strip()
            else:
                mtype      = getattr(message, "type", "")
                event      = getattr(message, "event", "")
                transcript = (getattr(message, "transcript", "") or "").strip()

            if mtype != "TurnInfo":
                return

            if event == "EndOfTurn" and transcript:
                log.info(f"[DG] EndOfTurn: '{transcript}'")
                self._final_queue.put_nowait(("final", transcript))
            elif transcript:   # Update / StartOfTurn with text = interim
                self._final_queue.put_nowait(("interim", transcript))
        except Exception as e:
            log.error(f"[DG] message parse error: {e}")

    async def send_audio(self, pcm_bytes: bytes):
        if self._conn is None or not pcm_bytes:
            return
        try:
            await self._conn.send_media(pcm_bytes)
        except Exception as e:
            log.error(f"[DG] send_audio failed: {e}")

    async def get_final(self, timeout: float = 1.0):
        try:
            return await asyncio.wait_for(self._final_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def close(self):
        try:
            import traceback
            log.info(f"[DG] close() called from:\n{''.join(traceback.format_stack()[-3:])}")  # ← temporary
            if self._listen_task and not self._listen_task.done():
                self._listen_task.cancel()
            if self._conn_cm is not None:
                await self._conn_cm.__aexit__(None, None, None)
        except Exception as e:
            log.error(f"[DG] close error: {e}")
        finally:
            self._conn = None
            self._conn_cm = None
            log.info("[DG] stream closed")