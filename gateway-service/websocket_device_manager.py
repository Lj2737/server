"""WebSocket device connection manager.

Hardware badges connect to:
    /badge/v1/algorithm/ws/device/{deviceNo}

This manager keeps the deviceNo -> WebSocket mapping, pushes broadcast audio,
pushes AI dialog audio, and maintains a lightweight text-frame heartbeat.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncGenerator, Dict, Optional

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from config import (
    TTS_TARGET_CHANNELS,
    TTS_TARGET_SAMPLE_RATE,
    TTS_TARGET_SAMPLE_WIDTH,
    TTS_PUSH_SPEED,
    TTS_REALTIME_PUSH,
    WS_HEARTBEAT_FAIL_THRESHOLD,
    WS_HEARTBEAT_INTERVAL,
    WS_PING_TIMEOUT,
)


class _DeviceConnection:
    """Runtime state for one connected badge device."""

    def __init__(self, device_no: str, websocket: WebSocket):
        self.device_no = device_no
        self.websocket = websocket
        self.connected_at = time.time()
        self.last_pong_time = time.time()
        self.missed_pings = 0

    def record_pong(self) -> None:
        self.last_pong_time = time.time()
        self.missed_pings = 0

    def record_activity(self) -> None:
        self.last_pong_time = time.time()
        self.missed_pings = 0

    def increment_missed_ping(self) -> int:
        self.missed_pings += 1
        return self.missed_pings


class WebSocketDeviceManager:
    """Singleton manager for online badge WebSocket connections."""

    _instance: Optional["WebSocketDeviceManager"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_constructed", False):
            return
        self._connections: Dict[str, _DeviceConnection] = {}
        self._streaming_devices = set()
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._initialized = False
        self._constructed = True

    def initialize(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        logger.info(
            f"WebSocketDeviceManager initialized | heartbeatInterval={WS_HEARTBEAT_INTERVAL}s | "
            f"failThreshold={WS_HEARTBEAT_FAIL_THRESHOLD}"
        )

    async def register_device(self, device_no: str, websocket: WebSocket) -> None:
        """Register or replace a device WebSocket connection."""
        if device_no in self._connections:
            old_conn = self._connections[device_no]
            logger.warning(
                f"Replacing existing WebSocket connection | deviceNo={device_no} | "
                f"oldAge={time.time() - old_conn.connected_at:.0f}s"
            )
            try:
                await old_conn.websocket.close(code=4000, reason="duplicate device connection")
            except Exception:
                pass

        self._connections[device_no] = _DeviceConnection(device_no, websocket)
        logger.info(
            f"Device WebSocket registered | deviceNo={device_no} | "
            f"onlineCount={len(self._connections)}"
        )

    async def unregister_device(self, device_no: str, websocket: Optional[WebSocket] = None) -> None:
        """Remove a device WebSocket connection from the online map."""
        conn = self._connections.get(device_no)
        if conn is None:
            return
        if websocket is not None and conn.websocket is not websocket:
            logger.info(
                f"Skip unregister for stale WebSocket | deviceNo={device_no} | "
                f"onlineCount={len(self._connections)}"
            )
            return

        del self._connections[device_no]
        self._streaming_devices.discard(device_no)
        logger.info(
            f"Device WebSocket unregistered | deviceNo={device_no} | "
            f"onlineCount={len(self._connections)}"
        )

    @staticmethod
    async def _pace_audio_chunk(chunk_size: int) -> None:
        if not TTS_REALTIME_PUSH:
            return
        bytes_per_second = TTS_TARGET_SAMPLE_RATE * TTS_TARGET_SAMPLE_WIDTH * TTS_TARGET_CHANNELS
        if bytes_per_second <= 0 or chunk_size <= 0:
            return
        speed = TTS_PUSH_SPEED if TTS_PUSH_SPEED > 0 else 1.0
        await asyncio.sleep(chunk_size / bytes_per_second / speed)

    async def send_text(self, device_no: str, message: dict) -> bool:
        """Send a JSON text frame to a connected device."""
        conn = self._connections.get(device_no)
        if conn is None:
            logger.warning(f"Device text send skipped; device offline | deviceNo={device_no}")
            return False

        try:
            await conn.websocket.send_text(json.dumps(message, ensure_ascii=False))
            return True
        except WebSocketDisconnect:
            await self.unregister_device(device_no, websocket=conn.websocket)
            return False
        except Exception as exc:
            logger.error(
                f"Device text send failed | deviceNo={device_no} | "
                f"type={message.get('type')} | error={str(exc)[:200]}"
            )
            return False

    async def push_dialog_audio_stream(
        self,
        device_no: str,
        dialog_id: str,
        audio_stream: AsyncGenerator[bytes, None],
    ) -> bool:
        """Push AI dialog TTS audio and send dialog_end_ack when done."""
        conn = self._connections.get(device_no)
        if conn is None:
            logger.warning(f"AI dialog audio push skipped; device offline | deviceNo={device_no}")
            return False

        total_bytes = 0
        chunk_count = 0
        start_time = time.time()
        self._streaming_devices.add(device_no)
        try:
            async for pcm_chunk in audio_stream:
                await conn.websocket.send_bytes(pcm_chunk)
                total_bytes += len(pcm_chunk)
                chunk_count += 1
                await self._pace_audio_chunk(len(pcm_chunk))

            await conn.websocket.send_text(
                json.dumps(
                    {
                        "type": "dialog_end_ack",
                        "dialogId": dialog_id,
                    },
                    ensure_ascii=False,
                )
            )
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"AI dialog audio pushed | deviceNo={device_no} | dialogId={dialog_id} | "
                f"chunks={chunk_count} | bytes={total_bytes} | elapsed={elapsed_ms}ms"
            )
            return True
        except WebSocketDisconnect:
            logger.warning(
                f"AI dialog audio push interrupted; device disconnected | "
                f"deviceNo={device_no} | dialogId={dialog_id}"
            )
            await self.unregister_device(device_no, websocket=conn.websocket)
            return False
        except asyncio.CancelledError:
            logger.info(
                f"AI dialog audio push cancelled | deviceNo={device_no} | "
                f"dialogId={dialog_id} | chunks={chunk_count} | bytes={total_bytes}"
            )
            raise
        except Exception as exc:
            error_detail = str(exc)[:1200]
            logger.error(
                f"AI dialog audio push failed | deviceNo={device_no} | dialogId={dialog_id} | "
                f"chunks={chunk_count} | bytes={total_bytes} | error={error_detail}"
            )
            await self.send_text(
                device_no,
                {
                    "type": "dialog_error",
                    "dialogId": dialog_id,
                    "message": f"AI dialog audio push failed: {error_detail}",
                },
            )
            return False
        finally:
            self._streaming_devices.discard(device_no)

    async def push_audio_stream(
        self,
        device_no: str,
        audio_stream: AsyncGenerator[bytes, None],
        broadcast_content: str = "",
    ) -> bool:
        """Push broadcast audio to a device using broadcast_start/end frames."""
        conn = self._connections.get(device_no)
        if conn is None:
            logger.warning(f"Broadcast audio push skipped; device offline | deviceNo={device_no}")
            return False

        start_time = time.time()
        total_bytes = 0
        chunk_count = 0
        self._streaming_devices.add(device_no)

        try:
            await conn.websocket.send_text(
                json.dumps(
                    {
                        "type": "broadcast_start",
                        "content": broadcast_content,
                    },
                    ensure_ascii=False,
                )
            )

            async for pcm_chunk in audio_stream:
                await conn.websocket.send_bytes(pcm_chunk)
                total_bytes += len(pcm_chunk)
                chunk_count += 1
                await self._pace_audio_chunk(len(pcm_chunk))

            await conn.websocket.send_text(json.dumps({"type": "broadcast_end"}, ensure_ascii=False))

            elapsed_ms = int((time.time() - start_time) * 1000)
            audio_duration = total_bytes / (2 * 16000) if total_bytes > 0 else 0
            logger.info(
                f"Broadcast audio pushed | deviceNo={device_no} | chunks={chunk_count} | "
                f"bytes={total_bytes} | duration={audio_duration:.2f}s | elapsed={elapsed_ms}ms"
            )
            return True
        except WebSocketDisconnect:
            logger.warning(f"Broadcast audio push interrupted; device disconnected | deviceNo={device_no}")
            await self.unregister_device(device_no, websocket=conn.websocket)
            return False
        except Exception as exc:
            logger.error(
                f"Broadcast audio push failed | deviceNo={device_no} | chunks={chunk_count} | "
                f"bytes={total_bytes} | errorType={type(exc).__name__} | error={str(exc)[:200]}"
            )
            try:
                await conn.websocket.send_text(
                    json.dumps(
                        {
                            "type": "broadcast_error",
                            "message": "broadcast audio push failed",
                        },
                        ensure_ascii=False,
                    )
                )
            except Exception:
                pass
            return False
        finally:
            self._streaming_devices.discard(device_no)

    def is_device_online(self, device_no: str) -> bool:
        return device_no in self._connections

    def get_online_devices(self) -> list:
        return list(self._connections.keys())

    def get_online_count(self) -> int:
        return len(self._connections)

    def record_device_activity(self, device_no: str) -> None:
        conn = self._connections.get(device_no)
        if conn is not None:
            conn.record_activity()

    async def prune_stale_devices(self) -> None:
        stale_after = WS_HEARTBEAT_INTERVAL + WS_PING_TIMEOUT * WS_HEARTBEAT_FAIL_THRESHOLD
        now = time.time()
        stale_devices = [
            device_no
            for device_no, conn in list(self._connections.items())
            if device_no not in self._streaming_devices and now - conn.last_pong_time > stale_after
        ]
        for device_no in stale_devices:
            logger.warning(
                f"Pruning stale WebSocket device before status response | "
                f"deviceNo={device_no} | staleAfter={stale_after:.1f}s"
            )
            conn = self._connections.get(device_no)
            await self.unregister_device(device_no, websocket=conn.websocket if conn else None)

    async def start_heartbeat(self) -> None:
        """Start the background text-frame heartbeat loop."""
        if self._heartbeat_task is not None:
            logger.warning("WebSocket heartbeat is already running")
            return

        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(),
            name="ws-device-heartbeat",
        )
        logger.info(
            f"WebSocket heartbeat started | interval={WS_HEARTBEAT_INTERVAL}s | "
            f"pingTimeout={WS_PING_TIMEOUT}s | failThreshold={WS_HEARTBEAT_FAIL_THRESHOLD}"
        )

    async def stop_heartbeat(self) -> None:
        """Stop the background heartbeat loop."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
            logger.info("WebSocket heartbeat stopped")

    async def _heartbeat_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(WS_HEARTBEAT_INTERVAL)
                if not self._connections:
                    continue

                offline_devices = []
                for device_no, conn in list(self._connections.items()):
                    if device_no in self._streaming_devices:
                        conn.record_pong()
                        continue
                    try:
                        await conn.websocket.send_text(json.dumps({"type": "ping"}, ensure_ascii=False))
                        logger.debug(f"WebSocket ping sent | deviceNo={device_no}")

                        await asyncio.sleep(0)
                        time_since_last = time.time() - conn.last_pong_time
                        if time_since_last > WS_HEARTBEAT_INTERVAL + WS_PING_TIMEOUT:
                            missed = conn.increment_missed_ping()
                            logger.debug(
                                f"WebSocket heartbeat missed | deviceNo={device_no} | "
                                f"missed={missed} | lastPongAge={time_since_last:.1f}s"
                            )
                            if missed >= WS_HEARTBEAT_FAIL_THRESHOLD:
                                offline_devices.append(device_no)
                        elif conn.missed_pings > 0:
                            conn.record_pong()
                    except Exception as exc:
                        logger.warning(
                            f"WebSocket heartbeat send failed | deviceNo={device_no} | "
                            f"error={str(exc)[:100]}"
                        )
                        missed = conn.increment_missed_ping()
                        if missed >= WS_HEARTBEAT_FAIL_THRESHOLD:
                            offline_devices.append(device_no)

                for device_no in offline_devices:
                    logger.warning(
                        f"WebSocket heartbeat timeout; unregistering device | "
                        f"deviceNo={device_no} | missed={WS_HEARTBEAT_FAIL_THRESHOLD}"
                    )
                    conn = self._connections.get(device_no)
                    await self.unregister_device(device_no, websocket=conn.websocket if conn else None)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    f"WebSocket heartbeat loop error | errorType={type(exc).__name__} | "
                    f"error={str(exc)[:200]}"
                )

    def handle_pong(self, device_no: str) -> None:
        if device_no in self._connections:
            self._connections[device_no].record_pong()
            logger.debug(f"WebSocket pong received | deviceNo={device_no}")

    async def close_all(self) -> None:
        """Close all device WebSocket connections."""
        device_list = list(self._connections.keys())
        for device_no in device_list:
            try:
                conn = self._connections[device_no]
                await conn.websocket.close(code=1001, reason="server shutdown")
            except Exception:
                pass

        self._connections.clear()
        logger.info(f"All device WebSocket connections closed | count={len(device_list)}")
