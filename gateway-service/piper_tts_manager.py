"""TTS API manager.

The class name is kept as PiperTTSManager for compatibility with existing
router code. The implementation calls a non-streaming speech API, normalizes
the returned audio to PCM when possible, then yields chunks for WebSocket push.
"""
from __future__ import annotations

import base64
import asyncio
import importlib.util
import io
import json
import sys
import threading
import wave
from typing import AsyncGenerator, Optional

import httpx
import numpy as np
from loguru import logger

from config import (
    TTS_API_BASE_URL,
    TTS_API_KEY,
    TTS_API_TIMEOUT,
    TTS_LANGUAGE_TYPE,
    TTS_MODEL_NAME,
    TTS_PUSH_CHUNK_SIZE,
    TTS_REALTIME_MODEL_NAME,
    TTS_REALTIME_RESPONSE_FORMAT,
    TTS_REALTIME_SPEECH_RATE,
    TTS_REALTIME_VOICE,
    TTS_REALTIME_WS_URL,
    TTS_RESPONSE_FORMAT,
    TTS_TARGET_CHANNELS,
    TTS_TARGET_SAMPLE_RATE,
    TTS_TARGET_SAMPLE_WIDTH,
    TTS_VOICE,
    BROADCAST_MAX_AUDIO_SECONDS,
)


class PiperTTSManager:
    """Singleton TTS API client used by broadcast and AI dialog routes."""

    _instance: Optional["PiperTTSManager"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._client: Optional[httpx.AsyncClient] = None
        self._is_loaded = False
        self._load_error: Optional[str] = None
        self._initialized = True

    @staticmethod
    def _mask_secret(secret: str) -> str:
        if not secret:
            return "empty"
        if len(secret) <= 10:
            return f"len={len(secret)}"
        return f"{secret[:6]}...{secret[-4:]}(len={len(secret)})"

    async def load_model(self) -> bool:
        """Initialize the async TTS API client."""
        try:
            if not TTS_API_KEY or TTS_API_KEY.startswith("replace-with-"):
                raise RuntimeError("TTS_API_KEY is not configured in gateway-service/.env")

            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(TTS_API_TIMEOUT),
                headers={
                    "Authorization": f"Bearer {TTS_API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            self._is_loaded = True
            self._load_error = None
            logger.info(
                f"TTS API client initialized | model={TTS_MODEL_NAME} | "
                f"url={TTS_API_BASE_URL} | format={TTS_RESPONSE_FORMAT} | "
                f"key={self._mask_secret(TTS_API_KEY)}"
            )
            return True
        except Exception as exc:
            self._is_loaded = False
            self._load_error = str(exc)
            logger.error(f"TTS API client initialization failed | error={exc}")
            return False

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Synthesize text and yield PCM/audio chunks for WebSocket delivery."""
        if not self._is_loaded or self._client is None:
            raise RuntimeError(f"TTS API client is not initialized | error={self._load_error}")

        if not text or not text.strip():
            logger.warning("TTS text is empty")
            return

        audio_bytes = await self._request_tts(text.strip())
        pcm_bytes = self._normalize_audio_bytes(audio_bytes)
        duration_seconds = self._pcm_duration_seconds(pcm_bytes)
        if (
            BROADCAST_MAX_AUDIO_SECONDS > 0
            and duration_seconds > BROADCAST_MAX_AUDIO_SECONDS
        ):
            raise RuntimeError(
                "TTS audio duration exceeds broadcast limit. "
                f"duration={duration_seconds:.2f}s | "
                f"limit={BROADCAST_MAX_AUDIO_SECONDS:.2f}s | "
                f"textLength={len(text)}"
            )
        # from test_ import diagnostics
        # diagnostics.check_normalized_pcm(pcm_bytes,len(audio_bytes))

        # import numpy as np
        # pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
        # print(f"PCM数据预览（前100个采样点）：{pcm_array[:100]}")
        # print(f"PCM数据统计：min={pcm_array.min()}, max={pcm_array.max()}, mean={pcm_array.mean():.2f}")
        # print(f"音频时长：{len(pcm_bytes) / (16000 * 2):.2f}秒")

        # # 保存PCM数据到文件（可以用Audacity打开分析）
        # diagnostics.save_audio_to_disk(pcm_bytes, "pcm_normalized")
        # # === 诊断代码结束 ===
        for start in range(0, len(pcm_bytes), TTS_PUSH_CHUNK_SIZE):
            chunk = pcm_bytes[start:start + TTS_PUSH_CHUNK_SIZE]
            if chunk:
                yield chunk

        logger.info(
            f"TTS synthesis completed | textLength={len(text)} | "
            f"audioBytes={len(pcm_bytes)} | duration={duration_seconds:.2f}s"
        )

    async def synthesize_realtime_stream(
        self,
        text_stream: AsyncGenerator[str, None],
    ) -> AsyncGenerator[bytes, None]:
        """Synthesize a text stream with DashScope realtime TTS and yield PCM chunks."""
        if not self._is_loaded:
            raise RuntimeError(f"TTS API client is not initialized | error={self._load_error}")

        try:
            import dashscope
            from dashscope.audio.qwen_tts_realtime import AudioFormat, QwenTtsRealtime, QwenTtsRealtimeCallback
        except ImportError as exc:
            dashscope_spec = importlib.util.find_spec("dashscope")
            dashscope_location = dashscope_spec.origin if dashscope_spec else "not found"
            raise RuntimeError(
                "dashscope realtime TTS SDK import failed. "
                "Ensure dashscope>=1.25.11 is installed in the Python environment running gateway-service. "
                f"python={sys.executable} | dashscope={dashscope_location} | "
                f"Original import error: {exc}"
            ) from exc
        dashscope.api_key = TTS_API_KEY

        loop = asyncio.get_running_loop()
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        error_future: asyncio.Future[BaseException] = loop.create_future()
        stop_event = threading.Event()

        class _RealtimeTTSCallback(QwenTtsRealtimeCallback):
            def on_open(self) -> None:
                logger.debug("Realtime TTS websocket opened")

            def on_event(self, response) -> None:
                try:
                    event_type = response.get("type") if isinstance(response, dict) else ""
                    if event_type == "response.audio.delta":
                        audio_b64 = response.get("delta", "")
                        if audio_b64:
                            loop.call_soon_threadsafe(
                                audio_queue.put_nowait,
                                base64.b64decode(audio_b64),
                            )
                    elif event_type == "error":
                        error = RuntimeError(f"Realtime TTS failed: {response}")
                        if not error_future.done():
                            loop.call_soon_threadsafe(error_future.set_result, error)
                        loop.call_soon_threadsafe(audio_queue.put_nowait, None)
                    elif event_type == "session.finished":
                        loop.call_soon_threadsafe(audio_queue.put_nowait, None)
                except BaseException as exc:
                    if not error_future.done():
                        loop.call_soon_threadsafe(error_future.set_result, exc)
                    loop.call_soon_threadsafe(audio_queue.put_nowait, None)

            def on_close(self, close_status_code: int, close_msg: str) -> None:
                logger.debug(
                    f"Realtime TTS websocket closed | code={close_status_code} | msg={close_msg}"
                )
                loop.call_soon_threadsafe(audio_queue.put_nowait, None)

            def on_error(self, message: str) -> None:
                error = RuntimeError(f"Realtime TTS failed: {message}")
                if not error_future.done():
                    loop.call_soon_threadsafe(error_future.set_result, error)
                loop.call_soon_threadsafe(audio_queue.put_nowait, None)

        callback = _RealtimeTTSCallback()
        synthesizer_holder: dict[str, object] = {}
        producer_done = asyncio.Event()

        def _run_text_producer() -> None:
            synthesizer = None
            try:
                synthesizer = QwenTtsRealtime(
                    model=TTS_REALTIME_MODEL_NAME,
                    callback=callback,
                    url=TTS_REALTIME_WS_URL,
                )
                synthesizer_holder["synthesizer"] = synthesizer
                synthesizer.connect()
                synthesizer.update_session(
                    voice=TTS_REALTIME_VOICE,
                    response_format=self._realtime_audio_format(AudioFormat),
                    speech_rate=TTS_REALTIME_SPEECH_RATE,
                    language_type=TTS_LANGUAGE_TYPE,
                    mode="server_commit",
                )
                for item in text_iter:
                    if stop_event.is_set():
                        break
                    if item:
                        synthesizer.append_text(item)
                if not stop_event.is_set():
                    synthesizer.finish()
            except BaseException as exc:
                if not error_future.done():
                    loop.call_soon_threadsafe(error_future.set_result, exc)
                loop.call_soon_threadsafe(audio_queue.put_nowait, None)
            finally:
                loop.call_soon_threadsafe(producer_done.set)

        text_iter = self._sync_text_iter(loop, text_stream)
        producer_thread = threading.Thread(
            target=_run_text_producer,
            name="dashscope-realtime-tts",
            daemon=True,
        )
        producer_thread.start()

        total_bytes = 0
        try:
            while True:
                if error_future.done():
                    raise error_future.result()
                chunk = await audio_queue.get()
                if chunk is None:
                    if error_future.done():
                        raise error_future.result()
                    break
                normalized_chunk = self._normalize_realtime_pcm_chunk(chunk)
                total_bytes += len(normalized_chunk)
                if normalized_chunk:
                    yield normalized_chunk
        finally:
            stop_event.set()
            await text_iter.aclose()
            synthesizer = synthesizer_holder.get("synthesizer")
            if synthesizer is not None:
                for method_name in ("close", "cancel"):
                    close_method = getattr(synthesizer, method_name, None)
                    if callable(close_method):
                        try:
                            close_method()
                        except Exception as exc:
                            logger.debug(f"Realtime TTS {method_name} failed | error={exc}")
                        break
            await producer_done.wait()
            logger.info(f"Realtime TTS completed | audioBytes={total_bytes}")

    @staticmethod
    def _realtime_audio_format(audio_format_cls):
        format_name = TTS_REALTIME_RESPONSE_FORMAT.strip()
        if hasattr(audio_format_cls, format_name):
            return getattr(audio_format_cls, format_name)
        logger.warning(
            f"Realtime TTS audio format not found | format={format_name}; "
            "fallback=PCM_24000HZ_MONO_16BIT"
        )
        return getattr(audio_format_cls, "PCM_24000HZ_MONO_16BIT")

    @staticmethod
    def _realtime_source_sample_rate() -> int:
        format_name = TTS_REALTIME_RESPONSE_FORMAT.upper()
        if "24000HZ" in format_name:
            return 24000
        if "16000HZ" in format_name:
            return 16000
        if "8000HZ" in format_name:
            return 8000
        logger.warning(
            f"Realtime TTS sample rate is unknown | format={TTS_REALTIME_RESPONSE_FORMAT}; "
            "assume=24000"
        )
        return 24000

    def _normalize_realtime_pcm_chunk(self, pcm_bytes: bytes) -> bytes:
        source_sample_rate = self._realtime_source_sample_rate()
        if source_sample_rate == TTS_TARGET_SAMPLE_RATE:
            return pcm_bytes
        return self._resample_pcm16_mono(
            pcm_bytes,
            source_sample_rate=source_sample_rate,
            target_sample_rate=TTS_TARGET_SAMPLE_RATE,
        )

    @staticmethod
    def _resample_pcm16_mono(
        pcm_bytes: bytes,
        source_sample_rate: int,
        target_sample_rate: int,
    ) -> bytes:
        pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        target_len = int(len(pcm_array) * target_sample_rate / source_sample_rate)
        if target_len <= 0:
            return b""

        try:
            from scipy.signal import resample

            resampled = resample(pcm_array, target_len)
            return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()
        except Exception as exc:
            logger.warning(f"TTS PCM resample failed; using original PCM | error={exc}")
            return pcm_bytes

    @staticmethod
    def _sync_text_iter(
        loop: asyncio.AbstractEventLoop,
        text_stream: AsyncGenerator[str, None],
    ):
        class _SyncTextIterator:
            def __init__(self) -> None:
                self._closed = False

            def __iter__(self):
                return self

            def __next__(self) -> str:
                if self._closed:
                    raise StopIteration
                future = asyncio.run_coroutine_threadsafe(text_stream.__anext__(), loop)
                try:
                    return future.result()
                except StopAsyncIteration:
                    raise StopIteration

            async def aclose(self) -> None:
                self._closed = True
                await text_stream.aclose()

        return _SyncTextIterator()

    async def _request_tts(self, text: str) -> bytes:
        """Call the non-streaming speech API and return audio bytes."""
        if self._client is None:
            raise RuntimeError("TTS API client is not initialized")

        payload = self._build_payload(text)
        response = await self._client.post(TTS_API_BASE_URL, json=payload)
        if response.status_code >= 400:
            response_preview = response.text[:1000]
            request_id = response.headers.get("x-request-id") or response.headers.get("X-Request-Id") or ""
            if response.status_code == 401:
                raise RuntimeError(
                    "TTS API authorization failed (401). "
                    f"Check gateway-service/.env TTS_API_KEY. key={self._mask_secret(TTS_API_KEY)} | "
                    f"url={TTS_API_BASE_URL} | requestId={request_id} | response={response_preview}"
                )
            raise RuntimeError(
                "TTS API request failed. "
                f"status={response.status_code} | url={TTS_API_BASE_URL} | "
                f"model={TTS_MODEL_NAME} | voice={TTS_VOICE} | "
                f"response_format={TTS_RESPONSE_FORMAT} | requestId={request_id} | response={response_preview}"
            )

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return await self._extract_audio_from_json(response.json())
        return response.content

    @staticmethod
    def _uses_dashscope_native_api() -> bool:
        return "/api/v1/services/aigc/multimodal-generation/generation" in TTS_API_BASE_URL

    def _build_payload(self, text: str) -> dict:
        """Build request payload for DashScope native or OpenAI-compatible API."""
        if self._uses_dashscope_native_api():
            return {
                "model": TTS_MODEL_NAME,
                "input": {
                    "text": text,
                    "voice": TTS_VOICE,
                    "language_type": TTS_LANGUAGE_TYPE,
                },
            }
        return {
            "model": TTS_MODEL_NAME,
            "input": text,
            "voice": TTS_VOICE,
            "response_format": TTS_RESPONSE_FORMAT,
            "stream": False,
        }

    async def _extract_audio_from_json(self, payload: dict) -> bytes:
        """Extract audio bytes from JSON responses."""
        status_code = payload.get("status_code")
        if status_code is not None and int(status_code) >= 400:
            raise RuntimeError(
                "TTS API returned an error JSON response. "
                f"status={status_code} | code={payload.get('code')} | "
                f"message={payload.get('message')}"
            )

        audio_url = self._extract_audio_url(payload)
        if audio_url:
            logger.info("TTS API returned audio URL; downloading generated audio")
            return await self._download_audio(audio_url)

        candidates = [
            payload.get("audio"),
            payload.get("data", {}).get("audio") if isinstance(payload.get("data"), dict) else None,
            payload.get("output", {}).get("audio") if isinstance(payload.get("output"), dict) else None,
        ]
        audio_base64 = next(
            (
                value
                for value in candidates
                if isinstance(value, str) and not value.startswith(("http://", "https://"))
            ),
            None,
        )
        if audio_base64:
            return base64.b64decode(audio_base64)

        raise RuntimeError(
            "TTS API JSON response does not contain audio or audio URL. "
            f"shape={self._describe_json_shape(payload)} | "
            f"response={json.dumps(payload, ensure_ascii=False)[:500]}"
        )

    @staticmethod
    def _extract_audio_url(payload: dict) -> str:
        output = payload.get("output")
        if isinstance(output, dict):
            audio = output.get("audio")
            if isinstance(audio, dict):
                return str(audio.get("url") or audio.get("audio_url") or "")
            if isinstance(audio, str) and audio.startswith(("http://", "https://")):
                return audio
            for key in ("audio_url", "url"):
                if output.get(key):
                    return str(output[key])

        data = payload.get("data")
        if isinstance(data, dict):
            for key in ("audio_url", "url"):
                if data.get(key):
                    return str(data[key])

        for key in ("audio_url", "url"):
            if payload.get(key):
                return str(payload[key])
        return ""

    async def _download_audio(self, audio_url: str) -> bytes:
        async with httpx.AsyncClient(timeout=httpx.Timeout(TTS_API_TIMEOUT)) as client:
            response = await client.get(audio_url)
            if response.status_code >= 400:
                raise RuntimeError(
                    "TTS audio download failed. "
                    f"status={response.status_code} | url={audio_url} | response={response.text[:300]}"
                )
            return response.content

    @staticmethod
    def _describe_json_shape(payload: dict) -> str:
        output = payload.get("output")
        data = payload.get("data")
        output_audio_type = None
        data_audio_type = None
        if isinstance(output, dict):
            output_audio_type = type(output.get("audio")).__name__
        if isinstance(data, dict):
            data_audio_type = type(data.get("audio")).__name__
        return (
            f"keys={list(payload.keys())} | "
            f"outputAudioType={output_audio_type} | "
            f"dataAudioType={data_audio_type}"
        )

    def _normalize_audio_bytes(self, audio_bytes: bytes) -> bytes:
        """Normalize API response to bytes expected by the badge WebSocket path."""
        fmt = TTS_RESPONSE_FORMAT.lower()
        if fmt == "pcm":
            return audio_bytes

        if fmt == "wav" or audio_bytes[:4] == b"RIFF":
            return self._wav_to_pcm(audio_bytes)

        logger.warning(
            f"TTS response format is {fmt}; returning raw audio bytes. "
            "Use pcm or wav when the badge expects raw PCM frames."
        )
        return audio_bytes

    @staticmethod
    def _pcm_duration_seconds(pcm_bytes: bytes) -> float:
        bytes_per_second = (
            TTS_TARGET_SAMPLE_RATE * TTS_TARGET_SAMPLE_WIDTH * TTS_TARGET_CHANNELS
        )
        if bytes_per_second <= 0:
            return 0.0
        return len(pcm_bytes) / bytes_per_second

    @staticmethod
    def _wav_to_pcm(wav_bytes: bytes) -> bytes:
        """Strip WAV header and resample to configured sample rate when needed."""
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            pcm_bytes = wf.readframes(wf.getnframes())

        if channels != 1 or sample_width != 2:
            logger.warning(
                f"TTS WAV is not 16-bit mono | channels={channels} | "
                f"sampleWidth={sample_width}"
            )

        if sample_rate == TTS_TARGET_SAMPLE_RATE:
            return pcm_bytes

        return PiperTTSManager._resample_pcm16_mono(
            pcm_bytes,
            source_sample_rate=sample_rate,
            target_sample_rate=TTS_TARGET_SAMPLE_RATE,
        )

    def is_available(self) -> bool:
        return self._is_loaded and self._client is not None and not self._client.is_closed

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
        self._client = None
        self._is_loaded = False

    def release(self) -> None:
        """Compatibility wrapper used by the existing FastAPI shutdown path."""
        client = self._client
        self._client = None
        self._is_loaded = False
        if client is None or client.is_closed:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(client.aclose())
        except RuntimeError:
            logger.warning("TTS client release called without a running event loop")
