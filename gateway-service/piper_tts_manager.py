"""TTS API manager.

The class name is kept as PiperTTSManager for compatibility with existing
router code. The implementation calls a non-streaming speech API, normalizes
the returned audio to PCM when possible, then yields chunks for WebSocket push.
"""
from __future__ import annotations

import base64
import asyncio
import io
import json
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
    TTS_RESPONSE_FORMAT,
    TTS_TARGET_SAMPLE_RATE,
    TTS_VOICE,
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

        for start in range(0, len(pcm_bytes), TTS_PUSH_CHUNK_SIZE):
            chunk = pcm_bytes[start:start + TTS_PUSH_CHUNK_SIZE]
            if chunk:
                yield chunk

        logger.info(
            f"TTS synthesis completed | textLength={len(text)} | "
            f"audioBytes={len(pcm_bytes)}"
        )

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

        pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        target_len = int(len(pcm_array) * TTS_TARGET_SAMPLE_RATE / sample_rate)
        if target_len <= 0:
            return b""

        try:
            from scipy.signal import resample

            resampled = resample(pcm_array, target_len)
            return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()
        except Exception as exc:
            logger.warning(f"TTS WAV resample failed; using original PCM | error={exc}")
            return pcm_bytes

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
