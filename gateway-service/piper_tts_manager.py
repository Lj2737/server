"""
TTS API manager.

Keeps the old PiperTTSManager class name so existing router code can keep using
the same dependency, but the implementation now calls a non-streaming TTS API
and yields the returned audio in chunks for WebSocket delivery.
"""
import base64
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
    TTS_MODEL_NAME,
    TTS_PUSH_CHUNK_SIZE,
    TTS_RESPONSE_FORMAT,
    TTS_TARGET_SAMPLE_RATE,
    TTS_VOICE,
)


class PiperTTSManager:
    """
    TTS API管理器 - 单例模式。

    - load_model() 改为校验API配置并创建HTTP客户端
    - synthesize_stream() 调用非流式TTS API，拿到完整音频后按chunk推送
    - 优先要求API返回 pcm/wav，便于直接推送给设备
    """

    _instance: Optional["PiperTTSManager"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._client: Optional[httpx.AsyncClient] = None
        self._is_loaded = False
        self._load_error: Optional[str] = None
        self._initialized = True

    @staticmethod
    def _mask_secret(secret: str) -> str:
        """Return a non-sensitive preview for logs."""
        if not secret:
            return "empty"
        if len(secret) <= 10:
            return f"len={len(secret)}"
        return f"{secret[:6]}...{secret[-4:]}(len={len(secret)})"

    async def load_model(self) -> bool:
        """
        初始化TTS API客户端。

        保持原方法名，避免改动调用方。
        """
        try:
            if not TTS_API_KEY or TTS_API_KEY.startswith("replace-with-"):
                raise RuntimeError("TTS_API_KEY未配置，请在gateway-service/.env中设置")

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
                f"TTS API客户端初始化成功 | model={TTS_MODEL_NAME} | "
                f"url={TTS_API_BASE_URL} | format={TTS_RESPONSE_FORMAT} | "
                f"key={self._mask_secret(TTS_API_KEY)}"
            )
            return True
        except Exception as e:
            self._is_loaded = False
            self._load_error = str(e)
            logger.error(f"TTS API客户端初始化失败 | 错误={e}")
            return False

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        非流式TTS API合成后按chunk输出。

        Args:
            text: 播报文本

        Yields:
            适合WebSocket推送的音频字节。若API返回wav，会剥离WAV头并输出PCM。
        """
        if not self._is_loaded or self._client is None:
            raise RuntimeError(f"TTS API客户端未初始化 | 错误={self._load_error}")

        if not text or not text.strip():
            logger.warning("TTS合成文本为空，跳过")
            return

        audio_bytes = await self._request_tts(text.strip())
        pcm_bytes = self._normalize_audio_bytes(audio_bytes)

        for start in range(0, len(pcm_bytes), TTS_PUSH_CHUNK_SIZE):
            chunk = pcm_bytes[start:start + TTS_PUSH_CHUNK_SIZE]
            if chunk:
                yield chunk

        logger.info(
            f"TTS合成完成 | 文本长度={len(text)} | "
            f"音频大小={len(pcm_bytes)} bytes"
        )

    async def _request_tts(self, text: str) -> bytes:
        """Call the non-streaming speech API and return audio bytes."""
        if self._client is None:
            raise RuntimeError("TTS API客户端未初始化")

        payload = {
            "model": TTS_MODEL_NAME,
            "input": text,
            "voice": TTS_VOICE,
            "response_format": TTS_RESPONSE_FORMAT,
            "stream": False,
        }
        response = await self._client.post(TTS_API_BASE_URL, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if response.status_code == 401:
                raise RuntimeError(
                    "TTS API认证失败(401)。请检查gateway-service/.env中的"
                    f"TTS_API_KEY是否有效；当前key={self._mask_secret(TTS_API_KEY)}，"
                    f"url={TTS_API_BASE_URL}"
                ) from e
            raise

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return self._extract_audio_from_json(response.json())
        return response.content

    @staticmethod
    def _extract_audio_from_json(payload: dict) -> bytes:
        """
        Accept common JSON audio response shapes.

        Supported examples:
        - {"audio": "<base64>"}
        - {"data": {"audio": "<base64>"}}
        - {"output": {"audio": "<base64>"}}
        """
        candidates = [
            payload.get("audio"),
            payload.get("data", {}).get("audio") if isinstance(payload.get("data"), dict) else None,
            payload.get("output", {}).get("audio") if isinstance(payload.get("output"), dict) else None,
        ]
        audio_base64 = next((value for value in candidates if value), None)
        if not audio_base64:
            raise RuntimeError(f"TTS API JSON响应中未找到audio字段: {json.dumps(payload, ensure_ascii=False)[:300]}")
        return base64.b64decode(audio_base64)

    def _normalize_audio_bytes(self, audio_bytes: bytes) -> bytes:
        """
        Normalize API response to PCM bytes.

        - wav: strip WAV header and resample if needed
        - pcm: pass through
        - other formats: return raw bytes; device side must support that format
        """
        fmt = TTS_RESPONSE_FORMAT.lower()
        if fmt == "pcm":
            return audio_bytes

        if fmt == "wav" or audio_bytes[:4] == b"RIFF":
            return self._wav_to_pcm(audio_bytes)

        logger.warning(
            f"TTS返回格式={fmt}，当前未做解码，原始字节将直接推送；"
            "建议将TTS_RESPONSE_FORMAT配置为pcm或wav"
        )
        return audio_bytes

    @staticmethod
    def _wav_to_pcm(wav_bytes: bytes) -> bytes:
        """Convert WAV bytes to target-sample-rate PCM bytes."""
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            pcm_bytes = wf.readframes(wf.getnframes())

        if channels != 1 or sample_width != 2:
            logger.warning(
                f"TTS WAV格式非目标PCM | channels={channels} | sample_width={sample_width}"
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
        except Exception as e:
            logger.warning(f"TTS WAV重采样失败，使用原始PCM | 错误={e}")
            return pcm_bytes

    def is_available(self) -> bool:
        """TTS服务是否可用。"""
        return self._is_loaded and self._client is not None and not self._client.is_closed

    @property
    def load_error(self) -> Optional[str]:
        """TTS API初始化失败原因。"""
        return self._load_error

    def release(self) -> None:
        """关闭TTS API客户端。"""
        if self._client is not None and not self._client.is_closed:
            try:
                import asyncio
                asyncio.create_task(self._client.aclose())
            except RuntimeError:
                pass
        self._client = None
        self._is_loaded = False
        logger.info("TTS API客户端已释放")
