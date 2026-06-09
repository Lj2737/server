"""Gateway service configuration.

All local environment variables are loaded from gateway-service/.env by
env_loader.load_env_file().
"""

from typing import Dict, List
import os

from env_loader import load_env_file


load_env_file()


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Gateway server
GATEWAY_PORT: int = _env_int("GATEWAY_PORT", 8090)
GATEWAY_HOST: str = _env_str("GATEWAY_HOST", "0.0.0.0")
GATEWAY_NAME: str = _env_str("GATEWAY_NAME", "GW-001")
GATEWAY_REGION: str = _env_str("GATEWAY_REGION", "全店区域")
GATEWAY_ROLE: str = _env_str("GATEWAY_ROLE", "主网关")
GATEWAY_FIRMWARE_VERSION: str = _env_str("GATEWAY_FIRMWARE_VERSION", "v2.1.3")
GATEWAY_MANAGED_DEVICE_COUNT: int = _env_int("GATEWAY_MANAGED_DEVICE_COUNT", 0)


# Compute nodes
COMPUTE_NODES: List[str] = [
    node.strip()
    for node in _env_str("COMPUTE_NODES", "127.0.0.1:8091").split(",")
    if node.strip()
]
MAX_CONCURRENT_PER_NODE: int = _env_int("MAX_CONCURRENT_PER_NODE", 5)
MAX_CONCURRENT_TOTAL: int = _env_int("MAX_CONCURRENT_TOTAL", 20)


# Health checks
HEALTH_CHECK_INTERVAL: int = _env_int("HEALTH_CHECK_INTERVAL", 5)
HEALTH_CHECK_FAIL_THRESHOLD: int = _env_int("HEALTH_CHECK_FAIL_THRESHOLD", 2)
HEALTH_CHECK_RECOVER_THRESHOLD: int = _env_int("HEALTH_CHECK_RECOVER_THRESHOLD", 3)
HEALTH_CHECK_TIMEOUT: float = _env_float("HEALTH_CHECK_TIMEOUT", 3.0)


# HTTP client
REQUEST_TIMEOUT: float = _env_float("REQUEST_TIMEOUT", 10.0)
HTTP_CLIENT_MAX_CONNECTIONS: int = _env_int("HTTP_CLIENT_MAX_CONNECTIONS", 20)
HTTP_CLIENT_MAX_KEEPALIVE_CONNECTIONS: int = _env_int(
    "HTTP_CLIENT_MAX_KEEPALIVE_CONNECTIONS",
    10,
)


# Gateway-to-compute route mapping
ROUTE_MAPPING: Dict[str, str] = {
    "/badge/v1/gateway/behavior-recognition": "/badge/v1/internal/algorithm/inference/behavior-recognition",
}


# Logging
LOG_DIR: str = _env_str("LOG_DIR", "logs")
LOG_FILE_FORMAT: str = _env_str("LOG_FILE_FORMAT", "gateway_{time:YYYY-MM-DD}.log")
LOG_RETENTION_DAYS: str = _env_str("LOG_RETENTION_DAYS", "7 days")
LOG_ROTATION: str = _env_str("LOG_ROTATION", "00:00")
LOG_LEVEL: str = _env_str("LOG_LEVEL", "INFO")


# Common enums
class BehaviorType:
    STANDARD = "STANDARD"
    ABNORMAL = "ABNORMAL"
    CUSTOMER = "CUSTOMER"


class DimensionType:
    STRENGTH = "STRENGTH"
    WEAKNESS = "WEAKNESS"


class ConfigType:
    KEYWORD = "KEYWORD"


class AlarmStatus:
    ACTIVE = "ACTIVE"
    RECOVERED = "RECOVERED"


class EventType:
    HEARTBEAT = "HEARTBEAT"
    ALARM = "ALARM"


class BusinessType:
    BEHAVIOR = "BEHAVIOR"
    RECORDING = "RECORDING"


# Time formats
DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT: str = "%Y-%m-%d"


# Idempotency and request tracing
IDEMPOTENT_KEY_CALLBACK: str = "eventId"
IDEMPOTENT_KEY_REQUEST: str = "requestId"
REQUEST_ID_HEADER: str = "X-Request-ID"


# Backend service
BACKEND_BASE_URL: str = _env_str("BACKEND_BASE_URL", "http://127.0.0.1:8080")
BACKEND_AUTH_TOKEN: str = _env_str("BACKEND_AUTH_TOKEN", "")
BACKEND_AUTH_HEADER: str = _env_str("BACKEND_AUTH_HEADER", "Authorization")
BACKEND_REQUEST_TIMEOUT: float = _env_float("BACKEND_REQUEST_TIMEOUT", 10.0)
BACKEND_MAX_RETRIES: int = _env_int("BACKEND_MAX_RETRIES", 3)
BACKEND_RETRY_DELAYS: List[float] = [1.0, 3.0, 5.0]
BACKEND_HTTP_MAX_CONNECTIONS: int = _env_int("BACKEND_HTTP_MAX_CONNECTIONS", 10)
BACKEND_HTTP_MAX_KEEPALIVE_CONNECTIONS: int = _env_int(
    "BACKEND_HTTP_MAX_KEEPALIVE_CONNECTIONS",
    5,
)


# Temporary audio clips
AUDIO_TEMP_DIR: str = _env_str("AUDIO_TEMP_DIR", "temp/audio_clips")
AUDIO_TEMP_FILE_TTL: int = _env_int("AUDIO_TEMP_FILE_TTL", 3600)
AUDIO_TEMP_CLEANUP_INTERVAL: int = _env_int("AUDIO_TEMP_CLEANUP_INTERVAL", 3600)


# Backend callback paths
BEHAVIOR_CALLBACK_PATH: str = _env_str(
    "BEHAVIOR_CALLBACK_PATH",
    "/badge/v1/internal/ai/voice-behaviors",
)
DIALOG_COMPLETION_CALLBACK_PATH: str = _env_str(
    "DIALOG_COMPLETION_CALLBACK_PATH",
    "/badge/v1/internal/ai/dialog-completions",
)
RECORDING_UPLOAD_PATH: str = _env_str(
    "RECORDING_UPLOAD_PATH",
    "/badge/v1/internal/ai/recordings",
)
RECORDING_UPLOAD_TIMEOUT: float = _env_float("RECORDING_UPLOAD_TIMEOUT", 30.0)


# Device event forwarding to backend
DEVICE_EVENT_FORWARD_PATH: str = _env_str(
    "DEVICE_EVENT_FORWARD_PATH",
    "/badge/v1/internal/ai/device-events",
)
DEVICE_EVENT_FORWARD_MAX_RETRIES: int = _env_int("DEVICE_EVENT_FORWARD_MAX_RETRIES", 2)
DEVICE_EVENT_FORWARD_RETRY_INTERVAL: float = _env_float(
    "DEVICE_EVENT_FORWARD_RETRY_INTERVAL",
    1.0,
)


# Internal compute callbacks
DIALOG_COMPLETED_INTERNAL_PATH: str = _env_str(
    "DIALOG_COMPLETED_INTERNAL_PATH",
    "/badge/v1/internal/algorithm/dialog-completed",
)
KNOWLEDGE_BASE_INTERNAL_PATH: str = _env_str(
    "KNOWLEDGE_BASE_INTERNAL_PATH",
    "/badge/v1/internal/algorithm/knowledge-base",
)


# Knowledge base cache and query path
KNOWLEDGE_BASE_CACHE_TTL: int = _env_int("KNOWLEDGE_BASE_CACHE_TTL", 86400)
KNOWLEDGE_BASE_CACHE_CLEANUP_INTERVAL: int = _env_int(
    "KNOWLEDGE_BASE_CACHE_CLEANUP_INTERVAL",
    3600,
)
KNOWLEDGE_BASE_QUERY_PATH: str = _env_str(
    "KNOWLEDGE_BASE_QUERY_PATH",
    "/badge/v1/internal/ai/devices/knowledge-base",
)


# Diagnosis
DIAGNOSIS_INTERNAL_PATH: str = _env_str(
    "DIAGNOSIS_INTERNAL_PATH",
    "/badge/v1/internal/algorithm/inference/diagnosis-summary",
)
STORE_DIAGNOSIS_INTERNAL_PATH: str = _env_str(
    "STORE_DIAGNOSIS_INTERNAL_PATH",
    "/badge/v1/internal/algorithm/inference/store-diagnosis-summary",
)
DIAGNOSIS_REQUEST_TIMEOUT: float = _env_float("DIAGNOSIS_REQUEST_TIMEOUT", 60.0)


# AI dialog
AI_DIALOG_INTERNAL_PATH: str = _env_str(
    "AI_DIALOG_INTERNAL_PATH",
    "/badge/v1/internal/algorithm/dialog/ai-chat",
)
AI_DIALOG_REQUEST_TIMEOUT: float = _env_float("AI_DIALOG_REQUEST_TIMEOUT", 90.0)
AI_DIALOG_MAX_AUDIO_SECONDS: int = _env_int("AI_DIALOG_MAX_AUDIO_SECONDS", 60)
AI_DIALOG_SAMPLE_RATE: int = _env_int("AI_DIALOG_SAMPLE_RATE", 16000)
AI_DIALOG_SAMPLE_WIDTH: int = _env_int("AI_DIALOG_SAMPLE_WIDTH", 2)
AI_DIALOG_CHANNELS: int = _env_int("AI_DIALOG_CHANNELS", 1)


# Config sync
CONFIG_SYNC_INTERNAL_PATH: str = _env_str(
    "CONFIG_SYNC_INTERNAL_PATH",
    "/badge/v1/internal/algorithm/config/sync",
)
CONFIG_SYNC_REQUEST_TIMEOUT: float = _env_float("CONFIG_SYNC_REQUEST_TIMEOUT", 10.0)


# Compute TTS endpoint
TTS_REQUEST_TIMEOUT: float = _env_float("TTS_REQUEST_TIMEOUT", 30.0)
TTS_INTERNAL_PATH: str = _env_str(
    "TTS_INTERNAL_PATH",
    "/badge/v1/internal/algorithm/tts/broadcast",
)


# External TTS API
TTS_API_BASE_URL: str = _env_str(
    "TTS_API_BASE_URL",
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
)
TTS_API_KEY: str = _env_str("TTS_API_KEY")
TTS_MODEL_NAME: str = _env_str("TTS_MODEL_NAME", "qwen3-tts-flash")
TTS_REALTIME_MODEL_NAME: str = _env_str("TTS_REALTIME_MODEL_NAME", "qwen3-tts-flash-realtime")
TTS_REALTIME_WS_URL: str = _env_str(
    "TTS_REALTIME_WS_URL",
    "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
)
TTS_REALTIME_RESPONSE_FORMAT: str = _env_str(
    "TTS_REALTIME_RESPONSE_FORMAT",
    "PCM_24000HZ_MONO_16BIT",
)
TTS_VOICE: str = _env_str("TTS_VOICE", "Cherry")
TTS_REALTIME_VOICE: str = _env_str("TTS_REALTIME_VOICE", TTS_VOICE)
TTS_REALTIME_SPEECH_RATE: float = _env_float("TTS_REALTIME_SPEECH_RATE", 1.2)
TTS_LANGUAGE_TYPE: str = _env_str("TTS_LANGUAGE_TYPE", "Chinese")
TTS_API_TIMEOUT: float = _env_float("TTS_API_TIMEOUT", 60.0)
TTS_RESPONSE_FORMAT: str = _env_str("TTS_RESPONSE_FORMAT", "wav")
TTS_TARGET_SAMPLE_RATE: int = _env_int("TTS_TARGET_SAMPLE_RATE", 16000)
TTS_TARGET_SAMPLE_WIDTH: int = _env_int("TTS_TARGET_SAMPLE_WIDTH", 2)
TTS_TARGET_CHANNELS: int = _env_int("TTS_TARGET_CHANNELS", 1)
TTS_PUSH_CHUNK_SIZE: int = _env_int("TTS_PUSH_CHUNK_SIZE", 3200)
TTS_REALTIME_PUSH: bool = _env_bool("TTS_REALTIME_PUSH", True)
TTS_PUSH_SPEED: float = _env_float("TTS_PUSH_SPEED", 1.0)


# WebSocket
WS_HEARTBEAT_INTERVAL: int = _env_int("WS_HEARTBEAT_INTERVAL", 5)
WS_HEARTBEAT_FAIL_THRESHOLD: int = _env_int("WS_HEARTBEAT_FAIL_THRESHOLD", 2)
WS_PING_TIMEOUT: float = _env_float("WS_PING_TIMEOUT", 5.0)


# Broadcast validation
BROADCAST_DEVICE_NO_MAX_LENGTH: int = _env_int("BROADCAST_DEVICE_NO_MAX_LENGTH", 20)
BROADCAST_CONTENT_MAX_LENGTH: int = _env_int("BROADCAST_CONTENT_MAX_LENGTH", 200)
BROADCAST_MAX_AUDIO_SECONDS: float = _env_float("BROADCAST_MAX_AUDIO_SECONDS", 60.0)
BROADCAST_QUEUE_MAX_SIZE: int = _env_int("BROADCAST_QUEUE_MAX_SIZE", 5)


# Hardware API validation
HARDWARE_API_PREFIX: str = _env_str("HARDWARE_API_PREFIX", "/badge/v1/internal/hardware")
DEVICE_NO_MAX_LENGTH: int = _env_int("DEVICE_NO_MAX_LENGTH", 20)
DEVICE_NO_PATTERN: str = _env_str("DEVICE_NO_PATTERN", r"^[a-zA-Z0-9_]+$")
BATTERY_LEVEL_MIN: int = _env_int("BATTERY_LEVEL_MIN", 0)
BATTERY_LEVEL_MAX: int = _env_int("BATTERY_LEVEL_MAX", 100)
SIGNAL_LEVEL_MIN: int = _env_int("SIGNAL_LEVEL_MIN", 0)
SIGNAL_LEVEL_MAX: int = _env_int("SIGNAL_LEVEL_MAX", 5)


# Device status cache
DEVICE_STATUS_CACHE_TTL: int = _env_int("DEVICE_STATUS_CACHE_TTL", 600)
DEVICE_STATUS_CLEANUP_INTERVAL: int = _env_int("DEVICE_STATUS_CLEANUP_INTERVAL", 60)


# Raw audio uploads from hardware
RAW_AUDIO_TEMP_DIR: str = _env_str("RAW_AUDIO_TEMP_DIR", "temp/raw_uploads")
RAW_AUDIO_MAX_FILE_SIZE: int = _env_int("RAW_AUDIO_MAX_FILE_SIZE", 10 * 1024 * 1024)
RAW_AUDIO_TEMP_FILE_TTL: int = _env_int("RAW_AUDIO_TEMP_FILE_TTL", 3600)
RAW_AUDIO_CLEANUP_INTERVAL: int = _env_int("RAW_AUDIO_CLEANUP_INTERVAL", 3600)
RAW_AUDIO_METADATA_MAX_LENGTH: int = _env_int("RAW_AUDIO_METADATA_MAX_LENGTH", 4096)
WAV_HEADER_MAGIC: bytes = b"RIFF"
RAW_AUDIO_FORWARD_TIMEOUT: float = _env_float("RAW_AUDIO_FORWARD_TIMEOUT", 30.0)
