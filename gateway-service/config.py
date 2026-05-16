"""
智能胸牌服务管理系统 - 主网关节点配置文件
所有配置项集中管理，禁止硬编码
适应部署架构：1台主树莓派（网关） + 4台从树莓派（算力）

v3.2更新：
- 硬件接收路径恢复为 /internal/badge/hardware 前缀（硬件→算法，非算法→后端）
- 新增算力节点→主网关内部接口路径（dialog-completed、knowledge-base）
- 新增知识库缓存配置
- 新增算法查询设备知识库ID接口路径
- 录音上传接口已废弃（v3.1合并到voice-behaviors，配置项已移除）

v3.1更新：
- 接口路径统一为 /badge/v1/ 前缀（算法→后端路径）
- 异常行为片段录音合并到 voice-behaviors 接口（multipart/form-data）
- 新增 AI对话完成回调接口（dialog-completions）
"""
from typing import List, Dict


# ==================== 服务配置 ====================
# 网关服务监听端口（对外唯一暴露端口，仅对接后端）
GATEWAY_PORT: int = 8090
# 网关服务监听地址
GATEWAY_HOST: str = "0.0.0.0"


# ==================== 算力节点配置 ====================
# 算力节点列表（固定4台，对内端口8091）
# 【本地测试时改为127.0.0.1:8091，生产部署改回树莓派IP】
COMPUTE_NODES: List[str] = [
    "127.0.0.1:8091",
    # "192.168.1.101:8091",  # 树莓派算力节点1
    # "192.168.1.102:8091",  # 树莓派算力节点2
    # "192.168.1.103:8091",  # 树莓派算力节点3
    # "192.168.1.104:8091",  # 树莓派算力节点4
]

# 单台算力节点最大并发数
MAX_CONCURRENT_PER_NODE: int = 5
# 全局最大并发数（4台 × 5路 = 20路）
MAX_CONCURRENT_TOTAL: int = 20


# ==================== 健康检查配置 ====================
# 健康检查间隔（秒）
HEALTH_CHECK_INTERVAL: int = 5
# 连续失败次数阈值，达到后标记节点不可用并摘除
HEALTH_CHECK_FAIL_THRESHOLD: int = 2
# 连续成功次数阈值，达到后重新加入可用节点池
HEALTH_CHECK_RECOVER_THRESHOLD: int = 3
# 健康检查请求超时（秒）
HEALTH_CHECK_TIMEOUT: float = 3.0


# ==================== 请求转发配置 ====================
# 单请求转发超时时间（秒）
REQUEST_TIMEOUT: float = 10.0
# httpx连接池总大小
HTTP_CLIENT_MAX_CONNECTIONS: int = 20
# httpx单主机最大保持活跃连接数
HTTP_CLIENT_MAX_KEEPALIVE_CONNECTIONS: int = 10


# ==================== 路由映射配置 ====================
# 外部路径 → 算力节点内部路径映射
# 对外接口统一前缀 /api/v1/gateway/，内部统一前缀 /api/v1/internal/inference/
ROUTE_MAPPING: Dict[str, str] = {
    "/api/v1/gateway/behavior-recognition": "/api/v1/internal/inference/behavior-recognition",
}


# ==================== 日志配置 ====================
# 日志文件存放目录
LOG_DIR: str = "logs"
# 日志文件名格式（按天命名）
LOG_FILE_FORMAT: str = "gateway_{time:YYYY-MM-DD}.log"
# 日志保留天数（过期自动清理）
LOG_RETENTION_DAYS: str = "7 days"
# 日志轮转时间（每天0点分割）
LOG_ROTATION: str = "00:00"
# 日志级别
LOG_LEVEL: str = "INFO"


# ==================== 业务枚举值（强制使用，禁止自定义） ====================

class BehaviorType:
    """行为类型枚举"""
    STANDARD = "STANDARD"      # 标准行为
    ABNORMAL = "ABNORMAL"      # 异常行为
    CUSTOMER = "CUSTOMER"      # 顾客负面行为


class DimensionType:
    """维度类型枚举"""
    STRENGTH = "STRENGTH"      # 优势维度
    WEAKNESS = "WEAKNESS"      # 薄弱维度


class ConfigType:
    """配置类型枚举"""
    KEYWORD = "KEYWORD"        # 词库配置


class AlarmStatus:
    """告警状态枚举"""
    ACTIVE = "ACTIVE"          # 告警生效
    RECOVERED = "RECOVERED"    # 告警恢复


class EventType:
    HEARTBEAT = "HEARTBEAT"   # 心跳状态
    ALARM = "ALARM"           # 硬件告警


# ==================== 时间格式 ====================
# 日期时间格式：yyyy-MM-dd HH:mm:ss
DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"
# 日期格式：yyyy-MM-dd
DATE_FORMAT: str = "%Y-%m-%d"


# ==================== 幂等与重试配置 ====================
# 算法回调后端使用的幂等键Header名
IDEMPOTENT_KEY_CALLBACK = "eventId"
# 后端调用算法使用的幂等键Header名
IDEMPOTENT_KEY_REQUEST = "requestId"
# 请求ID Header名（网关内部使用）
REQUEST_ID_HEADER = "X-Request-ID"


# ==================== 后端接口通用客户端配置 ====================
# 后端服务基础地址（算法侧回调后端的地址，部署时按实际环境修改）
# 【本地测试时改为127.0.0.1:8080，生产部署改回树莓派IP】
BACKEND_BASE_URL: str = "http://127.0.0.1:8080"
# 后端接口鉴权Token（Header方式，部署时按后端分配修改）
BACKEND_AUTH_TOKEN: str = "your-internal-token-here"
# 鉴权Token的Header名
BACKEND_AUTH_HEADER: str = "Authorization"
# 后端请求默认超时时间（秒）
BACKEND_REQUEST_TIMEOUT: float = 10.0
# 后端请求最大重试次数
BACKEND_MAX_RETRIES: int = 3
# 后端请求重试间隔（秒），指数退避基数：1s, 3s, 5s
BACKEND_RETRY_DELAYS: List[float] = [1.0, 3.0, 5.0]
# 后端httpx连接池总大小
BACKEND_HTTP_MAX_CONNECTIONS: int = 10
# 后端httpx单主机最大保持活跃连接数
BACKEND_HTTP_MAX_KEEPALIVE_CONNECTIONS: int = 5


# ==================== 幂等ID生成器配置 ====================
# eventId格式前缀：AI_业务类型_
# 业务类型枚举
class BusinessType:
    """幂等ID业务类型枚举"""
    BEHAVIOR = "BEHAVIOR"      # 行为识别事件
    RECORDING = "RECORDING"    # 录音上传事件（v3.1已废弃，合并到voice-behaviors）


# ==================== 异常音频临时存储配置 ====================
# 临时文件存放目录
AUDIO_TEMP_DIR: str = "temp/audio_clips"
# 临时文件过期时间（秒），超过此时间的文件将被清理
AUDIO_TEMP_FILE_TTL: int = 3600      # 1小时
# 定时清理间隔（秒）
AUDIO_TEMP_CLEANUP_INTERVAL: int = 3600  # 每小时清理一次


# ==================== 算法→后端回调接口路径 ====================
# 语音行为识别结果回调后端接口路径（POST，Content-Type: multipart/form-data）
# v3.1：异常行为片段录音合并到此接口，ABNORMAL必须上传file字段
BEHAVIOR_CALLBACK_PATH: str = "/badge/v1/internal/ai/voice-behaviors"

# AI对话完成回调后端接口路径（POST，Content-Type: application/json）
# v3.1新增接口
DIALOG_COMPLETION_CALLBACK_PATH: str = "/badge/v1/internal/ai/dialog-completions"

# 录音上传后端接口（v3.1已废弃，异常录音合并到voice-behaviors，v3.2移除配置项）
# 如需录音上传功能，请使用 BEHAVIOR_CALLBACK_PATH（/badge/v1/internal/ai/voice-behaviors）


# ==================== 硬件状态透传后端配置 ====================
# 算法→后端透传接口路径：POST /badge/v1/internal/ai/device-events
DEVICE_EVENT_FORWARD_PATH: str = "/badge/v1/internal/ai/device-events"
# 透传最大重试次数（不含首次请求，最多重试2次）
DEVICE_EVENT_FORWARD_MAX_RETRIES: int = 2
# 透传重试间隔（秒），固定1秒
DEVICE_EVENT_FORWARD_RETRY_INTERVAL: float = 1.0


# ==================== 算力节点→主网关内部接口路径 ====================
# AI对话完成内部接口（算力节点调用主网关）
DIALOG_COMPLETED_INTERNAL_PATH: str = "/api/v1/internal/dialog-completed"
# 知识库查询内部接口（算力节点调用主网关）
KNOWLEDGE_BASE_INTERNAL_PATH: str = "/api/v1/internal/knowledge-base"

# ==================== 知识库缓存配置 ====================
# 知识库ID缓存过期时间（秒），默认24小时
KNOWLEDGE_BASE_CACHE_TTL: int = 86400  # 24小时
# 知识库ID缓存清理间隔（秒）
KNOWLEDGE_BASE_CACHE_CLEANUP_INTERVAL: int = 3600  # 1小时

# ==================== 算法查询设备知识库ID接口路径 ====================
# 算法查询设备知识库ID接口路径（算法→后端）
KNOWLEDGE_BASE_QUERY_PATH: str = "/badge/v1/internal/ai/devices/knowledge-base"

# ==================== 功能3：AI时段诊断总结配置 ====================
# AI诊断请求转发到算力节点的内部路径
DIAGNOSIS_INTERNAL_PATH: str = "/api/v1/internal/inference/diagnosis-summary"
# AI诊断请求超时时间（秒），LLM推理耗时较长，设置为60秒
DIAGNOSIS_REQUEST_TIMEOUT: float = 60.0


# ==================== 功能4：词库配置同步配置 ====================
# 词库配置同步到算力节点的内部路径
CONFIG_SYNC_INTERNAL_PATH: str = "/api/v1/internal/config/sync"
# 词库配置同步请求超时时间（秒）
CONFIG_SYNC_REQUEST_TIMEOUT: float = 10.0


# ==================== 功能5：值班播报文字转语音配置 ====================
# TTS播报请求超时时间（秒）
TTS_REQUEST_TIMEOUT: float = 30.0
# TTS播报请求转发到算力节点的内部路径
TTS_INTERNAL_PATH: str = "/api/v1/internal/tts/broadcast"


# ==================== Piper TTS本地部署配置 ====================
# Piper TTS模型文件路径（ONNX格式）
PIPER_MODEL_PATH: str = "compute-service/models/zh_CN-huayan-medium/model.onnx"
# Piper TTS模型配置文件路径（JSON格式）
PIPER_CONFIG_PATH: str = "compute-service/models/zh_CN-huayan-medium/model.onnx.json"
# 是否使用CUDA加速（树莓派仅CPU，强制False）
PIPER_USE_CUDA: bool = False
# 目标音频采样率（Hz），硬件直接播放要求16000Hz
PIPER_TARGET_SAMPLE_RATE: int = 16000
# 目标音频采样位深（字节），16bit = 2字节
PIPER_TARGET_SAMPLE_WIDTH: int = 2
# 目标音频声道数，单声道
PIPER_TARGET_CHANNELS: int = 1
# Piper合成参数：噪声缩放
PIPER_NOISE_SCALE: float = 0.667
# Piper合成参数：语速缩放
PIPER_LENGTH_SCALE: float = 1.0
# Piper合成参数：音量倍数
PIPER_VOLUME: float = 1.0


# ==================== WebSocket设备连接管理配置 ====================
# 心跳发送间隔（秒），每30秒发一次ping
WS_HEARTBEAT_INTERVAL: int = 30
# 心跳失败阈值，连续3次未回复pong判定离线
WS_HEARTBEAT_FAIL_THRESHOLD: int = 3
# 单次ping等待pong超时时间（秒）
WS_PING_TIMEOUT: float = 5.0


# ==================== 值班播报业务限制配置 ====================
# 设备编号最大长度
BROADCAST_DEVICE_NO_MAX_LENGTH: int = 20
# 播报内容最大字数
BROADCAST_CONTENT_MAX_LENGTH: int = 200


# ==================== 硬件状态上报配置 ====================
# 设备编号最大长度
DEVICE_NO_MAX_LENGTH: int = 20
# 设备编号正则：仅支持字母、数字、下划线
DEVICE_NO_PATTERN: str = r"^[a-zA-Z0-9_]+$"
# 电量百分比取值范围
BATTERY_LEVEL_MIN: int = 0
BATTERY_LEVEL_MAX: int = 100
# 信号等级取值范围
SIGNAL_LEVEL_MIN: int = 0
SIGNAL_LEVEL_MAX: int = 5

# ==================== 设备状态缓存配置 ====================
# 设备状态缓存过期时间（秒）
DEVICE_STATUS_CACHE_TTL: int = 600       # 10分钟
# 缓存定时清理间隔（秒）
DEVICE_STATUS_CLEANUP_INTERVAL: int = 60


# ==================== 原始异常语音上传配置 ====================
# 原始音频临时文件存放目录
RAW_AUDIO_TEMP_DIR: str = "temp/raw_uploads"
# 原始音频文件最大允许大小（字节），默认10MB
RAW_AUDIO_MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
# 临时文件过期时间（秒）
RAW_AUDIO_TEMP_FILE_TTL: int = 3600       # 1小时
# 定时清理间隔（秒）
RAW_AUDIO_CLEANUP_INTERVAL: int = 3600
# metadata JSON字段最大长度（字符数）
RAW_AUDIO_METADATA_MAX_LENGTH: int = 4096
# WAV音频文件头魔数（RIFF）
WAV_HEADER_MAGIC: bytes = b"RIFF"
# 算力节点行为识别转发超时时间（秒）
RAW_AUDIO_FORWARD_TIMEOUT: float = 30.0
