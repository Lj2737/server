"""
智能胸牌服务管理系统 - 主网关节点配置文件
所有配置项集中管理，禁止硬编码
适应部署架构：1台主树莓派（网关） + 4台从树莓派（算力）
"""
from typing import List, Dict


# ==================== 服务配置 ====================
# 网关服务监听端口（对外唯一暴露端口，仅对接后端）
GATEWAY_PORT: int = 8090
# 网关服务监听地址
GATEWAY_HOST: str = "0.0.0.0"


# ==================== 算力节点配置 ====================
# 算力节点列表（固定4台，对内端口8091）
COMPUTE_NODES: List[str] = [
    "192.168.1.101:8091",
    "192.168.1.102:8091",
    "192.168.1.103:8091",
    "192.168.1.104:8091",
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
    "/api/v1/gateway/diagnosis-summary": "/api/v1/internal/inference/diagnosis-summary",
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
