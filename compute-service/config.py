"""
智能胸牌服务管理系统 - 算力节点配置文件
所有配置项集中管理，禁止硬编码
部署架构：4台从树莓派（算力节点），每台完整部署ASR并通过API调用LLM
"""
import os
from typing import List, Dict

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


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: List[str]) -> List[str]:
    value = os.getenv(name)
    if value is None:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


# ==================== 服务配置 ====================
# 算力节点监听端口（对内暴露，仅与主网关通信）
SERVICE_PORT: int = _env_int("SERVICE_PORT", 8091)
# 算力节点监听地址
SERVICE_HOST: str = _env_str("SERVICE_HOST", "0.0.0.0")

# 当前节点IP（用于健康检查返回，部署时按实际IP修改）
NODE_IP: str = _env_str("NODE_IP", "127.0.0.1")


# ==================== 调试模式 ====================
# 开发调试时设为True，放行所有IP访问（生产环境必须设为False）
DEBUG_MODE: bool = _env_bool("DEBUG_MODE", True)


# ==================== 主网关IP白名单 ====================
# 仅允许主网关节点IP访问，禁止其他IP请求（跨域限制）
# 部署时将192.168.1.100替换为主网关实际IP
# 注意：DEBUG_MODE=True时白名单不生效，仅文档路径和/health始终放行
GATEWAY_IP_WHITELIST: List[str] = [
    "192.168.1.100",  # 主网关节点IP
    "127.0.0.1",      # 本地回环，调试用
    "192.168.43.11",
]
GATEWAY_IP_WHITELIST = _env_list("GATEWAY_IP_WHITELIST", GATEWAY_IP_WHITELIST)


# ==================== 并发控制 ====================
# 单节点最大并发推理路数（树莓派5B/8G资源限制）
MAX_CONCURRENT: int = 5


# ==================== 幂等缓存配置 ====================
# 幂等性缓存TTL（秒），5分钟内相同请求直接返回缓存
CACHE_TTL: int = 300
# 缓存清理间隔（秒）
CACHE_CLEANUP_INTERVAL: int = 60


# ==================== ASR模型配置（sherpa-onnx-sense-voice-small） ====================
# 模型文件目录（包含model.onnx.int8和tokens.txt）
ASR_MODEL_DIR: str = _env_str("ASR_MODEL_DIR", "E:/Project-python/server/compute-service/models/sherpa-onnx-sense-voice-small")
# 采样率（固定16000Hz，直接适配sense-voice输入要求）
ASR_SAMPLE_RATE: int = 16000
# 识别语言：zh=中文
ASR_LANGUAGE: str = "zh"
# 是否启用逆文本归一化（自动加标点）
ASR_USE_ITN: bool = True
# 推理线程数
ASR_NUM_THREADS: int = 4
# 推理设备（树莓派仅支持CPU）
ASR_PROVIDER: str = "cpu"
# ASR推理超时（秒）
ASR_TIMEOUT: float = 10.0


# ==================== LLM API配置 ====================
# OpenAI-compatible base URL, for example https://api.deepseek.com.
# If a full /chat/completions URL is configured, the caller keeps using it as-is.
LLM_API_BASE_URL: str = _env_str("LLM_API_BASE_URL", "https://api.deepseek.com")
# API Key必须通过环境变量或 compute-service/.env 配置；兼容DEEPSEEK_API_KEY别名
LLM_API_KEY: str = _env_str("LLM_API_KEY") or _env_str("DEEPSEEK_API_KEY")
# 模型名
LLM_MODEL_NAME: str = _env_str("LLM_MODEL_NAME", "deepseek-v4-flash")
# 请求超时（秒）
LLM_API_TIMEOUT: float = _env_float("LLM_API_TIMEOUT", 60.0)
# Optional DeepSeek reasoning parameters. These mirror the tested OpenAI SDK call:
# reasoning_effort="high", extra_body={"thinking": {"type": "enabled"}}.
LLM_REASONING_EFFORT: str = _env_str("LLM_REASONING_EFFORT", "")
LLM_THINKING_ENABLED: bool = _env_bool("LLM_THINKING_ENABLED", False)


# ==================== FastGPT API配置 ====================
FASTGPT_API_URL: str = _env_str(
    "FASTGPT_API_URL",
    "https://cloud.fastgpt.cn/api/v1/chat/completions",
)
FASTGPT_API_KEY: str = _env_str("FASTGPT_API_KEY")
FASTGPT_API_TIMEOUT: float = _env_float("FASTGPT_API_TIMEOUT", 60.0)
FASTGPT_DATASET_VARIABLE_NAME: str = _env_str("FASTGPT_DATASET_VARIABLE_NAME", "datasetid")
FASTGPT_DATASET_VARIABLE_KEY: str = _env_str("FASTGPT_DATASET_VARIABLE_KEY", "id1")


# ==================== 行为识别LLM推理参数 ====================
# 温度（低随机性，保证输出稳定）
BEHAVIOR_LLM_TEMPERATURE: float = _env_float("BEHAVIOR_LLM_TEMPERATURE", 0.1)
# 最大输出token数
BEHAVIOR_LLM_MAX_TOKENS: int = _env_int("BEHAVIOR_LLM_MAX_TOKENS", 512)
# 停止标记
BEHAVIOR_LLM_STOP: List[str] = ["<|im_end|>"]
# JSON输出校验失败后最大重试次数
BEHAVIOR_LLM_MAX_RETRIES: int = _env_int("BEHAVIOR_LLM_MAX_RETRIES", 1)


# ==================== 诊断总结LLM推理参数 ====================
# 温度（略高于行为识别，允许一定创造性但不过度发散）
DIAGNOSIS_LLM_TEMPERATURE: float = _env_float("DIAGNOSIS_LLM_TEMPERATURE", 0.3)
# 最大输出token数（诊断总结需要更长输出）
DIAGNOSIS_LLM_MAX_TOKENS: int = _env_int("DIAGNOSIS_LLM_MAX_TOKENS", 1024)
# 停止标记
DIAGNOSIS_LLM_STOP: List[str] = ["<|im_end|>"]
# JSON输出校验失败后最大重试次数
DIAGNOSIS_LLM_MAX_RETRIES: int = _env_int("DIAGNOSIS_LLM_MAX_RETRIES", 1)


# ==================== 音频格式要求 ====================
# 采样率：16000Hz
AUDIO_SAMPLE_RATE: int = 16000
# 采样位深：16bit = 2字节
AUDIO_SAMPLE_WIDTH: int = 2
# 声道数：输入支持1-2声道，多声道自动转单声道后送入ASR推理
AUDIO_CHANNELS: int = 1
# 允许的最大输入声道数（超过此值拒绝）
AUDIO_MAX_CHANNELS: int = 2
# 允许的音频格式扩展名
AUDIO_ALLOWED_EXTENSIONS: List[str] = [".wav", ".pcm"]


# ==================== 异常音频裁剪参数 ====================
# 触发前裁剪秒数
ABNORMAL_CLIP_BEFORE_SECONDS: float = 5.0
# 触发后裁剪秒数
ABNORMAL_CLIP_AFTER_SECONDS: float = 10.0
# 裁剪总时长（秒）
ABNORMAL_CLIP_TOTAL_SECONDS: float = ABNORMAL_CLIP_BEFORE_SECONDS + ABNORMAL_CLIP_AFTER_SECONDS


# ==================== 词库配置本地缓存 ====================
# 词库配置JSON文件路径（内存+本地双缓存，重启自动加载）
KEYWORD_CONFIG_FILE: str = "data/keyword_config.json"


# ==================== 日志配置 ====================
# 日志文件存放目录
LOG_DIR: str = "logs"
# 日志文件名格式（按天命名）
LOG_FILE_FORMAT: str = "compute_{time:YYYY-MM-DD}.log"
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


# ==================== 幂等键配置 ====================
# 请求ID Header名（与主网关保持一致）
REQUEST_ID_HEADER = "X-Request-ID"


# ==================== Prompt模板 ====================
# 【修改位置】行为识别 system_prompt —— 调整分析维度和输出格式在此修改
BEHAVIOR_SYSTEM_PROMPT: str = (
    "你是一个专业的餐饮服务行为分析助手。请根据员工的服务语音转写文本，"
    "结合本地词库配置，分析行为类型并输出JSON格式结果。\n"
    "行为类型枚举：STANDARD（标准行为）、ABNORMAL（异常行为）、CUSTOMER（顾客负面行为）\n"
    "输出JSON格式要求：\n"
    "{\n"
    '  "behavior_type": "枚举值",\n'
    '  "summary": "100字以内的行为摘要",\n'
    '  "config_item_id": "命中的配置项ID，未命中时为空字符串",\n'
    '  "keyword_content": "命中的词库关键词内容；精确命中或语义命中都必须填写本地词库配置中已有的关键词，未命中时为空字符串",\n'
    '  "is_abnormal": true或false\n'
    "}\n"
    "规则：可以进行语义匹配，不要求员工原话与词库关键词完全一致；"
    "但keyword_content必须来自本地词库配置中的keywords.content，"
    "且必须选择与员工语音语义最接近的词库关键词，禁止选择明显不相关的词库示例词。\n"
    "注意：仅输出JSON，不要输出任何其他自然语言解释、Markdown格式或代码块标记。"
)

# 【修改位置】行为识别 user_prompt 模板 —— 调整输入变量格式在此修改
BEHAVIOR_USER_PROMPT_TEMPLATE: str = (
    "员工语音转写文本：{asr_text}\n本地词库配置：{keyword_config}"
)

# 【修改位置】诊断总结 system_prompt —— 调整诊断维度和输出格式在此修改
DIAGNOSIS_SYSTEM_PROMPT: str = (
    "你是一个专业的餐饮员工服务能力诊断助手。请根据员工的历史行为数据、评分数据，"
    "生成多维度诊断总结。\n"
    "维度类型枚举：STRENGTH（优势维度）、WEAKNESS（薄弱维度）\n"
    "输出JSON格式要求：\n"
    "{\n"
    '  "summary": "该时间段总体服务表现总结，150字以内",\n'
    '  "dimensions": [\n'
    "    {\n"
    '      "dimension_code": "与入参一致的维度编码",\n'
    '      "dimension_type": "STRENGTH或WEAKNESS",\n'
    '      "summary": "该维度表现总结，100字以内",\n'
    '      "suggestion": "改进建议，仅dimension_type=WEAKNESS时必填，100字以内"\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "注意：仅输出JSON，不要输出任何其他自然语言解释、Markdown格式或代码块标记。"
)

# 【修改位置】诊断总结 user_prompt 模板 —— 调整输入变量格式在此修改
DIAGNOSIS_USER_PROMPT_TEMPLATE: str = (
    "员工编号：{employee_no}\n"
    "时间范围：{start_date}至{end_date}\n"
    "综合评分：{score}\n"
    "维度评分：{dimension_scores}\n"
    "行为统计：{behavior_stats}\n"
    "异常行为列表：{abnormal_behaviors}"
)
