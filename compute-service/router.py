"""
智能胸牌服务管理系统 - 算力节点API路由
核心接口：
1. GET /health - 健康检查
2. POST /badge/v1/internal/algorithm/inference/behavior-recognition - 语音行为识别推理（支持1-2声道，多声道自动转单声道）
3. POST /badge/v1/internal/algorithm/inference/diagnosis-summary - AI时段诊断总结推理
4. POST /badge/v1/internal/algorithm/config/sync - 词库配置同步
"""
import asyncio
import time

from fastapi import APIRouter, UploadFile, File, Form
from loguru import logger

from config import (
    MAX_CONCURRENT,
    NODE_IP,
    BehaviorType,
    DimensionType,
    ConfigType,
)
from models import asr_model, llm_model, get_model_status
from cache import IdempotencyCache
from keyword_config import KeywordConfigManager
from audio_utils import validate_wav_audio, clip_abnormal_audio, ensure_mono_audio
from exception import ComputeException, ErrorCode, ErrorMsg
from schemas import (
    DiagnosisSummaryRequest,
    DiagnosisSummaryResponse,
    DiagnosisResultData,
    DimensionItem,
    ConfigSyncRequest,
    ConfigSyncResponse,
    ConfigSyncResultData,
    HealthCheckResponse,
    BehaviorRecognitionResponse,
    BehaviorResultData,
)

# 创建API路由器
router = APIRouter()

# 全局并发控制信号量（单节点最大5路并发）
_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# 幂等性缓存实例
idempotency_cache = IdempotencyCache()

# 词库配置管理器实例
keyword_config_manager = KeywordConfigManager()

# 当前活跃连接数（用于健康检查返回）
_active_connections = 0


async def _try_acquire_semaphore(request_id: str) -> bool:
    """
    非阻塞方式尝试获取并发信号量
    - 获取成功返回True
    - 信号量已满（达到并发上限）返回False，由调用方返回429

    修复说明：
    原方案使用 asyncio.wait_for(semaphore.acquire(), timeout=0)，
    在Python 3.10中存在已知问题：即使信号量有可用槽位，
    timeout=0也可能因事件循环来不及执行acquire()协程而误触发TimeoutError。
    现改为先检查信号量内部计数器_value，在单线程async环境中
    _value>0时acquire()不会阻塞（不yield事件循环），无竞态风险。

    Args:
        request_id: 请求ID（用于日志）

    Returns:
        是否成功获取并发槽位
    """
    # 先检查信号量内部计数器是否有可用槽位
    # 在单线程async环境中，_value > 0 时 acquire() 不会阻塞，无竞态风险
    if _semaphore._value <= 0:
        logger.warning(
            f"并发超限拒绝 | request_id={request_id} | "
            f"最大并发={MAX_CONCURRENT} | 当前可用={_semaphore._value}"
        )
        return False
    # 有可用槽位，acquire()不会阻塞，直接获取
    await _semaphore.acquire()
    return True


def _increment_connections() -> None:
    """增加活跃连接计数"""
    global _active_connections
    _active_connections += 1


def _decrement_connections() -> None:
    """减少活跃连接计数"""
    global _active_connections
    _active_connections = max(0, _active_connections - 1)


# ==================== 健康检查接口 ====================

@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    健康检查接口（主网关负载均衡专用）
    返回节点健康状态、当前连接数、配置版本、模型加载状态
    模型加载失败时自动返回status: "unhealthy"
    """
    model_status = get_model_status()
    is_healthy = model_status == "loaded"

    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "node_ip": NODE_IP,
        "current_connections": _active_connections,
        "config_version": keyword_config_manager.get_current_version(),
        "model_status": model_status,
    }


# ==================== 语音行为识别推理接口 ====================

@router.post(
    "/badge/v1/internal/algorithm/inference/behavior-recognition",
    response_model=BehaviorRecognitionResponse,
)
async def behavior_recognition(
    audio_file: UploadFile = File(..., description="音频文件（16000Hz、16bit、1-2声道WAV，多声道自动转单声道）"),
    device_no: str = Form(..., description="设备编号"),
    event_time: str = Form(..., description="行为发生时间，格式yyyy-MM-dd HH:mm:ss"),
    request_id: str = Form(..., description="主网关生成的唯一请求ID"),
):
    """
    语音行为识别推理接口
    处理流程（严格按顺序）：
    ① 入参校验，音频格式不符合直接返回400错误
    ② 幂等性校验：5分钟内相同request_id直接返回缓存结果
    ③ 并发控制：超过单节点5路并发上限，返回429
    ④ ASR推理：音频→带标点中文文本
    ⑤ LLM推理：文本+词库→行为类型JSON结果
    ⑥ 异常音频裁剪（仅ABNORMAL时）
    ⑦ 封装结果返回
    """
    start_time = time.time()

    # ========== ① 入参校验 ==========
    # 读取音频文件字节
    audio_bytes = await audio_file.read()

    # 音频格式校验
    is_valid, msg = validate_wav_audio(audio_bytes)
    if not is_valid:
        logger.warning(
            f"音频格式校验失败 | request_id={request_id} | "
            f"device_no={device_no} | 原因={msg}"
        )
        raise ComputeException(
            code=ErrorCode.BAD_REQUEST,
            msg=f"{ErrorMsg.AUDIO_FORMAT_INVALID}: {msg}",
            request_id=request_id,
        )

    # 多声道自动转单声道（单声道直接返回原数据，零开销）
    audio_bytes = ensure_mono_audio(audio_bytes)

    # ========== ② 幂等性校验 ==========
    cached_result = await idempotency_cache.get(request_id)
    if cached_result is not None:
        logger.info(f"幂等缓存命中 | request_id={request_id}")
        return cached_result

    # ========== ③ 并发控制：非阻塞获取信号量 ==========
    acquired = await _try_acquire_semaphore(request_id)
    if not acquired:
        raise ComputeException(
            code=ErrorCode.TOO_MANY_REQUESTS,
            msg=ErrorMsg.TOO_MANY_REQUESTS,
            request_id=request_id,
        )

    # 增加活跃连接计数
    _increment_connections()

    try:
        try:
            # ========== ④ ASR推理 ==========
            logger.info(
                f"行为识别ASR推理开始 | request_id={request_id} | "
                f"device_no={device_no} | event_time={event_time}"
            )

            try:
                asr_text = await asr_model.inference(audio_bytes)
            except RuntimeError as e:
                logger.error(
                    f"ASR推理失败 | request_id={request_id} | 错误={e}"
                )
                raise ComputeException(
                    code=ErrorCode.ASR_INFERENCE_FAILED,
                    msg=f"{ErrorMsg.ASR_INFERENCE_FAILED}: {str(e)}",
                    request_id=request_id,
                )

            # ASR输出为空校验
            if not asr_text.strip():
                logger.error(f"ASR输出为空 | request_id={request_id}")
                raise ComputeException(
                    code=ErrorCode.ASR_INFERENCE_FAILED,
                    msg=ErrorMsg.ASR_OUTPUT_EMPTY,
                    request_id=request_id,
                )

            # ========== ⑤ LLM推理 ==========
            logger.info(
                f"行为识别LLM推理开始 | request_id={request_id} | "
                f"ASR文本长度={len(asr_text)}"
            )

            # 获取当前词库配置文本
            keyword_text = keyword_config_manager.get_keyword_text()

            try:
                llm_result = await llm_model.behavior_inference(
                    asr_text=asr_text,
                    keyword_config_text=keyword_text,
                )
            except RuntimeError as e:
                logger.error(
                    f"LLM推理失败 | request_id={request_id} | 错误={e}"
                )
                raise ComputeException(
                    code=ErrorCode.LLM_INFERENCE_FAILED,
                    msg=f"{ErrorMsg.LLM_INFERENCE_FAILED}: {str(e)}",
                    request_id=request_id,
                )

            # 强制校验behavior_type枚举值，非标准值回退为STANDARD
            behavior_type = llm_result.get("behavior_type", BehaviorType.STANDARD)
            if behavior_type not in [BehaviorType.STANDARD, BehaviorType.ABNORMAL, BehaviorType.CUSTOMER]:
                logger.warning(
                    f"behavior_type枚举值异常 | 原始值={behavior_type} | "
                    f"回退为STANDARD | request_id={request_id}"
                )
                behavior_type = BehaviorType.STANDARD

            # ========== ⑥ 异常音频裁剪 ==========
            abnormal_audio_clip = None
            if behavior_type == BehaviorType.ABNORMAL:
                # 仅异常行为时裁剪音频片段
                logger.info(f"异常行为音频裁剪 | request_id={request_id}")
                abnormal_audio_clip = clip_abnormal_audio(audio_bytes)
                if not abnormal_audio_clip:
                    logger.warning(f"异常音频裁剪失败，不返回音频片段 | request_id={request_id}")

            # ========== ⑦ 封装结果 ==========
            result_data = {
                "behavior_type": behavior_type,
                "summary": llm_result.get("summary", ""),
                "config_item_id": llm_result.get("config_item_id", llm_result.get("configItemId", "")),
                "keyword_content": llm_result.get("keyword_content", llm_result.get("keywordContent", "")),
                "is_abnormal": llm_result.get("is_abnormal", behavior_type == BehaviorType.ABNORMAL),
            }
            # 仅ABNORMAL时返回异常音频片段
            if behavior_type == BehaviorType.ABNORMAL and abnormal_audio_clip:
                result_data["abnormal_audio_clip"] = abnormal_audio_clip

            result = {
                "code": 200,
                "msg": "success",
                "data": result_data,
                "request_id": request_id,
            }

            # 写入幂等缓存
            await idempotency_cache.set(request_id, result)

            # 记录处理时长
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"行为识别完成 | request_id={request_id} | "
                f"behavior_type={behavior_type} | "
                f"is_abnormal={result_data['is_abnormal']} | "
                f"耗时={elapsed_ms}ms"
            )

            return result

        except ComputeException:
            raise
        except Exception as e:
            logger.exception(
                f"行为识别异常 | request_id={request_id} | 异常={e}"
            )
            raise ComputeException(
                code=ErrorCode.INTERNAL_ERROR,
                msg=ErrorMsg.INTERNAL_ERROR,
                request_id=request_id,
            )

    finally:
        # 无论成功/失败，释放信号量和连接计数
        _semaphore.release()
        _decrement_connections()


# ==================== AI时段诊断总结推理接口 ====================

@router.post(
    "/badge/v1/internal/algorithm/inference/diagnosis-summary",
    response_model=DiagnosisSummaryResponse,
)
async def diagnosis_summary(body: DiagnosisSummaryRequest):
    """
    AI时段诊断总结推理接口
    处理流程：
    ① FastAPI自动校验请求体（Pydantic模型：必填、类型、日期格式、枚举值）
    ② 幂等性校验：5分钟内相同请求直接返回缓存结果
    ③ 并发控制：超过单节点5路并发上限，返回429
    ④ LLM诊断推理
    ⑤ 封装结果返回
    """
    start_time = time.time()
    request_id = body.request_id

    # ========== ② 幂等性校验 ==========
    # 幂等键：employee_no + start_date + end_date + request_id
    idempotency_key = (
        f"diagnosis:{body.employee_no}:{body.start_date}:"
        f"{body.end_date}:{request_id}"
    )
    cached_result = await idempotency_cache.get(idempotency_key)
    if cached_result is not None:
        logger.info(f"诊断幂等缓存命中 | request_id={request_id}")
        return cached_result

    # ========== ③ 并发控制：非阻塞获取信号量 ==========
    acquired = await _try_acquire_semaphore(request_id)
    if not acquired:
        raise ComputeException(
            code=ErrorCode.TOO_MANY_REQUESTS,
            msg=ErrorMsg.TOO_MANY_REQUESTS,
            request_id=request_id,
        )

    # 增加活跃连接计数
    _increment_connections()

    try:
        try:
            # ========== ④ LLM诊断推理 ==========
            logger.info(
                f"诊断总结LLM推理开始 | request_id={request_id} | "
                f"employee_no={body.employee_no} | "
                f"时间范围={body.start_date}~{body.end_date}"
            )

            try:
                llm_result = await llm_model.diagnosis_inference(
                    employee_no=body.employee_no,
                    start_date=body.start_date,
                    end_date=body.end_date,
                    score=body.score,
                    dimension_scores=body.dimension_scores,
                    behavior_stats=body.behavior_stats,
                    abnormal_behaviors=body.abnormal_behaviors,
                )
            except RuntimeError as e:
                logger.error(
                    f"诊断LLM推理失败 | request_id={request_id} | 错误={e}"
                )
                raise ComputeException(
                    code=ErrorCode.LLM_INFERENCE_FAILED,
                    msg=f"{ErrorMsg.LLM_INFERENCE_FAILED}: {str(e)}",
                    request_id=request_id,
                )

            # ========== ⑤ 封装结果 ==========
            # 强制校验dimension_type枚举值
            dimensions = llm_result.get("dimensions", [])
            for dim in dimensions:
                dim_type = dim.get("dimension_type", "")
                if dim_type not in [DimensionType.STRENGTH, DimensionType.WEAKNESS]:
                    logger.warning(
                        f"dimension_type枚举值异常 | 原始值={dim_type} | "
                        f"回退为STRENGTH | request_id={request_id}"
                    )
                    dim["dimension_type"] = DimensionType.STRENGTH
                # WEAKNESS维度必须有suggestion
                if dim["dimension_type"] == DimensionType.WEAKNESS and not dim.get("suggestion"):
                    dim["suggestion"] = "暂无改进建议"

            result = {
                "code": 200,
                "msg": "success",
                "data": {
                    "summary": llm_result.get("summary", ""),
                    "dimensions": dimensions,
                },
                "request_id": request_id,
            }

            # 写入幂等缓存
            await idempotency_cache.set(idempotency_key, result)

            # 记录处理时长
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"诊断总结完成 | request_id={request_id} | "
                f"维度数={len(dimensions)} | 耗时={elapsed_ms}ms"
            )

            return result

        except ComputeException:
            raise
        except Exception as e:
            logger.exception(
                f"诊断总结异常 | request_id={request_id} | 异常={e}"
            )
            raise ComputeException(
                code=ErrorCode.INTERNAL_ERROR,
                msg=ErrorMsg.INTERNAL_ERROR,
                request_id=request_id,
            )

    finally:
        # 无论成功/失败，释放信号量和连接计数
        _semaphore.release()
        _decrement_connections()


# ==================== 词库配置同步接口 ====================

@router.post(
    "/badge/v1/internal/algorithm/config/sync",
    response_model=ConfigSyncResponse,
)
async def config_sync(body: ConfigSyncRequest):
    """
    词库配置同步接口（主网关广播用）
    处理流程：
    ① FastAPI自动校验请求体（Pydantic模型：config_type必须为KEYWORD、版本号必填、items至少1项）
    ② 版本号校验：传入版本低于当前生效版本，直接返回成功
    ③ 配置内容本地缓存到内存+本地JSON文件
    ④ 更新LLM推理模块的词库引用，实时生效
    ⑤ 返回同步结果
    """
    # ========== ② 版本号校验 ==========
    current_version = keyword_config_manager.get_current_version()
    if current_version and body.config_version <= current_version:
        logger.info(
            f"词库配置版本未更新，跳过同步 | "
            f"当前版本={current_version} | 传入版本={body.config_version}"
        )
        return {
            "code": 200,
            "msg": "success",
            "data": {
                "success": True,
                "config_version": current_version,
            },
        }

    # ========== ③④ 同步配置（内存+文件+实时生效） ==========
    config_data = {
        "sop": [item.model_dump() for item in body.sop],
        "forbidden": [item.model_dump() for item in body.forbidden],
        "customer": [item.model_dump() for item in body.customer],
    }

    success = keyword_config_manager.sync_config(
        config_type=body.config_type,
        config_version=body.config_version,
        config_data=config_data,
    )

    if not success:
        raise ComputeException(
            code=ErrorCode.INTERNAL_ERROR,
            msg="词库配置同步失败",
            request_id="",
        )

    # ========== ⑤ 返回同步结果 ==========
    return {
        "code": 200,
        "msg": "success",
        "data": {
            "success": True,
            "config_version": keyword_config_manager.get_current_version(),
        },
    }
