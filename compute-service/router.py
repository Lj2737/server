"""
智能胸牌服务管理系统 - 算力节点API路由
核心接口：
1. GET /health - 健康检查
2. POST /badge/v1/internal/algorithm/inference/behavior-recognition - 语音行为识别推理（支持1-2声道，多声道自动转单声道）
3. POST /badge/v1/internal/algorithm/inference/diagnosis-summary - AI时段诊断总结推理
4. POST /badge/v1/internal/algorithm/config/sync - 词库配置同步
"""
import asyncio
import difflib
import json
import re
import time

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from loguru import logger

from config import (
    MAX_CONCURRENT,
    NODE_IP,
    BehaviorType,
    DimensionType,
    ConfigType,
)
from models import asr_model, llm_model, get_model_status
from fastgpt_client import fastgpt_client
from cache import IdempotencyCache
from keyword_config import KeywordConfigManager
from audio_utils import validate_wav_audio, clip_abnormal_audio, ensure_mono_audio
from exception import ComputeException, ErrorCode, ErrorMsg
from schemas import (
    DiagnosisSummaryRequest,
    DiagnosisSummaryResponse,
    DiagnosisResultData,
    DimensionItem,
    StoreDiagnosisSummaryRequest,
    StoreDiagnosisSummaryResponse,
    ConfigSyncRequest,
    ConfigSyncResponse,
    ConfigSyncResultData,
    HealthCheckResponse,
    BehaviorRecognitionResponse,
    BehaviorResultData,
)

# 创建API路由器
router = APIRouter()


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _select_diagnosis_dimensions(dimension_scores: list[dict]) -> list[dict]:
    """
    按产品规则筛选员工诊断维度：
    - 优势：score >= 85 且 score - avg_score >= 5，最多2个，差值高者优先
    - 薄弱：score < 75 且 avg_score - score >= 5，最多2个，差值高者优先
    """
    strengths = []
    weaknesses = []

    for item in dimension_scores or []:
        if not isinstance(item, dict):
            continue
        dimension_code = str(item.get("dimension_code", "")).strip()
        if not dimension_code:
            continue
        dimension_name = str(item.get("dimension_name") or dimension_code).strip()

        score = _to_float(item.get("score"))
        avg_score = _to_float(item.get("avg_score"))
        diff = score - avg_score
        normalized = {
            **item,
            "dimension_code": dimension_code,
            "dimension_name": dimension_name,
            "score": score,
            "avg_score": avg_score,
            "score_diff": round(diff, 2),
        }

        if score >= 85 and diff >= 5:
            normalized["dimension_type"] = DimensionType.STRENGTH
            strengths.append(normalized)
        elif score < 75 and -diff >= 5:
            normalized["dimension_type"] = DimensionType.WEAKNESS
            normalized["below_avg_diff"] = round(-diff, 2)
            weaknesses.append(normalized)

    strengths.sort(key=lambda x: (x["score_diff"], x["score"]), reverse=True)
    weaknesses.sort(key=lambda x: (x["below_avg_diff"], -x["score"]), reverse=True)
    return strengths[:2] + weaknesses[:2]


def _fallback_dimension_summary(dim: dict) -> str:
    dimension_name = str(dim.get("dimension_name") or dim.get("dimension_code") or "该维度").strip()
    score = _to_float(dim.get("score"))
    avg_score = _to_float(dim.get("avg_score"))
    diff = _to_float(dim.get("score_diff"))
    if dim.get("dimension_type") == DimensionType.STRENGTH:
        return f"{dimension_name}得分{score:g}分，高于平均分{avg_score:g}分，表现优于平均水平。"
    return f"{dimension_name}得分{score:g}分，低于平均分{avg_score:g}分，存在明显改进空间。"


def _fallback_weakness_suggestion(dim: dict) -> str:
    dimension_name = str(dim.get("dimension_name") or dim.get("dimension_code") or "该薄弱维度").strip()
    score = _to_float(dim.get("score"))
    avg_score = _to_float(dim.get("avg_score"))
    return (
        f"针对{dimension_name}得分{score:g}分、低于平均分{avg_score:g}分的薄弱表现，"
        "建议复盘相关服务场景，补充标准话术训练并持续跟踪改进效果。"
    )


def _keyword_content_in_asr(keyword_content: str, asr_text: str) -> bool:
    keyword = (keyword_content or "").strip()
    source = (asr_text or "").strip()
    if not keyword or not source:
        return False
    return keyword in source or keyword.replace(" ", "") in source.replace(" ", "")


def _extract_text_segments(asr_text: str) -> list[str]:
    source = (asr_text or "").strip()
    if not source:
        return []
    segments = [source]
    segments.extend(
        item.strip()
        for item in re.split(r"[。！？!?；;\n\r，,]", source)
        if item.strip()
    )
    return segments


def _keyword_similarity(keyword: str, text: str) -> float:
    keyword = (keyword or "").strip()
    text = (text or "").strip()
    if not keyword or not text:
        return 0.0
    if keyword in text:
        return 1.0
    keyword_chars = {char for char in keyword if not char.isspace()}
    text_chars = {char for char in text if not char.isspace()}
    overlap = len(keyword_chars & text_chars) / max(len(keyword_chars), 1)
    sequence = difflib.SequenceMatcher(None, keyword, text).ratio()
    return max(overlap, sequence)


def _find_semantic_keyword_from_config(
    keyword_config_manager: KeywordConfigManager,
    config_item_id: str,
    asr_text: str,
    min_score: float = 0.55,
) -> str:
    config_item_id = (config_item_id or "").strip()
    if not config_item_id:
        return ""

    segments = _extract_text_segments(asr_text)
    if not segments:
        return ""

    best_keyword = ""
    best_score = 0.0
    for config_items in keyword_config_manager.get_keyword_groups().values():
        for config_item in config_items:
            if str(config_item.get("configItemId", "")).strip() != config_item_id:
                continue
            for keyword in config_item.get("keywords", []) or []:
                content = str(keyword.get("content", "")).strip()
                if not content:
                    continue
                score = max(_keyword_similarity(content, segment) for segment in segments)
                if score > best_score:
                    best_score = score
                    best_keyword = content

    if best_keyword and best_score >= min_score:
        return best_keyword
    logger.warning(
        f"Semantic keyword rejected; no close keyword in config | "
        f"config_item_id={config_item_id} | best_keyword={best_keyword} | "
        f"best_score={best_score:.2f} | min_score={min_score}"
    )
    return ""


def _ensure_keyword_content_from_config(
    keyword_config_manager: KeywordConfigManager,
    config_item_id: str,
    asr_text: str,
    keyword_content: str,
) -> str:
    if _keyword_content_in_asr(keyword_content, asr_text):
        return keyword_content.strip()
    return _find_semantic_keyword_from_config(
        keyword_config_manager=keyword_config_manager,
        config_item_id=config_item_id,
        asr_text=asr_text,
    )

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

@router.post("/badge/v1/internal/algorithm/dialog/ai-chat")
async def ai_dialog_chat(
    audio_file: UploadFile = File(..., description="AI dialog WAV audio, 16kHz/16bit/mono"),
    device_no: str = Form(..., description="Device number"),
    dialog_id: str = Form(..., description="Dialog ID"),
    event_time: str = Form(..., description="Dialog time, yyyy-MM-dd HH:mm:ss"),
    request_id: str = Form(..., description="Request ID, usually same as dialog_id"),
    knowledge_base_id: str = Form("", description="FastGPT knowledge base ID"),
    knowledge_base_id_camel: str = Form("", alias="knowledgeBaseId", description="FastGPT knowledge base ID"),
    stream: bool = Form(False, description="Whether to stream FastGPT reply as NDJSON"),
):
    """
    AI对话推理接口：音频 -> ASR文本 -> FastGPT工作流回复。

    返回字段中data.id来自FastGPT响应id，data.content来自FastGPT响应choices[0].message.content。
    """
    start_time = time.time()
    audio_bytes = await audio_file.read()

    is_valid, msg = validate_wav_audio(audio_bytes)
    if not is_valid:
        raise ComputeException(
            code=ErrorCode.BAD_REQUEST,
            msg=f"{ErrorMsg.AUDIO_FORMAT_INVALID}: {msg}",
            request_id=request_id,
        )

    audio_bytes = ensure_mono_audio(audio_bytes)

    acquired = await _try_acquire_semaphore(request_id)
    if not acquired:
        raise ComputeException(
            code=ErrorCode.TOO_MANY_REQUESTS,
            msg=ErrorMsg.TOO_MANY_REQUESTS,
            request_id=request_id,
        )

    _increment_connections()
    release_in_stream = False
    try:
        try:
            logger.info(
                f"AI dialog ASR start | request_id={request_id} | "
                f"dialog_id={dialog_id} | device_no={device_no} | event_time={event_time}"
            )
            try:
                asr_text = await asr_model.inference(audio_bytes)
            except RuntimeError as e:
                raise ComputeException(
                    code=ErrorCode.ASR_INFERENCE_FAILED,
                    msg=f"{ErrorMsg.ASR_INFERENCE_FAILED}: {str(e)}",
                    request_id=request_id,
                )

            if not asr_text.strip():
                raise ComputeException(
                    code=ErrorCode.ASR_INFERENCE_FAILED,
                    msg=ErrorMsg.ASR_OUTPUT_EMPTY,
                    request_id=request_id,
                )

            chat_time = event_time.replace("-", "").replace(":", "").replace(" ", "")
            chat_id = f"{device_no}_{chat_time}" if chat_time else (dialog_id or device_no)
            resolved_knowledge_base_id = (
                knowledge_base_id.strip() or knowledge_base_id_camel.strip()
            )
            if not resolved_knowledge_base_id:
                raise ComputeException(
                    code=ErrorCode.BAD_REQUEST,
                    msg="FastGPT knowledgeBaseId is required",
                    request_id=request_id,
                )

            if stream:
                release_in_stream = True

                async def stream_reply():
                    reply_content_parts = []
                    reply_id = dialog_id
                    try:
                        async for chunk in fastgpt_client.stream_chat(
                            chat_id=chat_id,
                            user_content=asr_text,
                            knowledge_base_id=resolved_knowledge_base_id,
                            memory_key=device_no,
                        ):
                            reply_id = chunk.get("id") or reply_id
                            content = chunk.get("content") or ""
                            if not content:
                                continue
                            reply_content_parts.append(content)
                            yield json.dumps(
                                {
                                    "type": "delta",
                                    "id": reply_id,
                                    "content": content,
                                },
                                ensure_ascii=False,
                            ) + "\n"

                        reply_content = "".join(reply_content_parts).strip()
                        elapsed_ms = int((time.time() - start_time) * 1000)
                        logger.info(
                            f"AI dialog stream completed | request_id={request_id} | "
                            f"dialog_id={dialog_id} | fastgptId={reply_id} | "
                            f"asrLength={len(asr_text)} | replyLength={len(reply_content)} | "
                            f"elapsed={elapsed_ms}ms"
                        )
                        yield json.dumps(
                            {
                                "type": "done",
                                "id": reply_id,
                                "content": reply_content,
                                "asrText": asr_text,
                            },
                            ensure_ascii=False,
                        ) + "\n"
                    except Exception as e:
                        logger.error(
                            f"FastGPT stream failed | request_id={request_id} | "
                            f"dialog_id={dialog_id} | error={str(e)[:300]}"
                        )
                        yield json.dumps(
                            {
                                "type": "error",
                                "id": reply_id,
                                "message": f"FastGPT对话失败: {str(e)}",
                            },
                            ensure_ascii=False,
                        ) + "\n"
                    finally:
                        _semaphore.release()
                        _decrement_connections()

                return StreamingResponse(
                    stream_reply(),
                    media_type="application/x-ndjson",
                    headers={"X-Accel-Buffering": "no"},
                )

            try:
                fastgpt_result = await fastgpt_client.chat(
                    chat_id=chat_id,
                    user_content=asr_text,
                    knowledge_base_id=resolved_knowledge_base_id,
                    memory_key=device_no,
                )
            except Exception as e:
                logger.error(
                    f"FastGPT chat failed | request_id={request_id} | "
                    f"dialog_id={dialog_id} | error={str(e)[:300]}"
                )
                raise ComputeException(
                    code=ErrorCode.LLM_INFERENCE_FAILED,
                    msg=f"FastGPT对话失败: {str(e)}",
                    request_id=request_id,
                )

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"AI dialog completed | request_id={request_id} | "
                f"dialog_id={dialog_id} | fastgptId={fastgpt_result['id']} | "
                f"asrLength={len(asr_text)} | replyLength={len(fastgpt_result['content'])} | "
                f"elapsed={elapsed_ms}ms"
            )
            return {
                "code": 200,
                "msg": "success",
                "data": {
                    "id": fastgpt_result["id"],
                    "content": fastgpt_result["content"],
                    "asrText": asr_text,
                },
                "request_id": request_id,
            }
        except ComputeException:
            raise
        except Exception as e:
            logger.exception(f"AI dialog failed | request_id={request_id} | error={e}")
            raise ComputeException(
                code=ErrorCode.INTERNAL_ERROR,
                msg=ErrorMsg.INTERNAL_ERROR,
                request_id=request_id,
            )
    finally:
        if not release_in_stream:
            _semaphore.release()
            _decrement_connections()

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
                asr_result = await asr_model.inference_with_timestamps(audio_bytes)
                asr_text = str(asr_result.get("text") or "").strip()
                asr_tokens = asr_result.get("tokens") or []
                asr_timestamps = asr_result.get("timestamps") or []
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

            config_item_id = str(
                llm_result.get(
                    "config_item_id",
                    llm_result.get(
                        "configItemId",
                        llm_result.get("configitemid", llm_result.get("configItemID", "")),
                    ),
                )
            ).strip()
            keyword_content = str(
                llm_result.get(
                    "keyword_content",
                    llm_result.get(
                        "keywordContent",
                        llm_result.get("keywordcontent", llm_result.get("keywordContentl", "")),
                    ),
                )
            ).strip()
            summary = str(llm_result.get("summary", "")).strip()
            is_abnormal = llm_result.get("is_abnormal", behavior_type == BehaviorType.ABNORMAL)
            keyword_matches = keyword_config_manager.find_keyword_matches(asr_text)

            if keyword_matches:
                match_behavior_types = {match.get("behavior_type") for match in keyword_matches}
                if BehaviorType.ABNORMAL in match_behavior_types:
                    behavior_type = BehaviorType.ABNORMAL
                    is_abnormal = True
                elif BehaviorType.CUSTOMER in match_behavior_types:
                    behavior_type = BehaviorType.CUSTOMER
                    is_abnormal = False
                elif BehaviorType.STANDARD in match_behavior_types:
                    behavior_type = BehaviorType.STANDARD
                    is_abnormal = False

                preferred_match = next(
                    (
                        match
                        for target_type in (BehaviorType.ABNORMAL, BehaviorType.CUSTOMER, BehaviorType.STANDARD)
                        for match in keyword_matches
                        if match.get("behavior_type") == target_type
                    ),
                    keyword_matches[0],
                )
                config_item_id = str(preferred_match.get("config_item_id") or "").strip()
                keyword_content = str(preferred_match.get("keyword_content") or "").strip()
                logger.info(
                    f"Local keyword matches found | request_id={request_id} | "
                    f"count={len(keyword_matches)} | behavior_type={behavior_type} | "
                    f"first_config_item_id={config_item_id} | first_keyword_content={keyword_content}"
                )

            if behavior_type == BehaviorType.ABNORMAL and (
                not config_item_id or not keyword_content
            ):
                keyword_match = keyword_config_manager.find_keyword_match("forbidden", asr_text)
                if keyword_match:
                    config_item_id = keyword_match["config_item_id"]
                    keyword_content = keyword_match["keyword_content"]
                    logger.info(
                        f"ABNORMAL missing hit fields filled from forbidden keywords | "
                        f"request_id={request_id} | config_item_id={config_item_id} | "
                        f"keyword_content={keyword_content}"
                    )

            if config_item_id and not keyword_matches:
                corrected_keyword_content = _ensure_keyword_content_from_config(
                    keyword_config_manager=keyword_config_manager,
                    config_item_id=config_item_id,
                    asr_text=asr_text,
                    keyword_content=keyword_content,
                )
                if corrected_keyword_content != keyword_content:
                    logger.info(
                        f"Semantic keyword content corrected from config | request_id={request_id} | "
                        f"config_item_id={config_item_id} | "
                        f"original_keyword_content={keyword_content} | "
                        f"corrected_keyword_content={corrected_keyword_content}"
                    )
                    keyword_content = corrected_keyword_content

            if behavior_type == BehaviorType.ABNORMAL and (
                not config_item_id or not keyword_content
            ):
                logger.warning(
                    f"ABNORMAL missing config_item_id/keyword_content, callback will omit empty fields | "
                    f"request_id={request_id} | asr_text={asr_text[:100]}"
                )
            else:
                is_abnormal = behavior_type == BehaviorType.ABNORMAL

            # ========== ⑥ 异常音频裁剪 ==========
            abnormal_audio_clip = None
            if behavior_type == BehaviorType.ABNORMAL:
                # 仅异常行为时裁剪音频片段
                logger.info(f"异常行为音频裁剪 | request_id={request_id}")
                abnormal_audio_clip = clip_abnormal_audio(
                    audio_bytes,
                    asr_text=asr_text,
                    keyword_content=keyword_content,
                    asr_tokens=asr_tokens,
                    asr_timestamps=asr_timestamps,
                )
                if not abnormal_audio_clip:
                    logger.warning(f"异常音频裁剪失败，不返回音频片段 | request_id={request_id}")

            # ========== ⑦ 封装结果 ==========
            result_data = {
                "behavior_type": behavior_type,
                "summary": summary,
                "config_item_id": config_item_id,
                "keyword_content": keyword_content,
                "keyword_matches": keyword_matches,
                "is_abnormal": is_abnormal,
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

            selected_dimension_scores = _select_diagnosis_dimensions(
                body.dimension_scores
            )
            if not selected_dimension_scores:
                result = {
                    "code": 200,
                    "msg": "success",
                    "data": {
                        "summary": "该时间段暂无满足优势或薄弱规则的维度，整体表现未形成显著维度差异。",
                        "dimensions": [],
                    },
                    "request_id": request_id,
                }
                await idempotency_cache.set(idempotency_key, result)
                elapsed_ms = int((time.time() - start_time) * 1000)
                logger.info(
                    f"诊断总结完成 | request_id={request_id} | "
                    f"维度数=0 | 原因=无符合规则维度 | 耗时={elapsed_ms}ms"
                )
                return result

            try:
                llm_result = await llm_model.diagnosis_inference(
                    employee_no=body.employee_no,
                    start_date=body.start_date,
                    end_date=body.end_date,
                    score=body.score,
                    dimension_scores=selected_dimension_scores,
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
            # 按规则强制校正返回维度，最多2个优势+2个薄弱
            llm_dimensions = llm_result.get("dimensions", [])
            llm_dimension_map = {
                str(dim.get("dimension_code", "")).strip(): dim
                for dim in llm_dimensions
                if isinstance(dim, dict)
            }
            dimensions = []
            for selected_dim in selected_dimension_scores:
                dimension_code = selected_dim["dimension_code"]
                dim = dict(llm_dimension_map.get(dimension_code, {}))
                dim["dimension_code"] = dimension_code
                dim["dimension_type"] = selected_dim["dimension_type"]
                if not dim.get("summary"):
                    dim["summary"] = _fallback_dimension_summary(selected_dim)
                if dim["dimension_type"] == DimensionType.WEAKNESS and not dim.get("suggestion"):
                    dim["suggestion"] = _fallback_weakness_suggestion(selected_dim)
                if dim["dimension_type"] == DimensionType.STRENGTH:
                    dim["suggestion"] = ""
                dimensions.append(
                    {
                        "dimension_code": dim["dimension_code"],
                        "dimension_type": dim["dimension_type"],
                        "summary": dim["summary"],
                        "suggestion": dim.get("suggestion") or "",
                    }
                )

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


# ==================== 门店AI时段诊断总结推理接口 ====================

@router.post(
    "/badge/v1/internal/algorithm/inference/store-diagnosis-summary",
    response_model=StoreDiagnosisSummaryResponse,
)
async def store_diagnosis_summary(body: StoreDiagnosisSummaryRequest):
    """
    门店AI时段诊断总结推理接口
    根据门店员工行为记录生成门店综合分析和分析建议。
    """
    start_time = time.time()
    request_id = body.request_id

    idempotency_key = (
        f"store-diagnosis:{body.store_id}:{body.start_date}:"
        f"{body.end_date}:{request_id}"
    )
    cached_result = await idempotency_cache.get(idempotency_key)
    if cached_result is not None:
        logger.info(f"门店诊断幂等缓存命中 | request_id={request_id}")
        return cached_result

    acquired = await _try_acquire_semaphore(request_id)
    if not acquired:
        raise ComputeException(
            code=ErrorCode.TOO_MANY_REQUESTS,
            msg=ErrorMsg.TOO_MANY_REQUESTS,
            request_id=request_id,
        )

    _increment_connections()

    try:
        try:
            behaviors = [item.model_dump() for item in body.behaviors]
            logger.info(
                f"门店诊断LLM推理开始 | request_id={request_id} | "
                f"store_id={body.store_id} | store_name={body.store_name} | "
                f"时间范围={body.start_date}~{body.end_date} | 行为数={len(behaviors)}"
            )

            try:
                llm_result = await llm_model.store_diagnosis_inference(
                    store_id=body.store_id,
                    store_name=body.store_name,
                    start_date=body.start_date,
                    end_date=body.end_date,
                    behaviors=behaviors,
                )
            except RuntimeError as e:
                logger.error(
                    f"门店诊断LLM推理失败 | request_id={request_id} | 错误={e}"
                )
                raise ComputeException(
                    code=ErrorCode.LLM_INFERENCE_FAILED,
                    msg=f"{ErrorMsg.LLM_INFERENCE_FAILED}: {str(e)}",
                    request_id=request_id,
                )

            summary = str(llm_result.get("summary") or "").strip()
            if not summary:
                summary = "该时间段暂无足够行为数据，暂无法形成明确门店诊断。"

            raw_suggestions = llm_result.get("suggestions", [])
            suggestions = []
            if isinstance(raw_suggestions, list):
                suggestions = [
                    str(item).strip()
                    for item in raw_suggestions
                    if str(item).strip()
                ]
            elif isinstance(raw_suggestions, str) and raw_suggestions.strip():
                suggestions = [raw_suggestions.strip()]
            if not suggestions:
                suggestions = ["暂无改进建议"]

            result = {
                "code": 200,
                "msg": "success",
                "data": {
                    "summary": summary,
                    "suggestions": suggestions,
                },
                "request_id": request_id,
            }

            await idempotency_cache.set(idempotency_key, result)

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"门店诊断总结完成 | request_id={request_id} | "
                f"建议数={len(suggestions)} | 耗时={elapsed_ms}ms"
            )

            return result

        except ComputeException:
            raise
        except Exception as e:
            logger.exception(
                f"门店诊断总结异常 | request_id={request_id} | 异常={e}"
            )
            raise ComputeException(
                code=ErrorCode.INTERNAL_ERROR,
                msg=ErrorMsg.INTERNAL_ERROR,
                request_id=request_id,
            )

    finally:
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
