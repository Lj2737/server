"""
智能胸牌服务管理系统 - 原始异常语音上传路由
核心功能：
1. 接收胸牌硬件分段上传的原始WAV音频
2. 入参校验：file字段（WAV格式+大小限制）+ metadata字段（JSON格式+必传键校验）
3. 校验通过后保存到主网关临时目录
4. 异步转发给算力节点进行ASR转写和行为识别（不阻塞硬件响应）
5. 转发成功后删除临时文件，失败保留临时文件支持手动重试
6. 无论转发是否成功，硬件永远收到200响应

核心原则：
- 硬件仅负责音频采集、分段、上传，不做任何识别
- 所有ASR转写、异常检测由算力节点完成
- 复用已有的NodeManager算力节点管理器和httpx.AsyncClient连接池
- 转发失败不影响硬件上传接口的返回（硬件永远收到200）

接口规范：
- 接口路径：POST /internal/badge/hardware/raw-audio-upload
- Content-Type：multipart/form-data
- 表单必传字段：
  ① file：原始音频文件（WAV格式）
  ② metadata：JSON格式字符串
- metadata必传键：uploadId、deviceNo、startTime

使用示例：
    # Postman模拟硬件上传
    curl -X POST http://网关IP:8090/internal/badge/hardware/raw-audio-upload \
      -F "file=@/path/to/audio.wav" \
      -F 'metadata={"uploadId":"upload_001","deviceNo":"BADGE0001","startTime":"2026-05-13 15:00:00"}'
"""
import asyncio
import json
import os
import re
import time
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, UploadFile
from loguru import logger

from config import (
    RAW_AUDIO_MAX_FILE_SIZE,
    RAW_AUDIO_METADATA_MAX_LENGTH,
    RAW_AUDIO_FORWARD_TIMEOUT,
    WAV_HEADER_MAGIC,
    DEVICE_NO_MAX_LENGTH,
    DEVICE_NO_PATTERN,
)
from audio_temp_manager import AudioTempManager
from node_manager import NodeManager
from http_client import HttpClientSingleton
from behavior_callback import BehaviorCallback
from router import behavior_callback


# ==================== 常量定义 ====================

# metadata JSON中必须包含的键
METADATA_REQUIRED_KEYS = {"uploadId", "deviceNo", "startTime"}

# 算力节点行为识别接口路径
BEHAVIOR_RECOGNITION_PATH = "/api/v1/internal/inference/behavior-recognition"


# ==================== 路由器创建 ====================

raw_audio_router = APIRouter(
    tags=["原始异常语音上传接口"],
)

# 临时文件管理器单例
audio_temp_manager = AudioTempManager()

# 算力节点管理器单例（复用已有的NodeManager）
node_manager = NodeManager()

# 标记是否已初始化（在main.py lifespan中注入依赖后置True）
_initialized: bool = False


def initialize() -> None:
    """
    初始化原始异常语音上传模块
    在FastAPI lifespan startup阶段调用
    - 标记模块已初始化，允许转发逻辑执行
    """
    global _initialized
    _initialized = True
    logger.info("原始异常语音上传模块已初始化")


# ==================== POST接口：原始异常语音上传 ====================

@raw_audio_router.post("/raw-audio-upload")
async def raw_audio_upload(
    file: UploadFile = File(..., description="原始音频文件（WAV格式）"),
    metadata: str = Form(..., description="JSON格式字符串，包含uploadId、deviceNo、startTime"),
):
    """
    接收胸牌硬件分段上传的原始WAV音频

    对齐v4.2文档：
    - 硬件按分段上传原始WAV音频
    - 主网关接收后保存临时文件，异步转发给算力节点
    - 算力节点完成ASR转写和行为识别后，通过已有回调链路返回结果
    - 硬件永远收到200响应

    请求格式：multipart/form-data
    - file：原始WAV音频文件
    - metadata：JSON字符串，如 {"uploadId":"upload_001","deviceNo":"BADGE0001","startTime":"2026-05-13 15:00:00"}

    成功响应：
    {
        "code": 200,
        "message": "接收成功",
        "data": {
            "uploadId": "upload_001",
            "receiveTime": "2026-05-13 15:00:02"
        }
    }

    校验失败响应：
    {
        "code": 400,
        "message": "具体校验失败原因",
        "data": null
    }
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # ========== 步骤1：metadata字段校验 ==========
    metadata_dict, metadata_error = _validate_metadata(metadata, request_id)
    if metadata_error:
        return _build_error_response(400, metadata_error)

    upload_id = metadata_dict["uploadId"]
    device_no = metadata_dict["deviceNo"]

    # ========== 步骤2：file字段校验 ==========
    file_error = await _validate_audio_file(file, upload_id, device_no)
    if file_error:
        return _build_error_response(400, file_error)

    # ========== 步骤3：读取音频文件内容 ==========
    try:
        audio_bytes = await file.read()
    except Exception as e:
        logger.error(
            f"读取音频文件失败 | uploadId={upload_id} | "
            f"deviceNo={device_no} | 错误={str(e)[:200]}"
        )
        return _build_error_response(400, "音频文件读取失败")

    # ========== 步骤4：保存到临时目录 ==========
    temp_file_path: Optional[str] = None
    try:
        temp_file_path = await audio_temp_manager.save(
            upload_id=upload_id,
            device_no=device_no,
            audio_bytes=audio_bytes,
        )
    except IOError as e:
        logger.error(
            f"临时文件保存失败 | uploadId={upload_id} | "
            f"deviceNo={device_no} | 错误={str(e)[:200]}"
        )
        # 临时文件保存失败仍返回200，硬件不需要知道内部错误
        # 但这个情况比较严重，返回错误让硬件知道
        return _build_error_response(500, "临时文件保存失败")

    # ========== 步骤5：异步转发给算力节点 ==========
    # 核心原则：使用asyncio.create_task创建后台任务，不阻塞主网关事件循环
    # 转发失败不影响硬件上传接口的返回（硬件永远收到200）
    try:
        asyncio.create_task(
            _forward_to_compute_node_task(
                temp_file_path=temp_file_path,
                upload_id=upload_id,
                device_no=device_no,
                start_time_str=metadata_dict["startTime"],
                audio_size=len(audio_bytes),
            ),
            name=f"forward-raw-audio-{upload_id}-{device_no}",
        )
        logger.info(
            f"原始音频转发任务已创建 | uploadId={upload_id} | "
            f"deviceNo={device_no}"
        )
    except Exception as e:
        # 创建任务失败不影响返回，仅记录日志
        logger.error(
            f"原始音频转发任务创建失败（不影响接收响应）| "
            f"uploadId={upload_id} | deviceNo={device_no} | "
            f"错误={str(e)[:200]}"
        )

    # ========== 步骤6：返回标准响应（硬件永远收到200） ==========
    elapsed_ms = int((time.time() - start_time) * 1000)
    receive_time = _format_current_time()

    logger.info(
        f"原始异常语音上传处理完成 | uploadId={upload_id} | "
        f"deviceNo={device_no} | 文件大小={len(audio_bytes) / 1024:.1f}KB | "
        f"耗时={elapsed_ms}ms"
    )

    return _build_success_response(upload_id, receive_time)


# ==================== 入参校验函数 ====================

def _validate_metadata(metadata: str, request_id: str) -> tuple:
    """
    校验metadata字段

    校验规则：
    1. 非空
    2. 长度不超过RAW_AUDIO_METADATA_MAX_LENGTH
    3. 合法JSON格式
    4. 必须包含uploadId、deviceNo、startTime键
    5. uploadId非空
    6. deviceNo非空、仅字母数字下划线、≤20位
    7. startTime格式为yyyy-MM-dd HH:mm:ss

    Args:
        metadata: metadata表单字段值（JSON字符串）
        request_id: 请求ID（用于日志）

    Returns:
        (metadata_dict, error_message) 元组
        - 校验成功：metadata_dict为解析后的字典，error_message为None
        - 校验失败：metadata_dict为None，error_message为错误信息
    """
    # 非空校验
    if not metadata or not metadata.strip():
        error = "metadata字段不能为空"
        logger.warning(f"metadata校验失败 | request_id={request_id} | 原因={error}")
        return None, error

    # 长度校验
    if len(metadata) > RAW_AUDIO_METADATA_MAX_LENGTH:
        error = f"metadata长度超限（最大{RAW_AUDIO_METADATA_MAX_LENGTH}字符）"
        logger.warning(f"metadata校验失败 | request_id={request_id} | 原因={error}")
        return None, error

    # JSON格式校验
    try:
        metadata_dict = json.loads(metadata)
    except json.JSONDecodeError as e:
        error = f"metadata不是合法的JSON格式: {str(e)[:100]}"
        logger.warning(f"metadata校验失败 | request_id={request_id} | 原因={error}")
        return None, error

    # 必传键校验
    if not isinstance(metadata_dict, dict):
        error = "metadata必须是JSON对象（键值对）"
        logger.warning(f"metadata校验失败 | request_id={request_id} | 原因={error}")
        return None, error

    missing_keys = METADATA_REQUIRED_KEYS - set(metadata_dict.keys())
    if missing_keys:
        error = f"metadata缺少必传字段: {', '.join(sorted(missing_keys))}"
        logger.warning(f"metadata校验失败 | request_id={request_id} | 原因={error}")
        return None, error

    # uploadId校验
    upload_id = metadata_dict.get("uploadId", "")
    if not upload_id or not str(upload_id).strip():
        error = "metadata.uploadId不能为空"
        logger.warning(f"metadata校验失败 | request_id={request_id} | 原因={error}")
        return None, error

    # deviceNo校验
    device_no = metadata_dict.get("deviceNo", "")
    if not device_no or not str(device_no).strip():
        error = "metadata.deviceNo不能为空"
        logger.warning(f"metadata校验失败 | request_id={request_id} | 原因={error}")
        return None, error
    if not re.match(DEVICE_NO_PATTERN, str(device_no)):
        error = f"metadata.deviceNo仅支持字母、数字、下划线，当前值：{device_no}"
        logger.warning(f"metadata校验失败 | request_id={request_id} | 原因={error}")
        return None, error
    if len(str(device_no)) > DEVICE_NO_MAX_LENGTH:
        error = f"metadata.deviceNo长度不能超过{DEVICE_NO_MAX_LENGTH}位"
        logger.warning(f"metadata校验失败 | request_id={request_id} | 原因={error}")
        return None, error

    # startTime格式校验
    start_time_str = metadata_dict.get("startTime", "")
    if not start_time_str or not str(start_time_str).strip():
        error = "metadata.startTime不能为空"
        logger.warning(f"metadata校验失败 | request_id={request_id} | 原因={error}")
        return None, error
    try:
        datetime.strptime(str(start_time_str), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        error = f"metadata.startTime格式必须为yyyy-MM-dd HH:mm:ss，当前值：{start_time_str}"
        logger.warning(f"metadata校验失败 | request_id={request_id} | 原因={error}")
        return None, error

    return metadata_dict, None


async def _validate_audio_file(
    file: UploadFile,
    upload_id: str,
    device_no: str,
) -> Optional[str]:
    """
    校验音频文件

    校验规则：
    1. 文件名非空
    2. 文件大小不超过RAW_AUDIO_MAX_FILE_SIZE
    3. WAV格式校验（RIFF头魔数）

    Args:
        file: 上传的文件对象
        upload_id: 上传ID
        device_no: 设备编号

    Returns:
        None: 校验通过
        str: 校验失败错误信息
    """
    # 文件名校验
    if not file.filename:
        error = "音频文件名不能为空"
        logger.warning(
            f"音频文件校验失败 | uploadId={upload_id} | "
            f"deviceNo={device_no} | 原因={error}"
        )
        return error

    # 文件大小校验
    try:
        # 读取文件内容以检查大小和格式
        audio_bytes = await file.read()
        file_size = len(audio_bytes)

        # 将文件指针重置，以便后续再次读取
        await file.seek(0)

        if file_size == 0:
            error = "音频文件为空"
            logger.warning(
                f"音频文件校验失败 | uploadId={upload_id} | "
                f"deviceNo={device_no} | 原因={error}"
            )
            return error

        if file_size > RAW_AUDIO_MAX_FILE_SIZE:
            max_mb = RAW_AUDIO_MAX_FILE_SIZE / (1024 * 1024)
            error = f"音频文件大小超限（最大{max_mb:.0f}MB，当前{file_size / (1024 * 1024):.1f}MB）"
            logger.warning(
                f"音频文件校验失败 | uploadId={upload_id} | "
                f"deviceNo={device_no} | 原因={error}"
            )
            return error

        # WAV格式校验：检查RIFF头魔数
        if not audio_bytes[:4] == WAV_HEADER_MAGIC:
            error = "音频文件格式错误，仅支持WAV格式（RIFF头校验失败）"
            logger.warning(
                f"音频文件校验失败 | uploadId={upload_id} | "
                f"deviceNo={device_no} | 原因={error} | "
                f"文件头={audio_bytes[:4].hex()}"
            )
            return error

    except Exception as e:
        error = f"音频文件校验异常: {str(e)[:100]}"
        logger.warning(
            f"音频文件校验失败 | uploadId={upload_id} | "
            f"deviceNo={device_no} | 原因={error}"
        )
        return error

    return None


# ==================== 异步转发算力节点逻辑 ====================

async def _forward_to_compute_node_task(
    temp_file_path: str,
    upload_id: str,
    device_no: str,
    start_time_str: str,
    audio_size: int,
) -> None:
    """
    异步转发原始音频到算力节点的后台任务
    由asyncio.create_task创建，不阻塞主网关事件循环

    处理流程：
    1. 选择可用算力节点（最小连接数策略）
    2. 读取临时WAV文件
    3. 构建multipart/form-data请求转发到算力节点行为识别接口
    4. 转发成功后删除临时文件
    5. 转发失败保留临时文件，记录完整错误日志

    核心原则：
    - 所有异常都要捕获，不能抛出到主事件循环
    - 转发失败不影响硬件上传接口的返回（硬件永远收到200）

    Args:
        temp_file_path: 临时WAV文件路径
        upload_id: 上传ID（metadata中的uploadId，作为转发请求的request_id）
        device_no: 设备编号
        start_time_str: 行为发生时间（metadata中的startTime）
        audio_size: 音频文件大小（字节，用于日志）
    """
    selected_node = None
    try:
        if not _initialized:
            logger.warning(
                f"原始音频转发跳过 | 原因=模块未初始化 | "
                f"uploadId={upload_id} | deviceNo={device_no}"
            )
            return

        # ========== 选择可用算力节点 ==========
        selected_node = node_manager.get_least_connection_node()
        if selected_node is None:
            logger.error(
                f"原始音频转发失败 | 原因=无可用算力节点 | "
                f"uploadId={upload_id} | deviceNo={device_no} | "
                f"临时文件保留={os.path.basename(temp_file_path)}"
            )
            return

        target_url = f"http://{selected_node}{BEHAVIOR_RECOGNITION_PATH}"

        # ========== 读取临时WAV文件 ==========
        if not await audio_temp_manager.exists(temp_file_path):
            logger.error(
                f"原始音频转发失败 | 原因=临时文件不存在 | "
                f"uploadId={upload_id} | deviceNo={device_no} | "
                f"路径={temp_file_path}"
            )
            return

        audio_bytes = await audio_temp_manager.read(temp_file_path)

        # ========== 构建转发请求 ==========
        # 使用uploadId作为算力节点行为识别的request_id
        # 转发参数：audio_file、device_no、event_time、request_id
        # 对齐算力节点接口 POST /api/v1/internal/inference/behavior-recognition
        files = {
            "audio_file": (
                f"{upload_id}_{device_no}.wav",  # 文件名
                audio_bytes,                      # 文件字节
                "audio/wav",                      # Content-Type
            ),
        }
        data = {
            "device_no": device_no,
            "event_time": start_time_str,
            "request_id": upload_id,  # 使用uploadId作为request_id
        }

        # ========== 增加节点连接计数 ==========
        await node_manager.increment_connection(selected_node)

        try:
            # ========== 发送转发请求 ==========
            client = await HttpClientSingleton.get_client()

            logger.info(
                f"原始音频转发开始 | uploadId={upload_id} | "
                f"deviceNo={device_no} | 目标节点={selected_node} | "
                f"文件大小={audio_size / 1024:.1f}KB"
            )

            response = await client.post(
                url=target_url,
                files=files,
                data=data,
                timeout=RAW_AUDIO_FORWARD_TIMEOUT,
            )

            elapsed_ms = 0  # 简化，不含内部耗时计算

            # ========== 检查转发结果 ==========
            if 200 <= response.status_code < 300:
                # 转发成功
                logger.info(
                    f"原始音频转发成功 | uploadId={upload_id} | "
                    f"deviceNo={device_no} | 节点={selected_node} | "
                    f"状态码={response.status_code}"
                )

                # 新增：触发行为识别回调
                try:
                    response_data = response.json()
                    if response_data.get("code") == 200 and response_data.get("data"):
                        # 异步触发回调（fire-and-forget）
                        asyncio.create_task(
                            behavior_callback.handle_result(
                                inference_data=response_data["data"],
                                device_no=device_no,
                                event_time=start_time_str,
                            )
                        )
                        logger.info(
                            f"行为识别回调已触发 | uploadId={upload_id} | "
                            f"deviceNo={device_no}"
                        )
                except Exception as e:
                    logger.error(
                        f"触发行为识别回调失败 | uploadId={upload_id} | "
                        f"错误={str(e)[:200]}"
                    )

                # 转发成功后删除临时文件
                await audio_temp_manager.delete(temp_file_path)
                logger.debug(
                    f"临时文件已删除（转发成功）| uploadId={upload_id} | "
                    f"文件={os.path.basename(temp_file_path)}"
                )

            else:
                # 转发失败（算力节点返回非2xx）
                error_body = response.text[:500]
                logger.error(
                    f"原始音频转发失败 | uploadId={upload_id} | "
                    f"deviceNo={device_no} | 节点={selected_node} | "
                    f"状态码={response.status_code} | 响应={error_body} | "
                    f"临时文件保留={os.path.basename(temp_file_path)}"
                )

        except Exception as e:
            # 转发请求异常（超时、连接失败等）
            logger.error(
                f"原始音频转发异常 | uploadId={upload_id} | "
                f"deviceNo={device_no} | 节点={selected_node} | "
                f"异常类型={type(e).__name__} | 错误={str(e)[:300]} | "
                f"异常栈={traceback.format_exc()[:500]} | "
                f"临时文件保留={os.path.basename(temp_file_path)}"
            )

        finally:
            # 无论成功/失败，减少节点连接计数
            if selected_node:
                await node_manager.decrement_connection(selected_node)

    except Exception as e:
        # 终极兜底：任何未预料的异常都不能抛到主事件循环
        logger.error(
            f"原始音频转发任务异常(终极兜底) | "
            f"uploadId={upload_id} | deviceNo={device_no} | "
            f"节点={selected_node or '未知'} | "
            f"异常类型={type(e).__name__} | 错误={str(e)[:200]} | "
            f"异常栈={traceback.format_exc()[:500]} | "
            f"临时文件保留={os.path.basename(temp_file_path) if temp_file_path else '未知'}"
        )


# ==================== 响应构建 ====================

def _build_success_response(upload_id: str, receive_time: str) -> Dict[str, Any]:
    """
    构建上传成功响应体

    {
        "code": 200,
        "message": "接收成功",
        "data": {
            "uploadId": "upload_001",
            "receiveTime": "2026-05-13 15:00:02"
        }
    }

    Args:
        upload_id: 上传ID
        receive_time: 算法接收时间

    Returns:
        统一格式的成功响应字典
    """
    return {
        "code": 200,
        "message": "接收成功",
        "data": {
            "uploadId": upload_id,
            "receiveTime": receive_time,
        },
    }


def _build_error_response(code: int, message: str) -> Dict[str, Any]:
    """
    构建上传校验失败响应体

    {
        "code": 400,
        "message": "具体校验失败原因",
        "data": null
    }

    Args:
        code: 错误码
        message: 错误信息

    Returns:
        统一格式的错误响应字典
    """
    return {
        "code": code,
        "message": message,
        "data": None,
    }


def _format_current_time() -> str:
    """
    格式化当前时间为yyyy-MM-dd HH:mm:ss

    Returns:
        格式化后的当前时间字符串
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
