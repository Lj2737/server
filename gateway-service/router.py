"""
智能胸牌服务管理系统 - 路由转发与基础接口定义
核心功能：
1. 推理请求的路由转发（行为识别、诊断总结）
2. 行为识别推理成功后触发回调后端
3. 网关健康检查接口
4. 节点状态查询接口
5. 词库配置同步广播（内部方法，非对外接口）
"""
import asyncio
import json
import uuid
import time

import httpx
from fastapi import APIRouter, Request, Response
from loguru import logger

from config import (
    ROUTE_MAPPING,
    REQUEST_TIMEOUT,
    REQUEST_ID_HEADER,
    ConfigType,
)
from node_manager import NodeManager
from http_client import HttpClientSingleton
from behavior_callback import BehaviorCallback
from piper_tts_manager import PiperTTSManager
from websocket_device_manager import WebSocketDeviceManager
from exception import (
    GatewayException,
    ErrorCode,
    ErrorMsg,
    build_error_response,
)

# 创建API路由器
router = APIRouter()

# 节点管理器单例
node_manager = NodeManager()

# 行为识别回调处理器单例（由main.py在启动时初始化BackendClient后注入）
behavior_callback = BehaviorCallback()

# Piper TTS管理器单例和WebSocket设备管理器单例
piper_tts_manager = PiperTTSManager()
ws_device_manager = WebSocketDeviceManager()

# 转发时需要排除的响应头（避免传递导致冲突的头信息）
_HOP_BY_HOP_HEADERS = frozenset({
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "content-encoding",
})


# ==================== 基础接口 ====================

@router.get("/health")
async def health_check():
    """
    网关健康检查接口
    返回网关服务状态、可用节点数量、总请求数、TTS状态、WebSocket设备在线数
    当有健康节点时状态为healthy，否则为degraded
    """
    healthy_count = node_manager.get_healthy_node_count()
    total_requests = node_manager.get_total_requests()
    tts_available = piper_tts_manager.is_available()
    ws_online_count = ws_device_manager.get_online_count()

    return {
        "code": 200,
        "msg": "ok",
        "data": {
            "status": "healthy" if healthy_count > 0 else "degraded",
            "healthy_nodes": healthy_count,
            "total_nodes": len(node_manager._nodes),
            "total_requests": total_requests,
            "tts_available": tts_available,
            "ws_online_devices": ws_online_count,
        },
        "request_id": str(uuid.uuid4()),
    }


@router.get("/api/v1/gateway/nodes")
async def get_nodes_status():
    """
    节点状态查询接口
    返回所有算力节点的详细状态信息，包括：
    健康状态、当前连接数、配置版本号、累计请求数、最近检查时间等
    """
    nodes = node_manager.get_nodes_status()
    return {
        "code": 200,
        "msg": "ok",
        "data": nodes,
        "request_id": str(uuid.uuid4()),
    }


@router.get("/api/v1/gateway/ws-devices")
async def get_ws_devices_status():
    """
    WebSocket设备在线状态查询接口
    返回当前所有在线设备的编号列表和数量
    """
    online_devices = ws_device_manager.get_online_devices()
    return {
        "code": 200,
        "msg": "ok",
        "data": {
            "online_count": len(online_devices),
            "online_devices": online_devices,
            "tts_available": piper_tts_manager.is_available(),
        },
        "request_id": str(uuid.uuid4()),
    }


# ==================== 语音行为识别推理转发（含回调后端） ====================

@router.post("/api/v1/gateway/behavior-recognition")
async def forward_behavior_recognition(request: Request):
    """
    语音行为识别推理请求转发 + 回调后端

    完整流程（严格按顺序）：
    1. 从原始请求中提取device_no、event_time（用于回调后端）
    2. 获取或生成请求唯一ID（X-Request-ID）
    3. 路由映射 + 负载均衡选择算力节点
    4. 完整转发请求到算力节点
    5. 推理成功后，触发异步回调后端（fire-and-forget，不阻塞响应）
    6. 原封不动返回算力节点响应

    回调触发条件：
    - 算力节点返回HTTP 200
    - 响应体code=200且data非空
    - BehaviorCallback已初始化

    回调内容（v3文档格式）：
    POST /internal/badge/ai/voice-behaviors
    {
        "eventId": "AI_BEHAVIOR_20260508103100_001234",
        "eventTime": "2026-05-08 10:31:00",
        "deviceNo": "BADGE0001",
        "behaviorType": "STANDARD/ABNORMAL/CUSTOMER",
        "summary": "行为摘要文本"
    }
    """
    # 获取或生成请求唯一ID（幂等键）
    request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())

    # 获取请求原始路径
    original_path = request.url.path

    # ========== 步骤1：从原始请求中提取device_no、event_time ==========
    # 行为识别请求使用multipart/form-data，先读body缓存，再解析表单
    device_no = ""
    event_time = ""
    try:
        # body()读取并缓存原始请求体，form()从缓存解析表单字段
        _ = await request.body()  # 确保body已缓存（stream消费后可被form复用）
        form = await request.form()
        device_no = form.get("device_no", "") or ""
        event_time = form.get("event_time", "") or ""
        logger.debug(
            f"行为识别表单字段提取 | request_id={request_id} | "
            f"device_no={device_no} | event_time={event_time}"
        )
    except Exception as e:
        logger.warning(
            f"行为识别表单字段提取失败（不影响转发）| "
            f"request_id={request_id} | 错误={str(e)[:200]}"
        )

    # ========== 步骤2~4：转发到算力节点 ==========
    response = await _forward_to_compute_node(request, original_path, request_id)

    # ========== 步骤5：推理成功后触发异步回调后端（fire-and-forget） ==========
    if response.status_code == 200:
        _trigger_behavior_callback(response, request_id, device_no, event_time)

    # ========== 步骤6：原封不动返回算力节点响应 ==========
    return _build_proxy_response(response)


# ==================== AI时段诊断总结推理转发 ====================

@router.post("/api/v1/gateway/diagnosis-summary")
async def forward_diagnosis_summary(request: Request):
    """
    AI时段诊断总结推理请求转发
    纯转发，无回调逻辑

    流程：
    1. 获取或生成请求唯一ID
    2. 路由映射 + 负载均衡选择算力节点
    3. 完整转发请求到算力节点
    4. 原封不动返回算力节点响应
    """
    request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
    original_path = request.url.path

    response = await _forward_to_compute_node(request, original_path, request_id)

    return _build_proxy_response(response)


# ==================== 通用转发核心实现 ====================

async def _forward_to_compute_node(
    request: Request,
    original_path: str,
    request_id: str,
) -> httpx.Response:
    """
    通用推理请求转发核心实现
    从路由映射、负载均衡、请求转发到异常处理，完整的转发链路

    Args:
        request: 原始FastAPI请求对象
        original_path: 请求原始路径（如 /api/v1/gateway/behavior-recognition）
        request_id: 请求唯一ID

    Returns:
        算力节点的HTTP响应

    Raises:
        GatewayException: 无可用节点(503) / 超时(504) / 节点错误(502)
    """
    # 根据路由映射表，获取算力节点内部路径
    internal_path = ROUTE_MAPPING.get(original_path)
    if internal_path is None:
        logger.warning(
            f"路由映射不存在 | 路径={original_path} | request_id={request_id}"
        )
        raise GatewayException(
            code=ErrorCode.NOT_FOUND,
            msg=f"路由不存在: {original_path}",
            request_id=request_id,
        )

    # 负载均衡：选择活跃连接数最少的最健康节点
    selected_node = node_manager.get_least_connection_node()
    if selected_node is None:
        logger.error(
            f"无可用算力节点 | request_id={request_id} | 路径={original_path}"
        )
        raise GatewayException(
            code=ErrorCode.NO_AVAILABLE_NODE,
            msg=ErrorMsg.NO_AVAILABLE_NODE,
            request_id=request_id,
        )

    # 构建算力节点完整请求URL
    target_url = f"http://{selected_node}{internal_path}"

    # 保留原始查询参数
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    # 记录请求开始时间
    start_time = time.time()

    # 增加节点活跃连接数
    await node_manager.increment_connection(selected_node)

    try:
        # 获取httpx异步客户端（复用连接池）
        client = await HttpClientSingleton.get_client()

        # 构建转发请求头：保留原始Header，确保X-Request-ID存在
        forward_headers = dict(request.headers)
        forward_headers[REQUEST_ID_HEADER] = request_id
        # 移除可能导致冲突的头
        forward_headers.pop("host", None)
        forward_headers.pop("content-length", None)

        # 读取原始请求体（支持JSON/FormData/二进制等所有格式）
        body = await request.body()

        # 发送转发请求
        logger.info(
            f"请求转发开始 | request_id={request_id} | "
            f"源路径={original_path} → 目标={target_url} | 节点={selected_node}"
        )

        response = await client.request(
            method=request.method,
            url=target_url,
            headers=forward_headers,
            content=body,
            timeout=REQUEST_TIMEOUT,
        )

        # 计算响应时长（毫秒）
        elapsed_ms = int((time.time() - start_time) * 1000)

        # 记录转发结果日志
        logger.info(
            f"请求转发完成 | request_id={request_id} | "
            f"节点={selected_node} | 状态码={response.status_code} | "
            f"耗时={elapsed_ms}ms"
        )

        return response

    except httpx.TimeoutException as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.error(
            f"请求转发超时 | request_id={request_id} | "
            f"节点={selected_node} | 超时类型={type(e).__name__} | 耗时={elapsed_ms}ms"
        )
        raise GatewayException(
            code=ErrorCode.GATEWAY_TIMEOUT,
            msg=ErrorMsg.GATEWAY_TIMEOUT,
            request_id=request_id,
        )

    except httpx.HTTPStatusError as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.error(
            f"算力节点响应错误 | request_id={request_id} | "
            f"节点={selected_node} | 状态码={e.response.status_code} | 耗时={elapsed_ms}ms"
        )
        raise GatewayException(
            code=ErrorCode.NODE_REQUEST_FAILED,
            msg=ErrorMsg.NODE_REQUEST_FAILED,
            request_id=request_id,
        )

    except GatewayException:
        raise

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.exception(
            f"请求转发异常 | request_id={request_id} | "
            f"节点={selected_node} | 异常类型={type(e).__name__} | 耗时={elapsed_ms}ms"
        )
        raise GatewayException(
            code=ErrorCode.NODE_REQUEST_FAILED,
            msg=ErrorMsg.NODE_REQUEST_FAILED,
            request_id=request_id,
        )

    finally:
        # 无论成功或失败，必须减少节点活跃连接数，防止连接数泄漏
        await node_manager.decrement_connection(selected_node)


def _build_proxy_response(response: httpx.Response) -> Response:
    """
    将算力节点的HTTP响应转换为FastAPI响应
    过滤掉hop-by-hop头，原封不动返回内容和状态码

    Args:
        response: 算力节点的httpx响应

    Returns:
        FastAPI Response对象
    """
    # 构建响应头：过滤掉hop-by-hop头，避免传递冲突
    response_headers = {
        k: v
        for k, v in response.headers.items()
        if k.lower() not in _HOP_BY_HOP_HEADERS
    }

    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=response_headers,
    )


def _trigger_behavior_callback(
    response: httpx.Response,
    request_id: str,
    device_no: str,
    event_time: str,
) -> None:
    """
    触发行为识别回调后端（fire-and-forget）
    解析算力节点响应，如果推理成功则异步回调后端

    触发条件：
    - 算力节点返回HTTP 200
    - 响应体JSON中code=200且data非空
    - BehaviorCallback已初始化

    失败处理：
    - 回调失败仅记日志，不影响主流程
    - 解析响应失败仅记日志，不影响响应返回

    Args:
        response: 算力节点的HTTP响应
        request_id: 请求唯一ID
        device_no: 设备编号（从原始请求表单提取）
        event_time: 行为发生时间（从原始请求表单提取）
    """
    if not behavior_callback.is_initialized():
        logger.debug("行为识别回调处理器未初始化，跳过回调")
        return

    try:
        # 解析算力节点响应体
        response_data = json.loads(response.content)

        # 校验响应体格式：code=200且data非空
        if response_data.get("code") != 200:
            logger.debug(
                f"算力节点返回非成功状态，跳过回调 | "
                f"request_id={request_id} | 响应code={response_data.get('code')}"
            )
            return

        inference_data = response_data.get("data")
        if not inference_data:
            logger.debug(
                f"算力节点返回data为空，跳过回调 | request_id={request_id}"
            )
            return

        # 异步触发回调（fire-and-forget，不阻塞响应返回）
        asyncio.create_task(
            behavior_callback.handle_result(
                inference_data=inference_data,
                device_no=device_no,
                event_time=event_time,
            )
        )
        logger.info(
            f"行为识别回调已提交后台执行 | request_id={request_id} | "
            f"device_no={device_no}"
        )

    except json.JSONDecodeError as e:
        logger.warning(
            f"算力节点响应JSON解析失败，跳过回调 | "
            f"request_id={request_id} | 错误={str(e)[:200]}"
        )
    except Exception as e:
        logger.warning(
            f"触发行为识别回调异常（不影响主流程）| "
            f"request_id={request_id} | 错误类型={type(e).__name__} | "
            f"错误={str(e)[:200]}"
        )


# ==================== 词库配置同步广播（内部方法） ====================

async def broadcast_config(config_data: dict, config_version: str) -> dict:
    """
    词库配置同步广播
    向所有健康的算力节点广播词库配置，等待所有节点返回结果

    流程：
    1. 筛选所有健康节点
    2. 并发向所有健康节点发送POST /api/v1/internal/config/sync
    3. 等待所有节点响应（不因单个节点失败而中断其他节点）
    4. 统计成功/失败数量，更新成功节点的配置版本号
    5. 返回广播结果汇总

    Args:
        config_data: 词库配置数据
        config_version: 配置版本号

    Returns:
        广播结果：成功/失败节点数量及详情
    """
    logger.info(f"词库配置广播开始 | 版本={config_version}")

    # 获取所有健康节点
    healthy_nodes = [
        addr
        for addr, node in node_manager._nodes.items()
        if node.is_healthy
    ]

    if not healthy_nodes:
        logger.warning("词库配置广播失败 | 无可用健康节点")
        return {
            "success_count": 0,
            "fail_count": 0,
            "details": [],
            "error": "无可用健康节点",
        }

    # 并发向所有健康节点发送配置同步请求
    tasks = [
        _sync_config_to_node(addr, config_data, config_version)
        for addr in healthy_nodes
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 统计结果
    success_count = 0
    fail_count = 0
    details = []

    for addr, result in zip(healthy_nodes, results):
        if isinstance(result, Exception):
            # 单节点同步失败
            fail_count += 1
            details.append({"node": addr, "status": "failed", "error": str(result)})
            logger.error(f"词库配置同步失败 | 节点={addr} | 错误={result}")
        else:
            # 单节点同步成功
            success_count += 1
            # 更新节点配置版本号
            await node_manager.update_node_config_version(addr, config_version)
            details.append({"node": addr, "status": "success"})
            logger.info(f"词库配置同步成功 | 节点={addr} | 版本={config_version}")

    logger.info(
        f"词库配置广播完成 | 版本={config_version} | "
        f"成功={success_count} | 失败={fail_count}"
    )

    return {
        "success_count": success_count,
        "fail_count": fail_count,
        "details": details,
    }


async def _sync_config_to_node(
    address: str, config_data: dict, config_version: str
) -> dict:
    """
    向单个算力节点同步词库配置
    请求路径：POST http://{address}/api/v1/internal/config/sync

    Args:
        address: 节点地址
        config_data: 配置数据
        config_version: 配置版本号

    Returns:
        节点响应内容
    """
    client = await HttpClientSingleton.get_client()
    url = f"http://{address}/api/v1/internal/config/sync"

    response = await client.post(
        url,
        json={
            "config_type": ConfigType.KEYWORD,    # 算力节点Pydantic模型字段名(snake_case)
            "config_version": config_version,     # 算力节点Pydantic模型字段名(snake_case)
            "items": config_data,                 # 算力节点期望items，不是configData
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()
