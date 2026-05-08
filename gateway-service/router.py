"""
智能胸牌服务管理系统 - 路由转发与基础接口定义
核心功能：
1. 推理请求的路由转发（行为识别、诊断总结）
2. 网关健康检查接口
3. 节点状态查询接口
4. 词库配置同步广播（内部方法，非对外接口）
"""
import asyncio
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
    返回网关服务状态、可用节点数量、总请求数
    当有健康节点时状态为healthy，否则为degraded
    """
    healthy_count = node_manager.get_healthy_node_count()
    total_requests = node_manager.get_total_requests()

    return {
        "code": 200,
        "msg": "ok",
        "data": {
            "status": "healthy" if healthy_count > 0 else "degraded",
            "healthy_nodes": healthy_count,
            "total_nodes": len(node_manager._nodes),
            "total_requests": total_requests,
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


# ==================== 推理请求转发 ====================

@router.post("/api/v1/gateway/behavior-recognition")
@router.post("/api/v1/gateway/diagnosis-summary")
async def forward_inference_request(request: Request):
    """
    推理请求统一转发入口
    处理行为识别和诊断总结两类推理请求，流程如下：
    1. 获取或生成请求唯一ID（X-Request-ID），确保幂等性
    2. 根据路由映射表，将外部路径映射为算力节点内部路径
    3. 通过「最小连接数」负载均衡选择最优节点
    4. 完整转发请求（Header、Body、FormData、Query参数）
    5. 原封不动返回算力节点响应
    6. 异常处理：无可用节点返回503，超时返回504
    """
    # 获取或生成请求唯一ID（幂等键），无则自动生成UUID填充
    request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())

    # 获取请求原始路径
    original_path = request.url.path

    # 根据路由映射表，获取算力节点内部路径
    internal_path = ROUTE_MAPPING.get(original_path)
    if internal_path is None:
        # 未找到路由映射，返回404
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
        # 所有算力节点均不可用，返回503
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

    # 记录请求开始时间，用于计算响应时长
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

        # 记录转发结果日志（包含关键信息）
        logger.info(
            f"请求转发完成 | request_id={request_id} | "
            f"节点={selected_node} | 状态码={response.status_code} | "
            f"耗时={elapsed_ms}ms"
        )

        # 构建响应头：过滤掉hop-by-hop头，避免传递冲突
        response_headers = {
            k: v
            for k, v in response.headers.items()
            if k.lower() not in _HOP_BY_HOP_HEADERS
        }

        # 原封不动返回算力节点的响应内容和状态码
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
        )

    except httpx.TimeoutException as e:
        # 请求超时，返回504 Gateway Timeout
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
        # 算力节点返回HTTP错误状态码，返回502 Bad Gateway
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
        # 已经是GatewayException，直接向上抛出
        raise

    except Exception as e:
        # 其他未知异常，返回502 Bad Gateway
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
            "configType": ConfigType.KEYWORD,  # 配置类型：词库
            "configVersion": config_version,
            "configData": config_data,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()
