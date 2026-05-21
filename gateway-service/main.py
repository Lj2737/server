"""
智能胸牌服务管理系统 - 主网关节点入口
FastAPI实例创建、路由注册、生命周期管理
对外唯一暴露端口：8090，仅对接后端

v3.2架构变更：
- WebSocket端点统一为 /badge/v1/algorithm/ws/device/{deviceNo}
- 新增算力节点→主网关内部接口：dialog-completed、knowledge-base
- 新增知识库ID本地缓存（24小时TTL）
- 新增BackendClient便捷方法：report_dialog_completion、get_knowledge_base_id

值班播报架构：
- 后端→网关调用非流式TTS API合成→WebSocket分块推送胸牌→返回success

硬件状态上报架构（v3.1）：
- 硬件→算法：POST /badge/v1/internal/hardware/device-events（格式校验+缓存+透传）
- 算法→后端：POST /badge/v1/internal/ai/device-events（异步透传原始数据，不阻塞硬件响应）
- 算法仅做格式校验+缓存+透传，不修改、不新增、不删除硬件上报的任何业务字段
- 透传失败不影响硬件上报接口的返回（硬件永远收到200）
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from loguru import logger

from config import (
    GATEWAY_HOST,
    GATEWAY_PORT,
    BACKEND_BASE_URL,
    BEHAVIOR_CALLBACK_PATH,
    DIAGNOSIS_REQUEST_TIMEOUT,
    DIAGNOSIS_INTERNAL_PATH,
    CONFIG_SYNC_INTERNAL_PATH,
    WS_HEARTBEAT_INTERVAL,
    WS_HEARTBEAT_FAIL_THRESHOLD,
    DEVICE_STATUS_CACHE_TTL,
    DEVICE_STATUS_CLEANUP_INTERVAL,
    DEVICE_EVENT_FORWARD_PATH,
    RAW_AUDIO_TEMP_DIR,
    KNOWLEDGE_BASE_CACHE_TTL,
    KNOWLEDGE_BASE_CACHE_CLEANUP_INTERVAL,
)
from logger import setup_logger
from router import router, behavior_callback, initialize_dialog_callback
from algorithm_call_backend_router import algorithm_call_backend_router
from backend_call_algorithm_router import (
    backend_call_algorithm_router,
    diagnosis_handler,
    config_sync_handler,
)
from algorithm_forward_device_to_backend_router import (
    algorithm_forward_device_to_backend_router,
)
from hardware_call_algorithm_router import hardware_call_algorithm_router
from node_manager import NodeManager
from http_client import HttpClientSingleton
from utils import BackendClient
from exception import (
    global_exception_handler,
    validation_exception_handler,
    gateway_exception_handler,
    GatewayException,
)
# 值班播报模块（TTS API + WebSocket推送）
from duty_broadcast_router import (
    initialize_ai_dialog,
    ws_router,
    piper_tts_manager,
    ws_device_manager,
)
# 硬件状态上报模块
from hardware_router import (
    device_status_cache,
    initialize_forward as hardware_initialize_forward,
)
# 原始异常语音上传模块
from raw_audio_router import (
    audio_temp_manager,
    initialize as raw_audio_initialize,
)
# 知识库缓存模块
from knowledge_base_cache import KnowledgeBaseCache
from router import knowledge_base_cache as router_knowledge_base_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI生命周期管理
    使用asynccontextmanager实现startup/shutdown两个阶段

    - startup：初始化日志系统、预建httpx连接池、启动算力节点健康检查、
              初始化后端客户端、初始化行为识别回调、初始化AI对话完成回调、
              初始化AI诊断处理器、初始化词库配置同步处理器、
              初始化TTS API客户端、启动WebSocket设备心跳
    - shutdown：停止心跳、释放TTS API客户端、关闭所有WebSocket连接、
               停止健康检查、关闭httpx连接池、关闭后端客户端
    """
    # ========== 启动阶段 ==========
    # 初始化日志系统（控制台+文件双输出）
    setup_logger()
    logger.info("=" * 60)
    logger.info("智能胸牌服务管理系统 - 主网关节点启动中...")
    logger.info(f"监听地址={GATEWAY_HOST}:{GATEWAY_PORT}")
    logger.info("=" * 60)

    # 预建httpx连接池（懒加载变主动初始化，避免首次请求延迟）
    await HttpClientSingleton.get_client()
    logger.info("httpx异步客户端初始化完成")

    # 启动算力节点健康检查定时任务
    node_manager = NodeManager()
    await node_manager.start_health_check()
    logger.info("算力节点健康检查已启动")

    # 初始化后端通用客户端（复用连接池、鉴权、重试）
    backend_client = BackendClient()
    await backend_client.initialize()
    logger.info(f"后端通用客户端已初始化 | 后端地址={BACKEND_BASE_URL}")

    # 初始化行为识别回调处理器（注入BackendClient，用于回调后端）
    behavior_callback.initialize(backend_client=backend_client)
    logger.info(
        f"行为识别回调处理器已初始化 | "
        f"回调路径={BEHAVIOR_CALLBACK_PATH}"
    )

    initialize_ai_dialog(backend_client=backend_client)
    logger.info("AI dialog WebSocket pipeline initialized")
    # 初始化AI对话完成回调（注入BackendClient，用于回调后端dialog-completions）
    initialize_dialog_callback(backend_client=backend_client)
    logger.info("AI对话完成回调功能已初始化")

    # ========== 初始化AI诊断处理器（注入NodeManager，转发到算力节点） ==========
    diagnosis_handler.initialize(node_manager=node_manager)
    logger.info(
        f"AI诊断处理器已初始化 | 推理方式=算力节点调用LLM API | "
        f"内部路径={DIAGNOSIS_INTERNAL_PATH} | "
        f"超时={DIAGNOSIS_REQUEST_TIMEOUT}s"
    )

    # 初始化词库配置同步处理器（注入NodeManager，用于广播配置到所有健康节点）
    config_sync_handler.initialize(node_manager=node_manager)
    logger.info(
        f"词库配置同步处理器已初始化 | "
        f"内部路径={CONFIG_SYNC_INTERNAL_PATH}"
    )

    # ========== 初始化TTS API客户端 ==========
    piper_load_success = await piper_tts_manager.load_model()
    if piper_load_success:
        logger.info("TTS API客户端初始化成功，值班播报功能可用")
    else:
        logger.error(
            f"TTS API客户端初始化失败 | "
            f"错误={piper_tts_manager.load_error} | "
            f"值班播报功能不可用，其他功能正常"
        )

    # ========== WebSocket设备连接管理 ==========
    ws_device_manager.initialize()
    await ws_device_manager.start_heartbeat()
    logger.info(
        f"WebSocket设备管理器已启动 | "
        f"心跳间隔={WS_HEARTBEAT_INTERVAL}s | "
        f"失败阈值={WS_HEARTBEAT_FAIL_THRESHOLD}"
    )

    # ========== 硬件状态上报 ==========
    device_status_cache.initialize()
    await device_status_cache.start_cleanup()
    logger.info(
        f"设备状态缓存已启动 | "
        f"缓存过期时间={DEVICE_STATUS_CACHE_TTL}s | "
        f"清理间隔={DEVICE_STATUS_CLEANUP_INTERVAL}s"
    )

    # ========== 硬件状态透传后端 ==========
    # 注入BackendClient到硬件路由模块，用于异步透传硬件状态到后端
    hardware_initialize_forward(backend_client=backend_client)
    logger.info(
        f"硬件状态透传后端已初始化 | "
        f"透传路径={DEVICE_EVENT_FORWARD_PATH}"
    )

    # ========== 原始异常语音上传 ==========
    # 初始化临时文件管理器（自动创建目录、启动定时清理）
    await audio_temp_manager.initialize()
    logger.info(
        f"原始音频临时文件管理器已初始化 | "
        f"临时目录={RAW_AUDIO_TEMP_DIR}"
    )

    # 初始化原始异常语音上传模块
    raw_audio_initialize()
    logger.info("原始异常语音上传模块已初始化")

    # ========== 知识库ID缓存 ==========
    knowledge_base_cache_instance = KnowledgeBaseCache()
    await knowledge_base_cache_instance.start_cleanup()
    logger.info(
        f"知识库ID缓存已启动 | "
        f"缓存过期时间={KNOWLEDGE_BASE_CACHE_TTL}s | "
        f"清理间隔={KNOWLEDGE_BASE_CACHE_CLEANUP_INTERVAL}s"
    )

    logger.info("主网关节点启动完成，开始接收请求")

    yield  # 应用运行中，接收并处理请求

    # ========== 关闭阶段 ==========
    logger.info("主网关节点关闭中...")

    # 关闭知识库ID缓存（v3.2新增）
    await knowledge_base_cache_instance.stop_cleanup()
    knowledge_base_cache_instance.clear_all()
    logger.info("知识库ID缓存已关闭")

    # 关闭原始音频临时文件管理器（停止定时清理）
    await audio_temp_manager.close()
    logger.info("原始音频临时文件管理器已关闭")

    # 关闭设备状态缓存（停止定时清理）
    await device_status_cache.stop_cleanup()
    device_status_cache.clear_all()
    logger.info("设备状态缓存已关闭")

    # 关闭WebSocket设备管理器（停止心跳 + 关闭所有设备连接）
    await ws_device_manager.stop_heartbeat()
    await ws_device_manager.close_all()
    logger.info("WebSocket设备管理器已关闭")

    # 释放TTS API客户端资源
    piper_tts_manager.release()
    logger.info("TTS API客户端资源已释放")

    # 关闭后端通用客户端，释放连接池资源
    await backend_client.close()
    logger.info("后端通用客户端已关闭")

    # 停止健康检查定时任务
    await node_manager.stop_health_check()
    logger.info("健康检查已停止")

    # 关闭httpx客户端，释放连接池资源
    await HttpClientSingleton.close_client()
    logger.info("httpx客户端已关闭")

    logger.info("主网关节点已安全关闭")


# 创建FastAPI应用实例
app = FastAPI(
    title="智能胸牌服务管理系统 - 主网关节点",
    description=(
        "算法侧网关服务，负责请求路由、负载均衡、结果汇总。"
        "部署架构：1台主树莓派（网关） + 4台从树莓派（算力）\n"
        "v3.2：算力节点调用LLM API诊断 + TTS API播报 + 异常录音合并到voice-behaviors"
    ),
    version="3.2.0",
    lifespan=lifespan,
)

# 注册路由 - 算法调用后端接口（行为识别回调、AI对话完成、知识库查询）
app.include_router(algorithm_call_backend_router)

# 注册路由 - 后端调用算法接口（AI诊断、词库同步、值班播报）
app.include_router(backend_call_algorithm_router)

# 注册路由 - 网关基础接口（健康检查、节点状态、行为识别转发）
app.include_router(router, tags=["网关基础接口"])

# 注册路由 - WebSocket设备连接（/badge/v1/algorithm/ws/device/{deviceNo}）
app.include_router(ws_router)

# 注册路由 - 算法转发设备到后端接口
app.include_router(algorithm_forward_device_to_backend_router)

# 注册路由 - 硬件调用算法接口
app.include_router(hardware_call_algorithm_router)

# 注册全局异常处理器（顺序：先具体后通用）
app.add_exception_handler(GatewayException, gateway_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=GATEWAY_HOST,
        port=GATEWAY_PORT,
        log_level="info",
        access_log=False,  # 使用loguru替代uvicorn默认访问日志
    )
