"""
智能胸牌服务管理系统 - 主网关节点入口
FastAPI实例创建、路由注册、生命周期管理
对外唯一暴露端口：8090，仅对接后端

值班播报架构：
- 旧架构（已废弃）：后端→网关→算力节点TTS→返回结果→网关转发给后端
- 新架构（当前）：后端→网关本地Piper TTS合成→WebSocket流式推送胸牌→返回success
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
    RECORDING_UPLOAD_PATH,
    DIAGNOSIS_INTERNAL_PATH,
    CONFIG_SYNC_INTERNAL_PATH,
    PIPER_MODEL_PATH,
    PIPER_TARGET_SAMPLE_RATE,
    PIPER_TARGET_SAMPLE_WIDTH,
    PIPER_TARGET_CHANNELS,
    WS_HEARTBEAT_INTERVAL,
    WS_HEARTBEAT_FAIL_THRESHOLD,
    DEVICE_STATUS_CACHE_TTL,
    DEVICE_STATUS_CLEANUP_INTERVAL,
    DEVICE_EVENT_FORWARD_PATH,
)
from logger import setup_logger
from router import router, behavior_callback
from backend_router import backend_router, diagnosis_handler, config_sync_handler
from recording_upload import RecordingUpload
from node_manager import NodeManager
from http_client import HttpClientSingleton
from utils import BackendClient, AudioTempStorage
from exception import (
    global_exception_handler,
    validation_exception_handler,
    gateway_exception_handler,
    GatewayException,
)
# 值班播报新架构模块
from duty_broadcast_router import (
    duty_broadcast_router,
    piper_tts_manager,
    ws_device_manager,
)
# 硬件状态上报模块
from hardware_router import (
    hardware_router,
    device_status_cache,
    initialize_forward as hardware_initialize_forward,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI生命周期管理
    使用asynccontextmanager实现startup/shutdown两个阶段

    - startup：初始化日志系统、预建httpx连接池、启动算力节点健康检查、
              初始化后端客户端、初始化音频临时存储、
              加载Piper TTS模型、启动WebSocket设备心跳
    - shutdown：停止心跳、关闭TTS模型、关闭所有WebSocket连接、
               停止健康检查、关闭httpx连接池、关闭后端客户端、关闭音频临时存储
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

    # 初始化异常音频临时存储（自动创建目录、启动定时清理）
    audio_storage = AudioTempStorage()
    await audio_storage.initialize()
    logger.info("异常音频临时存储已初始化")

    # 初始化行为识别回调处理器（注入BackendClient，用于回调后端）
    behavior_callback.initialize(backend_client=backend_client)
    logger.info(
        f"行为识别回调处理器已初始化 | "
        f"回调路径={BEHAVIOR_CALLBACK_PATH}"
    )

    # 初始化异常行为片段录音上传处理器（注入BackendClient和AudioTempStorage）
    recording_upload = RecordingUpload()
    await recording_upload.initialize(
        backend_client=backend_client,
        audio_storage=audio_storage,
    )
    logger.info(
        f"录音上传处理器已初始化 | "
        f"上传路径={RECORDING_UPLOAD_PATH}"
    )

    # 将录音上传处理器注册为BehaviorCallback的音频处理器
    # 当行为识别结果为ABNORMAL且包含abnormal_audio_clip时，自动触发上传
    behavior_callback.register_audio_clip_handler(recording_upload.handle_audio_clip)
    logger.info("录音上传处理器已注册为BehaviorCallback音频处理器，ABNORMAL行为将自动触发录音上传")

    # 初始化AI时段诊断处理器（注入NodeManager，用于负载均衡选择算力节点）
    diagnosis_handler.initialize(node_manager=node_manager)
    logger.info(
        f"AI诊断处理器已初始化 | "
        f"内部路径={DIAGNOSIS_INTERNAL_PATH}"
    )

    # 初始化词库配置同步处理器（注入NodeManager，用于广播配置到所有健康节点）
    config_sync_handler.initialize(node_manager=node_manager)
    logger.info(
        f"词库配置同步处理器已初始化 | "
        f"内部路径={CONFIG_SYNC_INTERNAL_PATH}"
    )

    # ========== 新架构：Piper TTS本地部署 ==========
    # 加载Piper TTS模型（服务启动时预加载，避免首次请求延迟）
    tts_load_success = await piper_tts_manager.load_model()
    if tts_load_success:
        logger.info(
            f"Piper TTS模型加载成功 | "
            f"模型={PIPER_MODEL_PATH} | "
            f"输出格式={PIPER_TARGET_SAMPLE_RATE}Hz/"
            f"{PIPER_TARGET_SAMPLE_WIDTH * 8}bit/"
            f"{PIPER_TARGET_CHANNELS}ch PCM"
        )
    else:
        logger.error(
            f"Piper TTS模型加载失败 | 模型={PIPER_MODEL_PATH} | "
            f"错误={piper_tts_manager.load_error} | "
            f"值班播报功能不可用，其他功能正常"
        )

    # ========== 新架构：WebSocket设备连接管理 ==========
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

    logger.info("主网关节点启动完成，开始接收请求")

    yield  # 应用运行中，接收并处理请求

    # ========== 关闭阶段 ==========
    logger.info("主网关节点关闭中...")

    # 关闭设备状态缓存（停止定时清理）
    await device_status_cache.stop_cleanup()
    device_status_cache.clear_all()
    logger.info("设备状态缓存已关闭")

    # 关闭WebSocket设备管理器（停止心跳 + 关闭所有设备连接）
    await ws_device_manager.stop_heartbeat()
    await ws_device_manager.close_all()
    logger.info("WebSocket设备管理器已关闭")

    # 释放Piper TTS模型资源
    piper_tts_manager.release()
    logger.info("Piper TTS模型资源已释放")

    # 关闭异常音频临时存储
    await audio_storage.close()
    logger.info("音频临时存储已关闭")

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
        "值班播报：本地Piper TTS + WebSocket流式推送"
    ),
    version="4.1.0",
    lifespan=lifespan,
)

# 注册路由 - 算力节点转发（行为识别、诊断总结、健康检查、节点状态查询）
app.include_router(router)

# 注册路由 - 后端调用算法接口（AI诊断、词库同步、旧架构TTS）
app.include_router(backend_router)

# 注册路由 - 值班播报新架构（Piper TTS本地合成 + WebSocket推送）
app.include_router(duty_broadcast_router)

# 注册路由 - 硬件状态上报
app.include_router(hardware_router)

# 注册全局异常处理器
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
