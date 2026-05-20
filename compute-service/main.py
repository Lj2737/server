"""
智能胸牌服务管理系统 - 算力节点主程序入口
FastAPI实例创建、路由注册、生命周期管理
监听0.0.0.0:8091，仅与主网关通信
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from loguru import logger

from config import SERVICE_HOST, SERVICE_PORT
from logger import setup_logger
from router import router, idempotency_cache
import router as router_module
from models import asr_model, llm_model
from middleware import IPWhitelistMiddleware
from exception import (
    global_exception_handler,
    validation_exception_handler,
    compute_exception_handler,
    ComputeException,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI生命周期管理
    - startup：初始化日志、预加载模型、启动缓存清理
    - shutdown：释放模型资源、停止缓存清理
    """
    # ========== 启动阶段 ==========
    # 初始化日志系统
    setup_logger()
    logger.info("=" * 60)
    logger.info("智能胸牌服务管理系统 - 算力节点启动中...")
    logger.info(f"监听地址={SERVICE_HOST}:{SERVICE_PORT}")
    logger.info("=" * 60)

    # 预加载ASR模型（sherpa-onnx-sense-voice-small）
    asr_loaded = await asr_model.load()
    if asr_loaded:
        logger.info("ASR模型预加载成功")
    else:
        logger.error(f"ASR模型预加载失败 | 错误={asr_model.load_error}")

    # 初始化LLM API客户端
    llm_loaded = await llm_model.load()
    if llm_loaded:
        logger.info("LLM API客户端初始化成功")
    else:
        logger.error(f"LLM API客户端初始化失败 | 错误={llm_model.load_error}")

    # 模型加载状态汇总
    if asr_loaded and llm_loaded:
        logger.info("所有模型预加载完成，服务健康可用")
    else:
        logger.warning(
            "部分模型加载失败，健康检查将返回unhealthy，"
            "主网关将自动摘除本节点"
        )

    # 启动幂等缓存定时清理
    await idempotency_cache.start_cleanup_task()

    logger.info("算力节点启动完成，开始接收请求")

    yield  # 应用运行中

    # ========== 关闭阶段（优雅停机） ==========
    logger.info("算力节点关闭中...")

    # 停止幂等缓存清理任务
    await idempotency_cache.stop_cleanup_task()

    # 释放模型资源
    asr_model.release()
    llm_model.release()

    # 等待现有请求处理完成（简单等待活跃连接归零，最多等10秒）
    wait_time = 0
    max_wait = 10
    while router_module._active_connections > 0 and wait_time < max_wait:
        logger.info(f"等待现有请求完成 | 活跃连接={router_module._active_connections}")
        await asyncio.sleep(1)
        wait_time += 1

    if router_module._active_connections > 0:
        logger.warning(f"优雅停机超时 | 剩余活跃连接={router_module._active_connections}，强制关闭")

    logger.info("算力节点已安全关闭")


# 创建FastAPI应用实例
app = FastAPI(
    title="智能胸牌服务管理系统 - 算力节点",
    description=(
        "算力节点推理服务，部署ASR并通过API调用LLM。"
        "仅与主网关通信，不直接对接后端。"
        "部署架构：1台主树莓派（网关） + 4台从树莓派（算力）"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# 注册IP白名单中间件（仅允许主网关IP访问）
app.add_middleware(IPWhitelistMiddleware)

# 注册路由
app.include_router(router)

# 注册全局异常处理器
app.add_exception_handler(ComputeException, compute_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        log_level="info",
        access_log=False,  # 使用loguru替代uvicorn默认访问日志
    )
