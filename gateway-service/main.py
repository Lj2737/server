"""
智能胸牌服务管理系统 - 主网关节点入口
FastAPI实例创建、路由注册、生命周期管理
对外唯一暴露端口：8090，仅对接后端
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from loguru import logger

from config import GATEWAY_HOST, GATEWAY_PORT
from logger import setup_logger
from router import router
from node_manager import NodeManager
from http_client import HttpClientSingleton
from exception import (
    global_exception_handler,
    validation_exception_handler,
    gateway_exception_handler,
    GatewayException,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI生命周期管理
    使用asynccontextmanager实现startup/shutdown两个阶段

    - startup：初始化日志系统、预建httpx连接池、启动算力节点健康检查
    - shutdown：停止健康检查、关闭httpx连接池释放资源
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

    logger.info("主网关节点启动完成，开始接收请求")

    yield  # 应用运行中，接收并处理请求

    # ========== 关闭阶段 ==========
    logger.info("主网关节点关闭中...")

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
        "部署架构：1台主树莓派（网关） + 4台从树莓派（算力）"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# 注册路由（推理转发、健康检查、节点状态查询）
app.include_router(router)

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
