"""
智能胸牌服务管理系统 - 算力节点管理器
核心功能：
1. 维护固定的4台算力节点列表
2. 实现「最小连接数」负载均衡算法
3. 自动健康检查：每5秒检查，连续2次失败摘除，连续3次成功恢复
4. 提供节点状态查询接口
设计模式：单例模式，线程安全，支持并发操作
"""
import asyncio
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from threading import Lock

import httpx
from loguru import logger

from config import (
    COMPUTE_NODES,
    HEALTH_CHECK_INTERVAL,
    HEALTH_CHECK_FAIL_THRESHOLD,
    HEALTH_CHECK_RECOVER_THRESHOLD,
    HEALTH_CHECK_TIMEOUT,
    DATETIME_FORMAT,
)
from http_client import HttpClientSingleton


@dataclass
class NodeInfo:
    """
    算力节点信息数据类
    记录每个节点的健康状态、连接数、配置版本等关键指标
    """
    address: str                          # 节点地址，格式：ip:port
    is_healthy: bool = False              # 当前是否健康（在可用节点池中）
    active_connections: int = 0           # 当前活跃连接数
    consecutive_failures: int = 0         # 连续健康检查失败次数
    consecutive_successes: int = 0        # 连续健康检查成功次数
    config_version: str = ""              # 节点当前词库配置版本号
    total_requests: int = 0              # 累计处理请求数
    last_check_time: float = 0.0         # 上次健康检查时间戳
    last_healthy_time: float = 0.0       # 上次健康状态时间戳


class NodeManager:
    """
    算力节点管理器 - 单例模式
    核心职责：
    1. 维护4台算力节点列表，管理节点健康状态
    2. 实现「最小连接数」负载均衡算法：每次请求选择当前活跃连接数最少的健康节点
    3. 自动健康检查：每5秒并发检查所有节点，连续2次失败摘除，连续3次成功恢复
    4. 维护每个节点的活跃连接数：请求开始+1，请求结束-1（无论成功/失败）
    5. 提供节点状态查询接口
    """

    _instance: Optional["NodeManager"] = None  # 单例实例
    _init_lock: Lock = Lock()                  # 初始化线程安全锁
    _initialized: bool = False                 # 是否已初始化标记

    def __new__(cls, *args, **kwargs):
        """单例模式：确保全局只有一个NodeManager实例"""
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self):
        """
        初始化节点管理器（仅首次创建时执行）
        - 创建4台算力节点的NodeInfo
        - 初始化异步锁用于并发安全
        """
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return

        # 初始化节点字典：address → NodeInfo
        self._nodes: Dict[str, NodeInfo] = {}
        for node_addr in COMPUTE_NODES:
            self._nodes[node_addr] = NodeInfo(address=node_addr)

        # 健康检查定时任务句柄
        self._health_check_task: Optional[asyncio.Task] = None

        # 异步锁，用于并发操作节点状态时的安全保护
        self._async_lock = asyncio.Lock()

        # 标记初始化完成
        self._initialized = True
        logger.info(
            f"节点管理器初始化完成 | 节点数量={len(self._nodes)} | "
            f"节点列表={COMPUTE_NODES}"
        )

    # ==================== 健康检查 ====================

    async def start_health_check(self) -> None:
        """
        启动健康检查定时任务
        在FastAPI lifespan startup阶段调用
        """
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info("健康检查定时任务已启动")

    async def stop_health_check(self) -> None:
        """
        停止健康检查定时任务
        在FastAPI lifespan shutdown阶段调用
        """
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            logger.info("健康检查定时任务已停止")

    async def _health_check_loop(self) -> None:
        """
        健康检查主循环
        每HEALTH_CHECK_INTERVAL秒对所有节点执行一次并发健康检查
        异常不会导致循环退出（catch后继续下一轮）
        """
        while True:
            try:
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
                await self._check_all_nodes()
            except asyncio.CancelledError:
                # 任务被取消（shutdown），正常退出循环
                break
            except Exception as e:
                # 防止健康检查循环因意外异常退出
                logger.exception(f"健康检查循环异常 | 异常信息={e}")

    async def _check_all_nodes(self) -> None:
        """
        并发检查所有算力节点的健康状态
        使用asyncio.gather并发请求，提升检查效率
        return_exceptions=True确保单个节点异常不影响其他节点检查
        """
        tasks = [self._check_single_node(addr) for addr in self._nodes]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_single_node(self, address: str) -> None:
        """
        检查单个节点的健康状态
        - 请求 GET http://{address}/health
        - 成功（200）：连续成功计数+1，达到HEALTH_CHECK_RECOVER_THRESHOLD后恢复节点
        - 失败（非200/超时/异常）：连续失败计数+1，达到HEALTH_CHECK_FAIL_THRESHOLD后摘除节点

        Args:
            address: 节点地址，格式 ip:port
        """
        node = self._nodes[address]
        node.last_check_time = time.time()

        try:
            client = await HttpClientSingleton.get_client()
            url = f"http://{address}/health"
            response = await client.get(url, timeout=HEALTH_CHECK_TIMEOUT)

            if response.status_code == 200:
                # 健康检查成功
                async with self._async_lock:
                    node.consecutive_successes += 1
                    node.consecutive_failures = 0  # 成功时重置失败计数
                    node.last_healthy_time = time.time()

                    # 连续成功达到恢复阈值，且当前不在可用池 → 恢复节点
                    if (
                        not node.is_healthy
                        and node.consecutive_successes >= HEALTH_CHECK_RECOVER_THRESHOLD
                    ):
                        node.is_healthy = True
                        logger.info(
                            f"节点恢复上线 | 节点={address} | "
                            f"连续成功次数={node.consecutive_successes}"
                        )
                    else:
                        logger.debug(
                            f"节点健康检查成功 | 节点={address} | "
                            f"连续成功={node.consecutive_successes}"
                        )
            else:
                # 响应状态码非200，视为检查失败
                await self._handle_check_failure(
                    node, address, f"HTTP状态码={response.status_code}"
                )

        except Exception as e:
            # 请求异常（超时、连接拒绝等），视为检查失败
            await self._handle_check_failure(node, address, str(e))

    async def _handle_check_failure(
        self, node: NodeInfo, address: str, reason: str
    ) -> None:
        """
        处理健康检查失败
        - 累加连续失败计数，重置连续成功计数
        - 达到HEALTH_CHECK_FAIL_THRESHOLD时摘除节点（标记为不可用）

        Args:
            node: 节点信息对象
            address: 节点地址
            reason: 失败原因（仅供日志记录）
        """
        async with self._async_lock:
            node.consecutive_failures += 1
            node.consecutive_successes = 0  # 失败时重置成功计数

            # 连续失败达到摘除阈值，且当前还在可用池 → 摘除节点
            if (
                node.is_healthy
                and node.consecutive_failures >= HEALTH_CHECK_FAIL_THRESHOLD
            ):
                node.is_healthy = False
                logger.warning(
                    f"节点摘除下线 | 节点={address} | "
                    f"连续失败次数={node.consecutive_failures} | 原因={reason}"
                )
            else:
                logger.debug(
                    f"节点健康检查失败 | 节点={address} | "
                    f"连续失败={node.consecutive_failures} | 原因={reason}"
                )

    # ==================== 负载均衡 ====================

    def get_least_connection_node(self) -> Optional[str]:
        """
        「最小连接数」负载均衡算法
        从所有健康的节点中，选择当前活跃连接数最少的节点进行请求转发

        Returns:
            选中的节点地址（ip:port），无可用健康节点时返回None
        """
        # 筛选健康节点
        healthy_nodes = {
            addr: node for addr, node in self._nodes.items() if node.is_healthy
        }

        if not healthy_nodes:
            logger.warning("负载均衡失败 | 无可用健康节点")
            return None

        # 选择活跃连接数最少的节点
        selected_addr = min(
            healthy_nodes,
            key=lambda addr: healthy_nodes[addr].active_connections,
        )
        logger.debug(
            f"负载均衡选节点 | 选中={selected_addr} | "
            f"活跃连接数={healthy_nodes[selected_addr].active_connections} | "
            f"可用节点数={len(healthy_nodes)}"
        )
        return selected_addr

    # ==================== 连接计数管理 ====================

    async def increment_connection(self, address: str) -> None:
        """
        增加节点活跃连接数 + 累计请求数
        请求开始转发时调用

        Args:
            address: 节点地址
        """
        async with self._async_lock:
            if address in self._nodes:
                self._nodes[address].active_connections += 1
                self._nodes[address].total_requests += 1
                logger.debug(
                    f"节点连接数+1 | 节点={address} | "
                    f"当前连接数={self._nodes[address].active_connections}"
                )

    async def decrement_connection(self, address: str) -> None:
        """
        减少节点活跃连接数
        请求结束（无论成功/失败）时调用，确保连接数不会泄漏

        Args:
            address: 节点地址
        """
        async with self._async_lock:
            if address in self._nodes:
                # 防止意外减到负数
                self._nodes[address].active_connections = max(
                    0, self._nodes[address].active_connections - 1
                )
                logger.debug(
                    f"节点连接数-1 | 节点={address} | "
                    f"当前连接数={self._nodes[address].active_connections}"
                )

    # ==================== 状态查询 ====================

    def get_nodes_status(self) -> List[dict]:
        """
        获取所有节点的详细状态信息
        供 GET /badge/v1/gateway/nodes 接口使用

        Returns:
            节点状态列表，每个节点包含：地址、健康状态、连接数、配置版本等
        """
        return [
            {
                "address": node.address,
                "is_healthy": node.is_healthy,
                "active_connections": node.active_connections,
                "consecutive_failures": node.consecutive_failures,
                "consecutive_successes": node.consecutive_successes,
                "config_version": node.config_version,
                "total_requests": node.total_requests,
                # 时间戳转为可读格式，0表示尚未检查
                "last_check_time": (
                    time.strftime(DATETIME_FORMAT, time.localtime(node.last_check_time))
                    if node.last_check_time > 0
                    else ""
                ),
                "last_healthy_time": (
                    time.strftime(DATETIME_FORMAT, time.localtime(node.last_healthy_time))
                    if node.last_healthy_time > 0
                    else ""
                ),
            }
            for node in self._nodes.values()
        ]

    def get_healthy_node_count(self) -> int:
        """
        获取当前健康可用节点数量
        供 GET /health 接口使用

        Returns:
            健康节点数量
        """
        return sum(1 for node in self._nodes.values() if node.is_healthy)

    def get_total_requests(self) -> int:
        """
        获取所有节点累计请求总数
        供 GET /health 接口使用

        Returns:
            累计请求总数
        """
        return sum(node.total_requests for node in self._nodes.values())

    # ==================== 配置版本管理 ====================

    async def update_node_config_version(self, address: str, version: str) -> None:
        """
        更新节点的配置版本号
        词库配置同步成功后调用

        Args:
            address: 节点地址
            version: 新的配置版本号
        """
        async with self._async_lock:
            if address in self._nodes:
                self._nodes[address].config_version = version
                logger.info(f"节点配置版本更新 | 节点={address} | 版本={version}")
