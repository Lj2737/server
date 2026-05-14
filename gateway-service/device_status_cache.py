"""
智能胸牌服务管理系统 - 本地设备状态缓存
核心功能：
1. 内存字典缓存每个设备的最新完整状态，key为deviceNo，value为硬件上报的原始数据
2. 自动过期：超过配置时间（默认10分钟）未上报的设备自动从缓存中删除
3. 提供内部查询方法：get_device_status(device_no) -> Optional[dict]
4. 定时清理过期设备，避免内存泄漏

设计约束：
- 单例模式，全局唯一实例
- 纯内存缓存，无磁盘IO，适配树莓派资源
- 全异步实现，定时清理不阻塞事件循环
- 缓存数据为硬件上报的原始数据，不做任何修改

使用示例：
    from device_status_cache import DeviceStatusCache

    # 获取单例
    cache = DeviceStatusCache()

    # 初始化（在FastAPI lifespan startup阶段调用）
    cache.initialize()

    # 更新设备状态（硬件上报时调用）
    cache.update_device_status("BADGE0001", {
        "deviceNo": "BADGE0001",
        "eventType": "HEARTBEAT",
        "reportTime": "2026-05-13 15:00:00",
        "payload": {"batteryLevel": 86, "signalLevel": 4}
    })

    # 查询设备最新状态
    status = cache.get_device_status("BADGE0001")

    # 获取所有在线设备
    devices = cache.get_all_devices()

    # 停止定时清理（服务关闭时调用）
    await cache.stop_cleanup()
"""
import asyncio
import time
from typing import Dict, List, Optional

from loguru import logger

from config import DEVICE_STATUS_CACHE_TTL, DEVICE_STATUS_CLEANUP_INTERVAL


class _DeviceStatusEntry:
    """
    单个设备缓存条目
    包含设备状态数据和最后更新时间戳

    Attributes:
        device_no: 设备编号
        data: 硬件上报的原始数据
        updated_at: 最后更新时间戳（秒）
    """

    __slots__ = ("device_no", "data", "updated_at")

    def __init__(self, device_no: str, data: dict):
        self.device_no = device_no
        self.data = data
        self.updated_at = time.time()

    def touch(self) -> None:
        """更新最后访问时间"""
        self.updated_at = time.time()

    def is_expired(self, ttl: int) -> bool:
        """
        判断是否已过期

        Args:
            ttl: 过期时间（秒）

        Returns:
            True: 已过期
            False: 未过期
        """
        return (time.time() - self.updated_at) > ttl


class DeviceStatusCache:
    """
    本地设备状态缓存 - 单例模式

    职责：
    - 内存字典缓存每个设备的最新完整状态
    - 自动过期：超过配置时间未上报的设备自动删除
    - 提供内部查询方法给其他模块使用
    - 定时清理过期设备，避免内存泄漏

    设计约束：
    - 单例模式，全局唯一实例
    - 纯内存缓存，无磁盘IO
    - 全异步实现
    - 缓存数据为硬件上报的原始数据，不修改
    - 线程安全：FastAPI事件循环单线程，无需加锁
    """

    _instance: Optional["DeviceStatusCache"] = None

    def __new__(cls, *args, **kwargs):
        """单例模式：确保全局只有一个DeviceStatusCache实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化（仅首次创建时执行）"""
        if hasattr(self, "_initialized") and self._initialized:
            return
        # deviceNo → _DeviceStatusEntry 映射
        self._cache: Dict[str, _DeviceStatusEntry] = {}
        # 定时清理任务句柄
        self._cleanup_task: Optional[asyncio.Task] = None
        self._initialized = False

    def initialize(self) -> None:
        """
        初始化设备状态缓存
        在FastAPI lifespan startup阶段调用
        """
        if self._initialized:
            return
        self._initialized = True
        logger.info(
            f"设备状态缓存初始化完成 | "
            f"过期时间={DEVICE_STATUS_CACHE_TTL}s | "
            f"清理间隔={DEVICE_STATUS_CLEANUP_INTERVAL}s"
        )

    # ==================== 状态更新 ====================

    def update_device_status(self, device_no: str, data: dict) -> None:
        """
        更新设备状态缓存
        硬件上报时调用，更新或新增设备缓存条目

        Args:
            device_no: 设备编号
            data: 硬件上报的原始数据（完整JSON，包含deviceNo/eventType/reportTime/payload）
        """
        if device_no in self._cache:
            # 更新已有条目
            entry = self._cache[device_no]
            entry.data = data
            entry.touch()
            logger.debug(
                f"设备状态缓存更新 | deviceNo={device_no} | "
                f"eventType={data.get('eventType', '')} | "
                f"reportTime={data.get('reportTime', '')}"
            )
        else:
            # 新增条目
            self._cache[device_no] = _DeviceStatusEntry(device_no, data)
            logger.debug(
                f"设备状态缓存新增 | deviceNo={device_no} | "
                f"当前缓存设备数={len(self._cache)}"
            )

    # ==================== 状态查询 ====================

    def get_device_status(self, device_no: str) -> Optional[dict]:
        """
        查询设备最新状态
        供其他模块内部调用（如后续转发后端时使用）

        Args:
            device_no: 设备编号

        Returns:
            设备最新状态数据（原始上报数据），设备不存在或已过期返回None
        """
        entry = self._cache.get(device_no)
        if entry is None:
            return None

        # 检查是否过期
        if entry.is_expired(DEVICE_STATUS_CACHE_TTL):
            # 过期则删除并返回None
            del self._cache[device_no]
            logger.debug(
                f"设备状态缓存过期删除 | deviceNo={device_no}"
            )
            return None

        return entry.data

    def get_all_devices(self) -> List[dict]:
        """
        获取所有未过期的设备状态列表

        Returns:
            所有未过期设备的最新状态数据列表
        """
        self._cleanup_expired()
        return [entry.data for entry in self._cache.values()]

    def get_device_count(self) -> int:
        """
        获取当前缓存中的设备数量（包含可能过期但尚未清理的）

        Returns:
            缓存中的设备数量
        """
        return len(self._cache)

    def is_device_cached(self, device_no: str) -> bool:
        """
        检查设备是否在缓存中且未过期

        Args:
            device_no: 设备编号

        Returns:
            True: 设备在缓存中且未过期
            False: 设备不在缓存中或已过期
        """
        return self.get_device_status(device_no) is not None

    # ==================== 定时清理 ====================

    async def start_cleanup(self) -> None:
        """
        启动定时清理过期设备的后台任务
        在FastAPI lifespan startup阶段调用
        """
        if self._cleanup_task is not None:
            logger.warning("设备状态缓存清理任务已在运行，跳过重复启动")
            return

        self._cleanup_task = asyncio.create_task(
            self._cleanup_loop(),
            name="device-status-cleanup",
        )
        logger.info(
            f"设备状态缓存清理任务已启动 | "
            f"清理间隔={DEVICE_STATUS_CLEANUP_INTERVAL}s | "
            f"过期时间={DEVICE_STATUS_CACHE_TTL}s"
        )

    async def stop_cleanup(self) -> None:
        """
        停止定时清理任务
        在FastAPI lifespan shutdown阶段调用
        """
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("设备状态缓存清理任务已停止")

    async def _cleanup_loop(self) -> None:
        """
        定时清理过期设备的后台循环
        每DEVICE_STATUS_CLEANUP_INTERVAL秒执行一次清理
        """
        while True:
            try:
                await asyncio.sleep(DEVICE_STATUS_CLEANUP_INTERVAL)
                removed_count = self._cleanup_expired()
                if removed_count > 0:
                    logger.info(
                        f"设备状态缓存清理 | 删除过期设备={removed_count} | "
                        f"剩余设备={len(self._cache)}"
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"设备状态缓存清理异常 | "
                    f"错误类型={type(e).__name__} | 错误={str(e)[:200]}"
                )

    def _cleanup_expired(self) -> int:
        """
        清理所有过期的设备缓存条目

        Returns:
            清理的过期设备数量
        """
        expired_devices = [
            device_no
            for device_no, entry in self._cache.items()
            if entry.is_expired(DEVICE_STATUS_CACHE_TTL)
        ]

        for device_no in expired_devices:
            del self._cache[device_no]
            logger.debug(f"设备状态缓存过期清理 | deviceNo={device_no}")

        return len(expired_devices)

    # ==================== 资源清理 ====================

    def clear_all(self) -> None:
        """
        清空所有缓存
        在服务关闭或需要重置时调用
        """
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"设备状态缓存已清空 | 清理数量={count}")
