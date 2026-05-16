"""
智能胸牌服务管理系统 - 知识库ID本地缓存
核心功能：
1. 内存字典缓存设备→知识库ID的映射关系，key为deviceNo，value为(knowledgeBaseId, cached_timestamp)
2. 自动过期：超过配置时间（默认24小时）的缓存条目自动失效
3. 提供查询方法：get(device_no) -> Optional[str]，未过期返回knowledgeBaseId，过期返回None
4. 定时清理过期条目，避免内存泄漏

设计约束：
- 单例模式，全局唯一实例（与DeviceStatusCache、AudioTempManager风格一致）
- 纯内存缓存，无磁盘IO，适配树莓派资源
- 全异步实现，定时清理不阻塞事件循环
- 线程安全：FastAPI事件循环单线程，无需加锁

使用示例：
    from knowledge_base_cache import KnowledgeBaseCache

    # 获取单例
    cache = KnowledgeBaseCache()

    # 存入缓存（从后端查询到knowledgeBaseId后调用）
    cache.set("BADGE0001", "dataset-001")

    # 查询缓存（算力节点请求知识库ID时先查缓存）
    knowledge_base_id = cache.get("BADGE0001")  # 返回 "dataset-001" 或 None（未命中/已过期）

    # 启动定时清理（在FastAPI lifespan startup阶段调用）
    await cache.start_cleanup()

    # 停止定时清理（在FastAPI lifespan shutdown阶段调用）
    await cache.stop_cleanup()
    cache.clear_all()
"""
import asyncio
import time
from typing import Dict, Optional, Tuple

from loguru import logger

from config import KNOWLEDGE_BASE_CACHE_TTL, KNOWLEDGE_BASE_CACHE_CLEANUP_INTERVAL


class KnowledgeBaseCache:
    """
    知识库ID本地缓存 - 单例模式

    职责：
    - 内存字典缓存设备→知识库ID的映射
    - 自动过期：超过KNOWLEDGE_BASE_CACHE_TTL（默认24小时）的条目自动失效
    - 提供get/set方法给其他模块使用
    - 定时清理过期条目，避免内存泄漏

    设计约束：
    - 单例模式，全局唯一实例
    - 纯内存缓存，无磁盘IO
    - 全异步实现
    - 线程安全：FastAPI事件循环单线程，无需加锁
    """

    _instance: Optional["KnowledgeBaseCache"] = None

    def __new__(cls, *args, **kwargs):
        """单例模式：确保全局只有一个KnowledgeBaseCache实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化（仅首次创建时执行）"""
        if hasattr(self, "_initialized") and self._initialized:
            return
        # deviceNo → (knowledgeBaseId, cached_timestamp) 映射
        self._cache: Dict[str, Tuple[str, float]] = {}
        # 定时清理任务句柄
        self._cleanup_task: Optional[asyncio.Task] = None
        self._initialized = False

    # ==================== 缓存操作 ====================

    def get(self, device_no: str) -> Optional[str]:
        """
        查询设备对应的知识库ID

        如果缓存命中且未过期，返回knowledgeBaseId
        如果缓存未命中或已过期，返回None

        Args:
            device_no: 设备编号

        Returns:
            knowledgeBaseId字符串，未命中或已过期返回None
        """
        entry = self._cache.get(device_no)
        if entry is None:
            logger.debug(f"知识库缓存未命中 | deviceNo={device_no}")
            return None

        knowledge_base_id, cached_at = entry
        # 检查是否过期
        if (time.time() - cached_at) > KNOWLEDGE_BASE_CACHE_TTL:
            # 过期则删除并返回None
            del self._cache[device_no]
            logger.debug(
                f"知识库缓存已过期 | deviceNo={device_no} | "
                f"knowledgeBaseId={knowledge_base_id}"
            )
            return None

        logger.debug(
            f"知识库缓存命中 | deviceNo={device_no} | "
            f"knowledgeBaseId={knowledge_base_id}"
        )
        return knowledge_base_id

    def set(self, device_no: str, knowledge_base_id: str) -> None:
        """
        存入设备→知识库ID的映射

        Args:
            device_no: 设备编号
            knowledge_base_id: 知识库ID
        """
        self._cache[device_no] = (knowledge_base_id, time.time())
        logger.debug(
            f"知识库缓存写入 | deviceNo={device_no} | "
            f"knowledgeBaseId={knowledge_base_id} | "
            f"当前缓存条目数={len(self._cache)}"
        )

    # ==================== 定时清理 ====================

    async def start_cleanup(self) -> None:
        """
        启动定时清理过期条目的后台任务
        在FastAPI lifespan startup阶段调用
        """
        if self._cleanup_task is not None:
            logger.warning("知识库缓存清理任务已在运行，跳过重复启动")
            return

        self._cleanup_task = asyncio.create_task(
            self._cleanup_loop(),
            name="knowledge-base-cache-cleanup",
        )
        logger.info(
            f"知识库缓存清理任务已启动 | "
            f"清理间隔={KNOWLEDGE_BASE_CACHE_CLEANUP_INTERVAL}s | "
            f"过期时间={KNOWLEDGE_BASE_CACHE_TTL}s"
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
            logger.info("知识库缓存清理任务已停止")

    async def _cleanup_loop(self) -> None:
        """
        定时清理过期条目的后台循环
        每KNOWLEDGE_BASE_CACHE_CLEANUP_INTERVAL秒执行一次清理
        """
        while True:
            try:
                await asyncio.sleep(KNOWLEDGE_BASE_CACHE_CLEANUP_INTERVAL)
                removed_count = self._cleanup_expired()
                if removed_count > 0:
                    logger.info(
                        f"知识库缓存清理 | 删除过期条目={removed_count} | "
                        f"剩余条目={len(self._cache)}"
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"知识库缓存清理异常 | "
                    f"错误类型={type(e).__name__} | 错误={str(e)[:200]}"
                )

    def _cleanup_expired(self) -> int:
        """
        清理所有过期的缓存条目

        Returns:
            清理的过期条目数量
        """
        now = time.time()
        expired_keys = [
            device_no
            for device_no, (_, cached_at) in self._cache.items()
            if (now - cached_at) > KNOWLEDGE_BASE_CACHE_TTL
        ]

        for device_no in expired_keys:
            del self._cache[device_no]
            logger.debug(f"知识库缓存过期清理 | deviceNo={device_no}")

        return len(expired_keys)

    # ==================== 资源清理 ====================

    def clear_all(self) -> None:
        """
        清空所有缓存
        在服务关闭或需要重置时调用
        """
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"知识库缓存已清空 | 清理数量={count}")
