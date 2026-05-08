"""
智能胸牌服务管理系统 - 幂等性缓存
实现内存版本幂等缓存，5分钟内相同请求直接返回缓存结果
预留Redis扩展接口，当前仅使用内存字典实现
"""
import time
import asyncio
from typing import Any, Optional, Dict, Tuple
from loguru import logger

from config import CACHE_TTL, CACHE_CLEANUP_INTERVAL


class IdempotencyCache:
    """
    幂等性缓存 - 内存实现
    - 基于请求唯一键缓存结果，5分钟内相同请求直接返回缓存
    - 定时清理过期缓存，防止内存泄漏
    - 预留Redis扩展接口：get/set方法签名与Redis操作一致，后续可无缝替换

    扩展说明：
    如需切换为Redis实现，只需：
    1. 将_cache字典替换为Redis客户端操作
    2. 利用Redis的TTL机制自动过期
    3. 本类的get/set/cleanup方法签名保持不变
    """

    def __init__(self, ttl: int = CACHE_TTL):
        """
        Args:
            ttl: 缓存存活时间（秒），默认5分钟
        """
        self._ttl = ttl
        # 缓存字典：key → (result, timestamp)
        self._cache: Dict[str, Tuple[Any, float]] = {}
        # 定时清理任务句柄
        self._cleanup_task: Optional[asyncio.Task] = None
        # 异步锁，保证并发安全
        self._lock = asyncio.Lock()

    async def start_cleanup_task(self) -> None:
        """启动定时清理过期缓存任务"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info(f"幂等缓存清理任务已启动 | TTL={self._ttl}s | 清理间隔={CACHE_CLEANUP_INTERVAL}s")

    async def stop_cleanup_task(self) -> None:
        """停止定时清理任务"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("幂等缓存清理任务已停止")

    async def _cleanup_loop(self) -> None:
        """定时清理过期缓存主循环"""
        while True:
            try:
                await asyncio.sleep(CACHE_CLEANUP_INTERVAL)
                await self.cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"缓存清理循环异常 | 异常信息={e}")

    async def get(self, key: str) -> Optional[Any]:
        """
        获取缓存结果
        - 缓存存在且未过期：返回缓存结果
        - 缓存不存在或已过期：返回None

        Args:
            key: 缓存键（如request_id）

        Returns:
            缓存的结果，不存在或过期返回None
        """
        async with self._lock:
            if key in self._cache:
                result, timestamp = self._cache[key]
                # 检查是否过期
                if time.time() - timestamp < self._ttl:
                    logger.debug(f"幂等缓存命中 | key={key}")
                    return result
                # 已过期，删除
                del self._cache[key]
                logger.debug(f"幂等缓存已过期 | key={key}")
        return None

    async def set(self, key: str, value: Any) -> None:
        """
        写入缓存
        - 写入时记录当前时间戳，用于TTL判断

        Args:
            key: 缓存键
            value: 缓存结果
        """
        async with self._lock:
            self._cache[key] = (value, time.time())
            logger.debug(f"幂等缓存写入 | key={key}")

    async def cleanup(self) -> None:
        """清理所有过期缓存项"""
        now = time.time()
        async with self._lock:
            expired_keys = [
                k for k, (_, ts) in self._cache.items()
                if now - ts >= self._ttl
            ]
            for k in expired_keys:
                del self._cache[k]
            if expired_keys:
                logger.info(f"清理过期缓存 | 数量={len(expired_keys)}")

    def get_cache_size(self) -> int:
        """获取当前缓存条目数量"""
        return len(self._cache)
