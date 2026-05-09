"""
智能胸牌服务管理系统 - 异常音频片段临时存储工具
核心功能：
1. 临时存储从算力节点返回的异常音频片段base64，解码成WAV文件保存到本地
2. 临时目录可配置，自动创建不存在的目录
3. 定时清理：每小时清理一次1小时前的临时文件
4. 适配树莓派存储，避免临时文件占满空间
5. 提供文件路径供后续上传到后端使用

严格对齐v3文档：
- 5.3 异常行为片段录音上传：算法确认异常行为后，截取异常行为语音片段
- 上传接口 POST /internal/badge/ai/recordings 使用 multipart/form-data

使用示例：
    from utils import AudioTempStorage  # 或 from utils.audio_temp_storage import AudioTempStorage

    # 初始化（在FastAPI lifespan startup中调用）
    storage = AudioTempStorage()
    await storage.initialize()

    # 保存异常音频片段
    file_path = await storage.save(
        event_id="AI_BEHAVIOR_20260508103100_001234",
        behavior_event_id="AI_BEHAVIOR_20260508103100_001234",
        base64_audio="UklGRi4AAABXQVZFZm10...",
    )
    # file_path = "temp/audio_clips/AI_BEHAVIOR_20260508103100_001234.wav"

    # 读取文件用于上传
    file_data = await storage.read(file_path)

    # 上传完毕后删除临时文件
    await storage.delete(file_path)

    # 关闭（在FastAPI lifespan shutdown中调用）
    await storage.close()
"""
import os
import asyncio
import base64
import time
from typing import Optional

from loguru import logger

from config import (
    AUDIO_TEMP_DIR,
    AUDIO_TEMP_FILE_TTL,
    AUDIO_TEMP_CLEANUP_INTERVAL,
)


class AudioTempStorage:
    """
    异常音频片段临时存储工具
    - 将base64音频数据解码保存为WAV文件
    - 定时清理过期临时文件，防止占满树莓派存储
    - 单例模式，全局统一管理
    """

    _instance: Optional["AudioTempStorage"] = None

    def __new__(cls, *args, **kwargs):
        """单例模式：确保全局只有一个AudioTempStorage实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        初始化临时存储（仅首次创建时执行）
        """
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._cleanup_task: Optional[asyncio.Task] = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        初始化临时存储
        - 创建临时目录
        - 启动定时清理任务
        在FastAPI lifespan startup阶段调用
        """
        if self._initialized:
            return

        # 创建临时目录（含子目录，自动创建不存在的父目录）
        os.makedirs(AUDIO_TEMP_DIR, exist_ok=True)
        logger.info(f"音频临时存储目录就绪 | 路径={AUDIO_TEMP_DIR}")

        # 启动定时清理任务
        self._start_cleanup_task()
        self._initialized = True
        logger.info(
            f"音频临时存储初始化完成 | "
            f"文件过期={AUDIO_TEMP_FILE_TTL}s | "
            f"清理间隔={AUDIO_TEMP_CLEANUP_INTERVAL}s"
        )

    async def close(self) -> None:
        """
        关闭临时存储
        - 停止定时清理任务
        在FastAPI lifespan shutdown阶段调用
        """
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self._initialized = False
        logger.info("音频临时存储已关闭")

    # ==================== 核心方法 ====================

    async def save(
        self,
        event_id: str,
        behavior_event_id: str,
        base64_audio: str,
    ) -> str:
        """
        保存异常音频片段到临时文件
        将base64编码的音频数据解码后保存为WAV文件

        Args:
            event_id: 录音上传的事件ID（幂等键），如 AI_RECORDING_20260508103200_005678
            behavior_event_id: 关联的行为事件ID，如 AI_BEHAVIOR_20260508103100_001234
            base64_audio: base64编码的WAV音频数据（从算力节点返回）

        Returns:
            保存的文件绝对路径

        Raises:
            ValueError: base64解码失败
            IOError: 文件写入失败
        """
        if not base64_audio:
            raise ValueError("音频数据为空，无法保存")

        try:
            # 解码base64
            audio_bytes = base64.b64decode(base64_audio)
        except Exception as e:
            logger.error(f"音频base64解码失败 | event_id={event_id} | 错误={e}")
            raise ValueError(f"音频base64解码失败: {e}")

        # 构建文件名：{event_id}.wav
        # 使用recording的eventId作为文件名，与v3文档录音上传的eventId对齐
        filename = f"{event_id}.wav"
        file_path = os.path.join(AUDIO_TEMP_DIR, filename)

        # 异步写入文件（避免阻塞事件循环）
        try:
            await asyncio.to_thread(self._write_file_sync, file_path, audio_bytes)
        except Exception as e:
            logger.error(f"音频临时文件写入失败 | event_id={event_id} | 路径={file_path} | 错误={e}")
            raise IOError(f"音频文件写入失败: {e}")

        file_size_kb = len(audio_bytes) / 1024
        logger.info(
            f"异常音频片段已保存 | event_id={event_id} | "
            f"behavior_event_id={behavior_event_id} | "
            f"文件={filename} | 大小={file_size_kb:.1f}KB"
        )
        return file_path

    async def read(self, file_path: str) -> bytes:
        """
        读取临时音频文件内容
        用于上传到后端时读取文件数据

        Args:
            file_path: 文件路径（由save方法返回）

        Returns:
            文件二进制内容

        Raises:
            FileNotFoundError: 文件不存在（可能已被清理）
        """
        if not os.path.exists(file_path):
            logger.warning(f"音频临时文件不存在 | 路径={file_path}")
            raise FileNotFoundError(f"音频临时文件不存在: {file_path}")

        content = await asyncio.to_thread(self._read_file_sync, file_path)
        logger.debug(f"读取音频临时文件 | 路径={file_path} | 大小={len(content) / 1024:.1f}KB")
        return content

    async def delete(self, file_path: str) -> None:
        """
        删除临时音频文件
        上传到后端成功后调用，释放存储空间

        Args:
            file_path: 文件路径
        """
        try:
            await asyncio.to_thread(self._delete_file_sync, file_path)
            logger.debug(f"音频临时文件已删除 | 路径={file_path}")
        except FileNotFoundError:
            # 文件已被定时清理删除，忽略
            pass
        except Exception as e:
            logger.warning(f"音频临时文件删除失败 | 路径={file_path} | 错误={e}")

    async def exists(self, file_path: str) -> bool:
        """
        检查临时音频文件是否存在

        Args:
            file_path: 文件路径

        Returns:
            文件是否存在
        """
        return os.path.exists(file_path)

    # ==================== 同步文件操作（在子线程中执行） ====================

    @staticmethod
    def _write_file_sync(file_path: str, data: bytes) -> None:
        """同步写入文件（在子线程中执行，避免阻塞事件循环）"""
        with open(file_path, "wb") as f:
            f.write(data)

    @staticmethod
    def _read_file_sync(file_path: str) -> bytes:
        """同步读取文件（在子线程中执行）"""
        with open(file_path, "rb") as f:
            return f.read()

    @staticmethod
    def _delete_file_sync(file_path: str) -> None:
        """同步删除文件（在子线程中执行）"""
        os.remove(file_path)

    # ==================== 定时清理机制 ====================

    def _start_cleanup_task(self) -> None:
        """启动定时清理任务"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        """
        定时清理主循环
        每AUDIO_TEMP_CLEANUP_INTERVAL秒扫描临时目录，
        删除超过AUDIO_TEMP_FILE_TTL秒的过期文件
        """
        while True:
            try:
                await asyncio.sleep(AUDIO_TEMP_CLEANUP_INTERVAL)
                await self._cleanup_expired_files()
            except asyncio.CancelledError:
                # 任务被取消（shutdown），正常退出
                break
            except Exception as e:
                # 防止清理循环因意外异常退出
                logger.exception(f"音频临时文件清理循环异常 | 错误={e}")

    async def _cleanup_expired_files(self) -> None:
        """
        清理过期的临时音频文件
        - 扫描临时目录下所有.wav文件
        - 文件修改时间超过AUDIO_TEMP_FILE_TTL秒的删除
        - 记录清理数量和释放空间
        """
        if not os.path.exists(AUDIO_TEMP_DIR):
            return

        now = time.time()
        deleted_count = 0
        freed_bytes = 0

        try:
            # 在子线程中执行文件扫描，避免阻塞事件循环
            result = await asyncio.to_thread(self._scan_and_delete_sync, now)
            deleted_count, freed_bytes = result
        except Exception as e:
            logger.error(f"音频临时文件清理异常 | 错误={e}")
            return

        if deleted_count > 0:
            freed_kb = freed_bytes / 1024
            logger.info(
                f"音频临时文件清理完成 | 删除={deleted_count}个 | "
                f"释放空间={freed_kb:.1f}KB"
            )
        else:
            logger.debug("音频临时文件清理完成 | 无过期文件")

    def _scan_and_delete_sync(self, now: float) -> tuple:
        """
        同步扫描并删除过期文件（在子线程中执行）

        Args:
            now: 当前时间戳

        Returns:
            (删除文件数, 释放字节数)
        """
        deleted_count = 0
        freed_bytes = 0

        for filename in os.listdir(AUDIO_TEMP_DIR):
            if not filename.endswith(".wav"):
                continue

            file_path = os.path.join(AUDIO_TEMP_DIR, filename)
            try:
                # 获取文件修改时间
                mtime = os.path.getmtime(file_path)
                age = now - mtime

                if age > AUDIO_TEMP_FILE_TTL:
                    # 文件已过期，删除
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    deleted_count += 1
                    freed_bytes += file_size
                    logger.debug(
                        f"删除过期音频文件 | 文件={filename} | "
                        f"存活时间={age:.0f}s | 大小={file_size / 1024:.1f}KB"
                    )
            except Exception as e:
                logger.warning(f"清理音频文件失败 | 文件={filename} | 错误={e}")

        return (deleted_count, freed_bytes)

    # ==================== 状态查询 ====================

    async def get_stats(self) -> dict:
        """
        获取临时存储的统计信息
        用于健康检查和管理接口

        Returns:
            统计信息字典
        """
        if not os.path.exists(AUDIO_TEMP_DIR):
            return {
                "temp_dir": AUDIO_TEMP_DIR,
                "file_count": 0,
                "total_size_kb": 0.0,
            }

        file_count = 0
        total_size = 0

        for filename in os.listdir(AUDIO_TEMP_DIR):
            if filename.endswith(".wav"):
                file_path = os.path.join(AUDIO_TEMP_DIR, filename)
                try:
                    total_size += os.path.getsize(file_path)
                    file_count += 1
                except Exception:
                    pass

        return {
            "temp_dir": AUDIO_TEMP_DIR,
            "file_count": file_count,
            "total_size_kb": round(total_size / 1024, 1),
            "ttl_seconds": AUDIO_TEMP_FILE_TTL,
            "cleanup_interval_seconds": AUDIO_TEMP_CLEANUP_INTERVAL,
        }
