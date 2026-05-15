"""
智能胸牌服务管理系统 - 原始异常语音临时文件管理器
核心功能：
1. 管理硬件上传的原始WAV音频临时文件（与异常音频片段分开存储）
2. 临时文件命名规则：{uploadId}_{deviceNo}.wav
3. 自动创建不存在的临时目录
4. 定时清理：每小时清理一次1小时前的临时文件，避免占满树莓派存储
5. 提供文件保存、删除、存在性检查等操作

设计约束：
- 单例模式，全局统一管理
- 全异步实现，文件IO使用asyncio.to_thread避免阻塞事件循环
- 适配树莓派存储资源，定期清理过期文件
- 所有配置项从config.py读取，禁止硬编码

与utils/AudioTempStorage的区别：
- AudioTempStorage：管理异常音频片段（base64→WAV），命名{eventId}.wav
- AudioTempManager：管理原始上传音频（直接二进制），命名{uploadId}_{deviceNo}.wav
- 两者独立目录、独立清理，互不影响

使用示例：
    from audio_temp_manager import AudioTempManager

    # 初始化（在FastAPI lifespan startup中调用）
    manager = AudioTempManager()
    await manager.initialize()

    # 保存原始音频文件
    file_path = await manager.save(
        upload_id="upload_001",
        device_no="BADGE0001",
        audio_bytes=b"RIFF...",
    )
    # file_path = "temp/raw_uploads/upload_001_BADGE0001.wav"

    # 删除临时文件（转发成功后调用）
    await manager.delete(file_path)

    # 关闭（在FastAPI lifespan shutdown中调用）
    await manager.close()
"""
import os
import asyncio
import time
from typing import Optional

from loguru import logger

from config import (
    RAW_AUDIO_TEMP_DIR,
    RAW_AUDIO_TEMP_FILE_TTL,
    RAW_AUDIO_CLEANUP_INTERVAL,
)


class AudioTempManager:
    """
    原始异常语音临时文件管理器 - 单例模式

    职责：
    - 管理硬件上传的原始WAV音频文件
    - 命名规则：{uploadId}_{deviceNo}.wav
    - 定时清理过期临时文件
    - 适配树莓派存储，避免占满空间
    """

    _instance: Optional["AudioTempManager"] = None

    def __new__(cls, *args, **kwargs):
        """单例模式：确保全局只有一个AudioTempManager实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化（仅首次创建时执行）"""
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._cleanup_task: Optional[asyncio.Task] = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        初始化临时文件管理器
        - 创建临时目录
        - 启动定时清理任务
        在FastAPI lifespan startup阶段调用
        """
        if self._initialized:
            return

        # 创建临时目录（含子目录，自动创建不存在的父目录）
        os.makedirs(RAW_AUDIO_TEMP_DIR, exist_ok=True)
        logger.info(f"原始音频临时目录就绪 | 路径={RAW_AUDIO_TEMP_DIR}")

        # 启动定时清理任务
        self._start_cleanup_task()
        self._initialized = True
        logger.info(
            f"原始音频临时文件管理器初始化完成 | "
            f"文件过期={RAW_AUDIO_TEMP_FILE_TTL}s | "
            f"清理间隔={RAW_AUDIO_CLEANUP_INTERVAL}s"
        )

    async def close(self) -> None:
        """
        关闭临时文件管理器
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
        logger.info("原始音频临时文件管理器已关闭")

    # ==================== 核心方法 ====================

    async def save(
        self,
        upload_id: str,
        device_no: str,
        audio_bytes: bytes,
    ) -> str:
        """
        保存原始音频文件到临时目录

        临时文件命名规则：{uploadId}_{deviceNo}.wav
        例如：upload_001_BADGE0001.wav

        Args:
            upload_id: 上传唯一ID（来自硬件metadata的uploadId）
            device_no: 设备编号（来自硬件metadata的deviceNo）
            audio_bytes: 原始音频二进制数据

        Returns:
            保存的文件绝对路径

        Raises:
            IOError: 文件写入失败
        """
        # 构建文件名：{uploadId}_{deviceNo}.wav
        filename = f"{upload_id}_{device_no}.wav"
        file_path = os.path.join(RAW_AUDIO_TEMP_DIR, filename)

        # 异步写入文件（避免阻塞事件循环）
        try:
            await asyncio.to_thread(self._write_file_sync, file_path, audio_bytes)
        except Exception as e:
            logger.error(
                f"原始音频临时文件写入失败 | uploadId={upload_id} | "
                f"deviceNo={device_no} | 路径={file_path} | 错误={e}"
            )
            raise IOError(f"音频文件写入失败: {e}")

        file_size_kb = len(audio_bytes) / 1024
        logger.info(
            f"原始音频已保存 | uploadId={upload_id} | "
            f"deviceNo={device_no} | 文件={filename} | "
            f"大小={file_size_kb:.1f}KB"
        )
        return file_path

    async def read(self, file_path: str) -> bytes:
        """
        读取临时音频文件内容

        Args:
            file_path: 文件路径（由save方法返回）

        Returns:
            文件二进制内容

        Raises:
            FileNotFoundError: 文件不存在（可能已被清理）
        """
        if not os.path.exists(file_path):
            logger.warning(f"原始音频临时文件不存在 | 路径={file_path}")
            raise FileNotFoundError(f"音频临时文件不存在: {file_path}")

        content = await asyncio.to_thread(self._read_file_sync, file_path)
        logger.debug(
            f"读取原始音频临时文件 | 路径={file_path} | "
            f"大小={len(content) / 1024:.1f}KB"
        )
        return content

    async def delete(self, file_path: str) -> None:
        """
        删除临时音频文件
        转发成功后调用，释放存储空间

        Args:
            file_path: 文件路径
        """
        try:
            await asyncio.to_thread(self._delete_file_sync, file_path)
            logger.debug(f"原始音频临时文件已删除 | 路径={file_path}")
        except FileNotFoundError:
            # 文件已被定时清理删除，忽略
            pass
        except Exception as e:
            logger.warning(
                f"原始音频临时文件删除失败 | 路径={file_path} | 错误={e}"
            )

    async def exists(self, file_path: str) -> bool:
        """
        检查临时音频文件是否存在

        Args:
            file_path: 文件路径

        Returns:
            文件是否存在
        """
        return os.path.exists(file_path)

    def get_file_path(self, upload_id: str, device_no: str) -> str:
        """
        根据uploadId和deviceNo构建文件路径

        Args:
            upload_id: 上传ID
            device_no: 设备编号

        Returns:
            文件绝对路径
        """
        filename = f"{upload_id}_{device_no}.wav"
        return os.path.join(RAW_AUDIO_TEMP_DIR, filename)

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
        每RAW_AUDIO_CLEANUP_INTERVAL秒扫描临时目录，
        删除超过RAW_AUDIO_TEMP_FILE_TTL秒的过期文件
        """
        while True:
            try:
                await asyncio.sleep(RAW_AUDIO_CLEANUP_INTERVAL)
                await self._cleanup_expired_files()
            except asyncio.CancelledError:
                # 任务被取消（shutdown），正常退出
                break
            except Exception as e:
                # 防止清理循环因意外异常退出
                logger.exception(f"原始音频临时文件清理循环异常 | 错误={e}")

    async def _cleanup_expired_files(self) -> None:
        """
        清理过期的临时音频文件
        - 扫描临时目录下所有.wav文件
        - 文件修改时间超过RAW_AUDIO_TEMP_FILE_TTL秒的删除
        - 记录清理数量和释放空间
        """
        if not os.path.exists(RAW_AUDIO_TEMP_DIR):
            return

        now = time.time()

        try:
            # 在子线程中执行文件扫描，避免阻塞事件循环
            result = await asyncio.to_thread(self._scan_and_delete_sync, now)
            deleted_count, freed_bytes = result
        except Exception as e:
            logger.error(f"原始音频临时文件清理异常 | 错误={e}")
            return

        if deleted_count > 0:
            freed_kb = freed_bytes / 1024
            logger.info(
                f"原始音频临时文件清理完成 | 删除={deleted_count}个 | "
                f"释放空间={freed_kb:.1f}KB"
            )
        else:
            logger.debug("原始音频临时文件清理完成 | 无过期文件")

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

        for filename in os.listdir(RAW_AUDIO_TEMP_DIR):
            if not filename.endswith(".wav"):
                continue

            file_path = os.path.join(RAW_AUDIO_TEMP_DIR, filename)
            try:
                # 获取文件修改时间
                mtime = os.path.getmtime(file_path)
                age = now - mtime

                if age > RAW_AUDIO_TEMP_FILE_TTL:
                    # 文件已过期，删除
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    deleted_count += 1
                    freed_bytes += file_size
                    logger.debug(
                        f"删除过期原始音频文件 | 文件={filename} | "
                        f"存活时间={age:.0f}s | 大小={file_size / 1024:.1f}KB"
                    )
            except Exception as e:
                logger.warning(
                    f"清理原始音频文件失败 | 文件={filename} | 错误={e}"
                )

        return (deleted_count, freed_bytes)

    # ==================== 状态查询 ====================

    async def get_stats(self) -> dict:
        """
        获取临时存储的统计信息
        用于健康检查和管理接口

        Returns:
            统计信息字典
        """
        if not os.path.exists(RAW_AUDIO_TEMP_DIR):
            return {
                "temp_dir": RAW_AUDIO_TEMP_DIR,
                "file_count": 0,
                "total_size_kb": 0.0,
            }

        file_count = 0
        total_size = 0

        for filename in os.listdir(RAW_AUDIO_TEMP_DIR):
            if filename.endswith(".wav"):
                file_path = os.path.join(RAW_AUDIO_TEMP_DIR, filename)
                try:
                    total_size += os.path.getsize(file_path)
                    file_count += 1
                except Exception:
                    pass

        return {
            "temp_dir": RAW_AUDIO_TEMP_DIR,
            "file_count": file_count,
            "total_size_kb": round(total_size / 1024, 1),
            "ttl_seconds": RAW_AUDIO_TEMP_FILE_TTL,
            "cleanup_interval_seconds": RAW_AUDIO_CLEANUP_INTERVAL,
        }
