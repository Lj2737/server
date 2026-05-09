"""
智能胸牌服务管理系统 - WebSocket设备连接管理器
核心功能：
1. 胸牌硬件通过WebSocket连接主网关，路径：WS /ws/device/{deviceNo}
2. 维护deviceNo → WebSocket连接的映射，支持根据deviceNo推送音频
3. 心跳保活：每30秒发送ping包，连续3次无pong应答判定离线，自动清理连接
4. 断连自动清理：连接关闭时删除映射关系，释放资源
5. 支持异步流式推送：边合成音频边推送给胸牌，无需等待整段合成完成

WebSocket协议约定：
- 连接路径：ws://网关IP:8090/ws/device/{deviceNo}
- 心跳：服务端每30秒发ping，客户端回pong
- 音频推送：服务端发二进制帧（PCM裸流），客户端直接播放
- 播报开始标记：服务端发文本帧 "{"type":"broadcast_start","content":"播报内容"}"
- 播报结束标记：服务端发文本帧 "{"type":"broadcast_end"}"
- 播报错误标记：服务端发文本帧 "{"type":"broadcast_error","message":"错误信息"}"

使用示例：
    from websocket_device_manager import WebSocketDeviceManager

    # 获取单例
    manager = WebSocketDeviceManager()

    # 启动心跳（在FastAPI lifespan startup阶段调用）
    await manager.start_heartbeat()

    # 注册设备连接（在WebSocket端点中调用）
    await manager.register_device(device_no, websocket)

    # 流式推送音频
    success = await manager.push_audio_stream(device_no, audio_stream)

    # 检查设备在线
    if manager.is_device_online(device_no):
        ...
"""
import asyncio
import json
import time
from typing import AsyncGenerator, Dict, Optional

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from config import (
    WS_HEARTBEAT_INTERVAL,
    WS_HEARTBEAT_FAIL_THRESHOLD,
    WS_PING_TIMEOUT,
)


class _DeviceConnection:
    """
    单个设备连接信息封装
    包含WebSocket连接对象和心跳状态

    Attributes:
        device_no: 设备编号
        websocket: WebSocket连接对象
        connected_at: 连接建立时间戳
        last_pong_time: 最近一次收到pong的时间戳
        missed_pings: 连续未回复ping的次数
    """

    def __init__(self, device_no: str, websocket: WebSocket):
        self.device_no = device_no
        self.websocket = websocket
        self.connected_at = time.time()
        self.last_pong_time = time.time()
        self.missed_pings = 0

    def record_pong(self) -> None:
        """记录收到pong，重置missed_pings计数"""
        self.last_pong_time = time.time()
        self.missed_pings = 0

    def increment_missed_ping(self) -> int:
        """增加未回复ping计数，返回当前计数"""
        self.missed_pings += 1
        return self.missed_pings


class WebSocketDeviceManager:
    """
    WebSocket设备连接管理器 - 单例模式

    职责：
    - 维护deviceNo → WebSocket连接映射
    - 设备连接注册与注销
    - 异步流式推送PCM音频到指定设备
    - 心跳保活定时任务
    - 断连自动清理

    设计约束：
    - 单例模式，全局唯一实例
    - 全异步实现，不阻塞事件循环
    - 单设备单连接：同一deviceNo新连接会踢掉旧连接
    - 线程安全：FastAPI事件循环单线程，无需加锁
    """

    _instance: Optional["WebSocketDeviceManager"] = None

    def __new__(cls, *args, **kwargs):
        """单例模式：确保全局只有一个WebSocketDeviceManager实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化（仅首次创建时执行）"""
        if hasattr(self, "_initialized") and self._initialized:
            return
        # deviceNo → _DeviceConnection 映射
        self._connections: Dict[str, _DeviceConnection] = {}
        # 心跳定时任务句柄
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._initialized = False

    def initialize(self) -> None:
        """
        初始化WebSocket设备管理器
        在FastAPI lifespan startup阶段调用
        """
        if self._initialized:
            return
        self._initialized = True
        logger.info(
            f"WebSocket设备管理器初始化完成 | "
            f"心跳间隔={WS_HEARTBEAT_INTERVAL}s | "
            f"心跳失败阈值={WS_HEARTBEAT_FAIL_THRESHOLD}"
        )

    # ==================== 设备注册/注销 ====================

    async def register_device(self, device_no: str, websocket: WebSocket) -> None:
        """
        注册设备WebSocket连接

        处理逻辑：
        1. 如果该deviceNo已有连接，先关闭旧连接（踢掉旧设备）
        2. 创建新的_DeviceConnection并加入映射
        3. 记录注册日志

        Args:
            device_no: 设备编号
            websocket: WebSocket连接对象
        """
        # 踢掉旧连接：同一设备只保留最新连接
        if device_no in self._connections:
            old_conn = self._connections[device_no]
            logger.warning(
                f"设备重复连接，踢掉旧连接 | deviceNo={device_no} | "
                f"旧连接时长={time.time() - old_conn.connected_at:.0f}s"
            )
            try:
                await old_conn.websocket.close(
                    code=4000,
                    reason="设备新连接建立，旧连接被踢掉"
                )
            except Exception:
                pass  # 旧连接可能已断开，忽略关闭异常

        # 注册新连接
        conn = _DeviceConnection(device_no, websocket)
        self._connections[device_no] = conn

        logger.info(
            f"设备WebSocket注册成功 | deviceNo={device_no} | "
            f"当前在线设备数={len(self._connections)}"
        )

    async def unregister_device(self, device_no: str) -> None:
        """
        注销设备WebSocket连接
        连接关闭时调用，删除映射关系

        Args:
            device_no: 设备编号
        """
        if device_no in self._connections:
            del self._connections[device_no]
            logger.info(
                f"设备WebSocket注销 | deviceNo={device_no} | "
                f"当前在线设备数={len(self._connections)}"
            )

    # ==================== 音频流式推送 ====================

    async def push_audio_stream(
        self,
        device_no: str,
        audio_stream: AsyncGenerator[bytes, None],
        broadcast_content: str = "",
    ) -> bool:
        """
        异步流式推送PCM音频到指定设备
        边合成边推送，无需等待整段合成完成

        推送协议：
        1. 先发文本帧：{"type":"broadcast_start","content":"播报内容"}
        2. 逐chunk发二进制帧：PCM裸流（16bit/16000Hz/mono）
        3. 最后发文本帧：{"type":"broadcast_end"}
        4. 任何步骤失败发文本帧：{"type":"broadcast_error","message":"错误信息"}

        Args:
            device_no: 设备编号
            audio_stream: PCM音频异步生成器（由PiperTTSManager.synthesize_stream返回）
            broadcast_content: 播报内容文本（用于通知设备播报内容，可选）

        Returns:
            True: 推送成功
            False: 推送失败（设备不在线或推送过程中出错）
        """
        if device_no not in self._connections:
            logger.warning(
                f"音频推送失败 | 设备不在线 | deviceNo={device_no}"
            )
            return False

        conn = self._connections[device_no]
        start_time = time.time()
        total_bytes = 0
        chunk_count = 0

        try:
            # 步骤1：发送播报开始标记
            start_msg = json.dumps(
                {
                    "type": "broadcast_start",
                    "content": broadcast_content,
                },
                ensure_ascii=False,
            )
            await conn.websocket.send_text(start_msg)
            logger.debug(
                f"播报开始标记已发送 | deviceNo={device_no} | "
                f"内容={broadcast_content[:50]}"
            )

            # 步骤2：逐chunk推送PCM音频
            async for pcm_chunk in audio_stream:
                await conn.websocket.send_bytes(pcm_chunk)
                total_bytes += len(pcm_chunk)
                chunk_count += 1

            # 步骤3：发送播报结束标记
            end_msg = json.dumps({"type": "broadcast_end"}, ensure_ascii=False)
            await conn.websocket.send_text(end_msg)

            elapsed_ms = int((time.time() - start_time) * 1000)
            # PCM时长估算：字节数 / (2字节/采样 * 16000采样/秒)
            audio_duration = total_bytes / (2 * 16000) if total_bytes > 0 else 0

            logger.info(
                f"音频流式推送完成 | deviceNo={device_no} | "
                f"chunks={chunk_count} | 总字节={total_bytes} | "
                f"音频时长={audio_duration:.2f}s | 耗时={elapsed_ms}ms"
            )
            return True

        except WebSocketDisconnect:
            logger.warning(
                f"音频推送中断 | 设备断开连接 | deviceNo={device_no}"
            )
            await self.unregister_device(device_no)
            return False

        except Exception as e:
            logger.error(
                f"音频推送异常 | deviceNo={device_no} | "
                f"已推送chunks={chunk_count} | 已推送字节={total_bytes} | "
                f"错误类型={type(e).__name__} | 错误={str(e)[:200]}"
            )
            # 尝试发送错误标记
            try:
                error_msg = json.dumps(
                    {
                        "type": "broadcast_error",
                        "message": "音频推送异常，播报中断",
                    },
                    ensure_ascii=False,
                )
                await conn.websocket.send_text(error_msg)
            except Exception:
                pass  # 发送错误标记失败，忽略
            return False

    # ==================== 设备在线查询 ====================

    def is_device_online(self, device_no: str) -> bool:
        """
        检查设备是否在线

        Args:
            device_no: 设备编号

        Returns:
            True: 设备在线
            False: 设备不在线
        """
        return device_no in self._connections

    def get_online_devices(self) -> list:
        """
        获取所有在线设备编号列表

        Returns:
            在线设备编号列表
        """
        return list(self._connections.keys())

    def get_online_count(self) -> int:
        """
        获取在线设备数量

        Returns:
            在线设备数量
        """
        return len(self._connections)

    # ==================== 心跳保活 ====================

    async def start_heartbeat(self) -> None:
        """
        启动心跳保活定时任务
        在FastAPI lifespan startup阶段调用

        心跳逻辑：
        1. 每WS_HEARTBEAT_INTERVAL秒（默认30秒）遍历所有设备连接
        2. 向每个连接发送ping帧
        3. 等待WS_PING_TIMEOUT秒（默认5秒）后检查pong回复
        4. 连续WS_HEARTBEAT_FAIL_THRESHOLD次（默认3次）未回复判定离线
        5. 离线设备自动清理连接
        """
        if self._heartbeat_task is not None:
            logger.warning("心跳定时任务已在运行，跳过重复启动")
            return

        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(),
            name="ws-device-heartbeat",
        )
        logger.info(
            f"WebSocket设备心跳任务已启动 | "
            f"间隔={WS_HEARTBEAT_INTERVAL}s | "
            f"超时={WS_PING_TIMEOUT}s | "
            f"失败阈值={WS_HEARTBEAT_FAIL_THRESHOLD}"
        )

    async def stop_heartbeat(self) -> None:
        """
        停止心跳保活定时任务
        在FastAPI lifespan shutdown阶段调用
        """
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
            logger.info("WebSocket设备心跳任务已停止")

    async def _heartbeat_loop(self) -> None:
        """
        心跳保活循环（后台任务）

        每轮循环：
        1. 遍历所有设备连接
        2. 发送心跳文本消息（FastAPI WebSocket不支持send_ping，用文本消息代替）
        3. 检查上次收到消息的时间
        4. 超时未收到消息的累加missed_pings计数
        5. 超过阈值的设备判定离线并清理
        """
        while True:
            try:
                await asyncio.sleep(WS_HEARTBEAT_INTERVAL)

                if not self._connections:
                    continue

                offline_devices = []

                for device_no, conn in list(self._connections.items()):
                    try:
                        # FastAPI WebSocket不支持send_ping()方法
                        # 改用文本消息作为心跳：发送 {"type":"ping"}
                        # 客户端应回复 {"type":"pong"}
                        ping_msg = json.dumps({"type": "ping"}, ensure_ascii=False)
                        await conn.websocket.send_text(ping_msg)
                        logger.debug(
                            f"心跳ping已发送 | deviceNo={device_no}"
                        )

                        # 检查距上次收到消息的时间
                        time_since_last = time.time() - conn.last_pong_time

                        if time_since_last > WS_HEARTBEAT_INTERVAL * 2:
                            missed = conn.increment_missed_ping()
                            logger.debug(
                                f"设备心跳未回复 | deviceNo={device_no} | "
                                f"连续未回复={missed} | "
                                f"距上次消息={time_since_last:.1f}s"
                            )
                            if missed >= WS_HEARTBEAT_FAIL_THRESHOLD:
                                offline_devices.append(device_no)
                        else:
                            # 设备近期有消息，重置计数
                            if conn.missed_pings > 0:
                                conn.record_pong()

                    except Exception as e:
                        logger.warning(
                            f"设备心跳发送失败 | deviceNo={device_no} | "
                            f"错误={str(e)[:100]}"
                        )
                        missed = conn.increment_missed_ping()
                        if missed >= WS_HEARTBEAT_FAIL_THRESHOLD:
                            offline_devices.append(device_no)

                # 清理离线设备
                for device_no in offline_devices:
                    logger.warning(
                        f"设备心跳超时离线 | deviceNo={device_no} | "
                        f"连续未回复={WS_HEARTBEAT_FAIL_THRESHOLD}"
                    )
                    await self.unregister_device(device_no)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"心跳循环异常 | 错误类型={type(e).__name__} | "
                    f"错误={str(e)[:200]}"
                )

    def handle_pong(self, device_no: str) -> None:
        """
        处理设备pong回复
        在WebSocket端点收到pong帧时调用

        Args:
            device_no: 设备编号
        """
        if device_no in self._connections:
            self._connections[device_no].record_pong()
            logger.debug(f"设备pong回复 | deviceNo={device_no}")

    # ==================== 资源清理 ====================

    async def close_all(self) -> None:
        """
        关闭所有设备连接
        在服务关闭时调用
        """
        device_list = list(self._connections.keys())
        for device_no in device_list:
            try:
                conn = self._connections[device_no]
                await conn.websocket.close(
                    code=1001,
                    reason="服务关闭，连接断开"
                )
            except Exception:
                pass

        self._connections.clear()
        logger.info(
            f"所有设备连接已关闭 | 清理数量={len(device_list)}"
        )
