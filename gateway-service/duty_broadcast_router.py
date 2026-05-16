"""
智能胸牌服务管理系统 - 值班播报路由
核心功能：
1. WebSocket端点：胸牌硬件通过 WS /ws/device/{deviceNo} 连接主网关
2. POST接口：后端调用 POST /badge/v1/algorithm/duty-broadcasts/tts 触发播报
3. 入参校验：deviceNo非空且长度≤20，broadcastContent非空且长度≤200字
4. 联动逻辑：入参校验 → TTS可用性检查 → 设备在线检查 → 流式合成 → 流式推送

接口规范（对齐v3.1文档6.2节）：
- 接口路径：POST /badge/v1/algorithm/duty-broadcasts/tts
- Content-Type：application/json
- 入参：{"deviceNo": "BADGE0001", "broadcastContent": "播报内容"}
- 出参：{"code": 200, "msg": "ok", "data": {"success": true}, "request_id": "..."}

WebSocket端点：
- 路径：WS /ws/device/{deviceNo}
- 协议：ping/pong心跳 + 二进制PCM音频帧 + 文本控制帧

处理流程（严格按顺序）：
    ① 入参校验：deviceNo非空且长度≤20，broadcastContent非空且长度≤200字
    ② 检查TTS服务可用性：不可用返回success=false
    ③ 检查设备在线状态：不在线返回success=false
    ④ 调用PiperTTSManager.synthesize_stream获取本地PCM音频流
    ⑤ 调用WebSocketDeviceManager.push_audio_stream流式推送给对应设备
    ⑥ 推送完成返回success=true；任何步骤失败返回success=false

使用示例：
    # WebSocket连接测试
    # ws://网关IP:8090/ws/device/BADGE0001

    # POST播报请求
    # curl -X POST http://网关IP:8090/badge/v1/algorithm/duty-broadcasts/tts \
    #   -H "Content-Type: application/json" \
    #   -d '{"deviceNo":"BADGE0001","broadcastContent":"播报内容"}'
"""
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger
from pydantic import BaseModel, Field

from config import (
    BROADCAST_DEVICE_NO_MAX_LENGTH,
    BROADCAST_CONTENT_MAX_LENGTH,
)
from piper_tts_manager import PiperTTSManager
from websocket_device_manager import WebSocketDeviceManager


# ==================== Pydantic 请求模型 ====================

class DutyBroadcastRequest(BaseModel):
    """
    值班播报文字转语音请求体模型
    后端 → 主网关，Content-Type: application/json
    严格对齐v3.1文档6.2节后端请求必传字段

    必传字段：
    - deviceNo: 设备编号，非空，长度≤20
    - broadcastContent: 播报内容，非空，长度≤200字
    """
    deviceNo: str = Field(
        ...,
        min_length=1,
        max_length=BROADCAST_DEVICE_NO_MAX_LENGTH,
        description="设备编号",
        examples=["BADGE0001"],
    )
    broadcastContent: str = Field(
        ...,
        min_length=1,
        max_length=BROADCAST_CONTENT_MAX_LENGTH,
        description="播报内容，最多200字",
        examples=["王五需要去做前厅消防通道检查，请及时完成"],
    )


# ==================== 路由器创建 ====================

duty_broadcast_router = APIRouter(
    prefix="/badge/v1/algorithm",
    tags=["值班播报接口"],
)

# WebSocket路由器（v3.2：独立路由器，无前缀，WebSocket端点在根路径 /ws/device/{deviceNo}）
ws_router = APIRouter(
    prefix = "/algorithm/badge",
    tags=["WebSocket设备连接"],
)

# 处理器单例（由 main.py 在 lifespan 中初始化后赋值）
piper_tts_manager = PiperTTSManager()
ws_device_manager = WebSocketDeviceManager()


# ==================== WebSocket端点：胸牌设备连接（v3.2：根路径 /ws/device/{deviceNo}） ====================

@ws_router.websocket("/ws/device/{device_no}")
async def device_websocket(websocket: WebSocket, device_no: str):
    """
    胸牌设备WebSocket连接端点

    连接流程：
    1. 接受WebSocket连接
    2. 注册设备连接到WebSocketDeviceManager
    3. 进入消息循环，监听客户端消息（pong、播报回执等）
    4. 连接断开时自动注销设备

    协议约定：
    - 服务端 → 客户端：
      - 二进制帧：PCM音频裸流（16bit/16000Hz/mono）
      - 文本帧：控制消息（broadcast_start/broadcast_end/broadcast_error）
      - Ping帧：心跳保活
    - 客户端 → 服务端：
      - Pong帧：心跳回复（自动处理）
      - 文本帧：播报回执（预留扩展）

    Args:
        websocket: FastAPI WebSocket连接对象
        device_no: 设备编号，从URL路径提取
    """
    # 步骤1：接受连接
    await websocket.accept()
    logger.info(f"设备WebSocket连接请求 | deviceNo={device_no}")

    # 步骤2：注册设备
    await ws_device_manager.register_device(device_no, websocket)

    try:
        # 步骤3：消息循环
        while True:
            try:
                # 接收客户端消息
                data = await websocket.receive()

                # 处理不同类型的消息
                if "text" in data:
                    # 文本消息：播报回执等
                    text = data["text"]
                    logger.debug(
                        f"设备文本消息 | deviceNo={device_no} | "
                        f"内容={text[:200]}"
                    )
                    # 【预留扩展位】播报回执处理
                    # TODO: 与硬件确认回执格式后实现
                    # 示例回执格式：{"type":"broadcast_ack","success":true}
                    _handle_device_message(device_no, text)

                elif "bytes" in data:
                    # 二进制消息：暂不处理
                    logger.debug(
                        f"设备二进制消息 | deviceNo={device_no} | "
                        f"长度={len(data['bytes'])}"
                    )

                elif data.get("type") == "websocket.disconnect":
                    logger.info(
                        f"设备主动断开 | deviceNo={device_no}"
                    )
                    break

            except WebSocketDisconnect:
                logger.info(
                    f"设备WebSocket断开 | deviceNo={device_no}"
                )
                break

    finally:
        # 步骤4：注销设备
        await ws_device_manager.unregister_device(device_no)


def _handle_device_message(device_no: str, text: str) -> None:
    """
    处理设备发送的文本消息

    消息类型：
    - {"type":"pong"}：心跳回复（更新设备last_pong_time）
    - {"type":"broadcast_ack"}：播报回执（预留扩展）
    - 其他：记录日志

    Args:
        device_no: 设备编号
        text: 消息文本
    """
    try:
        import json
        msg = json.loads(text)
        msg_type = msg.get("type", "unknown")

        if msg_type == "pong":
            # 心跳回复，更新last_pong_time
            ws_device_manager.handle_pong(device_no)
            logger.debug(f"设备pong回复 | deviceNo={device_no}")
            return

        logger.info(
            f"设备消息处理 | deviceNo={device_no} | "
            f"type={msg_type} | 原始={text[:200]}"
        )

        # 【预留扩展位】播报回执处理
        # 需要和硬件确认：
        # 1. 播报成功是否需要回执？
        # 2. 播报失败回执格式？
        # 3. 播报超时判定逻辑？

    except Exception as e:
        logger.warning(
            f"设备消息解析失败 | deviceNo={device_no} | "
            f"内容={text[:100]} | 错误={str(e)[:100]}"
        )


# ==================== POST接口：后端调用播报 ====================

@duty_broadcast_router.post("/duty-broadcasts/tts")
async def duty_broadcast_tts(request: DutyBroadcastRequest):
    """
    值班播报文字转语音

    完整流程：
    ① 入参校验：deviceNo非空且长度≤20，broadcastContent非空且长度≤200字
    ② 检查TTS服务可用性：不可用返回success=false
    ③ 检查设备在线状态：不在线返回success=false
    ④ 调用PiperTTSManager.synthesize_stream获取本地PCM音频流
    ⑤ 调用WebSocketDeviceManager.push_audio_stream流式推送给对应设备
    ⑥ 推送完成返回success=true；任何步骤失败返回success=false

    请求体（对齐v3.1文档6.2节）：
    {
        "deviceNo": "BADGE0001",
        "broadcastContent": "王五需要去做前厅消防通道检查，请及时完成"
    }

    成功响应：
    {
        "code": 200,
        "msg": "ok",
        "data": {"success": true},
        "request_id": "..."
    }

    失败响应（TTS不可用/设备离线/合成失败/推送失败）：
    {
        "code": 200,
        "msg": "ok",
        "data": {"success": false, "reason": "失败原因"},
        "request_id": "..."
    }
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    device_no = request.deviceNo
    broadcast_content = request.broadcastContent

    logger.info(
        f"收到值班播报请求 | request_id={request_id} | "
        f"deviceNo={device_no} | "
        f"播报内容长度={len(broadcast_content)} | "
        f"内容={broadcast_content[:50]}"
    )

    # ========== ② 检查TTS服务可用性 ==========
    if not piper_tts_manager.is_available():
        error_msg = f"TTS服务不可用 | 错误={piper_tts_manager.load_error}"
        logger.error(
            f"值班播报失败 | request_id={request_id} | "
            f"deviceNo={device_no} | {error_msg}"
        )
        return _build_broadcast_response(
            request_id=request_id,
            success=False,
            reason="TTS服务不可用",
        )

    # ========== ③ 检查设备在线状态 ==========
    if not ws_device_manager.is_device_online(device_no):
        logger.warning(
            f"值班播报失败 | 设备不在线 | request_id={request_id} | "
            f"deviceNo={device_no}"
        )
        return _build_broadcast_response(
            request_id=request_id,
            success=False,
            reason="设备不在线",
        )

    # ========== ④⑤ 流式合成 + 流式推送 ==========
    try:
        # 获取本地Piper TTS音频流（异步生成器）
        audio_stream = piper_tts_manager.synthesize_stream(broadcast_content)

        # 流式推送到设备
        push_success = await ws_device_manager.push_audio_stream(
            device_no=device_no,
            audio_stream=audio_stream,
            broadcast_content=broadcast_content,
        )

        # ========== ⑥ 返回结果 ==========
        elapsed_ms = int((time.time() - start_time) * 1000)

        if push_success:
            logger.info(
                f"值班播报完成 | request_id={request_id} | "
                f"deviceNo={device_no} | 耗时={elapsed_ms}ms"
            )
            return _build_broadcast_response(
                request_id=request_id,
                success=True,
            )
        else:
            logger.warning(
                f"值班播报推送失败 | request_id={request_id} | "
                f"deviceNo={device_no} | 耗时={elapsed_ms}ms"
            )
            return _build_broadcast_response(
                request_id=request_id,
                success=False,
                reason="音频推送失败",
            )

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.error(
            f"值班播报异常 | request_id={request_id} | "
            f"deviceNo={device_no} | "
            f"异常类型={type(e).__name__} | "
            f"错误={str(e)[:200]} | 耗时={elapsed_ms}ms"
        )
        return _build_broadcast_response(
            request_id=request_id,
            success=False,
            reason="播报服务异常",
        )


# ==================== 响应构建 ====================

def _build_broadcast_response(
    request_id: str,
    success: bool,
    reason: str = "",
) -> Dict[str, Any]:
    """
    构建值班播报响应体

    成功：{"code": 200, "msg": "ok", "data": {"success": true}, "request_id": "..."}
    失败：{"code": 200, "msg": "ok", "data": {"success": false, "reason": "..."}, "request_id": "..."}

    注意：即使播报失败，HTTP状态码仍为200（业务成功但播报失败），
    通过data.success区分成功/失败，对齐v3.1文档6.2节算法返回规范

    Args:
        request_id: 请求ID
        success: 是否成功
        reason: 失败原因（仅success=False时有值）

    Returns:
        统一格式的响应字典
    """
    data: Dict[str, Any] = {"success": success}
    if not success and reason:
        data["reason"] = reason

    return {
        "code": 200,
        "msg": "ok",
        "data": data,
        "request_id": request_id,
    }
