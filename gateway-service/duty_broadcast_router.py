"""
智能胸牌服务管理系统 - 值班播报路由
核心功能：
1. WebSocket端点：胸牌硬件通过 WS /badge/v1/algorithm/ws/device/{deviceNo} 连接主网关
2. POST接口：后端调用 POST /badge/v1/algorithm/duty-broadcasts/tts 触发播报
3. 入参校验：deviceNo非空且长度≤20，broadcastContent非空且长度≤200字
4. 联动逻辑：入参校验 → TTS可用性检查 → 设备在线检查 → 非流式合成 → 分块推送

接口规范（对齐v3.1文档6.2节）：
- 接口路径：POST /badge/v1/algorithm/duty-broadcasts/tts
- Content-Type：application/json
- 入参：{"deviceNo": "BADGE0001", "broadcastContent": "播报内容"}
- 出参：{"code": 200, "msg": "ok", "data": {"success": true}, "request_id": "..."}

WebSocket端点：
- 路径：WS /badge/v1/algorithm/ws/device/{deviceNo}
- 协议：ping/pong心跳 + 二进制PCM音频帧 + 文本控制帧

处理流程（严格按顺序）：
    ① 入参校验：deviceNo非空且长度≤20，broadcastContent非空且长度≤200字
    ② 检查TTS服务可用性：不可用返回success=false
    ③ 检查设备在线状态：不在线返回success=false
    ④ 调用PiperTTSManager.synthesize_stream获取TTS API音频分块
    ⑤ 调用WebSocketDeviceManager.push_audio_stream流式推送给对应设备
    ⑥ 推送完成返回success=true；任何步骤失败返回success=false

使用示例：
    # WebSocket连接测试
    # ws://网关IP:8090/badge/v1/algorithm/ws/device/BADGE0001

    # POST播报请求
    # curl -X POST http://网关IP:8090/badge/v1/algorithm/duty-broadcasts/tts \
    #   -H "Content-Type: application/json" \
    #   -d '{"deviceNo":"BADGE0001","broadcastContent":"播报内容"}'
"""
import asyncio
import io
import json
import time
import wave
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger
from pydantic import BaseModel, Field

from config import (
    AI_DIALOG_CHANNELS,
    AI_DIALOG_INTERNAL_PATH,
    AI_DIALOG_MAX_AUDIO_SECONDS,
    AI_DIALOG_REQUEST_TIMEOUT,
    AI_DIALOG_SAMPLE_RATE,
    AI_DIALOG_SAMPLE_WIDTH,
    BROADCAST_DEVICE_NO_MAX_LENGTH,
    BROADCAST_CONTENT_MAX_LENGTH,
)
from http_client import HttpClientSingleton
from node_manager import NodeManager
from piper_tts_manager import PiperTTSManager
from utils import BackendClient
from websocket_device_manager import WebSocketDeviceManager


# ==================== Pydantic 请求模型 ====================

class DutyBroadcastRequest(BaseModel):
    """Backend request body for TTS broadcast."""
    deviceNo: str = Field(
        ...,
        min_length=1,
        max_length=BROADCAST_DEVICE_NO_MAX_LENGTH,
        description="Device number",
        examples=["BADGE0001"],
    )
    broadcastContent: str = Field(
        ...,
        min_length=1,
        max_length=BROADCAST_CONTENT_MAX_LENGTH,
        description="Broadcast text, max 200 characters",
        examples=["Please check the front hall fire passage."],
    )


# ==================== 路由器创建 ====================

duty_broadcast_router = APIRouter(
    prefix="/badge/v1/algorithm",
    tags=["值班播报接口"],
)

# WebSocket路由器：统一使用 /badge/v1 前缀
ws_router = APIRouter(
    prefix="/badge/v1/algorithm",
    tags=["WebSocket设备连接"],
)

# 处理器单例（由 main.py 在 lifespan 中初始化后赋值）
piper_tts_manager = PiperTTSManager()
ws_device_manager = WebSocketDeviceManager()
node_manager = NodeManager()
_backend_client: Optional[BackendClient] = None
_dialog_sessions: Dict[str, Dict[str, Any]] = {}
_dialog_tasks: Dict[str, asyncio.Task] = {}
_dialog_task_dialog_ids: Dict[str, str] = {}
_dialog_versions: Dict[str, int] = {}
_dialog_locks: Dict[str, asyncio.Lock] = {}
_DIALOG_INTERRUPT_WAIT_SECONDS = 1.0
_DIALOG_START_DEBOUNCE_SECONDS = 0.35


def _get_dialog_lock(device_no: str) -> asyncio.Lock:
    lock = _dialog_locks.get(device_no)
    if lock is None:
        lock = asyncio.Lock()
        _dialog_locks[device_no] = lock
    return lock


def initialize_ai_dialog(backend_client: BackendClient) -> None:
    """Inject backend client for knowledge-base lookup and dialog completion callback."""
    global _backend_client
    _backend_client = backend_client
    backend_client.set_badge_binding_missing_handler(_broadcast_badge_binding_missing)
    logger.info("AI dialog WebSocket pipeline initialized")


async def _broadcast_badge_binding_missing(
    device_no: str,
    broadcast_content: str,
    source_path: str,
) -> None:
    """Broadcast backend badge-binding-missing errors without blocking the caller."""
    try:
        if not piper_tts_manager.is_available():
            logger.warning(
                f"Badge binding missing broadcast skipped | deviceNo={device_no} | "
                f"path={source_path} | reason=TTS unavailable | error={piper_tts_manager.load_error}"
            )
            return

        if not ws_device_manager.is_device_online(device_no):
            logger.warning(
                f"Badge binding missing broadcast skipped | deviceNo={device_no} | "
                f"path={source_path} | reason=device offline | content={broadcast_content}"
            )
            return

        audio_stream = piper_tts_manager.synthesize_stream(broadcast_content)
        success = await ws_device_manager.push_audio_stream(
            device_no=device_no,
            audio_stream=audio_stream,
            broadcast_content=broadcast_content,
        )
        if not success:
            logger.warning(
                f"Badge binding missing broadcast push failed | deviceNo={device_no} | "
                f"path={source_path}"
            )
            return

        logger.info(
            f"Badge binding missing broadcast completed | deviceNo={device_no} | "
            f"path={source_path}"
        )
    except Exception as exc:
        logger.error(
            f"Badge binding missing broadcast exception | deviceNo={device_no} | "
            f"path={source_path} | errorType={type(exc).__name__} | error={str(exc)[:300]}"
        )


def _next_dialog_version(device_no: str) -> int:
    version = _dialog_versions.get(device_no, 0) + 1
    _dialog_versions[device_no] = version
    return version


def _is_current_dialog(device_no: str, dialog_id: str, version: int) -> bool:
    current_task = asyncio.current_task()
    tracked_task = _dialog_tasks.get(device_no)
    return (
        _dialog_versions.get(device_no) == version
        and _dialog_task_dialog_ids.get(device_no) == dialog_id
        and (current_task is None or tracked_task is current_task)
    )


def _track_dialog_task(device_no: str, dialog_id: str, task: asyncio.Task) -> None:
    _dialog_tasks[device_no] = task
    _dialog_task_dialog_ids[device_no] = dialog_id

    def _cleanup(done_task: asyncio.Task) -> None:
        if _dialog_tasks.get(device_no) is done_task:
            _dialog_tasks.pop(device_no, None)
            _dialog_task_dialog_ids.pop(device_no, None)
        try:
            done_task.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(
                f"AI dialog task ended with unhandled exception | "
                f"deviceNo={device_no} | dialogId={dialog_id} | "
                f"error={type(exc).__name__}: {str(exc)[:300]}"
            )

    task.add_done_callback(_cleanup)


async def _cancel_active_dialog(device_no: str, reason: str) -> Optional[str]:
    session = _dialog_sessions.pop(device_no, None)
    session_dialog_id = str(session.get("dialog_id", "")) if session else ""

    task = _dialog_tasks.pop(device_no, None)
    task_dialog_id = _dialog_task_dialog_ids.pop(device_no, "")
    interrupted_dialog_id = task_dialog_id or session_dialog_id or None

    if session_dialog_id:
        logger.info(
            f"AI dialog recording state cleared | deviceNo={device_no} | "
            f"dialogId={session_dialog_id} | reason={reason}"
        )

    if task is None:
        if interrupted_dialog_id:
            _next_dialog_version(device_no)
        return interrupted_dialog_id

    _next_dialog_version(device_no)
    if task.done():
        return interrupted_dialog_id

    logger.info(
        f"AI dialog interruption requested | deviceNo={device_no} | "
        f"dialogId={task_dialog_id or 'unknown'} | reason={reason}"
    )
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=_DIALOG_INTERRUPT_WAIT_SECONDS)
    except asyncio.CancelledError:
        pass
    except asyncio.TimeoutError:
        logger.warning(
            f"AI dialog task cancellation timed out | deviceNo={device_no} | "
            f"dialogId={task_dialog_id or 'unknown'} | "
            f"timeout={_DIALOG_INTERRUPT_WAIT_SECONDS}s"
        )
    except Exception as exc:
        logger.warning(
            f"AI dialog task finished during interruption | deviceNo={device_no} | "
            f"dialogId={task_dialog_id or 'unknown'} | "
            f"error={type(exc).__name__}: {str(exc)[:200]}"
        )
    return interrupted_dialog_id


async def _cancel_active_dialog_locked(device_no: str, reason: str) -> Optional[str]:
    async with _get_dialog_lock(device_no):
        return await _cancel_active_dialog(device_no, reason)


# ==================== WebSocket端点：胸牌设备连接 ====================

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
                if not ws_device_manager.is_device_online(device_no):
                    logger.warning(
                        f"Device WebSocket activity restored after offline pruning | "
                        f"deviceNo={device_no}"
                    )
                    await ws_device_manager.register_device(device_no, websocket)
                else:
                    ws_device_manager.record_device_activity(device_no)

                # 处理不同类型的消息
                if "text" in data:
                    # 文本消息：播报回执等
                    text = data["text"]
                    logger.debug(
                        f"设备文本消息 | deviceNo={device_no} | "
                        f"内容={text[:200]}"
                    )
                    await _handle_device_message(device_no, websocket, text)
                    # TODO: 与硬件确认回执格式后实现
                elif "bytes" in data:
                    binary = data["bytes"]
                    if device_no in _dialog_sessions:
                        session = _dialog_sessions[device_no]
                        session["chunks"].append(binary)
                        session["total_bytes"] += len(binary)
                        if session["total_bytes"] > _max_dialog_audio_bytes():
                            dialog_id = session.get("dialog_id", "")
                            _dialog_sessions.pop(device_no, None)
                            await ws_device_manager.send_text(
                                device_no,
                                {
                                    "type": "dialog_error",
                                    "dialogId": dialog_id,
                                    "message": "dialog audio exceeds max duration",
                                },
                            )
                        else:
                            logger.debug(
                                f"AI dialog audio chunk | deviceNo={device_no} | "
                                f"dialogId={session.get('dialog_id')} | bytes={len(binary)} | "
                                f"total={session['total_bytes']}"
                            )
                    else:
                        logger.debug(
                            f"device binary message | deviceNo={device_no} | length={len(binary)}"
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
        if ws_device_manager.get_device_websocket(device_no) is websocket:
            await _cancel_active_dialog_locked(device_no, reason="websocket_disconnect")
        await ws_device_manager.unregister_device(device_no, websocket=websocket)


async def _handle_device_message(device_no: str, websocket: WebSocket, text: str) -> None:
    """Handle device text messages, including AI dialog control frames."""
    try:
        msg = json.loads(text)
        msg_type = msg.get("type", "unknown")

        if msg_type == "pong":
            ws_device_manager.handle_pong(device_no)
            logger.debug(f"device pong | deviceNo={device_no}")
            return

        if msg_type == "dialog_start":
            dialog_id = str(msg.get("dialogId", "")).strip()
            payload_device_no = str(msg.get("deviceNo", device_no)).strip()
            if not dialog_id:
                await ws_device_manager.send_text(device_no, {"type": "dialog_start_ack", "dialogId": "", "success": False, "message": "dialogId is required"})
                return
            if payload_device_no and payload_device_no != device_no:
                await ws_device_manager.send_text(device_no, {"type": "dialog_start_ack", "dialogId": dialog_id, "success": False, "message": "deviceNo does not match websocket path"})
                return

            now = time.time()
            async with _get_dialog_lock(device_no):
                active_session = _dialog_sessions.get(device_no)
                if active_session is not None:
                    active_dialog_id = str(active_session.get("dialog_id", ""))
                    active_started_at = float(active_session.get("started_at") or 0)
                    if now - active_started_at < _DIALOG_START_DEBOUNCE_SECONDS:
                        await ws_device_manager.send_text(
                            device_no,
                            {
                                "type": "dialog_start_ack",
                                "dialogId": dialog_id,
                                "success": False,
                                "message": "dialog_start too frequent",
                                "activeDialogId": active_dialog_id,
                            },
                        )
                        logger.warning(
                            f"AI dialog start rejected by debounce | deviceNo={device_no} | "
                            f"dialogId={dialog_id} | activeDialogId={active_dialog_id}"
                        )
                        return

                interrupted_dialog_id = await _cancel_active_dialog(device_no, reason="new_dialog_start")
                session_version = _next_dialog_version(device_no)
                _dialog_sessions[device_no] = {
                    "dialog_id": dialog_id,
                    "version": session_version,
                    "chunks": [],
                    "total_bytes": 0,
                    "started_at": now,
                }
            await ws_device_manager.send_text(device_no, {"type": "dialog_start_ack", "dialogId": dialog_id, "success": True})
            logger.info(
                f"AI dialog started | deviceNo={device_no} | dialogId={dialog_id} | "
                f"version={session_version} | interruptedDialogId={interrupted_dialog_id or ''}"
            )
            return

        if msg_type == "dialog_end":
            dialog_id = str(msg.get("dialogId", "")).strip()
            async with _get_dialog_lock(device_no):
                session = _dialog_sessions.pop(device_no, None)
                if not session:
                    await ws_device_manager.send_text(device_no, {"type": "dialog_error", "dialogId": dialog_id, "message": "dialog session not found"})
                    return
                if dialog_id and dialog_id != session.get("dialog_id"):
                    _dialog_sessions[device_no] = session
                    await ws_device_manager.send_text(device_no, {"type": "dialog_error", "dialogId": dialog_id, "message": "dialogId does not match active session"})
                    return

            pcm_bytes = b"".join(session["chunks"])
            if not pcm_bytes:
                await ws_device_manager.send_text(device_no, {"type": "dialog_error", "dialogId": session["dialog_id"], "message": "dialog audio is empty"})
                return

            async with _get_dialog_lock(device_no):
                task = asyncio.create_task(
                    _handle_ai_dialog_task(
                        device_no=device_no,
                        dialog_id=session["dialog_id"],
                        version=int(session.get("version", 0)),
                        pcm_bytes=pcm_bytes,
                    ),
                    name=f"ai-dialog-{device_no}-{session['dialog_id']}",
                )
                _track_dialog_task(device_no, session["dialog_id"], task)
            logger.info(f"AI dialog audio accepted | deviceNo={device_no} | dialogId={session['dialog_id']} | bytes={len(pcm_bytes)}")
            return

        logger.info(f"device text message | deviceNo={device_no} | type={msg_type} | raw={text[:200]}")

    except Exception as e:
        logger.warning(f"device message parse failed | deviceNo={device_no} | content={text[:100]} | error={str(e)[:100]}")


def _max_dialog_audio_bytes() -> int:
    return AI_DIALOG_SAMPLE_RATE * AI_DIALOG_SAMPLE_WIDTH * AI_DIALOG_CHANNELS * AI_DIALOG_MAX_AUDIO_SECONDS


def _pcm_to_wav(pcm_bytes: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(AI_DIALOG_CHANNELS)
        wf.setsampwidth(AI_DIALOG_SAMPLE_WIDTH)
        wf.setframerate(AI_DIALOG_SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    return buffer.getvalue()


def _current_time_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def _send_current_dialog_error(
    device_no: str,
    dialog_id: str,
    version: int,
    message: str,
) -> bool:
    if not _is_current_dialog(device_no, dialog_id, version):
        logger.info(
            f"AI dialog stale error suppressed | deviceNo={device_no} | "
            f"dialogId={dialog_id} | version={version} | message={message}"
        )
        return False
    return await ws_device_manager.send_text(
        device_no,
        {"type": "dialog_error", "dialogId": dialog_id, "message": message},
    )


async def _iter_compute_dialog_text(
    response,
    dialog_id: str,
    state: Dict[str, str],
) -> AsyncGenerator[str, None]:
    async for line in response.aiter_lines():
        if not line or not line.strip():
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            logger.debug(f"AI dialog stream line ignored | dialogId={dialog_id} | line={line[:200]}")
            continue

        event_type = event.get("type")
        if event.get("id"):
            state["reply_id"] = str(event["id"])

        if event_type == "error":
            raise RuntimeError(str(event.get("message") or "AI dialog stream failed"))

        if event_type == "delta":
            content = str(event.get("content") or "")
            if not content:
                continue
            state["reply_content"] += content
            yield content
            continue

        if event_type == "done":
            final_content = str(event.get("content") or "")
            if final_content and len(final_content) > len(state["reply_content"]):
                yield final_content[len(state["reply_content"]):]
                state["reply_content"] = final_content
            return


async def _dialog_audio_stream_from_compute_response(
    response,
    device_no: str,
    dialog_id: str,
    version: int,
    state: Dict[str, str],
) -> AsyncGenerator[bytes, None]:
    text_stream = _iter_compute_dialog_text(response, dialog_id, state)
    emitted_audio = False
    async for pcm_chunk in piper_tts_manager.synthesize_realtime_stream(text_stream):
        if not _is_current_dialog(device_no, dialog_id, version):
            return
        emitted_audio = True
        yield pcm_chunk
    if not emitted_audio and _is_current_dialog(device_no, dialog_id, version):
        raise RuntimeError("AI reply is empty")


async def _handle_ai_dialog_task(device_no: str, dialog_id: str, version: int, pcm_bytes: bytes) -> None:
    """Forward dialog audio to compute, push FastGPT reply as TTS, then callback backend."""
    selected_node = None
    try:
        if not piper_tts_manager.is_available():
            await _send_current_dialog_error(device_no, dialog_id, version, "TTS service is unavailable")
            return

        selected_node = node_manager.get_least_connection_node()
        if selected_node is None:
            await _send_current_dialog_error(device_no, dialog_id, version, "no available compute node")
            return

        knowledge_base_id = ""
        if _backend_client is not None:
            knowledge_base_id = await _backend_client.get_knowledge_base_id(device_no) or ""
        if not _is_current_dialog(device_no, dialog_id, version):
            logger.info(
                f"AI dialog task became stale after knowledge lookup | "
                f"deviceNo={device_no} | dialogId={dialog_id} | version={version}"
            )
            return
        if not knowledge_base_id:
            logger.error(
                f"AI dialog knowledge base id missing | deviceNo={device_no} | dialogId={dialog_id}"
            )
            await _send_current_dialog_error(device_no, dialog_id, version, "knowledge base id not found")
            return

        wav_bytes = _pcm_to_wav(pcm_bytes)
        event_time = _current_time_str()
        target_url = f"http://{selected_node}{AI_DIALOG_INTERNAL_PATH}"
        files = {"audio_file": (f"{dialog_id}_{device_no}.wav", wav_bytes, "audio/wav")}
        data = {
            "device_no": device_no,
            "dialog_id": dialog_id,
            "event_time": event_time,
            "request_id": dialog_id,
            "knowledge_base_id": knowledge_base_id,
            "knowledgeBaseId": knowledge_base_id,
            "stream": "true",
        }

        await node_manager.increment_connection(selected_node)
        try:
            client = await HttpClientSingleton.get_client()
            async with client.stream(
                "POST",
                url=target_url,
                files=files,
                data=data,
                timeout=AI_DIALOG_REQUEST_TIMEOUT,
            ) as response:
                if not (200 <= response.status_code < 300):
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    logger.error(
                        f"AI dialog compute failed | deviceNo={device_no} | "
                        f"dialogId={dialog_id} | node={selected_node} | "
                        f"status={response.status_code} | body={body[:500]}"
                    )
                    await _send_current_dialog_error(device_no, dialog_id, version, "AI dialog inference failed")
                    return

                if not _is_current_dialog(device_no, dialog_id, version):
                    logger.info(
                        f"AI dialog task became stale after compute response | "
                        f"deviceNo={device_no} | dialogId={dialog_id} | version={version}"
                    )
                    return

                stream_state = {
                    "reply_id": dialog_id,
                    "reply_content": "",
                }
                audio_stream = _dialog_audio_stream_from_compute_response(
                    response=response,
                    device_no=device_no,
                    dialog_id=dialog_id,
                    version=version,
                    state=stream_state,
                )
                push_success = await ws_device_manager.push_dialog_audio_stream(
                    device_no=device_no,
                    dialog_id=dialog_id,
                    audio_stream=audio_stream,
                )
                if not push_success:
                    return
        finally:
            await node_manager.decrement_connection(selected_node)

        reply_id = stream_state.get("reply_id") or dialog_id
        reply_content = (stream_state.get("reply_content") or "").strip()
        if not reply_content:
            await _send_current_dialog_error(device_no, dialog_id, version, "AI reply is empty")
            return

        if not _is_current_dialog(device_no, dialog_id, version):
            logger.info(
                f"AI dialog task became stale after audio push | "
                f"deviceNo={device_no} | dialogId={dialog_id} | version={version}"
            )
            return

        if _backend_client is not None:
            await _backend_client.report_dialog_completion(device_no, _current_time_str())
        logger.info(f"AI dialog pipeline completed | deviceNo={device_no} | dialogId={dialog_id} | replyId={reply_id}")

    except asyncio.CancelledError:
        logger.info(
            f"AI dialog pipeline cancelled | deviceNo={device_no} | "
            f"dialogId={dialog_id} | version={version} | node={selected_node or 'unknown'}"
        )
        raise
    except Exception as e:
        logger.error(f"AI dialog pipeline exception | deviceNo={device_no} | dialogId={dialog_id} | node={selected_node or 'unknown'} | error={type(e).__name__}: {str(e)[:300]}")
        await _send_current_dialog_error(device_no, dialog_id, version, "AI dialog service error")


# ==================== POST接口：后端调用播报 ====================

@duty_broadcast_router.post("/duty-broadcasts/tts")
async def duty_broadcast_tts(request: DutyBroadcastRequest):
    """
    值班播报文字转语音

    完整流程：
    ① 入参校验：deviceNo非空且长度≤20，broadcastContent非空且长度≤200字
    ② 检查TTS服务可用性：不可用返回success=false
    ③ 检查设备在线状态：不在线返回success=false
    ④ 调用PiperTTSManager.synthesize_stream获取TTS API音频分块
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
        # 获取TTS API音频分块（异步生成器）
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
