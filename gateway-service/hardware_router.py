"""
智能胸牌服务管理系统 - 硬件状态上报路由
核心功能：
1. 接收胸牌硬件的心跳和告警状态上报
2. Pydantic模型入参校验（deviceNo/eventType/reportTime/payload）
3. 根据eventType做差异化payload校验（HEARTBEAT/ALARM）
4. 校验通过后返回标准响应（无论后续是否转发后端成功，都返回code=200）
5. 本地缓存设备最新状态（DeviceStatusCache）
6. 异步透传原始数据到后端（复用BackendClient，不阻塞硬件响应）

接口规范：
- 接口路径：POST /internal/badge/hardware/device-events
- Content-Type：application/json
- 入参：{"deviceNo":"BADGE0001","eventType":"HEARTBEAT","reportTime":"...","payload":{...}}
- 出参：{"code":200,"message":"接收成功","data":{"receiveTime":"..."}}

核心原则：
- 算法仅做格式校验+缓存+透传，不修改、不新增、不删除硬件上报的任何业务字段
- 无论后续是否转发后端成功，只要算法成功接收并校验通过，就必须返回code=200
- 透传后端失败不影响硬件上报接口的返回（硬件永远收到200）

使用示例：
    # 正常心跳上报
    curl -X POST http://网关IP:8090/internal/badge/hardware/device-events \
      -H "Content-Type: application/json" \
      -d '{
        "deviceNo": "BADGE0001",
        "eventType": "HEARTBEAT",
        "reportTime": "2026-05-13 15:00:00",
        "payload": {"batteryLevel": 86, "signalLevel": 4}
      }'

    # 正常告警上报
    curl -X POST http://网关IP:8090/internal/badge/hardware/device-events \
      -H "Content-Type: application/json" \
      -d '{
        "deviceNo": "BADGE0001",
        "eventType": "ALARM",
        "reportTime": "2026-05-13 15:01:00",
        "payload": {"alarmCode": "MIC_ERROR", "alarmStatus": "ACTIVE"}
      }'
"""
import asyncio
import re
import time
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, Literal, Optional, Union

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator

from config import (
    DEVICE_NO_MAX_LENGTH,
    DEVICE_NO_PATTERN,
    BATTERY_LEVEL_MIN,
    BATTERY_LEVEL_MAX,
    SIGNAL_LEVEL_MIN,
    SIGNAL_LEVEL_MAX,
)
from device_status_cache import DeviceStatusCache
from utils import BackendClient


# ==================== 枚举定义（对齐v4.1文档7.1节） ====================

class EventType:
    """事件类型枚举"""
    HEARTBEAT = "HEARTBEAT"   # 心跳状态
    ALARM = "ALARM"           # 硬件告警


class AlarmCode:
    """告警编码枚举"""
    MIC_ERROR = "MIC_ERROR"                   # 麦克风异常
    AUDIO_UPLOAD_FAILED = "AUDIO_UPLOAD_FAILED"  # 音频上传失败


class AlarmStatus:
    """告警状态枚举"""
    ACTIVE = "ACTIVE"         # 告警产生
    RECOVERED = "RECOVERED"   # 告警恢复


# ==================== Pydantic Payload模型 ====================

class HeartbeatPayload(BaseModel):
    """
    HEARTBEAT事件payload模型
    """
    batteryLevel: int = Field(
        ...,
        ge=BATTERY_LEVEL_MIN,
        le=BATTERY_LEVEL_MAX,
        description="电量百分比，0-100的整数",
        examples=[86],
    )
    signalLevel: int = Field(
        ...,
        ge=SIGNAL_LEVEL_MIN,
        le=SIGNAL_LEVEL_MAX,
        description="信号等级，0-5的整数（0=无信号，5=满格）",
        examples=[4],
    )


class AlarmPayload(BaseModel):
    """
    ALARM事件payload模型
    """
    alarmCode: str = Field(
        ...,
        description="告警编码，枚举：MIC_ERROR / AUDIO_UPLOAD_FAILED",
        examples=["MIC_ERROR"],
    )
    alarmStatus: str = Field(
        ...,
        description="告警状态，枚举：ACTIVE / RECOVERED",
        examples=["ACTIVE"],
    )

    @field_validator("alarmCode")
    @classmethod
    def validate_alarm_code(cls, v: str) -> str:
        """校验alarmCode必须为合法枚举值"""
        valid_codes = [AlarmCode.MIC_ERROR, AlarmCode.AUDIO_UPLOAD_FAILED]
        if v not in valid_codes:
            raise ValueError(
                f"alarmCode必须为{valid_codes}之一，当前值：{v}"
            )
        return v

    @field_validator("alarmStatus")
    @classmethod
    def validate_alarm_status(cls, v: str) -> str:
        """校验alarmStatus必须为合法枚举值"""
        valid_statuses = [AlarmStatus.ACTIVE, AlarmStatus.RECOVERED]
        if v not in valid_statuses:
            raise ValueError(
                f"alarmStatus必须为{valid_statuses}之一，当前值：{v}"
            )
        return v


# ==================== Pydantic 请求模型 ====================

class DeviceEventRequest(BaseModel):
    """
    硬件设备事件上报请求体模型

    硬件 → 算法，Content-Type: application/json
    核心原则：算法仅做格式校验+缓存+透传，不修改业务字段
    """
    deviceNo: str = Field(
        ...,
        min_length=1,
        max_length=DEVICE_NO_MAX_LENGTH,
        description="设备编号，字母/数字/下划线，≤20位",
        examples=["BADGE0001"],
    )
    eventType: str = Field(
        ...,
        description="事件类型，枚举：HEARTBEAT / ALARM",
        examples=["HEARTBEAT"],
    )
    reportTime: str = Field(
        ...,
        description="上报时间，格式yyyy-MM-dd HH:mm:ss",
        examples=["2026-05-13 15:00:00"],
    )
    payload: Dict[str, Any] = Field(
        ...,
        description="事件数据，根据eventType差异化校验",
        examples=[{"batteryLevel": 86, "signalLevel": 4}],
    )

    @field_validator("deviceNo")
    @classmethod
    def validate_device_no(cls, v: str) -> str:
        """校验deviceNo：仅支持字母、数字、下划线"""
        if not re.match(DEVICE_NO_PATTERN, v):
            raise ValueError(
                f"deviceNo仅支持字母、数字、下划线，当前值：{v}"
            )
        return v

    @field_validator("eventType")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        """校验eventType必须为HEARTBEAT或ALARM"""
        valid_types = [EventType.HEARTBEAT, EventType.ALARM]
        if v not in valid_types:
            raise ValueError(
                f"eventType必须为{valid_types}之一，当前值：{v}"
            )
        return v

    @field_validator("reportTime")
    @classmethod
    def validate_report_time(cls, v: str) -> str:
        """校验reportTime格式必须为yyyy-MM-dd HH:mm:ss"""
        try:
            datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise ValueError(
                f"reportTime格式必须为yyyy-MM-dd HH:mm:ss，当前值：{v}"
            )
        return v

    @model_validator(mode="after")
    def validate_payload_by_event_type(self) -> "DeviceEventRequest":
        """
        根据eventType对payload做差异化校验
        HEARTBEAT → 校验batteryLevel(0-100) + signalLevel(0-5)
        ALARM → 校验alarmCode(枚举) + alarmStatus(枚举)
        """
        if self.eventType == EventType.HEARTBEAT:
            # 校验HEARTBEAT的payload
            try:
                HeartbeatPayload(**self.payload)
            except Exception as e:
                raise ValueError(
                    f"HEARTBEAT事件payload校验失败：{str(e)}"
                )
        elif self.eventType == EventType.ALARM:
            # 校验ALARM的payload
            try:
                AlarmPayload(**self.payload)
            except Exception as e:
                raise ValueError(
                    f"ALARM事件payload校验失败：{str(e)}"
                )
        return self


# ==================== 路由器创建 ====================

hardware_router = APIRouter(
    prefix="/internal/badge/hardware",
    tags=["硬件状态上报接口"],
)

# 设备状态缓存单例
device_status_cache = DeviceStatusCache()

# 后端通用客户端引用（在main.py lifespan startup中通过initialize_forward注入）
_backend_client: Optional[BackendClient] = None


def initialize_forward(backend_client: BackendClient) -> None:
    """
    初始化硬件状态透传后端功能
    在FastAPI lifespan startup阶段调用，注入BackendClient实例

    Args:
        backend_client: 后端通用客户端单例
    """
    global _backend_client
    _backend_client = backend_client
    logger.info("硬件状态透传后端功能已初始化")


async def _forward_device_event_task(event_data: dict) -> None:
    """
    异步透传硬件状态到后端的后台任务
    由asyncio.create_task创建，不阻塞主网关事件循环

    核心原则：
    - 绝对不修改、不新增、不删除硬件上报的任何业务字段
    - 转发失败不影响硬件上报接口的返回
    - 所有异常都要捕获，不能抛出到主事件循环

    Args:
        event_data: 硬件上报的原始数据字典（与硬件上报完全一致）
    """
    device_no = event_data.get("deviceNo", "未知")
    event_type = event_data.get("eventType", "未知")

    try:
        if _backend_client is None:
            logger.warning(
                f"硬件状态转发跳过 | 原因=后端客户端未初始化 | "
                f"deviceNo={device_no} | eventType={event_type}"
            )
            return

        success = await _backend_client.forward_device_event(event_data)

        if success:
            logger.debug(
                f"硬件状态透传任务完成(成功) | "
                f"deviceNo={device_no} | eventType={event_type}"
            )
        else:
            logger.warning(
                f"硬件状态透传任务完成(失败) | "
                f"deviceNo={device_no} | eventType={event_type} | "
                f"原始数据={event_data}"
            )

    except Exception as e:
        # 终极兜底：任何未预料的异常都不能抛到主事件循环
        logger.error(
            f"硬件状态透传任务异常(终极兜底) | "
            f"deviceNo={device_no} | eventType={event_type} | "
            f"异常={type(e).__name__}: {str(e)[:200]} | "
            f"异常栈={traceback.format_exc()[:500]} | "
            f"原始数据={event_data}"
        )


# ==================== POST接口：硬件设备事件上报 ====================

@hardware_router.post("/device-events")
async def receive_device_event(request: DeviceEventRequest):
    """
    接收硬件设备事件上报（心跳/告警）


    - 硬件按约定频率产生心跳状态时调用
    - 设备产生或恢复硬件告警时立即调用
    - 网络中断恢复后，按时间顺序补传所有缓存的状态数据

    核心原则：
    - 算法仅做格式校验+缓存+透传，不修改、不新增、不删除硬件上报的任何业务字段
    - 无论后续是否转发后端成功，只要算法成功接收并校验通过，就必须返回code=200

    请求体：
    {
        "deviceNo": "BADGE0001",
        "eventType": "HEARTBEAT",
        "reportTime": "2026-05-13 15:00:00",
        "payload": {"batteryLevel": 86, "signalLevel": 4}
    }

    成功响应：
    {
        "code": 200,
        "message": "接收成功",
        "data": {"receiveTime": "2026-05-13 15:00:02"}
    }

    校验失败响应：
    {
        "code": 400,
        "message": "具体校验失败原因",
        "data": null
    }
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    device_no = request.deviceNo
    event_type = request.eventType
    report_time = request.reportTime

    logger.info(
        f"收到硬件状态上报 | request_id={request_id} | "
        f"deviceNo={device_no} | eventType={event_type} | "
        f"reportTime={report_time}"
    )

    # ========== 缓存设备最新状态 ==========
    try:
        raw_data = request.model_dump()
        device_status_cache.update_device_status(device_no, raw_data)
        logger.info(
            f"硬件状态缓存成功 | request_id={request_id} | "
            f"deviceNo={device_no} | eventType={event_type} | "
            f"缓存设备数={device_status_cache.get_device_count()}"
        )
    except Exception as e:
        # 缓存失败不影响返回，仅记录日志
        logger.error(
            f"硬件状态缓存失败（不影响接收响应）| request_id={request_id} | "
            f"deviceNo={device_no} | 错误={str(e)[:200]}"
        )
        # 即使缓存失败，也需要raw_data用于透传
        raw_data = request.model_dump()

    # ========== 异步透传原始数据到后端 ==========
    # 核心原则：
    # 1. 算法仅做透传，绝对不修改、不新增、不删除硬件上报的任何业务字段
    # 2. 后端收到的数据必须与硬件原始上报完全一致
    # 3. 使用asyncio.create_task创建后台任务，不阻塞主事件循环
    # 4. 转发失败不影响硬件上报接口的返回（硬件永远收到200）
    try:
        asyncio.create_task(
            _forward_device_event_task(raw_data),
            name=f"forward-device-event-{device_no}-{event_type}",
        )
        logger.info(
            f"硬件状态透传任务已创建 | request_id={request_id} | "
            f"deviceNo={device_no} | eventType={event_type}"
        )
    except Exception as e:
        logger.error(
            f"硬件状态透传任务创建失败（不影响接收响应）| request_id={request_id} | "
            f"deviceNo={device_no} | 错误={str(e)[:200]}"
        )

    # ========== 返回标准响应 ==========
    elapsed_ms = int((time.time() - start_time) * 1000)
    receive_time = _format_current_time()

    logger.info(
        f"硬件状态上报处理完成 | request_id={request_id} | "
        f"deviceNo={device_no} | eventType={event_type} | "
        f"耗时={elapsed_ms}ms"
    )

    return _build_success_response(receive_time)


# ==================== 响应构建 ====================

def _build_success_response(receive_time: str) -> Dict[str, Any]:
    """
    构建硬件上报成功响应体
    {
        "code": 200,
        "message": "接收成功",
        "data": {"receiveTime": "2026-05-13 15:00:02"}
    }

    Args:
        receive_time: 算法接收时间

    Returns:
        统一格式的成功响应字典
    """
    return {
        "code": 200,
        "message": "接收成功",
        "data": {
            "receiveTime": receive_time,
        },
    }


def _format_current_time() -> str:
    """
    格式化当前时间为yyyy-MM-dd HH:mm:ss

    Returns:
        格式化后的当前时间字符串
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
